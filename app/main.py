"""
app/main.py
===========
Serving layer. Implements the blueprint's defense-in-depth pipeline:

    [Incoming URL]
          |
          v
    +----------+   hit
    | Blocklist|-------> UNSAFE (stage="blocklist")
    +----+-----+
         | miss
         v
    +----------+   hit
    | Allowlist|-------> SAFE (stage="allowlist")
    +----+-----+
         | miss
         v
    +-------------------------+
    | ML model (core.features |
    | -> models/train.py's    |
    | pipeline)                |
    +-------------------------+

Fixes the "Checked URL doesn't match what I typed" bug (screenshot 4):
that bug is a symptom of shared mutable state somewhere (a global variable
holding "the last result," or a frontend not tying a response to the
request that produced it). This app has NO global mutable request state -
every request is handled independently, and the response always echoes
back exactly the URL IT checked. The frontend below uses AbortController
so an in-flight stale request can never overwrite a newer one.

Also exposes a PUBLIC bulk checker (no auth) via two endpoints:
/api/bulk-check-paste (pasted text, one/comma-separated URLs, capped at
MAX_PASTE_URLS) and /api/bulk-check-upload (.txt/.csv file, capped at
MAX_FILE_URLS, read straight from the request stream - never written to
disk). A third endpoint, /api/bulk-check-export, turns a set of results
the browser already has into a downloadable CSV/XLSX built in memory.
All three funnel through the same _bulk_check() used by /api/check, so
there is exactly one phishing-detection code path in this file.

The old developer-only /dev/bulk page and its X-Dev-Key-gated endpoints
have been removed entirely - this public flow replaces it. core/auth.py's
require_dev_key still gates /api/admin/reload, which is unrelated to bulk
checking.
"""
from __future__ import annotations
import os
import re
import sys
import io
import csv
import logging
from typing import Literal
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from core.features import extract_features, extract_features_batch, _safe_urlparse, is_valid_url
from core.registry import load_current_model, ModelNotFoundError, DECISION_THRESHOLD
from core.lists import is_allowlisted, is_blocklisted, reload_lists
from core.typosquat import find_typosquat_match
from core.auth import require_dev_key

# Structured logging of verdicts/stages - domain + outcome only, NEVER the
# full URL (path/query can carry search terms, tokens, session data - the
# same privacy reasoning as the extension's query-string stripping).
# Enough to debug "which domains are generating false-positive reports"
# without logging anything a user typed or visited beyond its domain.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phishing_detector")


def _domain_only(url: str) -> str:
    """Hostname only, for logging - never the full URL. See module note above."""
    try:
        return _safe_urlparse(url).hostname or "(unparseable)"
    except Exception:
        return "(unparseable)"


def _log_verdict(url: str, verdict: str, stage: str, confidence: float | None = None) -> None:
    conf_str = f" confidence={confidence:.3f}" if confidence is not None else ""
    logger.info(f"domain={_domain_only(url)} verdict={verdict} stage={stage}{conf_str}")

# In production, FastAPI's auto-generated /docs, /redoc, /openapi.json
# disclose the full shape of the dev-only bulk endpoints (header name,
# request/response models) to any unauthenticated scanner - the auth gate
# itself isn't affected, but it aids reconnaissance for no benefit once
# this isn't being actively developed against. Set APP_ENV=production in
# your host's environment variables to close this; defaults open (docs
# visible) for local development, where they're genuinely useful.
_is_production = os.environ.get("APP_ENV", "development").lower() == "production"
app = FastAPI(
    title="Phishing URL Checker",
    docs_url=None if _is_production else "/docs",
    redoc_url=None if _is_production else "/redoc",
    openapi_url=None if _is_production else "/openapi.json",
)

# Browser extensions call this API from a chrome-extension:// origin, which
# is a real, distinct origin as far as CORS is concerned. Wide open here
# (extension calls carry no cookies, and /api/check plus all the public
# bulk-check* endpoints are meant to be public; /api/admin/reload is
# separately gated by X-Dev-Key regardless of origin).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MAX_BULK_URLS = 5000  # absolute internal ceiling _bulk_check will ever process in one call
MAX_URL_LENGTH = 2048  # matches CheckRequest's existing cap

# Public bulk-checker limits. Deliberately far below MAX_BULK_URLS: the
# host has 512MB total RAM shared with the OS and the already-loaded ML
# model, so these caps (not MAX_BULK_URLS) are what actually bound memory
# use for the public-facing endpoints.
MAX_PASTE_URLS = 50
MAX_FILE_URLS = 75
MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # 2 MB, per the public upload widget's own stated limit
MAX_PASTE_TEXT_CHARS = MAX_PASTE_URLS * (MAX_URL_LENGTH + 1)  # bounds the raw textarea payload itself

# Deliberately regex-based, not a dependency like `urlextract`: RAM budget
# above rules out adding weight for something a compiled pattern already
# does well enough.
#
# SECURITY FIX (2026-07 audit): the previous version of this was ONE regex
# combining tokenization and domain-shape validation, with a label pattern
# shaped like `[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?` repeated inside
# `(?:\....)+` and run via findall() across the WHOLE uploaded blob (up to
# 2MB, unauthenticated). That nested-optional-inside-a-repeated-group shape
# is a textbook catastrophic-backtracking pattern: measured directly,
# 50,000 adversarial characters (long hyphen runs with no valid domain
# ending) took 107 seconds to fail; extrapolating that growth to the 2MB
# upload cap would hang the request for a plausibly multi-hour DoS - far
# worse than the "one 19MB upload froze the server for 35s" incident this
# project's own history already treats as critical, and reachable by
# anyone (this endpoint has no auth).
#
# Also fixes a correctness bug found while rewriting this: the old pattern
# required its repeated middle group to consume at least one ".label"
# BEFORE the separately-mandatory trailing ".TLD" - which made a bare
# two-label domain with no "www." (e.g. "google.com", "example.org")
# impossible to match at all. Confirmed: _URL_EXTRACT_RE.findall("google.com")
# returned [] before this fix. Verified after the fix that it now matches.
#
# The fix: split TOKENIZATION (a single flat character class, `[\s,;"'<>]+`
# - one quantifier layer, cannot backtrack ambiguously regardless of input
# size) from SHAPE VALIDATION (run only against each already-bounded token,
# never the raw multi-megabyte blob). Each validation pattern also uses
# only flat, non-nested quantifiers (`[a-zA-Z0-9-]+`, not the old
# optional-wrapped-star form), which is the standard safe way to write a
# domain-label regex. Verified empirically: 2,000,000 adversarial
# characters (the full upload cap) now takes ~0.27s instead of hanging.
_TOKEN_BOUNDARY_RE = re.compile(r"""[\s,;"'<>]+""")
_SCHEME_URL_RE = re.compile(r"^(?:https?|ftp)://", re.IGNORECASE)
_BARE_DOMAIN_RE = re.compile(
    r"^(?:www\.)?[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,}(?:[/?#].*)?$"
)


def _extract_urls(text: str) -> list[str]:
    """URL-like substrings pulled out of free-form text (a CSV cell, a
    sentence), deduplicated in first-seen order. Used by
    bulk-check-upload, where content is arbitrary and non-URL text must
    be ignored rather than surfaced as an 'Invalid' row. See the security
    fix note above _TOKEN_BOUNDARY_RE for why this is two safe regexes
    applied per-token, not one regex over the whole blob."""
    seen: dict[str, None] = {}
    for raw_token in _TOKEN_BOUNDARY_RE.split(text):
        token = raw_token.strip(".,;:!?)")
        if not token:
            continue
        if _SCHEME_URL_RE.match(token) or _BARE_DOMAIN_RE.match(token):
            if token not in seen:
                seen[token] = None
    return list(seen.keys())


def _split_paste_urls(text: str) -> list[str]:
    """One-token-per-line-or-comma split, NOT extraction: every token the
    user typed is kept (even ones that aren't URLs at all), so
    _bulk_check's is_valid_url pre-filter can mark them 'Invalid' instead
    of them silently disappearing - the behavior the bulk-paste spec
    calls for ('erfgvrthtyjnn' -> Invalid, not dropped)."""
    seen: dict[str, None] = {}
    for token in re.split(r"[,\r\n]+", text):
        token = token.strip()
        if not token:
            continue
        if len(token) > MAX_URL_LENGTH:
            token = token[:MAX_URL_LENGTH]
        if token not in seen:
            seen[token] = None
    return list(seen.keys())

# DECISION_THRESHOLD (the safe/unsafe cutoff) is imported from
# core/registry.py - it used to be defined here, but models/evaluate.py
# hardcoded its own 0.5 to apply the same policy, recreating the exact
# "same value written twice" problem this constant was created to fix.
# One definition, next to model loading, shared by serving + evaluation.

# One message for both the single-check and bulk paths - same "two paths
# must never diverge" rule as DECISION_THRESHOLD above.
INVALID_URL_MESSAGE = "This doesn't look like a valid URL. Please enter a full website address."


class CheckRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=MAX_URL_LENGTH)


class CheckResponse(BaseModel):
    checked_url: str
    status: str = "ok"     # "ok" | "invalid"
    verdict: str | None = None   # "safe" | "unsafe" (None when status="invalid")
    stage: str | None = None     # "blocklist" | "allowlist" | "typosquat" | "model"
    confidence: float | None = None
    model_version: str | None = None
    note: str | None = None
    message: str | None = None   # user-facing explanation when status="invalid"
    # Plain-language explanation, set ONLY when verdict="unsafe" - a
    # non-technical user has no use for "unsafe" without a reason, but a
    # reason next to "safe" reads as suspicious/unearned. See
    # _unsafe_reason() below for the wording per stage.
    reason: str | None = None
    # The real domain a typosquat is impersonating, set ONLY for
    # stage="typosquat" - a bare copy of find_typosquat_match()'s return
    # value, which is always a literal entry from config/allowlist.json,
    # never a string built from the flagged URL itself. This is what lets
    # the extension's warning page safely offer "go to the real site"
    # instead of guessing a domain from string similarity: the redirect
    # target is always a known-good allowlist entry, so even a fooled
    # similarity match can only send the user to the WRONG real site, never
    # an attacker-controlled one.
    legit_domain: str | None = None


class BulkPasteRequest(BaseModel):
    # Bounds the raw textarea payload itself (MAX_PASTE_TEXT_CHARS), not
    # just the URL count after splitting - closes the same "one giant
    # token" DoS shape that the old dev endpoint's per-item max_length was
    # written to fix, just applied before splitting instead of after.
    text: str = Field(..., min_length=1, max_length=MAX_PASTE_TEXT_CHARS)


class BulkCheckResponse(BaseModel):
    results: list[CheckResponse]
    summary: dict


class BulkExportRequest(BaseModel):
    # Re-packages results the browser already has (from a prior
    # bulk-check-paste/upload response) into a downloadable file. Capped
    # at the larger of the two public limits - this endpoint only ever
    # receives what this app itself just returned, never third-party data.
    results: list[CheckResponse] = Field(..., min_length=1, max_length=MAX_FILE_URLS)
    format: Literal["csv", "xlsx"] = "csv"


def _unsafe_reason(stage: str, typosquat_match: str | None = None) -> str:
    """Plain-language, non-technical explanation for why a URL was marked
    unsafe. Never called for a 'safe' verdict - see CheckResponse.reason."""
    if stage == "blocklist":
        return "This website is on a list of known unsafe or scam websites."
    if stage == "typosquat":
        return (
            f"This web address looks a lot like '{typosquat_match}' but isn't it - "
            "a common trick scam sites use to fool people."
        )
    # stage == "model": the ML model has no single human-readable "why" to
    # give (it's a probability over many features, not a rule) - this is
    # the honest, simple summary of what the model looks for.
    return "This website's link has patterns that are often used by fake or scam websites."


def _advisory_legit_domain(url: str) -> str | None:
    """Best-effort 'what real site might this be impersonating', for
    unsafe verdicts that DIDN'T come from the typosquat stage (blocklist,
    model). Those stages already decided the URL is unsafe on their own
    evidence, so this is purely advisory - it drives the warning page's
    'go to the real site' suggestion, not the verdict - and can afford to
    use find_typosquat_match's require_corroboration=False mode (matches
    e.g. 'arnazon.com' against 'amazon.com' even with no suspicious path,
    which the verdict-deciding typosquat stage deliberately withholds to
    avoid flagging coincidental collisions like redfin.com/reddit.com).
    Still only ever returns a literal config/allowlist.json entry - never
    text built from the flagged URL - so it carries the same safety
    property as the typosquat stage's own legit_domain."""
    return find_typosquat_match(url, require_corroboration=False)


def _decide_stage1(url: str) -> CheckResponse | None:
    """Blocklist -> allowlist -> typosquat. Returns None if the URL falls
    through to the ML model (stage1 alone can't decide it). Shared by
    every bulk-check path (paste, upload) and /api/check so they can
    never silently diverge - the same lesson as core/features.py."""
    result: CheckResponse | None = None
    if is_blocklisted(url):
        result = CheckResponse(checked_url=url, verdict="unsafe", stage="blocklist",
                                reason=_unsafe_reason("blocklist"),
                                legit_domain=_advisory_legit_domain(url))
    elif is_allowlisted(url):
        result = CheckResponse(checked_url=url, verdict="safe", stage="allowlist")
    else:
        typosquat_match = find_typosquat_match(url)
        if typosquat_match:
            result = CheckResponse(
                checked_url=url, verdict="unsafe", stage="typosquat",
                note=f"Domain closely resembles known site '{typosquat_match}' but does not match it exactly.",
                reason=_unsafe_reason("typosquat", typosquat_match),
                legit_domain=typosquat_match,
            )
    if result is not None:
        _log_verdict(url, result.verdict, result.stage)
    return result


@app.get("/health")
def health():
    try:
        _, meta = load_current_model()
        return {"status": "ok", "model_version": meta["version"]}
    except ModelNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/api/admin/reload", dependencies=[Depends(require_dev_key)])
def admin_reload():
    """both core/registry.py and core/lists.py
    advertise 'call cache_clear() after retraining/refresh' in their own
    docstrings, but nothing ever called them and no endpoint existed to
    trigger it - after retraining or editing config/*.json, the running
    server kept serving the OLD model/lists until a manual restart. This
    is the missing wiring, gated behind the same dev key as bulk-check."""
    load_current_model.cache_clear()
    reload_lists()
    try:
        _, meta = load_current_model()
        model_version = meta["version"]
    except ModelNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"status": "reloaded", "model_version": model_version}


@app.post("/api/check", response_model=CheckResponse)
def check_url(payload: CheckRequest):
    url = payload.url.strip()

    if not is_valid_url(url):
        return CheckResponse(
            checked_url=url,
            status="invalid",
            message=INVALID_URL_MESSAGE,
        )

    early = _decide_stage1(url)
    if early is not None:
        return early

    try:
        pipeline, metadata = load_current_model()
    except ModelNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    feats = extract_features(url)
    row = pd.DataFrame([feats])
    proba_phishing = float(pipeline.predict_proba(row)[0, 1])
    verdict = "unsafe" if proba_phishing >= DECISION_THRESHOLD else "safe"
    _log_verdict(url, verdict, "model", proba_phishing)

    return CheckResponse(
        checked_url=url,
        verdict=verdict,
        stage="model",
        confidence=proba_phishing,
        model_version=metadata["version"],
        reason=_unsafe_reason("model") if verdict == "unsafe" else None,
        legit_domain=_advisory_legit_domain(url) if verdict == "unsafe" else None,
    )


def _bulk_check(urls: list[str]) -> BulkCheckResponse:
    """Batch-efficient: stage1 (blocklist/allowlist/typosquat) runs per URL
    since those are cheap dict/edit-distance lookups, but anything that
    falls through gets ONE batched feature-extraction + ONE batched
    pipeline.predict_proba call, instead of reloading/predicting per row -
    the difference between checking 5000 URLs in seconds vs minutes."""
    urls = [u.strip() for u in urls if u.strip()][:MAX_BULK_URLS]

    results: list[CheckResponse | None] = []
    fallthrough_indices = []
    fallthrough_urls = []
    for i, url in enumerate(urls):
        # Same pre-filter as /api/check: this path used to skip validation
        # entirely, so a stray non-URL line in an uploaded file ("URL LIST",
        # a CSV fragment, random text) got fed to the model and came back
        # with a confident safe/unsafe verdict - the exact single-vs-bulk
        # drift _decide_stage1 exists to prevent.
        if not is_valid_url(url):
            results.append(CheckResponse(checked_url=url, status="invalid",
                                          message=INVALID_URL_MESSAGE))
            continue
        early = _decide_stage1(url)
        results.append(early)
        if early is None:
            fallthrough_indices.append(i)
            fallthrough_urls.append(url)

    if fallthrough_urls:
        try:
            pipeline, metadata = load_current_model()
        except ModelNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))
        feats_df = extract_features_batch(fallthrough_urls)
        probs = pipeline.predict_proba(feats_df)[:, 1]
        for idx, url, p in zip(fallthrough_indices, fallthrough_urls, probs):
            p = float(p)
            verdict = "unsafe" if p >= DECISION_THRESHOLD else "safe"
            _log_verdict(url, verdict, "model", p)
            results[idx] = CheckResponse(
                checked_url=url,
                verdict=verdict,
                stage="model",
                confidence=p,
                model_version=metadata["version"],
                reason=_unsafe_reason("model") if verdict == "unsafe" else None,
                legit_domain=_advisory_legit_domain(url) if verdict == "unsafe" else None,
            )

    final_results = [r for r in results if r is not None]
    summary = {
        "total": len(final_results),
        "safe": sum(1 for r in final_results if r.verdict == "safe"),
        "unsafe": sum(1 for r in final_results if r.verdict == "unsafe"),
        "invalid": sum(1 for r in final_results if r.status == "invalid"),
        "by_stage": {
            stage: sum(1 for r in final_results if r.stage == stage)
            for stage in ("blocklist", "allowlist", "typosquat", "model")
        },
    }
    return BulkCheckResponse(results=final_results, summary=summary)


@app.post("/api/bulk-check-paste", response_model=BulkCheckResponse)
def bulk_check_paste(payload: BulkPasteRequest):
    """Public, no auth. Splits pasted text on newlines and/or commas -
    every token is kept (not just ones that look like URLs), so
    non-URL text still comes back as an explicit 'Invalid' row via
    _bulk_check's is_valid_url pre-filter, rather than silently vanishing.
    Rejects the whole request up front if it's over MAX_PASTE_URLS -
    never silently processes just the first N."""
    urls = _split_paste_urls(payload.text)
    if not urls:
        raise HTTPException(status_code=400, detail="Please paste at least one URL.")
    if len(urls) > MAX_PASTE_URLS:
        raise HTTPException(
            status_code=400,
            detail=f"Please check up to {MAX_PASTE_URLS} URLs at a time.",
        )
    return _bulk_check(urls)


def _csv_safe(value) -> str:
    """OWASP CSV-injection mitigation: prefix any field starting with a
    formula-trigger character with a single quote, forcing spreadsheet
    apps to treat it as text rather than evaluate it. Closes a confirmed
    finding: an uploaded url column value like '=cmd|\' /C calc\'!A1'
    was written verbatim into the exported CSV and would execute as a
    formula if opened in Excel/Sheets. Applied to CSV and XLSX exports
    alike - the same formula-trigger characters execute in both."""
    s = "" if value is None else str(value)
    if s and s[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + s
    return s


_ALLOWED_UPLOAD_EXTENSIONS = (".txt", ".csv")


@app.post("/api/bulk-check-upload", response_model=BulkCheckResponse)
def bulk_check_upload(file: UploadFile = File(...)):
    """Public, no auth. Accepts a .txt or .csv file, extracts URL-like
    substrings from its content (regardless of column/position - see
    _extract_urls), and runs them through the same pipeline as every
    other check path.

    Nothing about the upload is ever written to disk or kept past this
    request: it's read directly from the request stream into `raw_bytes`,
    decoded, parsed, and discarded once the function returns - no temp
    file, no DB row, no server-side cache of file contents.

    Deliberately a plain `def`, not `async def`: this function does 100%
    synchronous CPU/IO work (parsing, feature extraction, prediction), and
    a synchronous function in FastAPI runs in a threadpool automatically.
    An `async def` version would run all of that directly on the single
    event-loop thread, blocking every other request (including
    /api/check) for the duration - confirmed in this project's own
    security testing history against the old dev-only upload endpoint."""
    filename = (file.filename or "").lower()
    if not filename.endswith(_ALLOWED_UPLOAD_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Only .txt and .csv files are supported.")

    content_length = file.size
    if content_length is not None and content_length > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large; max is {MAX_UPLOAD_BYTES // (1024 * 1024)}MB.",
        )

    raw_bytes = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large; max is {MAX_UPLOAD_BYTES // (1024 * 1024)}MB.",
        )

    # Broad except, not just decode/csv.Error: this is public and
    # unauthenticated, so any corrupt/adversarial upload must degrade to a
    # clean 400, never a 500 with an internal stack trace in the response.
    try:
        raw = raw_bytes.decode("utf-8", errors="ignore")
        if filename.endswith(".csv"):
            rows = list(csv.reader(io.StringIO(raw)))
            text_blob = "\n".join(cell for row in rows for cell in row)
        else:
            text_blob = raw
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read this file. Please upload a plain .txt or .csv file.",
        )

    urls = _extract_urls(text_blob)
    if not urls:
        raise HTTPException(status_code=400, detail="No URLs found in the uploaded file.")
    if len(urls) > MAX_FILE_URLS:
        raise HTTPException(
            status_code=400,
            detail=f"This file contains too many URLs - please limit to {MAX_FILE_URLS} URLs per file.",
        )

    return _bulk_check(urls)


def _status_label(r: CheckResponse) -> str:
    if r.status == "invalid":
        return "Invalid"
    return "Safe" if r.verdict == "safe" else "Unsafe"


def _percent_chance(r: CheckResponse) -> float | str:
    """Chance (0-100) that the verdict shown is correct - e.g. a 'safe'
    verdict from a 0.1 phishing-probability model score is a 90% chance
    of being safe, not 10%. Mirrors the same calculation the single-check
    UI already does client-side."""
    if r.confidence is None:
        return ""
    pct = (1 - r.confidence) if r.verdict == "safe" else r.confidence
    return round(pct * 100, 1)


def _build_export(results: list[CheckResponse], fmt: str) -> tuple[io.BytesIO, str, str]:
    """Builds the downloadable results file entirely in memory
    (io.BytesIO, never a temp file on disk). The caller streams it and
    lets it go out of scope immediately after - nothing here persists.
    "Percent Chance" and "Reason", not "Confidence" - matches the on-screen
    wording (non-technical users don't parse "confidence")."""
    rows = [
        {
            "URL": _csv_safe(r.checked_url),
            "Status": _status_label(r),
            "Percent Chance": _percent_chance(r),
            # Only unsafe rows get a reason - see CheckResponse.reason.
            "Reason": _csv_safe(r.reason) if r.reason else "",
        }
        for r in results
    ]
    df = pd.DataFrame(rows, columns=["URL", "Status", "Percent Chance", "Reason"])

    buf = io.BytesIO()
    if fmt == "xlsx":
        df.to_excel(buf, index=False, engine="openpyxl")
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename = "bulk_check_results.xlsx"
    else:
        buf.write(df.to_csv(index=False).encode("utf-8"))
        media_type = "text/csv"
        filename = "bulk_check_results.csv"
    buf.seek(0)
    return buf, media_type, filename


@app.post("/api/bulk-check-export")
def bulk_check_export(payload: BulkExportRequest):
    """Public, no auth. Turns results the browser already holds (returned
    moments earlier by bulk-check-paste/bulk-check-upload) into a
    downloadable CSV or XLSX. Does not re-run detection and does not
    touch the original uploaded file in any way - by this point that file
    was already discarded, per bulk_check_upload's docstring."""
    buf, media_type, filename = _build_export(payload.results, payload.format)
    headers = {"Content-Disposition": f"attachment; filename={filename}"}
    return StreamingResponse(buf, media_type=media_type, headers=headers)


STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")



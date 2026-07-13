"""
core/features.py
=================
THE SINGLE SOURCE OF TRUTH for turning a URL into model features.

Architectural rule this file exists to enforce:
    Training code and serving code must call THIS SAME FUNCTION.
    Never re-implement feature logic in a second place "for speed" or
    "for the API." The moment there are two implementations, they will
    drift, and the model will silently start scoring garbage. This is
    the #1 root cause of "the model flags everything as phishing" bugs.

Every feature below is:
  - Computed ONLY from the URL string itself (no network calls, no page
    fetch) -> cannot be blocked by a WAF, cannot time out, safe to run
    synchronously in the request path.
  - Deterministic and exactly documented -> no undocumented formulas
    borrowed from a paper that can't actually be reproduced (see
    this file's own commit history for a worked example of why that matters).

Two exported functions:
  extract_features(url)        -> dict, for a single URL (serving path)
  extract_features_batch(urls) -> DataFrame, for many URLs (training path)
extract_features_batch literally calls extract_features per row, so
there is no way for the two paths to disagree.
"""
from __future__ import annotations
import re
import math
import json
import hashlib
import ipaddress
from urllib.parse import urlparse
from collections import Counter
import pandas as pd
from core.wordplay import (
    normalize_confusables, has_mixed_script, is_punycode,
    count_confusable_chars, contains_obfuscated_suspicious_term,
)

# Small, curated, path/query-only keyword list. Deliberately NOT applied to
# the host/domain, so a legitimate bank whose brand name happens to contain
# a suspicious-sounding substring is never penalized just for existing.
# This directly implements the "sanitize what you feed the model" principle:
# structural syntax of the PATH is signal; the company name in the HOST is not.
SUSPICIOUS_PATH_KEYWORDS = [
    "login", "signin", "verify", "secure", "account", "update", "confirm",
    "password", "banking", "webscr", "suspend", "unlock", "authenticate",
]

# Static reference set - no fit-time statistics, so zero leakage risk and
# zero "was this computed on train or test" ambiguity. Deliberately narrow;
# false negatives here (an uncommon-but-legitimate TLD) are fine because
# this is one signal among many, not a gate.
#
# Single-label only: the TLD extraction below always takes just the LAST
# domain label (e.g. "in" from "example.gov.in", never the compound
# "gov.in"). An earlier version of this set included "gov.in"/"co.uk"/
# "co.in" as if compound TLDs could match - they never could, since
# nothing here produces a compound string to check against them. Removed
# as dead/misleading data; behavior is unchanged, since the single-label
# equivalents ("in", "uk") were already present and cover the same domains.
COMMON_TLDS = {
    "com", "org", "net", "edu", "gov", "io", "co", "uk", "de", "in",
    "ca", "au", "us", "info", "biz",
}


def _fingerprint_constants(keywords: list[str], tlds: set[str]) -> str:
    """SUSPICIOUS_PATH_KEYWORDS and COMMON_TLDS are
    code constants; the model metadata itself says 'expand via config, not
    by editing code' as a known limitation, but changing either one
    silently invalidates a trained model - the feature semantics shift
    under a frozen model, with no way to detect the mismatch at serve
    time. This produces a short, order-independent fingerprint of both,
    recorded in model metadata at train time (see models/train.py) and
    checked against the live values whenever a model is loaded."""
    normalized = json.dumps({"keywords": sorted(keywords), "tlds": sorted(tlds)}, sort_keys=True)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def feature_constants_fingerprint() -> str:
    """The fingerprint of the CURRENTLY LOADED code's constants - compare
    this against a model's saved metadata['feature_constants_fingerprint']
    to detect feature-list drift (someone edited the keyword/TLD lists
    without retraining)."""
    return _fingerprint_constants(SUSPICIOUS_PATH_KEYWORDS, COMMON_TLDS)

FEATURE_NAMES = [
    "url_length", "domain_length", "path_length", "query_length",
    "num_dots", "num_hyphens", "num_underscores", "num_slashes",
    "num_digits", "num_letters", "digit_ratio", "letter_ratio",
    "num_special_chars", "special_char_ratio", "num_subdomains",
    "is_ip_address", "is_https", "tld_length", "num_equals",
    "num_question_marks", "num_ampersands", "num_percent_encoded",
    "has_obfuscation", "domain_entropy", "has_at_symbol",
    "num_path_segments", "suspicious_keyword_count", "has_port",
    "is_common_tld",
    # Wordplay/character-substitution features (core/wordplay.py) - see
    # General technique detection (core/wordplay.py), not tied to any
    # specific brand list.
    "has_mixed_script", "is_punycode", "num_confusable_chars",
    "confusable_char_ratio", "domain_has_obfuscated_suspicious_term",
]

# Text used for the TF-IDF side of the model. Path + query ONLY - see
# module docstring. Never include host/domain here. (Computed inline at
# the call site, not via a wrapper function - kept it simple after
# finding the original _path_query_text() helper was defined but never
# actually called anywhere.)


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _safe_urlparse(url: str):
    """urlparse() raises ValueError on some malformed inputs (notably
    invalid IPv6 literals like 'https://[::1/').
    A URL that can't be parsed is itself a signal, not a reason to 500.
    On failure we retry against a bracket-stripped copy so the rest of
    feature extraction still gets real values off the recoverable parts;
    if even that fails, we parse an empty string (all-empty components).
    Well-formed URLs are unaffected - identical output to a bare
    urlparse(), so the trained model is not invalidated."""
    try:
        return urlparse(url)
    except ValueError:
        pass
    try:
        return urlparse(url.replace("[", "").replace("]", ""))
    except ValueError:
        return urlparse("")


def is_valid_url(url: str) -> bool:
    """True if `url` looks like an actual checkable web address - a
    hostname with a dot and a plausible TLD (or a literal IP address),
    not random text with no domain structure at all (e.g.
    "erfgvrthtyjnn"). Mirrors extract_features' own bare-domain tolerance
    (prepends http:// when no scheme is present) so validation and
    feature extraction always agree on what counts as a URL - this is a
    pre-filter in front of the model, not a second feature pipeline.

    Deliberately NOT ASCII-only: a homoglyph/IDN domain (e.g. Cyrillic
    'o's standing in for Latin ones) must still be considered a valid
    URL and reach the typosquat/model stages - that's precisely the kind
    of domain this tool exists to catch, not something to pre-reject.

    Deliberately permissive when a scheme was explicitly given but the
    host still can't be extracted (e.g. "https://[::1/", an unbracketed
    IPv6 literal): _safe_urlparse already has documented, tested recovery
    behavior for exactly this case, feeding the model a "no host" signal
    rather than a verdict-less rejection. A string the user bothered to
    prefix with a real scheme is a URL attempt, not a random word - let
    the existing pipeline judge it instead of a second, stricter gate."""
    url = (url or "").strip()
    if not url:
        return False
    had_scheme = bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url))
    candidate = url if had_scheme else "http://" + url
    parsed = _safe_urlparse(candidate)
    host = parsed.hostname or ""
    if not host:
        return had_scheme
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    if "." not in host:
        return False
    tld = host.rsplit(".", 1)[-1]
    return len(tld) >= 2 and not tld.isdigit()


def _safe_port(parsed) -> int | None:
    """parsed.port raises ValueError for out-of-range (>65535) or
    non-integer ports. An unparseable port
    means 'no valid port' -> has_port=0, which is also a mild signal.
    Well-formed URLs are unaffected."""
    try:
        return parsed.port
    except ValueError:
        return None


def extract_features(url: str) -> dict:
    """Turn one URL string into the full feature dict. This is the ONLY
    place feature logic should ever be written. Both train.py and the
    FastAPI serving layer import this exact function."""
    url = (url or "").strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url  # tolerate bare domains typed by users

    parsed = _safe_urlparse(url)
    host = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    # See count_basis_url note below: a bare "/" path is semantically empty.
    if path == "/" and not query:
        path = ""

    # Structural fix, not a data patch (same pattern as the trailing-slash
    # fix below): PhiUSIIL's legitimate class has a "www." artifact -
    # 100% of its benign examples have a www. prefix; ~59% of its phishing
    # examples don't . "www." is a cosmetic
    # subdomain convention, not a real phishing signal - openai.com,
    # stripe.com, and discord.com are all legitimate with no www. So we
    # compute features on the www-stripped host, removing the model's
    # ability to use "has www" as a proxy for "is legitimate."
    norm_host = host[4:] if host.startswith("www.") else host
    # urlparse().hostname is already lowercased,
    # but the raw url string isn't - a plain str.replace(host, norm_host)
    # silently finds nothing for e.g. "HTTP://WWW.GOOGLE.COM/" (searching
    # for lowercase "www.google.com" inside an uppercase string), so the
    # www-stripping doesn't happen and count-based features differ
    # between two byte-different but semantically identical URLs. Fixed
    # with a case-insensitive match on the actual (possibly-uppercase)
    # host substring, so the real characters get replaced regardless of
    # case - not by lowercasing the whole url, which would also be wrong
    # (path/query case can legitimately carry meaning).
    if norm_host != host:
        url_www_normalized = re.sub(re.escape(host), norm_host, url, count=1, flags=re.IGNORECASE)
    else:
        url_www_normalized = url

    # Structural fix, not a data patch: PhiUSIIL's legitimate class has a
    # trailing-slash/slash-count artifact  that no
    # amount of augmentation fully closes, because a tree ensemble
    # memorizes the specific augmented examples rather than learning "a
    # trailing slash on an otherwise-bare root is semantically identical to
    # no trailing slash." So we remove the distinction at its source:
    # "https://x.com" and "https://x.com/" must produce byte-identical
    # count-based features, because they ARE the same URL to a browser.
    # Only the bare-root case is normalized; a real path's slash count is
    # untouched and still carries real signal.
    count_basis_url = (url_www_normalized.rstrip("/")
                        if (parsed.path in ("", "/") and not query)
                        else url_www_normalized)
    if "://" not in count_basis_url:
        count_basis_url = count_basis_url + "://"  # pathological edge guard

    letters = sum(c.isalpha() for c in count_basis_url)
    digits = sum(c.isdigit() for c in count_basis_url)
    url_len = len(count_basis_url) or 1  # avoid /0
    special = sum(not c.isalnum() for c in count_basis_url)

    labels = [p for p in norm_host.split(".") if p]
    # subdomain count = parts before the registrable domain+TLD (approx: all
    # but the last two labels; good enough without a public-suffix-list dep,
    # and -- critically -- computed identically at train and serve time)
    num_subdomains = max(len(labels) - 2, 0)
    tld = labels[-1] if labels else ""

    path_query_text = f"{path} {query}".lower()
    # Leetspeak-aware: check both the raw text AND its normalized form, so
    # "v3rify-acc0unt" still matches "verify"/"account" even though the
    # literal substring never appears. This is a general fix, not tied to
    # any specific brand - see core/wordplay.py.
    normalized_path_query = normalize_confusables(path_query_text)
    suspicious_count = sum(
        kw in path_query_text or kw in normalized_path_query
        for kw in SUSPICIOUS_PATH_KEYWORDS
    )

    obf_matches = re.findall(r"%[0-9a-fA-F]{2}", url)

    # Wordplay/character-substitution features - general technique
    # detection (core/wordplay.py), independent of any brand list.
    confusables_in_domain = count_confusable_chars(norm_host)

    feats = {
        "url_length": url_len,
        "domain_length": len(norm_host),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": count_basis_url.count("."),
        "num_hyphens": count_basis_url.count("-"),
        "num_underscores": count_basis_url.count("_"),
        "num_slashes": count_basis_url.count("/"),
        "num_digits": digits,
        "num_letters": letters,
        "digit_ratio": digits / url_len,
        "letter_ratio": letters / url_len,
        "num_special_chars": special,
        "special_char_ratio": special / url_len,
        "num_subdomains": num_subdomains,
        "is_ip_address": int(_is_ip(host)),
        "is_https": int(parsed.scheme == "https"),
        "tld_length": len(tld),
        "num_equals": count_basis_url.count("="),
        "num_question_marks": count_basis_url.count("?"),
        "num_ampersands": count_basis_url.count("&"),
        "num_percent_encoded": len(obf_matches),
        "has_obfuscation": int(len(obf_matches) > 0),
        "domain_entropy": _shannon_entropy(norm_host),
        "has_at_symbol": int("@" in count_basis_url),
        "num_path_segments": len([p for p in path.split("/") if p]),
        "suspicious_keyword_count": suspicious_count,
        "has_port": int(_safe_port(parsed) is not None),
        "is_common_tld": int(tld in COMMON_TLDS),
        "has_mixed_script": int(has_mixed_script(norm_host)),
        "is_punycode": int(is_punycode(norm_host)),
        "num_confusable_chars": confusables_in_domain,
        "confusable_char_ratio": confusables_in_domain / len(norm_host) if norm_host else 0.0,
        "domain_has_obfuscated_suspicious_term": int(
            contains_obfuscated_suspicious_term(norm_host)
        ),
        # carried alongside the numeric vector for the TF-IDF stage;
        # dropped before the numeric model sees it (see train.py)
        "_path_query_text": path_query_text,
        "_tld": tld,
    }
    return feats


def extract_features_batch(urls) -> pd.DataFrame:
    """Vectorized-looking convenience wrapper. Internally this is a plain
    loop calling extract_features() per row on purpose -- see module
    docstring for why that's a feature, not a missed optimization."""
    return pd.DataFrame([extract_features(u) for u in urls])

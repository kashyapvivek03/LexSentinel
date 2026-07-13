"""
tests/test_p2_fixes.py
========================
Regression tests for the P2 tier: 2.3 (rate limiter memory
leak + proxy IP + counts successful requests), 2.5 (extension cache
unbounded growth, JS), 2.6 (no admin reload endpoint + metadata_file
FileNotFoundError), 2.7 (duplicated 0.5 threshold constant), 3.3
(case-sensitivity feature drift), 3.4 (typosquat Check 3 unreachable for
short brand cores).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from fastapi.testclient import TestClient
from app.main import app
from core.auth import get_or_create_dev_key, _request_log

client = TestClient(app)


# ---------------------------------------------------------------- 2.3 ----
def test_rate_limiter_purges_stale_client_entries():
    """Review 2.3(a): _request_log never deletes keys - a scanner cycling
    spoofed/rotating IPs grows it forever. After enough distinct clients
    with fully-expired windows, stale entries must be purged, not kept
    forever."""
    from core import auth
    auth._request_log.clear()
    # Simulate many distinct "clients" whose rate-limit window has fully
    # expired (all timestamps far in the past).
    import time
    ancient = time.monotonic() - (auth.RATE_LIMIT_WINDOW_SECONDS * 10)
    for i in range(auth.STALE_CLIENT_PURGE_THRESHOLD + 5):
        auth._request_log[f"fake-client-{i}"].append(ancient)
    # One more real check should trigger a purge of expired-and-empty entries.
    auth._check_rate_limit("real-client")
    assert len(auth._request_log) < auth.STALE_CLIENT_PURGE_THRESHOLD + 6, (
        f"Stale entries were not purged: {len(auth._request_log)} entries remain"
    )
    auth._request_log.clear()


def test_rate_limiter_only_counts_failed_attempts():
    """Review 2.3(c): counting successful requests too means a legitimate
    scripting session hits 429 after 20 calls/min even with a correct key
    every time. Only failed auth attempts should count. Exercised against
    /api/admin/reload - the one remaining dev-key-gated endpoint now that
    bulk-check is public - rather than the old dev-only bulk endpoints."""
    _request_log.clear()
    key = get_or_create_dev_key()
    statuses = []
    for _ in range(30):  # more than RATE_LIMIT_MAX_REQUESTS, all with the CORRECT key
        resp = client.post("/api/admin/reload", headers={"X-Dev-Key": key})
        statuses.append(resp.status_code)
    assert 429 not in statuses, "Legitimate correctly-keyed requests should never be rate-limited"
    _request_log.clear()


# ---------------------------------------------------------------- 2.6 ----
def test_admin_reload_endpoint_exists_and_is_dev_key_gated():
    """Review 2.6: hot-swap is documented ('call cache_clear() after
    retraining') but nothing wires it up - no admin endpoint exists."""
    resp = client.post("/api/admin/reload")
    assert resp.status_code == 401, "Reload endpoint must require the dev key"

    key = get_or_create_dev_key()
    resp2 = client.post("/api/admin/reload", headers={"X-Dev-Key": key})
    assert resp2.status_code == 200, f"{resp2.status_code}: {resp2.text}"


def test_missing_metadata_file_gives_503_not_500():
    """Review 2.6: load_current_model checks preprocessor/xgb existence
    but not metadata_file - a half-written current.json throws a raw
    FileNotFoundError (-> 500) instead of the intended ModelNotFoundError
    (-> 503)."""
    from core.registry import load_current_model, ModelNotFoundError
    artifacts = Path(__file__).resolve().parents[1] / "models" / "artifacts"
    real_current = json.loads((artifacts / "current.json").read_text(encoding="utf-8"))
    fake_current = dict(real_current)
    fake_current["metadata_file"] = "does_not_exist.metadata.json"
    backup = (artifacts / "current.json").read_text(encoding="utf-8")
    try:
        (artifacts / "current.json").write_text(json.dumps(fake_current), encoding="utf-8")
        load_current_model.cache_clear()
        try:
            load_current_model()
            assert False, "Expected ModelNotFoundError"
        except ModelNotFoundError:
            pass  # correct
        except Exception as e:
            assert False, f"Raised {type(e).__name__} instead of ModelNotFoundError: {e}"
    finally:
        (artifacts / "current.json").write_text(backup, encoding="utf-8")
        load_current_model.cache_clear()


# ---------------------------------------------------------------- 2.7 ----
def test_decision_threshold_is_a_single_constant():
    """Review 2.7: 0.5 was written twice (single-check + bulk-check paths)
    - the exact 'two paths silently diverge' failure mode this codebase
    otherwise guards against everywhere else."""
    import app.main as main_module
    assert hasattr(main_module, "DECISION_THRESHOLD"), (
        "No module-level DECISION_THRESHOLD constant found"
    )
    main_py = Path(__file__).resolve().parents[1].joinpath("app", "main.py").read_text(encoding="utf-8")
    # after the constant's own definition line, no other bare "0.5" should
    # appear as a verdict comparison
    import re
    bare_threshold_comparisons = re.findall(r'[<>]=?\s*0\.5\b', main_py)
    assert len(bare_threshold_comparisons) == 0, (
        f"Found hardcoded 0.5 comparisons instead of using DECISION_THRESHOLD: {bare_threshold_comparisons}"
    )


# ---------------------------------------------------------------- 3.3 ----
def test_feature_extraction_case_insensitive_for_scheme_and_host():
    """Review 3.3: urlparse().hostname is lowercased but the raw url isn't
    - url.replace(host, norm_host, 1) silently fails to find the lowercase
    host inside an uppercase URL, so www-stripping doesn't happen and
    count-based features differ between two byte-different but
    semantically identical URLs."""
    from core.features import extract_features
    a = extract_features("http://www.google.com/")
    b = extract_features("HTTP://WWW.GOOGLE.COM/")
    diffs = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
    assert not diffs, f"Case-sensitivity feature drift: {diffs}"


# ---------------------------------------------------------------- 3.4 ----
def test_typosquat_check3_reachable_for_short_brand_cores():
    """Review 3.4: for protected cores of length <=4 (sbi, rbi, irs, usa,
    nih, ajio, jio), Check 2's blanket `continue` skips straight to the
    next protected domain, so Check 3 (leetspeak/homoglyph exact match)
    never runs for them. '5b!' normalizes exactly to 'sbi' under this
    codebase's own LEETSPEAK_MAP (5->s, !->i) but was previously missed
    entirely."""
    from core.typosquat import find_typosquat_match
    result = find_typosquat_match("https://www.5b!.co.in/")
    assert result == "sbi.co.in", f"Expected 'sbi.co.in', got {result}"


# ---------------------------------------------------------------- 2.5 ----
def test_extension_cache_is_bounded():
    """Review 2.5: urlCache in chrome.storage.session accumulates one
    entry per distinct URL visited, TTL-checked on read but never
    evicted. chrome.storage.session has a ~10MB quota; heavy browsing
    eventually makes storage.session.set start throwing. setCacheEntry
    must prune expired/excess entries, capped at a fixed size."""
    bg_js = Path(__file__).resolve().parents[1].joinpath("extension", "background.js").read_text(encoding="utf-8")
    assert "MAX_CACHE_ENTRIES" in bg_js, "No cache size cap constant found in background.js"
    assert "prune" in bg_js.lower() or "evict" in bg_js.lower(), (
        "setCacheEntry does not appear to prune/evict old entries"
    )

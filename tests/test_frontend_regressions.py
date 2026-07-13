"""
tests/test_frontend_regressions.py
====================================
Source-inspection regression tests for the frontend bugs fixed in the
2026-07-09 audit pass (same approach as test_p1_fixes.py: real XSS/DOM
behavior needs a browser, but each of these bugs has an unambiguous
source-level signature that a plain text check can pin down).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def test_bulk_modal_renders_invalid_rows():
    """The public bulk-check UI (app/static/index.html, replacing the old
    dev-only bulk.html) must handle status='invalid' rows (verdict null)
    without calling .toUpperCase() on null - same bug class as
    test_popup_handles_invalid_status_without_crashing below."""
    index = _read("app/static/index.html")
    assert "r.status === 'invalid'" in index, (
        "index.html's bulk results rendering has no handling for invalid rows"
    )


def test_popup_handles_invalid_status_without_crashing():
    """popup.js renderStatus() crashed (TypeError on
    result.verdict.toUpperCase()) when the backend answered
    status='invalid' with verdict=null - e.g. a manual check on an
    intranet host or localhost."""
    popup = _read("extension/popup.js")
    assert 'result.status === "invalid"' in popup, (
        "popup.js does not handle the backend's status='invalid' response"
    )


def test_index_submits_on_enter_key():
    """The URL input isn't inside a <form>, so Enter did nothing -
    clicking the button was the only way to submit."""
    index = _read("app/static/index.html")
    assert "keydown" in index and "'Enter'" in index, (
        "index.html has no Enter-key submit handler"
    )


def test_index_handles_http_error_responses_deliberately():
    """Non-2xx responses (422 URL-too-long, 429, 503) previously 'worked'
    only because data.verdict.toUpperCase() threw a TypeError that
    happened to land in the catch block."""
    index = _read("app/static/index.html")
    assert "!res.ok" in index, "index.html never checks res.ok"


def test_warning_page_shows_the_reason_it_was_blocked():
    """2026-07 audit: background.js's redirectToWarning() built note/stage/
    confidence into the warning page's URL params, but warning.js never
    read them and warning.html had no element to show them - a user
    blocked by the extension saw only a generic hardcoded sentence, never
    the actual reason. Now `reason` (plain-language, set for every unsafe
    stage) is built, read, and rendered."""
    background = _read("extension/background.js")
    assert "reason: result.reason" in background, (
        "background.js's redirectToWarning() does not pass 'reason' to the warning page"
    )
    warning_js = _read("extension/warning.js")
    assert 'params.get("reason")' in warning_js, (
        "warning.js does not read the 'reason' query param"
    )
    warning_html = _read("extension/warning.html")
    assert 'id="reasonBox"' in warning_html, (
        "warning.html has no element to display the block reason"
    )


def test_popup_shows_reason_for_every_unsafe_stage():
    """2026-07 audit: popup.js only ever rendered result.note, which is
    set ONLY for the typosquat stage - a blocklist- or model-flagged
    unsafe result showed no explanation at all. result.reason is set for
    every unsafe stage (see app/main.py's _unsafe_reason)."""
    popup = _read("extension/popup.js")
    assert "result.reason" in popup, (
        "popup.js does not reference result.reason - unsafe verdicts from "
        "the blocklist/model stages would show no explanation"
    )


def test_options_page_validates_backend_url_before_saving():
    """options.js saved any string as the backend URL; a typo silently
    broke every subsequent check (fail-open '?' badge on every site)."""
    options = _read("extension/options.js")
    assert "new URL(" in options, "options.js does not validate the URL"
    for proto_check in ['u.protocol === "http:"', 'u.protocol === "https:"']:
        assert proto_check in options, "options.js does not restrict to http(s)"

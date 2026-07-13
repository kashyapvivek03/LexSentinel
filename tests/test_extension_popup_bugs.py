"""
tests/test_extension_popup_bugs.py
====================================
Regression tests for a bug found through real usage (not the code review):
the extension popup showed "chrome://newtab/ - UNSAFE - model (100.0%)"
for every site checked, regardless of which real site was actually open.

Root cause chain:
1. MANUAL_CHECK (unlike the auto-check path in onBeforeNavigate) never
   filtered out browser-internal schemes (chrome://, about://, etc.) -
   background.js. If ever triggered while a tab was on chrome://newtab/,
   it sent that to the backend, which - never having seen anything like
   it in training - confidently (100%) called it unsafe, and cached it.
2. popup.js's init() never checked cache-entry freshness (isCacheFresh())
   before rendering a cached result - unlike every other read path in the
   extension. So that one bad chrome://newtab/ result never expired from
   the DISPLAY's perspective, and resurfaced any time the popup happened
   to catch the real timing race (checking a tab before its navigation to
   a newly-typed URL has actually committed, which is a genuine, expected
   browser behavior, not itself a bug).

These are source-inspection tests (JS, not Python-executable) since
there's no JS test runner in this project - see tests/test_p1_fixes.py
for the same pattern used elsewhere.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_manual_check_filters_ignored_schemes():
    """background.js's MANUAL_CHECK handler must reject/skip
    browser-internal schemes the same way onBeforeNavigate already does -
    it must not be possible to send chrome://, about://, etc. to the
    backend and get a nonsensical cached verdict for it."""
    bg_js = Path(__file__).resolve().parents[1].joinpath("extension", "background.js").read_text(encoding="utf-8")
    manual_check_start = bg_js.index('message.type === "MANUAL_CHECK"')
    manual_check_block = bg_js[manual_check_start:manual_check_start + 600]
    assert "IGNORED_SCHEMES" in manual_check_block, (
        "MANUAL_CHECK handler does not filter IGNORED_SCHEMES - "
        "browser-internal URLs can still be sent to the backend and cached"
    )


def test_popup_checks_cache_freshness_before_rendering():
    """popup.js must call isCacheFresh() (or equivalent) before displaying
    a cached result - an expired/stale entry must not be shown as if it
    were current. This is what let one bad chrome://newtab/ result from
    long ago keep resurfacing."""
    popup_js = Path(__file__).resolve().parents[1].joinpath("extension", "popup.js").read_text(encoding="utf-8")
    assert "isCacheFresh" in popup_js, (
        "popup.js does not check cache freshness before rendering a cached result"
    )


def test_popup_shows_clear_message_for_unsupported_pages():
    """When the active tab is a browser-internal page (new tab, chrome://
    settings, etc.), the popup should say so clearly rather than silently
    querying a cache keyed by that URL and potentially showing an
    unrelated/stale result."""
    popup_js = Path(__file__).resolve().parents[1].joinpath("extension", "popup.js").read_text(encoding="utf-8")
    assert "IGNORED_SCHEMES" in popup_js or "cannot be checked" in popup_js.lower() or "can't be checked" in popup_js.lower(), (
        "popup.js has no handling for browser-internal pages"
    )

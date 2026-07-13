"""
tests/test_malformed_urls.py
=============================
Regression tests for the P0 tier: malformed URLs raised an
unhandled ValueError from urlparse/parsed.port, causing HTTP 500 on
/api/check and — worse — failing an ENTIRE bulk-check batch if one
row was malformed.

The four reproduction inputs are taken verbatim from the review.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient
from app.main import app
from core.features import extract_features

client = TestClient(app)

MALFORMED_URLS = [
    "http://example.com:99999/",   # Port out of range 0-65535
    "http://example.com:abc/",     # Port could not be cast to integer
    "https://[::1/",               # Invalid IPv6 URL (crashes even blocklist stage)
    "http://[::1",                 # Invalid IPv6 URL, no trailing slash
]


@pytest.mark.parametrize("url", MALFORMED_URLS)
def test_extract_features_does_not_raise(url):
    """extract_features must never raise on a malformed URL - a URL that
    can't be parsed is itself a signal, not a crash."""
    feats = extract_features(url)  # must not raise
    assert isinstance(feats, dict)
    assert "url_length" in feats


@pytest.mark.parametrize("url", MALFORMED_URLS)
def test_api_check_does_not_500(url):
    """POST /api/check with a malformed URL must return 200 with a verdict,
    not a 500."""
    resp = client.post("/api/check", json={"url": url})
    assert resp.status_code == 200, f"{url} -> {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data["verdict"] in ("safe", "unsafe")
    assert data["checked_url"] == url


def test_bulk_check_one_malformed_url_does_not_fail_whole_batch():
    """A single malformed URL in a batch must not fail all the others."""
    urls = [
        "https://www.google.com/",
        "http://example.com:99999/",   # malformed, in the middle
        "https://www.wikipedia.org/",
    ]
    resp = client.post("/api/bulk-check-paste", json={"text": "\n".join(urls)})
    assert resp.status_code == 200, f"{resp.status_code}: {resp.text}"
    data = resp.json()
    assert data["summary"]["total"] == 3
    verdicts = {r["checked_url"]: r["verdict"] for r in data["results"]}
    assert verdicts["https://www.google.com/"] == "safe"
    assert verdicts["https://www.wikipedia.org/"] == "safe"

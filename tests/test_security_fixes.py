"""
tests/test_security_fixes.py
==============================
Regression tests for findings from a 2026-07-07 red
team review).16 for the full writeup.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import io
import inspect
from fastapi.testclient import TestClient
from app.main import app, bulk_check_upload, MAX_URL_LENGTH
from core.auth import _request_log

client = TestClient(app)


def test_bulk_check_upload_is_not_async():
    """HIGH finding: the file-upload endpoint used to be `async def` but
    did 100% synchronous work, blocking the event loop for the whole
    server (confirmed: one 19MB upload froze /health for 35s). A sync
    `def` runs in FastAPI's threadpool automatically instead."""
    assert not inspect.iscoroutinefunction(bulk_check_upload)


def test_oversized_upload_rejected():
    big_content = ("https://example.com/\n" * 1).encode() + b"a" * (3 * 1024 * 1024)
    resp = client.post(
        "/api/bulk-check-upload",
        files={"file": ("big.txt", io.BytesIO(big_content), "text/plain")},
    )
    assert resp.status_code == 413


def test_oversized_paste_url_truncated_not_a_dos():
    """Per-token length cap on the paste path - this is what made the DoS
    reachable without any file upload at all. A single wildly oversized
    pasted token must be truncated and handled fast, not hang the
    request or 500."""
    huge_url = "http://example.com/" + "a" * (MAX_URL_LENGTH + 1)
    resp = client.post("/api/bulk-check-paste", json={"text": huge_url})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 1
    assert len(data["results"][0]["checked_url"]) == MAX_URL_LENGTH


def test_csv_injection_escaped():
    """OWASP CSV-injection mitigation: formula-trigger characters at the
    start of a field get a leading single quote so spreadsheet apps treat
    them as text, not formulas.

    /api/bulk-check-export takes results the browser already holds (from
    a prior paste/upload response) and re-packages them into a
    downloadable file WITHOUT re-running detection - so it must sanitize
    on the way out regardless of where the checked_url text came from,
    the same as any other public endpoint that shouldn't trust its input.
    Tested by POSTing formula-shaped checked_url values directly, since
    the upload/paste extraction paths themselves only ever produce
    URL-shaped strings and wouldn't reproduce this on their own."""
    results = [
        {"checked_url": "=cmd|'/C calc'!A1", "status": "ok", "verdict": "unsafe", "confidence": 0.9},
        {"checked_url": "+2+5", "status": "ok", "verdict": "safe", "confidence": 0.1},
        {"checked_url": "-2+3", "status": "ok", "verdict": "safe", "confidence": 0.1},
        {"checked_url": "@SUM(1+1)", "status": "ok", "verdict": "safe", "confidence": 0.1},
    ]
    resp = client.post("/api/bulk-check-export", json={"results": results, "format": "csv"})
    assert resp.status_code == 200, resp.text
    body = resp.text
    assert "'=cmd" in body
    assert "'+2+5" in body
    assert "'-2+3" in body
    assert "'@SUM" in body
    # and none of the RAW (unescaped) forms appear at the start of a line
    for line in body.splitlines():
        assert not line.startswith(("=", "+", "-", "@"))


def test_xff_client_id_uses_last_hop_not_client_supplied_first(monkeypatch):
    """2026-07-09 audit: in production the rate-limit client ID was the
    FIRST X-Forwarded-For entry - which is whatever the client claims.
    An attacker could rotate fake IPs to bypass the limit, or spoof the
    real developer's IP to lock them out. The platform proxy (Render)
    APPENDS the IP it actually saw, so the LAST entry is the trustworthy
    one."""
    import core.auth as auth

    class FakeClient:
        host = "10.0.0.1"

    class FakeRequest:
        headers = {"x-forwarded-for": "6.6.6.6, 203.0.113.9"}
        client = FakeClient()

    monkeypatch.setattr(auth, "_TRUST_PROXY_HEADERS", True)
    assert auth._get_client_id(FakeRequest()) == "203.0.113.9", (
        "Client ID must come from the proxy-appended (last) XFF hop, "
        "never the client-supplied first hop"
    )
    monkeypatch.setattr(auth, "_TRUST_PROXY_HEADERS", False)
    assert auth._get_client_id(FakeRequest()) == "10.0.0.1", (
        "Outside production, XFF must be ignored entirely (spoofable)"
    )


def test_dev_key_rate_limited():
    """LOW finding: no throttling on repeated wrong-key attempts. Exercised
    against /api/admin/reload, the one remaining dev-key-gated endpoint
    now that bulk-check is public."""
    _request_log.clear()
    statuses = []
    for _ in range(25):
        resp = client.post("/api/admin/reload", headers={"X-Dev-Key": "wrong"})
        statuses.append(resp.status_code)
    assert 429 in statuses
    _request_log.clear()  # don't leak rate-limit state into other tests

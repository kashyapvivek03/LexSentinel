"""
core/auth.py
============
Gates developer-only features behind a secret key, so this isn't
accidentally exposed to anyone who finds the URL. Currently that's just
/api/admin/reload (hot-swap the model/lists after a retrain) - bulk
checking used to live behind this too, but is now a public feature (see
app/main.py's bulk-check-paste/upload/export). The key is auto-generated
on first run and stored in config/dev_key.txt, which is gitignored -
nothing to accidentally commit or hardcode.

This is intentionally simple (a single shared secret, not per-user
accounts) because the stated requirement is "only me/developers," not
"multiple users with different permission levels." If that need grows,
swap this for real auth (e.g. FastAPI's OAuth2/JWT support) rather than
extending this file.
"""
from __future__ import annotations
import os
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path
from fastapi import Header, HTTPException, Request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KEY_PATH = PROJECT_ROOT / "config" / "dev_key.txt"
ENV_VAR_NAME = "PHISHING_DETECTOR_DEV_KEY"

# Simple in-memory sliding-window rate limit on dev-key endpoints. Brute
# force isn't realistically feasible against a 192-bit token (see
# get_or_create_dev_key), so this is defense-in-depth/log-hygiene, not a
# response to a real bypass risk - closes a LOW finding from security
# testing (30 rapid wrong-key requests all returned plain 401s with no
# throttling). In-memory is fine for this single-instance, single-developer
# tool; a multi-instance deployment would need a shared store instead.
#
# Three sub-issues fixed here:
# (a) _request_log never deleted keys - a scanner cycling spoofed/rotating
#     IPs grew it forever. Now purged once the table gets large.
# (b) On Render (the documented deployment target) the app sits behind a
#     proxy, so request.client.host was the PROXY's IP - every real client
#     shared one bucket, and an attacker could lock the real developer out.
#     Now reads X-Forwarded-For's last hop when APP_ENV=production (the
#     same flag already used to gate /docs - Render is always behind a
#     proxy, so bundling this under one existing flag avoids requiring a
#     second env var). NOT trusted by default, since blindly trusting
#     X-Forwarded-For when NOT actually behind a real proxy lets a client
#     spoof any IP it wants to reset its own limit.
# (c) the limit used to count SUCCESSFUL requests too, so a legitimate
#     bulk-scripting session with the correct key hit 429 after 20
#     calls/min. Now only failed auth attempts are counted - see
#     require_dev_key below, which checks the key BEFORE touching the
#     rate limiter at all.
RATE_LIMIT_MAX_REQUESTS = 20
RATE_LIMIT_WINDOW_SECONDS = 60
STALE_CLIENT_PURGE_THRESHOLD = 10_000
_request_log: dict[str, deque] = defaultdict(deque)
_TRUST_PROXY_HEADERS = os.environ.get("APP_ENV", "development").lower() == "production"


def _get_client_id(request: Request) -> str:
    if _TRUST_PROXY_HEADERS:
        xff = request.headers.get("x-forwarded-for")
        if xff:
            # LAST entry, not first: the platform proxy (Render) APPENDS
            # the IP it actually saw, while everything to the left is
            # whatever the client claims. Taking the first entry let a
            # client send "X-Forwarded-For: <anything>" to pick its own
            # rate-limit bucket - rotating fake IPs to bypass the limit,
            # or spoofing the real developer's IP to lock them out (the
            # exact attack scenario (b) above is meant to prevent).
            return xff.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def _purge_stale_clients(now: float) -> None:
    stale_ids = [
        cid for cid, log in _request_log.items()
        if not log or now - log[-1] > RATE_LIMIT_WINDOW_SECONDS
    ]
    for cid in stale_ids:
        del _request_log[cid]


def _check_rate_limit(client_id: str) -> None:
    now = time.monotonic()
    log = _request_log[client_id]
    while log and now - log[0] > RATE_LIMIT_WINDOW_SECONDS:
        log.popleft()
    if len(log) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {RATE_LIMIT_MAX_REQUESTS} requests "
                    f"per {RATE_LIMIT_WINDOW_SECONDS}s on this endpoint.",
        )
    log.append(now)
    if len(_request_log) > STALE_CLIENT_PURGE_THRESHOLD:
        _purge_stale_clients(now)


def get_or_create_dev_key() -> str:
    # Production (Render, etc.): filesystem resets on every deploy, so an
    # env var set once in the host's dashboard is the only thing that
    # actually persists. Checked first for that reason.
    env_key = os.environ.get(ENV_VAR_NAME)
    if env_key:
        return env_key.strip()

    if KEY_PATH.exists():
        return KEY_PATH.read_text(encoding="utf-8").strip()
    key = secrets.token_urlsafe(24)
    KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    KEY_PATH.write_text(key, encoding="utf-8")
    print(f"\n[phishing_detector] Generated new dev key at {KEY_PATH}")
    print("[phishing_detector] Read the key from that file, or run:")
    print("[phishing_detector]   python -c \"from core.auth import get_or_create_dev_key; print(get_or_create_dev_key())\"")
    print("[phishing_detector] Use it in the X-Dev-Key header for /api/admin/reload")
    print("[phishing_detector] (Not printed here directly - on hosts like Render, stdout")
    print("[phishing_detector]  goes into persistent platform logs.)\n")
    return key


def require_dev_key(request: Request, x_dev_key: str = Header(default=None)) -> None:
    """FastAPI dependency - raises 401 if the header is missing/wrong,
    429 if this client has exceeded the FAILED-attempt rate limit.

    Key checked FIRST, rate limiter only touched on the failure path -
    see 2.3(c) above for why (a correct key must never count against the
    limit, or legitimate scripted use gets locked out)."""
    expected = get_or_create_dev_key()
    if x_dev_key and secrets.compare_digest(x_dev_key, expected):
        return
    client_id = _get_client_id(request)
    _check_rate_limit(client_id)
    raise HTTPException(status_code=401, detail="Missing or invalid X-Dev-Key header.")

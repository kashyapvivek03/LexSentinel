"""
core/lists.py
=============
Fixes audit finding: "Large hardcoded allowlists become stale... Move to
configuration/database with automated updates."

Lists live in config/*.json (data), not in Python source (code). Loading
is cached but explicitly reloadable, so a refresh job can call
reload_lists() after updating the JSON/DB without restarting the process,
or POST /api/admin/reload (dev-key gated) which does this for you.

This is the front gate from the blueprint's defense-in-depth diagram:
    blocklist hit  -> UNSAFE immediately, skip the model entirely
    allowlist hit  -> SAFE immediately, skip the model entirely
    neither        -> fall through to the ML model

Matching is by registrable-ish domain (host with a leading "www." stripped),
so "https://www.discord.com/anything?x=1" matches the "discord.com" entry.
"""
from __future__ import annotations
import json
from pathlib import Path
from functools import lru_cache
from core.features import _safe_urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"


def _normalize_host(url_or_host: str) -> str:
    if "://" not in url_or_host:
        url_or_host = "http://" + url_or_host
    host = _safe_urlparse(url_or_host).hostname or ""
    return host[4:] if host.startswith("www.") else host


def _host_matches_entry(host: str, entry: str) -> bool:
    """Exact match OR host is a subdomain of the entry (en.wikipedia.org
    matches wikipedia.org; evil-wikipedia.org must NOT match, hence the
    dot-boundary check rather than a bare .endswith(entry)). Kept for
    reference/tests - the actual matching path now uses _matches_any,
    which implements the identical semantics via suffix-walking."""
    return host == entry or host.endswith("." + entry)


def _host_suffixes(host: str) -> list[str]:
    """All dot-boundary suffixes of host, most-specific first, e.g.
    'en.wikipedia.org' -> ['en.wikipedia.org', 'wikipedia.org', 'org'].
    this is what turns matching into O(#labels)
    average-case SET LOOKUPS instead of an O(n) scan across every entry
    in the list - the file's own docstring says production should sync
    the allowlist from the Tranco Top 1M, at which point a linear scan
    becomes a million-iteration check per URL. Set membership doesn't
    care how large the set is."""
    labels = host.split(".")
    return [".".join(labels[i:]) for i in range(len(labels))]


def _matches_any(host: str, domain_set: set) -> bool:
    """Same semantics as any(_host_matches_entry(host, e) for e in
    domain_set), but O(#labels) instead of O(len(domain_set))."""
    return any(suffix in domain_set for suffix in _host_suffixes(host))


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    path = CONFIG_DIR / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_domain_set"] = set(data["domains"])
    return data


def reload_lists():
    """Call after updating config/*.json or after a scheduled refresh sync
    (see docstring: production should sync allowlist from Tranco/Umbrella
    and blocklist from PhishTank/OpenPhish/URLHaus on a short interval)."""
    _load.cache_clear()


def is_allowlisted(url: str) -> bool:
    host = _normalize_host(url)
    return _matches_any(host, _load("allowlist")["_domain_set"])


def is_blocklisted(url: str) -> bool:
    host = _normalize_host(url)
    return _matches_any(host, _load("blocklist")["_domain_set"])

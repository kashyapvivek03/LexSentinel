"""
core/wordplay.py
=================
General-purpose defense against character-level obfuscation in phishing
URLs: leetspeak-style digit/symbol substitution ("s3cure", "acc0unt"),
Unicode homoglyphs (Cyrillic "а" standing in for Latin "a" - the classic
IDN homograph attack), and punycode-encoded lookalike domains.

Deliberately NOT a big fuzzy-match-against-a-huge-dictionary system: that
was considered and rejected  because it raises false
positives on legitimate short/coincidental-looking domains without
actually helping the domain-name case (brand names aren't in a standard
English dictionary to begin with). Instead, normalization is applied to a
small, well-scoped set of security-relevant terms and to Unicode
structure - the TECHNIQUE generalizes, without needing a huge reference
set that would itself become a source of noise.
"""
from __future__ import annotations

# Leetspeak-style substitutions: character -> canonical letter it's
# standing in for. Deliberately conservative (only the extremely common
# ones) to avoid over-normalizing legitimate alphanumeric strings.
LEETSPEAK_MAP = {
    "0": "o", "1": "l", "3": "e", "4": "a", "5": "s",
    "7": "t", "8": "b", "@": "a", "$": "s", "!": "i",
}

# Common Unicode homoglyphs used in IDN homograph attacks - Cyrillic and
# Greek letters that are visually near-identical to Latin letters. Not
# exhaustive (the full Unicode confusables table has thousands of entries)
# but covers the characters actually easy to type/paste and seen in the
# wild, which is what matters for a synchronous, no-lookup-table check.
HOMOGLYPH_MAP = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",  # Cyrillic
    "ѕ": "s", "і": "i", "ј": "j", "ԁ": "d", "ѡ": "w",
    "α": "a", "ο": "o", "ε": "e", "і": "i",  # Greek (some overlap with above)
}

CONFUSABLES_MAP = {**LEETSPEAK_MAP, **HOMOGLYPH_MAP}

# Security/action terms worth catching even when NOT tied to a specific
# protected brand - generic "this domain/path is trying to look like a
# login/verification/banking flow" signal. Small and conservative on
# purpose (false-positive risk on real company names goes up with size).
GENERIC_SUSPICIOUS_TERMS = [
    "secure", "verify", "login", "signin", "account", "bank", "banking",
    "password", "confirm", "update", "wallet", "credential", "authenticate",
    # Added after real-user testing found "arnazon-orders.com" wasn't
    # caught: the original list skewed toward login/banking lures and had
    # no coverage for common e-commerce/delivery phishing patterns.
    "order", "orders", "delivery", "shipment", "tracking", "invoice",
    "billing", "refund", "suspended", "expired",
]

# Unicode code point ranges for scripts commonly abused in homograph
# attacks. Latin covers ASCII + Latin-1 supplement + extended-A (accented
# European letters, which are legitimate in many real domains).
_SCRIPT_RANGES = {
    "latin": [(0x0041, 0x024F), (0x00C0, 0x00FF)],
    "cyrillic": [(0x0400, 0x04FF)],
    "greek": [(0x0370, 0x03FF)],
}


def _char_script(ch: str) -> str | None:
    cp = ord(ch)
    for script, ranges in _SCRIPT_RANGES.items():
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return script
    return None


def normalize_confusables(s: str) -> str:
    """Maps leetspeak digits/symbols and common homoglyphs to their
    canonical letter. 'g00gle' -> 'google', 'аmazon' (Cyrillic а) ->
    'amazon'. Case-insensitive (lowercases first)."""
    s = s.lower()
    return "".join(CONFUSABLES_MAP.get(c, c) for c in s)


def count_confusable_chars(s: str) -> int:
    """How many characters in s are potential substitution characters -
    a density signal independent of what they might be substituting for."""
    s = s.lower()
    return sum(1 for c in s if c in CONFUSABLES_MAP)


def has_mixed_script(host: str) -> bool:
    """True if the host mixes Latin with Cyrillic/Greek - a strong,
    brand-agnostic signal of an IDN homograph attack. Legitimate domains
    essentially never do this; it's not a stylistic choice anyone makes
    for a real brand."""
    scripts_present = set()
    for c in host:
        if c.isalpha():
            script = _char_script(c)
            if script:
                scripts_present.add(script)
    return len(scripts_present) > 1


def is_punycode(host: str) -> bool:
    """Punycode-encoded (IDNA ASCII form of a non-ASCII domain) - not
    inherently malicious (legitimate international domains use it too),
    but worth surfacing as a feature since it's exactly the mechanism IDN
    homograph attacks rely on."""
    return any(label.startswith("xn--") for label in host.split("."))


def contains_obfuscated_suspicious_term(s: str) -> bool:
    """True if s, AFTER normalization, contains a generic suspicious term
    (secure/verify/login/etc.) - AND s in its RAW form already contains at
    least one substitution character. That second condition matters: it's
    what keeps this from flagging a legitimate word that plainly contains
    'secure' or 'bank' in its ordinary spelling (extremely common in real
    company names) - only the deliberately-obfuscated form is flagged."""
    if count_confusable_chars(s) == 0:
        return False
    normalized = normalize_confusables(s)
    return any(term in normalized for term in GENERIC_SUSPICIOUS_TERMS)

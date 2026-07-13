"""
tests/test_wordplay.py
========================
Tests the general character-substitution/homoglyph defense (core/wordplay.py,
the new features in core/features.py, and the synthetic training
augmentation in core/wordplay_training_data.py) - covers both catching
the attack technique broadly (not just specific brands) and NOT
false-positiving on real legitimate numeric branding.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Novel patterns - deliberately NOT the exact ones in
# core/wordplay_training_data.py's generator, to test generalization
# rather than memorization.
NOVEL_WORDPLAY_ATTACKS = [
    "http://p4ypal-security.info/verify/acc0unt.php",
    "http://microsft-support.online/sign1n",
    "http://netfl1x-billing.xyz/update-p4yment",
    "http://l1nkedin-jobs.top/apply",
]

LEGITIMATE_NUMERIC_BRANDS = [
    "https://1password.com/",
    "https://www.9gag.com/",
    "https://auth0.com/",
    "https://id.me/",
    "https://www.23andme.com/",
    "https://www.office365.com/",
]


def test_novel_wordplay_attacks_caught():
    """The model should generalize to leetspeak patterns it never saw
    during training, not just memorize the synthetic generator's output."""
    for url in NOVEL_WORDPLAY_ATTACKS:
        resp = client.post("/api/check", json={"url": url})
        data = resp.json()
        assert data["verdict"] == "unsafe", f"{url} -> {data}"


def test_legitimate_numeric_brands_not_flagged():
    """1Password, 9gag, Auth0, etc. legitimately contain digits and/or
    normalize to something resembling a suspicious term - must not be
    penalized just for that. Found during testing: 1password.com would
    otherwise trip domain_has_obfuscated_suspicious_term."""
    for url in LEGITIMATE_NUMERIC_BRANDS:
        resp = client.post("/api/check", json={"url": url})
        assert resp.json()["verdict"] == "safe", f"{url} -> {resp.json()}"


def test_homoglyph_domain_detected():
    """Cyrillic 'о' standing in for Latin 'o' - a classic IDN homograph
    attack. Caught via typosquat.py's normalized brand comparison."""
    resp = client.post("/api/check", json={"url": "https://gооgle.com/"})  # Cyrillic о x2
    data = resp.json()
    assert data["verdict"] == "unsafe", data


def test_mixed_script_feature_detects_non_brand_homograph():
    """A mixed-script domain NOT matching any protected brand should still
    be caught by the ML model via the has_mixed_script feature, not just
    by brand-specific typosquat matching."""
    from core.features import extract_features
    f = extract_features("https://sоmerandomsite.com/")  # Cyrillic о
    assert f["has_mixed_script"] == 1


def test_punycode_flag_computed():
    from core.features import extract_features
    f = extract_features("http://xn--80ak6aa92e.com/")
    assert f["is_punycode"] == 1
    f2 = extract_features("https://www.google.com/")
    assert f2["is_punycode"] == 0

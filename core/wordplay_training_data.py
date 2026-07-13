"""
core/wordplay_training_data.py
================================
PhiUSIIL almost certainly has few or no examples of leetspeak/homoglyph
character-substitution phishing (it predates this being flagged as a gap
here), so the new features in core/features.py (has_mixed_script,
is_punycode, num_confusable_chars, domain_has_obfuscated_suspicious_term)
would have no real training signal to learn from without this. This
generates synthetic-but-structurally-realistic examples of the ATTACK
TECHNIQUE - not scraped from anywhere, not modeled on any single real
phishing campaign - purely mechanical substitution applied to generic
terms and (separately) to already-public brand names from our own
allowlist, which is standard, legitimate ML security practice (adversarial
data augmentation), the same category of thing as the SYNTHETIC_SUSPICIOUS
test cases already in tests/.

Equally important: generates LEGITIMATE counter-examples (1Password,
9gag, Web3 Foundation, etc.) so the model learns to distinguish
deliberate obfuscation from ordinary numeric branding, rather than
over-penalizing any digit-near-letters pattern
for the 1Password false-positive this caught during testing.
"""
import random
from core.wordplay import GENERIC_SUSPICIOUS_TERMS

LEET_GEN_MAP = {"e": "3", "a": "4", "o": "0", "i": "1", "s": "5", "t": "7"}

SUSPICIOUS_TLDS = ["tk", "ml", "ga", "cf", "top", "xyz"]
NEUTRAL_TLDS = ["com", "net", "info", "online"]

BRAND_CORES_SAMPLE = [
    "google", "amazon", "instagram", "facebook", "paypal", "flipkart",
    "netflix", "microsoft", "apple", "whatsapp", "linkedin", "dropbox",
]

# Cyrillic homoglyphs for a few brands - real IDN-homograph-style examples
HOMOGLYPH_BRANDS = {
    "google": "gооgle",     # Cyrillic о
    "amazon": "аmazon",     # Cyrillic а
    "paypal": "paypаl",     # Cyrillic а
    "facebook": "facebооk",  # Cyrillic о
}


def _leetspeak_variant(word: str, rate: float = 0.5) -> str:
    """Substitutes SOME (not all) substitutable letters - phishers
    typically obfuscate a character or two, not the whole word, since
    over-obfuscating makes the deception less convincing to a human."""
    chars = list(word)
    substitutable = [i for i, c in enumerate(chars) if c in LEET_GEN_MAP]
    if not substitutable:
        return word
    n_to_sub = max(1, int(len(substitutable) * rate))
    for i in random.sample(substitutable, min(n_to_sub, len(substitutable))):
        chars[i] = LEET_GEN_MAP[chars[i]]
    return "".join(chars)


def generate_phishing_examples() -> list[str]:
    # random.seed() used to live at MODULE IMPORT
    # time, which only seeds the generator ONCE - a second call in the
    # same process (or any other random.* call happening first, which
    # depends on import order) produced DIFFERENT output. Training must
    # be reproducible regardless of call count/order, so the seed is set
    # at the START of every call instead.
    random.seed(42)
    urls = []

    # Generic suspicious terms, leetspeak-obfuscated, in both domain and
    # path positions - this is the general (not brand-specific) case.
    #
    # the original ~4 fixed f-string templates risk
    # teaching a tree ensemble to memorize the template SKELETON (e.g.
    # "-portal." + ".tk") instead of the substitution technique -
    # confirmed by measurement: only 14 distinct structural skeletons
    # across 239 URLs (test_p3_fixes.py collapses every substituted
    # word/tld/id uniformly to reveal this). Expanded to a wider set of
    # structurally DISTINCT shapes: different separators (hyphen vs none
    # vs subdomain), different path depths, different extensions (.php
    # vs .html vs none), different query styles (id= vs ref= vs none vs
    # multi-param), with vs without "www.".
    for term in GENERIC_SUSPICIOUS_TERMS:
        for _ in range(3):
            variant = _leetspeak_variant(term)
            if variant == term:
                continue
            tld = random.choice(SUSPICIOUS_TLDS + NEUTRAL_TLDS)
            other_term = random.choice(GENERIC_SUSPICIOUS_TERMS)
            other_term2 = random.choice(GENERIC_SUSPICIOUS_TERMS)
            rid = random.randint(1000, 9999)

            # domain-level obfuscation - several distinct domain shapes
            urls.append(f"http://{variant}-portal.{tld}/")
            urls.append(f"http://user-{variant}.{tld}/index.php")
            urls.append(f"http://www.{variant}online.{tld}/")
            urls.append(f"http://{variant}.{other_term}-{tld}support.com/")
            urls.append(f"http://my-{variant}-center.{tld}/")

            # path-level obfuscation - several distinct path/query shapes
            urls.append(f"http://myaccount-support.{tld}/{variant}/{other_term}.php?id={rid}")
            urls.append(f"http://{other_term}-alert.{tld}/{variant}.html")
            urls.append(f"http://{other_term}.{tld}/{variant}/{other_term2}/{rid}")
            urls.append(f"http://www.{other_term}-notice.{tld}/action/{variant}?ref={rid}&step=2")
            urls.append(f"http://{other_term}.{tld}/{variant}")
            urls.append(f"http://{tld}-{other_term}.com/{variant}/{other_term2}.php")

    # Brand impersonation via leetspeak - several distinct shapes per brand
    for brand in BRAND_CORES_SAMPLE:
        for _ in range(3):
            variant = _leetspeak_variant(brand, rate=0.4)
            if variant == brand:
                continue
            tld = random.choice(SUSPICIOUS_TLDS + NEUTRAL_TLDS)
            other_term = random.choice(GENERIC_SUSPICIOUS_TERMS)
            urls.append(f"http://{variant}.{tld}/")
            urls.append(f"http://{variant}-support.{tld}/login")
            urls.append(f"http://secure-{variant}.{tld}/account/verify")
            urls.append(f"http://www.{variant}-{other_term}.{tld}/")
            urls.append(f"http://{variant}.{tld}/{other_term}/{other_term}.html")
            urls.append(f"http://help-{variant}.{tld}/{other_term}?case={random.randint(100,999)}")

    # Brand impersonation via Unicode homoglyphs - a couple of shapes each
    for brand, homoglyph_variant in HOMOGLYPH_BRANDS.items():
        tld = random.choice(NEUTRAL_TLDS)
        other_term = random.choice(GENERIC_SUSPICIOUS_TERMS)
        urls.append(f"http://{homoglyph_variant}.{tld}/")
        urls.append(f"http://{homoglyph_variant}.{tld}/login")
        urls.append(f"http://www.{homoglyph_variant}.{tld}/{other_term}/{other_term}")

    # A few realistic punycode-style examples (structurally representative
    # xn-- prefixed hosts, as a real IDN-homograph URL would appear)
    punycode_examples = [
        "xn--pypal-4ve.com", "xn--gogle-qta.com", "xn--mazon-3ve.com",
    ]
    for host in punycode_examples:
        urls.append(f"http://{host}/")
        urls.append(f"http://{host}/secure/login")
        urls.append(f"http://www.{host}/{random.choice(GENERIC_SUSPICIOUS_TERMS)}")

    return urls


def generate_legitimate_counter_examples() -> list[str]:
    """Real, legitimate domains that would trip a naive digit-substitution
    or suspicious-keyword heuristic if the model isn't taught the
    distinction. Found via testing: 1password.com legitimately normalizes
    to contain 'password' after leetspeak normalization - without this,
    the new features would likely over-penalize real numeric branding."""
    return [
        "https://1password.com/",
        "https://1password.com/features",
        "https://www.9gag.com/",
        "https://web3.foundation/",
        "https://auth0.com/",
        "https://auth0.com/docs",
        "https://id.me/",
        "https://www.23andme.com/",
        "https://c3.ai/",
        "https://www.office365.com/",
        "https://www.windows11.com/",
        "https://www.7-eleven.com/",
        "https://www.4chan.org/",
        "https://www.w3.org/",
        "https://www.3m.com/",
        "https://www.20thcenturystudios.com/",
    ]

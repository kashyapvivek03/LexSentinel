"""
tests/test_p3_fixes.py
========================
Regression tests for the P3 tier: 3.1 (realistic held-out
evaluation set), 3.2 (synthetic wordplay data diversity + reproducibility),
3.5 (feature-constant versioning in model metadata), 2.2 (scalable
allowlist/blocklist matching).

Note on test cost: 3.1's real evaluation runs inference (cheap) against
the ALREADY-TRAINED current model, not a fresh training run - retraining
is not part of this suite (too slow for every test invocation). 3.2's
feature-importance verification (does num_confusable_chars/has_mixed_script
actually carry gain) was done as a one-time manual check while
implementing the fix, not baked into the permanent suite for the same
reason - noted explicitly rather than silently skipped.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time


# ---------------------------------------------------------------- 2.2 ----
def test_list_matching_correctness_matches_old_linear_semantics():
    """Review 2.2: the new (fast) matching must return IDENTICAL results
    to the old (correct but slow) linear scan for every case that
    mattered - exact match, subdomain match, and the dot-boundary
    exclusion (evil-wikipedia.org must NOT match wikipedia.org)."""
    from core.lists import is_allowlisted, _host_matches_entry, _load
    domain_set = _load("allowlist")["_domain_set"]

    test_hosts = [
        "google.com", "www.google.com", "mail.google.com",
        "evil-google.com", "googlecom", "notgoogle.com",
        "en.wikipedia.org", "evil-wikipedia.org", "wikipediaorg.com",
        "a.b.c.google.com", "sbi.co.in", "evil.sbi.co.in", "sbi.co.in.evil.com",
    ]
    for host in test_hosts:
        old_result = any(_host_matches_entry(host, e) for e in domain_set)
        new_result = is_allowlisted("https://" + host + "/")
        assert old_result == new_result, f"Mismatch for {host}: old={old_result} new={new_result}"


def test_list_matching_scales_independent_of_list_size():
    """Review 2.2: at Tranco-Top-1M scale, a linear scan becomes a
    million-iteration check per URL. The fix must be near-constant-time
    regardless of list size - verified by building a large synthetic
    list and confirming lookup time doesn't scale with it."""
    from core.lists import _matches_any

    small_set = {f"legit-domain-{i}.com" for i in range(100)}
    large_set = {f"legit-domain-{i}.com" for i in range(200_000)}
    large_set.add("findme.com")

    # warm up
    _matches_any("nomatch.com", small_set)

    t0 = time.perf_counter()
    for _ in range(200):
        _matches_any("nomatch-at-all.example.org", large_set)
    large_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(200):
        _matches_any("nomatch-at-all.example.org", small_set)
    small_elapsed = time.perf_counter() - t0

    # generous margin (not a strict big-O proof, just guards against an
    # accidental regression back to linear scan) - large set must not be
    # dramatically slower than the small one.
    assert large_elapsed < small_elapsed * 20 + 0.5, (
        f"Matching time scales with list size: small={small_elapsed:.4f}s large={large_elapsed:.4f}s"
    )
    assert _matches_any("www.findme.com", large_set) is True


# ---------------------------------------------------------------- 3.2 ----
def test_wordplay_generator_is_reproducible_across_calls():
    """Review 3.2: random.seed(42) at MODULE import time only seeds once -
    calling the generator a second time in the same process (or after any
    other random.* call) produces different output. Training must be
    reproducible regardless of call order/count."""
    from core.wordplay_training_data import generate_phishing_examples
    a = generate_phishing_examples()
    b = generate_phishing_examples()
    assert a == b, "Generator is not reproducible across repeated calls"


def test_wordplay_generator_has_diverse_structural_templates():
    """Review 3.2: ~4-8 fixed f-string templates risk teaching a tree
    ensemble to memorize the template SKELETON (e.g. '-portal.' + '.tk')
    instead of the substitution technique. Measured by collapsing every
    substituted word/variant/tld/id to a placeholder, uniformly - what
    remains is the pure structural skeleton (dots, hyphens, slashes,
    ?/&/= placement). Before the fix: 14 distinct skeletons across 239
    URLs. Must be meaningfully higher after diversifying."""
    import re
    from core.wordplay_training_data import generate_phishing_examples
    urls = generate_phishing_examples()

    def structural_shape(url: str) -> str:
        return re.sub(r"[A-Za-z0-9]+", "W", url)

    shapes = {structural_shape(u) for u in urls}
    assert len(shapes) >= 25, f"Only {len(shapes)} distinct structural skeletons found: {shapes}"


# ---------------------------------------------------------------- 3.5 ----
def test_feature_constants_are_fingerprinted_in_model_metadata():
    """Review 3.5: COMMON_TLDS/SUSPICIOUS_PATH_KEYWORDS are code constants;
    changing them silently invalidates a trained model (feature semantics
    shift under a frozen model) with no way to detect the mismatch at
    serve time. A fingerprint of the constants used at train time must be
    recorded in metadata, and a live-vs-recorded check must exist."""
    from core.registry import load_current_model
    from core.features import feature_constants_fingerprint

    _, metadata = load_current_model()
    assert "feature_constants_fingerprint" in metadata, (
        "Model metadata does not record a feature-constants fingerprint"
    )
    live_fingerprint = feature_constants_fingerprint()
    assert metadata["feature_constants_fingerprint"] == live_fingerprint, (
        "Live COMMON_TLDS/SUSPICIOUS_PATH_KEYWORDS do not match what the "
        "currently-loaded model was trained with - feature-list drift, "
        "model should be retrained"
    )


def test_fingerprint_changes_when_constants_change():
    """The fingerprint function must actually be sensitive to the
    constants - not a no-op that always returns the same value."""
    from core.features import _fingerprint_constants
    a = _fingerprint_constants(["login", "verify"], {"com", "org"})
    b = _fingerprint_constants(["login", "verify", "extra"], {"com", "org"})
    assert a != b


# ---------------------------------------------------------------- 3.1 ----
def test_realistic_evaluation_module_exists_and_runs():
    """Review 3.1: PhiUSIIL's test split has the SAME artifacts
    (bare-homepage-only benign class) training worked around, so reported
    AUC measures performance on an artifact-laden distribution, never the
    realistic one. models/evaluate.py must exist and produce metrics on a
    held-out set that never touched training."""
    import importlib
    evaluate = importlib.import_module("models.evaluate")
    assert hasattr(evaluate, "REALISTIC_HELDOUT_URLS"), (
        "No realistic held-out URL set defined"
    )
    assert hasattr(evaluate, "run_evaluation"), "No run_evaluation() function found"

    # The held-out set must be genuinely disjoint from what training uses,
    # or this isn't testing anything the artifact-laden PhiUSIIL split
    # doesn't already cover.
    from core.augmentation_data import REAL_BENIGN_URLS_WITH_PATHS, REAL_BENIGN_ROOT_URLS_WITH_TRAILING_SLASH
    training_urls = set(REAL_BENIGN_URLS_WITH_PATHS) | set(REAL_BENIGN_ROOT_URLS_WITH_TRAILING_SLASH)
    heldout_urls = {u for u, _ in evaluate.REALISTIC_HELDOUT_URLS}
    overlap = training_urls & heldout_urls
    assert not overlap, f"Held-out set overlaps with training augmentation data: {overlap}"

    report = evaluate.run_evaluation()
    assert "phiusiil_test" in report
    assert "realistic_heldout" in report
    for key in ("accuracy", "precision", "recall", "f1"):
        assert key in report["realistic_heldout"], f"Missing metric {key} in realistic_heldout report"

"""
tests/test_feature_parity.py
=============================
Trivially true today (extract_features_batch literally calls
extract_features per row) - that's the point. This test exists to catch
FUTURE regressions: if someone "optimizes" the batch path later and it
silently diverges from the single-URL path, this fails immediately
instead of surfacing as a mystery production bug.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.features import extract_features, extract_features_batch, FEATURE_NAMES

SAMPLE_URLS = [
    "https://www.google.com/",
    "http://192.168.1.1/wp-admin/login.php?redirect=x",
    "https://en.wikipedia.org/wiki/Main_Page",
    "https://accounts.google.com/signin/v2/identifier?service=mail",
]


def test_batch_matches_single_row_exactly():
    batch_df = extract_features_batch(SAMPLE_URLS)
    for i, url in enumerate(SAMPLE_URLS):
        single = extract_features(url)
        for col in FEATURE_NAMES:
            assert batch_df.iloc[i][col] == single[col], (
                f"PARITY BREAK on '{url}', feature '{col}': "
                f"batch={batch_df.iloc[i][col]!r} single={single[col]!r}. "
                "Training and serving would now see different features."
            )

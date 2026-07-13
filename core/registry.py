"""
core/registry.py
=================
Fixes audit finding: "ML model likely referenced with relative path based
on prior inspection... Resolve paths using application root (e.g., pathlib)."

Every path here is derived from THIS FILE's own location, never from
os.getcwd(). It doesn't matter what directory you launch the app from -
`python app/main.py`, a systemd unit, a Docker WORKDIR, a test runner from
a different folder - the model always resolves the same way.

Also fixes: "two model files, which one is real?" There is exactly one
entry point: current.json, written atomically by train.py, read here.
"""
from __future__ import annotations
import json
from pathlib import Path
from functools import lru_cache
import joblib
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"

# The safe/unsafe operating point applied to the model's phishing
# probability. Lives HERE, next to model loading, because both the
# serving layer (app/main.py) and the offline evaluation
# (models/evaluate.py) must apply the SAME cutoff - otherwise reported
# metrics stop describing what the product actually does.
#
# TODO (deferred - a prior review's own suggestion): 0.5 is almost never
# the right operating point for a security product with asymmetric
# false-positive/false-negative costs, and there's no "uncertain" band -
# a 50.1% score renders identically to 99.9%. Tuning this from the
# validation PR curve at a target precision, and adding a three-way
# verdict (safe/suspicious/unsafe), is real follow-up work.
DECISION_THRESHOLD = 0.5


class ModelNotFoundError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_current_model():
    """Loads the model pointed to by current.json and reconstructs a
    Pipeline-like object. Cached so repeated calls don't re-deserialize;
    call load_current_model.cache_clear() after retraining to hot-swap,
    or POST /api/admin/reload (dev-key gated) which does this for you.

    Loaded from TWO files, not one: the sklearn preprocessing (joblib,
    portable) and the XGBoost model (native UBJSON via load_model(),
    explicitly cross-platform-safe). A single joblib blob containing the
    whole pipeline embeds XGBoost's raw internal buffer, which is NOT
    guaranteed portable across operating systems even at an identical
    library version - that's what caused "input stream corrupted" when
    serving a Linux-trained model on Windows. Don't go back to one blob."""
    pointer_path = ARTIFACTS_DIR / "current.json"
    if not pointer_path.exists():
        raise ModelNotFoundError(
            f"No current.json at {pointer_path}. Run models/train.py first."
        )
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    preprocessor_path = ARTIFACTS_DIR / pointer["preprocessor_file"]
    xgb_path = ARTIFACTS_DIR / pointer["xgb_model_file"]
    meta_path = ARTIFACTS_DIR / pointer["metadata_file"]
    # this used to only check preprocessor/xgb
    # existence, not metadata_file - a half-written current.json (e.g. a
    # crashed/interrupted train.py run) raised a raw FileNotFoundError
    # when meta_path.read_text(encoding="utf-8") was called below, which propagates as
    # an unhandled 500 instead of the intended ModelNotFoundError -> 503.
    if not preprocessor_path.exists() or not xgb_path.exists() or not meta_path.exists():
        raise ModelNotFoundError(
            f"current.json points to missing file(s): {preprocessor_path} / {xgb_path} / {meta_path}"
        )

    preprocessor = joblib.load(preprocessor_path)
    clf = XGBClassifier()
    clf.load_model(str(xgb_path))
    pipeline = Pipeline([("features", preprocessor), ("clf", clf)])
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return pipeline, metadata

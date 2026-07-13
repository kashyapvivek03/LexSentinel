"""
models/evaluate.py
====================
PhiUSIIL's test split has the SAME distribution
artifacts training worked around (100% bare-homepage benign class,
100% www-prefixed, etc.). Reported
ROC-AUC measures performance on an artifact-laden distribution, not real
traffic. Meanwhile the augmentation URLs (the ones actually shaped like
real-world traffic) go ONLY into training - the model is never *scored*
on the distribution it was patched to handle.

This adds a second, independent evaluation: a small held-out set of real
legitimate URLs (fresh search results, disjoint from every URL in
core/augmentation_data.py) and freshly-written phishing-shaped URLs
(disjoint from core/wordplay_training_data.py's generator output and
from tests/test_regression_known_sites.py's SYNTHETIC_SUSPICIOUS_CASES).
Neither list is used anywhere in training - this is a genuine held-out
check, not a re-measurement of training performance.

Run directly: python models/evaluate.py
Or import run_evaluation() for programmatic use (e.g. a future CI step).
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from core.registry import load_current_model, DECISION_THRESHOLD
from core.features import extract_features_batch

# ---------------------------------------------------------------------
# Real legitimate URLs, freshly pulled (2026-07-08) from two categories
# NOT covered anywhere in core/augmentation_data.py (freelance/gig-work
# taxes, puppy obedience training) - genuinely held-out, not a re-slice
# of what training already saw.
# ---------------------------------------------------------------------
_FRESH_LEGITIMATE_URLS = [
    "https://www.irs.gov/businesses/small-businesses-self-employed/self-employed-individuals-tax-center",
    "https://turbotax.intuit.com/tax-tips/self-employment-taxes/a-freelancers-guide-to-taxes/L6ACNfKVW",
    "https://quickbooks.intuit.com/r/taxes/freelance-taxes/",
    "https://www.jacksonhewitt.com/tax-help/tax-tips-topics/employment/budget-for-freelance-taxes/",
    "https://1800accountant.com/blog/taxes-for-freelancers-what-the-self-employed-need-to-know",
    "https://www.sdocpa.com/freelancer-tax-guide/",
    "https://www.eztaxreturn.com/blog/freelance-taxes-101-a-guide-to-paying-taxes-when-youre-self-employed/",
    "https://www.nerdwallet.com/taxes/learn/freelance-taxes",
    "https://www.irs.gov/businesses/small-businesses-self-employed/manage-taxes-for-your-gig-work",
    "https://www.collective.com/blog/tax-tips/a-freelancers-guide-to-getting-taxes-right",
    "https://www.akc.org/expert-advice/training/teach-your-puppy-these-5-basic-commands/",
    "https://www.diggs.pet/blogs/posts/puppy-obedience-training",
    "https://dogsinc.org/blog/ask-the-trainer/puppy-school-13-commands/",
    "https://www.thepuppyacademy.com/blog/2021/2/1/puppy-training-101-giving-your-puppy-commands-the-right-way",
    "https://www.petlandflorida.com/how-to-teach-your-puppy-the-5-basic-commands/",
    "https://www.eukanuba.com/au/articles/training/the-top-10-commands-to-teach-your-puppy-first",
    "https://www.pawcbd.com/blogs/posts/dog-commands-for-puppy-obedience-training",
    "https://www.pdsa.org.uk/pet-help-and-advice/looking-after-your-pet/puppies-dogs/basic-training-for-puppies",
    "https://www.doggoneproblems.com/miles/",
]

# ---------------------------------------------------------------------
# Freshly written phishing-shaped URLs (2026-07-08) - different domains,
# TLDs, and topical hooks than core/wordplay_training_data.py's generator
# output and tests/test_regression_known_sites.py's
# SYNTHETIC_SUSPICIOUS_CASES, so this isn't re-testing the same examples
# the model was trained/tested on elsewhere. Structurally representative
# of real-world lure categories: shipping/delivery, tax-refund, HR/payroll,
# subscription-renewal, cloud-storage-share, IP-based.
# ---------------------------------------------------------------------
_FRESH_PHISHING_URLS = [
    "http://track-your-package.delivery-notice.top/parcel/confirm.php?id=8834920",
    "http://irs-taxrefund-status.online/refund/claim.php?ref=aa8391kd",
    "http://payroll-hr-portal.xyz/employee/verify.php?emp=00219",
    "http://www.netflix-billing-update.info/subscription/renew?token=7f2a9c",
    "http://drive-shared-file.cloudstorage-alert.top/view.php?doc=q9281",
    "http://192.168.4.22/employee/payroll/login.php",
    "http://amaz0n-orderissue.com/account/resolve.php?order=113-2938471",
    "http://c0nfirm-your-package.tk/tracking/update?id=3321",
    "http://hr-benefits-enroll1ment.net/portal/signin.php?dept=finance",
    "http://secure-cloudfile-share.ga/document/access.php?ref=91ka2",
    "http://tax-refund-cla1m.ml/status/verify.php?ssn_last4=incomplete",
    "http://www.paypa1-disputecenter.com/case/review.php?case=772819",
    "http://microsft-teams-invite.online/meeting/join?id=4471-teams",
    "http://apple-idlocked.top/unlock/verify.php?device=iphone15",
    "http://bank0famerica-alert.info/account/secure.php?ref=abc123def",
]

REALISTIC_HELDOUT_URLS: list[tuple[str, int]] = (
    [(u, 0) for u in _FRESH_LEGITIMATE_URLS]
    + [(u, 1) for u in _FRESH_PHISHING_URLS]
)


def _compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    # FPR/FNR computed directly (not via sklearn's confusion_matrix) so a
    # single-class edge case (e.g. an all-benign URL list) can't crash on
    # matrix shape. FPR = benign URLs wrongly called phishing / all benign;
    # FNR = phishing wrongly called safe / all phishing - the two costs a
    # phishing checker actually trades off (a bare accuracy number hides
    # which side is failing).
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    n_benign = sum(1 for t in y_true if t == 0)
    n_phish = sum(1 for t in y_true if t == 1)
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": (fp / n_benign) if n_benign else 0.0,
        "false_negative_rate": (fn / n_phish) if n_phish else 0.0,
        "n_false_positives": fp,
        "n_false_negatives": fn,
    }


def _evaluate_realistic_heldout(pipeline) -> dict:
    urls = [u for u, _ in REALISTIC_HELDOUT_URLS]
    y_true = [label for _, label in REALISTIC_HELDOUT_URLS]
    feats_df = extract_features_batch(urls)
    probs = pipeline.predict_proba(feats_df)[:, 1]
    # Same cutoff the serving layer applies - a hardcoded 0.5 here would
    # silently diverge from production the day the threshold is tuned.
    y_pred = [1 if p >= DECISION_THRESHOLD else 0 for p in probs]
    metrics = _compute_metrics(y_true, y_pred)
    metrics["misclassified"] = [
        {"url": u, "true_label": t, "predicted": p, "probability": float(prob)}
        for u, t, p, prob in zip(urls, y_true, y_pred, probs) if t != p
    ]
    return metrics


def _evaluate_phiusiil_test(pipeline) -> dict:
    """Reproduces the same held-out split train.py uses, IF the dataset
    file is present. Gracefully reports unavailable rather than failing -
    the 56MB PhiUSIIL CSV isn't always present in every environment this
    might run in (e.g. CI), and the realistic_heldout evaluation above
    needs no external data at all."""
    import os
    from sklearn.model_selection import train_test_split
    ROOT = Path(__file__).resolve().parents[1]
    data_path = Path(os.environ.get(
        "PHISHING_DETECTOR_DATASET", ROOT / "dataset" / "PhiUSIIL_Phishing_URL_Dataset.csv"
    ))
    if not data_path.exists():
        return {"status": "unavailable", "reason": f"dataset not found at {data_path}"}

    raw = pd.read_csv(data_path)
    y = 1 - raw["label"]  # label=1 is legitimate in this file
    _, test_idx = train_test_split(
        raw.index, test_size=0.2, random_state=42, stratify=y
    )
    test_urls = raw.loc[test_idx, "URL"].tolist()
    y_true = y.loc[test_idx].tolist()
    feats_df = extract_features_batch(test_urls)
    probs = pipeline.predict_proba(feats_df)[:, 1]
    y_pred = [1 if p >= DECISION_THRESHOLD else 0 for p in probs]
    return _compute_metrics(y_true, y_pred)


def run_evaluation() -> dict:
    pipeline, metadata = load_current_model()
    return {
        "model_version": metadata["version"],
        "phiusiil_test": _evaluate_phiusiil_test(pipeline),
        "realistic_heldout": _evaluate_realistic_heldout(pipeline),
    }


if __name__ == "__main__":
    import json
    report = run_evaluation()
    print(json.dumps(report, indent=2))
    print("\n--- Summary ---")
    print(f"Model version: {report['model_version']}")
    ph = report["phiusiil_test"]
    if ph.get("status") == "unavailable":
        print(f"PhiUSIIL test:      unavailable ({ph['reason']})")
    else:
        print(f"PhiUSIIL test:      acc={ph['accuracy']:.3f} prec={ph['precision']:.3f} "
              f"recall={ph['recall']:.3f} f1={ph['f1']:.3f}  (n={ph['n']})")
    rh = report["realistic_heldout"]
    print(f"Realistic held-out: acc={rh['accuracy']:.3f} prec={rh['precision']:.3f} "
          f"recall={rh['recall']:.3f} f1={rh['f1']:.3f}  (n={rh['n']})")
    if rh["misclassified"]:
        print(f"\nMisclassified on realistic held-out set ({len(rh['misclassified'])}):")
        for m in rh["misclassified"]:
            print(f"  true={m['true_label']} pred={m['predicted']} p={m['probability']:.3f}  {m['url']}")

#!/usr/bin/env bash
# Closes packaging findings: excludes .git, IDE metadata, logs, the venv,
# the secret dev key, the (large, redistributable) dataset, and .env from
# the release archive. Run from the project root.
#
# PROJECT_REVIEW.md 5.6: this used to exclude LESS than .gitignore did,
# so a release tarball built from a working directory shipped the secret
# dev key, the venv, and the dataset. If you want a release built strictly
# from tracked files (the more robust option), use `git archive` instead -
# it respects .gitignore-tracked state exactly and can't accidentally
# include an untracked secret sitting in the working directory.
set -euo pipefail
VERSION=$(python3 -c "import json; print(json.load(open('models/artifacts/current.json'))['version'])" 2>/dev/null || echo "unversioned")
OUT="release_${VERSION}.tar.gz"
tar --exclude='.git' --exclude='.claude' --exclude='__pycache__' \
    --exclude='logs' --exclude='*.log' --exclude='.pytest_cache' \
    --exclude='venv' --exclude='config/dev_key.txt' --exclude='dataset' \
    --exclude='.env' --exclude='x' \
    -czf "$OUT" .
echo "Wrote $OUT"

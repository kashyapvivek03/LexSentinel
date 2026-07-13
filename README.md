# LexSentinel — Phishing URL Detector

A machine-learning-based phishing URL detection system, built as a final-year
B.Tech project (Asansol Engineering College, MAKAUT). Full academic writeup:
`final_year_thesis_report.pdf` (also available as `LexSentinel_Thesis_Report.docx`).

## Summary

LexSentinel classifies URLs as benign or phishing in real time, combining a
trained gradient-boosted (XGBoost) classifier with rule-based signal layers —
allowlists/blocklists, typosquat detection (e.g. catching `sbl.co.in`
impersonating `sbi.co.in`), and general character-substitution/homoglyph
("wordplay") detection. It ships as a FastAPI backend, a web UI for
single and bulk URL checks, and a Manifest V3 Chrome extension that checks
every site a user visits automatically before it loads.

**Dataset & training**: models are trained on the PhiUSIIL Phishing URL
Dataset, with additional benign-with-path URLs augmented in to counter
false positives seen on real-world sites. Features are extracted from the
URL string alone (lexical/structural features — no live page fetch needed
at inference time), through a single canonical feature-extraction module
shared identically between training and serving code, so the two can never
drift apart.

**Model comparison**: the thesis benchmarks Logistic Regression, Random
Forest, and XGBoost head-to-head on the same data; XGBoost was selected and
shipped. Reported results: ~98.3% accuracy on the PhiUSIIL test split and
~94% accuracy on a realistic, independently constructed held-out set (URLs
disjoint from training/augmentation data). Feature-importance analysis
shows `is_https` as the single most influential signal (~46% gain),
flagged in the report as a limitation — reliance on HTTPS presence alone is
not fully robust as attackers increasingly use valid TLS certificates too.

**Engineering hardening**: beyond the initial model, the system went
through a technical audit, an independent red-team security assessment,
and a 100,000-URL evaluation pass — fixing issues like inconsistent bulk
paste parsing, typosquat false positives on legitimate subdomains
(`id.atlassian.com`, `outlook.live.com`), a rate-limiting bypass via
spoofed `X-Forwarded-For` headers, and API response inconsistencies. See
`PROJECT_UPGRADE_REPORT.md` for the detailed changelog.

**Report structure** (9 chapters): Preface, Literature Review, Dataset
Analysis, Theories/Algorithms, Proposed Framework & Methodology,
Technology Used, Implementation & Results, Conclusion, References.

## Features at a glance

- Single URL check (`POST /api/check`) and bulk checking (paste up to 50
  URLs, or upload a `.txt`/`.csv` file up to 75 URLs) with CSV/Excel export
- Browser extension that auto-checks every page you browse to
- Developer-only hot-reload endpoint for swapping in a retrained model
  without downtime
- Regression test suite that pins down previously-misclassified real sites
  (`perplexity.ai`, `discord.com`, `india.gov.in`, `icici.bank.in`, etc.)

## How to download and run this on your own computer

These steps assume no prior setup — just a computer with internet access.

### 1. Install prerequisites

- **Git**: [git-scm.com/downloads](https://git-scm.com/downloads)
- **Python 3.10+**: [python.org/downloads](https://python.org/downloads) —
  on Windows, tick "Add Python to PATH" during install.

Verify both installed correctly by opening a terminal (Command Prompt,
PowerShell, or Terminal on Mac/Linux) and running:

```bash
git --version
python --version
```

### 2. Download the project from GitHub

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

(Replace the URL with this project's actual GitHub repository URL. If you
don't have it in GitHub yet, this project's own README section above
under "Browser extension" explains how to push it there first.)

### 3. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it:

- Windows (PowerShell): `venv\Scripts\Activate.ps1`
- Windows (cmd.exe): `venv\Scripts\activate.bat`
- Mac/Linux: `source venv/bin/activate`

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the server

```bash
uvicorn app.main:app --reload --port 8000
```

### 6. Use it

Open a browser and go to:

```
http://localhost:8000/
```

You'll see the web UI where you can paste a single URL, or click the **+**
next to the search bar to bulk-check multiple URLs at once (paste a list,
or upload a `.txt`/`.csv` file).

### 7. (Optional) Retrain the model yourself

If you want to reproduce the training pipeline instead of using the
committed model artifact:

```bash
python models/train.py
```

This reads `dataset/PhiUSIIL_Phishing_URL_Dataset.csv`, extracts features,
trains a fresh model, and saves a new versioned artifact under
`models/artifacts/`.

### 8. (Optional) Install the browser extension

Load `extension/` as an unpacked Manifest V3 extension in Chrome
(`chrome://extensions` → enable Developer Mode → "Load unpacked" → select
the `extension/` folder). By default it needs a backend reachable over the
internet, not `localhost` — see the "Browser extension" section of the
project's technical documentation (or `extension/README.md`) for deploying
a free always-on backend via Render.

### 9. Run the test suite (optional, for developers)

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

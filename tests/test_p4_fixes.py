"""
tests/test_p4_fixes.py
========================
Regression tests for PROJECT_REVIEW.md P4 tier: 5.2 (dangling doc
references - verified as part of the fix itself, spot-checked here too),
5.4 (requirements split), 5.5 (CI workflow), 5.7 (embedded HTML moved to
real static files), 5.8 (structured logging, domain-only for privacy).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging


def test_no_dangling_doc_references_remain():
    """5.2: code comments referenced AUDIT_NOTES.md/PROJECT_REVIEW.md/etc.
    in ~60 places, but those files are gitignored and not in the public
    repo - every reference was a dead link for a collaborator. All must
    be gone or self-contained now."""
    import re
    root = Path(__file__).resolve().parents[1]
    pattern = re.compile(r"AUDIT_NOTES\.md|SECURITY_ASSESSMENT\.md|MODEL_EVALUATION\.md|PROJECT_REVIEW\.md")
    offenders = []
    for pyfile in list(root.glob("core/*.py")) + list(root.glob("app/*.py")) + \
                  list(root.glob("models/*.py")) + list(root.glob("extension/*.js")):
        text = pyfile.read_text(encoding="utf-8")
        if pattern.search(text):
            offenders.append(str(pyfile))
    assert not offenders, f"Dangling doc references remain in: {offenders}"


def test_requirements_split_runtime_vs_dev():
    """5.4: requirements.txt mixed runtime and dev-only deps (pytest ships
    to production). Split into requirements.txt (runtime) and
    requirements-dev.txt (adds test tooling)."""
    root = Path(__file__).resolve().parents[1]
    runtime = (root / "requirements.txt").read_text(encoding="utf-8")
    assert "pytest" not in runtime, "pytest should not be in runtime requirements.txt"
    assert "httpx" not in runtime, "httpx should not be in runtime requirements.txt"
    assert "fastapi" in runtime, "fastapi (an actual runtime dep) missing from requirements.txt"

    dev_path = root / "requirements-dev.txt"
    assert dev_path.exists(), "requirements-dev.txt does not exist"
    dev = dev_path.read_text(encoding="utf-8")
    assert "pytest" in dev
    assert "httpx" in dev


def test_python_version_pinned_for_deployment():
    """5.4: nothing pinned the Python version; Render/other host defaults
    may silently differ from what this was built/tested against."""
    root = Path(__file__).resolve().parents[1]
    runtime_txt = root / "runtime.txt"
    assert runtime_txt.exists(), "No runtime.txt pinning the Python version"
    assert "3.12" in runtime_txt.read_text(encoding="utf-8")


def test_ci_workflow_exists_and_runs_pytest():
    """5.5: tests existed but nothing ran them automatically."""
    import yaml
    root = Path(__file__).resolve().parents[1]
    workflow_path = root / ".github" / "workflows" / "tests.yml"
    assert workflow_path.exists(), "No CI workflow found"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    assert "jobs" in workflow
    workflow_text = workflow_path.read_text(encoding="utf-8")
    assert "pytest" in workflow_text, "CI workflow does not appear to run pytest"


def test_html_extracted_from_python_strings():
    """5.7: ~250 lines of HTML/JS were embedded as Python triple-quoted
    strings in app/main.py - not editable, lintable, or
    syntax-highlighted as HTML/JS. Moved to a real static file."""
    root = Path(__file__).resolve().parents[1]
    static_dir = root / "app" / "static"
    assert (static_dir / "index.html").exists()

    main_py = (root / "app" / "main.py").read_text(encoding="utf-8")
    assert "_INDEX_HTML" not in main_py, "Embedded HTML string still present in main.py"

    # and the app must actually still serve it correctly
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    r1 = client.get("/")
    assert r1.status_code == 200 and "<title>" in r1.text


def test_logging_emits_domain_and_verdict_never_full_url(caplog):
    """5.8: zero logging existed anywhere except print() in auth.py.
    Added structured logging of verdict/stage - domain only, NEVER the
    full URL (path/query can carry search terms, tokens, session data -
    same privacy reasoning as the extension's query-string stripping)."""
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    sensitive_path = "/reset-password?token=supersecret12345"
    test_url = f"https://example.com{sensitive_path}"

    with caplog.at_level(logging.INFO, logger="phishing_detector"):
        client.post("/api/check", json={"url": test_url})

    assert len(caplog.records) > 0, "No log record emitted for a check"
    log_text = "\n".join(r.message for r in caplog.records)
    assert "example.com" in log_text, "Domain should be logged"
    assert "supersecret12345" not in log_text, "Query string leaked into logs"
    assert "reset-password" not in log_text, "Path leaked into logs"
    assert "verdict=" in log_text
    assert "stage=" in log_text

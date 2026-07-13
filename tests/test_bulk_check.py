"""
tests/test_bulk_check.py
=========================
Covers the public bulk-checker feature that replaced the old dev-only
/dev/bulk page: bulk-check-paste (pasted text, no auth), bulk-check-upload
(.txt/.csv file, no auth), and bulk-check-export (CSV/XLSX download built
in memory from results the browser already has). No X-Dev-Key is required
anywhere here - see test_p2_fixes.py / test_security_fixes.py for the
still-gated /api/admin/reload.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import io
from fastapi.testclient import TestClient
from app.main import app, MAX_PASTE_URLS, MAX_FILE_URLS

client = TestClient(app)

TEST_URLS = [
    "https://www.google.com/",
    "https://www.sbl.co.in/",
    "http://192.168.10.5/wp-admin/login.php?redirect=confirm",
]


# --------------------------------------------------------- bulk-check-paste --
def test_bulk_paste_basic_check():
    resp = client.post("/api/bulk-check-paste", json={"text": "\n".join(TEST_URLS)})
    assert resp.status_code == 200
    data = resp.json()
    assert data["summary"]["total"] == len(TEST_URLS)
    verdicts = {r["checked_url"]: r["verdict"] for r in data["results"]}
    assert verdicts["https://www.google.com/"] == "safe"
    assert verdicts["https://www.sbl.co.in/"] == "unsafe"
    assert verdicts["http://192.168.10.5/wp-admin/login.php?redirect=confirm"] == "unsafe"


def test_bulk_paste_reason_only_set_for_unsafe():
    """reason is a plain-language 'why', shown to non-technical users -
    it must be present for every unsafe row and absent for safe/invalid
    ones (a reason next to 'safe' would read as unearned suspicion)."""
    resp = client.post(
        "/api/bulk-check-paste",
        json={"text": "\n".join(TEST_URLS + ["not a url"])},
    )
    assert resp.status_code == 200
    by_url = {r["checked_url"]: r for r in resp.json()["results"]}

    safe_row = by_url["https://www.google.com/"]
    assert safe_row["verdict"] == "safe"
    assert safe_row["reason"] is None

    unsafe_row = by_url["https://www.sbl.co.in/"]
    assert unsafe_row["verdict"] == "unsafe"
    assert unsafe_row["reason"], "Unsafe row must have a plain-language reason"
    assert "sbi" in unsafe_row["reason"].lower() or "look" in unsafe_row["reason"].lower()

    invalid_row = by_url["not a url"]
    assert invalid_row["status"] == "invalid"
    assert invalid_row["reason"] is None


def test_bulk_paste_matches_single_check_no_drift():
    """The paste path shares _decide_stage1 and the same model call
    pattern as /api/check - this pins that they can't silently diverge."""
    bulk_resp = client.post("/api/bulk-check-paste", json={"text": "\n".join(TEST_URLS)})
    bulk_by_url = {r["checked_url"]: r for r in bulk_resp.json()["results"]}
    for url in TEST_URLS:
        single = client.post("/api/check", json={"url": url}).json()
        assert single["verdict"] == bulk_by_url[url]["verdict"]
        assert single["stage"] == bulk_by_url[url]["stage"]


def test_bulk_paste_handles_comma_separated():
    resp = client.post("/api/bulk-check-paste", json={"text": ",".join(TEST_URLS)})
    assert resp.status_code == 200
    assert resp.json()["summary"]["total"] == len(TEST_URLS)


def test_bulk_paste_handles_mixed_newline_and_comma():
    text = f"{TEST_URLS[0]}, {TEST_URLS[1]}\n{TEST_URLS[2]}"
    resp = client.post("/api/bulk-check-paste", json={"text": text})
    assert resp.status_code == 200
    assert resp.json()["summary"]["total"] == 3


def test_bulk_paste_deduplicates():
    text = "\n".join([TEST_URLS[0], TEST_URLS[0], TEST_URLS[0]])
    resp = client.post("/api/bulk-check-paste", json={"text": text})
    assert resp.status_code == 200
    assert resp.json()["summary"]["total"] == 1


def test_bulk_paste_marks_non_urls_invalid_instead_of_scoring_them():
    resp = client.post(
        "/api/bulk-check-paste",
        json={"text": "https://www.google.com/\ndefinitely not a url"},
    )
    assert resp.status_code == 200
    data = resp.json()
    by_url = {r["checked_url"]: r for r in data["results"]}
    bad = by_url["definitely not a url"]
    assert bad["status"] == "invalid"
    assert bad["verdict"] is None
    assert data["summary"]["invalid"] == 1
    assert by_url["https://www.google.com/"]["verdict"] == "safe"


def test_bulk_paste_empty_text_rejected():
    resp = client.post("/api/bulk-check-paste", json={"text": ""})
    assert resp.status_code == 422  # pydantic min_length=1


def test_bulk_paste_whitespace_only_rejected():
    resp = client.post("/api/bulk-check-paste", json={"text": "   \n\n  ,, \n"})
    assert resp.status_code == 400


def test_bulk_paste_rejects_whole_request_over_cap():
    """Exceeding MAX_PASTE_URLS must reject the WHOLE request with a clear
    message, not silently process just the first N."""
    too_many = "\n".join(f"https://example{i}.com/" for i in range(MAX_PASTE_URLS + 1))
    resp = client.post("/api/bulk-check-paste", json={"text": too_many})
    assert resp.status_code == 400
    assert str(MAX_PASTE_URLS) in resp.json()["detail"]


def test_bulk_paste_accepts_exactly_the_cap():
    exactly = "\n".join(f"https://example{i}.com/" for i in range(MAX_PASTE_URLS))
    resp = client.post("/api/bulk-check-paste", json={"text": exactly})
    assert resp.status_code == 200
    assert resp.json()["summary"]["total"] == MAX_PASTE_URLS


def test_bulk_paste_requires_no_auth():
    """This replaced a dev-only, X-Dev-Key-gated endpoint - confirm the
    new one is genuinely public."""
    resp = client.post("/api/bulk-check-paste", json={"text": "https://www.google.com/"})
    assert resp.status_code == 200


# -------------------------------------------------------- bulk-check-upload --
def test_bulk_upload_txt_file():
    content = "\n".join(TEST_URLS).encode()
    resp = client.post(
        "/api/bulk-check-upload",
        files={"file": ("urls.txt", io.BytesIO(content), "text/plain")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["summary"]["total"] == len(TEST_URLS)


def test_bulk_upload_csv_file_any_column():
    """URLs must be found anywhere in the CSV, not assumed to be in a
    specific column."""
    csv_content = "name,notes,url\nGoogle,n/a,https://www.google.com/\nTyposquat,see below,https://www.sbl.co.in/\n"
    resp = client.post(
        "/api/bulk-check-upload",
        files={"file": ("urls.csv", io.BytesIO(csv_content.encode()), "text/csv")},
    )
    assert resp.status_code == 200
    data = resp.json()
    urls = {r["checked_url"] for r in data["results"]}
    assert "https://www.google.com/" in urls
    assert "https://www.sbl.co.in/" in urls


def test_bulk_upload_extracts_bare_two_label_domain_without_www():
    """2026-07 audit regression: the original _URL_EXTRACT_RE required its
    repeated middle group to consume a ".label" BEFORE the separately
    mandatory trailing ".TLD" - which made a bare two-label domain with no
    "www." prefix (e.g. "google.com") impossible to match at all, so it
    silently vanished from upload results instead of being checked."""
    content = b"Some notes.\nOur main site is google.com for now.\n"
    resp = client.post(
        "/api/bulk-check-upload",
        files={"file": ("notes.txt", io.BytesIO(content), "text/plain")},
    )
    assert resp.status_code == 200
    urls = {r["checked_url"] for r in resp.json()["results"]}
    assert "google.com" in urls


def test_bulk_upload_extraction_is_not_a_dos_on_adversarial_content():
    """2026-07 audit: the original _URL_EXTRACT_RE nested an optional
    group inside a repeated group - a catastrophic-backtracking shape.
    Measured directly before the fix: 50,000 adversarial characters (long
    hyphen runs with no valid domain ending) took 107 seconds. This pins
    that a large adversarial upload (up to the 2MB cap) resolves quickly,
    not hangs the worker thread - this endpoint is public and unauthenticated."""
    import time
    adversarial = (("a-" * 50_000) + "." + ("a-" * 50_000) + "!").encode()
    start = time.time()
    resp = client.post(
        "/api/bulk-check-upload",
        files={"file": ("adversarial.txt", io.BytesIO(adversarial), "text/plain")},
    )
    elapsed = time.time() - start
    assert elapsed < 5, f"URL extraction took {elapsed:.1f}s on adversarial input - regex backtracking regression"
    assert resp.status_code == 400  # no real URLs in this payload


def test_bulk_upload_extracts_urls_mixed_with_other_text():
    content = b"Some notes here.\nCheck out https://www.google.com/ before Friday.\nAlso see www.sbl.co.in for more.\n"
    resp = client.post(
        "/api/bulk-check-upload",
        files={"file": ("notes.txt", io.BytesIO(content), "text/plain")},
    )
    assert resp.status_code == 200
    urls = {r["checked_url"] for r in resp.json()["results"]}
    assert any("google.com" in u for u in urls)
    assert any("sbl.co.in" in u for u in urls)


def test_bulk_upload_deduplicates():
    content = ("https://www.google.com/\n" * 5).encode()
    resp = client.post(
        "/api/bulk-check-upload",
        files={"file": ("dupes.txt", io.BytesIO(content), "text/plain")},
    )
    assert resp.status_code == 200
    assert resp.json()["summary"]["total"] == 1


def test_bulk_upload_rejects_wrong_file_type():
    resp = client.post(
        "/api/bulk-check-upload",
        files={"file": ("urls.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")},
    )
    assert resp.status_code == 400


def test_bulk_upload_rejects_over_75_urls():
    content = "\n".join(f"https://example{i}.com/" for i in range(MAX_FILE_URLS + 1)).encode()
    resp = client.post(
        "/api/bulk-check-upload",
        files={"file": ("many.txt", io.BytesIO(content), "text/plain")},
    )
    assert resp.status_code == 400
    assert str(MAX_FILE_URLS) in resp.json()["detail"]


def test_bulk_upload_accepts_exactly_75_urls():
    content = "\n".join(f"https://example{i}.com/" for i in range(MAX_FILE_URLS)).encode()
    resp = client.post(
        "/api/bulk-check-upload",
        files={"file": ("exactly.txt", io.BytesIO(content), "text/plain")},
    )
    assert resp.status_code == 200
    assert resp.json()["summary"]["total"] == MAX_FILE_URLS


def test_bulk_upload_no_urls_found_rejected():
    content = b"Just some notes with no links at all, nothing to see here."
    resp = client.post(
        "/api/bulk-check-upload",
        files={"file": ("notes.txt", io.BytesIO(content), "text/plain")},
    )
    assert resp.status_code == 400
    assert "No URLs found" in resp.json()["detail"]


def test_bulk_upload_requires_no_auth():
    resp = client.post(
        "/api/bulk-check-upload",
        files={"file": ("urls.txt", io.BytesIO(b"https://www.google.com/"), "text/plain")},
    )
    assert resp.status_code == 200


def test_bulk_upload_never_written_to_disk():
    """CRITICAL memory rule: nothing about an upload may persist. Best a
    black-box test can do is confirm no new file appears anywhere under
    the project root during/after an upload."""
    root = Path(__file__).resolve().parents[1]
    before = {str(p) for p in root.rglob("*") if p.is_file()}
    content = "\n".join(TEST_URLS).encode()
    resp = client.post(
        "/api/bulk-check-upload",
        files={"file": ("urls.txt", io.BytesIO(content), "text/plain")},
    )
    assert resp.status_code == 200
    after = {str(p) for p in root.rglob("*") if p.is_file()}
    assert after - before == set(), f"New file(s) appeared on disk after upload: {after - before}"


# -------------------------------------------------------- bulk-check-export --
def test_export_csv_contains_results():
    paste_resp = client.post("/api/bulk-check-paste", json={"text": "\n".join(TEST_URLS)})
    results = paste_resp.json()["results"]
    resp = client.post("/api/bulk-check-export", json={"results": results, "format": "csv"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    body = resp.text
    header = body.splitlines()[0]
    assert "URL" in header
    assert "www.google.com" in body


def test_export_uses_percent_chance_and_reason_columns():
    """The export header must say 'Percent Chance', not 'Confidence' -
    non-technical users don't parse that term. A Reason column must be
    present and populated only for unsafe rows."""
    import csv as _csv
    paste_resp = client.post("/api/bulk-check-paste", json={"text": "\n".join(TEST_URLS)})
    results = paste_resp.json()["results"]
    resp = client.post("/api/bulk-check-export", json={"results": results, "format": "csv"})
    assert resp.status_code == 200

    rows = list(_csv.reader(io.StringIO(resp.text)))
    header, data_rows = rows[0], rows[1:]
    assert header == ["URL", "Status", "Percent Chance", "Reason"]

    by_url = {r[0]: r for r in data_rows}
    safe_row = by_url["https://www.google.com/"]
    assert safe_row[1] == "Safe"
    assert safe_row[3] == ""  # no reason for a safe verdict

    unsafe_row = by_url["https://www.sbl.co.in/"]
    assert unsafe_row[1] == "Unsafe"
    assert unsafe_row[3] != ""  # unsafe rows must have a reason


def test_export_xlsx_returns_spreadsheet():
    paste_resp = client.post("/api/bulk-check-paste", json={"text": "\n".join(TEST_URLS)})
    results = paste_resp.json()["results"]
    resp = client.post("/api/bulk-check-export", json={"results": results, "format": "xlsx"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert len(resp.content) > 0


def test_export_defaults_to_csv():
    paste_resp = client.post("/api/bulk-check-paste", json={"text": TEST_URLS[0]})
    results = paste_resp.json()["results"]
    resp = client.post("/api/bulk-check-export", json={"results": results})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")


def test_export_empty_results_rejected():
    resp = client.post("/api/bulk-check-export", json={"results": [], "format": "csv"})
    assert resp.status_code == 422  # pydantic min_length=1


# ----------------------------------------------------- /dev/bulk removed --
def test_dev_bulk_route_no_longer_exists():
    resp = client.get("/dev/bulk")
    assert resp.status_code == 404


def test_old_dev_bulk_endpoints_no_longer_exist():
    resp = client.post("/api/bulk-check", json={"urls": ["https://example.com/"]})
    assert resp.status_code == 404
    resp = client.post(
        "/api/bulk-check-file",
        files={"file": ("x.txt", io.BytesIO(b"https://example.com/"), "text/plain")},
    )
    assert resp.status_code == 404

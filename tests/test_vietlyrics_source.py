"""Focused contracts for VietLyrics source onboarding (acquisition + audit).

Covers pinned revision, CSV hash verification, header/row parsing, strict WPM
parsing, strict Zing-ID extraction, duplicate-ID detection, train/val overlap
detection and zero-LLM guarantees. No P7 QA or density thresholds are involved.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.common.config import ROOT
from src.common.zero_llm import scan_modules_for_forbidden_calls
from src.sauvi_perception.datasets.vietlyrics.vietlyrics_source import (
    EXPECTED_HEADERS,
    REVISION,
    VietLyricsSourceError,
    audit_zing_ids,
    cross_split_audit,
    extract_zing_id,
    load_provenance,
    load_rows,
    parse_wpm,
    verify_source,
)

AUDIT_DIR = ROOT / "outputs" / "source_audits" / "vietlyrics" / REVISION


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=EXPECTED_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in EXPECTED_HEADERS})


def _row(link: str, title: str = "t", artist: str = "a") -> dict[str, str]:
    row = {key: "" for key in EXPECTED_HEADERS}
    row.update({"link": link, "title": title, "artist": artist, "wpm": "80.0"})
    return row


# --- Revision / provenance --------------------------------------------------


def test_revision_is_pinned():
    prov = load_provenance()
    assert prov["revision"] == REVISION == "d52817ad9f63c4e1bdb7f77991e122a919d5dcf6"
    assert prov["integrity_authority"] == "sha256"


def test_provenance_retains_both_hashes():
    prov = load_provenance()
    for key in ("train", "validation"):
        assert len(prov[key]["sha256"]) == 64
        assert len(prov[key]["git_blob_sha1"]) == 40
        assert prov[key]["bytes"] > 0


def test_provenance_has_no_absolute_paths_or_timestamps():
    text = (ROOT / "resources" / "sources" / "vietlyrics.json").read_text(
        encoding="utf-8"
    )
    assert "G:\\" not in text and "G:/" not in text
    assert "audio_qa_mvp\\" not in text
    assert "2026" not in text


# --- Hash verification ------------------------------------------------------


def test_verify_source_rejects_tampered_bytes(tmp_path):
    tampered = tmp_path / "train_7k_metadata_authors.csv"
    tampered.write_bytes(b"not the real bytes")
    with pytest.raises(VietLyricsSourceError):
        verify_source(train_path=tampered)


# --- Parsing ----------------------------------------------------------------


def test_load_rows_roundtrip(tmp_path):
    path = tmp_path / "x.csv"
    _write_csv(path, [_row("https://zingmp3.vn/bai-hat/slug/ZW8IAWZ0.html")])
    rows = load_rows(path)
    assert len(rows) == 1
    assert rows[0]["link"].endswith("ZW8IAWZ0.html")


def test_load_rows_rejects_unexpected_header(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(VietLyricsSourceError):
        load_rows(path)


# --- WPM --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "status", "value"),
    [
        ("18", "valid", 18.0),
        ("18.5", "valid", 18.5),
        (" 20 ", "valid", 20.0),
        ("2.533235", "valid", 2.533235),
        (None, "missing", None),
        ("", "missing", None),
        ("   ", "missing", None),
        ("NaN", "malformed", None),
        ("nan", "malformed", None),
        ("inf", "malformed", None),
        ("-inf", "malformed", None),
        ("abc", "malformed", None),
        ("25 wpm", "malformed", None),
        ("0", "nonpositive", None),
        ("-3", "nonpositive", None),
    ],
)
def test_parse_wpm(raw, status, value):
    assert parse_wpm(raw) == (status, value)


# --- Zing ID ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("link", "expected"),
    [
        (
            "https://zingmp3.vn/bai-hat/1-ngay-ben-em-Trungg-I-U-Rum/Z7UEF8ID.html",
            "Z7UEF8ID",
        ),
        ("https://zingmp3.vn/bai-hat/slug/ZW8IAWZ0.html/", "ZW8IAWZ0"),
        ("https://example.com/foo", None),
        ("https://zingmp3.vn/bai-hat/slug/short.html", None),
        ("https://zingmp3.vn/ZW8IAWZ0", None),
        ("", None),
        (None, None),
    ],
)
def test_extract_zing_id(link, expected):
    assert extract_zing_id(link) == expected


# --- Duplicate IDs ----------------------------------------------------------


def test_audit_zing_ids_detects_duplicates():
    rows = [
        _row("https://zingmp3.vn/bai-hat/a/ZW8IAWZ0.html"),
        _row("https://zingmp3.vn/bai-hat/b/ZW8IAWZ0.html"),
        _row("https://zingmp3.vn/bai-hat/c/Z7UEF8ID.html"),
    ]
    audit = audit_zing_ids(rows)
    assert audit["parsed_ids"] == 3
    assert audit["unique_ids"] == 2
    assert audit["duplicate_ids"] == 1
    assert audit["duplicate_rows"] == 2


def test_audit_zing_ids_malformed_and_missing():
    rows = [
        _row("not a link"),
        _row(""),
        _row("https://zingmp3.vn/bai-hat/x/Z7UEF8ID.html"),
    ]
    audit = audit_zing_ids(rows)
    assert audit["malformed_link"] == 1
    assert audit["missing_link"] == 1
    assert audit["parsed_ids"] == 1


# --- Cross-split ------------------------------------------------------------


def test_cross_split_detects_overlap():
    train = [_row("https://zingmp3.vn/bai-hat/a/ZW8IAWZ0.html")]
    val = [_row("https://zingmp3.vn/bai-hat/b/ZW8IAWZ0.html")]
    audit = cross_split_audit(train, val)
    assert audit["zing_id_overlap_count"] == 1
    assert audit["blocking"] is True


def test_cross_split_clean():
    train = [_row("https://zingmp3.vn/bai-hat/a/ZW8IAWZ0.html")]
    val = [_row("https://zingmp3.vn/bai-hat/b/Z7UEF8ID.html")]
    audit = cross_split_audit(train, val)
    assert audit["zing_id_overlap_count"] == 0
    assert audit["blocking"] is False


# --- Zero LLM ---------------------------------------------------------------


def test_zero_llm_module_scan():
    result = scan_modules_for_forbidden_calls(modules=("src/sauvi_perception/datasets/vietlyrics/vietlyrics_source.py",))
    assert result["status"] == "PASS"
    assert result["llm_calls"] == 0


def test_source_has_no_llm_or_audio_download():
    text = (ROOT / "src" / "sauvi_perception" / "datasets" / "vietlyrics" / "vietlyrics_source.py").read_text(encoding="utf-8")
    assert "openai" not in text
    assert "llm_client" not in text
    assert "import yt_dlp" not in text
    assert "yt_dlp." not in text
    assert "soundfile" not in text


# --- Generated audit artifacts ---------------------------------------------


def test_source_audit_artifacts_present():
    audit_path = AUDIT_DIR / "source_audit.json"
    if not audit_path.is_file():
        pytest.skip("VietLyrics source audit not generated")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["revision"] == REVISION
    assert audit["files"]["train"]["logical_records"] == 7433
    assert audit["files"]["validation"]["logical_records"] == 995
    assert audit["wpm"]["train"]["valid_wpm"] == 7433
    assert audit["wpm"]["validation"]["valid_wpm"] == 995
    # Blocking finding surfaced by the audit (source-level split overlap).
    assert audit["cross_split"]["zing_id_overlap_count"] == 102
    assert audit["cross_split"]["blocking"] is True
    assert audit["zing_ids"]["train"]["duplicate_ids"] == 366

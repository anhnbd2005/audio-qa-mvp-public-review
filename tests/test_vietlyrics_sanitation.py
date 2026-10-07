"""Focused contracts for VietLyrics source sanitation + SAUVI contamination audit.

Covers asset grouping, VAL precedence, duplicate WPM canonicalization/conflict,
Decimal WPM equality, metadata-only disagreement, deterministic canonical rows,
SAUVI asset-ID extraction, all-task reservation, complete accounting, disjoint
final pool, zero-LLM and zero-audio guarantees.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from src.common.config import ROOT
from src.common.zero_llm import scan_modules_for_forbidden_calls
from src.sauvi_perception.datasets.vietlyrics.vietlyrics_sanitation import (
    REVISION,
    build_asset_inventory,
    build_clean_assets,
    build_exclusion_audit,
    build_sauvi_reserved,
    canonical_row,
    classify_all,
    extract_sauvi_asset_id,
    independent_audit,
    parse_wpm_decimal,
)

OUT_DIR = (
    ROOT / "outputs" / "source_audits" / "vietlyrics" / REVISION / "asset_sanitation"
)


def _row(link: str, wpm: str, **extra) -> dict[str, str]:
    row = {
        "link": link,
        "wpm": wpm,
        "title": extra.get("title", "t"),
        "artist": extra.get("artist", "a"),
        "song": extra.get("song", "s"),
        "genre": extra.get("genre", "g"),
        "duration_mins": extra.get("duration_mins", "3.0"),
        "token_count": extra.get("token_count", "300.0"),
    }
    return row


def _link(asset_id: str) -> str:
    return f"https://zingmp3.vn/bai-hat/slug/{asset_id}.html"


def _sauvi_row(asset_id: str, subtask: str = "music tempo detection") -> dict:
    return {
        "dataset": "VietLyrics",
        "audio_id": f"audio/music/audio_zing_{asset_id}.wav",
        "sub-category": subtask,
    }


# --- WPM (Decimal) ----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("80", Decimal(80)),
        ("80.0", Decimal("80.0")),
        ("80.000", Decimal("80.000")),
        (" 126.5 ", Decimal("126.5")),
        ("0", None),
        ("-3", None),
        ("NaN", None),
        ("inf", None),
        ("abc", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_wpm_decimal(raw, expected):
    assert parse_wpm_decimal(raw) == expected


def test_decimal_equality_numeric():
    assert parse_wpm_decimal("80") == parse_wpm_decimal("80.000")
    assert parse_wpm_decimal("80") != parse_wpm_decimal("80.1")


# --- SAUVI ID extraction ----------------------------------------------------


@pytest.mark.parametrize(
    ("audio_id", "expected"),
    [
        ("audio/music/audio_zing_ZW8IAWZ0.wav", "ZW8IAWZ0"),
        ("zing_ZW8IAWZ0.mp3", "ZW8IAWZ0"),
        ("/abs/path/audio_zing_Z6BCC6OE.wav", "Z6BCC6OE"),
        ("audio/music/audio_fb73d75b917b195d2177.wav", None),
        ("", None),
        (None, None),
    ],
)
def test_extract_sauvi_asset_id(audio_id, expected):
    assert extract_sauvi_asset_id(audio_id) == expected


# --- Asset grouping + precedence -------------------------------------------


def test_asset_grouping():
    train = [_row(_link("AAAA1111"), "80"), _row(_link("AAAA1111"), "80.0")]
    val = [_row(_link("BBBB2222"), "70")]
    inventory, un_train, un_val = build_asset_inventory(train, val)
    assert set(inventory) == {"AAAA1111", "BBBB2222"}
    assert len(inventory["AAAA1111"]["train"]) == 2
    assert len(inventory["BBBB2222"]["val"]) == 1
    assert un_train == [] and un_val == []


def test_val_precedence_reserves_asset():
    train = [_row(_link("AAAA1111"), "80")]
    val = [_row(_link("AAAA1111"), "80")]
    inventory, _u1, _u2 = build_asset_inventory(train, val)
    classifications = classify_all(inventory, set())
    assert classifications["AAAA1111"]["terminal_status"] == "VAL_RESERVED"
    source_only, final = build_clean_assets(classifications)
    assert source_only == [] and final == []


def test_train_only_singleton_is_clean():
    inventory, _u1, _u2 = build_asset_inventory([_row(_link("AAAA1111"), "80")], [])
    classifications = classify_all(inventory, set())
    source_only, final = build_clean_assets(classifications)
    assert classifications["AAAA1111"]["terminal_status"] == "TRAIN_CLEAN"
    assert len(source_only) == 1 and len(final) == 1


# --- Duplicate handling -----------------------------------------------------


def test_duplicate_same_wpm_canonicalized():
    train = [_row(_link("AAAA1111"), "80"), _row(_link("AAAA1111"), "80.0")]
    inventory, _u1, _u2 = build_asset_inventory(train, [])
    classifications = classify_all(inventory, set())
    cls = classifications["AAAA1111"]
    assert cls["wpm_conflict"] is False
    assert cls["terminal_status"] == "TRAIN_CLEAN"
    assert cls["canonical_row"]["source_row_index"] == 0
    source_only, _final = build_clean_assets(classifications)
    assert len(source_only) == 1


def test_duplicate_differing_wpm_conflict_excluded():
    train = [_row(_link("AAAA1111"), "80"), _row(_link("AAAA1111"), "81")]
    inventory, _u1, _u2 = build_asset_inventory(train, [])
    classifications = classify_all(inventory, set())
    cls = classifications["AAAA1111"]
    assert cls["wpm_conflict"] is True
    assert cls["terminal_status"] == "TRAIN_WPM_CONFLICT"
    source_only, final = build_clean_assets(classifications)
    assert source_only == [] and final == []
    excluded = build_exclusion_audit(inventory, classifications)
    assert {r["disposition"] for r in excluded} == {"EXCLUDED_WPM_CONFLICT"}
    assert len(excluded) == 2


def test_metadata_only_disagreement_is_report_only():
    train = [
        _row(_link("AAAA1111"), "80", title="one"),
        _row(_link("AAAA1111"), "80.0", title="two"),
    ]
    inventory, _u1, _u2 = build_asset_inventory(train, [])
    classifications = classify_all(inventory, set())
    cls = classifications["AAAA1111"]
    assert cls["metadata_inconsistency_report_only"] is True
    assert cls["terminal_status"] == "TRAIN_CLEAN"


def test_canonical_row_is_lowest_index():
    members = [
        {"source_row_index": 9, "wpm_raw": "80"},
        {"source_row_index": 3, "wpm_raw": "80"},
        {"source_row_index": 7, "wpm_raw": "80"},
    ]
    assert canonical_row(members)["source_row_index"] == 3


# --- SAUVI reservation ------------------------------------------------------


def test_sauvi_reserved_all_subtasks():
    rows = [
        _sauvi_row("AAAA1111", "music tempo detection"),
        _sauvi_row("BBBB2222", "vietnamese music genre reasoning"),
        _sauvi_row("CCCC3333", "vietnamese ethnic reasoning"),
        _sauvi_row("AAAA1111", "vietnamese ethnic reasoning"),
    ]
    reserved = build_sauvi_reserved(rows)
    assert reserved["total_rows"] == 4
    assert reserved["unique_asset_ids"] == 3
    assert set(reserved["reserved_asset_ids"]) == {"AAAA1111", "BBBB2222", "CCCC3333"}
    assert set(reserved["by_subtask"]) == {
        "music tempo detection",
        "vietnamese music genre reasoning",
        "vietnamese ethnic reasoning",
    }


def test_sauvi_reserved_asset_excluded_from_final():
    train = [_row(_link("AAAA1111"), "80")]
    inventory, _u1, _u2 = build_asset_inventory(train, [])
    classifications = classify_all(inventory, {"AAAA1111"})
    assert classifications["AAAA1111"]["terminal_status"] == "SAUVI_RESERVED"
    source_only, final = build_clean_assets(classifications)
    assert len(source_only) == 1  # source-clean before SAUVI filtering
    assert final == []  # excluded after SAUVI filtering


def test_val_precedence_beats_sauvi():
    train = [_row(_link("AAAA1111"), "80")]
    val = [_row(_link("AAAA1111"), "80")]
    inventory, _u1, _u2 = build_asset_inventory(train, val)
    classifications = classify_all(inventory, {"AAAA1111"})
    assert classifications["AAAA1111"]["terminal_status"] == "VAL_RESERVED"


# --- Independent audit / accounting ----------------------------------------


def test_independent_audit_accounting_and_disjointness():
    train = [
        _row(_link("AAAA1111"), "80"),  # clean
        _row(_link("BBBB2222"), "80"),  # sauvi reserved
        _row(_link("CCCC3333"), "80"),  # wpm conflict
        _row(_link("CCCC3333"), "81"),
    ]
    val = [_row(_link("DDDD4444"), "70")]
    inventory, _u1, _u2 = build_asset_inventory(train, val)
    classifications = classify_all(inventory, {"BBBB2222"})
    sauvi = {"reserved_asset_ids": ["BBBB2222"]}
    audit = independent_audit(train, val, classifications, inventory, sauvi)
    assert audit["all_train_rows_accounted"] is True
    assert audit["all_val_rows_accounted"] is True
    assert audit["no_val_asset_survives"] is True
    assert audit["no_sauvi_asset_survives"] is True
    assert audit["no_wpm_conflict_survives"] is True
    assert audit["final_clean_train_ids"] == 1


# --- Zero LLM / zero audio --------------------------------------------------


def test_zero_llm_module_scan():
    result = scan_modules_for_forbidden_calls(modules=("src/sauvi_perception/datasets/vietlyrics/vietlyrics_sanitation.py",))
    assert result["status"] == "PASS"
    assert result["llm_calls"] == 0


def test_no_llm_or_audio_in_source():
    text = (ROOT / "src" / "sauvi_perception" / "datasets" / "vietlyrics" / "vietlyrics_sanitation.py").read_text(encoding="utf-8")
    assert "openai" not in text
    assert "llm_client" not in text
    assert "soundfile" not in text
    assert "import yt_dlp" not in text


# --- Generated artifacts ----------------------------------------------------


def test_sanitation_artifacts_present():
    summary_path = OUT_DIR / "sanitation_summary.json"
    if not summary_path.is_file():
        pytest.skip("VietLyrics sanitation artifacts not generated")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["raw_rows"] == {"train": 7433, "val": 995}
    assert summary["asset_level"]["train_val_overlap_ids"] == 102
    assert summary["sauvi"]["unique_vietlyrics_ids"] == 237
    assert summary["final"]["final_clean_train_assets"] == 6452
    independent = json.loads(
        (OUT_DIR / "independent_audit.json").read_text(encoding="utf-8")
    )
    assert independent["all_train_rows_accounted"] is True
    assert independent["final_clean_intersect_val"] == []
    assert independent["final_clean_intersect_sauvi"] == []
    assert independent["no_wpm_conflict_survives"] is True

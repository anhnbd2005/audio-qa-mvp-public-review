"""Focused contracts for the final SAUVI P7 training-QA production release.

Covers frozen policy/thresholds, frozen clean-pool integrity, one-asset-one-QA,
exact boundary semantics, deterministic template/choice rendering, model-facing
leakage exclusion, split isolation, zero-LLM/no-audio architecture, source-order
invariance and byte-identical double runs.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

from src.common.config import ROOT
from src.sauvi_perception.tasks.p7_tempo.p7_production import (
    CLASS_ORDER,
    CLEAN_POOL,
    EXPECTED_CLASS_COUNTS,
    EXPECTED_POLICY_SHA256,
    EXPECTED_POOL_COUNT,
    EXPECTED_POOL_SHA256,
    EXPECTED_THRESHOLDS,
    POLICY_PATH,
    build_record,
    classify,
    compute_audit,
    generate_records,
    load_clean_assets,
    load_policy,
    permute_choices,
    select_template,
    write_release,
)
from src.common.zero_llm import scan_modules_for_forbidden_calls

OUT = ROOT / "outputs" / "training_qa" / "vietlyrics" / "p7" / "current"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _policy() -> dict:
    return {
        "policy_version": "p7_lyric_density_policy_v1",
        "policy_origin": "BENCHMARK_DERIVED_SEMANTIC_POLICY",
        "class_order": list(CLASS_ORDER),
        "thresholds": {key: str(value) for key, value in EXPECTED_THRESHOLDS.items()},
    }


def _asset(asset_id: str, wpm: str) -> dict:
    return {"asset_id": asset_id, "wpm_raw": wpm}


def _isolation() -> dict[str, set[str]]:
    return {"val_ids": set(), "sauvi_ids": set(), "conflict_ids": set()}


# --- Policy -----------------------------------------------------------------


def test_policy_hash_and_thresholds():
    assert _sha256(POLICY_PATH) == EXPECTED_POLICY_SHA256
    policy = load_policy()
    assert policy["policy_origin"] == "BENCHMARK_DERIVED_SEMANTIC_POLICY"
    assert policy["thresholds"] == {
        "t1": "56.1097976955994875",
        "t2": "66.91883456630708",
        "t3": "80.66592441889507",
    }
    assert EXPECTED_THRESHOLDS == {
        "t1": Decimal("56.1097976955994875"),
        "t2": Decimal("66.91883456630708"),
        "t3": Decimal("80.66592441889507"),
    }


def test_no_threshold_recomputation():
    # classify uses the frozen constants only
    assert classify(Decimal(50), EXPECTED_THRESHOLDS) == "Thưa"
    assert classify(Decimal(100), EXPECTED_THRESHOLDS) == "Rất dày"


# --- Source -----------------------------------------------------------------


def test_clean_pool_hash_and_count():
    assert _sha256(CLEAN_POOL) == EXPECTED_POOL_SHA256
    assets = load_clean_assets()
    assert len(assets) == EXPECTED_POOL_COUNT == 6452


def test_one_asset_one_qa():
    assets = [_asset("AAAA1111", "50"), _asset("BBBB2222", "100")]
    records = generate_records(assets, _policy(), EXPECTED_POLICY_SHA256)
    assert len(records) == 2
    assert len({r["source_asset_id"] for r in records}) == 2
    assert len({r["id"] for r in records}) == 2


# --- Gold boundary semantics ------------------------------------------------


@pytest.mark.parametrize(
    ("wpm", "expected"),
    [
        ("56.0", "Thưa"),
        ("56.1097976955994875", "Vừa"),  # == t1 -> Vừa
        ("56.2", "Vừa"),
        ("66.91883456630708", "Dày"),  # == t2 -> Dày
        ("67.0", "Dày"),
        ("80.66592441889507", "Rất dày"),  # == t3 -> Rất dày
        ("80.7", "Rất dày"),
    ],
)
def test_boundary_semantics(wpm, expected):
    assert classify(Decimal(wpm), EXPECTED_THRESHOLDS) == expected


def test_independent_derivation_matches_build_record():
    asset = _asset("AAAA1111", "70.0")
    record = build_record(asset, _policy(), EXPECTED_POLICY_SHA256)
    assert record["answer"] == classify(Decimal("70.0"), EXPECTED_THRESHOLDS)


# --- QA rendering -----------------------------------------------------------


def test_four_choices_and_answer_in_choices():
    records = generate_records(
        [_asset("AAAA1111", "50"), _asset("BBBB2222", "70"), _asset("CCCC3333", "100")],
        _policy(),
        EXPECTED_POLICY_SHA256,
    )
    for record in records:
        assert len(record["choices"]) == 4
        assert set(record["choices"]) == set(CLASS_ORDER)
        assert record["answer"] in record["choices"]


def test_deterministic_template_and_choices():
    assert select_template("AAAA1111") == select_template("AAAA1111")
    assert permute_choices("AAAA1111") == permute_choices("AAAA1111")
    assert (
        len(
            {
                entry["template_id"]
                for entry in [select_template(f"A{i}") for i in range(50)]
            }
        )
        > 1
    )
    assert set(permute_choices("AAAA1111")) == set(CLASS_ORDER)


def test_qa_id_not_from_row_number():
    a = build_record(_asset("AAAA1111", "50"), _policy(), EXPECTED_POLICY_SHA256)
    b = build_record(_asset("BBBB2222", "50"), _policy(), EXPECTED_POLICY_SHA256)
    assert a["id"] != b["id"]
    assert a["id"].startswith("p7_vietlyrics_")


# --- Leakage ----------------------------------------------------------------


def test_model_facing_has_no_metadata_leakage():
    record = build_record(_asset("AAAA1111", "70"), _policy(), EXPECTED_POLICY_SHA256)
    model = {
        "id": record["id"],
        "task": record["task"],
        "audio_id": record["audio_id"],
        "question": record["question"],
        "choices": record["choices"],
        "answer": record["answer"],
    }
    text = json.dumps(model, ensure_ascii=False).lower()
    for token in ("wpm", "title", "artist", "genre", "zingmp3", "threshold", "http"):
        assert token not in text
    assert set(model) == {"id", "task", "audio_id", "question", "choices", "answer"}


# --- Isolation / architecture ----------------------------------------------


def test_isolation_overlap_detected():
    assets = [_asset("AAAA1111", "50")]
    records = generate_records(assets, _policy(), EXPECTED_POLICY_SHA256)
    model = [
        {
            "id": r["id"],
            "task": r["task"],
            "audio_id": r["audio_id"],
            "question": r["question"],
            "choices": r["choices"],
            "answer": r["answer"],
        }
        for r in records
    ]
    isolation = {"val_ids": {"AAAA1111"}, "sauvi_ids": set(), "conflict_ids": set()}
    audit = compute_audit(records, model, isolation)
    assert audit["isolation"]["val_overlap"] == 1


def test_production_never_reads_sauvi_manifest():
    text = (ROOT / "src" / "sauvi_perception" / "tasks" / "p7_tempo" / "p7_production.py").read_text(encoding="utf-8")
    assert "sauvi_all" not in text
    assert "manifest_v2" not in text
    assert "music-ld" not in text
    assert "openai" not in text
    assert "llm_client" not in text
    assert "soundfile" not in text
    assert "import yt_dlp" not in text


def test_zero_llm_module_scan():
    result = scan_modules_for_forbidden_calls(modules=("src/sauvi_perception/tasks/p7_tempo/p7_production.py",))
    assert result["status"] == "PASS"
    assert result["llm_calls"] == 0


# --- Reproducibility --------------------------------------------------------


def test_source_order_invariance():
    assets = [
        _asset("AAAA1111", "50"),
        _asset("BBBB2222", "70"),
        _asset("CCCC3333", "100"),
    ]
    forward = generate_records(assets, _policy(), EXPECTED_POLICY_SHA256)
    reversed_ = generate_records(
        list(reversed(assets)), _policy(), EXPECTED_POLICY_SHA256
    )
    assert forward == reversed_
    assert [r["source_asset_id"] for r in forward] == sorted(
        a["asset_id"] for a in assets
    )


def test_byte_identical_double_run(tmp_path):
    assets = [_asset("AAAA1111", "50"), _asset("BBBB2222", "100")]
    records = generate_records(assets, _policy(), EXPECTED_POLICY_SHA256)
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    write_release(
        dir_a, records, assets, _policy(), EXPECTED_POLICY_SHA256, _isolation()
    )
    write_release(
        dir_b, records, assets, _policy(), EXPECTED_POLICY_SHA256, _isolation()
    )
    for name in ("qa_internal.jsonl", "qa_model_facing.jsonl", "audit.json"):
        assert (dir_a / name).read_bytes() == (dir_b / name).read_bytes()


# --- Generated release ------------------------------------------------------


def test_release_class_distribution():
    path = OUT / "class_distribution.json"
    if not path.is_file():
        pytest.skip("P7 release not generated")
    distribution = json.loads(path.read_text(encoding="utf-8"))
    assert distribution["counts"] == EXPECTED_CLASS_COUNTS
    assert distribution["total"] == 6452


def test_release_audit_and_independent():
    if not (OUT / "audit.json").is_file():
        pytest.skip("P7 release not generated")
    audit = json.loads((OUT / "audit.json").read_text(encoding="utf-8"))
    assert audit["isolation"]["val_overlap"] == 0
    assert audit["isolation"]["sauvi_overlap"] == 0
    assert audit["isolation"]["wpm_conflict_overlap"] == 0
    assert audit["isolation"]["sauvi_manifest_accessed"] is False
    independent = json.loads(
        (OUT / "independent_audit.json").read_text(encoding="utf-8")
    )
    assert independent["wrong_derived_class"] == 0
    assert independent["wrong_wpm"] == 0
    assert independent["answer_not_in_choices"] == 0

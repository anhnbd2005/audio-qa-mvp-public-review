"""Tests for canonical structured dataset row loader and relation evidence wiring.

Verifies:
A. Canonical metadata artifact exists -> load_dataset_rows returns actual rows.
B. Returned rows contain expected canonical schema keys without audio bytes.
C. Relation validator receives these rows.
D. Generic many-to-one relation can validate from loaded rows.
E. Missing production artifact does NOT fall back to sample.jsonl.
F. Mock/test can still explicitly pass sample rows.
G. Direct/equality behavior remains unchanged.
H. No real LLM calls.
"""

from pathlib import Path
import pytest

from src.autonomous_qa.compiler.pipeline_runner import (
    ROOT,
    _RunBase,
)
from src.autonomous_qa.datasets.context_adapter import (
    dataset_manifest_path,
    load_dataset_rows,
    load_sample,
)
from src.autonomous_qa.core.validity import (
    SCHEMA_FIELDS,
    gate_round,
    validate_derived_relation,
    validate_question_type,
)


def test_a_canonical_metadata_artifact_loads():
    """A: canonical metadata artifact exists -> load_dataset_rows returns actual rows."""
    rows = load_dataset_rows("vimd")
    assert isinstance(rows, list)
    assert len(rows) == 2026, f"Expected 2026 rows, got {len(rows)}"


def test_b_returned_rows_contain_expected_schema_keys():
    """B: returned rows contain expected canonical schema keys (no audio bytes)."""
    rows = load_dataset_rows("vimd")
    assert len(rows) > 0
    expected_keys = {"audio", "text", "gender", "region", "province", "speakerID"}

    for r in rows[:50]:
        assert expected_keys.issubset(r.keys()), f"Row missing keys: {r}"
        assert isinstance(r["audio"], str)
        assert isinstance(r["text"], str)
        assert r["gender"] in ("female", "male")
        assert isinstance(r["region"], str)
        assert isinstance(r["province"], str)
        assert isinstance(r["speakerID"], str)
        # Ensure no binary audio data is attached
        assert "bytes" not in r
        assert not isinstance(r["audio"], bytes)


def test_c_relation_validator_receives_rows():
    """C: relation validator receives loaded dataset rows during validation."""
    rows = load_dataset_rows("vimd")
    cache = {}
    qtype = {
        "id": "QS_D1",
        "name": "Province to Region",
        "uses": ["audio", "province", "region"],
        "input_count": 1,
        "answer_rule": "derive region",
        "answer": {"kind": "derived_field", "source_key": "province", "target_key": "region"},
    }
    errs = validate_question_type(qtype, dataset_rows=rows, relation_cache=cache)
    assert errs == []
    assert ("province", "region") in cache
    entry = cache[("province", "region")]
    assert entry["valid"] is True
    assert entry["distinct_source_count"] == 63
    assert entry["distinct_target_count"] == 3


def test_d_generic_many_to_one_relation_validates_from_loaded_rows():
    """D: generic many-to-one relation validates; vacuous relation rejects."""
    rows = load_dataset_rows("vimd")

    # Province -> Region is a valid many-to-one mapping across all 2026 rows
    res_valid = validate_derived_relation("province", "region", rows, SCHEMA_FIELDS)
    assert res_valid["valid"] is True
    assert res_valid["reason"] is None
    assert res_valid["distinct_source_count"] == 63
    assert res_valid["distinct_target_count"] == 3
    assert res_valid["mapping_conflicts"] == 0
    assert res_valid["reusable_source_values"] == 63

    # Text -> Gender is vacuous across rows (text is row-unique, reusable=0)
    res_vacuous = validate_derived_relation("text", "gender", rows, SCHEMA_FIELDS)
    assert res_vacuous["valid"] is False
    assert res_vacuous["reason"] == "invalid_derived_nonreusable_source"
    assert res_vacuous["reusable_source_values"] == 0


def test_e_missing_production_artifact_no_fallback(capsys, isolated_root, monkeypatch):
    """E: missing production artifact does NOT fall back to sample.jsonl."""
    monkeypatch.setattr("src.autonomous_qa.datasets.context_adapter.ROOT", isolated_root)
    # 1. Nonexistent dataset
    rows_unknown = load_dataset_rows("nonexistent_dataset")
    assert rows_unknown == []
    out = capsys.readouterr().out
    assert "structured_dataset_rows_unavailable:nonexistent_dataset" in out

    # 2. Manifest deleted in isolated root
    isolated_manifest = isolated_root / "data" / "vimd" / "manifest.jsonl"
    if isolated_manifest.exists():
        isolated_manifest.unlink()

    rows_missing = load_dataset_rows("vimd")
    assert rows_missing == []
    out2 = capsys.readouterr().out
    assert "structured_dataset_rows_unavailable:vimd" in out2

    # Verify sample.jsonl exists in isolated_root but was NOT returned
    sample_path = isolated_root / "data" / "vimd" / "sample.jsonl"
    assert sample_path.is_file()
    sample_rows = load_sample("vimd")
    assert len(sample_rows) == 26
    assert rows_missing != sample_rows


def test_f_mock_test_explicitly_pass_sample_rows():
    """F: mock/test can still explicitly pass sample rows."""
    sample_rows = load_sample("vimd")
    assert len(sample_rows) == 26

    # Test that _RunBase accepts explicitly supplied rows
    run = _RunBase(
        dataset="vimd",
        run_id="mock_run",
        out_dir=None,
        raw_dir=None,
        rounds_dir=None,
        readme="",
        rows=sample_rows,
        client=None,
    )
    assert run.rows == sample_rows

    # Test gate_round with sample_rows
    qtype = {
        "id": "QS_D1",
        "name": "Province to Region",
        "uses": ["audio", "province", "region"],
        "input_count": 1,
        "answer_rule": "derive region",
        "answer": {"kind": "derived_field", "source_key": "province", "target_key": "region"},
    }
    gate = gate_round(
        [qtype],
        [{"template_id": "T1", "question_type_id": "QS_D1", "text": "Hỏi [KEY]"}],
        type_key_realizations={"QS_D1": "vùng"},
        dataset_rows=sample_rows,
    )
    assert len(gate["valid_types"]) == 1


def test_g_direct_and_equality_behavior_unchanged():
    """G: direct/equality behavior remains unchanged regardless of rows."""
    direct = {
        "id": "QS_G",
        "name": "Gender",
        "uses": ["audio", "gender"],
        "input_count": 1,
        "answer_rule": "gender",
        "answer": {"kind": "field_value", "key": "gender"},
    }
    eq = {
        "id": "QS_E",
        "name": "Speaker Verification",
        "uses": ["audio", "speakerID"],
        "input_count": 2,
        "answer_rule": "same speaker",
        "answer": {"kind": "equality", "keys": ["speakerID_1", "speakerID_2"]},
    }

    # Valid with empty rows
    assert validate_question_type(direct, dataset_rows=[]) == []
    assert validate_question_type(eq, dataset_rows=[]) == []

    # Valid with full loaded rows
    full_rows = load_dataset_rows("vimd")
    assert validate_question_type(direct, dataset_rows=full_rows) == []
    assert validate_question_type(eq, dataset_rows=full_rows) == []


def test_h_no_real_llm_calls():
    """H: row loading and validation make exactly 0 real LLM calls."""
    from src.autonomous_qa.authoring.llm_client import BUDGET

    initial_calls = BUDGET.count
    rows = load_dataset_rows("vimd")
    validate_derived_relation("province", "region", rows, SCHEMA_FIELDS)
    validate_derived_relation("text", "gender", rows, SCHEMA_FIELDS)
    assert BUDGET.count == initial_calls == 0

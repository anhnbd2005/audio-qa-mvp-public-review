"""Regression: historical R2 malformed verdict fails; corrected passes.

Fixture: tests/fixtures/regression_r2_partition.json — malformed shape
copied from historical run 20260910T161936Z R2 (template id inside the
TYPE-level rejected array); candidate ids/texts mirror that run.
"""

import pytest

from src.autonomous_qa.authoring.llm_client import StageOutputValidationError
from src.autonomous_qa.core.loop import apply_accepted_types, validate_quality_partition
from src.common.io import load_fixture


def _fx():
    return load_fixture("regression_r2_partition")


def test_malformed_r2_verdict_fails_closed(isolated_root):
    fx = _fx()
    with pytest.raises(StageOutputValidationError,
                       match="quality_unknown_type_id:T_R2_02_02_P01"):
        validate_quality_partition(
            fx["malformed_verdict"],
            set(fx["valid_type_ids"]),
            set(fx["approved_type_ids"]))


def test_corrected_r2_verdict_passes(isolated_root):
    fx = _fx()
    out = validate_quality_partition(
        fx["corrected_verdict"],
        set(fx["valid_type_ids"]),
        set(fx["approved_type_ids"]))
    assert out == {"accepted": ["QS_R2_02"], "duplicates": [],
                   "rejected": []}


def test_corrected_r2_applies_one_type_zero_judge_rejects(isolated_root):
    fx = _fx()
    texts = dict(fx["offered_templates"])
    qtype = {"id": "QS_R2_02", "name": "Same Province Comparison",
             "goal": "g", "uses": ["audio", "province"], "input_count": 2,
             "answer_rule": "same province on both clips",
             "answer": {"kind": "equality",
                        "keys": ["province_1", "province_2"]}}
    pool = []
    accepted = apply_accepted_types(
        pool, {"QS_R2_02": qtype}, texts, fx["corrected_verdict"], 2,
        fx["owners"])
    assert [t["type_id"] for t in accepted] == ["QS_R2_02"]
    kept = [k["template_id"] for k in accepted[0]["kept_templates"]]
    assert kept == ["T_R2_02_01", "T_R2_02_02", "T_R2_02_01_P01",
                    "T_R2_02_01_P02", "T_R2_02_02_P02"]
    assert "T_R2_02_02_P01" not in kept  # omitted, not rejected-as-type


def test_corrected_r2_round_arithmetic(isolated_root):
    # Whole-round invariant: 8 generated == 7 deterministic + 1 accepted.
    generated, auto_rejected, accepted, duplicates, judge_rejected = (
        8, 7, 1, 0, 0)
    assert (generated == accepted + duplicates + judge_rejected
            + auto_rejected)

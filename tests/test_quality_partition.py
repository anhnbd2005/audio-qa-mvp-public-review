"""Quality partition validation: type verdicts partition V exactly (1-19)."""

import pytest

from src.autonomous_qa.authoring.llm_client import StageOutputValidationError
from src.autonomous_qa.core.loop import validate_quality_partition

V = {"QS_R2_01", "QS_R2_02"}
P = {"QS_R1_01", "QS_R1_04"}


def _verdict(acc=(), dup=(), rej=()):
    return {
        "accepted_new": [
            {"type_id": t, "keep_template_ids": ["T"]} for t in acc],
        "duplicates": [
            {"type_id": t, "duplicate_of": o} for t, o in dup],
        "rejected": [
            {"type_id": t, "reason": "r"} for t in rej],
    }


def test_accepted_type_id_not_in_v_fails():
    with pytest.raises(StageOutputValidationError,
                       match="quality_unknown_type_id:GHOST"):
        validate_quality_partition(
            _verdict(acc=["GHOST"], rej=["QS_R2_01", "QS_R2_02"]), V, P)


def test_duplicate_type_id_not_in_v_fails():
    with pytest.raises(StageOutputValidationError,
                       match="quality_unknown_type_id:GHOST"):
        validate_quality_partition(
            {"accepted_new": [], "duplicates": [
                {"type_id": "GHOST", "duplicate_of": "QS_R1_01"}],
             "rejected": [{"type_id": "QS_R2_01", "reason": "r"},
                          {"type_id": "QS_R2_02", "reason": "r"}]},
            V, P)


def test_rejected_type_id_not_in_v_fails():
    # The real R2 bug: a TEMPLATE id inside the TYPE-level rejected array.
    with pytest.raises(StageOutputValidationError,
                       match="quality_unknown_type_id:T_R2_02_02_P01"):
        validate_quality_partition(
            {"accepted_new": [{"type_id": "QS_R2_02",
                               "keep_template_ids": ["T1"]}],
             "duplicates": [],
             "rejected": [{"type_id": "T_R2_02_02_P01", "reason": "weak"}]},
            {"QS_R2_02"}, P)


def test_duplicate_of_outside_approved_set_fails():
    with pytest.raises(StageOutputValidationError,
                       match="quality_unknown_duplicate_target:QS_R9_99"):
        validate_quality_partition(
            {"accepted_new": [{"type_id": "QS_R2_01",
                               "keep_template_ids": ["T1"]}],
             "duplicates": [{"type_id": "QS_R2_02",
                             "duplicate_of": "QS_R9_99"}],
             "rejected": []},
            V, P)


def test_template_id_as_rejected_type_id_fails():
    with pytest.raises(StageOutputValidationError,
                       match="quality_unknown_type_id"):
        validate_quality_partition(
            _verdict(acc=["QS_R2_01"], rej=["T_X"]), V, P)


def test_template_id_as_duplicate_type_id_fails():
    with pytest.raises(StageOutputValidationError,
                       match="quality_unknown_type_id"):
        validate_quality_partition(
            {"accepted_new": [], "duplicates": [
                {"type_id": "T_X", "duplicate_of": "QS_R1_01"}],
             "rejected": [{"type_id": "QS_R2_01", "reason": "r"},
                          {"type_id": "QS_R2_02", "reason": "r"}]},
            V, P)


def test_template_id_as_duplicate_of_fails():
    with pytest.raises(StageOutputValidationError,
                       match="quality_unknown_duplicate_target"):
        validate_quality_partition(
            {"accepted_new": [{"type_id": "QS_R2_01",
                               "keep_template_ids": ["T1"]}],
             "duplicates": [{"type_id": "QS_R2_02",
                             "duplicate_of": "T_X"}],
             "rejected": []},
            V, P)


def test_same_type_in_accepted_and_rejected_fails():
    with pytest.raises(StageOutputValidationError,
                       match="quality_overlapping_verdict:QS_R2_01"):
        validate_quality_partition(
            _verdict(acc=["QS_R2_01", "QS_R2_02"],
                     rej=["QS_R2_01"]), V, P)


def test_same_type_in_accepted_and_duplicates_fails():
    with pytest.raises(StageOutputValidationError,
                       match="quality_overlapping_verdict:QS_R2_02"):
        validate_quality_partition(
            {"accepted_new": [{"type_id": "QS_R2_02",
                               "keep_template_ids": ["T1"]}],
             "duplicates": [{"type_id": "QS_R2_02",
                             "duplicate_of": "QS_R1_01"}],
             "rejected": [{"type_id": "QS_R2_01", "reason": "r"}]},
            V, P)


def test_same_type_in_duplicate_and_rejected_fails():
    with pytest.raises(StageOutputValidationError,
                       match="quality_overlapping_verdict:QS_R2_01"):
        validate_quality_partition(
            {"accepted_new": [{"type_id": "QS_R2_02",
                               "keep_template_ids": ["T1"]}],
             "duplicates": [{"type_id": "QS_R2_01",
                             "duplicate_of": "QS_R1_01"}],
             "rejected": [{"type_id": "QS_R2_01", "reason": "r"}]},
            V, P)


def test_valid_type_omitted_from_all_arrays_fails():
    with pytest.raises(StageOutputValidationError,
                       match="quality_missing_verdict:QS_R2_02"):
        validate_quality_partition(
            _verdict(acc=["QS_R2_01"]), V, P)


def test_exact_partition_passes():
    out = validate_quality_partition(
        _verdict(acc=["QS_R2_01"], dup=[("QS_R2_02", "QS_R1_04")]), V, P)
    assert out == {"accepted": ["QS_R2_01"], "duplicates": ["QS_R2_02"],
                   "rejected": []}


def test_all_rejected_partition_passes():
    out = validate_quality_partition(
        _verdict(rej=["QS_R2_01", "QS_R2_02"]), V, P)
    assert out == {"accepted": [], "duplicates": [],
                   "rejected": ["QS_R2_01", "QS_R2_02"]}


def test_accepted_type_requires_nonempty_keep_list():
    from src.autonomous_qa.core.loop import apply_accepted_types
    qtype = {"id": "QS_R2_02", "name": "N", "goal": "g",
             "uses": ["audio", "province"], "input_count": 2,
             "answer_rule": "r",
             "answer": {"kind": "equality",
                        "keys": ["province_1", "province_2"]}}
    with pytest.raises(StageOutputValidationError,
                       match="quality_no_kept_template"):
        apply_accepted_types(
            [], {"QS_R2_02": qtype}, {"T1": "t"},
            {"accepted_new": [{"type_id": "QS_R2_02",
                               "keep_template_ids": []}]},
            2, {"T1": "QS_R2_02"})


def test_foreign_keep_template_fails():
    from src.autonomous_qa.core.loop import apply_accepted_types
    qtype = {"id": "QS_R2_02", "name": "N", "goal": "g",
             "uses": ["audio", "province"], "input_count": 2,
             "answer_rule": "r",
             "answer": {"kind": "equality",
                        "keys": ["province_1", "province_2"]}}
    with pytest.raises(StageOutputValidationError,
                       match="quality_foreign_template_id"):
        apply_accepted_types(
            [], {"QS_R2_02": qtype}, {"T1": "t", "T9": "t9"},
            {"accepted_new": [{"type_id": "QS_R2_02",
                               "keep_template_ids": ["T9"]}]},
            2, {"T1": "QS_R2_02", "T9": "QS_R9_99"})


def test_unknown_keep_template_fails():
    from src.autonomous_qa.core.loop import apply_accepted_types
    qtype = {"id": "QS_R2_02", "name": "N", "goal": "g",
             "uses": ["audio", "province"], "input_count": 2,
             "answer_rule": "r",
             "answer": {"kind": "equality",
                        "keys": ["province_1", "province_2"]}}
    with pytest.raises(StageOutputValidationError,
                       match="quality_unknown_template_id"):
        apply_accepted_types(
            [], {"QS_R2_02": qtype}, {"T1": "t"},
            {"accepted_new": [{"type_id": "QS_R2_02",
                               "keep_template_ids": ["T1", "GHOST"]}]},
            2, {"T1": "QS_R2_02"})


def test_duplicate_keep_template_fails():
    from src.autonomous_qa.core.loop import apply_accepted_types
    qtype = {"id": "QS_R2_02", "name": "N", "goal": "g",
             "uses": ["audio", "province"], "input_count": 2,
             "answer_rule": "r",
             "answer": {"kind": "equality",
                        "keys": ["province_1", "province_2"]}}
    with pytest.raises(StageOutputValidationError,
                       match="quality_duplicate_keep_id"):
        apply_accepted_types(
            [], {"QS_R2_02": qtype}, {"T1": "t"},
            {"accepted_new": [{"type_id": "QS_R2_02",
                               "keep_template_ids": ["T1", "T1"]}]},
            2, {"T1": "QS_R2_02"})


def test_unselected_template_needs_no_rejected_entry():
    # QS_R2_02 keeps 5 of 6 offered; the omitted id needs no verdict row.
    from src.autonomous_qa.core.loop import apply_accepted_types
    qtype = {"id": "QS_R2_02", "name": "N", "goal": "g",
             "uses": ["audio", "province"], "input_count": 2,
             "answer_rule": "r",
             "answer": {"kind": "equality",
                        "keys": ["province_1", "province_2"]}}
    texts = {f"T{i}": f"t{i}" for i in range(6)}
    owners = {f"T{i}": "QS_R2_02" for i in range(6)}
    verdict = {"accepted_new": [{"type_id": "QS_R2_02",
                                 "keep_template_ids": ["T0", "T1", "T2",
                                                       "T3", "T4"]}],
               "duplicates": [], "rejected": []}
    pool = []
    accepted = apply_accepted_types(
        pool, {"QS_R2_02": qtype}, texts, verdict, 2, owners)
    assert [k["template_id"] for k in accepted[0]["kept_templates"]] == [
        "T0", "T1", "T2", "T3", "T4"]
    assert verdict["rejected"] == []


def test_unselected_template_does_not_increment_rejections():
    # Template omission must not leak into type-level rejection counts.
    from src.autonomous_qa.core.loop import build_round_stats
    stats = build_round_stats(
        round_idx=2, generated_types=1, accepted_new_types=1, duplicates=0,
        rejected=0, new_type_rate=1.0, total_approved_types=8)
    assert stats["rejected"] == 0
    assert stats["accepted_new_types"] == 1

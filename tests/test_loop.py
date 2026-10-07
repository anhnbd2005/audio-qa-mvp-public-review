"""Discovery controller + approved-type pool mechanics (pure Python)."""


from src.autonomous_qa.core.loop import (
    LoopController,
    apply_accepted_types,
    build_round_stats,
    build_type_fewshot,
    flatten_type_bindings,
    normalize_template,
    split_exact_duplicates,
)


def _controller(**over):
    cfg = {
        "max_rounds": 3,
        "min_rate": 0.15,
        "saturation_patience": 2,
        "immediate_stop_if_zero_new": True,
        "zero_reason": "zero_new_types",
        "saturated_reason": "type_diversity_saturated",
    }
    cfg.update(over)
    return LoopController(**cfg)


def test_zero_new_types_stops_immediately():
    out = _controller().register_round(1, accepted_new=0, generated=8)
    assert out["stop"] is True
    assert out["stop_reason"] == "zero_new_types"


def test_one_low_round_does_not_stop():
    out = _controller().register_round(1, accepted_new=1, generated=8)
    assert out["stop"] is False
    assert out["stop_reason"] is None
    assert out["new_type_rate"] == 0.125


def test_two_low_rounds_stop_saturated():
    ctrl = _controller()
    ctrl.register_round(1, accepted_new=1, generated=8)  # 0.125
    out = ctrl.register_round(2, accepted_new=1, generated=8)  # 0.125
    assert out["stop"] is True
    assert out["stop_reason"] == "type_diversity_saturated"


def test_gain_recovery_resets_streak():
    ctrl = _controller()
    ctrl.register_round(1, accepted_new=1, generated=8)  # 0.125
    out = ctrl.register_round(2, accepted_new=4, generated=8)  # 0.50
    assert out["stop"] is False
    assert out["low_gain_streak"] == 0


def test_hard_max_rounds_stops_despite_gain():
    ctrl = _controller()
    ctrl.register_round(1, accepted_new=8, generated=8)
    ctrl.register_round(2, accepted_new=8, generated=8)
    out = ctrl.register_round(3, accepted_new=8, generated=8)
    assert out["stop"] is True
    assert out["stop_reason"] == "max_rounds"


def test_zero_generated_counts_as_zero_new():
    out = _controller().register_round(1, accepted_new=0, generated=0)
    assert out["stop"] is True
    assert out["stop_reason"] == "zero_new_types"


def test_normalize_template():
    assert normalize_template("  Hay   XAC dinh [KEY] ") == "hay xac dinh [key]"


def test_split_exact_duplicates_collapses_batch():
    unique, exact = split_exact_duplicates(
        [{"template_id": "A", "text": "Hay xac dinh [KEY]?"},
         {"template_id": "B", "text": "Hay  XAC DINH [KEY]?"}],
        [],
    )
    assert exact == 1
    assert [c["template_id"] for c in unique] == ["A"]


def test_dynamic_fewshot_is_minimal():
    pool = [{
        "type_id": "QS_R1_01", "name": "Gender", "goal": "g",
        "uses": ["audio", "gender"], "input_count": 1,
        "answer_rule": "the gender label", "answer": {"kind": "field_value"},
        "round_accepted": 1, "kept_templates": [{"template_id": "T",
                                                 "text": "t"}],
        "bindings": [],
    }]
    assert build_type_fewshot(pool) == [{
        "name": "Gender", "uses": ["audio", "gender"], "goal": "g",
        "answer_rule": "the gender label"}]


def test_apply_accepted_types_builds_bindings_with_owners():
    pool = []
    qtype = {"id": "QS_R1_01", "name": "Gender", "goal": "g",
             "uses": ["audio", "gender"], "input_count": 1,
             "answer_rule": "the gender label",
             "answer": {"kind": "field_value", "key": "gender"}}
    texts = {"T1": "Hãy xác định [KEY]?", "T1_P1": "[KEY] là gì?"}
    owners = {"T1": "QS_R1_01", "T1_P1": "QS_R1_01"}
    verdict = {"accepted_new": [
        {"type_id": "QS_R1_01", "keep_template_ids": ["T1", "T1_P1"]},
    ]}
    accepted = apply_accepted_types(
        pool, {"QS_R1_01": qtype}, texts, verdict, 1, owners)
    assert len(accepted) == 1
    item = accepted[0]
    assert item["round_accepted"] == 1
    assert [k["template_id"] for k in item["kept_templates"]] == ["T1", "T1_P1"]
    assert item["bindings"] == [
        {"question_type_id": "QS_R1_01", "key": "gender", "template_id": "T1"},
        {"question_type_id": "QS_R1_01", "key": "gender",
         "template_id": "T1_P1"},
    ]
    flat = flatten_type_bindings(pool)
    assert len(flat) == 2


def test_apply_accepted_types_rejects_unknown_type():
    import pytest

    from src.autonomous_qa.authoring.llm_client import StageOutputValidationError
    with pytest.raises(StageOutputValidationError, match="unknown type"):
        apply_accepted_types(
            [], {"QS_R1_01": {"id": "QS_R1_01"}}, {"T1": "t"},
            {"accepted_new": [{"type_id": "QS_R9_99",
                               "keep_template_ids": ["T1"]}]},
            1, {"T1": "QS_R1_01"})


def test_apply_accepted_types_rejects_ghost_keep_id():
    import pytest

    from src.autonomous_qa.authoring.llm_client import StageOutputValidationError
    with pytest.raises(StageOutputValidationError, match="unknown"):
        apply_accepted_types(
            [], {"QS_R1_01": {"id": "QS_R1_01"}}, {"T1": "t"},
            {"accepted_new": [{"type_id": "QS_R1_01",
                               "keep_template_ids": ["T1", "GHOST"]}]},
            1, {"T1": "QS_R1_01"})


def test_apply_accepted_types_rejects_foreign_keep_id():
    import pytest

    from src.autonomous_qa.authoring.llm_client import StageOutputValidationError
    with pytest.raises(StageOutputValidationError, match="does not belong"):
        apply_accepted_types(
            [], {"QS_R1_01": {"id": "QS_R1_01"},
                 "QS_R1_02": {"id": "QS_R1_02"}},
            {"T1": "t", "T2": "t2"},
            {"accepted_new": [{"type_id": "QS_R1_01",
                               "keep_template_ids": ["T2"]}]},
            1, {"T1": "QS_R1_01", "T2": "QS_R1_02"})


def test_apply_accepted_ignores_repeated_type():
    pool = [{"type_id": "QS_R1_01", "kept_templates": [], "bindings": []}]
    accepted = apply_accepted_types(
        pool, {"QS_R1_01": {"id": "QS_R1_01"}}, {"T1": "t"},
        {"accepted_new": [{"type_id": "QS_R1_01",
                           "keep_template_ids": ["T1"]}]}, 2,
        {"T1": "QS_R1_01"})
    assert accepted == []
    assert len(pool) == 1


def test_build_round_stats_type_fields():
    stats = build_round_stats(
        round_idx=2, generated_types=8, accepted_new_types=2, duplicates=5,
        rejected=1, new_type_rate=0.25, total_approved_types=7)
    assert stats == {
        "round": 2, "generated_types": 8, "accepted_new_types": 2,
        "duplicates": 5, "deterministic_duplicates": 0, "rejected": 1,
        "new_type_rate": 0.25, "total_approved_types": 7,
    }

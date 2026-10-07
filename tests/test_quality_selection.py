"""Fix 2 tests: Quality keep_template_ids actually selects (15-24)."""

import pytest

from src.autonomous_qa.authoring.llm_client import StageOutputValidationError
from src.autonomous_qa.core.loop import apply_accepted_types


def _qtype(tid="QS_R1_01"):
    return {"id": tid, "name": "Gender", "goal": "g",
            "uses": ["audio", "gender"], "input_count": 1,
            "answer_rule": "the gender label",
            "answer": {"kind": "field_value", "key": "gender"}}


def _owners(*ids, owner="QS_R1_01"):
    return {i: owner for i in ids}


def test_subset_keep_results_in_pool_with_only_those():
    pool = []
    texts = {"A": "tA", "B": "tB", "C": "tC"}
    verdict = {"accepted_new": [
        {"type_id": "QS_R1_01", "keep_template_ids": ["A", "C"]}]}
    accepted = apply_accepted_types(
        pool, {"QS_R1_01": _qtype()}, texts, verdict, 1,
        _owners("A", "B", "C"))
    assert [k["template_id"] for k in accepted[0]["kept_templates"]] == [
        "A", "C"]
    assert [k["text"] for k in accepted[0]["kept_templates"]] == ["tA", "tC"]
    flat = [b["template_id"] for b in accepted[0]["bindings"]]
    assert flat == ["A", "C"]


def test_candidate_not_retained_merely_because_type_accepted():
    pool = []
    texts = {"A": "tA", "B": "tB"}
    verdict = {"accepted_new": [
        {"type_id": "QS_R1_01", "keep_template_ids": ["A"]}]}
    apply_accepted_types(
        pool, {"QS_R1_01": _qtype()}, texts, verdict, 1,
        _owners("A", "B"))
    kept = [k["template_id"] for k in pool[0]["kept_templates"]]
    assert "B" not in kept
    assert kept == ["A"]


def test_empty_keep_fails_validation():
    with pytest.raises(StageOutputValidationError, match="empty"):
        apply_accepted_types(
            [], {"QS_R1_01": _qtype()}, {"A": "tA"},
            {"accepted_new": [{"type_id": "QS_R1_01",
                               "keep_template_ids": []}]},
            1, {"A": "QS_R1_01"})


def test_nonexistent_keep_id_fails_validation():
    with pytest.raises(StageOutputValidationError, match="unknown"):
        apply_accepted_types(
            [], {"QS_R1_01": _qtype()}, {"A": "tA"},
            {"accepted_new": [{"type_id": "QS_R1_01",
                               "keep_template_ids": ["GHOST"]}]},
            1, {"A": "QS_R1_01"})


def test_foreign_type_keep_id_fails_validation():
    with pytest.raises(StageOutputValidationError,
                       match="does not belong"):
        apply_accepted_types(
            [], {"QS_R1_01": _qtype(),
                 "QS_R1_02": _qtype("QS_R1_02")},
            {"A": "tA", "B": "tB"},
            {"accepted_new": [{"type_id": "QS_R1_01",
                               "keep_template_ids": ["B"]}]},
            1, {"A": "QS_R1_01", "B": "QS_R1_02"})


def test_dropped_template_id_cannot_be_resurrected():
    # "B" was deterministically dropped: absent from texts/owners.
    with pytest.raises(StageOutputValidationError, match="unknown"):
        apply_accepted_types(
            [], {"QS_R1_01": _qtype()}, {"A": "tA"},
            {"accepted_new": [{"type_id": "QS_R1_01",
                               "keep_template_ids": ["A", "B"]}]},
            1, {"A": "QS_R1_01"})


def test_dedup_removed_paraphrase_cannot_be_resurrected():
    with pytest.raises(StageOutputValidationError, match="unknown"):
        apply_accepted_types(
            [], {"QS_R1_01": _qtype()}, {"A": "tA"},
            {"accepted_new": [{"type_id": "QS_R1_01",
                               "keep_template_ids": ["A_P1"]}]},
            1, {"A": "QS_R1_01"})


def test_duplicate_type_stays_type_level():
    # A duplicate verdict references the TYPE id; paraphrase equivalence
    # never creates duplicate entries and never touches the pool.
    pool = [{"type_id": "QS_R1_04", "kept_templates": [{"template_id": "T",
                                                       "text": "t"}],
             "bindings": []}]
    texts = {"A": "tA", "A_P1": "tA variant"}
    verdict = {"accepted_new": [],
               "duplicates": [{"type_id": "QS_R2_03",
                               "duplicate_of": "QS_R1_04"}],
               "rejected": []}
    accepted = apply_accepted_types(
        pool, {}, texts, verdict, 2, {"A": "QS_R2_03", "A_P1": "QS_R2_03"})
    assert accepted == []
    assert len(pool) == 1  # duplicates never mutate the approved pool


def test_partially_bad_type_still_accepted_with_good_keep():
    pool = []
    texts = {"GOOD": "t-good"}
    verdict = {"accepted_new": [
        {"type_id": "QS_R1_01", "keep_template_ids": ["GOOD"]}]}
    accepted = apply_accepted_types(
        pool, {"QS_R1_01": _qtype()}, texts, verdict, 1,
        {"GOOD": "QS_R1_01"})
    assert [t["type_id"] for t in accepted] == ["QS_R1_01"]


def test_final_pool_built_exactly_from_selected_ids(isolated_root):
    from src.autonomous_qa.core.loop import flatten_type_bindings
    from src.common.io import load_fixture
    pool = []
    for r in (1, 2):
        style = load_fixture(f"loop/round_{r:02d}/style")
        tpl = load_fixture(f"loop/round_{r:02d}/template")
        para = load_fixture(f"loop/round_{r:02d}/paraphrase")
        verdict = load_fixture(f"loop/round_{r:02d}/quality")
        from src.autonomous_qa.core.loop import filter_paraphrases_for_judging
        from src.autonomous_qa.compiler.pipeline_runner import load_sample
        from src.autonomous_qa.core.validity import gate_round
        rows = load_sample("vimd")
        gate = gate_round(style["question_types"], tpl["templates"], dataset_rows=rows)
        filt = filter_paraphrases_for_judging(
            gate["valid_templates"], para["paraphrases"])
        texts = {t["template_id"]: t["text"]
                 for t in gate["valid_templates"]}
        texts.update(filt["texts"])
        owners = {t["template_id"]: t["question_type_id"]
                  for t in gate["valid_templates"]}
        owners.update(filt["owners"])
        by_id = {t["id"]: t for t in style["question_types"]}
        apply_accepted_types(pool, by_id, texts, verdict, r, owners)
    # Every pooled template id was explicitly selected by some verdict.
    selected = set()
    for r in (1, 2):
        verdict = load_fixture(f"loop/round_{r:02d}/quality")
        for a in verdict["accepted_new"]:
            selected.update(a["keep_template_ids"])
    pooled = {k["template_id"] for t in pool for k in t["kept_templates"]}
    assert pooled <= selected
    assert pooled  # non-empty: selection actually retained content
    assert len(flatten_type_bindings(pool)) == len(pooled)

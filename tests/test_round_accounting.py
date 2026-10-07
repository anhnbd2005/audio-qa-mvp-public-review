"""Round accounting invariant + base dedup scope + failure resume (§41-43)."""

import json

import pytest

import src.autonomous_qa.authoring.llm_client as llm_client
from src.autonomous_qa.compiler.pipeline_runner import _assert_round_accounting, run_pipeline
from src.autonomous_qa.core.validity import gate_round
from tests.test_safety import ScriptedLLM, _reset_budget, clean_budget  # noqa: F401


def test_accounting_pass_case_a():
    # generated=8, auto=7, accepted=1, dup=0, judge=0 -> PASS.
    _assert_round_accounting(2, 8, 1, 0, 0, 7, "quality")


def test_accounting_fail_case_b():
    # generated=8, auto=7, accepted=1, dup=0, judge=1 -> 9 != 8 -> FAIL.
    with pytest.raises(ValueError, match="round accounting violated"):
        _assert_round_accounting(2, 8, 1, 0, 1, 7, "quality")


def test_accounting_pass_case_c():
    # generated=8, auto=6, accepted=1, dup=1, judge=0 -> PASS.
    _assert_round_accounting(2, 8, 1, 1, 0, 6, "quality")


def test_template_omission_does_not_change_type_arithmetic():
    # offered=6, kept=5: template-level omission is invisible to the
    # generated/accepted/duplicates/rejected type counts.
    _assert_round_accounting(2, 8, 1, 0, 0, 7, "quality")


def _t(tid, owner, text):
    return {"template_id": tid, "question_type_id": owner, "text": text}


def _type(tid, name="N"):
    return {"id": tid, "name": name, "goal": "g", "uses": ["audio", "gender"],
            "input_count": 1, "answer_rule": "the gender label",
            "answer": {"kind": "field_value", "key": "gender"}}


def test_base_dedup_scoped_per_type_not_global():
    # Identical generic text under two types: both survive, because each
    # binds a different concept.
    gate = gate_round(
        [_type("QA", "A"), _type("QB", "B")],
        [_t("TA", "QA", "Người nói thuộc [KEY] nào?"),
         _t("TB", "QB", "Người nói thuộc [KEY] nào?")])
    assert [t["template_id"] for t in gate["valid_templates"]] == [
        "TA", "TB"]
    assert [d for d in gate["dropped_templates"]
            if d["reason"].startswith("exact_duplicate:")] == []


def test_base_dedup_collapses_same_type_normalized_match():
    gate = gate_round(
        [_type("QA", "A")],
        [_t("TA1", "QA", "Người nói thuộc [KEY] nào?"),
         _t("TA2", "QA", "  người nói   thuộc [KEY] nào?  ")])
    assert [t["template_id"] for t in gate["valid_templates"]] == ["TA1"]
    assert gate["dropped_templates"] == [
        {"template_id": "TA2", "type_id": "QA",
         "reason": "exact_duplicate:TA1"}]


def _bad_style_verdict_pair(isolated_root):
    from tests.test_safety import _payload
    style = _payload(isolated_root, "loop", "round_01", "style")
    tpl = _payload(isolated_root, "loop", "round_01", "template")
    para = _payload(isolated_root, "loop", "round_01", "paraphrase")
    bad_verdict = {
        "round": 1,
        "accepted_new": [{"type_id": "QS_R1_01",
                          "keep_template_ids": ["T_R1_01_01"]}],
        "duplicates": [],
        # Malformed: a TEMPLATE id inside the TYPE-level rejected array.
        "rejected": [{"type_id": "T_R1_01_01_P01",
                      "reason": "weak wording"}],
    }
    return style, tpl, para, bad_verdict


def test_malformed_partition_fails_before_pool_mutation(
        isolated_root, clean_budget, monkeypatch):
    style, tpl, para, bad_verdict = _bad_style_verdict_pair(isolated_root)
    script = [style, tpl, para, bad_verdict]

    def fake(client, stage, prompt, response_schema):
        from types import SimpleNamespace
        item = script.pop(0)
        llm_client.BUDGET.consume(stage)
        return SimpleNamespace(text=json.dumps(item), parsed=None,
                               candidates=[], usage_metadata=None)

    monkeypatch.setattr(llm_client, "call_llm", fake)
    monkeypatch.setattr(llm_client, "create_llm_client", lambda: object())

    from src.autonomous_qa.authoring.llm_client import StageOutputValidationError
    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "malformed-q"
    with pytest.raises(StageOutputValidationError,
                       match="quality_unknown_type_id"):
        run_pipeline(dataset="vimd", real_llm=True, run_id="malformed-q")

    # Raw attempt saved, paid attempt counted, nothing applied downstream.
    raws = sorted((out_dir / "raw").glob("quality_round_1_attempt_*.txt"))
    assert len(raws) == 1
    assert llm_client.ATTEMPTS.attempted_calls_total == 4
    assert (out_dir / "rounds" / "round_01" / "04_quality.json").exists() is False
    assert (out_dir / "rounds" / "round_01" / "approved_types_after.json"
            ).exists() is False
    assert (out_dir / "final_approved_types.json").exists() is False
    # No automatic retry happened: exactly the 3 calls up to quality.
    assert llm_client.BUDGET.stage_counts["quality"] == 1

    # Manual resume with a corrected verdict proceeds normally.
    # R2 proposes one gender duplicate -> zero new -> clean stop.
    _reset_budget()
    llm_client.reset_attempts()
    good_verdict = {
        "round": 1,
        "accepted_new": [{"type_id": "QS_R1_01",
                          "keep_template_ids": ["T_R1_01_01"]}],
        "duplicates": [],
        "rejected": [{"type_id": tid, "reason": "out of scope for retry"}
                     for tid in ("QS_R1_02", "QS_R1_03", "QS_R1_04",
                                 "QS_R1_05")],
    }
    r2_type = {
        "id": "QS_R2_01", "name": "Gender Again", "goal": "g",
        "uses": ["audio", "gender"], "input_count": 1,
        "answer_rule": "the gender label",
        "answer": {"kind": "field_value", "key": "gender"}}
    script2 = [
        good_verdict,
        {"round": 2, "question_types": [r2_type]},
        {"round": 2, "key_realizations": {"QS_R2_01": "giới tính"},
         "templates": [{"template_id": "T2", "question_type_id": "QS_R2_01",
                        "text": "Hãy xác định [KEY]?"}]},
        {"round": 2, "paraphrases": [
            {"template_id": "T2_P1", "source_template_id": "T2",
             "text": "Cho biết [KEY]?"}]},
        {"round": 2, "accepted_new": [], "duplicates": [
            {"type_id": "QS_R2_01", "duplicate_of": "QS_R1_01"}],
         "rejected": []},
    ]

    def fake2(client, stage, prompt, response_schema):
        from types import SimpleNamespace
        item = script2.pop(0)
        llm_client.BUDGET.consume(stage)
        return SimpleNamespace(text=json.dumps(item), parsed=None,
                               candidates=[], usage_metadata=None)

    monkeypatch.setattr(llm_client, "call_llm", fake2)
    preview = run_pipeline(dataset="vimd", real_llm=True,
                           resume_run="malformed-q")
    assert len(preview) > 0
    # Resume replays rounds 1 (pending quality) — needs rounds 2-3 too.
    final_types = json.loads(
        (out_dir / "final_approved_types.json").read_text(encoding="utf-8"))
    assert "QS_R1_01" in [t["type_id"]
                          for t in final_types["approved_types"]]

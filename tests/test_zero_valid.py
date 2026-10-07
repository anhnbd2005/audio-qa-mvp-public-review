"""Zero-valid short-circuit: gate with nothing to pass stops the run.

No Paraphrase/Quality calls, no fake calls, no budget spent beyond the
two design calls. Terminal checkpoint (stopped=true, next_action=null)
refuses resume with zero new requests.
"""

import json

import pytest

import src.autonomous_qa.authoring.llm_client as llm_client
from src.autonomous_qa.compiler.pipeline_runner import run_pipeline
from tests.test_safety import ScriptedLLM, _reset_budget, clean_budget  # noqa: F401


def _bad_style():
    return {
        "round": 1,
        "question_types": [
            {"id": "QS_R1_01", "name": "Duration Guess",
             "goal": "Guess utterance length.",
             "uses": ["audio", "duration"],
             "input_count": 1,
             "answer_rule": "The duration value of the audio.",
             "answer": {"kind": "field_value", "key": "duration"}},
            {"id": "QS_R1_02", "name": "Emotion Guess",
             "goal": "Guess speaker emotion.",
             "uses": ["audio", "emotion"],
             "input_count": 1,
             "answer_rule": "The emotion label of the speaker.",
             "answer": {"kind": "field_value", "key": "emotion"}},
        ],
    }


def _template_for_bad():
    return {
        "round": 1,
        "key_realizations": {},
        "templates": [
            {"template_id": "T_R1_01_01", "question_type_id": "QS_R1_01",
             "text": "Đoạn âm thanh dài bao lâu?"},
            {"template_id": "T_R1_02_01", "question_type_id": "QS_R1_02",
             "text": "Người nói đang có cảm xúc gì?"},
        ],
    }


def test_zero_valid_stops_before_paraphrase_quality(
        isolated_root, clean_budget, monkeypatch):
    fake = ScriptedLLM([_bad_style(), _template_for_bad()])
    monkeypatch.setattr(llm_client, "call_llm", fake)
    monkeypatch.setattr(llm_client, "create_llm_client", lambda: object())

    preview = run_pipeline(dataset="vimd", real_llm=True, run_id="zero-v")

    # Exactly the 2 design calls; nothing downstream, nothing faked.
    assert fake.calls == ["question_style", "question_template"]
    assert preview == []
    assert llm_client.BUDGET.stage_counts["question_style"] == 1
    assert llm_client.BUDGET.stage_counts["question_template"] == 1
    assert llm_client.BUDGET.stage_counts["paraphrase"] == 0
    assert llm_client.BUDGET.stage_counts["quality"] == 0
    assert llm_client.ATTEMPTS.attempted_calls_total == 2
    assert llm_client.ATTEMPTS.successful_calls_total == 2

    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "zero-v"
    rdir = out_dir / "rounds" / "round_01"
    # Audit trail persists; downstream artifacts never created.
    assert (rdir / "01_question_styles.json").is_file()
    assert (rdir / "02_templates.json").is_file()
    assert not (rdir / "03_paraphrases.json").exists()
    assert not (rdir / "04_quality.json").exists()

    summary = json.loads((out_dir / "loop_summary.json").read_text(
        encoding="utf-8"))
    assert summary["stop_reason"] == "no_valid_types_after_gate"
    assert summary["rounds"][0]["accepted_new_types"] == 0
    assert summary["rounds"][0]["generated_types"] == 2

    state = json.loads((out_dir / "run_state.json").read_text(
        encoding="utf-8"))
    assert state["stopped"] is True
    assert state["stop_reason"] == "no_valid_types_after_gate"
    assert state["next_action"] is None


def test_zero_valid_terminal_run_refuses_resume_with_zero_calls(
        isolated_root, clean_budget, monkeypatch):
    fake = ScriptedLLM([_bad_style(), _template_for_bad()])
    monkeypatch.setattr(llm_client, "call_llm", fake)
    monkeypatch.setattr(llm_client, "create_llm_client", lambda: object())
    run_pipeline(dataset="vimd", real_llm=True, run_id="zero-term")

    def _boom(*a, **k):
        raise AssertionError("no LLM request may be issued on resume")

    monkeypatch.setattr(llm_client, "call_llm", _boom)
    monkeypatch.setattr(llm_client, "create_llm_client", _boom)
    with pytest.raises(ValueError, match="already finished"):
        run_pipeline(dataset="vimd", real_llm=True, resume_run="zero-term")


def test_nonzero_gate_continues_and_excludes_rejects(
        isolated_root, clean_budget, monkeypatch):
    good = {
        "id": "QS_R1_01", "name": "Gender", "goal": "g",
        "uses": ["audio", "gender"], "input_count": 1,
        "answer_rule": "the gender label",
        "answer": {"kind": "field_value", "key": "gender"},
    }
    bad = {
        "id": "QS_R1_02", "name": "Emotion Guess", "goal": "g",
        "uses": ["audio", "emotion"], "input_count": 1,
        "answer_rule": "the emotion label",
        "answer": {"kind": "field_value", "key": "emotion"},
    }
    script = [
        {"round": 1, "question_types": [good, bad]},
        {"round": 1, "key_realizations": {"QS_R1_01": "giới tính"},
         "templates": [
             {"template_id": "T1", "question_type_id": "QS_R1_01",
              "text": "Hãy xác định [KEY] của người nói."},
             {"template_id": "T2", "question_type_id": "QS_R1_02",
              "text": "Cảm xúc là gì?"}]},
        {"round": 1, "paraphrases": [
            {"template_id": "T1_P1", "source_template_id": "T1",
             "text": "Cho biết [KEY] của người nói."}]},
        {"round": 1,
         "accepted_new": [{"type_id": "QS_R1_01",
                           "keep_template_ids": ["T1", "T1_P1"]}],
         "duplicates": [], "rejected": []},
        # Round 2 proposes only gate-invalid types -> short-circuit.
        {"round": 2, "question_types": [bad]},
        {"round": 2, "key_realizations": {},
         "templates": [
             {"template_id": "T2b", "question_type_id": "QS_R1_02",
              "text": "Cảm xúc là gì?"}]},
    ]
    fake = ScriptedLLM(script)
    monkeypatch.setattr(llm_client, "call_llm", fake)
    monkeypatch.setattr(llm_client, "create_llm_client", lambda: object())

    preview = run_pipeline(dataset="vimd", real_llm=True, run_id="mixed")
    assert fake.calls == ["question_style", "question_template",
                          "paraphrase", "quality",
                          "question_style", "question_template"]
    assert len(preview) > 0

    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "mixed"
    record = json.loads((out_dir / "rounds" / "round_01"
                         / "04_quality.json").read_text(encoding="utf-8"))
    assert [a["type_id"] for a in record["auto_rejected"]] == ["QS_R1_02"]
    final_types = json.loads((out_dir / "final_approved_types.json")
                             .read_text(encoding="utf-8"))
    assert [t["type_id"] for t in final_types["approved_types"]] == [
        "QS_R1_01"]
    summary = json.loads((out_dir / "loop_summary.json").read_text(
        encoding="utf-8"))
    assert summary["stop_reason"] == "no_valid_types_after_gate"
    assert summary["rounds"][-1]["accepted_new_types"] == 0

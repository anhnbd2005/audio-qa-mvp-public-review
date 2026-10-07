"""Zero-valid Gate short-circuit parity: fresh == resume (§33-§47).

Gate valid==0 MUST stop before Paraphrase/Quality in BOTH paths via the
ONE shared _post_gate_terminate helper. All LLM traffic faked (0 real).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.autonomous_qa.authoring.llm_client as llm_client
from src.autonomous_qa.authoring.llm_client import StageOutputValidationError
from src.autonomous_qa.compiler.pipeline_runner import run_pipeline
from tests.test_safety import ScriptedLLM, _reset_budget, clean_budget  # noqa: F401


def _payload(root: Path, *parts: str) -> dict:
    return json.loads(
        (root / "tests" / "fixtures" / Path(*parts)).with_suffix(
            ".json").read_text(encoding="utf-8"))


def _ns(item) -> SimpleNamespace:
    return SimpleNamespace(text=json.dumps(item, ensure_ascii=False),
                           parsed=None, candidates=[], usage_metadata=None)


def _zero_valid_style(n: int = 6) -> dict:
    return {
        "round": 3,
        "question_types": [
            {"id": f"QS_R3_{i:02d}", "name": f"Bad {i}", "goal": "g",
             "uses": ["audio", "emotion"], "input_count": 1,
             "answer_rule": "the emotion label",
             "answer": {"kind": "field_value", "key": "emotion"}}
            for i in range(1, n + 1)],
    }


def _zero_valid_template() -> dict:
    return {
        "round": 3,
        "key_realizations": {},
        "templates": [
            {"template_id": f"T3_{i:02d}", "question_type_id": f"QS_R3_{i:02d}",
             "text": "Cảm xúc là gì?"}
            for i in range(1, 7)],
    }


def _r2_scripts(root):
    """R1 full + R2 style/template/paraphrase + bad R2 quality."""
    style = _payload(root, "loop", "round_01", "style")
    tpl = _payload(root, "loop", "round_01", "template")
    para = _payload(root, "loop", "round_01", "paraphrase")
    qual = _payload(root, "loop", "round_01", "quality")
    r2style = _payload(root, "loop", "round_02", "style")
    r2tpl = _payload(root, "loop", "round_02", "template")
    r2para = _payload(root, "loop", "round_02", "paraphrase")
    r2qual = _payload(root, "loop", "round_02", "quality")
    bad_qual = json.loads(json.dumps(r2qual, ensure_ascii=False))
    bad_qual["rejected"] = [{"type_id": "T_R2_01_01_P01",
                             "reason": "template id as type"}]
    return [style, tpl, para, qual, r2style, r2tpl, r2para, bad_qual], r2qual


# §35/§36/§37/§38/§39 — exact real sequence, 11 paid calls, no R3 P/Q.
def test_35_real_sequence_r2_retry_then_r3_zero_valid(
        isolated_root, clean_budget, monkeypatch):
    script, good_r2qual = _r2_scripts(isolated_root)
    calls: list[str] = []

    def fake(client, stage, prompt, response_schema):
        item = script.pop(0)
        llm_client.BUDGET.consume(stage)
        calls.append(stage)
        return _ns(item)

    monkeypatch.setattr(llm_client, "call_llm", fake)
    monkeypatch.setattr(llm_client, "create_llm_client", lambda: object())
    with pytest.raises(StageOutputValidationError,
                       match="quality_unknown_type_id"):
        run_pipeline(dataset="vimd", real_llm=True, run_id="parity-r3")
    # R1(4) + R2 S/T/P + bad Q = 8 paid before resume.
    assert len(calls) == 8

    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "parity-r3"
    _reset_budget()
    llm_client.reset_attempts()
    script2 = [good_r2qual, _zero_valid_style(), _zero_valid_template()]
    calls2: list[str] = []

    def fake2(client, stage, prompt, response_schema):
        item = script2.pop(0)
        llm_client.BUDGET.consume(stage)
        calls2.append(stage)
        return _ns(item)

    monkeypatch.setattr(llm_client, "call_llm", fake2)
    preview = run_pipeline(dataset="vimd", real_llm=True,
                           resume_run="parity-r3")

    # R2 Quality retry first, then R3 Style + Template, then STOP.
    assert calls2[0] == "quality"
    assert calls2.count("question_style") == 1  # R3 only
    assert calls2.count("question_template") == 1
    r3_calls = calls2[1:]
    assert r3_calls == ["question_style", "question_template"]
    # Total paid: 8 + 1 retry + 2 = 11, NOT 13.
    assert len(calls) + len(calls2) == 11
    assert llm_client.BUDGET.count == 11

    # Raw inventory: R3 style/template exist (global attempt 03);
    # R3 paraphrase/quality absent.
    assert (out_dir / "raw" / "round_03_style_attempt_03.txt").is_file()
    assert (out_dir / "raw" / "round_03_template_attempt_03.txt").is_file()
    assert list((out_dir / "raw").glob("paraphrase_round_3_attempt_*.txt")) == []
    assert list((out_dir / "raw").glob("quality_round_3_attempt_*.txt")) == []

    # No fake empty semantic artifacts for skipped stages.
    assert not (out_dir / "rounds" / "round_03"
                / "03_paraphrases.json").exists()
    assert not (out_dir / "rounds" / "round_03"
                / "04_quality.json").exists()

    # Stop reason is the Gate reason, never zero_new_types.
    state = json.loads((out_dir / "run_state.json").read_text(
        encoding="utf-8"))
    assert state["stop_reason"] == "no_valid_types_after_gate"

    # Round accounting: 6 == 6 + 0 + 0 + 0.
    summary = json.loads((out_dir / "loop_summary.json").read_text(
        encoding="utf-8"))
    r3stats = summary["rounds"][-1]
    assert r3stats["round"] == 3
    assert r3stats["generated_types"] == 6
    assert r3stats["rejected"] == 6
    assert r3stats["accepted_new_types"] == 0
    assert r3stats["duplicates"] == 0
    assert (r3stats["generated_types"] == r3stats["rejected"]
            + r3stats["accepted_new_types"] + r3stats["duplicates"])

    # Finalization ran on the prior pool; R3 added nothing.
    final_types = json.loads((out_dir / "final_approved_types.json")
                             .read_text(encoding="utf-8"))
    assert all(not t["type_id"].startswith("QS_R3_")
               for t in final_types["approved_types"])
    assert len(preview) > 0


def _r1_quality_no_derived_keep() -> dict:
    """R1 verdict avoiding kept derived-[KEY] templates.

    Accepts gender + slot-free types only, so finalization needs just the
    gender binding. (A kept derived-[KEY] template renders under its
    source key, which is unrelated to this patch's control flow.)
    """
    return {
        "round": 1,
        "accepted_new": [
            {"type_id": "QS_R1_01",
             "keep_template_ids": ["T_R1_01_01", "T_R1_01_01_P01"]},
            {"type_id": "QS_R1_03",
             "keep_template_ids": ["T_R1_03_01", "T_R1_03_01_P01"]},
            {"type_id": "QS_R1_04",
             "keep_template_ids": ["T_R1_04_01", "T_R1_04_01_P01"]},
        ],
        "duplicates": [],
        "rejected": [{"type_id": tid, "reason": "judge reject"}
                     for tid in ("QS_R1_02", "QS_R1_05")],
    }


# §34 — resumed zero-valid behaves exactly like fresh.
def test_34_resumed_zero_valid_matches_fresh(
        isolated_root, clean_budget, monkeypatch):
    # Fresh path first: R1 good, R2 zero-valid.
    style = _payload(isolated_root, "loop", "round_01", "style")
    tpl = _payload(isolated_root, "loop", "round_01", "template")
    para = _payload(isolated_root, "loop", "round_01", "paraphrase")
    qual = _r1_quality_no_derived_keep()
    r2s = _payload(isolated_root, "loop", "round_02", "style")
    r2s_bad = json.loads(json.dumps(r2s, ensure_ascii=False))
    for t in r2s_bad["question_types"]:
        t["uses"] = ["audio", "emotion"]
        t["answer"] = {"kind": "field_value", "key": "emotion"}
    r2t_bad = {"round": 2, "key_realizations": {}, "templates": [
        {"template_id": "TB", "question_type_id": "QS_R2_01",
         "text": "Cảm xúc là gì?"}]}
    fake = ScriptedLLM([style, tpl, para, qual, r2s_bad, r2t_bad])
    monkeypatch.setattr(llm_client, "call_llm", fake)
    monkeypatch.setattr(llm_client, "create_llm_client", lambda: object())
    preview_fresh = run_pipeline(dataset="vimd", real_llm=True,
                                 run_id="fresh-zv")
    assert fake.calls == ["question_style", "question_template",
                          "paraphrase", "quality",
                          "question_style", "question_template"]
    out_fresh = isolated_root / "outputs" / "runs" / "vimd" / "fresh-zv"
    state_fresh = json.loads((out_fresh / "run_state.json").read_text(
        encoding="utf-8"))
    assert state_fresh["stop_reason"] == "no_valid_types_after_gate"

    # Resumed path: crash before R2 style (provider boom), then same R2.
    _reset_budget()
    llm_client.reset_attempts()
    boom = RuntimeError("simulated provider failure")
    fake2 = ScriptedLLM([style, tpl, para, qual, boom])
    monkeypatch.setattr(llm_client, "call_llm", fake2)
    with pytest.raises(RuntimeError, match="simulated provider failure"):
        run_pipeline(dataset="vimd", real_llm=True, run_id="resume-zv")

    _reset_budget()
    llm_client.reset_attempts()
    fake3 = ScriptedLLM([r2s_bad, r2t_bad])
    monkeypatch.setattr(llm_client, "call_llm", fake3)
    preview_res = run_pipeline(dataset="vimd", real_llm=True,
                               resume_run="resume-zv")
    assert fake3.calls == ["question_style", "question_template"]
    out_res = isolated_root / "outputs" / "runs" / "vimd" / "resume-zv"
    state_res = json.loads((out_res / "run_state.json").read_text(
        encoding="utf-8"))
    assert state_res["stop_reason"] == state_fresh["stop_reason"]
    assert ([t["type_id"] for t in json.loads(
        (out_res / "final_approved_types.json").read_text(
            encoding="utf-8"))["approved_types"]] == [t["type_id"] for t in json.loads(
        (out_fresh / "final_approved_types.json").read_text(
            encoding="utf-8"))["approved_types"]])
    assert preview_res == preview_fresh


# §40 — Gate>0 + Quality accepts 0 still yields zero_new_types.
def test_40_quality_zero_new_preserved(isolated_root, clean_budget,
                                       monkeypatch):
    style = _payload(isolated_root, "loop", "round_01", "style")
    tpl = _payload(isolated_root, "loop", "round_01", "template")
    para = _payload(isolated_root, "loop", "round_01", "paraphrase")
    qual = _payload(isolated_root, "loop", "round_01", "quality")
    qual_zero = json.loads(json.dumps(qual, ensure_ascii=False))
    qual_zero["accepted_new"] = []
    qual_zero["duplicates"] = []
    # Valid R1 types per gate: 01-05 (06/07/08 auto-rejected).
    qual_zero["rejected"] = [
        {"type_id": tid, "reason": "judge reject"}
        for tid in ("QS_R1_01", "QS_R1_02", "QS_R1_03", "QS_R1_04",
                    "QS_R1_05")]
    fake = ScriptedLLM([style, tpl, para, qual_zero])
    monkeypatch.setattr(llm_client, "call_llm", fake)
    monkeypatch.setattr(llm_client, "create_llm_client", lambda: object())
    run_pipeline(dataset="vimd", real_llm=True, run_id="zero-new")
    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "zero-new"
    state = json.loads((out_dir / "run_state.json").read_text(
        encoding="utf-8"))
    assert state["stop_reason"] == "zero_new_types"


# §42/§43 — crash after terminal Gate resumes at finalization, no LLM.
def test_42_crash_after_terminal_gate_resumes_finalize(
        isolated_root, clean_budget, monkeypatch):
    from src.autonomous_qa.compiler.pipeline_runner import TEMPLATE_CONTRACT_VERSION, _ResumedRun, load_sample

    fx = isolated_root / "tests" / "fixtures" / "loop"
    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "crash-term"
    raw_dir = out_dir / "raw"
    (out_dir / "rounds" / "round_01").mkdir(parents=True)
    (out_dir / "rounds" / "round_02").mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    for kind, name in (("style", "01_question_styles.json"),
                       ("template", "02_templates.json"),
                       ("paraphrase", "03_paraphrases.json"),
                       ("quality", "04_quality.json")):
        (out_dir / "rounds" / "round_01" / name).write_bytes(
            (fx / "round_01" / f"{kind}.json").read_bytes())
    # R1 verdict without kept derived-[KEY] templates (control-flow only).
    (out_dir / "rounds" / "round_01" / "04_quality.json").write_text(
        json.dumps(_r1_quality_no_derived_keep()), encoding="utf-8")
    # R2 zero-valid gate artifacts only (no paraphrase/quality files).
    (out_dir / "rounds" / "round_02" / "01_question_styles.json"
     ).write_text(json.dumps(_zero_valid_style()), encoding="utf-8")
    (out_dir / "rounds" / "round_02" / "02_templates.json"
     ).write_text(json.dumps(_zero_valid_template()), encoding="utf-8")
    for name in ("round_01_style_attempt_01.txt",
                 "round_01_template_attempt_01.txt",
                 "paraphrase_round_1_attempt_01.txt",
                 "quality_round_1_attempt_01.txt",
                 "round_02_style_attempt_02.txt",
                 "round_02_template_attempt_02.txt"):
        (raw_dir / name).write_text("{}", encoding="utf-8")
    (out_dir / "run_state.json").write_text(json.dumps({
        "run_id": "crash-term", "dataset": "vimd", "mode": "real",
        "completed": ["style_round_1", "template_round_1",
                      "paraphrase_round_1", "quality_round_1",
                      "style_round_2", "template_round_2"],
        "current_round": 2, "next_action": "finalize",
        "low_gain_streak": 0, "stopped": True,
        "stop_reason": "no_valid_types_after_gate",
        "spent_calls": {"question_style": 2, "question_template": 2,
                        "paraphrase": 1, "quality": 1},
        "stage_attempts": {"question_style": 2, "question_template": 2,
                           "paraphrase": 1, "quality": 1},
        "attempted_calls_total": 6, "successful_calls_total": 5,
        "last_finish_reason": "STOP", "last_failed_stage": None,
        "loop_config": {"max_rounds": 3, "min_new_type_rate": 0.15,
                        "saturation_patience": 2,
                        "immediate_stop_if_zero_new": True},
            "template_contract_version": TEMPLATE_CONTRACT_VERSION,
            "answer_reference_contract_version": "kind-scoped-v1",
            "answer_shape_contract_version": "canonical-v1",
            "executable_signature_contract_version": "v1",
            "derived_field_contract_version": "data-backed-v1",
        }), encoding="utf-8")

    run = _ResumedRun(dataset="vimd", out_dir=out_dir, raw_dir=raw_dir,
                      rounds_dir=out_dir / "rounds", readme="",
                      rows=load_sample("vimd"), client=None)
    run.load_and_validate()
    assert run._next_action == "finalize"

    def _boom(*a, **k):
        raise AssertionError("no LLM request may be issued on resume")

    monkeypatch.setattr(llm_client, "call_llm", _boom)
    monkeypatch.setattr(llm_client, "create_llm_client", lambda: object())
    preview = run_pipeline(dataset="vimd", real_llm=True,
                           resume_run="crash-term")
    assert len(preview) > 0
    state = json.loads((out_dir / "run_state.json").read_text(
        encoding="utf-8"))
    assert state["stop_reason"] == "no_valid_types_after_gate"
    # Still no R2 paraphrase/quality evidence after finalization resume.
    assert list(raw_dir.glob("paraphrase_round_2_attempt_*.txt")) == []
    assert list(raw_dir.glob("quality_round_2_attempt_*.txt")) == []


# §47 — a normal mock run can reach a Gate-0 round and short-circuit.
def test_47_mock_zero_valid_short_circuit(isolated_root, clean_budget):
    fx = isolated_root / "tests" / "fixtures" / "loop" / "round_03"
    (fx / "style.json").write_text(
        json.dumps(_zero_valid_style()), encoding="utf-8")
    (fx / "template.json").write_text(
        json.dumps(_zero_valid_template()), encoding="utf-8")
    preview = run_pipeline(dataset="vimd", real_llm=False)
    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "mock"
    summary = json.loads((out_dir / "loop_summary.json").read_text(
        encoding="utf-8"))
    assert summary["stop_reason"] == "no_valid_types_after_gate"
    assert summary["rounds"][-1]["generated_types"] == 6
    assert summary["rounds"][-1]["rejected"] == 6
    assert not (out_dir / "rounds" / "round_03"
                / "03_paraphrases.json").exists()
    assert not (out_dir / "rounds" / "round_03"
                / "04_quality.json").exists()
    assert len(preview) == 10

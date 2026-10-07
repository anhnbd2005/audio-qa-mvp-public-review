"""Paid attempts + thinking budgets. All LLM traffic is faked (0 real calls)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.autonomous_qa.authoring.llm_client as llm_client
from src.common.config import CONFIG
from src.autonomous_qa.authoring.llm_client import StageOutputValidationError, reset_attempts
from src.autonomous_qa.compiler.pipeline_runner import run_pipeline


@pytest.fixture()
def clean_all():
    saved_b = (llm_client.BUDGET.count, dict(llm_client.BUDGET.stage_counts),
               llm_client.BUDGET.display_total)
    llm_client.BUDGET.count = 0
    llm_client.BUDGET.stage_counts = {
        k: 0 for k in llm_client.BUDGET.stage_counts}
    saved_a = (
        llm_client.ATTEMPTS.attempted_calls_total,
        llm_client.ATTEMPTS.successful_calls_total,
        dict(llm_client.ATTEMPTS.stage_attempts),
        dict(llm_client.ATTEMPTS.stage_successes),
        llm_client.ATTEMPTS.last_finish_reason,
        llm_client.ATTEMPTS.last_failed_stage,
    )
    reset_attempts()
    try:
        yield
    finally:
        (llm_client.BUDGET.count, counts, llm_client.BUDGET.display_total) = (
            saved_b[0], saved_b[1], saved_b[2])
        llm_client.BUDGET.stage_counts.clear()
        llm_client.BUDGET.stage_counts.update(counts)
        reset_attempts(
            attempted=saved_a[0], successful=saved_a[1],
            stage_attempts=saved_a[2], stage_successes=saved_a[3],
            last_finish_reason=saved_a[4], last_failed_stage=saved_a[5])


def _reset_budget():
    llm_client.BUDGET.count = 0
    llm_client.BUDGET.stage_counts = {
        k: 0 for k in llm_client.BUDGET.stage_counts}


def _payload(root: Path, *parts: str) -> dict:
    return json.loads(
        (root / "tests" / "fixtures" / Path(*parts)).with_suffix(
            ".json").read_text(encoding="utf-8"))


def _full_script(root: Path) -> list:
    script = []
    for r in (1, 2, 3):
        for kind in ("style", "template", "paraphrase", "quality"):
            script.append(_payload(root, "loop", f"round_{r:02d}", kind))
    return script


def _maxtokens_call(calls: list, truncated: str):
    def fake(client, stage, prompt, response_schema):
        llm_client.BUDGET.consume(stage)
        calls.append(stage)
        return SimpleNamespace(
            text=truncated, parsed=None,
            candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
            usage_metadata=None)
    return fake


def test_maxtokens_attempt_counted_despite_parse_failure(
        isolated_root, clean_all, monkeypatch):
    style = _payload(isolated_root, "loop", "round_01", "style")
    truncated = json.dumps(style, ensure_ascii=False)[:400]
    calls: list[str] = []
    monkeypatch.setattr(llm_client, "call_llm",
                        _maxtokens_call(calls, truncated))
    monkeypatch.setattr(llm_client, "create_llm_client", lambda: object())

    with pytest.raises(StageOutputValidationError):
        run_pipeline(dataset="vimd", real_llm=True, run_id="att-fail")

    assert calls == ["question_style"]
    assert llm_client.ATTEMPTS.attempted_calls_total == 1
    assert llm_client.ATTEMPTS.stage_attempts == {"question_style": 1}
    assert llm_client.ATTEMPTS.successful_calls_total == 0
    assert llm_client.ATTEMPTS.last_finish_reason == "MAX_TOKENS"
    assert llm_client.ATTEMPTS.last_failed_stage == "question_style"

    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "att-fail"
    raw = (out_dir / "raw" / "round_01_style_attempt_01.txt").read_text(
        encoding="utf-8")
    assert raw == truncated  # failed raw preserved, never parsed
    assert not (out_dir / "rounds" / "round_01"
                / "01_question_styles.json").exists()

    state = json.loads((out_dir / "run_state.json").read_text(
        encoding="utf-8"))
    assert state["completed"] == []
    assert state["next_action"] == "style_round_1"
    assert state["attempted_calls_total"] == 1
    assert state["stage_attempts"] == {"question_style": 1}
    assert state["successful_calls_total"] == 0
    assert state["last_finish_reason"] == "MAX_TOKENS"
    assert state["last_failed_stage"] == "question_style"


def test_failed_raw_preserved_and_resume_retries_only_failed_stage(
        isolated_root, clean_all, monkeypatch):
    style = _payload(isolated_root, "loop", "round_01", "style")
    truncated = json.dumps(style, ensure_ascii=False)[:400]
    calls: list[str] = []
    monkeypatch.setattr(llm_client, "call_llm",
                        _maxtokens_call(calls, truncated))
    monkeypatch.setattr(llm_client, "create_llm_client", lambda: object())

    with pytest.raises(StageOutputValidationError):
        run_pipeline(dataset="vimd", real_llm=True, run_id="att-retry")

    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "att-retry"
    attempt_01 = (out_dir / "raw" / "round_01_style_attempt_01.txt")
    before = attempt_01.read_bytes()

    # Resume in a fresh process: budget reset, good payloads from here on.
    _reset_budget()
    reset_attempts()
    script = _full_script(isolated_root)

    def fake_good(client, stage, prompt, response_schema):
        item = script.pop(0)
        llm_client.BUDGET.consume(stage)
        calls.append(stage)
        return SimpleNamespace(text=json.dumps(item), parsed=None,
                               candidates=[], usage_metadata=None)

    monkeypatch.setattr(llm_client, "call_llm", fake_good)
    preview = run_pipeline(dataset="vimd", real_llm=True,
                           resume_run="att-retry")

    assert calls == ["question_style"] + [
        "question_style", "question_template", "paraphrase", "quality",
        "question_style", "question_template", "paraphrase", "quality",
        "question_style", "question_template"]
    assert attempt_01.read_bytes() == before  # attempt_01 never overwritten
    attempt_02 = out_dir / "raw" / "round_01_style_attempt_02.txt"
    assert json.loads(attempt_02.read_text(encoding="utf-8")) == style
    from src.autonomous_qa.core.schemas import StyleRoundOutput
    assert json.loads((out_dir / "rounds" / "round_01"
                       / "01_question_styles.json").read_text(
        encoding="utf-8")) == StyleRoundOutput(**style).model_dump()
    assert len(preview) == 10
    # R3 is a terminal signature-dedup round: Style+Template only.
    assert llm_client.BUDGET.stage_counts == {
        "question_style": 4, "question_template": 3,
        "paraphrase": 2, "quality": 2}
    assert llm_client.BUDGET.count == 11  # 10 successes + 1 failed attempt
    assert llm_client.ATTEMPTS.attempted_calls_total == 11
    assert llm_client.ATTEMPTS.successful_calls_total == 10


def test_completed_style_cannot_be_rerun(isolated_root, clean_all,
                                        monkeypatch):
    from tests.test_safety import ScriptedLLM

    fake = ScriptedLLM(_full_script(isolated_root))
    monkeypatch.setattr(llm_client, "call_llm", fake)
    monkeypatch.setattr(llm_client, "create_llm_client", lambda: object())
    run_pipeline(dataset="vimd", real_llm=True, run_id="no-rerun")

    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "no-rerun"
    state_path = out_dir / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "style_round_1" in state["completed"]
    state["next_action"] = "style_round_1"  # tamper: rewind a success
    state_path.write_text(json.dumps(state), encoding="utf-8")

    fake.calls.clear()
    with pytest.raises(ValueError, match="already completed"):
        run_pipeline(dataset="vimd", real_llm=True, resume_run="no-rerun")
    assert fake.calls == []


class _CaptureModels:
    def __init__(self, outer, payload):
        self.outer = outer
        self.payload = payload

    def generate_content(self, **kwargs):
        self.outer.configs.append(kwargs["config"])
        return SimpleNamespace(text=json.dumps(self.payload), parsed=None,
                               candidates=[], usage_metadata=None)


class _CaptureClient:
    def __init__(self, payload):
        self.configs: list = []
        self.models = _CaptureModels(self, payload)


@pytest.mark.parametrize("stage,fixture_parts,runner", [
    ("question_style", ("loop", "round_01", "style"), "style"),
    ("question_template", ("loop", "round_01", "template"), "template"),
    ("paraphrase", ("loop", "round_01", "paraphrase"), "paraphrase"),
    ("quality", ("loop", "round_01", "quality"), "quality"),
])
def test_thinking_budget_reaches_config(isolated_root, clean_all, tmp_path,
                                        stage, fixture_parts, runner):
    import src.autonomous_qa.language.paraphrase as paraphrase_mod
    import src.autonomous_qa.certification.quality as quality_mod
    import src.autonomous_qa.language.question_style as style_mod
    import src.autonomous_qa.language.question_template as template_mod

    payload = _payload(isolated_root, *fixture_parts)
    client = _CaptureClient(payload)
    raw = tmp_path / "raw" / f"{stage}.txt"

    if runner == "style":
        style_mod.run_question_style_round(
            client, readme="r", dynamic_fewshots=[], round_idx=1,
            real_llm=True, raw_path=raw)
    elif runner == "template":
        template_mod.run_question_template_round(
            client, readme="r", round_types=[], template_bank=[],
            round_idx=1, real_llm=True, raw_path=raw)
    elif runner == "paraphrase":
        paraphrase_mod.run_paraphrase_round(
            client, templates=[], round_idx=1,
            real_llm=True, raw_path=raw)
    else:
        quality_mod.run_quality_round(
            client, bundles=[], approved_fewshots=[], round_idx=1,
            real_llm=True, raw_path=raw)

    assert len(client.configs) == 1
    thinking = client.configs[0].thinking_config
    expected = CONFIG["stages"][stage]["thinking_budget"]
    assert thinking.thinking_budget == expected
    assert thinking.include_thoughts is not True

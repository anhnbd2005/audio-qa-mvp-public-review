"""Safety: output isolation, unique run_ids, generic checkpoint/resume.

All LLM traffic is scripted fakes (real_llm=True path, zero network).
Global BUDGET is saved/restored around every test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.autonomous_qa.authoring.llm_client as llm_client
from src.autonomous_qa.compiler.pipeline_runner import run_pipeline


@pytest.fixture()
def clean_budget():
    saved = (llm_client.BUDGET.count, dict(llm_client.BUDGET.stage_counts),
             llm_client.BUDGET.display_total)
    llm_client.BUDGET.count = 0
    llm_client.BUDGET.stage_counts = {
        k: 0 for k in llm_client.BUDGET.stage_counts}
    saved_attempts = (
        llm_client.ATTEMPTS.attempted_calls_total,
        llm_client.ATTEMPTS.successful_calls_total,
        dict(llm_client.ATTEMPTS.stage_attempts),
        dict(llm_client.ATTEMPTS.stage_successes),
        llm_client.ATTEMPTS.last_finish_reason,
        llm_client.ATTEMPTS.last_failed_stage,
    )
    llm_client.reset_attempts()
    try:
        yield llm_client.BUDGET
    finally:
        (llm_client.BUDGET.count, counts, llm_client.BUDGET.display_total) = (
            saved[0], saved[1], saved[2])
        llm_client.BUDGET.stage_counts.clear()
        llm_client.BUDGET.stage_counts.update(counts)
        llm_client.reset_attempts(
            attempted=saved_attempts[0],
            successful=saved_attempts[1],
            stage_attempts=saved_attempts[2],
            stage_successes=saved_attempts[3],
            last_finish_reason=saved_attempts[4],
            last_failed_stage=saved_attempts[5],
        )


def _payload(root: Path, *parts: str) -> dict:
    return json.loads(
        (root / "tests" / "fixtures" / Path(*parts)).with_suffix(
            ".json").read_text(encoding="utf-8"))


def _full_script(root: Path) -> list:
    """12 design calls: 4 stages x 3 rounds, converging at round 3."""
    script = []
    for r in (1, 2, 3):
        for kind in ("style", "template", "paraphrase", "quality"):
            script.append(_payload(root, "loop", f"round_{r:02d}", kind))
    return script


class ScriptedLLM:
    """Fake call_llm: spends budget, serves scripted payloads, records."""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[str] = []

    def __call__(self, client, stage, prompt, response_schema):
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        llm_client.BUDGET.consume(stage)
        self.calls.append(stage)
        return SimpleNamespace(text=json.dumps(item), parsed=None,
                               candidates=[], usage_metadata=None)


@pytest.fixture()
def scripted(monkeypatch):
    fake = ScriptedLLM([])
    monkeypatch.setattr(llm_client, "call_llm", fake)
    monkeypatch.setattr(llm_client, "create_llm_client", lambda: object())
    return fake


def _reset_budget():
    llm_client.BUDGET.count = 0
    llm_client.BUDGET.stage_counts = {
        k: 0 for k in llm_client.BUDGET.stage_counts}
    # A new invocation is a new process: attempt numbering restarts too.
    llm_client.reset_attempts()


def _snapshot_tree(out_dir: Path) -> dict:
    return {
        p.relative_to(out_dir).as_posix(): hashlib.sha256(
            p.read_bytes()).hexdigest()
        for p in sorted(out_dir.rglob("*")) if p.is_file()
    }


def test_mock_never_touches_real_tree(isolated_root, clean_budget):
    preview = run_pipeline(dataset="vimd", real_llm=False)
    assert len(preview) == 10
    assert (isolated_root / "outputs" / "runs" / "vimd" / "mock"
            / "preview.jsonl").is_file()
    run_root = isolated_root / "outputs" / "runs" / "vimd"
    assert {p.name for p in run_root.iterdir()} == {"mock"}


def test_two_real_run_ids_never_overwrite(isolated_root, clean_budget,
                                          scripted):
    scripted.script = _full_script(isolated_root)
    run_pipeline(dataset="vimd", real_llm=True, run_id="run-a")
    assert scripted.calls.count("question_style") == 3
    assert scripted.calls.count("question_template") == 3
    assert scripted.calls.count("paraphrase") == 2
    assert scripted.calls.count("quality") == 2
    assert len(scripted.calls) == 10

    with pytest.raises(FileExistsError):
        run_pipeline(dataset="vimd", real_llm=True, run_id="run-a")

    _reset_budget()  # a new run is a new process with a fresh budget
    scripted.script = _full_script(isolated_root)
    run_pipeline(dataset="vimd", real_llm=True, run_id="run-b")
    dir_a = isolated_root / "outputs" / "runs" / "vimd" / "run-a"
    dir_b = isolated_root / "outputs" / "runs" / "vimd" / "run-b"
    assert (dir_a / "preview.jsonl").is_file()
    assert (dir_b / "preview.jsonl").is_file()

    with pytest.raises(ValueError, match="already finished"):
        run_pipeline(dataset="vimd", real_llm=True, resume_run="run-b")


def test_crash_before_quality_r2_resumes_exact_stage(
        isolated_root, clean_budget, scripted):
    script = _full_script(isolated_root)
    # Crash instead of the round-2 quality request (index 7:
    # s1,t1,p1,q1,s2,t2,p2 | q2 ...).
    boom = RuntimeError("simulated provider failure")
    scripted.script = script[:7] + [boom] + script[7:]
    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "crash-r2"

    with pytest.raises(RuntimeError, match="simulated provider failure"):
        run_pipeline(dataset="vimd", real_llm=True, run_id="crash-r2")

    state = json.loads((out_dir / "run_state.json").read_text(
        encoding="utf-8"))
    assert state["next_action"] == "quality_round_2"
    assert (out_dir / "rounds" / "round_02"
            / "03_paraphrases.json").is_file()
    assert not (out_dir / "rounds" / "round_02" / "04_quality.json").exists()
    before = _snapshot_tree(out_dir)
    assert scripted.calls == ["question_style", "question_template",
                              "paraphrase", "quality",
                              "question_style", "question_template",
                              "paraphrase"]

    # Resume is a new invocation with a fresh budget object.
    _reset_budget()
    scripted.script = script[7:]
    scripted.calls.clear()
    preview = run_pipeline(dataset="vimd", real_llm=True,
                           resume_run="crash-r2")
    assert scripted.calls == ["quality", "question_style",
                              "question_template"]
    assert len(preview) == 10

    after = _snapshot_tree(out_dir)
    for path, digest in before.items():
        if "round_02/04_quality.json" in path:
            continue  # did not exist before
        if "round_02/approved_types_after.json" in path:
            continue  # did not exist before
        if path in ("run_state.json", "run_meta.json"):
            continue  # checkpoints legitimately advance
        if path.startswith("rounds/round_03"):
            continue
        if path in ("final_approved_types.json", "final_template_pool.json",
                    "loop_summary.json", "preview.jsonl",
                    "approved_type_candidates.json"):
            continue  # finalize outputs did not exist before
        assert after[path] == digest, f"completed artifact changed: {path}"

    totals = llm_client.BUDGET.stage_counts
    assert totals == {"question_style": 3, "question_template": 3,
                      "paraphrase": 2, "quality": 2}


def test_streak_restored_from_disk(isolated_root, clean_budget):
    from src.autonomous_qa.compiler.pipeline_runner import _ResumedRun

    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "streak-1"
    raw_dir = out_dir / "raw"
    (out_dir / "rounds" / "round_01").mkdir(parents=True)
    raw_dir.mkdir(parents=True)

    fx = isolated_root / "tests" / "fixtures" / "loop"
    for kind, name in (("style", "01_question_styles.json"),
                       ("template", "02_templates.json"),
                       ("paraphrase", "03_paraphrases.json")):
        (out_dir / "rounds" / "round_01" / name).write_bytes(
            (fx / "round_01" / f"{kind}.json").read_bytes())
    # Durable paid-attempt evidence (1 per stage) so resume inventory
    # matches the checkpoint (D == C).
    (raw_dir / "round_01_style_attempt_01.txt").write_text("{}", encoding="utf-8")
    (raw_dir / "round_01_template_attempt_01.txt").write_text("{}", encoding="utf-8")
    (raw_dir / "paraphrase_round_1_attempt_01.txt").write_text("{}", encoding="utf-8")
    (raw_dir / "quality_round_1_attempt_01.txt").write_text("{}", encoding="utf-8")

    # Round 1 as judged: 8 proposed, 5 accepted -> rate 0.625, streak 0.
    # Hand-edit the verdict down to 1 accepted (others rejected, so the
    # partition still covers every valid type) to force a low-gain streak.
    quality = json.loads(
        (fx / "round_01" / "quality.json").read_text(encoding="utf-8"))
    kept_first = quality["accepted_new"][:1]
    rest_ids = [a["type_id"] for a in quality["accepted_new"][1:]]
    quality["accepted_new"] = kept_first
    quality["duplicates"] = []
    quality["rejected"] = [
        {"type_id": tid, "reason": "hand-edited low-gain fixture"}
        for tid in rest_ids]
    (out_dir / "rounds" / "round_01" / "04_quality.json").write_text(
        json.dumps(quality), encoding="utf-8")
    # 1/8 accepted -> rate 0.125 < 0.15 -> low_gain_streak becomes 1.
    from src.autonomous_qa.compiler.pipeline_runner import TEMPLATE_CONTRACT_VERSION
    (out_dir / "run_state.json").write_text(json.dumps({
        "run_id": "streak-1", "dataset": "vimd", "mode": "real",
        "completed": ["style_round_1", "template_round_1",
                      "paraphrase_round_1", "quality_round_1"],
        "current_round": 1, "next_action": "style_round_2",
        "low_gain_streak": 1, "stopped": False, "stop_reason": None,
        "spent_calls": {"question_style": 1, "question_template": 1,
                        "paraphrase": 1, "quality": 1},
        "loop_config": {
            "max_rounds": 3, "min_new_type_rate": 0.15,
            "saturation_patience": 2, "immediate_stop_if_zero_new": True},
        "template_contract_version": TEMPLATE_CONTRACT_VERSION,
        "answer_reference_contract_version": "kind-scoped-v1",
        "answer_shape_contract_version": "canonical-v1",
        "executable_signature_contract_version": "v1",
        "derived_field_contract_version": "data-backed-v1",
    }), encoding="utf-8")

    from src.autonomous_qa.compiler.pipeline_runner import load_sample
    run = _ResumedRun(dataset="vimd", out_dir=out_dir, raw_dir=raw_dir,
                      rounds_dir=out_dir / "rounds", readme="",
                      rows=load_sample("vimd"), client=None)
    run.load_and_validate()
    assert run.controller.low_gain_streak == 1
    assert run._next_action == "style_round_2"
    assert len(run.history) == 1
    assert run.history[0]["new_type_rate"] == 0.125
    assert len(run.approved_pool) == 1


def test_tampered_spent_budget_refuses_resume(isolated_root, clean_budget,
                                              scripted):
    scripted.script = _full_script(isolated_root)
    run_pipeline(dataset="vimd", real_llm=True, run_id="tamper")
    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "tamper"

    state_path = out_dir / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["spent_calls"]["paraphrase"] = 5  # above cap, disk shows 3
    # Rewind past finalize so the spent check (not the rerun guard) fires.
    state["completed"].remove("finalize")
    state["next_action"] = "finalize"  # keep otherwise plausible
    state_path.write_text(json.dumps(state), encoding="utf-8")

    scripted.calls.clear()  # refusal must issue zero new requests
    with pytest.raises(ValueError, match="spent calls"):
        run_pipeline(dataset="vimd", real_llm=True, resume_run="tamper")
    assert scripted.calls == []


def test_legacy_v2_checkpoint_refuses_resume(isolated_root, clean_budget,
                                             scripted):
    """Checkpoints from the old template-pool architecture cannot resume."""
    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "legacy"
    out_dir.mkdir(parents=True)
    (out_dir / "run_state.json").write_text(json.dumps({
        "run_id": "legacy", "dataset": "vimd", "mode": "real",
        "completed": ["question_style", "question_template"],
        "current_round": 0, "next_action": "paraphrase_round_1",
        "low_gain_streak": 0, "stopped": False, "stop_reason": None,
        "spent_calls": {"question_style": 1, "question_template": 1},
        "loop_config": {
            "max_rounds": 3, "min_new_unique_rate": 0.15,
            "saturation_patience": 2, "immediate_stop_if_zero_new": True},
    }), encoding="utf-8")

    scripted.calls.clear()
    with pytest.raises(ValueError, match="Loop config changed"):
        run_pipeline(dataset="vimd", real_llm=True, resume_run="legacy")
    assert scripted.calls == []

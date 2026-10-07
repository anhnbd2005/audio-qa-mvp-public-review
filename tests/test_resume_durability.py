"""Resume durability: raw-before-validate state machine (spec §28-§50).

All LLM traffic is faked (0 real calls). Uses isolated_root so the broken
real run is never touched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.autonomous_qa.authoring.llm_client as llm_client
from src.autonomous_qa.authoring.llm_client import (
    StageOutputValidationError,
    atomic_write_new_file,
    record_paid_response,
    reset_attempts,
    scan_raw_attempts,
    require_contiguous,
)
from src.autonomous_qa.compiler.pipeline_runner import TEMPLATE_CONTRACT_VERSION, run_pipeline


@pytest.fixture()
def clean_all():
    saved_b = (llm_client.BUDGET.count, dict(llm_client.BUDGET.stage_counts),
               llm_client.BUDGET.display_total)
    llm_client.BUDGET.count = 0
    llm_client.BUDGET.stage_counts = {k: 0 for k in llm_client.BUDGET.stage_counts}
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


def _payload(root: Path, *parts: str) -> dict:
    return json.loads(
        (root / "tests" / "fixtures" / Path(*parts)).with_suffix(
            ".json").read_text(encoding="utf-8"))


def _reset_budget():
    llm_client.BUDGET.count = 0
    llm_client.BUDGET.stage_counts = {k: 0 for k in llm_client.BUDGET.stage_counts}
    llm_client.reset_attempts()


def _ns(payload_or_text, finish="STOP", parsed=None):
    text = payload_or_text if isinstance(payload_or_text, str) else json.dumps(
        payload_or_text, ensure_ascii=False)
    cands = [] if finish is None else [SimpleNamespace(finish_reason=finish)]
    return SimpleNamespace(text=text, parsed=parsed, candidates=cands,
                           usage_metadata=None)


def _run_r1_plus_r2style(root):
    """Script payloads: R1 full round + R2 style (5 valid calls)."""
    return [_payload(root, "loop", "round_01", k)
            for k in ("style", "template", "paraphrase", "quality")] + [
        _payload(root, "loop", "round_02", "style")]


def _bad_template(root):
    tpl = _payload(root, "loop", "round_02", "template")
    tpl = json.loads(json.dumps(tpl, ensure_ascii=False))
    tpl["key_realizations"] = {"KEY": "gioi tinh"}
    return tpl


# --- TEST 1: exact observed template bug (§28) -----------------------------
def test_01_malformed_key_template_is_paid_and_durable(
        isolated_root, clean_all, monkeypatch):
    script = _run_r1_plus_r2style(isolated_root) + [_bad_template(isolated_root)]
    calls: list[str] = []

    def fake(client, stage, prompt, response_schema):
        item = script.pop(0)
        llm_client.BUDGET.consume(stage)
        calls.append(stage)
        return _ns(item)

    monkeypatch.setattr(llm_client, "call_llm", fake)
    monkeypatch.setattr(llm_client, "create_llm_client", lambda: object())

    with pytest.raises(StageOutputValidationError, match="unknown_key_realization_type:KEY"):
        run_pipeline(dataset="vimd", real_llm=True, run_id="t1-badkey")

    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "t1-badkey"
    # provider called exactly once for R2 template (6 total calls)
    assert calls.count("question_template") == 2
    assert len(calls) == 6
    raws = sorted((out_dir / "raw").glob("round_02_template_attempt_*.txt"))
    assert len(raws) == 1  # malformed attempt_02 durable
    raw_text = raws[0].read_text(encoding="utf-8")
    assert "KEY" in raw_text
    # paid count incremented + checkpoint saved
    state = json.loads((out_dir / "run_state.json").read_text(encoding="utf-8"))
    assert state["stage_attempts"]["question_template"] == 2
    assert state["spent_calls"]["question_template"] == 2
    # no successful R2 template artifact
    assert not (out_dir / "rounds" / "round_02" / "02_templates.json").exists()
    # next action still the unfinished template stage
    assert state["next_action"] == "template_round_2"
    # no auto retry
    assert len(calls) == 6
    # approved state unchanged: only R1 pool snapshot exists
    assert (out_dir / "rounds" / "round_01" / "approved_types_after.json").is_file()


# --- TEST 2+3: resume retries same stage with next number, append-only ----
def test_02_03_resume_retries_same_stage_append_only(
        isolated_root, clean_all, monkeypatch):
    script = _run_r1_plus_r2style(isolated_root) + [_bad_template(isolated_root)]

    def fake_bad(client, stage, prompt, response_schema):
        item = script.pop(0)
        llm_client.BUDGET.consume(stage)
        return _ns(item)

    monkeypatch.setattr(llm_client, "call_llm", fake_bad)
    monkeypatch.setattr(llm_client, "create_llm_client", lambda: object())
    with pytest.raises(StageOutputValidationError):
        run_pipeline(dataset="vimd", real_llm=True, run_id="t23-base")

    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "t23-base"
    a01 = out_dir / "raw" / "round_01_template_attempt_01.txt"
    a02 = out_dir / "raw" / "round_02_template_attempt_02.txt"
    assert a01.is_file() and a02.is_file()
    h1 = hashlib.sha256(a01.read_bytes()).hexdigest()
    h2 = hashlib.sha256(a02.read_bytes()).hexdigest()

    _reset_budget()
    reset_attempts()
    good_tpl = _payload(isolated_root, "loop", "round_02", "template")
    rest = [_payload(isolated_root, "loop", "round_02", k)
            for k in ("paraphrase", "quality")] + [
        _payload(isolated_root, "loop", "round_03", k)
        for k in ("style", "template", "paraphrase", "quality")]
    script2 = [good_tpl] + rest
    calls2: list[str] = []

    def fake_good(client, stage, prompt, response_schema):
        item = script2.pop(0)
        llm_client.BUDGET.consume(stage)
        calls2.append(stage)
        return _ns(item)

    monkeypatch.setattr(llm_client, "call_llm", fake_good)
    preview = run_pipeline(dataset="vimd", real_llm=True, resume_run="t23-base")
    # R1 + R2 style NOT called again; first call is R2 template retry
    assert calls2[0] == "question_template"
    assert "question_style" in calls2  # R3 style later, but no R1/R2 style
    assert len([c for c in calls2 if c == "question_style"]) == 1  # only R3
    # new attempt number = previous + 1
    a03 = out_dir / "raw" / "round_02_template_attempt_03.txt"
    assert a03.is_file()
    # append-only: earlier attempts unchanged
    assert hashlib.sha256(a01.read_bytes()).hexdigest() == h1
    assert hashlib.sha256(a02.read_bytes()).hexdigest() == h2
    assert len(preview) == 10


# --- TEST 4: quality partition failure (§31) --------------------------------
def test_04_quality_partition_failure_durable(
        isolated_root, clean_all, monkeypatch):
    style = _payload(isolated_root, "loop", "round_01", "style")
    tpl = _payload(isolated_root, "loop", "round_01", "template")
    para = _payload(isolated_root, "loop", "round_01", "paraphrase")
    bad_q = {"round": 1,
             "accepted_new": [{"type_id": "QS_R1_01",
                               "keep_template_ids": ["T_R1_01_01"]}],
             "duplicates": [],
             "rejected": [{"type_id": "T_R1_01_01_P01", "reason": "x"}]}
    script = [style, tpl, para, bad_q]

    def fake(client, stage, prompt, response_schema):
        item = script.pop(0)
        llm_client.BUDGET.consume(stage)
        return _ns(item)

    monkeypatch.setattr(llm_client, "call_llm", fake)
    monkeypatch.setattr(llm_client, "create_llm_client", lambda: object())
    with pytest.raises(StageOutputValidationError, match="quality_unknown_type_id"):
        run_pipeline(dataset="vimd", real_llm=True, run_id="t4-badq")

    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "t4-badq"
    raws = list((out_dir / "raw").glob("quality_round_1_attempt_*.txt"))
    assert len(raws) == 1
    state = json.loads((out_dir / "run_state.json").read_text(encoding="utf-8"))
    assert state["stage_attempts"]["quality"] == 1
    assert not (out_dir / "rounds" / "round_01" / "04_quality.json").exists()
    assert not (out_dir / "rounds" / "round_01" / "approved_types_after.json").exists()

    # resume retries same quality stage with next attempt number
    _reset_budget()
    reset_attempts()
    good_q = _payload(isolated_root, "loop", "round_01", "quality")
    r2 = [_payload(isolated_root, "loop", "round_02", k)
          for k in ("style", "template", "paraphrase", "quality")]
    r3 = [_payload(isolated_root, "loop", "round_03", k)
          for k in ("style", "template", "paraphrase", "quality")]
    script2 = [good_q] + r2 + r3
    calls2: list[str] = []

    def fake2(client, stage, prompt, response_schema):
        item = script2.pop(0)
        llm_client.BUDGET.consume(stage)
        calls2.append(stage)
        return _ns(item)

    monkeypatch.setattr(llm_client, "call_llm", fake2)
    run_pipeline(dataset="vimd", real_llm=True, resume_run="t4-badq")
    assert calls2[0] == "quality"
    assert (out_dir / "raw" / "quality_round_1_attempt_02.txt").is_file()


# --- TEST 5: JSON parse failure (§32) ---------------------------------------
def test_05_json_parse_failure_durable(isolated_root, clean_all, monkeypatch):
    calls: list[str] = []

    def fake(client, stage, prompt, response_schema):
        llm_client.BUDGET.consume(stage)
        calls.append(stage)
        return _ns("{bad json", finish="STOP")

    monkeypatch.setattr(llm_client, "call_llm", fake)
    monkeypatch.setattr(llm_client, "create_llm_client", lambda: object())
    with pytest.raises(StageOutputValidationError):
        run_pipeline(dataset="vimd", real_llm=True, run_id="t5-badjson")
    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "t5-badjson"
    raw = out_dir / "raw" / "round_01_style_attempt_01.txt"
    assert raw.is_file()
    assert raw.read_text(encoding="utf-8") == "{bad json"
    state = json.loads((out_dir / "run_state.json").read_text(encoding="utf-8"))
    assert state["stage_attempts"]["question_style"] == 1
    assert not (out_dir / "rounds" / "round_01" / "01_question_styles.json").exists()
    assert state["next_action"] == "style_round_1"


# --- TEST 6: Pydantic failure (§33) -----------------------------------------
def test_06_pydantic_failure_durable(isolated_root, clean_all, monkeypatch):
    def fake(client, stage, prompt, response_schema):
        llm_client.BUDGET.consume(stage)
        return _ns(json.dumps({"round": 1}))  # missing required fields

    monkeypatch.setattr(llm_client, "call_llm", fake)
    monkeypatch.setattr(llm_client, "create_llm_client", lambda: object())
    with pytest.raises(Exception):
        run_pipeline(dataset="vimd", real_llm=True, run_id="t6-pyd")
    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "t6-pyd"
    assert (out_dir / "raw" / "round_01_style_attempt_01.txt").is_file()
    state = json.loads((out_dir / "run_state.json").read_text(encoding="utf-8"))
    assert state["stage_attempts"]["question_style"] == 1
    assert not (out_dir / "rounds" / "round_01" / "01_question_styles.json").exists()


# --- helpers to build minimal resumable dirs for reconciliation tests ------
def _min_dir(root, run_id, c_counts, d_lists):
    """Hand-build a 1-round style-complete dir.

    c_counts: checkpoint stage_attempts; d_lists: raw attempt numbers per stage.
    Style artifact always present (round complete through style only).
    """
    from src.autonomous_qa.compiler.pipeline_runner import _ResumedRun  # noqa
    out_dir = root / "outputs" / "runs" / "vimd" / run_id
    raw_dir = out_dir / "raw"
    rdir = out_dir / "rounds" / "round_01"
    rdir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    fx = root / "tests" / "fixtures" / "loop"
    (rdir / "01_question_styles.json").write_bytes(
        (fx / "round_01" / "style.json").read_bytes())
    name_map = {"question_style": "round_01_style",
                "question_template": "round_01_template",
                "paraphrase": "paraphrase_round_1",
                "quality": "quality_round_1"}
    for stage, nums in d_lists.items():
        for n in nums:
            (raw_dir / f"{name_map[stage]}_attempt_{n:02d}.txt").write_text(
                "{}", encoding="utf-8")
    (out_dir / "run_state.json").write_text(json.dumps({
        "run_id": run_id, "dataset": "vimd", "mode": "real",
        "completed": ["style_round_1"], "current_round": 1,
        "next_action": "template_round_1", "low_gain_streak": 0,
        "stopped": False, "stop_reason": None,
        "spent_calls": dict(c_counts), "stage_attempts": dict(c_counts),
        "attempted_calls_total": sum(c_counts.values()),
        "successful_calls_total": 1,
        "last_finish_reason": "STOP", "last_failed_stage": None,
        "loop_config": {"max_rounds": 3, "min_new_type_rate": 0.15,
                        "saturation_patience": 2,
                        "immediate_stop_if_zero_new": True},
        "template_contract_version": TEMPLATE_CONTRACT_VERSION,
        "answer_reference_contract_version": "kind-scoped-v1",
        "answer_shape_contract_version": "canonical-v1",
        "executable_signature_contract_version": "v1",
        "derived_field_contract_version": "data-backed-v1"}),
        encoding="utf-8")
    return out_dir


def _try_resume(root, run_id):
    from src.autonomous_qa.compiler.pipeline_runner import _ResumedRun
    out_dir = root / "outputs" / "runs" / "vimd" / run_id
    run = _ResumedRun(dataset="vimd", out_dir=out_dir,
                      raw_dir=out_dir / "raw",
                      rounds_dir=out_dir / "rounds", readme="",
                      rows=[], client=None)
    run.load_and_validate()
    return run


# --- TEST 7: raw ahead by one (§34) -----------------------------------------
def test_07_raw_ahead_by_one_reconciles(isolated_root, clean_all):
    c = {"question_style": 1, "question_template": 0, "paraphrase": 0, "quality": 0}
    d = {"question_style": [1, 2], "question_template": [],
         "paraphrase": [], "quality": []}
    out_dir = _min_dir(isolated_root, "t7-ahead1", c, d)
    # style artifact for round 1 exists; extra raw style attempt_02 simulates
    # crash after raw write before checkpoint write. Resume must reconcile
    # without any LLM call and without modifying attempt files.
    before = (out_dir / "raw" / "round_01_style_attempt_01.txt").read_bytes()
    run = _try_resume(isolated_root, "t7-ahead1")
    assert run.state["stage_attempts"]["question_style"] == 2
    assert (out_dir / "raw" / "round_01_style_attempt_01.txt").read_bytes() == before
    assert (out_dir / "raw" / "round_01_style_attempt_02.txt").is_file()


# --- TEST 8: checkpoint ahead (§35) -----------------------------------------
def test_08_checkpoint_ahead_refuses(isolated_root, clean_all):
    c = {"question_style": 2, "question_template": 0, "paraphrase": 0, "quality": 0}
    d = {"question_style": [1], "question_template": [],
         "paraphrase": [], "quality": []}
    _min_dir(isolated_root, "t8-ckahead", c, d)
    with pytest.raises(ValueError, match="refusing to resume"):
        _try_resume(isolated_root, "t8-ckahead")


# --- TEST 9: disk ahead by more than one (§36) ------------------------------
def test_09_disk_ahead_by_two_refuses(isolated_root, clean_all):
    c = {"question_style": 1, "question_template": 0, "paraphrase": 0, "quality": 0}
    d = {"question_style": [1, 2, 3], "question_template": [],
         "paraphrase": [], "quality": []}
    _min_dir(isolated_root, "t9-dahead2", c, d)
    with pytest.raises(ValueError, match="refusing to resume"):
        _try_resume(isolated_root, "t9-dahead2")


# --- TEST 10: noncontiguous (§37) -------------------------------------------
def test_10_noncontiguous_refuses(isolated_root, clean_all):
    c = {"question_style": 2, "question_template": 0, "paraphrase": 0, "quality": 0}
    out_dir = _min_dir(isolated_root, "t10-gap",
                       {"question_style": 1, "question_template": 0,
                        "paraphrase": 0, "quality": 0},
                       {"question_style": [1], "question_template": [],
                        "paraphrase": [], "quality": []})
    # manually add attempt_03 -> [1,3] gap
    (out_dir / "raw" / "round_01_style_attempt_03.txt").write_text(
        "{}", encoding="utf-8")
    with pytest.raises(ValueError, match="[Nn]oncontiguous|refusing to resume"):
        _try_resume(isolated_root, "t10-gap")


# --- TEST 11: global budget (§38) -------------------------------------------
def test_11_budget_cap_includes_malformed(isolated_root, clean_all, monkeypatch):
    # Spend the full 16-call allowance (4 per stage), then verify no 17th call.
    for stage in ("question_style", "question_template", "paraphrase", "quality"):
        for _ in range(4):
            llm_client.BUDGET.consume(stage)
    assert llm_client.BUDGET.count == 16
    invoked: list[str] = []

    class _Client:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                invoked.append("x")
                raise AssertionError("must not be invoked")

    from src.autonomous_qa.core.schemas import StyleRoundOutput
    with pytest.raises(RuntimeError, match="[Bb]udget|cap|exceeded"):
        llm_client.complete_structured_stage(
            _Client(), stage="question_style", prompt="p",
            response_schema=StyleRoundOutput,
            raw_path=isolated_root / "x.txt")
    assert invoked == []


# --- TEST 12: MAX_TOKENS (§39) ----------------------------------------------
def test_12_maxtokens_returned_is_paid(isolated_root, clean_all, monkeypatch):
    style = _payload(isolated_root, "loop", "round_01", "style")
    truncated = json.dumps(style, ensure_ascii=False)[:400]

    def fake(client, stage, prompt, response_schema):
        llm_client.BUDGET.consume(stage)
        return SimpleNamespace(text=truncated, parsed=None,
                               candidates=[SimpleNamespace(
                                   finish_reason="MAX_TOKENS")],
                               usage_metadata=None)

    monkeypatch.setattr(llm_client, "call_llm", fake)
    monkeypatch.setattr(llm_client, "create_llm_client", lambda: object())
    with pytest.raises(StageOutputValidationError):
        run_pipeline(dataset="vimd", real_llm=True, run_id="t12-max")
    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "t12-max"
    assert (out_dir / "raw" / "round_01_style_attempt_01.txt").is_file()
    state = json.loads((out_dir / "run_state.json").read_text(encoding="utf-8"))
    assert state["stage_attempts"]["question_style"] == 1
    assert state["last_finish_reason"] == "MAX_TOKENS"


# --- TEST 13: provider throws before response (§40) -------------------------
def test_13_provider_exception_makes_no_file(
        isolated_root, clean_all, monkeypatch):
    def fake(client, stage, prompt, response_schema):
        raise RuntimeError("simulated provider failure")

    monkeypatch.setattr(llm_client, "call_llm", fake)
    monkeypatch.setattr(llm_client, "create_llm_client", lambda: object())
    with pytest.raises(RuntimeError, match="simulated provider failure"):
        run_pipeline(dataset="vimd", real_llm=True, run_id="t13-boom")
    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "t13-boom"
    assert list((out_dir / "raw").glob("*_attempt_*.txt")) == []
    # No response returned => no paid checkpoint is required to exist.
    # The run must not fabricate a raw file.
    state_path = out_dir / "run_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert sum(state.get("stage_attempts", {}).values()) == 0


# --- TEST 49: atomic helper (§49) -------------------------------------------
def test_49_atomic_helper_no_overwrite_and_tmp_ignored(tmp_path):
    final = tmp_path / "question_template_attempt_01.txt"
    atomic_write_new_file(final, "hello")
    assert final.read_text(encoding="utf-8") == "hello"
    with pytest.raises(FileExistsError):
        atomic_write_new_file(final, "other")
    assert final.read_text(encoding="utf-8") == "hello"
    # temp file must not count
    (tmp_path / "question_template_attempt_02.txt.tmp").write_text(
        "tmp", encoding="utf-8")
    scanned = scan_raw_attempts(tmp_path)
    assert scanned.get("question_template") == [1]
    assert require_contiguous([1], "question_template") == 1


# --- TEST 50: inventory ignores non-canonical (§50) -------------------------
def test_50_inventory_ignores_junk(tmp_path):
    (tmp_path / "question_template_attempt_01.txt").write_text("a", encoding="utf-8")
    (tmp_path / "question_template_attempt_02.txt.tmp").write_text("t", encoding="utf-8")
    (tmp_path / "question_template_attempt_02.txt.bak").write_text("b", encoding="utf-8")
    (tmp_path / "02_templates.json").write_text("{}", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    scanned = scan_raw_attempts(tmp_path)
    assert scanned == {"question_template": [1]}
    # shared recorder refuses overwrite
    with pytest.raises(FileExistsError):
        record_paid_response("question_template", 1, "zzz", tmp_path / "question_template.txt")

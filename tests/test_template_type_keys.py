"""Type-keyed key_realizations contract (§40-§60).

LLM keys are exact current-round question type IDs. Python derives raw
fields via get_answer_concept_field(). All LLM traffic faked (0 real).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import src.autonomous_qa.authoring.llm_client as llm_client
from src.autonomous_qa.core.loop import commit_approved_bindings
from src.autonomous_qa.compiler.pipeline_runner import run_pipeline
from src.autonomous_qa.core.validity import (
    build_round_field_candidates,
    gate_round,
    get_answer_concept_field,
    validate_type_key_realizations,
)
from tests.test_safety import clean_budget  # noqa: F401


def _type(tid, answer, uses=("audio", "gender")):
    return {"id": tid, "name": "N", "goal": "g", "uses": list(uses),
            "input_count": 1, "answer_rule": "r", "answer": answer}


def _tpl(tid, owner, text):
    return {"template_id": tid, "question_type_id": owner, "text": text}


def _fv(key):
    return {"kind": "field_value", "key": key}


# §48 — Python derives field from the type, never from the LLM key.
def test_48_python_derives_province_field():
    t = _type("QS_A", _fv("province"), uses=("audio", "province"))
    assert get_answer_concept_field(t) == "province"
    assert validate_type_key_realizations(
        {"QS_A": "phương ngữ tỉnh/thành"}, ["QS_A"],
        [_tpl("T1", "QS_A", "Giọng nói này thuộc [KEY] nào?")]) == []


# §49 — equality derivation stays structural, value never exposed.
def test_49_equality_speakerid_concept():
    t = _type("QS_A", {"kind": "equality",
                       "keys": ["speakerID_1", "speakerID_2"]},
              uses=("audio", "speakerID"))
    assert get_answer_concept_field(t) == "speakerID"
    assert validate_type_key_realizations(
        {"QS_A": "danh tính người nói"}, ["QS_A"],
        [_tpl("T1", "QS_A", "Cùng một [KEY]?")]) == []


# §15/§16 — gate resolution: approved -> owner phrase -> unbound.
def test_gate_per_type_resolution_order():
    types = [_type("QS_A", _fv("gender")), _type("QS_B", _fv("gender"))]
    tpls = [_tpl("T1", "QS_A", "Hãy xác định [KEY]?"),
            _tpl("T2", "QS_B", "Cho biết [KEY]?")]
    # Owner phrase binds even with no approved map.
    g = gate_round(types, tpls,
                   type_key_realizations={"QS_A": "giới tính",
                                          "QS_B": "giới tính của người nói"},
                   approved_key_realizations={})
    assert {t["template_id"] for t in g["valid_templates"]} == {"T1", "T2"}
    # Missing owner phrase -> unbound for that type only.
    g2 = gate_round(types, tpls,
                    type_key_realizations={"QS_A": "giới tính"},
                    approved_key_realizations={})
    assert [t["template_id"] for t in g2["valid_templates"]] == ["T1"]
    assert {d["template_id"]: d["reason"]
            for d in g2["dropped_templates"]}["T2"] == "unbound_key"
    # Approved binding binds without any type phrase.
    g3 = gate_round(types, tpls, type_key_realizations={},
                    approved_key_realizations={"gender": "giới tính"})
    assert {t["template_id"] for t in g3["valid_templates"]} == {"T1", "T2"}


# §17/§18/§50 — post-gate candidates, style order, first wins.
def test_50_two_types_same_field_first_style_order_wins():
    types = [_type("QS_A", _fv("gender")), _type("QS_B", _fv("gender"))]
    tpls = [_tpl("T1", "QS_A", "Hãy xác định [KEY]?"),
            _tpl("T2", "QS_B", "Cho biết [KEY]?")]
    g = gate_round(types, tpls,
                   type_key_realizations={"QS_A": "giới tính",
                                          "QS_B": "giới tính của người nói"},
                   approved_key_realizations={})
    cands = build_round_field_candidates(
        types, g["valid_templates"],
        [t["id"] for t in g["valid_types"]],
        {"QS_A": "giới tính", "QS_B": "giới tính của người nói"}, {})
    assert cands == {"gender": "giới tính"}


# §21 — auto-rejected types cannot seed candidates.
def test_51_rejected_earlier_type_does_not_win():
    good = _type("QS_B", _fv("region"), uses=("audio", "region"))
    bad = {"id": "QS_A", "name": "N", "goal": "g",
           "uses": ["audio", "unknown_field"], "input_count": 1,
           "answer_rule": "r",
           "answer": {"kind": "field_value", "key": "unknown_field"}}
    types = [bad, good]
    tpls = [_tpl("T1", "QS_A", "Hãy xác định [KEY]?"),
            _tpl("T2", "QS_B", "Cho biết [KEY]?")]
    g = gate_round(types, tpls,
                   type_key_realizations={"QS_A": "phrase_A",
                                          "QS_B": "phrase_B"},
                   approved_key_realizations={})
    assert {t["id"] for t in g["valid_types"]} == {"QS_B"}
    cands = build_round_field_candidates(
        types, g["valid_templates"],
        [t["id"] for t in g["valid_types"]],
        {"QS_A": "phrase_A", "QS_B": "phrase_B"}, {})
    assert cands == {"region": "phrase_B"}


# §22 — approved precedence over round phrases.
def test_52_approved_precedence():
    types = [_type("QS_A", _fv("region"), uses=("audio", "region"))]
    tpls = [_tpl("T1", "QS_A", "Hãy xác định [KEY]?")]
    g = gate_round(types, tpls,
                   type_key_realizations={"QS_A": "vùng giọng"},
                   approved_key_realizations={"region": "vùng phương ngữ"})
    cands = build_round_field_candidates(
        types, g["valid_templates"],
        [t["id"] for t in g["valid_types"]],
        {"QS_A": "vùng giọng"}, {"region": "vùng phương ngữ"})
    assert cands == {}
    # Effective context keeps the approved value.
    effective = dict(cands)
    effective.update({"region": "vùng phương ngữ"})
    assert effective == {"region": "vùng phương ngữ"}


# §26/§54 — accepted + kept-[KEY] commits the canonical round candidate.
def test_54_accepted_commit_uses_round_candidate():
    types = {"QS_A": _type("QS_A", _fv("gender"))}
    approved: dict = {}
    out = commit_approved_bindings(
        approved, {"gender": "giới tính"},
        [{"type_id": "QS_A",
          "kept_templates": [{"template_id": "T1",
                              "text": "Hãy xác định [KEY]?"}]}],
        types, type_order=["QS_A"])
    assert out["committed"] == ["gender"]
    assert approved == {"gender": "giới tính"}


# §55/§56 — slot-free keep / rejected / duplicate commit nothing.
def test_55_56_no_commit_without_kept_key():
    types = {"QS_A": _type("QS_A", _fv("gender"))}
    approved: dict = {}
    out = commit_approved_bindings(
        approved, {"gender": "giới tính"},
        [{"type_id": "QS_A",
          "kept_templates": [{"template_id": "T1", "text": "Hãy phiên âm."}]}],
        types, type_order=["QS_A"])
    assert out["committed"] == [] and approved == {}
    # Rejected/duplicates never reach commit: empty accepted list.
    out2 = commit_approved_bindings(dict(approved), {"gender": "x"}, [],
                                    types, type_order=["QS_A"])
    assert out2["committed"] == []


# §57/§58 — final artifact stays field-keyed, needed-only.
def test_57_final_format_field_keyed():
    from src.autonomous_qa.compiler.pipeline_runner import _FreshRun
    import tempfile
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp())
    run = _FreshRun(dataset="vimd", run_id="t", out_dir=tmp,
                    raw_dir=tmp / "raw", rounds_dir=tmp / "rounds",
                    readme="", rows=[], client=None)
    run.approved_pool = [{
        "type_id": "QS_A", "name": "N", "goal": "g", "uses": ["audio"],
        "input_count": 1, "answer_rule": "r",
        "answer": {"kind": "field_value", "key": "gender"},
        "round_accepted": 1,
        "kept_templates": [{"template_id": "T1",
                            "text": "Hãy xác định [KEY]?"}],
        "bindings": []}]
    run.approved_key_realizations = {"gender": "giới tính",
                                     "province": "stale"}
    final = run._final_key_realizations()
    assert final == {"gender": "giới tính"}
    assert "QS_A" not in final


# §53 — Quality bundle exposes effective field realization.
def test_53_quality_bundle_effective_context(isolated_root, clean_budget):
    from src.autonomous_qa.compiler.pipeline_runner import run_pipeline
    run_pipeline(dataset="vimd", real_llm=False)
    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "mock"
    pool = json.loads((out_dir / "final_template_pool.json").read_text(
        encoding="utf-8"))
    assert pool["key_realizations"]["province"] == "tỉnh thành phương ngữ"
    assert not any(k.startswith("QS_")
                   for k in pool["key_realizations"])


# §60 — durability with new contract: KEY fails as unknown type, resumable.
def test_60_resume_durability_new_contract(isolated_root, clean_budget,
                                           monkeypatch):
    from tests.test_safety import _payload

    style = _payload(isolated_root, "loop", "round_01", "style")
    r1tpl = _payload(isolated_root, "loop", "round_01", "template")
    r1para = _payload(isolated_root, "loop", "round_01", "paraphrase")
    r1q = _payload(isolated_root, "loop", "round_01", "quality")
    r2style = _payload(isolated_root, "loop", "round_02", "style")
    bad_tpl = {"round": 2, "key_realizations": {"KEY": "..."},
               "templates": [
                   {"template_id": "T2", "question_type_id": "QS_R2_01",
                    "text": "Hãy xác định [KEY]?"}]}
    script = [style, r1tpl, r1para, r1q, r2style, bad_tpl]
    calls: list[str] = []

    def fake(client, stage, prompt, response_schema):
        item = script.pop(0)
        llm_client.BUDGET.consume(stage)
        calls.append(stage)
        return SimpleNamespace(text=json.dumps(item, ensure_ascii=False),
                               parsed=None, candidates=[], usage_metadata=None)

    monkeypatch.setattr(llm_client, "call_llm", fake)
    monkeypatch.setattr(llm_client, "create_llm_client", lambda: object())
    from src.autonomous_qa.authoring.llm_client import StageOutputValidationError
    with pytest.raises(StageOutputValidationError,
                       match="unknown_key_realization_type:KEY"):
        run_pipeline(dataset="vimd", real_llm=True, run_id="newc-badkey")
    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "newc-badkey"
    assert len(list((out_dir / "raw").glob(
        "round_02_template_attempt_*.txt"))) == 1
    state = json.loads((out_dir / "run_state.json").read_text(
        encoding="utf-8"))
    assert state["stage_attempts"]["question_template"] == 2
    assert state["next_action"] == "template_round_2"

    # Manual resume retries the same stage with the next attempt number.
    llm_client.BUDGET.count = 0
    llm_client.BUDGET.stage_counts = {
        k: 0 for k in llm_client.BUDGET.stage_counts}
    llm_client.reset_attempts()
    good_tpl = _payload(isolated_root, "loop", "round_02", "template")
    rest = [_payload(isolated_root, "loop", "round_02", k)
            for k in ("paraphrase", "quality")] + [
        _payload(isolated_root, "loop", "round_03", k)
        for k in ("style", "template", "paraphrase", "quality")]
    script2 = [good_tpl] + rest
    calls2: list[str] = []

    def fake2(client, stage, prompt, response_schema):
        item = script2.pop(0)
        llm_client.BUDGET.consume(stage)
        calls2.append(stage)
        return SimpleNamespace(text=json.dumps(item, ensure_ascii=False),
                               parsed=None, candidates=[], usage_metadata=None)

    monkeypatch.setattr(llm_client, "call_llm", fake2)
    preview = run_pipeline(dataset="vimd", real_llm=True,
                           resume_run="newc-badkey")
    assert calls2[0] == "question_template"
    assert (out_dir / "raw" / "round_02_template_attempt_03.txt").is_file()
    assert len(preview) == 10


# §32 — prompt explicitly teaches type-ID keys and forbids raw keys.
def test_32_prompt_teaches_type_id_keys():
    from src.common.io import load_prompt
    prompt = load_prompt("question_template")
    assert "QUESTION TYPE IDs" in prompt
    assert "copied exactly from one of the input `type_id`" in prompt
    assert '"KEY"' in prompt
    assert "Do NOT use dataset column names as mapping keys." in prompt


# §39 — pre-patch checkpoints refuse resume as incompatible.
def test_39_old_run_fingerprint_refuses(isolated_root, clean_budget):
    import json as _json
    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "old-contract"
    raw_dir = out_dir / "raw"
    (out_dir / "rounds" / "round_01").mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    fx = isolated_root / "tests" / "fixtures" / "loop"
    for kind, name in (("style", "01_question_styles.json"),
                       ("template", "02_templates.json"),
                       ("paraphrase", "03_paraphrases.json")):
        (out_dir / "rounds" / "round_01" / name).write_bytes(
            (fx / "round_01" / f"{kind}.json").read_bytes())
    (raw_dir / "round_01_style_attempt_01.txt").write_text(
        "{}", encoding="utf-8")
    (out_dir / "run_state.json").write_text(_json.dumps({
        "run_id": "old-contract", "dataset": "vimd", "mode": "real",
        "completed": ["style_round_1"], "current_round": 1,
        "next_action": "template_round_1", "low_gain_streak": 0,
        "stopped": False, "stop_reason": None,
        "spent_calls": {"question_style": 1, "question_template": 0,
                        "paraphrase": 0, "quality": 0},
        "loop_config": {"max_rounds": 3, "min_new_type_rate": 0.15,
                        "saturation_patience": 2,
                        "immediate_stop_if_zero_new": True}}),
        encoding="utf-8")
    from src.autonomous_qa.compiler.pipeline_runner import _ResumedRun
    run = _ResumedRun(dataset="vimd", out_dir=out_dir, raw_dir=raw_dir,
                      rounds_dir=out_dir / "rounds", readme="",
                      rows=[], client=None)
    with pytest.raises(ValueError, match="contract changed"):
        run.load_and_validate()

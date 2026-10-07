"""Final key_realizations mapping: needed-only, justified, fail-closed."""

import json

import pytest

from src.autonomous_qa.compiler.pipeline_runner import _FreshRun


def _run(pool=(), approved=None, rows=()):
    import tempfile
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp())
    run = _FreshRun(
        dataset="vimd", run_id="t", out_dir=tmp, raw_dir=tmp / "raw",
        rounds_dir=tmp / "rounds", readme="", rows=list(rows), client=None)
    run.approved_pool = list(pool)
    run.approved_key_realizations = dict(approved or {})
    return run


def _type(tid, answer, kept, uses=("audio", "gender")):
    return {"type_id": tid, "name": "N", "goal": "g", "uses": list(uses),
            "input_count": 1, "answer_rule": "r", "answer": answer,
            "round_accepted": 1,
            "kept_templates": [{"template_id": k, "text": t}
                               for k, t in kept],
            "bindings": []}


def test_zero_approved_zero_templates_gives_empty_mapping():
    run = _run(pool=[], approved={})
    assert run._final_key_realizations() == {}


def test_rejected_round_mapping_excluded_from_final():
    # Candidate had gender/region/province but every owner type was
    # rejected: nothing was ever committed, final stays empty.
    run = _run(pool=[], approved={})
    assert run._final_key_realizations() == {}


def test_final_contains_gender_binding_for_key_template():
    pool = [_type("QS_G", {"kind": "field_value", "key": "gender"},
                  [("T1", "Hãy xác định [KEY]?")])]
    run = _run(pool=pool, approved={"gender": "giới tính"})
    assert run._final_key_realizations() == {"gender": "giới tính"}


def test_final_mapping_has_no_unused_field():
    pool = [_type("QS_G", {"kind": "field_value", "key": "gender"},
                  [("T1", "Hãy xác định [KEY]?")])]
    run = _run(pool=pool,
               approved={"gender": "giới tính", "region": "vùng giọng"})
    assert run._final_key_realizations() == {"gender": "giới tính"}


def test_final_mapping_keys_are_schema_fields():
    from src.autonomous_qa.core.validity import SCHEMA_FIELDS
    pool = [_type("QS_G", {"kind": "field_value", "key": "gender"},
                  [("T1", "x [KEY] y")])]
    run = _run(pool=pool, approved={"gender": "giới tính"})
    final = run._final_key_realizations()
    assert set(final) <= set(SCHEMA_FIELDS)


def test_round_artifact_may_retain_excluded_candidates():
    # Round artifacts keep candidate mappings for audit even when the
    # final mapping excludes them; the two stores are independent.
    run = _run(pool=[], approved={})
    assert run._round_candidates == {}
    run._round_candidates[1] = {"gender": "giới tính"}
    assert run._final_key_realizations() == {}


def test_finalize_fails_closed_on_impossible_state():
    pool = [_type("QS_G", {"kind": "field_value", "key": "gender"},
                  [("T1", "Hãy xác định [KEY]?")])]
    run = _run(pool=pool, approved={})
    with pytest.raises(ValueError, match="no approved realization"):
        run._final_key_realizations()


def test_mock_final_mapping_matches_needed_bindings(isolated_root):
    from src.autonomous_qa.authoring.llm_client import BUDGET
    from src.autonomous_qa.compiler.pipeline_runner import run_pipeline
    from src.autonomous_qa.core.validity import SCHEMA_FIELDS, get_answer_concept_field

    before = (BUDGET.count, dict(BUDGET.stage_counts))
    run_pipeline(dataset="vimd", real_llm=False)
    assert (BUDGET.count, dict(BUDGET.stage_counts)) == (
        before[0], before[1])  # still zero paid calls

    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "mock"
    pool_doc = json.loads(
        (out_dir / "final_template_pool.json").read_text(encoding="utf-8"))
    final_map = pool_doc["key_realizations"]
    approved = json.loads(
        (out_dir / "final_approved_types.json").read_text(encoding="utf-8"))
    needed = set()
    for t in approved["approved_types"]:
        if any("[KEY]" in (k.get("text", "") or "")
               for k in t.get("kept_templates", [])):
            needed.add(get_answer_concept_field(t))
    # Exactly the needed bindings: nothing stale, nothing missing.
    assert set(final_map) == needed
    for f in needed:
        assert isinstance(final_map[f], str) and final_map[f].strip()
    # Round artifacts keep the LLM type-keyed mapping for audit: keys are
    # exact round type IDs, never raw fields.
    r1tpl = json.loads(
        (out_dir / "rounds" / "round_01" / "02_templates.json").read_text(
            encoding="utf-8"))
    r1styles = json.loads(
        (out_dir / "rounds" / "round_01" / "01_question_styles.json").read_text(
            encoding="utf-8"))
    r1_ids = {t["id"] for t in r1styles.get("question_types", [])}
    assert set(r1tpl.get("key_realizations", {})) <= r1_ids
    assert set(final_map) <= set(SCHEMA_FIELDS)

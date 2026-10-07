"""Answer-reference contract: kind-scoped field grammar (§41-§57).

field_value / derived_field: EXACT raw fields only.
equality: canonical [F_1, F_2] pair references only.
Gate and renderer implement the same grammar (parity).
All data hand-built; zero LLM calls.
"""

from __future__ import annotations

import pytest

from src.autonomous_qa.production.render_preview import _gold_answer, render_preview
from src.autonomous_qa.compiler.pipeline_runner import ANSWER_REFERENCE_CONTRACT_VERSION
from src.autonomous_qa.core.validity import (
    SCHEMA_FIELDS,
    gate_round,
    get_answer_concept_field,
    parse_pair_field_reference,
    validate_answer_references,
    validate_question_type,
)

S = {"audio", "text", "gender", "region", "province", "speakerID"}


def _t(**over):
    base = {"id": "QX", "name": "N", "goal": "g",
            "uses": ["audio", "gender"], "input_count": 1,
            "answer_rule": "r",
            "answer": {"kind": "field_value", "key": "gender"}}
    base.update(over)
    return base


# §41 — field_value exact raw field.
def test_41_field_value_exact():
    assert validate_answer_references(
        {"kind": "field_value", "key": "province"}, S) == []
    assert validate_answer_references(
        {"kind": "field_value", "key": "province_1"}, S) == [
        "unknown_answer_field:province_1"]
    assert validate_answer_references(
        {"kind": "field_value", "key": "gender_2"}, S) == [
        "unknown_answer_field:gender_2"]


# §62 — literal _1 field wins by exact membership.
def test_62_literal_underscore_field_valid():
    schema = set(S) | {"feature_1"}
    assert validate_answer_references(
        {"kind": "field_value", "key": "feature_1"}, schema) == []


# §42 — derived exact raw fields.
def test_42_derived_exact():
    good = {"kind": "derived_field", "source_key": "province",
            "target_key": "region"}
    assert validate_answer_references(good, S) == []
    assert validate_answer_references(
        {"kind": "derived_field", "source_key": "province_1",
         "target_key": "region"}, S) == ["unknown_answer_field:province_1"]
    assert validate_answer_references(
        {"kind": "derived_field", "source_key": "province",
         "target_key": "region_1"}, S) == ["unknown_answer_field:region_1"]
    assert validate_answer_references(
        {"kind": "derived_field", "source_key": "province_1",
         "target_key": "region_1"}, S) == [
        "unknown_answer_field:province_1",
        "unknown_answer_field:region_1"]


# §43 — real crash regression is gate-rejected.
def test_43_real_regression_rejected_at_gate():
    bad = _t(id="QS_BAD", uses=["audio", "province", "region"],
             answer_rule="province of audio 1 maps to region",
             answer={"kind": "derived_field", "source_key": "province_1",
                     "target_key": "region"})
    errs = validate_question_type(bad)
    assert "unknown_answer_field:province_1" in errs
    gate = gate_round(
        [bad], [{"template_id": "T1", "question_type_id": "QS_BAD",
                 "text": "Vùng giọng chung là gì?"}],
        type_key_realizations={}, approved_key_realizations={})
    assert [t["id"] for t in gate["valid_types"]] == []
    assert any(a["type_id"] == "QS_BAD" for a in gate["auto_rejected"])


# §44/§63 — equality valid + parser behavior.
def test_44_equality_valid_and_parser():
    assert validate_answer_references(
        {"kind": "equality", "keys": ["province_1", "province_2"]}, S) == []
    assert (get_answer_concept_field(_t(
        answer={"kind": "equality",
                "keys": ["province_1", "province_2"]})) == "province")
    assert parse_pair_field_reference("province_1", S) == ("province", 1)
    assert parse_pair_field_reference("province_2", S) == ("province", 2)
    assert parse_pair_field_reference("province", S) is None
    assert parse_pair_field_reference("fake_1", S) is None
    assert parse_pair_field_reference("province_3", S) is None
    # §63: one suffix only; literal feature_1 base works.
    assert parse_pair_field_reference(
        "feature_1_1", set(S) | {"feature_1"}) == ("feature_1", 1)


# §45/§46/§47/§48/§49 — equality rejections.
def test_45_49_equality_rejections():
    v = validate_answer_references
    assert v({"kind": "equality",
              "keys": ["province_2", "province_1"]}, S) == [
        "invalid_equality_instances"]
    assert v({"kind": "equality",
              "keys": ["province_1", "province_1"]}, S) == [
        "invalid_equality_instances"]
    assert v({"kind": "equality",
              "keys": ["province_1", "region_2"]}, S) == [
        "equality_field_mismatch"]
    assert v({"kind": "equality",
              "keys": ["province", "province"]}, S) == [
        "invalid_equality_instances"]
    assert v({"kind": "equality",
              "keys": ["fake_1", "fake_2"]}, S) == [
        "unknown_answer_field:fake_1", "unknown_answer_field:fake_2"]
    assert v({"kind": "equality", "keys": ["province_1"]}, S) == [
        "invalid_equality_arity"]


# §50 — concept field semantics.
def test_50_concept_field_exact():
    c = get_answer_concept_field
    assert c(_t(answer={"kind": "field_value",
                       "key": "province"})) == "province"
    assert c(_t(answer={"kind": "field_value",
                       "key": "province_1"})) == "province_1"
    assert c(_t(answer={"kind": "derived_field",
                       "source_key": "province",
                       "target_key": "region"})) == "region"
    assert c(_t(answer={"kind": "derived_field",
                       "source_key": "province_1",
                       "target_key": "region_1"})) == "region_1"
    assert c(_t(answer={"kind": "equality",
                       "keys": ["province_1",
                                "province_2"]})) == "province"
    assert c(_t(answer={"kind": "equality",
                       "keys": ["province_1",
                                "region_2"]})) is None
    assert c(_t(answer={"kind": "transcription"})) is None


# §51 — renderer field_value exact + defensive error.
def test_51_render_field_value():
    q = {"type_id": "Q",
         "answer": {"kind": "field_value", "key": "province"}}
    gold, _ = _gold_answer(q, {"province": "Hà Nội"}, None, [])
    assert gold == "Hà Nội"
    bad = {"type_id": "Q",
           "answer": {"kind": "field_value", "key": "province_1"}}
    with pytest.raises(ValueError, match="render_missing_answer_field"):
        _gold_answer(bad, {"province": "Hà Nội"}, None, [])
    try:
        _gold_answer(bad, {"province": "Hà Nội"}, None, [])
    except ValueError as exc:
        assert "KeyError" not in type(exc).__name__
        assert "province_1" in str(exc)


# §52 — renderer derived exact + defensive error.
def test_52_render_derived():
    rows = [{"province": "A", "region": "X"},
            {"province": "B", "region": "Y"}]
    q = {"type_id": "Q",
         "answer": {"kind": "derived_field", "source_key": "province",
                    "target_key": "region"}}
    gold, _ = _gold_answer(q, rows[0], None, rows)
    assert gold == "X"
    bad = {"type_id": "Q",
           "answer": {"kind": "derived_field", "source_key": "province_1",
                      "target_key": "region"}}
    with pytest.raises(ValueError, match="render_missing_answer_field"):
        _gold_answer(bad, rows[0], None, rows)


# §53 — renderer equality reads row1[F]/row2[F].
def test_53_render_equality():
    q = {"type_id": "Q",
         "answer": {"kind": "equality",
                    "keys": ["province_1", "province_2"]}}
    r1 = {"province": "A", "audio": "a.wav"}
    r2 = {"province": "A", "audio": "b.wav"}
    r3 = {"province": "B", "audio": "c.wav"}
    gold, src = _gold_answer(q, r1, r2, [r1, r2, r3])
    assert gold == "Có" and src["field"] == "province"
    gold2, _ = _gold_answer(q, r1, r3, [r1, r2, r3])
    assert gold2 == "Không"


# §54 — parity: every valid form validates, gates, and renders.
def test_54_gate_render_parity_valid():
    rows = [
        {"audio": "a1.wav", "province": "A", "region": "X",
         "gender": "g", "speakerID": "s", "text": "t"},
        {"audio": "a2.wav", "province": "A", "region": "X",
         "gender": "g", "speakerID": "s", "text": "t"},
        {"audio": "b1.wav", "province": "B", "region": "X",
         "gender": "g", "speakerID": "s", "text": "t"},
        {"audio": "b2.wav", "province": "B", "region": "X",
         "gender": "g", "speakerID": "s", "text": "t"},
        {"audio": "c1.wav", "province": "C", "region": "Y",
         "gender": "g", "speakerID": "s", "text": "t"},
        {"audio": "c2.wav", "province": "C", "region": "Y",
         "gender": "g", "speakerID": "s", "text": "t"},
    ]
    forms = [
        _t(id="Q1", uses=["audio", "province"],
           answer={"kind": "field_value", "key": "province"}),
        _t(id="Q2", uses=["audio", "province", "region"],
           answer={"kind": "derived_field", "source_key": "province",
                   "target_key": "region"}),
        _t(id="Q3", uses=["audio", "province"], input_count=2,
           answer={"kind": "equality",
                   "keys": ["province_1", "province_2"]}),
    ]
    for q in forms:
        assert validate_answer_references(
            q["answer"], SCHEMA_FIELDS) == [], q["id"]
        assert validate_question_type(q, dataset_rows=rows) == [], q["id"]
    # field_value + derived render on a single row.
    for q in forms[:2]:
        gold, _ = _gold_answer(q, rows[0], None, rows)
        assert isinstance(gold, str) and gold
    # equality renders on a pair.
    gold, _ = _gold_answer(forms[2], rows[0], rows[0], rows)
    assert gold in ("Có", "Không")


# §55 — invalid parity: gate rejects all malformed forms.
def test_55_gate_rejects_all_malformed():
    cases = [
        {"kind": "field_value", "key": "province_1"},
        {"kind": "derived_field", "source_key": "province_1",
         "target_key": "region"},
        {"kind": "derived_field", "source_key": "province",
         "target_key": "region_1"},
        {"kind": "equality", "keys": ["province_1", "region_2"]},
    ]
    for i, ans in enumerate(cases):
        q = _t(id=f"QB{i}", answer=ans)
        assert validate_question_type(q), ans
        gate = gate_round(
            [q], [{"template_id": f"T{i}", "question_type_id": f"QB{i}",
                   "text": "Text?"}],
            type_key_realizations={}, approved_key_realizations={})
        assert gate["valid_types"] == [], ans


# §56 — finalization preflight fails before render.
def test_56_finalization_preflight():
    import tempfile
    from pathlib import Path
    from src.autonomous_qa.compiler.pipeline_runner import _FreshRun
    tmp = Path(tempfile.mkdtemp())
    run = _FreshRun(dataset="vimd", run_id="t", out_dir=tmp,
                    raw_dir=tmp / "raw", rounds_dir=tmp / "rounds",
                    readme="", rows=[], client=None)
    run.approved_pool = [{
        "type_id": "QS_BAD", "name": "N", "goal": "g",
        "uses": ["audio", "province", "region"], "input_count": 1,
        "answer_rule": "r",
        "answer": {"kind": "derived_field", "source_key": "province_1",
                   "target_key": "region"},
        "round_accepted": 1,
        "kept_templates": [{"template_id": "T1", "text": "Text?"}],
        "bindings": []}]
    run.approved_key_realizations = {"region": "vùng giọng"}
    with pytest.raises(ValueError, match="final_invalid_answer_reference"):
        run._finalize()
    assert not (tmp / "preview.jsonl").exists()


# §57 — version stored and enforced.
def test_57_contract_version():
    assert ANSWER_REFERENCE_CONTRACT_VERSION == "kind-scoped-v1"


# §65 — mock carries a malformed derived auto-reject; Quality unaffected.
def test_65_mock_malformed_derived_auto_rejected(isolated_root):
    from src.autonomous_qa.compiler.pipeline_runner import run_pipeline
    from src.autonomous_qa.core.validity import validate_question_type
    from src.common.io import load_fixture
    style = load_fixture("loop/round_01/style")
    by_id = {t["id"]: t for t in style["question_types"]}
    errs = validate_question_type(by_id["QS_R1_06"])
    assert "unknown_answer_field:province_1" in errs
    run_pipeline(dataset="vimd", real_llm=False)
    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "mock"
    import json as _json
    record = _json.loads(
        (out_dir / "rounds" / "round_01" / "04_quality.json").read_text(
            encoding="utf-8"))
    assert "QS_R1_06" not in [
        a["type_id"] for a in record["accepted_new"]]

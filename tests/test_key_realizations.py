"""key_realizations: QUESTION TYPE ID -> phrase (§2, §7-§11, §30).

The LLM NEVER outputs raw dataset field names. Keys are exact
current-round type IDs; Python derives raw fields via
get_answer_concept_field().
"""

import pytest

from src.autonomous_qa.authoring.llm_client import StageOutputValidationError
from src.autonomous_qa.core.validity import (
    SCHEMA_FIELDS,
    key_required_type_ids,
    validate_key_realizations,
    validate_type_key_realizations,
)
from tests.test_safety import ScriptedLLM, _reset_budget, clean_budget  # noqa: F401

import src.autonomous_qa.authoring.llm_client as llm_client
from src.autonomous_qa.compiler.pipeline_runner import run_pipeline


def _fv(tid, key="gender", uses=("audio", "gender")):
    return {"id": tid, "name": "N", "goal": "g", "uses": list(uses),
            "input_count": 1, "answer_rule": "r",
            "answer": {"kind": "field_value", "key": key}}


def _tpl(tid, owner, text):
    return {"template_id": tid, "question_type_id": owner, "text": text}


def test_schema_fixture_shape():
    assert SCHEMA_FIELDS == {
        "audio", "text", "gender", "region", "province", "speakerID"}


def test_basic_type_keyed_mapping_passes():
    types = [_fv("QS_A")]
    tpls = [_tpl("T1", "QS_A", "Hãy xác định [KEY] của người nói.")]
    assert validate_type_key_realizations(
        {"QS_A": "giới tính"}, ["QS_A"], tpls) == []


def test_old_raw_field_key_rejected():
    # "gender" is a raw field, not a current type ID -> unknown type.
    types = [_fv("QS_A")]
    tpls = [_tpl("T1", "QS_A", "Hãy xác định [KEY] của người nói.")]
    assert validate_type_key_realizations(
        {"gender": "giới tính"}, ["QS_A"], tpls) == [
        "unknown_key_realization_type:gender",
        "missing_key_realization_for_type:QS_A"]


def test_historical_key_failure():
    types = [_fv("QS_A")]
    tpls = [_tpl("T1", "QS_A", "Hãy xác định [KEY] của người nói.")]
    assert validate_type_key_realizations(
        {"KEY": "giới tính"}, ["QS_A"], tpls) == [
        "unknown_key_realization_type:KEY",
        "missing_key_realization_for_type:QS_A"]


def test_historical_natural_phrase_key_failure():
    key = "vùng phương ngữ của phương ngữ cấp tỉnh"
    assert validate_type_key_realizations(
        {key: "x"}, ["QS_A"],
        [_tpl("T1", "QS_A", "Hãy xác định [KEY]?")]) == [
        f"unknown_key_realization_type:{key}",
        "missing_key_realization_for_type:QS_A"]


def test_missing_for_key_template_fails():
    assert validate_type_key_realizations(
        {}, ["QS_A"],
        [_tpl("T1", "QS_A", "Hãy xác định [KEY] của người nói.")]) == [
        "missing_key_realization_for_type:QS_A"]


def test_slot_free_type_needs_no_mapping():
    assert validate_type_key_realizations(
        {}, ["QS_A"],
        [_tpl("T1", "QS_A", "Hãy phiên âm đoạn âm thanh.")]) == []


def test_known_but_unused_extra_ignored():
    # QS_B is known but slot-free-only: extra phrase is harmless.
    types = [_fv("QS_A"), _fv("QS_B")]
    tpls = [_tpl("T1", "QS_A", "Hãy xác định [KEY]?"),
            _tpl("T2", "QS_B", "Hãy phiên âm.")]
    assert validate_type_key_realizations(
        {"QS_A": "giới tính", "QS_B": "nội dung"},
        ["QS_A", "QS_B"], tpls) == []


def test_empty_type_phrase_fails():
    assert validate_type_key_realizations(
        {"QS_A": ""}, ["QS_A"],
        [_tpl("T1", "QS_A", "Hãy xác định [KEY]?")]) == [
        "empty_key_realization:QS_A"]
    assert validate_type_key_realizations(
        {"QS_A": "   "}, ["QS_A"],
        [_tpl("T1", "QS_A", "Hãy xác định [KEY]?")]) == [
        "empty_key_realization:QS_A"]


def test_key_required_ids_from_templates():
    tpls = [_tpl("T1", "QS_A", "Hãy xác định [KEY]?"),
            _tpl("T2", "QS_B", "Hãy phiên âm.")]
    assert key_required_type_ids(tpls) == {"QS_A"}


def test_old_validator_alias_requires_type_ids():
    with pytest.raises(TypeError):
        validate_key_realizations({"gender": "giới tính"})
    assert validate_key_realizations(
        {"QS_A": "giới tính"}, ["QS_A"],
        [_tpl("T1", "QS_A", "Hãy xác định [KEY]?")]) == []


def _boom(*a, **k):
    raise AssertionError("no LLM request may be issued")


def test_template_stage_rejects_malformed_mapping_no_fuzzy_repair(
        isolated_root, clean_budget, monkeypatch):
    from tests.test_safety import _payload

    style = _payload(isolated_root, "loop", "round_01", "style")
    bad_template = {
        "round": 1,
        "key_realizations": {"giới tính": "giới tính"},
        "templates": [],
    }
    calls = []

    def fake(client, stage, prompt, response_schema):
        from types import SimpleNamespace
        import json as _json
        item = [style, bad_template][len(calls)]
        llm_client.BUDGET.consume(stage)
        calls.append(stage)
        return SimpleNamespace(text=_json.dumps(item), parsed=None,
                               candidates=[], usage_metadata=None)

    monkeypatch.setattr(llm_client, "call_llm", fake)
    monkeypatch.setattr(llm_client, "create_llm_client", lambda: object())
    with pytest.raises(StageOutputValidationError, match="malformed"):
        run_pipeline(dataset="vimd", real_llm=True, run_id="bad-kr")
    # Failed at template validation: exactly 2 paid attempts, no guessing.
    assert calls == ["question_style", "question_template"]
    assert llm_client.ATTEMPTS.attempted_calls_total == 2

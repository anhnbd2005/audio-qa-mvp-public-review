"""Fix 1 tests: [VALUE] is never part of the question contract (1-14)."""

import pytest

from src.autonomous_qa.production.render_preview import render_question
from src.common.io import load_fixture, load_prompt
from src.autonomous_qa.core.validity import validate_template_text


def _regression():
    return load_fixture("regression_surface")


def test_template_prompt_forbids_value(isolated_root):
    text = load_prompt("question_template")
    assert "[VALUE] is FORBIDDEN in question text" in text
    assert "Gold values are resolved later" in text


def test_paraphrase_prompt_forbids_value(isolated_root):
    text = load_prompt("paraphrase")
    assert "Never introduce [VALUE]" in text
    assert "not a question+answer string" in text


def test_base_value_template_dropped_as_answer_leakage(isolated_root):
    for tpl in _regression()["leak_base_templates"]:
        assert validate_template_text(tpl["text"]) == ["answer_leakage"], tpl


def test_paraphrase_value_template_dropped_as_answer_leakage(isolated_root):
    for p in _regression()["leak_paraphrases"]:
        assert validate_template_text(p["text"]) == ["answer_leakage"], p


def test_key_remains_supported():
    assert validate_template_text("Hãy xác định [KEY] của người nói.") == []


def test_slot_free_template_remains_supported():
    assert validate_template_text(
        "Hai đoạn âm thanh có phải do cùng một người nói không?") == []


def test_key_source_remains_unknown_placeholder():
    assert validate_template_text("Hãy cho biết [KEY_SOURCE]?") == [
        "unknown_placeholder:KEY_SOURCE"]


def test_type_survives_partial_value_leak():
    from src.autonomous_qa.core.validity import gate_round
    gate = gate_round(
        [{"id": "QS_X", "name": "X", "goal": "g", "uses": ["audio", "gender"],
          "input_count": 1, "answer_rule": "r",
          "answer": {"kind": "field_value", "key": "gender"}}],
        [{"template_id": "GOOD", "question_type_id": "QS_X",
          "text": "Hãy xác định [KEY]?"},
         {"template_id": "LEAK", "question_type_id": "QS_X",
          "text": "Giới tính là [VALUE]?"}])
    assert [t["id"] for t in gate["valid_types"]] == ["QS_X"]
    assert [t["template_id"] for t in gate["valid_templates"]] == ["GOOD"]
    assert gate["dropped_templates"] == [
        {"template_id": "LEAK", "type_id": "QS_X",
         "reason": "answer_leakage"}]


def test_type_without_usable_template_rejected():
    from src.autonomous_qa.core.validity import gate_round
    gate = gate_round(
        [{"id": "QS_X", "name": "X", "goal": "g", "uses": ["audio", "gender"],
          "input_count": 1, "answer_rule": "r",
          "answer": {"kind": "field_value", "key": "gender"}}],
        [{"template_id": "LEAK", "question_type_id": "QS_X",
          "text": "[KEY] là [VALUE]?"}])
    assert gate["valid_types"] == []
    assert gate["auto_rejected"] == [
        {"type_id": "QS_X", "reason": "no_renderable_template"}]


def test_renderer_source_has_no_value_substitution():
    import inspect
    import src.autonomous_qa.production.render_preview as rp
    src_text = inspect.getsource(rp.render_question)
    assert ".replace(\"[VALUE]\"" not in src_text
    assert ".replace('[VALUE]'" not in src_text


def test_rendered_question_never_contains_gold():
    q, _ = render_question(
        "Hãy xác định [KEY] của người nói?", "T1", "gender",
        {"gender": "giới tính"}, "female")
    assert q == "Hãy xác định giới tính của người nói?"
    assert "female" not in q


def test_gold_still_resolved_from_field_value():
    from src.autonomous_qa.production.render_preview import _gold_answer
    qtype = {"type_id": "Q", "answer": {"kind": "field_value",
                                       "key": "gender"}}
    row = {"gender": "male", "audio": "a.wav"}
    gold, source = _gold_answer(qtype, row, None, [row])
    assert gold == "male"
    assert source == {"kind": "field_value", "field": "gender"}


def test_gold_still_resolved_from_equality():
    from src.autonomous_qa.production.render_preview import _gold_answer
    qtype = {"type_id": "Q",
             "answer": {"kind": "equality",
                        "keys": ["speakerID_1", "speakerID_2"]}}
    r1 = {"speakerID": "s1", "audio": "a.wav"}
    r2 = {"speakerID": "s1", "audio": "b.wav"}
    r3 = {"speakerID": "s2", "audio": "c.wav"}
    gold, _ = _gold_answer(qtype, r1, r2, [r1, r2, r3])
    assert gold == "Có"
    gold, _ = _gold_answer(qtype, r1, r3, [r1, r2, r3])
    assert gold == "Không"


def test_unresolved_value_reaching_preview_is_hard_failure():
    with pytest.raises(ValueError, match="\\[VALUE\\]"):
        render_question(
            "Nội dung là [VALUE].", "T9", None, {}, "anything")

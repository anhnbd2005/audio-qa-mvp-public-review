"""get_answer_concept_field + [KEY] binding rules (§45-46)."""

from src.autonomous_qa.core.validity import binding_key_for, gate_round, get_answer_concept_field


def _type(answer, uses=("audio", "gender")):
    return {"id": "QX", "name": "N", "goal": "g", "uses": list(uses),
            "input_count": 1, "answer_rule": "r", "answer": answer}


def test_field_value_gender_concept():
    assert get_answer_concept_field(
        _type({"kind": "field_value", "key": "gender"})) == "gender"


def test_field_value_region_concept():
    assert get_answer_concept_field(
        _type({"kind": "field_value", "key": "region"},
              uses=("audio", "region"))) == "region"


def test_field_value_province_concept():
    assert get_answer_concept_field(
        _type({"kind": "field_value", "key": "province"},
              uses=("audio", "province"))) == "province"


def test_field_value_text_concept():
    assert get_answer_concept_field(
        _type({"kind": "field_value", "key": "text"},
              uses=("audio", "text"))) == "text"


def test_equality_gender_concept():
    assert get_answer_concept_field(_type(
        {"kind": "equality", "keys": ["gender_1", "gender_2"]})) == "gender"


def test_equality_region_concept():
    assert get_answer_concept_field(_type(
        {"kind": "equality", "keys": ["region_1", "region_2"]},
        uses=("audio", "region"))) == "region"


def test_equality_speakerid_concept_stays_structured():
    assert get_answer_concept_field(_type(
        {"kind": "equality", "keys": ["speakerID_1", "speakerID_2"]},
        uses=("audio", "speakerID"))) == "speakerID"


def test_derived_concept_is_target():
    assert get_answer_concept_field(_type(
        {"kind": "derived_field", "source_key": "province",
         "target_key": "region"},
        uses=("audio", "province", "region"))) == "region"


def test_key_template_survives_with_candidate():
    gate = gate_round(
        [_type({"kind": "field_value", "key": "gender"})],
        [{"template_id": "T1", "question_type_id": "QX",
          "text": "Hãy xác định [KEY] của người nói."}],
        key_context={"gender": "giới tính"})
    assert [t["template_id"] for t in gate["valid_templates"]] == ["T1"]
    assert gate["valid_types"]


def test_key_template_dropped_without_any_binding():
    gate = gate_round(
        [_type({"kind": "field_value", "key": "gender"})],
        [{"template_id": "T1", "question_type_id": "QX",
          "text": "Hãy xác định [KEY] của người nói."}],
        key_context={})
    assert gate["valid_templates"] == []
    assert gate["auto_rejected"] == [
        {"type_id": "QX", "reason": "no_renderable_template"}]
    assert gate["dropped_templates"] == [
        {"template_id": "T1", "type_id": "QX", "reason": "unbound_key"}]


def test_slot_free_template_needs_no_mapping():
    gate = gate_round(
        [_type({"kind": "field_value", "key": "gender"})],
        [{"template_id": "T1", "question_type_id": "QX",
          "text": "Giới tính của người nói là gì?"}],
        key_context={})
    assert [t["template_id"] for t in gate["valid_templates"]] == ["T1"]


def test_speakerid_equality_key_template_binds_concept_not_value():
    # A generic [KEY] template on a speakerID equality task binds the
    # *concept* speakerID for rendering; the VALUE is never exposed.
    t = _type({"kind": "equality", "keys": ["speakerID_1", "speakerID_2"]},
              uses=("audio", "speakerID"))
    assert get_answer_concept_field(t) == "speakerID"
    assert binding_key_for(t) is None
    gate = gate_round(
        [t],
        [{"template_id": "T1", "question_type_id": "QX",
          "text": "Hai đoạn âm thanh có phải do cùng một [KEY] không?"}],
        key_context={"speakerID": "same speaker/person"})
    assert [x["template_id"] for x in gate["valid_templates"]] == ["T1"]

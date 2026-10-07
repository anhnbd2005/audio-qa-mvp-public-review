"""Hard validity gate: deterministic checks, zero LLM (pure Python).

Core invariant: answer_rule is opaque prose. ONLY non-emptiness is
checked. All machine validation reads the structured `answer` object.
"""

from src.autonomous_qa.core.validity import (
    SCHEMA_FIELDS,
    binding_key_for,
    gate_round,
    normalize_field_reference,
    validate_question_type,
    validate_template_text,
)


def _type(**over):
    base = {
        "id": "QS_R1_01",
        "name": "Speaker Gender Recognition",
        "goal": "Identify the speaker's gender from the audio.",
        "uses": ["audio", "gender"],
        "input_count": 1,
        "answer_rule": "the gender label of the speaker",
        "answer": {"kind": "field_value", "key": "gender"},
    }
    base.update(over)
    return base


def test_answer_rule_prose_is_never_parsed_for_fields():
    # Words like "ground", "associated", "Equality", "given" inside
    # answer_rule must not create unknown fields.
    rules = [
        "The ground-truth transcription of the audio.",
        "The gender label associated with the speaker.",
        "Equality of speakerID values for the two recordings.",
        "Whatever prose a human writes here, as long as it is non-empty.",
    ]
    for rule in rules:
        assert validate_question_type(_type(answer_rule=rule)) == [], rule


def test_empty_answer_rule_rejected():
    assert validate_question_type(_type(answer_rule="")) == [
        "empty_answer_rule"]
    assert validate_question_type(_type(answer_rule="   ")) == [
        "empty_answer_rule"]


def test_field_value_kinds_pass():
    for key, uses in (("gender", ["audio", "gender"]),
                      ("region", ["audio", "region"]),
                      ("province", ["audio", "province"]),
                      ("text", ["audio", "text"])):
        assert validate_question_type(_type(
            uses=uses, answer_rule="some human prose here",
            answer={"kind": "field_value", "key": key})) == [], key


def test_field_value_speaker_id_rejected_as_id_exposure():
    errs = validate_question_type(_type(
        uses=["audio", "speakerID"],
        answer_rule="the speaker identifier",
        answer={"kind": "field_value", "key": "speakerID"}))
    assert errs == ["id_exposure"]


def test_unknown_answer_field_rejected():
    errs = validate_question_type(_type(
        uses=["audio", "gender"],
        answer_rule="the gender label",
        answer={"kind": "field_value", "key": "emotion"}))
    assert errs == ["unknown_answer_field:emotion"]


def _eq_type(keys, uses=("audio", "speakerID"), inputs=2):
    return _type(
        id="QS_EQ", uses=list(uses), input_count=inputs,
        answer_rule="equality prose",
        answer={"kind": "equality", "keys": list(keys)})


def test_equality_speaker_ids_passes():
    assert validate_question_type(
        _eq_type(["speakerID_1", "speakerID_2"])) == []


def test_equality_indexed_same_attribute_passes():
    assert validate_question_type(_eq_type(
        ["gender_1", "gender_2"], uses=["audio", "gender"])) == []
    assert validate_question_type(_eq_type(
        ["region_1", "region_2"], uses=["audio", "region"])) == []


def test_equality_mixed_attributes_rejected():
    errs = validate_question_type(_eq_type(
        ["gender_1", "region_2"], uses=["audio", "gender", "region"]))
    assert errs == ["equality_field_mismatch"]


def test_equality_wrong_arity_rejected():
    assert validate_question_type(_eq_type(["speakerID_1"])) == [
        "invalid_equality_arity"]
    assert validate_question_type(
        _eq_type(["a", "b", "c"])) == ["invalid_equality_arity"]


def test_equality_unknown_field_rejected():
    errs = validate_question_type(_eq_type(["foo_1", "foo_2"]))
    assert errs == ["unknown_answer_field:foo_1",
                    "unknown_answer_field:foo_2"]


def test_equality_same_instance_suffix_rejected():
    errs = validate_question_type(_eq_type(["gender_1", "gender_1"],
                                           uses=["audio", "gender"]))
    assert errs == ["invalid_equality_instances"]


def test_equality_unindexed_pair_rejected():
    # Indexed F_1/F_2 references are mandatory; bare fields are not a
    # canonical pair.
    assert validate_question_type(_eq_type(
        ["speakerID", "speakerID"])) == ["invalid_equality_instances"]


def test_equality_reversed_instances_rejected():
    errs = validate_question_type(_eq_type(["gender_2", "gender_1"],
                                           uses=["audio", "gender"]))
    assert errs == ["invalid_equality_instances"]


def test_pairwise_audio_alias_forms_valid():
    # Generic uses=["audio","speakerID"] is valid for verification...
    assert validate_question_type(_type(
        id="QS_V", uses=["audio", "speakerID"], input_count=2,
        answer_rule="same speakerID => yes, otherwise no",
        answer={"kind": "equality",
                "keys": ["speakerID_1", "speakerID_2"]})) == []
    # ...and so is the explicit alias form.
    assert validate_question_type(_type(
        id="QS_V", uses=["audio_1", "audio_2", "speakerID"], input_count=2,
        answer_rule="same speakerID => yes, otherwise no",
        answer={"kind": "equality",
                "keys": ["speakerID_1", "speakerID_2"]})) == []
    # Same-gender comparison over plain audio uses.
    assert validate_question_type(_type(
        id="QS_G", uses=["audio", "gender"], input_count=2,
        answer_rule="same gender on both clips",
        answer={"kind": "equality",
                "keys": ["gender_1", "gender_2"]})) == []


def test_metadata_only_mapping_rejected_for_missing_audio():
    errs = validate_question_type(_type(
        id="QS_R1_06", name="Province to Region Mapping",
        uses=["province", "region"],
        answer_rule="The broad dialect region that the given province "
                    "belongs to.",
        answer={"kind": "derived_field", "source_key": "province",
                "target_key": "region"}))
    assert "missing_audio_input" in errs


def test_unknown_uses_field_rejected():
    for bad in ("duration", "age", "emotion"):
        errs = validate_question_type(_type(uses=["audio", bad]))
        assert any(e == f"unknown_uses_field:{bad}" for e in errs), bad


def test_normalize_field_reference_exact_spec():
    n = normalize_field_reference
    assert n("gender", SCHEMA_FIELDS) == "gender"
    assert n("gender_1", SCHEMA_FIELDS) == "gender"
    assert n("gender_2", SCHEMA_FIELDS) == "gender"
    assert n("speakerID_1", SCHEMA_FIELDS) == "speakerID"
    assert n("region_2", SCHEMA_FIELDS) == "region"
    assert n("foo_1", SCHEMA_FIELDS) is None
    assert n("associated", SCHEMA_FIELDS) is None
    # Only a FINAL _1/_2 is stripped; nothing else.
    assert n("a_1_b", SCHEMA_FIELDS) is None
    assert n("x_12", SCHEMA_FIELDS) is None
    assert n("audio_3", SCHEMA_FIELDS) is None
    assert n(None, SCHEMA_FIELDS) is None
    assert n(3, SCHEMA_FIELDS) is None
    # Exact schema fields are never rewritten.
    assert n("audio", {"audio", "audio_1"}) == "audio"
    assert n("audio_1", {"audio", "audio_1"}) == "audio_1"


def test_unknown_placeholder_rejected():
    assert validate_template_text("Hãy cho biết [KEY_SOURCE] của người nói.") == [
        "unknown_placeholder:KEY_SOURCE"]
    assert validate_template_text("Hãy xác định [KEY] của người nói.") == []
    assert validate_template_text("Nội dung là [VALUE].") == [
        "answer_leakage"]
    assert validate_template_text("   ") == ["empty_template"]


def test_binding_key_mapping():
    assert binding_key_for(_type()) == "gender"
    assert binding_key_for(_type(
        answer={"kind": "derived_field", "source_key": "province",
                "target_key": "region"})) == "region"
    assert binding_key_for(_type(
        answer={"kind": "equality",
                "keys": ["speakerID_1", "speakerID_2"]})) is None
    assert binding_key_for(_type(answer={"kind": "transcription"})) is None


def test_gate_drops_bad_templates_and_orphan_types():
    types = [_type(), _type(id="QS_R1_02", uses=["audio", "age"],
                            answer_rule="the age label",
                            answer={"kind": "field_value", "key": "age"})]
    templates = [
        {"template_id": "T1", "question_type_id": "QS_R1_01",
         "text": "Hãy xác định [KEY] của người nói."},
        {"template_id": "T2", "question_type_id": "QS_R1_01",
         "text": "Hãy cho biết [KEY_SOURCE]."},
        {"template_id": "T3", "question_type_id": "QS_R9_99",
         "text": "Text không thuộc type nào."},
    ]
    gate = gate_round(types, templates)
    assert [t["id"] for t in gate["valid_types"]] == ["QS_R1_01"]
    assert [t["template_id"] for t in gate["valid_templates"]] == ["T1"]
    assert {a["type_id"] for a in gate["auto_rejected"]} == {"QS_R1_02"}
    dropped = {d["template_id"]: d["reason"]
               for d in gate["dropped_templates"]}
    assert dropped["T2"].startswith("unknown_placeholder:")
    assert dropped["T3"] == "unknown_type"


def test_gate_rejects_type_left_without_templates():
    types = [_type()]
    gate = gate_round(types, [
        {"template_id": "T1", "question_type_id": "QS_R1_01",
         "text": "Hãy cho biết [KEY_SOURCE]."}])
    assert gate["valid_types"] == []
    assert gate["auto_rejected"] == [
        {"type_id": "QS_R1_01", "reason": "no_renderable_template"}]


def test_gate_rejects_key_template_for_keyless_type():
    pair = _type(id="QS_R1_03", uses=["audio", "speakerID"], input_count=2,
                 answer_rule="same speakerID => yes, otherwise no",
                 answer={"kind": "equality",
                         "keys": ["speakerID_1", "speakerID_2"]})
    gate = gate_round([pair], [
        {"template_id": "P1", "question_type_id": "QS_R1_03",
         "text": "Hai đoạn này có cùng [KEY] không?"}])
    assert gate["valid_types"] == []
    assert gate["auto_rejected"][0]["reason"] == "no_renderable_template"
    assert gate["dropped_templates"][0]["reason"] == "unbound_key"

"""Style `uses` contract: build+score resources, audio necessity (§41-42)."""

from src.autonomous_qa.language.question_style import build_question_style_prompt
from src.autonomous_qa.core.validity import validate_question_type


def _prompt():
    return build_question_style_prompt("README", [], 1)


def test_uses_is_not_gold_fields_only():
    text = _prompt()
    assert "union of build + score resources" in text


def test_uses_includes_evidence_and_annotation():
    text = _prompt()
    assert "evidence presented to / consumed by the QA model" in text
    assert "annotation or hidden fields" in text


def test_direct_example_includes_audio_label():
    text = _prompt()
    assert 'uses = ["audio", "text"]' in text
    assert 'uses = ["audio", "gender"]' in text


def test_pairwise_example_includes_audio_hidden_key():
    text = _prompt()
    assert 'uses = ["audio", "speakerID"]' in text


def test_metadata_tasks_must_not_add_fake_audio():
    text = _prompt()
    assert "Do not add" in text
    assert "merely to satisfy" in text
    assert "INCORRECT workaround" in text


def test_audio_necessity_self_check_exists():
    text = _prompt()
    assert "AUDIO NECESSITY TEST" in text
    assert "without listening" in text.replace("\n", " ")
    assert "DO NOT RETURN THIS PROPOSAL" in text
    assert "`uses` MUST explicitly include" in text


def test_few_shots_remain_examples_not_taxonomy():
    text = _prompt()
    assert "NOT a taxonomy" in text


def _type(uses, answer, rule="some rule"):
    return {"id": "QX", "name": "N", "goal": "g", "uses": uses,
            "input_count": 1, "answer_rule": rule, "answer": answer}


def test_audio_gender_passes():
    assert validate_question_type(_type(
        ["audio", "gender"],
        {"kind": "field_value", "key": "gender"})) == []


def test_bare_gender_fails_missing_audio():
    assert validate_question_type(_type(
        ["gender"], {"kind": "field_value", "key": "gender"})) == [
        "missing_audio_input"]


def test_audio_speakerid_supports_verification():
    assert validate_question_type({
        "id": "QX", "name": "N", "goal": "g",
        "uses": ["audio", "speakerID"], "input_count": 2,
        "answer_rule": "same speaker",
        "answer": {"kind": "equality",
                   "keys": ["speakerID_1", "speakerID_2"]}}) == []


def test_bare_speakerid_fails_missing_audio():
    errs = validate_question_type(_type(
        ["speakerID"], {"kind": "field_value", "key": "gender"}))
    assert "missing_audio_input" in errs


def test_province_region_only_fails_missing_audio():
    errs = validate_question_type(_type(
        ["province", "region"],
        {"kind": "derived_field", "source_key": "province",
         "target_key": "region"}))
    assert "missing_audio_input" in errs


def test_python_never_auto_adds_audio():
    from src.autonomous_qa.core.validity import gate_round
    t = {"id": "QX", "name": "N", "goal": "g", "uses": ["province", "region"],
         "input_count": 1, "answer_rule": "r",
         "answer": {"kind": "derived_field", "source_key": "province",
                    "target_key": "region"}}
    before = list(t["uses"])
    gate_round([t], [])
    assert t["uses"] == before  # gate rejects; never injects audio
    assert before == ["province", "region"]


def test_goal_words_do_not_satisfy_audio():
    errs = validate_question_type({
        "id": "QX", "name": "speaker voice recording",
        "goal": "listen to the speaker voice recording",
        "uses": ["gender"], "input_count": 1,
        "answer_rule": "audio speaker voice",
        "answer": {"kind": "field_value", "key": "gender"}})
    assert "missing_audio_input" in errs

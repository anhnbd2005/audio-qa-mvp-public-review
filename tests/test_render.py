"""Preview sampler: per-type coverage, pairwise pos/neg, hard placeholder failures."""

import pytest

from src.autonomous_qa.production.render_preview import render_preview, render_question


def _rows():
    return [
        {"audio": "a1.wav", "text": "xin chao", "gender": "female",
         "region": "Central", "province": "Hue", "speakerID": "spk1"},
        {"audio": "a2.wav", "text": "tam biet", "gender": "male",
         "region": "North", "province": "Hanoi", "speakerID": "spk1"},
        {"audio": "a3.wav", "text": "cam on", "gender": "male",
         "region": "South", "province": "Saigon", "speakerID": "spk2"},
    ]


def _type(tid, kind="field_value", **over):
    base = {
        "type_id": tid, "name": tid, "goal": "g",
        "uses": ["audio", "gender"], "input_count": 1,
        "answer_rule": "the gender label",
        "answer": {"kind": "field_value", "key": "gender"},
        "kept_templates": [{"template_id": f"{tid}_T",
                            "text": "Hãy xác định [KEY] của người nói."}],
    }
    base.update(over)
    if kind == "pair":
        base.update({
            "uses": ["audio", "speakerID"], "input_count": 2,
            "answer_rule": "same speakerID => yes, otherwise no",
            "answer": {"kind": "equality",
                       "keys": ["speakerID_1", "speakerID_2"]},
            "kept_templates": [{
                "template_id": f"{tid}_T",
                "text": "Hai đoạn này có cùng một người nói không?"}],
        })
    return base


def test_key_substituted_and_gold_never_in_question():
    q, kr = render_question(
        "Hãy xác định [KEY] của người nói?", "T1", "gender",
        {"gender": "giới tính"}, "female")
    assert q == "Hãy xác định giới tính của người nói?"
    assert kr == "giới tính"
    assert "female" not in q  # gold answer never leaks into the question


def test_value_placeholder_is_hard_failure():
    with pytest.raises(ValueError, match="\\[VALUE\\]"):
        render_question(
            "Hãy xác định [KEY] ([VALUE])?", "T1", "gender",
            {"gender": "giới tính"}, "female")
    with pytest.raises(ValueError, match="\\[VALUE\\]"):
        render_question(
            "Giới tính của người nói là gì? [VALUE]", "T2", "gender",
            {"gender": "giới tính"}, "female")


def test_unknown_placeholder_is_hard_failure():
    with pytest.raises(ValueError, match="unresolved placeholders"):
        render_question("Hãy cho biết [KEY_SOURCE]?", "T1", "gender",
                        {"gender": "giới tính"}, "female")


def test_unbalanced_bracket_leftovers_fail():
    with pytest.raises(ValueError, match="unresolved placeholder"):
        render_question("Hãy cho biết [KEY] [", "T1", "gender",
                        {"gender": "giới tính"}, "female")


def test_missing_key_realization_fails():
    with pytest.raises(ValueError, match="realization"):
        render_question("Hãy xác định [KEY]?", "T1", "province", {}, "Hue")


def test_pairwise_answers_use_accented_boolean():
    out = render_preview(
        _rows(), [_type("QS_P", kind="pair")], {}, n=10, random_seed=1)
    answers = [i["answer"] for i in out]
    assert "Có" in answers and "Không" in answers
    assert "Co" not in answers and "Khong" not in answers


def test_each_type_covered_when_quota_allows():
    types = [_type(f"QS_{i}") for i in range(4)]
    out = render_preview(_rows(), types, {"gender": "giới tính"},
                         n=10, random_seed=42)
    covered = {i["question_type_id"] for i in out}
    assert covered == {f"QS_{i}" for i in range(4)}


def test_preview_has_no_unresolved_placeholders_and_cap():
    types = [_type(f"QS_{i}") for i in range(12)]
    out = render_preview(_rows(), types, {"gender": "giới tính"},
                         n=10, random_seed=42)
    assert len(out) == 10
    import re
    for item in out:
        assert not re.search(r"\[[A-Za-z_][A-Za-z0-9_]*\]", item["question"])
        assert item["answer_source"]["kind"] in (
            "field_value", "equality", "derived_field")


def test_pairwise_without_any_pair_yields_nothing():
    out = render_preview(_rows()[:1], [_type("QS_P", kind="pair")], {},
                         n=10, random_seed=1)
    assert out == []


def test_pairwise_negative_only_when_no_same_speaker_pair():
    rows = [dict(r, speakerID=f"spk{i}") for i, r in enumerate(_rows())]
    out = render_preview(rows, [_type("QS_P", kind="pair")], {},
                         n=10, random_seed=1)
    assert {i["answer"] for i in out} == {"Không"}


def test_field_value_gold_matches_row():
    out = render_preview(_rows(), [_type("QS_G")], {"gender": "giới tính"},
                         n=1, random_seed=42)
    assert out[0]["answer_source"] == {
        "kind": "field_value", "field": "gender"}
    assert out[0]["key_realization"] == "giới tính"

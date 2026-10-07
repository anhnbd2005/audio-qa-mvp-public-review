"""Fix 4 tests: preview pass 2 opposite-class priority (41-51)."""

from src.autonomous_qa.production.render_preview import render_preview

KR = {"gender": "giới tính", "region": "vùng giọng",
      "province": "tỉnh thành phương ngữ"}


def _rows():
    def r(audio, text, gender, region, province, speakerID):
        return {"audio": audio, "text": text, "gender": gender,
                "region": region, "province": province,
                "speakerID": speakerID}
    return [
        r("a0.wav", "t0", "male", "North", "Hue", "spkA"),
        r("a1.wav", "t1", "male", "North", "Hue", "spkA"),
        r("a2.wav", "t2", "female", "South", "Saigon", "spkB"),
        r("a3.wav", "t3", "male", "South", "Hanoi", "spkC"),
        r("a4.wav", "t4", "female", "Central", "DaNang", "spkD"),
        r("a5.wav", "t5", "male", "Central", "Hue", "spkE"),
    ]


def _eq(tid, name, keys, uses):
    return {"type_id": tid, "name": name, "goal": "g", "uses": uses,
            "input_count": 2,
            "answer_rule": "equality prose",
            "answer": {"kind": "equality", "keys": list(keys)},
            "kept_templates": [{"template_id": tid + "_T",
                                "text": "Hai đoạn này có cùng không?"}]}


def _direct(tid, key):
    return {"type_id": tid, "name": tid, "goal": "g",
            "uses": ["audio", key], "input_count": 1,
            "answer_rule": "direct prose",
            "answer": {"kind": "field_value", "key": key},
            "kept_templates": [{"template_id": tid + "_T",
                                "text": "Hãy xác định [KEY] của người nói."}]}


def _eight_types():
    return [
        _eq("QS_E1", "Speaker Verification",
            ["speakerID_1", "speakerID_2"], ["audio", "speakerID"]),
        _direct("QS_D1", "gender"),
        _eq("QS_E2", "Banana Check",  # name says nothing; kind decides
            ["gender_1", "gender_2"], ["audio", "gender"]),
        _direct("QS_D2", "region"),
        _eq("QS_E3", "Same Region Check",
            ["region_1", "region_2"], ["audio", "region"]),
        _direct("QS_D3", "province"),
        _eq("QS_E4", "Same Province Check",
            ["province_1", "province_2"], ["audio", "province"]),
        {"type_id": "QS_D4", "name": "Transcription", "goal": "g",
         "uses": ["audio", "text"], "input_count": 1,
         "answer_rule": "transcribe prose",
         "answer": {"kind": "field_value", "key": "text"},
         "kept_templates": [{"template_id": "QS_D4_T",
                             "text": "Hãy chép lại nội dung."}]},
    ]


def test_pass1_covers_all_eight_types():
    out = render_preview(_rows(), _eight_types(), KR, n=10, random_seed=7)
    first8 = [i["question_type_id"] for i in out[:8]]
    assert first8 == [t["type_id"] for t in _eight_types()]


def test_only_two_slots_remain_after_coverage():
    out = render_preview(_rows(), _eight_types(), KR, n=10, random_seed=7)
    assert len(out) == 10  # 8 coverage + exactly 2 extra


def test_extra_slots_are_opposite_class_equality():
    out = render_preview(_rows(), _eight_types(), KR, n=10, random_seed=7)
    extras = out[8:]
    assert len(extras) == 2
    for item in extras:
        assert item["answer_source"]["kind"] == "equality", item
    # Each extra is the opposite class of its type's pass-1 answer.
    for item in extras:
        tid = item["question_type_id"]
        first = next(i for i in out[:8] if i["question_type_id"] == tid)
        assert {first["answer"], item["answer"]} == {"Có", "Không"}


def test_no_second_direct_while_pairwise_missing_exists():
    out = render_preview(_rows(), _eight_types(), KR, n=10, random_seed=7)
    extras = out[8:]
    kinds = [i["answer_source"]["kind"] for i in extras]
    assert kinds == ["equality", "equality"], kinds


def test_pairwise_selection_uses_kind_not_names():
    out = render_preview(_rows(), _eight_types(), KR, n=10, random_seed=7)
    banana = [i for i in out if i["question_type_id"] == "QS_E2"]
    # "Banana Check" has no pairwise-sounding name but kind == equality:
    # its items carry pair audio + boolean answers.
    assert banana
    assert all(isinstance(i["audio"], list) for i in banana)
    assert all(i["answer"] in ("Có", "Không") for i in banana)


def test_extra_seeks_opposite_of_first_example():
    out = render_preview(_rows(), _eight_types(), KR, n=10, random_seed=7)
    by_type = {}
    for i in out:
        by_type.setdefault(i["question_type_id"], []).append(i["answer"])
    for tid, answers in by_type.items():
        if len(answers) == 2 and all(
                a in ("Có", "Không") for a in answers):
            assert set(answers) == {"Có", "Không"}, (tid, answers)


def test_unavailable_opposite_class_skips_to_next_type():
    # One same-speaker pair (verification can do both classes);
    # every row male (gender equality can only answer Có).
    rows = [
        {"audio": "a0.wav", "text": "t0", "gender": "male",
         "region": "North", "province": "Hue", "speakerID": "spkA"},
        {"audio": "a1.wav", "text": "t1", "gender": "male",
         "region": "North", "province": "Hue", "speakerID": "spkA"},
        {"audio": "a2.wav", "text": "t2", "gender": "male",
         "region": "South", "province": "Saigon", "speakerID": "spkB"},
        {"audio": "a3.wav", "text": "t3", "gender": "male",
         "region": "Central", "province": "DaNang", "speakerID": "spkC"},
    ]
    types = [_eq("QS_E1", "Speaker Verification",
                 ["speakerID_1", "speakerID_2"], ["audio", "speakerID"]),
             _eq("QS_E2", "Same Gender Check",
                 ["gender_1", "gender_2"], ["audio", "gender"]),
             _direct("QS_D1", "gender")]
    out = render_preview(rows, types, KR, n=10, random_seed=7)
    by_type = {}
    for i in out:
        by_type.setdefault(i["question_type_id"], []).append(i["answer"])
    # Gender type: only Có possible -> never a fabricated Không.
    assert set(by_type["QS_E2"]) == {"Có"}
    # Verification type got the opposite-class extra instead.
    assert set(by_type["QS_E1"]) == {"Có", "Không"}


def test_normal_fill_resumes_after_pairwise_exhausted():
    rows = [dict(r, gender="male", speakerID=f"spk{i}")
            for i, r in enumerate(_rows())]
    types = [_eq("QS_E1", "Speaker Verification",
                 ["speakerID_1", "speakerID_2"], ["audio", "speakerID"]),
             _direct("QS_D1", "gender")]
    out = render_preview(rows, types, KR, n=5, random_seed=7)
    # pass1: 2 items; pass2: verification opposite (gender skipped);
    # pass3: ordinary fill to quota.
    assert len(out) == 5
    assert {i["question_type_id"] for i in out} == {"QS_E1", "QS_D1"}


def test_preview_count_capped_and_boolean_exact():
    out = render_preview(_rows(), _eight_types(), KR, n=10, random_seed=7)
    assert len(out) <= 10
    for i in out:
        if i["answer_source"]["kind"] == "equality":
            assert i["answer"] in ("Có", "Không")
            assert i["answer"] not in ("Co", "Khong")

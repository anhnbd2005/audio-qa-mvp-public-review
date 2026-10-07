from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.autonomous_qa.language.template_contracts import (
    OperatorTemplateOutput,
    QuestionTemplateSpec,
    TypeContract,
    build_operator_prompt,
    build_type_contracts,
    instantiate_type,
    model_facing,
    normalize_value,
    operator_contracts,
    validate_instance,
    validate_operator_output,
)

REGISTRY = Path("resources/semantics/vimd_type_registry.json")


def contracts():
    return build_type_contracts(json.loads(REGISTRY.read_text(encoding="utf-8")))


def supported():
    return [c for c in contracts() if c.template_status == "ELIGIBLE"]


def make_contract(operator="DIRECT", field="region", type_id="t1"):
    return TypeContract(
        type_id=type_id,
        operator=operator,
        semantic_field=field,
        semantic_description="semantic attribute",
        semantic_phrase_vi="thuộc tính",
        entity_scope="speaker",
        audio_input_count=1 if operator in {"DIRECT", "TARGET_MATCH"} else 2,
        condition_fields=[field]
        if operator in {"TARGET_MATCH", "PAIRWISE_SELECTION"}
        else [],
        gold_source_fields=[field],
        answer_kind={
            "DIRECT": "field_value",
            "EQUALITY": "boolean",
            "TARGET_MATCH": "boolean",
            "PAIRWISE_SELECTION": "audio_index",
        }[operator],
        answer_mode={
            "DIRECT": "FIELD_VALUE",
            "EQUALITY": "BOOLEAN",
            "TARGET_MATCH": "BOOLEAN",
            "PAIRWISE_SELECTION": "A_B_SELECTION",
        }[operator],
        template_status="ELIGIBLE",
        template_policy={},
        instantiation_policy={
            "avoid_same_speaker": False,
            "text_negative_length_bucket": field == "text",
        },
        source_status="SUPPORTED",
    )


def make_template(contract, text=None, template_id="tpl1"):
    target = contract.operator in {"TARGET_MATCH", "PAIRWISE_SELECTION"}
    defaults = {
        "DIRECT": "Đoạn âm thanh này có thuộc tính nào?",
        "EQUALITY": "Hai đoạn âm thanh có cùng thuộc tính hay không?",
        "TARGET_MATCH": "Đoạn âm thanh này có khớp với [TARGET_VALUE] hay không?",
        "PAIRWISE_SELECTION": "Đoạn âm thanh nào, A hoặc B, có thuộc tính [TARGET_VALUE]?",
    }
    return QuestionTemplateSpec(
        template_id=template_id,
        type_id=contract.type_id,
        operator=contract.operator,
        question_text=text or defaults[contract.operator],
        answer_mode=contract.answer_mode,
        audio_reference_style="attached",
        required_placeholders=["[TARGET_VALUE]"] if target else [],
        forbidden_placeholders=[] if target else ["[TARGET_VALUE]"],
        allowed_answer_forms=[],
    )


def make_rows(tmp_path):
    rows = []
    specs = [
        ("a", "North", "P1", "xin chào bạn", "s1"),
        ("b", "North", "P1", "xin chào mọi người", "s2"),
        ("c", "South", "P2", "hôm nay trời đẹp", "s3"),
        ("d", "South", "P2", "ngày mai trời đẹp", "s4"),
    ]
    for sid, region, province, text, speaker in specs:
        audio = tmp_path / f"{sid}.wav"
        audio.write_bytes(b"RIFF")
        rows.append(
            {
                "dataset": "x",
                "split": "train",
                "sample_id": sid,
                "audio_path": str(audio),
                "metadata": {
                    "region": region,
                    "province_name": province,
                    "text": text,
                    "speakerID": speaker,
                },
            }
        )
    return rows


def output_for(contract, templates):
    return OperatorTemplateOutput(operator=contract.operator, templates=templates)


def test_five_operator_contracts_exist():
    assert set(operator_contracts()) == {
        "COMPOSITE",
        "DIRECT",
        "EQUALITY",
        "PAIRWISE_SELECTION",
        "TARGET_MATCH",
    }


def test_operator_core_has_no_dataset_specific_fields():
    raw = {k: v.model_dump() for k, v in operator_contracts().items()}
    leaves = (
        json.dumps(list(raw.values()))
        .replace("context_roles", "")
        .replace("target_role", "")
    )
    assert all(
        x not in leaves for x in ("province_name", '"region"', "speakerID", '"text"')
    )


def test_every_supported_type_maps_to_operator():
    assert len(supported()) == 11
    assert all(c.operator in operator_contracts() for c in supported())


def test_review_types_deferred():
    deferred = [c for c in contracts() if c.template_status == "DEFERRED_DATA_CAPACITY"]
    assert len(deferred) == 2
    assert {c.semantic_field for c in deferred} == {"speakerID", "text"}


def test_prompt_has_schema_last_and_no_values():
    c = make_contract()
    prompt = build_operator_prompt("DIRECT", [c])
    assert "JSON SCHEMA" in prompt and "North" not in prompt
    assert prompt.rfind("$defs") > prompt.find("TYPE CONTRACTS")


def test_complete_template_accounting_passes():
    c = make_contract()
    templates = [
        make_template(
            c, template_id=f"t{i}", text=f"Đoạn âm thanh này có thuộc tính nào{'?' * i}"
        )
        for i in (1, 2)
    ]
    _, errors = validate_operator_output(output_for(c, templates), [c])
    assert errors == []


def test_unused_optional_audio_placeholder_is_valid():
    c = make_contract("EQUALITY")
    templates = []
    texts = [
        "Hai đoạn âm thanh có cùng thuộc tính hay không?",
        "Thuộc tính trong hai đoạn âm thanh có giống nhau không?",
    ]
    for index, text in enumerate(texts, 1):
        template = make_template(
            c,
            text=text,
            template_id=f"t{index}",
        )
        template.optional_placeholders = ["[AUDIO_A]", "[AUDIO_B]"]
        templates.append(template)
    checked, errors = validate_operator_output(output_for(c, templates), [c])
    assert errors == []
    assert all(row["validation_status"] == "PASS" for row in checked)


def test_unknown_type_rejected():
    c = make_contract()
    bad = make_template(c)
    bad.type_id = "unknown"
    _, errors = validate_operator_output(output_for(c, [bad]), [c])
    assert {e["code"] for e in errors} >= {"UNKNOWN_TYPE", "MISSING_TYPE"}


def test_missing_type_rejected():
    c = make_contract()
    _, errors = validate_operator_output(output_for(c, []), [c])
    assert errors[0]["code"] == "MISSING_TYPE"


def test_unknown_placeholder_rejected():
    c = make_contract()
    t = make_template(c, "Đoạn âm thanh [REGION] thuộc loại nào?")
    t.required_placeholders = ["[REGION]"]
    checked, _ = validate_operator_output(
        output_for(c, [t, make_template(c, template_id="t2")]), [c]
    )
    assert "NO_UNKNOWN_PLACEHOLDER" in checked[0]["reasons"]


@pytest.mark.parametrize("token", ["speakerID", "filename", "province_code", "abc.wav"])
def test_hidden_or_provenance_rejected(token):
    c = make_contract()
    t = make_template(c, f"Đoạn âm thanh có {token} là gì?")
    checked, _ = validate_operator_output(
        output_for(c, [t, make_template(c, template_id="t2")]), [c]
    )
    assert checked[0]["validation_status"] == "REJECT"


def test_operator_semantic_drift_rejected():
    c = make_contract("TARGET_MATCH")
    t = make_template(c, "Đoạn âm thanh thuộc vùng [TARGET_VALUE] nào?")
    checked, _ = validate_operator_output(
        output_for(c, [t, make_template(c, template_id="t2")]), [c]
    )
    assert "OPERATOR_SEMANTIC_DRIFT" in checked[0]["reasons"]


@pytest.mark.parametrize("operator", ["DIRECT", "EQUALITY"])
def test_target_placeholder_forbidden(operator):
    c = make_contract(operator)
    t = make_template(
        c,
        "Hai đoạn âm thanh có khớp [TARGET_VALUE] không?"
        if operator == "EQUALITY"
        else "Đoạn âm thanh [TARGET_VALUE] là gì?",
    )
    t.required_placeholders = ["[TARGET_VALUE]"]
    checked, _ = validate_operator_output(
        output_for(c, [t, make_template(c, template_id="t2")]), [c]
    )
    assert "TARGET_PLACEHOLDER_FORBIDDEN" in checked[0]["reasons"]


@pytest.mark.parametrize("operator", ["TARGET_MATCH", "PAIRWISE_SELECTION"])
def test_target_placeholder_required(operator):
    c = make_contract(operator)
    t = make_template(
        c,
        "Hai đoạn âm thanh A hoặc B phù hợp?"
        if operator == "PAIRWISE_SELECTION"
        else "Đoạn âm thanh có phù hợp hay không?",
    )
    t.required_placeholders = []
    checked, _ = validate_operator_output(
        output_for(c, [t, make_template(c, template_id="t2")]), [c]
    )
    assert "TARGET_PLACEHOLDER_REQUIRED" in checked[0]["reasons"]


def test_direct_derives_correct_gold(tmp_path):
    c = make_contract()
    records = instantiate_type(
        c, make_template(c), make_rows(tmp_path), {"instances": 4, "seed": 42}
    )
    assert all(
        r.gold.value == r.derivation_evidence["hidden_values"][0] for r in records
    )


def test_equality_positive_and_negative_correct(tmp_path):
    c = make_contract("EQUALITY")
    records = instantiate_type(
        c, make_template(c), make_rows(tmp_path), {"instances": 4, "seed": 42}
    )
    assert [r.gold.value for r in records] == [True, False, True, False]
    assert all(not validate_instance(r) for r in records)


def test_identical_audio_cannot_pair_with_itself(tmp_path):
    c = make_contract("EQUALITY")
    records = instantiate_type(
        c, make_template(c), make_rows(tmp_path), {"instances": 8, "seed": 42}
    )
    assert all(len(set(r.audio_ids)) == 2 for r in records)


def test_target_match_positive_and_negative(tmp_path):
    c = make_contract("TARGET_MATCH")
    records = instantiate_type(
        c, make_template(c), make_rows(tmp_path), {"instances": 4, "seed": 42}
    )
    assert [r.gold.value for r in records] == [True, False, True, False]
    assert all(not validate_instance(r) for r in records)


def test_selection_exactly_one_match_and_ab_balance(tmp_path):
    c = make_contract("PAIRWISE_SELECTION")
    records = instantiate_type(
        c, make_template(c), make_rows(tmp_path), {"instances": 4, "seed": 42}
    )
    assert [r.gold.value for r in records] == ["A", "B", "A", "B"]
    assert all(sum(r.derivation_evidence["matches"]) == 1 for r in records)


@pytest.mark.parametrize("matches", [[True, True], [False, False]])
def test_selection_both_or_neither_rejected(tmp_path, matches):
    c = make_contract("PAIRWISE_SELECTION")
    record = instantiate_type(
        c, make_template(c), make_rows(tmp_path), {"instances": 1, "seed": 42}
    )[0]
    record.derivation_evidence["matches"] = matches
    assert "INVALID_SELECTION_CARDINALITY" in validate_instance(record)


def test_ab_order_randomization_preserves_gold(tmp_path):
    c = make_contract("PAIRWISE_SELECTION")
    records = instantiate_type(
        c, make_template(c), make_rows(tmp_path), {"instances": 2, "seed": 1}
    )
    for r in records:
        expected = "A" if r.derivation_evidence["matches"] == [True, False] else "B"
        assert r.gold.value == expected


def test_seed_is_deterministic(tmp_path):
    c = make_contract("TARGET_MATCH")
    args = (c, make_template(c), make_rows(tmp_path), {"instances": 8, "seed": 42})
    a = [r.model_dump() for r in instantiate_type(*args)]
    b = [r.model_dump() for r in instantiate_type(*args)]
    assert a == b


def test_sampler_does_not_materialize_pair_space(tmp_path):
    c = make_contract("EQUALITY")
    records = instantiate_type(
        c, make_template(c), make_rows(tmp_path), {"instances": 3, "seed": 2}
    )
    assert len(records) == 3
    assert all(r.generation["sampler"] == "indexed_bounded_v1" for r in records)


def test_speaker_id_filename_and_provenance_not_model_facing(tmp_path):
    c = make_contract()
    record = instantiate_type(
        c, make_template(c), make_rows(tmp_path), {"instances": 1, "seed": 1}
    )[0]
    public = model_facing(record)
    raw = json.dumps(public)
    assert (
        "source_rows" not in public
        and "speakerID" not in raw
        and "derivation_evidence" not in public
    )


def test_candidate_hidden_label_not_visible(tmp_path):
    c = make_contract("EQUALITY")
    record = instantiate_type(
        c, make_template(c), make_rows(tmp_path), {"instances": 1, "seed": 1}
    )[0]
    assert all(
        str(v) not in record.question
        for v in record.derivation_evidence["hidden_values"]
    )


def test_visible_target_allowed_for_match_and_select(tmp_path):
    for operator in ("TARGET_MATCH", "PAIRWISE_SELECTION"):
        c = make_contract(operator)
        record = instantiate_type(
            c, make_template(c), make_rows(tmp_path), {"instances": 1, "seed": 1}
        )[0]
        assert str(record.visible_context["target_value"]) in record.question
        assert not validate_instance(record)


def test_target_alone_cannot_determine_answer(tmp_path):
    c = make_contract("TARGET_MATCH")
    records = instantiate_type(
        c, make_template(c), make_rows(tmp_path), {"instances": 2, "seed": 1}
    )
    assert {r.gold.value for r in records} == {True, False}
    assert all(r.visible_context for r in records)


def test_transcript_direct_does_not_expose_text(tmp_path):
    c = make_contract("DIRECT", "text")
    record = instantiate_type(
        c,
        make_template(c, "Hãy ghi lại nội dung được nói trong đoạn âm thanh."),
        make_rows(tmp_path),
        {"instances": 1, "seed": 1},
    )[0]
    assert record.gold.value not in record.question


def test_transcript_target_match_exposes_target_not_hidden_transcript(tmp_path):
    c = make_contract("TARGET_MATCH", "text")
    record = instantiate_type(
        c,
        make_template(c, "Đoạn âm thanh có nói [TARGET_VALUE] hay không?"),
        make_rows(tmp_path),
        {"instances": 2, "seed": 1},
    )[1]
    assert record.visible_context["target_value"] in record.question
    assert record.derivation_evidence["hidden_values"][0] not in record.question


def test_text_normalization_whitespace_case():
    assert normalize_value("  Xin   CHÀO ", "text") == normalize_value(
        "xin chào", "text"
    )

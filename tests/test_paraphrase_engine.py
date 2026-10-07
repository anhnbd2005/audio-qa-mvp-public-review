from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.autonomous_qa.language.paraphrase_engine import (
    ParaphraseCandidate,
    audit_capacity,
    mock_output,
    normalized_lexical,
    semantic_contract_hash,
    strict_parse,
    validate_output,
)
from src.autonomous_qa.language.template_engine import vimd_field_specs
from src.autonomous_qa.compiler.semantic_field_specs import SemanticFieldSpec
from src.autonomous_qa.language.template_contracts import TypeContract, build_type_contracts
from src.autonomous_qa.language.template_renderer import load_library, render_blueprint

LIB = Path("resources/language/template_library.json")
GUARD = Path("resources/language/semantic_guard.json")
REGISTRY = Path("resources/semantics/vimd_type_registry.json")


def library():
    return load_library(LIB)


def guard():
    return json.loads(GUARD.read_text(encoding="utf-8"))


def sources(operator="TARGET_MATCH", semantic_class="categorical_attribute"):
    return library().lookup(operator, semantic_class)


def contracts():
    raw = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return [
        item for item in build_type_contracts(raw) if item.source_status == "SUPPORTED"
    ]


def validate_mock(operator="TARGET_MATCH", semantic_class="categorical_attribute"):
    src = sources(operator, semantic_class)
    return validate_output(mock_output(operator, src), src, guard(), "mock", "hash")


def test_capacity_axes_are_separate():
    from src.autonomous_qa.language.template_renderer import RenderedTemplate

    assert {"preview_capacity_status", "full_train_capacity_status"} <= set(
        RenderedTemplate.model_fields
    )


def test_preview_insufficiency_does_not_imply_full_train_insufficiency():
    state = {
        "preview": "INSUFFICIENT_POSITIVE_CAPACITY",
        "full_train": "NOT_AUDITED",
    }
    assert state["preview"] != state["full_train"]


def test_partial_capacity_uses_group_counts_without_pairs(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [
        {"province_name": "A"},
        {"province_name": "A"},
        {"province_name": "B"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    c = next(item for item in contracts() if item.type_id == "vimd-v2-s1-006")
    item = audit_capacity([c], path)["types"][0]
    assert item["positive_capacity"] and item["negative_capacity"]
    assert item["full_train_capacity_status"] == "NOT_AUDITED"
    assert item["evidence_summary"]["groups_with_count_ge_2"] == 1
    assert item["evidence_summary"]["pair_enumeration"] == "NOT_PERFORMED"
    assert item["semantic_status"] == "SUPPORTED"


def test_complete_train_province_equality_is_sufficient():
    path = Path(
        "data/materialized/vimd/train.jsonl"
    )
    result = audit_capacity(contracts(), path)
    province = next(
        item for item in result["types"] if item["type_id"] == "vimd-v2-s1-006"
    )
    assert result["metadata_rows"] == 15023
    assert province["full_train_capacity_status"] == "SUFFICIENT"
    assert province["evidence_summary"]["groups_with_count_ge_2"] > 0


def test_candidate_requires_source_id():
    with pytest.raises(ValidationError):
        ParaphraseCandidate(
            operator="DIRECT",
            semantic_class="categorical_attribute",
            paraphrase_pattern="x",
            required_slots=[],
            answer_kind="field_value",
        )


def test_semantic_contract_hash_stable():
    source = sources()[0]
    assert semantic_contract_hash(source) == semantic_contract_hash(source)


def test_all_28_mock_sources_account_and_pass():
    total = 0
    for operator in ("DIRECT", "EQUALITY", "PAIRWISE_SELECTION", "TARGET_MATCH"):
        src = sorted(
            library().lookup(operator, "categorical_attribute")
            + library().lookup(operator, "numeric_attribute")
            + library().lookup(operator, "text_content"),
            key=lambda item: item.blueprint_id,
        )
        results, accounting = validate_output(
            mock_output(operator, src), src, guard(), "m", "h"
        )
        assert accounting["complete"]
        assert all(item.paraphrase_status == "PASS" for item in results)
        total += len(results)
    assert total == 30


@pytest.mark.parametrize("kind", ["missing", "unknown", "duplicate"])
def test_accounting_failure(kind):
    src = sources()
    output = mock_output("TARGET_MATCH", src)
    if kind == "missing":
        output.paraphrases.pop()
    elif kind == "unknown":
        output.paraphrases[0].source_blueprint_id = "unknown"
    else:
        output.paraphrases.append(output.paraphrases[0].model_copy())
    with pytest.raises(ValueError, match="ACCOUNTING"):
        validate_output(output, src, guard(), "c", "h")


def test_strict_parser_only_strips_fence():
    output = mock_output("TARGET_MATCH", sources())
    raw = json.dumps(output.model_dump())
    assert strict_parse(f"```json\n{raw}\n```").operator == "TARGET_MATCH"
    with pytest.raises((ValidationError, json.JSONDecodeError)):
        strict_parse(json.dumps({"foreign": output.model_dump()}))


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("operator", "DIRECT", "OPERATOR_UNCHANGED"),
        ("semantic_class", "numeric_attribute", "SEMANTIC_CLASS_UNCHANGED"),
        ("answer_kind", "field_value", "ANSWER_KIND_UNCHANGED"),
    ],
)
def test_source_contract_mutations_rejected(field, value, code):
    src = sources()
    output = mock_output("TARGET_MATCH", src)
    setattr(output.paraphrases[0], field, value)
    results, _ = validate_output(output, src, guard(), "c", "h")
    assert code in results[0].reason_codes


def test_required_slot_set_preserved():
    src = sources()
    output = mock_output("TARGET_MATCH", src)
    output.paraphrases[0].required_slots = []
    results, _ = validate_output(output, src, guard(), "c", "h")
    assert "REQUIRED_SLOT_SET_PRESERVED" in results[0].reason_codes


@pytest.mark.parametrize("operator", ["TARGET_MATCH", "PAIRWISE_SELECTION"])
def test_target_slot_cannot_disappear(operator):
    src = sources(operator)[0:1]
    output = mock_output(operator, src)
    output.paraphrases[0].paraphrase_pattern = output.paraphrases[
        0
    ].paraphrase_pattern.replace("[TARGET_VALUE]", "mục tiêu")
    results, _ = validate_output(output, src, guard(), "c", "h")
    assert results[0].paraphrase_status == "REJECTED"


def test_direct_cannot_gain_target_and_unknown_slot_rejected():
    src = sources("DIRECT")[0:1]
    output = mock_output("DIRECT", src)
    output.paraphrases[0].paraphrase_pattern += " [TARGET_VALUE] [REGION]"
    results, _ = validate_output(output, src, guard(), "c", "h")
    assert {"SLOT_OCCURRENCE_PRESERVED", "NO_UNKNOWN_SLOT"} <= set(
        results[0].reason_codes
    )


@pytest.mark.parametrize(
    ("operator", "pattern", "code"),
    [
        (
            "DIRECT",
            "[ATTRIBUTE_PHRASE] [ENTITY_PHRASE] có đúng không?",
            "DIRECT_OPERATOR_DRIFT",
        ),
        (
            "TARGET_MATCH",
            "Hãy xác định [ATTRIBUTE_PHRASE] của [ENTITY_PHRASE] [TARGET_VALUE].",
            "TARGET_MATCH_OPERATOR_DRIFT",
        ),
        ("EQUALITY", "Chọn A hoặc B có [ATTRIBUTE_PHRASE].", "EQUALITY_OPERATOR_DRIFT"),
    ],
)
def test_operator_drift_rejected(operator, pattern, code):
    src = sources(operator)[0:1]
    output = mock_output(operator, src)
    output.paraphrases[0].paraphrase_pattern = pattern
    results, _ = validate_output(output, src, guard(), "c", "h")
    assert code in results[0].reason_codes


def test_selection_requires_ab_semantics():
    src = sources("PAIRWISE_SELECTION")[0:1]
    output = mock_output("PAIRWISE_SELECTION", src)
    output.paraphrases[
        0
    ].paraphrase_pattern = "Đoạn này có [ATTRIBUTE_PHRASE] [TARGET_VALUE] không?"
    results, _ = validate_output(output, src, guard(), "c", "h")
    assert "PAIRWISE_SELECTION_CONTEXT_DRIFT" in results[0].reason_codes


def test_exact_numeric_approximation_rejected():
    src = sources("TARGET_MATCH", "numeric_attribute")[0:1]
    output = mock_output("TARGET_MATCH", src)
    output.paraphrases[0].paraphrase_pattern += " khoảng"
    results, _ = validate_output(output, src, guard(), "c", "h")
    assert "NUMERIC_EXACT_POLICY_DRIFT" in results[0].reason_codes


def test_exact_text_containment_rejected():
    src = sources("TARGET_MATCH", "text_content")[0:1]
    output = mock_output("TARGET_MATCH", src)
    output.paraphrases[
        0
    ].paraphrase_pattern = (
        "Đoạn âm thanh có chứa [CONTENT_PHRASE] [TARGET_VALUE] không?"
    )
    results, _ = validate_output(output, src, guard(), "c", "h")
    assert "TEXT_EXACT_CONTAINMENT_DRIFT" in results[0].reason_codes


def test_exact_selection_similarity_rejected():
    src = sources("PAIRWISE_SELECTION")[0:1]
    output = mock_output("PAIRWISE_SELECTION", src)
    output.paraphrases[0].paraphrase_pattern += " phù hợp hơn"
    results, _ = validate_output(output, src, guard(), "c", "h")
    assert "SELECTION_MATCH_POLICY_DRIFT" in results[0].reason_codes


def test_exact_categorical_equality_accepted():
    results, _ = validate_mock("EQUALITY")
    assert all(item.paraphrase_status == "PASS" for item in results)


def test_punctuation_duplicate_normalizes_and_rejects_source():
    assert normalized_lexical("Xin chào?") == normalized_lexical("xin chào!!!")
    src = sources("DIRECT")[0:1]
    output = mock_output("DIRECT", src)
    output.paraphrases[0].paraphrase_pattern = src[0].pattern + "!!!"
    results, _ = validate_output(output, src, guard(), "c", "h")
    assert "LEXICAL_DUPLICATE_SOURCE" in results[0].reason_codes


@pytest.mark.parametrize(
    ("name", "semantic_class", "phrase", "unit"),
    [
        ("region", "categorical_attribute", "vùng phương ngữ", None),
        ("gender", "categorical_attribute", "giới tính", None),
        ("emotion", "categorical_attribute", "cảm xúc", None),
        ("language", "categorical_attribute", "ngôn ngữ", None),
        ("age", "numeric_attribute", "độ tuổi", "tuổi"),
    ],
)
def test_zero_llm_paraphrase_rendering(name, semantic_class, phrase, unit):
    source = sources("DIRECT", semantic_class)[0]
    para = validate_output(
        mock_output("DIRECT", [source]), [source], guard(), "c", "h"
    )[0][0]
    spec = SemanticFieldSpec(
        field_name=name,
        semantic_class=semantic_class,
        entity_scope="speaker",
        entity_phrase="người nói",
        attribute_phrase=phrase,
        value_phrase=phrase,
        unit=unit,
    )
    contract = TypeContract(
        type_id=f"demo-{name}",
        operator="DIRECT",
        semantic_field=name,
        semantic_description="demo",
        semantic_phrase_vi=phrase,
        entity_scope="speaker",
        audio_input_count=1,
        condition_fields=[],
        gold_source_fields=[name],
        answer_kind="field_value",
        answer_mode="FIELD_VALUE",
        template_status="ELIGIBLE",
        template_policy={},
        instantiation_policy={},
        source_status="SUPPORTED",
    )
    rendered = render_blueprint(
        para.as_template_blueprint(), spec, contract, "canonical"
    )
    assert phrase in rendered.question_pattern.casefold()


def test_text_paraphrase_renders_vimd_spec():
    source = sources("DIRECT", "text_content")[0]
    para = validate_output(
        mock_output("DIRECT", [source]), [source], guard(), "c", "h"
    )[0][0]
    contract = next(item for item in contracts() if item.type_id == "vimd-v2-s1-001")
    rendered = render_blueprint(
        para.as_template_blueprint(),
        vimd_field_specs()["text"],
        contract,
        "canonical",
    )
    assert "nội dung" in rendered.question_pattern.casefold()


def test_ineligible_field_cannot_be_enabled_by_paraphrase():
    source = sources("DIRECT")[0]
    para = validate_output(
        mock_output("DIRECT", [source]), [source], guard(), "c", "h"
    )[0][0]
    spec = SemanticFieldSpec(
        field_name="gender",
        semantic_class="categorical_attribute",
        entity_scope="speaker",
        entity_phrase="người nói",
        attribute_phrase="giới tính",
        semantic_status="UNSUPPORTED_ENCODING",
    )
    contract = TypeContract(
        type_id="gender",
        operator="DIRECT",
        semantic_field="gender",
        semantic_description="x",
        semantic_phrase_vi="x",
        entity_scope="speaker",
        audio_input_count=1,
        condition_fields=[],
        gold_source_fields=["gender"],
        answer_kind="field_value",
        answer_mode="FIELD_VALUE",
        template_status="ELIGIBLE",
        template_policy={},
        instantiation_policy={},
        source_status="SUPPORTED",
    )
    with pytest.raises(ValueError, match="SEMANTIC_FIELD_NOT_SUPPORTED"):
        render_blueprint(para.as_template_blueprint(), spec, contract, "canonical")

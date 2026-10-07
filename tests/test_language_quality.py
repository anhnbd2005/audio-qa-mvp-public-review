from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.common.config import ROOT
from src.autonomous_qa.language.language_quality import (
    LanguageQualityDecision,
    OperatorLanguageQualityOutput,
    ProductionGenerationConfig,
    approved_patterns_for_contract,
    build_quality_pairs,
    build_registry,
    deterministic_language_reasons,
    language_realization_identity,
    load_language_registry,
    load_paraphrase_library,
    mock_output,
    render_approved_patterns,
    semantic_instance_identity,
    strict_parse,
    validate_accounting,
)
from src.autonomous_qa.language.paraphrase_engine import semantic_contract_hash
from src.autonomous_qa.language.template_engine import sha256_file, vimd_field_specs
from src.autonomous_qa.language.template_contracts import build_type_contracts
from src.autonomous_qa.language.template_renderer import TemplateBlueprint, load_library

CANONICAL_PATH = ROOT / "resources" / "language" / "template_library.json"
PARAPHRASE_PATH = ROOT / "resources" / "language" / "paraphrase_library.json"
TYPE_REGISTRY_PATH = ROOT / "resources" / "semantics" / "vimd_type_registry.json"


@pytest.fixture(scope="module")
def libraries():
    return load_library(CANONICAL_PATH), load_paraphrase_library(PARAPHRASE_PATH)


@pytest.fixture(scope="module")
def pairs(libraries):
    result, audit = build_quality_pairs(*libraries)
    assert audit["rejected"] == 0
    return result


def _decision(pair, decision="PASS"):
    fail = decision == "REJECT"
    review = decision == "REVIEW"
    return LanguageQualityDecision(
        canonical_id=pair.canonical_id,
        paraphrase_id=pair.paraphrase_id,
        decision=decision,
        naturalness="FAIL" if fail else "REVIEW" if review else "PASS",
        semantic_faithfulness="PASS",
        ambiguity="PASS",
        slot_bindability="PASS",
        operator_preservation="PASS",
        match_policy_preservation="PASS",
        reason_codes=[decision] if decision != "PASS" else [],
        short_reason="test",
    )


def _changed_para(libraries, source_id, **changes):
    _, paras = libraries
    para = next(p for p in paras.paraphrases if p.source_blueprint_id == source_id)
    return para.model_copy(update=changes, deep=True)


def _guard(libraries, source_id, pattern):
    canonical, _ = libraries
    source = next(p for p in canonical.blueprints if p.blueprint_id == source_id)
    para = _changed_para(libraries, source_id, pattern=pattern)
    return deterministic_language_reasons(source, para)


# A. Terminology / separation
def test_style_discovery_is_not_language_quality(pairs):
    assert all(not hasattr(pair, "style_discovery_status") for pair in pairs)


def test_language_quality_cannot_create_semantic_type():
    fields = LanguageQualityDecision.model_fields
    assert "semantic_class" not in fields and "new_type" not in fields


def test_language_quality_cannot_change_operator():
    assert "operator" not in LanguageQualityDecision.model_fields


def test_language_quality_cannot_change_field_eligibility():
    assert "field_eligibility" not in LanguageQualityDecision.model_fields


# B. Strict accounting
def test_each_input_has_one_decision(pairs):
    subset = [p for p in pairs if p.operator == "DIRECT"]
    out = mock_output("DIRECT", subset)
    assert validate_accounting(out, "DIRECT", subset)["complete"]


def test_missing_id_fails(pairs):
    subset = [p for p in pairs if p.operator == "DIRECT"]
    out = mock_output("DIRECT", subset)
    out.evaluations.pop()
    with pytest.raises(ValueError, match="ACCOUNTING_FAILURE"):
        validate_accounting(out, "DIRECT", subset)


def test_extra_id_fails(pairs):
    subset = [p for p in pairs if p.operator == "DIRECT"]
    out = mock_output("DIRECT", subset)
    out.evaluations[0].paraphrase_id = "unknown"
    with pytest.raises(ValueError, match="ACCOUNTING_FAILURE"):
        validate_accounting(out, "DIRECT", subset)


def test_duplicate_id_fails(pairs):
    subset = [p for p in pairs if p.operator == "DIRECT"]
    out = mock_output("DIRECT", subset)
    out.evaluations.append(out.evaluations[0].model_copy(deep=True))
    with pytest.raises(ValueError, match="ACCOUNTING_FAILURE"):
        validate_accounting(out, "DIRECT", subset)


def test_empty_nonempty_fails(pairs):
    subset = [p for p in pairs if p.operator == "DIRECT"]
    out = OperatorLanguageQualityOutput(operator="DIRECT", evaluations=[])
    with pytest.raises(ValueError, match="ACCOUNTING_FAILURE"):
        validate_accounting(out, "DIRECT", subset)


def test_strict_parser_only_accepts_schema(pairs):
    out = mock_output("DIRECT", [p for p in pairs if p.operator == "DIRECT"])
    assert strict_parse(f"```json\n{out.model_dump_json()}\n```").operator == "DIRECT"
    with pytest.raises((ValidationError, ValueError, json.JSONDecodeError)):
        strict_parse('{"operator":"DIRECT","evaluations":[],"repair":true}')


# C. Structural preservation
@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("operator", "TARGET_MATCH", "OPERATOR_MISMATCH"),
        ("semantic_class", "numeric_attribute", "SEMANTIC_CLASS_MISMATCH"),
        ("answer_kind", "boolean", "ANSWER_KIND_MISMATCH"),
        ("required_slots", ["[TARGET_VALUE]"], "REQUIRED_SLOT_MISMATCH"),
    ],
)
def test_structural_changes_rejected(libraries, field, value, reason):
    canonical, paras = libraries
    changed = _changed_para(libraries, "vi1_dir_cat_01", **{field: value})
    altered = paras.model_copy(update={"paraphrases": [changed]}, deep=True)
    _, audit = build_quality_pairs(canonical, altered)
    assert reason in audit["rejections"][0]["reason_codes"]


def test_audio_arity_is_operator_contract(pairs):
    assert all(
        p.required_audio_input_count
        == (2 if p.operator in {"EQUALITY", "PAIRWISE_SELECTION"} else 1)
        for p in pairs
    )


def test_target_visibility_preserved(libraries):
    reasons = _guard(
        libraries,
        "vi1_tm_cat_01",
        "[ATTRIBUTE_PHRASE] của [ENTITY_PHRASE] là gì?",
    )
    assert "TARGET_VISIBILITY_DRIFT" in reasons


# D. Text exactness
def test_whole_text_equality_accepted(libraries):
    reasons = _guard(
        libraries,
        "vi1_tm_text_01",
        "Đoạn âm thanh có nói đúng nội dung “[TARGET_VALUE]” không?",
    )
    assert "TEXT_EXACT_CONTAINMENT_DRIFT" not in reasons


@pytest.mark.parametrize("term", ["chứa", "từ khóa", "nhắc đến"])
def test_text_containment_or_keyword_drift_rejected(libraries, term):
    reasons = _guard(
        libraries,
        "vi1_tm_text_01",
        f"Đoạn âm thanh có {term} “[TARGET_VALUE]” không?",
    )
    assert "TEXT_EXACT_CONTAINMENT_DRIFT" in reasons


# E. Numeric exactness
def test_numeric_exact_accepted(libraries):
    reasons = _guard(
        libraries,
        "vi1_tm_num_01",
        "[ATTRIBUTE_PHRASE] của [ENTITY_PHRASE] có bằng [TARGET_VALUE] [UNIT] không?",
    )
    assert "NUMERIC_EXACT_POLICY_DRIFT" not in reasons


@pytest.mark.parametrize("term", ["khoảng", "xấp xỉ"])
def test_numeric_approximation_rejected(libraries, term):
    reasons = _guard(
        libraries,
        "vi1_tm_num_01",
        f"[ATTRIBUTE_PHRASE] có {term} [TARGET_VALUE] [UNIT] không?",
    )
    assert "NUMERIC_EXACT_POLICY_DRIFT" in reasons


# F. Operator drift
def test_direct_to_target_match_fails(libraries):
    assert "DIRECT_TO_TARGET_MATCH_DRIFT" in _guard(
        libraries,
        "vi1_dir_cat_01",
        "[ATTRIBUTE_PHRASE] có phải [TARGET_VALUE] không?",
    )


def test_target_match_to_direct_fails(libraries):
    assert "TARGET_VISIBILITY_DRIFT" in _guard(
        libraries,
        "vi1_tm_cat_01",
        "[ATTRIBUTE_PHRASE] của [ENTITY_PHRASE] là gì?",
    )


def test_equality_to_selection_fails(libraries):
    assert "EQUALITY_TO_SELECTION_DRIFT" in _guard(
        libraries,
        "vi1_eq_cat_01",
        "Trong hai đoạn A và B, hãy chọn đoạn nào có [ATTRIBUTE_PHRASE].",
    )


def test_selection_to_boolean_fails(libraries):
    reasons = _guard(
        libraries,
        "vi1_sel_cat_01",
        "Hai đoạn A và B có [ATTRIBUTE_PHRASE] là [TARGET_VALUE] không?",
    )
    assert "SELECTION_ANSWER_KIND_DRIFT" in reasons


# G. Registry
def _registry(libraries, pairs, overrides=None):
    decisions = [
        _decision(pair, (overrides or {}).get(pair.paraphrase_id, "PASS"))
        for pair in pairs
    ]
    return build_registry(*libraries, decisions, "mock", ["call"])


def test_pass_entry_active(libraries, pairs):
    registry, _ = _registry(libraries, pairs)
    assert all(item.enabled for item in registry.entries)


def test_review_entry_stored_inactive(libraries, pairs):
    registry, _ = _registry(libraries, pairs, {pairs[0].paraphrase_id: "REVIEW"})
    item = next(x for x in registry.entries if x.source_id == pairs[0].paraphrase_id)
    assert item.quality_status == "PRODUCTION_REVIEW" and not item.enabled


def test_reject_not_runtime_active(libraries, pairs):
    registry, rejected = _registry(libraries, pairs, {pairs[0].paraphrase_id: "REJECT"})
    assert not any(x.source_id == pairs[0].paraphrase_id for x in registry.entries)
    assert rejected[0]["quality_status"] == "PRODUCTION_REJECT"


def test_bad_paraphrase_does_not_remove_canonical(libraries, pairs):
    registry, _ = _registry(libraries, pairs, {pairs[0].paraphrase_id: "REJECT"})
    source = pairs[0].canonical_id
    assert any(x.enabled and x.source_id == source for x in registry.entries)


def test_registry_hash_deterministic(libraries, pairs):
    first, _ = _registry(libraries, pairs)
    second, _ = _registry(libraries, pairs)
    assert first.registry_hash == second.registry_hash == first.computed_hash()


# H. Runtime
def _contracts():
    raw = json.loads(TYPE_REGISTRY_PATH.read_text(encoding="utf-8"))
    return [c for c in build_type_contracts(raw) if c.source_status == "SUPPORTED"]


def test_production_renderer_reads_approved_registry(libraries, pairs):
    registry, _ = _registry(libraries, pairs, {pairs[0].paraphrase_id: "REVIEW"})
    contract = _contracts()[0]
    entries = approved_patterns_for_contract(
        registry, vimd_field_specs()[contract.semantic_field], contract
    )
    assert entries and all(item.enabled for item in entries)


def test_new_field_does_not_trigger_llm(libraries, pairs):
    registry, _ = _registry(libraries, pairs)
    assert registry.active("DIRECT", "categorical_attribute")


def test_new_qa_does_not_trigger_llm(libraries, pairs):
    registry, _ = _registry(libraries, pairs)
    assert "quality_calls" not in registry.model_dump()


def test_missing_operator_class_returns_explicit_error(libraries, pairs):
    registry, _ = _registry(libraries, pairs)
    spec = vimd_field_specs()["region"].model_copy(
        update={"semantic_class": "numeric_attribute"}
    )
    contract = _contracts()[0].model_copy(
        update={"operator": "EQUALITY", "answer_kind": "boolean"}
    )
    registry.entries = [
        x
        for x in registry.entries
        if not (x.operator == "EQUALITY" and x.semantic_class == "numeric_attribute")
    ]
    with pytest.raises(ValueError, match="LANGUAGE_LIBRARY_COVERAGE_MISSING"):
        approved_patterns_for_contract(registry, spec, contract)


def test_production_config_schema_loads_without_arbitrary_budgets():
    raw = yaml.safe_load(
        (ROOT / "configs" / "qa_generation_production.yaml").read_text(encoding="utf-8")
    )
    cfg = ProductionGenerationConfig.model_validate(raw)
    assert cfg.total_target_qa is None and cfg.language_preflight_artifact


def test_semantic_and_language_identity_are_separate():
    semantic = semantic_instance_identity(
        type_id="type", operator="DIRECT", audio_ids=["a"], target=None, gold="x"
    )
    first = language_realization_identity(semantic, "lang_a")
    second = language_realization_identity(semantic, "lang_b")
    assert first != second
    assert semantic == semantic_instance_identity(
        type_id="type", operator="DIRECT", audio_ids=["a"], target=None, gold="x"
    )


def test_canonical_registry_renders_all_supported_vimd_types():
    path = ROOT / "resources" / "language" / "production_registry.json"
    registry = load_language_registry(path)
    specs = vimd_field_specs()
    rendered = [
        render_approved_patterns(registry, specs[contract.semantic_field], contract)
        for contract in _contracts()
    ]
    assert len(rendered) == 11 and all(rendered)


# I. Canonical integrity
def test_canonical_template_hash(libraries):
    assert (
        libraries[0].library_hash
        == "290e99933d713bdbbd1794c497577c0b90a48b82ce29fe98b644950331db7b8b"
    )


def test_canonical_paraphrase_hash_is_valid(libraries):
    assert (
        libraries[1].library_hash
        == "3d5fd4f86a2f18614cac5fd2bacd553dd9a7e3038c7c8678732f6d5fcb744922"
    )
    assert (
        sha256_file(PARAPHRASE_PATH)
        == "6262ee1031e1f036da5a173d1da8ee7638d9e788ae168832ede684afe0ea6b65"
    )


def test_decision_schema_forbids_rewritten_pattern(pairs):
    payload = _decision(pairs[0]).model_dump()
    payload["better_wording"] = "forbidden"
    with pytest.raises(ValidationError):
        LanguageQualityDecision.model_validate(payload)


def test_decision_consistency_is_strict(pairs):
    payload = _decision(pairs[0]).model_dump()
    payload["decision"] = "PASS"
    payload["naturalness"] = "FAIL"
    with pytest.raises(ValidationError, match="INCONSISTENT_QUALITY_DECISION"):
        LanguageQualityDecision.model_validate(payload)


def test_semantic_contract_hash_remains_source_derived(libraries):
    canonical, _ = libraries
    source = canonical.blueprints[0]
    assert semantic_contract_hash(source) == semantic_contract_hash(
        TemplateBlueprint.model_validate(source.model_dump())
    )


def test_registry_round_trip_hash(tmp_path: Path, libraries, pairs):
    registry, _ = _registry(libraries, pairs)
    path = tmp_path / "registry.json"
    path.write_text(registry.model_dump_json(), encoding="utf-8")
    assert load_language_registry(path).registry_hash == registry.registry_hash

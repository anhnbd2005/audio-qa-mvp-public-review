"""Generic training-data compiler core tests."""

from __future__ import annotations

import pytest

from src.autonomous_qa.production import production_zero_llm as pzl
from src.autonomous_qa.language import template_bank
from src.autonomous_qa.certification.comparator_collision import comparator_collision_audit
from src.autonomous_qa.certification.contract_closure import close_contract, closure_complete
from src.autonomous_qa.certification.guardrails import GuardrailRegistry
from src.autonomous_qa.core.information_value import (
    information_metrics,
    information_value_decision,
)
from src.autonomous_qa.compiler.semantic_comparators import load_comparator_registry
from src.autonomous_qa.compiler.semantic_task import SemanticTaskSpec
from src.autonomous_qa.compiler.semantic_tiers import validate_pair
from src.common.severity import max_severity
from src.autonomous_qa.certification.task_sanitation import (
    BLOCKED_BY_CONTRACT,
    DROP_LOCAL_ANOMALY,
    SYSTEMATIC_CONTRACT_PATTERN,
    SanitationPolicy,
    TaskEligibilityMap,
    classify_anomalies,
    task_scoped_eligibility,
)


def _atomic_spec(**over):
    base = {
        "type_id": "t_atomic",
        "proposition_id": "p",
        "proposition_description": "d",
        "operator": "TARGET_MATCH",
        "classification": "PRIMITIVE_RELATION",
        "kind": "ATOMIC",
        "audio_arity": 1,
        "visible_context_roles": ["candidate_target_text"],
        "outputs": [{"role": "matches", "kind": "boolean"}],
        "comparator_id": "exact_normalized_text",
        "invariances": ["surrounding_whitespace"],
        "source_role_mapping": {"source_field": "observed_transcription_norm"},
    }
    base.update(over)
    return SemanticTaskSpec.model_validate(base)


# --- tiers / gold origin -----------------------------------------------------


def test_tier_gold_origin_validation():
    assert validate_pair("T1_PERCEPTION", "SOURCE") == []
    assert validate_pair("T2_DERIVED", "DERIVED_SOURCE") == []
    assert validate_pair("T3_RELATIONAL", "DERIVED_SOURCE") == []
    assert validate_pair("T1_PERCEPTION", "DERIVED_SOURCE") != []
    assert validate_pair("BOGUS", "SOURCE") != []


def test_severity_aggregation():
    assert max_severity(["PASS", "REVIEW", "BLOCKING"]) == "BLOCKING"
    assert max_severity(["PASS", "REPORT_ONLY"]) == "REPORT_ONLY"
    assert max_severity([]) == "PASS"


# --- information value -------------------------------------------------------


def test_information_value_adaptive_over_80_rule():
    # C3-like: 84% majority but usable minority -> NOT rejected.
    c3 = information_metrics([True] * 481 + [False] * 2530)
    decision = information_value_decision(c3)
    assert decision["severity"] == "PASS"
    assert "PASS_WITH_PLANNER_BALANCING_CAPABILITY" in decision["reasons"]
    assert decision["usable_balanced_capacity"] == 962
    # Tiny minority -> reject.
    tiny = information_metrics(["a"] * 1000 + ["b"] * 3)
    assert information_value_decision(tiny)["severity"] == "AUTO_REJECT_TASK"
    # Constant label -> reject.
    const = information_metrics(["a"] * 1000)
    assert information_value_decision(const)["severity"] == "AUTO_REJECT_TASK"


# --- comparator collision ----------------------------------------------------


def test_comparator_collision_detects_and_allows_declared():
    comparator = load_comparator_registry().by_id("normalized_text_exact_casefold")
    audit = comparator_collision_audit(["Ab", "ab", "ac"], comparator)
    assert audit["collision_class_count"] == 1
    assert audit["severity"] == "BLOCKING"
    allowed = comparator_collision_audit(
        ["Ab", "ab", "ac"], comparator, allowed_collapse_groups=[["Ab", "ab"]]
    )
    assert allowed["severity"] == "PASS"


# --- sanitation --------------------------------------------------------------


def test_task_scoped_eligibility_per_task():
    elmap = TaskEligibilityMap()
    elmap.set("r1", "taskA", DROP_LOCAL_ANOMALY)
    elmap.set("r1", "taskB", "ELIGIBLE")
    assert not elmap.is_eligible("r1", "taskA")
    assert elmap.is_eligible("r1", "taskB")


def test_systematic_pattern_not_local_noise():
    systematic = classify_anomalies(
        ["PRESENTATION_ONLY"] * 599, 3181, SanitationPolicy()
    )
    assert systematic["classification"] == SYSTEMATIC_CONTRACT_PATTERN
    assert systematic["severity"] == "BLOCKING"
    local = classify_anomalies(
        [f"sig_{i % 9}" for i in range(10)], 1000, SanitationPolicy()
    )
    assert local["classification"] == "LOCAL_HETEROGENEOUS_NOISE"
    assert local["severity"] == "AUTO_DROP_ROW"


def test_task_scoped_systematic_blocks_whole_task():
    anomalies = [
        {"row_id": f"r{i}", "task_id": "tA", "signature": "PRESENTATION_ONLY"}
        for i in range(10)
    ]
    result = task_scoped_eligibility(anomalies, SanitationPolicy())
    assert result["tasks"]["tA"]["classification"] == SYSTEMATIC_CONTRACT_PATTERN
    assert result["eligibility"]["r0"]["tA"]["status"] == BLOCKED_BY_CONTRACT


# --- contract closure --------------------------------------------------------


def test_contract_closure_complete_and_hash_stable():
    closure = close_contract(
        task_id="t",
        tier="T2_DERIVED",
        gold_origin="DERIVED_SOURCE",
        proposition="p",
        operator="TARGET_MATCH",
    )
    assert closure.contract_hash
    assert closure_complete(closure) == []
    closure2 = close_contract(
        task_id="t",
        tier="T2_DERIVED",
        gold_origin="DERIVED_SOURCE",
        proposition="p",
        operator="TARGET_MATCH",
    )
    assert closure.contract_hash == closure2.contract_hash


def test_contract_closure_incomplete_raises():
    with pytest.raises(ValueError, match="CONTRACT_CLOSURE_INCOMPLETE"):
        close_contract(
            task_id="t",
            tier="T2_DERIVED",
            gold_origin="DERIVED_SOURCE",
            proposition="",
            operator="X",
        )
    with pytest.raises(ValueError, match="CONTRACT_CLOSURE_INCOMPLETE"):
        close_contract(
            task_id="t",
            tier="T1_PERCEPTION",
            gold_origin="DERIVED_SOURCE",
            proposition="p",
            operator="X",
        )


# --- template bank -----------------------------------------------------------


def test_template_assignment_deterministic_and_balanced():
    ids = [f"q{i}" for i in range(2000)]
    a = template_bank.assign_templates(ids, 20, registry_hash="reg", seed=7)
    b = template_bank.assign_templates(ids, 20, registry_hash="reg", seed=7)
    assert a["assignment"] == b["assignment"]
    assert a["min_usage"] == a["max_usage"] == 100
    assert a["imbalance"] == 0
    # different registry hash can change assignment
    c = template_bank.assign_templates(ids, 20, registry_hash="other", seed=7)
    assert c["assignment"] != a["assignment"]


# --- guardrail registry applicability ----------------------------------------


def test_guardrail_registry_applicability():
    registry = GuardrailRegistry()
    atomic = _atomic_spec(tier="T2_DERIVED", gold_origin="DERIVED_SOURCE")
    ids = registry.applicable(atomic, {})
    assert "NEGATIVE_VALIDITY" in ids
    assert "SOURCE_SUPPORT" in ids

    composite = SemanticTaskSpec.model_validate(
        {
            "type_id": "t_comp",
            "proposition_id": "p",
            "proposition_description": "d",
            "operator": "COMPOSITE",
            "classification": "CHAIN_DERIVED",
            "kind": "STRUCTURED",
            "audio_arity": 2,
            "visible_context_roles": [],
            "outputs": [
                {"role": "a", "kind": "field_value"},
                {"role": "b", "kind": "boolean", "dependencies": ["a"]},
            ],
        }
    ).model_copy(update={"tier": "T3_RELATIONAL", "gold_origin": "DERIVED_SOURCE"})
    cids = registry.applicable(composite, {})
    assert "COMPOSITE_REDUNDANCY" in cids
    assert "PAIR_VALIDITY" in cids



def test_guardrail_registry_flags_component_visibility_and_redundancy():
    registry = GuardrailRegistry()
    spec = _atomic_spec(tier="T2_DERIVED", gold_origin="DERIVED_SOURCE")
    report = registry.run(spec, {"component_visible_types": ["t_atomic"]})
    assert report.status == "FAIL"
    assert report.by_id("COMPONENT_VISIBILITY").severity == "AUTO_REJECT_TASK"

    composite = SemanticTaskSpec.model_validate(
        {
            "type_id": "t_comp2",
            "proposition_id": "p",
            "proposition_description": "d",
            "operator": "COMPOSITE",
            "classification": "CHAIN_DERIVED",
            "kind": "STRUCTURED",
            "audio_arity": 1,
            "visible_context_roles": [],
            "outputs": [
                {"role": "a", "kind": "field_value"},
                {"role": "b", "kind": "boolean", "dependencies": ["a"]},
            ],
        }
    ).model_copy(update={"tier": "T3_RELATIONAL", "gold_origin": "DERIVED_SOURCE"})
    report2 = registry.run(
        composite, {"composite_classification": "REDUNDANT_COMPOSITE"}
    )
    assert report2.by_id("COMPOSITE_REDUNDANCY").severity == "AUTO_REJECT_TASK"


def test_multiclass_direct_information_value():
    metrics = information_metrics(["A"] * 500 + ["B"] * 400 + ["C"] * 300)
    decision = information_value_decision(metrics)
    assert decision["severity"] == "PASS"
    assert metrics["class_count"] == 3
    assert metrics["majority_rate"] < 0.8




# --- production zero LLM -----------------------------------------------------


def test_production_zero_llm_scan_passes():
    result = pzl.scan_production_zero_llm()
    assert result["status"] == "PASS", result["offenders"]
    pzl.assert_production_zero_llm()

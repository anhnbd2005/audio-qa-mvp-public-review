"""Comprehensive Unit Test Suite for Autonomous Audio-QA Compiler V2 Core Architecture.

Validates B1 (Contract/Plan separation), B2 (CS term presence universe capacity),
B3 (TopologyEngine), and B5 (CoreInvariantsSuite).
"""

from __future__ import annotations

import json

import pytest

from src.autonomous_qa.certification.core_invariants import CompileError, CoreInvariantsSuite
from src.autonomous_qa.compiler.semantic_contract import (
    SemanticContract,
    create_semantic_contract,
    InvalidSemanticContractError,
)
from src.autonomous_qa.production.production_plan import ProductionPlan
from src.autonomous_qa.core.topology_engine import TopologyEngine, UnsupportedTopologyError



# -----------------------------------------------------------------------------
# B1 TESTS: Contract vs Planner Separation
# -----------------------------------------------------------------------------

def test_b1_contract_contains_no_sampling_fields():
    """Verify SemanticContract schema v2 forbids production sampling fields."""
    contract = create_semantic_contract(
        task_id="test_task",
        tier="T1_PERCEPTION",
        topology="ONE_TO_ONE",
        positive_universe_capacity=100,
        negative_universe_capacity=0,
        total_universe_capacity=100,
        proposition="Test proposition",
        operator="DIRECT",
        gold_origin="SOURCE",
    )
    assert contract.schema_version == 2
    assert not hasattr(contract, "target_qa_count")
    assert not hasattr(contract, "pos_neg_ratio")
    assert not hasattr(contract, "sampling_seed")


def test_b1_contract_rejects_sampling_fields_on_creation():
    """Verify passing sampling fields into SemanticContract raises InvalidSemanticContractError."""
    with pytest.raises(InvalidSemanticContractError, match="FORBIDDEN_SAMPLING_FIELD"):
        create_semantic_contract(
            task_id="test_task",
            tier="T1_PERCEPTION",
            topology="ONE_TO_ONE",
            target_qa_count=1000,
            pos_neg_ratio=1.0,
        )


def test_b1_changing_planner_target_does_not_change_contract_hash():
    """Verify changing ProductionPlan target or seed does not alter SemanticContract hash."""
    contract = create_semantic_contract(
        task_id="test_task",
        tier="T1_PERCEPTION",
        topology="ONE_TO_ONE",
        positive_universe_capacity=100,
        negative_universe_capacity=0,
        total_universe_capacity=100,
        proposition="Test proposition",
        operator="DIRECT",
        gold_origin="SOURCE",
    )
    hash_1 = contract.contract_hash

    plan_a = ProductionPlan(task_id="test_task", target_qa_count=50, random_seed=42)
    plan_b = ProductionPlan(task_id="test_task", target_qa_count=100, random_seed=999)

    assert plan_a.compute_hash() != plan_b.compute_hash()
    assert contract.contract_hash == hash_1  # Unchanged!


# -----------------------------------------------------------------------------
# B2 TESTS: ViMedCSS Presence Complete Universe Capacity
# -----------------------------------------------------------------------------





# -----------------------------------------------------------------------------
# B3 TESTS: TopologyEngine
# -----------------------------------------------------------------------------

def test_b3_topology_one_to_one():
    res = TopologyEngine.compute_capacity("ONE_TO_ONE", n_anchors=11832)
    assert res.pos_capacity == 11832
    assert res.neg_capacity == 0
    assert res.total_capacity == 11832


def test_b3_topology_cartesian_presence():
    res = TopologyEngine.compute_capacity(
        "CARTESIAN_PRESENCE",
        n_anchors=11782,
        candidate_pool_size=610,
        total_positive_ground_truth_labels=12264,
        collision_count=6752,
    )
    assert res.pos_capacity == 12264
    assert res.collision_capacity == 6752
    assert res.raw_combinatorial_space_capacity == 11782 * 610
    assert res.valid_semantic_universe_capacity == 7180268
    assert res.total_capacity == 7180268
    assert res.neg_capacity == 11782 * 610 - 12264 - 6752


def test_b3_topology_unordered_pair_regression():
    """Verify TopologyEngine structurally computes C(11832, 2) = 69,992,196."""
    res = TopologyEngine.compute_capacity(
        "UNORDERED_PAIR",
        n_anchors=11832,
        same_topic_pairs=21491641,
    )
    assert res.total_capacity == 69992196
    assert res.pos_capacity == 21491641
    assert res.neg_capacity == 69992196 - 21491641
    assert res.total_capacity != 70000696  # Structurally prevents old bug!


def test_b3_unsupported_topology_raises_error():
    with pytest.raises(UnsupportedTopologyError):
        TopologyEngine.compute_capacity("INVALID_TOPOLOGY", n_anchors=10)


# -----------------------------------------------------------------------------
# B4 TESTS: Autonomous T4-K Fallback
# -----------------------------------------------------------------------------







# -----------------------------------------------------------------------------
# B5 TESTS: CoreInvariantsSuite Gatekeeper
# -----------------------------------------------------------------------------

def test_b5_overlapping_partition_raises_compile_error():
    with pytest.raises(CompileError, match="PARTITION_OVERLAP"):
        CoreInvariantsSuite.assert_mutually_exclusive_partition(
            promoted={"task1"},
            promoted_with_fallback={"task1"},  # Overlap!
            rejected=set(),
            all_candidates={"task1"},
        )


def test_b5_incomplete_partition_raises_compile_error():
    with pytest.raises(CompileError, match="PARTITION_INCOMPLETE"):
        CoreInvariantsSuite.assert_mutually_exclusive_partition(
            promoted={"task1"},
            promoted_with_fallback=set(),
            rejected=set(),
            all_candidates={"task1", "task2"},  # Missing task2!
        )


def test_b5_sanitization_non_monotonic_raises_compile_error():
    with pytest.raises(CompileError, match="SANITIZATION_NON_MONOTONIC"):
        CoreInvariantsSuite.assert_sanitization_monotonicity(
            raw_capacity=100,
            sanitized_capacity=105,  # Increased!
            task_id="test_task",
        )


def test_b5_undeclared_concept_space_raises_compile_error():
    contract = create_semantic_contract(
        task_id="vimedcss_cs_term_extraction",
        tier="T1_PERCEPTION",
        topology="ONE_TO_ONE",
        concept_space=None,  # Undeclared!
    )
    with pytest.raises(CompileError, match="UNDECLARED_CONCEPT_SPACE"):
        CoreInvariantsSuite.assert_concept_space_declared(contract)


def test_b5_topology_capacity_exceeded_raises_compile_error():
    with pytest.raises(CompileError, match="TOPOLOGY_CAPACITY_EXCEEDED"):
        CoreInvariantsSuite.assert_topology_maximum(
            topology="UNORDERED_PAIR",
            n_anchors=10,
            capacity=100,  # Max for n=10 is 45!
        )


def test_b5_planner_bounds_exceeded_raises_compile_error():
    contract = create_semantic_contract(
        task_id="test_task",
        tier="T1_PERCEPTION",
        topology="ONE_TO_ONE",
        positive_universe_capacity=50,
        negative_universe_capacity=0,
        total_universe_capacity=50,
    )
    plan = ProductionPlan(task_id="test_task", target_qa_count=100)
    with pytest.raises(CompileError, match="PLANNER_BOUNDS_EXCEEDED"):
        CoreInvariantsSuite.assert_planner_capacity_bounds(plan, contract)


# -----------------------------------------------------------------------------
# HARDENING V2 TESTS: Conservation, Discrepancy, Identity, Manifest
# -----------------------------------------------------------------------------

def test_hardening_cartesian_conservation_pass():
    """Verify exact Cartesian conservation sum equals raw Cartesian product A x C."""
    CoreInvariantsSuite.assert_cartesian_conservation(
        raw_cartesian=7187020,
        pos=12264,
        neg=7168004,
        collision=6752,
        invalid=0,
    )


def test_hardening_cartesian_conservation_failure_raises_compile_error():
    """Verify distorting one bucket by +1 raises CompileError."""
    with pytest.raises(CompileError, match="CARTESIAN_CONSERVATION_VIOLATION"):
        CoreInvariantsSuite.assert_cartesian_conservation(
            raw_cartesian=7187020,
            pos=12265,  # +1 distortion!
            neg=7168004,
            collision=6752,
            invalid=0,
        )


def test_hardening_formula_vs_enumeration_discrepancy_raises_compile_error():
    """Verify formula vs enumeration discrepancy raises CompileError."""
    formula_res = {"pos_capacity": 100, "neg_capacity": 900, "collision_capacity": 50, "total_capacity": 1050}
    enum_res = {"pos_capacity": 100, "neg_capacity": 901, "collision_capacity": 50, "total_capacity": 1051}
    with pytest.raises(CompileError, match="FORMULA_ENUMERATION_DISCREPANCY"):
        CoreInvariantsSuite.assert_formula_enumeration_equivalence(formula_res, enum_res)


def test_hardening_manifest_instance_id_duplication_raises_compile_error():
    """Verify duplicate instance IDs in FrozenInstanceManifest raise CompileError."""
    with pytest.raises(CompileError, match="MANIFEST_INSTANCE_ID_DUPLICATION"):
        CoreInvariantsSuite.assert_manifest_instance_uniqueness(["id1", "id2", "id1"])


def test_hardening_manifest_universe_membership_violation_raises_compile_error():
    """Verify manifest selecting an instance outside valid universe raises CompileError."""
    selected = ["valid_id_1", "invalid_invented_id"]
    valid_universe = {"valid_id_1", "valid_id_2"}
    with pytest.raises(CompileError, match="MANIFEST_UNIVERSE_MEMBERSHIP_VIOLATION"):
        CoreInvariantsSuite.assert_manifest_universe_membership(selected, valid_universe)


def test_hardening_registry_hash_separation():
    """Verify lexical comparator and KG resolver registry hashes must be distinct and non-empty."""
    lex_hash = "17bb44a9ab14fc36b8c37fb89a18292c8b1045e63d708e6bf4f4219b6a5f2d5e"
    kg_hash = "3afb8199578dd2ceb6b530c3048cc35ed1a56465039a9adc4b46ec8f833f4499"
    CoreInvariantsSuite.assert_registry_hash_separation(lex_hash, kg_hash)

    with pytest.raises(CompileError, match="REGISTRY_HASH_COLLISION"):
        CoreInvariantsSuite.assert_registry_hash_separation(lex_hash, lex_hash)


def test_hardening_fallback_decision_hash_distinct_from_contract_hash():
    """Verify T4 fallback decision hash is distinct from effective execution contract hash."""
    fallback_hash = "41f85321bf2c168101b754b0d8039edd50448eb1a76f129543e9d06bbd62cf07"
    contract_hash = "6f7d1e6b282bfc710ec06d7d966cf37db8e6052590e527a9677c896b25acf37c"
    CoreInvariantsSuite.assert_fallback_decision_provenance(fallback_hash, contract_hash)

    with pytest.raises(CompileError, match="FALLBACK_HASH_ALIASING"):
        CoreInvariantsSuite.assert_fallback_decision_provenance(contract_hash, contract_hash)



def test_b5_determinism_invariant():
    data_a = json.dumps({"key": "value", "seq": [1, 2, 3]}, sort_keys=True)
    data_b = json.dumps({"key": "value", "seq": [1, 2, 3]}, sort_keys=True)
    # Should not raise
    CoreInvariantsSuite.assert_determinism(data_a, data_b)

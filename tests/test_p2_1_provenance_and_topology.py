"""Tests for P2.1 Source-Universe Provenance and Topology Compatibility Invariants."""

from __future__ import annotations

import pytest
from src.autonomous_qa.certification.core_invariants import CompileError, CoreInvariantsSuite
from src.autonomous_qa.compiler.semantic_contract import create_semantic_contract
from src.autonomous_qa.core.topology_engine import TopologyEngine


def test_semantic_universe_source_provenance_invariant() -> None:
    """Forbidden production artifacts must trigger CompileError."""
    # Valid source provenance
    CoreInvariantsSuite.assert_semantic_universe_source_provenance("SOURCE_DATASET")
    CoreInvariantsSuite.assert_semantic_universe_source_provenance("SOURCE_PROFILE")

    # Forbidden production artifacts
    forbidden_kinds = [
        "HISTORICAL_QA_RELEASE",
        "HISTORICAL_GENERATION_PLAN",
        "PRODUCTION_PLAN",
        "FROZEN_INSTANCE_MANIFEST",
    ]
    for kind in forbidden_kinds:
        with pytest.raises(CompileError, match="FORBIDDEN_SOURCE_PROVENANCE_KIND"):
            CoreInvariantsSuite.assert_semantic_universe_source_provenance(kind)


def test_vimd_pair_capacity_arithmetic_exactness() -> None:
    """Verify exact formula C(13344, 2) == 89024496 (not legacy 89024928)."""
    res = TopologyEngine.compute_capacity("UNORDERED_PAIR", n_anchors=13344)
    expected_pairs = 13344 * 13343 // 2
    assert expected_pairs == 89024496
    assert res.total_capacity == 89024496
    assert res.total_capacity != 89024928


def test_topology_role_compatibility_invariant() -> None:
    """Contract role mapping must match topology structure."""
    # Valid UNORDERED_PAIR contract
    valid_pair = create_semantic_contract(
        task_id="valid_pair_task",
        topology="UNORDERED_PAIR",
        anchor_type="audio_segment",
        visible_roles=("audio",),
    )
    CoreInvariantsSuite.assert_topology_role_compatibility(valid_pair)

    # Invalid UNORDERED_PAIR contract with non-audio anchor domain
    invalid_pair = create_semantic_contract(
        task_id="invalid_pair_task",
        topology="UNORDERED_PAIR",
        anchor_type="text_concept",
        visible_roles=("text",),
    )
    with pytest.raises(CompileError, match="TOPOLOGY_ROLE_MISMATCH"):
        CoreInvariantsSuite.assert_topology_role_compatibility(invalid_pair)


def test_topology_usage_aggregation_invariant() -> None:
    """Sum of topology usage counts across datasets must equal total executable contracts (22)."""
    topology_counts = {
        "ONE_TO_ONE": 9,
        "UNORDERED_PAIR": 9,
        "CARTESIAN_PRESENCE": 4,
    }
    # Valid aggregation
    CoreInvariantsSuite.assert_topology_usage_aggregation(
        topology_counts, total_executable_contracts=22
    )

    # Mismatched aggregation raises CompileError
    with pytest.raises(CompileError, match="TOPOLOGY_USAGE_AGGREGATION_DISCREPANCY"):
        CoreInvariantsSuite.assert_topology_usage_aggregation(
            topology_counts, total_executable_contracts=20
        )


def test_separate_lanes_for_source_and_historical_replay() -> None:
    """Raw source count and historical plan selection count must remain separate metrics."""
    raw_source_count = 15023
    historical_plan_count = 13344
    assert raw_source_count != historical_plan_count

    # Capacities must derive strictly from raw_source_count
    source_cap = TopologyEngine.compute_capacity("ONE_TO_ONE", n_anchors=raw_source_count)
    assert source_cap.pos_capacity == 15023
    assert source_cap.pos_capacity != 13344

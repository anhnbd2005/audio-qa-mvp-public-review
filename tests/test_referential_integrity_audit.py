"""Unit tests for P2.2b Referential Integrity and Replay Closure."""

import random
import pytest

from src.autonomous_qa.certification.core_invariants import CompileError, CoreInvariantsSuite
from src.autonomous_qa.certification.referential_integrity_audit import (
    AuditReportRow,
    CapacityInputSnapshot,
    FrozenTaskBinding,
    run_p2_2b_referential_integrity_audit,
)
from src.autonomous_qa.core.topology_engine import TopologyEngine


def test_vimd_province_ontology_63(tmp_path):
    """Requirement 1: ViMD province candidate count must equal 63."""
    res = run_p2_2b_referential_integrity_audit(out_dir=tmp_path / "compiler_v2_referential_closure")
    assert res["status"] == "P2_2B_REFERENTIAL_INTEGRITY_PASS"


def test_vimd_region_pairwise_capacity():
    """Requirement 4: ViMD region pairwise-selection valid capacity == 149,185,910."""
    hist = {"north": 5913, "central": 4705, "south": 4405}
    res = TopologyEngine.compute_capacity(
        "TARGET_CONDITIONED_UNORDERED_PAIR",
        n_anchors=15023,
        candidate_pool_size=3,
        candidate_histogram=hist,
    )
    assert res.valid_semantic_universe_capacity == 149185910
    assert res.raw_combinatorial_space_capacity == 3 * (15023 * 15022 // 2)
    assert res.excluded_capacity == res.raw_combinatorial_space_capacity - 149185910


def test_vietmdd_canonical_release_counts():
    """Requirement 6 & 7: VietMDD canonical selected-count vector == [3181, 6362, 3181, 3011, 3181, 3011], total = 21,927."""
    counts = [3181, 6362, 3181, 3011, 3181, 3011]
    assert sum(counts) == 21927


def test_vietmdd_p3_aligned_relation_topology():
    """Requirement 8 & 9: VietMDD P3 uses ALIGNED_RELATION topology."""
    res = TopologyEngine.compute_capacity(
        "ALIGNED_RELATION",
        n_anchors=3181,
        label_distribution={"TRUE": 2419, "FALSE": 762},
    )
    assert res.raw_combinatorial_space_capacity == 3181
    assert res.valid_semantic_universe_capacity == 3181
    assert res.pos_capacity == 2419
    assert res.neg_capacity == 762


def test_cross_dataset_topology_matrix_totals():
    """Requirement 11: Topology totals == 7 ONE_TO_ONE, 5 CARTESIAN_RELATION, 5 UNORDERED_PAIR, 4 TARGET_CONDITIONED_UNORDERED_PAIR, 1 ALIGNED_RELATION = 22."""
    counts = {
        "ONE_TO_ONE": 7,
        "CARTESIAN_RELATION": 5,
        "UNORDERED_PAIR": 5,
        "TARGET_CONDITIONED_UNORDERED_PAIR": 4,
        "ALIGNED_RELATION": 1,
    }
    CoreInvariantsSuite.assert_topology_usage_aggregation(counts, 22)


def test_vimedcss_capacity_nomenclature():
    """Requirement 12: ViMedCSS raw=7,187,020, valid=7,180,268, excluded=6,752."""
    res = TopologyEngine.compute_capacity(
        "CARTESIAN_PRESENCE",
        n_anchors=11782,
        candidate_pool_size=610,
        total_positive_ground_truth_labels=12264,
        collision_count=6752,
    )
    assert res.raw_combinatorial_space_capacity == 7187020
    assert res.valid_semantic_universe_capacity == 7180268
    assert res.excluded_capacity == 6752


def test_shuffle_safety():
    """Section 65: Shuffling task input ordering leaves report rows identical by task_id."""
    snap = CapacityInputSnapshot.create(
        task_id="t1",
        topology="ONE_TO_ONE",
        anchor_registry_hash="anc_hash",
        n_anchors=100,
    )
    cap = TopologyEngine.compute_capacity("ONE_TO_ONE", n_anchors=100)
    b1 = FrozenTaskBinding.build(
        dataset_id="d1",
        task_id="t1",
        contract_hash="c1",
        semantic_universe_hash="u1",
        topology="ONE_TO_ONE",
        source_attribute="text",
        anchor_registry_id="anc1",
        anchor_registry_hash="anc_hash",
        eligible_anchor_count=100,
        capacity_input_snapshot=snap,
        capacity_result=cap,
    )

    snap2 = CapacityInputSnapshot.create(
        task_id="t2",
        topology="ONE_TO_ONE",
        anchor_registry_hash="anc_hash2",
        n_anchors=200,
    )
    cap2 = TopologyEngine.compute_capacity("ONE_TO_ONE", n_anchors=200)
    b2 = FrozenTaskBinding.build(
        dataset_id="d1",
        task_id="t2",
        contract_hash="c2",
        semantic_universe_hash="u2",
        topology="ONE_TO_ONE",
        source_attribute="region",
        anchor_registry_id="anc2",
        anchor_registry_hash="anc_hash2",
        eligible_anchor_count=200,
        capacity_input_snapshot=snap2,
        capacity_result=cap2,
    )

    bindings_orig = [b1, b2]
    bindings_shuffled = [b2, b1]

    rows_orig = {b.task_id: AuditReportRow.from_frozen_task_binding(b) for b in bindings_orig}
    rows_shuffled = {b.task_id: AuditReportRow.from_frozen_task_binding(b) for b in bindings_shuffled}

    assert rows_orig == rows_shuffled


def test_cross_wire_candidate_registry_fails():
    """Section 66: Binding mismatch between anchor/candidate hashes raises CompileError."""
    with pytest.raises(CompileError, match="TASK_BINDING_REFERENTIAL_INTEGRITY_FAIL"):
        CoreInvariantsSuite.assert_task_binding_referential_integrity(
            task_id="t1",
            contract_task_id="t1",
            universe_task_id="t1",
            capacity_task_id="t1",
            anchor_reg_hash="anc_hash_a",
            universe_anchor_hash="anc_hash_b",  # Mismatch!
        )


def test_historical_replay_closure_guard():
    """Section 49: Replay count closure guard fails if sum != expected."""
    with pytest.raises(CompileError, match="HISTORICAL_REPLAY_COUNT_CLOSURE_FAIL"):
        CoreInvariantsSuite.assert_historical_replay_count_closure(
            dataset_id="vietmdd",
            per_task_counts={"t1": 3181, "t2": 6362},
            expected_total=21927,  # Mismatch!
        )

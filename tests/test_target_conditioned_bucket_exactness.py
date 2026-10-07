"""Unit tests for Target-Conditioned Bucket Exactness Closure (P2.2c)."""

import pytest
from src.autonomous_qa.certification.core_invariants import CompileError, CoreInvariantsSuite
from src.autonomous_qa.core.topology_engine import CandidatePairBucket, CapacityResult, TopologyEngine


def test_cancelling_error_synthetic_fails():
    """Section 24: Aggregate conservation alone passes if both/neither cancel out, but bucket exactness invariant fails."""
    hist = {"north": 5913, "central": 4705, "south": 4405}
    cap_res = TopologyEngine.compute_capacity(
        "TARGET_CONDITIONED_UNORDERED_PAIR",
        n_anchors=15023,
        candidate_pool_size=3,
        candidate_histogram=hist,
    )

    # Sanity check: valid baseline passes
    CoreInvariantsSuite.assert_target_conditioned_bucket_exactness(
        cap_res.candidate_buckets, cap_res
    )

    # Mutate capacity_result collision_capacity (both) and invalid_capacity (neither) by offset 10
    mutated_cap = CapacityResult(
        topology=cap_res.topology,
        pos_capacity=cap_res.pos_capacity,
        neg_capacity=cap_res.neg_capacity,
        collision_capacity=cap_res.collision_capacity - 10,  # Mutated!
        invalid_capacity=cap_res.invalid_capacity + 10,      # Mutated!
        total_capacity=cap_res.total_capacity,
        raw_combinatorial_space_capacity=cap_res.raw_combinatorial_space_capacity,
        valid_semantic_universe_capacity=cap_res.valid_semantic_universe_capacity,
        excluded_capacity=cap_res.excluded_capacity,
        label_capacities=cap_res.label_capacities,
        exclusion_capacities=cap_res.exclusion_capacities,
        candidate_buckets=cap_res.candidate_buckets,
        metadata=cap_res.metadata,
    )

    # Aggregate conservation (raw == valid + excluded) still passes!
    assert mutated_cap.raw_combinatorial_space_capacity == mutated_cap.valid_semantic_universe_capacity + mutated_cap.excluded_capacity

    # But TARGET_CONDITIONED_BUCKET_EXACTNESS fails as required!
    with pytest.raises(CompileError, match="TARGET_CONDITIONED_BUCKET_EXACTNESS_FAIL"):
        CoreInvariantsSuite.assert_target_conditioned_bucket_exactness(
            cap_res.candidate_buckets, mutated_cap
        )


def test_region_exact_numbers():
    """Section 25: Test exact ViMD region aggregate target-conditioned numbers."""
    hist = {"north": 5913, "central": 4705, "south": 4405}
    res = TopologyEngine.compute_capacity(
        "TARGET_CONDITIONED_UNORDERED_PAIR",
        n_anchors=15023,
        candidate_pool_size=3,
        candidate_histogram=hist,
    )

    assert res.raw_combinatorial_space_capacity == 338513259
    assert res.valid_semantic_universe_capacity == 149185910
    assert res.collision_capacity == 38244798   # both_match_excluded_capacity
    assert res.invalid_capacity == 151082551    # neither_match_excluded_capacity
    assert res.excluded_capacity == 189327349
    assert res.collision_capacity + res.invalid_capacity == res.excluded_capacity
    assert res.valid_semantic_universe_capacity + res.excluded_capacity == res.raw_combinatorial_space_capacity


def test_each_region_buckets():
    """Section 26: Test exact individual candidate buckets for north, central, south."""
    hist = {"north": 5913, "central": 4705, "south": 4405}
    res = TopologyEngine.compute_capacity(
        "TARGET_CONDITIONED_UNORDERED_PAIR",
        n_anchors=15023,
        candidate_pool_size=3,
        candidate_histogram=hist,
    )

    bucket_map = {b.candidate_id: b for b in res.candidate_buckets}

    north = bucket_map["north"]
    assert north.valid_pair_capacity == 53867430
    assert north.both_match_excluded_capacity == 17478828
    assert north.neither_match_excluded_capacity == 41491495

    central = bucket_map["central"]
    assert central.valid_pair_capacity == 48546190
    assert central.both_match_excluded_capacity == 11066160
    assert central.neither_match_excluded_capacity == 53225403

    south = bucket_map["south"]
    assert south.valid_pair_capacity == 46772290
    assert south.both_match_excluded_capacity == 9699810
    assert south.neither_match_excluded_capacity == 56365653


def test_text_target_conditioned_buckets():
    """Section 27: ViMD text pairwise selection exact bucket math."""
    hist = {f"text_{i}": 1 for i in range(15023)}
    res = TopologyEngine.compute_capacity(
        "TARGET_CONDITIONED_UNORDERED_PAIR",
        n_anchors=15023,
        candidate_pool_size=15023,
        candidate_histogram=hist,
    )

    assert res.raw_combinatorial_space_capacity == 1695161563319
    assert res.valid_semantic_universe_capacity == 225675506
    assert res.collision_capacity == 0           # both_match_excluded == 0
    assert res.invalid_capacity == 1694935887813  # neither_match_excluded
    assert res.excluded_capacity == 1694935887813


def test_province_target_conditioned_buckets():
    """Section 28: ViMD province pairwise selection derived from 63-class histogram."""
    province_identities = [f"province_{i}" for i in range(63)]
    province_hist = {p: 238 for p in province_identities}
    for i in range(29):
        province_hist[province_identities[i]] += 1
    assert sum(province_hist.values()) == 15023

    res = TopologyEngine.compute_capacity(
        "TARGET_CONDITIONED_UNORDERED_PAIR",
        n_anchors=15023,
        candidate_pool_size=63,
        candidate_histogram=province_hist,
    )

    assert res.raw_combinatorial_space_capacity == 7108778439
    assert res.valid_semantic_universe_capacity == 222108124
    assert res.collision_capacity == 1783691
    assert res.invalid_capacity == 6884886624
    assert res.excluded_capacity == 6886670315
    assert res.raw_combinatorial_space_capacity == res.valid_semantic_universe_capacity + res.excluded_capacity

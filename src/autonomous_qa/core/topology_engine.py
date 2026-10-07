"""Centralized Topology Engine for Autonomous Compiler Capacity Math.

No candidate module or dataset adapter may independently implement its own
combinatorial formulas. All compiler capacity math flows through TopologyEngine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TopologyType = Literal[
    "ONE_TO_ONE",
    "CARTESIAN_PRESENCE",
    "CARTESIAN_RELATION",
    "UNORDERED_PAIR",
    "TARGET_CONDITIONED_UNORDERED_PAIR",
    "ALIGNED_RELATION",
]


class UnsupportedTopologyError(ValueError):
    """Raised when an unknown or unsupported topology is requested."""


@dataclass(frozen=True)
class CandidatePairBucket:
    candidate_id: str
    positive_anchor_count: int
    negative_anchor_count: int
    raw_pair_capacity: int
    valid_pair_capacity: int
    both_match_excluded_capacity: int
    neither_match_excluded_capacity: int

    @classmethod
    def compute(
        cls, candidate_id: str, positive_anchor_count: int, total_anchors: int
    ) -> CandidatePairBucket:
        p_c = positive_anchor_count
        n_c = total_anchors - p_c
        if p_c < 0 or n_c < 0:
            raise ValueError(f"INVALID_ANCHOR_COUNTS:{p_c},{n_c}")

        valid = p_c * n_c
        both = (p_c * (p_c - 1) // 2) if p_c >= 2 else 0
        neither = (n_c * (n_c - 1) // 2) if n_c >= 2 else 0
        raw = total_anchors * (total_anchors - 1) // 2 if total_anchors >= 2 else 0

        if raw != valid + both + neither:
            raise ValueError(
                f"TARGET_CONDITIONED_PER_CANDIDATE_CONSERVATION_FAIL:{candidate_id}: "
                f"raw ({raw}) != valid ({valid}) + both ({both}) + neither ({neither})"
            )

        return cls(
            candidate_id=candidate_id,
            positive_anchor_count=p_c,
            negative_anchor_count=n_c,
            raw_pair_capacity=raw,
            valid_pair_capacity=valid,
            both_match_excluded_capacity=both,
            neither_match_excluded_capacity=neither,
        )


@dataclass(frozen=True)
class CapacityResult:
    topology: TopologyType
    pos_capacity: int
    neg_capacity: int
    collision_capacity: int = 0
    invalid_capacity: int = 0
    total_capacity: int = 0
    raw_combinatorial_space_capacity: int = 0
    valid_semantic_universe_capacity: int = 0
    excluded_capacity: int = 0
    label_capacities: dict[str, int] = field(default_factory=dict)
    exclusion_capacities: dict[str, int] = field(default_factory=dict)
    candidate_buckets: list[CandidatePairBucket] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Enforce conservation invariant raw == valid + excluded
        if self.raw_combinatorial_space_capacity > 0:
            expected_raw = (
                self.valid_semantic_universe_capacity + self.excluded_capacity
            )
            if self.raw_combinatorial_space_capacity != expected_raw:
                raise ValueError(
                    f"CAPACITY_CONSERVATION_ERROR: raw ({self.raw_combinatorial_space_capacity}) != "
                    f"valid ({self.valid_semantic_universe_capacity}) + excluded ({self.excluded_capacity})"
                )


class TopologyEngine:
    """Pure deterministic capacity math engine (zero LLM, zero sampling, zero RNG)."""

    @staticmethod
    def compute_capacity(
        topology: TopologyType,
        n_anchors: int,
        candidate_pool_size: int = 0,
        **kwargs: Any,
    ) -> CapacityResult:
        if n_anchors < 0:
            raise ValueError(f"INVALID_ANCHOR_COUNT:{n_anchors}")

        match topology:
            case "ONE_TO_ONE":
                pos = n_anchors
                neg = 0
                total = pos + neg
                return CapacityResult(
                    topology=topology,
                    pos_capacity=pos,
                    neg_capacity=neg,
                    collision_capacity=0,
                    invalid_capacity=0,
                    total_capacity=total,
                    raw_combinatorial_space_capacity=total,
                    valid_semantic_universe_capacity=total,
                    excluded_capacity=0,
                    label_capacities={"positive": pos, "negative": neg},
                    exclusion_capacities={"collision": 0, "invalid": 0},
                    metadata={"n_anchors": n_anchors},
                )

            case "ALIGNED_RELATION":
                # One source anchor aligned 1-to-1 with its own source-bound reference counterpart
                label_dist = kwargs.get("label_distribution", {})
                true_count = int(label_dist.get("TRUE", kwargs.get("pos_capacity", n_anchors)))
                false_count = int(label_dist.get("FALSE", kwargs.get("neg_capacity", 0)))
                total = n_anchors
                return CapacityResult(
                    topology=topology,
                    pos_capacity=true_count,
                    neg_capacity=false_count,
                    collision_capacity=0,
                    invalid_capacity=0,
                    total_capacity=total,
                    raw_combinatorial_space_capacity=total,
                    valid_semantic_universe_capacity=total,
                    excluded_capacity=0,
                    label_capacities={"TRUE": true_count, "FALSE": false_count},
                    exclusion_capacities={"collision": 0, "invalid": 0},
                    metadata={
                        "n_anchors": n_anchors,
                        "label_distribution": label_dist,
                    },
                )

            case "CARTESIAN_PRESENCE" | "CARTESIAN_RELATION":
                pos_count = int(kwargs.get("total_positive_ground_truth_labels", 0))
                collision_count = int(kwargs.get("collision_count", 0))
                invalid_count = int(kwargs.get("invalid_count", 0))
                raw_cartesian = n_anchors * candidate_pool_size

                neg_count = raw_cartesian - pos_count - collision_count - invalid_count
                if neg_count < 0:
                    # If total_positive_ground_truth_labels not passed explicitly, assume pos_count = n_anchors
                    if pos_count == 0:
                        pos_count = n_anchors
                        neg_count = max(0, raw_cartesian - pos_count - collision_count - invalid_count)
                    else:
                        raise ValueError(
                            f"NEGATIVE_CAPACITY_ERROR: raw_cartesian={raw_cartesian}, pos={pos_count}, collision={collision_count}, invalid={invalid_count}"
                        )

                excluded = collision_count + invalid_count
                valid = pos_count + neg_count

                if (valid + excluded) != raw_cartesian:
                    raise ValueError(
                        f"CARTESIAN_CONSERVATION_ERROR: sum={valid + excluded} != raw_cartesian={raw_cartesian}"
                    )

                return CapacityResult(
                    topology=topology,
                    pos_capacity=pos_count,
                    neg_capacity=neg_count,
                    collision_capacity=collision_count,
                    invalid_capacity=invalid_count,
                    total_capacity=valid,
                    raw_combinatorial_space_capacity=raw_cartesian,
                    valid_semantic_universe_capacity=valid,
                    excluded_capacity=excluded,
                    label_capacities={"positive": pos_count, "negative": neg_count},
                    exclusion_capacities={
                        "collision": collision_count,
                        "invalid": invalid_count,
                    },
                    metadata={
                        "n_anchors": n_anchors,
                        "candidate_pool_size": candidate_pool_size,
                        "raw_cartesian": raw_cartesian,
                        "pos_count": pos_count,
                        "collision_count": collision_count,
                        "invalid_count": invalid_count,
                        "neg_count": neg_count,
                        "conservation_verified": True,
                    },
                )

            case "UNORDERED_PAIR":
                # Mathematical formula for distinct unordered pairs: C(n, 2) = n*(n-1)/2
                total_pairs = n_anchors * (n_anchors - 1) // 2 if n_anchors >= 2 else 0
                same_topic = int(kwargs.get("same_topic_pairs", total_pairs))
                diff_topic = int(kwargs.get("different_topic_pairs", 0))
                if "same_topic_pairs" in kwargs and "different_topic_pairs" not in kwargs:
                    diff_topic = max(0, total_pairs - same_topic)

                return CapacityResult(
                    topology=topology,
                    pos_capacity=same_topic,
                    neg_capacity=diff_topic,
                    collision_capacity=0,
                    invalid_capacity=0,
                    total_capacity=total_pairs,
                    raw_combinatorial_space_capacity=total_pairs,
                    valid_semantic_universe_capacity=total_pairs,
                    excluded_capacity=0,
                    label_capacities={"same": same_topic, "different": diff_topic},
                    exclusion_capacities={"collision": 0, "invalid": 0},
                    metadata={
                        "n_anchors": n_anchors,
                        "combinatorial_formula": "C(n, 2) = n*(n-1)/2",
                        "total_unordered_pairs": total_pairs,
                        "same_topic_pairs": same_topic,
                        "different_topic_pairs": diff_topic,
                    },
                )

            case "TARGET_CONDITIONED_UNORDERED_PAIR":
                # Exact CandidatePairBucket aggregation:
                candidate_histogram = kwargs.get("candidate_histogram")
                if candidate_histogram and isinstance(candidate_histogram, dict):
                    buckets = []
                    for c, p_c in candidate_histogram.items():
                        bucket = CandidatePairBucket.compute(
                            candidate_id=str(c),
                            positive_anchor_count=p_c,
                            total_anchors=n_anchors,
                        )
                        buckets.append(bucket)

                    raw_space = sum(b.raw_pair_capacity for b in buckets)
                    valid_universe = sum(b.valid_pair_capacity for b in buckets)
                    both_match_excluded = sum(b.both_match_excluded_capacity for b in buckets)
                    neither_match_excluded = sum(b.neither_match_excluded_capacity for b in buckets)
                    excluded_space = both_match_excluded + neither_match_excluded

                    if raw_space != valid_universe + excluded_space:
                        raise ValueError(
                            f"TARGET_CONDITIONED_AGGREGATE_CONSERVATION_FAIL: raw ({raw_space}) != "
                            f"valid ({valid_universe}) + excluded ({excluded_space})"
                        )

                    return CapacityResult(
                        topology=topology,
                        pos_capacity=valid_universe,
                        neg_capacity=0,
                        collision_capacity=both_match_excluded,
                        invalid_capacity=neither_match_excluded,
                        total_capacity=valid_universe,
                        raw_combinatorial_space_capacity=raw_space,
                        valid_semantic_universe_capacity=valid_universe,
                        excluded_capacity=excluded_space,
                        label_capacities={"valid_selection_pairs": valid_universe},
                        exclusion_capacities={
                            "both_match_excluded": both_match_excluded,
                            "neither_match_excluded": neither_match_excluded,
                        },
                        candidate_buckets=buckets,
                        metadata={
                            "n_anchors": n_anchors,
                            "candidate_pool_size": len(buckets),
                            "combinatorial_formula": "sum_c Bucket.compute(c, P_c, N)",
                            "candidate_bucket_exactness_verified": True,
                        },
                    )

                # Fallback when histogram is not directly provided
                target_conditioned_pairs = int(kwargs.get("target_conditioned_pairs", 0))
                if target_conditioned_pairs == 0 and candidate_pool_size > 0:
                    pos_per_c = max(1, n_anchors // candidate_pool_size)
                    neg_per_c = n_anchors - pos_per_c
                    target_conditioned_pairs = (
                        candidate_pool_size * pos_per_c * neg_per_c
                    )

                total_pairs_per_cand = n_anchors * (n_anchors - 1) // 2 if n_anchors >= 2 else 0
                raw_space = candidate_pool_size * total_pairs_per_cand if candidate_pool_size > 0 else target_conditioned_pairs
                excluded_space = max(0, raw_space - target_conditioned_pairs)

                return CapacityResult(
                    topology=topology,
                    pos_capacity=target_conditioned_pairs,
                    neg_capacity=0,
                    collision_capacity=excluded_space,
                    invalid_capacity=0,
                    total_capacity=target_conditioned_pairs,
                    raw_combinatorial_space_capacity=raw_space,
                    valid_semantic_universe_capacity=target_conditioned_pairs,
                    excluded_capacity=excluded_space,
                    label_capacities={"valid_selection_pairs": target_conditioned_pairs},
                    exclusion_capacities={"excluded_pairs": excluded_space},
                    metadata={
                        "n_anchors": n_anchors,
                        "candidate_pool_size": candidate_pool_size,
                        "combinatorial_formula": "sum(P_c * N_c)",
                        "target_conditioned_pairs": target_conditioned_pairs,
                    },
                )

            case _:
                raise UnsupportedTopologyError(f"UNSUPPORTED_TOPOLOGY:{topology}")

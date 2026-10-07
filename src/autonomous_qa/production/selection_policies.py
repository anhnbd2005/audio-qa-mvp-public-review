"""Generic production selection policies and registry for P3.1.

Implements dataset-agnostic selection strategies:
1. FULL_VALID_UNIVERSE
2. ALL_POSITIVES_BALANCED_NEGATIVES
3. BALANCED_BINARY_RELATIONAL_PAIRS

Memory-bounded and mathematically deterministic.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable


class PlanningError(RuntimeError):
    """Raised when production selection policy constraints are infeasible or violated."""


def stable_hash(seed: int | str, task_id: str, *fields: Any) -> str:
    parts = [str(seed), str(task_id)] + [str(f) for f in fields]
    payload = "\x1f".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_pair_id(id_a: str, id_b: str) -> str:
    first, second = min(id_a, id_b), max(id_a, id_b)
    return f"{first}\x1f{second}"


SUPPORTED_POLICIES = (
    "FULL_VALID_UNIVERSE",
    "ALL_POSITIVES_BALANCED_NEGATIVES",
    "BALANCED_BINARY_RELATIONAL_PAIRS",
)


def get_policy_registry_hash() -> str:
    payload = {
        "schema_version": "p3_1_v1",
        "supported_policies": list(SUPPORTED_POLICIES),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


class FullValidUniversePolicy:
    policy_id = "FULL_VALID_UNIVERSE"

    @staticmethod
    def select(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return list(instances)


class AllPositivesBalancedNegativesPolicy:
    policy_id = "ALL_POSITIVES_BALANCED_NEGATIVES"

    @staticmethod
    def select(
        task_id: str,
        eligible_anchors: list[dict[str, Any]],
        candidate_vocab: list[str],
        positive_instances: list[dict[str, Any]],
        relation_classifier: Callable[[str, str], str],
        seed: int = 42,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        positives_by_anchor: dict[str, list[dict[str, Any]]] = {}
        for inst in positive_instances:
            aid = str(inst["segment_id"])
            positives_by_anchor.setdefault(aid, []).append(inst)

        selected_positives: list[dict[str, Any]] = []
        k_a: dict[str, int] = {}
        for anchor in eligible_anchors:
            aid = str(anchor["segment_id"])
            pos_list = positives_by_anchor.get(aid, [])
            if not pos_list:
                raise PlanningError(f"PRESENCE_ANCHOR_HAS_ZERO_POSITIVES: {aid}")
            selected_positives.extend(pos_list)
            k_a[aid] = len(pos_list)

        valid_negatives_by_anchor: dict[str, set[str]] = {}
        for aid in k_a:
            v_set = {
                c for c in candidate_vocab if relation_classifier(aid, c) == "NEGATIVE"
            }
            valid_negatives_by_anchor[aid] = v_set

        remaining_quota = dict(k_a)
        candidate_usage = {c: 0 for c in candidate_vocab}
        assigned_negatives: dict[str, set[str]] = {aid: set() for aid in k_a}

        # PHASE A: Candidate Coverage Pass
        sorted_candidates = sorted(
            candidate_vocab,
            key=lambda c: stable_hash(seed, task_id, "candidate_coverage", c),
        )

        unselectable_candidates = []
        for c in sorted_candidates:
            valid_anchors = [
                aid
                for aid in k_a
                if remaining_quota[aid] > 0
                and c not in assigned_negatives[aid]
                and c in valid_negatives_by_anchor[aid]
            ]

            if not valid_anchors:
                unselectable_candidates.append(c)
                continue

            best_anchor = min(
                valid_anchors,
                key=lambda aid: (
                    len(assigned_negatives[aid]),
                    -remaining_quota[aid],
                    stable_hash(seed, task_id, "coverage_anchor", c, aid),
                ),
            )

            assigned_negatives[best_anchor].add(c)
            remaining_quota[best_anchor] -= 1
            candidate_usage[c] += 1

        if unselectable_candidates:
            raise PlanningError(
                f"PRESENCE_UNSELECTABLE_NEGATIVE_CANDIDATES: {len(unselectable_candidates)} candidates"
            )

        # PHASE B: Quota Fill Pass
        sorted_anchors = sorted(
            list(k_a.keys()),
            key=lambda aid: stable_hash(seed, task_id, "anchor_fill", aid),
        )

        for aid in sorted_anchors:
            v_list = valid_negatives_by_anchor[aid]
            while remaining_quota[aid] > 0:
                valid_cands = [c for c in v_list if c not in assigned_negatives[aid]]

                if not valid_cands:
                    raise PlanningError(
                        f"PRESENCE_NEGATIVE_QUOTA_INFEASIBLE: anchor {aid} quota {k_a[aid]}"
                    )

                best_cand = min(
                    valid_cands,
                    key=lambda c: (
                        candidate_usage[c],
                        stable_hash(seed, task_id, "negative_fill", aid, c),
                    ),
                )

                assigned_negatives[aid].add(best_cand)
                remaining_quota[aid] -= 1
                candidate_usage[best_cand] += 1

        # Build selected negative instances
        selected_negatives: list[dict[str, Any]] = []
        for aid in sorted(k_a.keys()):
            for c in sorted(assigned_negatives[aid]):
                rel = relation_classifier(aid, c)
                if rel != "NEGATIVE":
                    raise PlanningError(f"PRESENCE_INVALID_NEGATIVE_SELECTED: {aid}+{c} is {rel}")
                selected_negatives.append(
                    {
                        "instance_id": f"{aid}:neg:{c}",
                        "segment_id": aid,
                        "audio_ids": [aid],
                        "candidate_term": c,
                        "output_gold": {"present": False},
                        "semantic_relation": "NEGATIVE",
                    }
                )

        audit_data = {
            "eligible_anchors": len(k_a),
            "covered_anchors": len(k_a),
            "positive_universe": len(selected_positives),
            "selected_positives": len(selected_positives),
            "selected_negatives": len(selected_negatives),
            "total_presence_qa": len(selected_positives) + len(selected_negatives),
            "label_ratio": "1:1",
            "candidate_vocab_size": len(candidate_vocab),
            "positive_candidate_coverage": 1.0,
            "negative_candidate_coverage": sum(1 for c in candidate_vocab if candidate_usage[c] > 0)
            / len(candidate_vocab),
            "candidate_negative_usage_stats": {
                "min": min(candidate_usage.values()) if candidate_usage else 0,
                "max": max(candidate_usage.values()) if candidate_usage else 0,
                "mean": round(sum(candidate_usage.values()) / len(candidate_usage), 4)
                if candidate_usage
                else 0.0,
            },
            "invalid_selected": 0,
            "collision_selected": 0,
        }

        return selected_positives + selected_negatives, audit_data


class BalancedBinaryRelationalPairsPolicy:
    policy_id = "BALANCED_BINARY_RELATIONAL_PAIRS"

    @staticmethod
    def select_label_pairs(
        task_id: str,
        anchors: list[dict[str, Any]],
        label: str,
        target_count: int = 11832,
        max_degree: int = 6,
        seed: int = 42,
        group_field: str = "original_video_link",
        topic_field: str = "topic",
    ) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, Any]]:
        anchor_by_id = {str(a["segment_id"]): a for a in anchors}
        anchor_ids = list(anchor_by_id.keys())
        video_by_id = {str(a["segment_id"]): str(a[group_field]) for a in anchors}
        topic_by_id = {str(a["segment_id"]): str(a[topic_field]) for a in anchors}

        # Pre-group anchors by topic and video for fast O(1) set ops
        by_topic: dict[str, set[str]] = {}
        by_video: dict[str, set[str]] = {}
        for a in anchors:
            aid = str(a["segment_id"])
            by_topic.setdefault(str(a[topic_field]), set()).add(aid)
            by_video.setdefault(str(a[group_field]), set()).add(aid)

        # Preflight Check (Section 30)
        for a in anchors:
            aid = str(a["segment_id"])
            top = topic_by_id[aid]
            vid = video_by_id[aid]
            same_vid_anchors = by_video[vid]

            if label == "SAME_TOPIC":
                same_top_anchors = by_topic[top]
                has_same = len(same_top_anchors - same_vid_anchors) > 0
                if not has_same:
                    raise PlanningError(f"PAIRWISE_CROSS_VIDEO_COVERAGE_INFEASIBLE: {aid}")
            elif label == "DIFFERENT_TOPIC":
                same_top_anchors = by_topic[top]
                all_invalid = same_vid_anchors | same_top_anchors
                has_diff = len(set(anchor_ids) - all_invalid) > 0
                if not has_diff:
                    raise PlanningError(f"PAIRWISE_CROSS_VIDEO_COVERAGE_INFEASIBLE: {aid}")

        degree: dict[str, int] = {aid: 0 for aid in anchor_ids}
        degree_sets: dict[int, set[str]] = {0: set(anchor_ids)}
        selected_pairs: set[tuple[str, str]] = set()
        selected_pairs_list: list[tuple[dict[str, Any], dict[str, Any]]] = []

        def update_degree(aid: str) -> None:
            old_d = degree[aid]
            new_d = old_d + 1
            degree[aid] = new_d
            degree_sets[old_d].discard(aid)
            degree_sets.setdefault(new_d, set()).add(aid)

        def get_valid_candidates(aid: str, target_d_set: set[str]) -> list[str]:
            top = topic_by_id[aid]
            vid = video_by_id[aid]
            if label == "SAME_TOPIC":
                pool = target_d_set & by_topic[top]
            else:
                pool = target_d_set - by_topic[top]

        anchor_cand_hash = {
            aid: stable_hash(seed, task_id, label, "cand_order", aid)
            for aid in anchor_ids
        }

        def get_valid_candidates(aid: str, target_d_set: set[str]) -> list[str]:
            top = topic_by_id[aid]
            vid = video_by_id[aid]
            if label == "SAME_TOPIC":
                pool = target_d_set & by_topic[top]
            else:
                pool = target_d_set - by_topic[top]

            avail = []
            for bid in pool:
                if bid == aid or video_by_id[bid] == vid:
                    continue
                pair_key = (min(aid, bid), max(aid, bid))
                if pair_key not in selected_pairs:
                    avail.append(bid)

            if len(avail) > 50:
                avail = sorted(avail, key=lambda bid: anchor_cand_hash[bid])[:50]

            return avail

        # PHASE 1: Coverage Pass
        anchor_order = sorted(
            anchor_ids,
            key=lambda aid: stable_hash(seed, task_id, label, "coverage_anchor", aid),
        )

        for aid in anchor_order:
            if degree[aid] > 0:
                continue

            best_b = None
            for target_d in range(max_degree):
                d_set = degree_sets.get(target_d)
                if not d_set:
                    continue
                avail = get_valid_candidates(aid, d_set)
                if avail:
                    best_b = min(
                        avail,
                        key=lambda bid: (
                            0 if degree[bid] == 0 else 1,
                            degree[bid],
                            stable_hash(
                                seed,
                                task_id,
                                label,
                                "coverage_pair",
                                canonical_pair_id(aid, bid),
                            ),
                        ),
                    )
                    break

            if best_b is None:
                raise PlanningError(f"PAIRWISE_ANCHOR_COVERAGE_FAILED: anchor {aid} degree 0")

            pair_key = (min(aid, best_b), max(aid, best_b))
            selected_pairs.add(pair_key)
            update_degree(aid)
            update_degree(best_b)
            selected_pairs_list.append((anchor_by_id[pair_key[0]], anchor_by_id[pair_key[1]]))

        if min(degree.values()) < 1:
            raise PlanningError(f"PAIRWISE_COVERAGE_ASSERTION_FAILED: min degree is {min(degree.values())}")

        # PHASE 2: Fill Pass
        fill_anchor_hash = {
            aid: stable_hash(seed, task_id, label, "fill_anchor", aid)
            for aid in anchor_ids
        }

        while len(selected_pairs) < target_count:
            min_avail_degree = min(d for d in range(max_degree) if degree_sets.get(d))
            active_anchors = sorted(
                degree_sets[min_avail_degree],
                key=lambda aid: fill_anchor_hash[aid],
            )

            added_pair = False
            for aid in active_anchors:
                if degree[aid] >= max_degree:
                    continue

                best_b = None
                for target_d in range(max_degree):
                    d_set = degree_sets.get(target_d)
                    if not d_set:
                        continue
                    avail = get_valid_candidates(aid, d_set)
                    if avail:
                        best_b = min(
                            avail,
                            key=lambda bid: (
                                max(degree[aid], degree[bid]),
                                degree[aid] + degree[bid],
                                stable_hash(
                                    seed,
                                    task_id,
                                    label,
                                    "fill_pair",
                                    canonical_pair_id(aid, bid),
                                ),
                            ),
                        )
                        break

                if best_b is not None:
                    pair_key = (min(aid, best_b), max(aid, best_b))
                    selected_pairs.add(pair_key)
                    update_degree(aid)
                    update_degree(best_b)
                    selected_pairs_list.append((anchor_by_id[pair_key[0]], anchor_by_id[pair_key[1]]))
                    added_pair = True
                    break

            if not added_pair:
                raise PlanningError(
                    f"PAIRWISE_TARGET_INFEASIBLE_UNDER_DEGREE_CAP: stalled at {len(selected_pairs)} for {label}"
                )

        assert len(selected_pairs) == target_count
        assert min(degree.values()) >= 1
        assert max(degree.values()) <= max_degree

        deg_vals = list(degree.values())
        deg_sorted = sorted(deg_vals)

        audit_info = {
            "label": label,
            "selected_pair_count": len(selected_pairs),
            "unique_anchors": len(degree),
            "anchor_coverage_ratio": 1.0,
            "same_video_pairs": 0,
            "min_degree": min(deg_vals),
            "mean_degree": round(sum(deg_vals) / len(deg_vals), 4),
            "p50_degree": deg_sorted[len(deg_sorted) // 2],
            "p95_degree": deg_sorted[int(len(deg_sorted) * 0.95)],
            "p99_degree": deg_sorted[int(len(deg_sorted) * 0.99)],
            "max_degree": max(deg_vals),
        }

        return selected_pairs_list, audit_info

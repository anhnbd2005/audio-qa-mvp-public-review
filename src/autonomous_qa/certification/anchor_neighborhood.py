"""Generic anchor-neighborhood EQUALITY feasibility + final plan audits.

Dataset-agnostic: they operate on a semantic field, its value groups and the
generated plan records. No dataset names appear here.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

from src.autonomous_qa.certification.qa_sampling import TrainIndex, comb2


class AnchorPlanAuditError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}:{detail}" if detail else code)


def anchor_neighborhood_feasibility(
    index: TrainIndex,
    field: str,
    *,
    positive_per_anchor: int,
    negative_per_anchor: int,
) -> dict[str, Any]:
    """Metadata-only feasibility of the 2+2 per-anchor policy.

    No QA is rendered and no pair is enumerated.
    """
    anchors = sorted(index.valid_row_ids[field])
    total_rows = len(anchors)
    group_sizes = Counter(
        len(members) for members in index.rows_by_value[field].values()
    )
    sizes = sorted(len(m) for m in index.rows_by_value[field].values())
    min_group = sizes[0] if sizes else 0
    max_group = sizes[-1] if sizes else 0

    # SAME: pos distinct partners per anchor, each unordered pair used once.
    # Within a group of size m, unique pairs = comb2(m); need pos*m.
    same_capacity_ok = all(
        comb2(m) >= positive_per_anchor * m for m in sizes
    )
    # Count anchors (rows) that cannot get 2 same-topic partners.
    anchors_without_two_same = 0
    for members in index.rows_by_value[field].values():
        if len(members) < 3:
            anchors_without_two_same += len(members)
    # DIFFERENT: each anchor needs neg partners outside its own group.
    anchors_without_two_different = 0
    for members in index.rows_by_value[field].values():
        outside = total_rows - len(members)
        if outside < max(2, negative_per_anchor):
            anchors_without_two_different += len(members)

    feasible = (
        same_capacity_ok
        and anchors_without_two_same == 0
        and anchors_without_two_different == 0
    )
    return {
        "field": field,
        "train_rows": total_rows,
        "topic_groups": len(sizes),
        "min_group_size": min_group,
        "max_group_size": max_group,
        "group_size_histogram": {str(k): v for k, v in sorted(group_sizes.items())},
        "anchors_unable_two_same": anchors_without_two_same,
        "anchors_unable_two_different": anchors_without_two_different,
        "requested_same_pairs": positive_per_anchor * total_rows,
        "requested_different_pairs": negative_per_anchor * total_rows,
        "positive_per_anchor": positive_per_anchor,
        "negative_per_anchor": negative_per_anchor,
        "same_pair_capacity_ok": same_capacity_ok,
        "feasible": feasible,
        "pair_enumeration": "NOT_PERFORMED",
        "full_cartesian_pair_enumeration": False,
    }


def audit_split_isolation(
    records: list[dict[str, Any]],
    *,
    split: str,
    split_of_row: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Verify every record belongs to the expected split and pairs are same-split."""
    failures: list[str] = []
    bad_split = 0
    cross_split = 0
    for record in records:
        if record.get("split") != split:
            bad_split += 1
        if split_of_row is not None and len(record.get("source_row_ids", [])) >= 2:
            splits = {split_of_row(r) for r in record["source_row_ids"]}
            if len(splits) != 1 or split not in splits:
                cross_split += 1
    if bad_split:
        failures.append(f"SPLIT_MISMATCH:{bad_split}")
    if cross_split:
        failures.append(f"CROSS_SPLIT_PAIR:{cross_split}")
    return {
        "split": split,
        "records": len(records),
        "split_mismatches": bad_split,
        "cross_split_pairs": cross_split,
        "passed": not failures,
        "failures": failures,
    }


def audit_direct_coverage(
    records: list[dict[str, Any]],
    *,
    type_id: str,
    expected_rows: int,
) -> dict[str, Any]:
    typed = [r for r in records if r["type_id"] == type_id]
    occurrences: Counter = Counter()
    for record in typed:
        rows = record["source_row_ids"]
        if len(rows) != 1:
            occurrences["__arity__"] += 1
            continue
        occurrences[rows[0]] += 1
    occurrences.pop("__arity__", None)
    count = len(typed)
    unique = len(occurrences)
    min_occ = min(occurrences.values()) if occurrences else 0
    max_occ = max(occurrences.values()) if occurrences else 0
    passed = (
        count == expected_rows
        and unique == expected_rows
        and min_occ == 1
        and max_occ == 1
    )
    return {
        "type_id": type_id,
        "records": count,
        "unique_source_rows": unique,
        "min_row_occurrence": min_occ,
        "max_row_occurrence": max_occ,
        "expected_rows": expected_rows,
        "passed": passed,
    }


def audit_anchor_neighborhood_plan(
    records: list[dict[str, Any]],
    *,
    type_id: str,
    row_value_lookup: Callable[[str], Any],
    positive_per_anchor: int,
    negative_per_anchor: int,
    expected_anchors: int,
) -> dict[str, Any]:
    typed = [r for r in records if r["type_id"] == type_id]
    failures: list[str] = []
    anchor_counts: Counter = Counter()
    same_by_anchor: Counter = Counter()
    diff_by_anchor: Counter = Counter()
    seen_pairs: set[tuple[str, str]] = set()
    self_pairs = 0
    gold_mismatch = 0
    arity_bad = 0

    for record in typed:
        rows = record["source_row_ids"]
        if len(rows) != 2 or rows[0] == rows[1]:
            arity_bad += 1
            if len(rows) == 2 and rows[0] == rows[1]:
                self_pairs += 1
            continue
        anchor, partner = rows
        anchor_counts[anchor] += 1
        if bool(record["gold"]["value"]):
            same_by_anchor[anchor] += 1
        else:
            diff_by_anchor[anchor] += 1
        key = tuple(sorted((anchor, partner)))
        if key in seen_pairs:
            failures.append(f"DUPLICATE_UNORDERED_PAIR:{key}")
        seen_pairs.add(key)
        expected_gold = row_value_lookup(anchor) == row_value_lookup(partner)
        if bool(record["gold"]["value"]) != bool(expected_gold):
            gold_mismatch += 1

    anchors = set(anchor_counts)
    per_anchor_bad = 0
    for anchor in anchors:
        if (
            same_by_anchor[anchor] != positive_per_anchor
            or diff_by_anchor[anchor] != negative_per_anchor
            or anchor_counts[anchor]
            != (positive_per_anchor + negative_per_anchor)
        ):
            per_anchor_bad += 1

    total = len(typed)
    expected_total = expected_anchors * (positive_per_anchor + negative_per_anchor)
    global_same = sum(same_by_anchor.values())
    global_diff = sum(diff_by_anchor.values())
    expected_side = expected_anchors * positive_per_anchor

    if arity_bad:
        failures.append(f"ARITY_OR_SELF:{arity_bad}")
    if self_pairs:
        failures.append(f"SELF_PAIRS:{self_pairs}")
    if gold_mismatch:
        failures.append(f"GOLD_MISMATCH:{gold_mismatch}")
    if len(anchors) != expected_anchors:
        failures.append(f"ANCHOR_COVERAGE:{len(anchors)}!={expected_anchors}")
    if per_anchor_bad:
        failures.append(f"PER_ANCHOR_COUNTS_BAD:{per_anchor_bad}")
    if total != expected_total:
        failures.append(f"TOTAL:{total}!={expected_total}")
    if global_same != expected_side or global_diff != expected_side:
        failures.append(f"GLOBAL_SIDE:{global_same}/{global_diff}!={expected_side}")

    duplicate_pairs = sum(1 for f in failures if f.startswith("DUPLICATE_UNORDERED_PAIR"))
    return {
        "type_id": type_id,
        "records": total,
        "anchors": len(anchors),
        "expected_anchors": expected_anchors,
        "same_topic": global_same,
        "different_topic": global_diff,
        "unique_unordered_pairs": len(seen_pairs),
        "duplicate_unordered_pairs": duplicate_pairs,
        "self_pairs": self_pairs,
        "gold_mismatches": gold_mismatch,
        "per_anchor_bad": per_anchor_bad,
        "expected_total": expected_total,
        "passed": not failures,
        "failures": failures,
    }

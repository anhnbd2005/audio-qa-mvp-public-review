"""Comparator-Collision Gate.

Normalization must not merge semantically distinct authoritative labels unless
the contract explicitly declares them equivalent.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from src.autonomous_qa.compiler.semantic_comparators import ComparatorSpec, normalize_with_comparator


def comparator_collision_audit(
    raw_values: Iterable,
    comparator: ComparatorSpec,
    *,
    allowed_collapse_groups: Iterable[Iterable] = (),
) -> dict:
    """Group authoritative raw values by their normalized key.

    A collision group is acceptable only if it is a subset of an explicitly
    allowed collapse group. Any unexpected collision is BLOCKING.
    """
    raw = [str(v) for v in raw_values]
    groups: dict[str, set[str]] = defaultdict(set)
    for value in raw:
        groups[normalize_with_comparator(comparator, value)].add(value)

    collisions = {key: sorted(vals) for key, vals in groups.items() if len(vals) > 1}
    allowed = [{str(x) for x in group} for group in allowed_collapse_groups]

    unexpected: dict[str, list[str]] = {}
    for key, vals in collisions.items():
        if not any(set(vals) <= group for group in allowed):
            unexpected[key] = vals

    return {
        "comparator_id": comparator.comparator_id,
        "raw_unique": len(set(raw)),
        "normalized_unique": len(groups),
        "collision_class_count": len(collisions),
        "unexpected_collision_class_count": len(unexpected),
        "collisions": collisions,
        "unexpected_collisions": unexpected,
        "severity": "BLOCKING" if unexpected else "PASS",
    }

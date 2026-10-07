"""Deterministic, balanced template assignment.

Production must never use uncontrolled random.choice(). Given the same plan,
language registry and seed, rendered bytes must be identical. Assignment is
also balanced so no single template dominates.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable


def stable_hash_int(*parts: str) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def choose_template(
    qa_id: str, registry_hash: str, seed: int, template_count: int
) -> int:
    if template_count <= 0:
        raise ValueError("template_count must be positive")
    return stable_hash_int(qa_id, registry_hash, str(seed)) % template_count


def assign_templates(
    qa_ids: Iterable[str],
    template_count: int,
    *,
    registry_hash: str = "",
    seed: int = 0,
) -> dict:
    """Balanced deterministic assignment.

    Processes QA ids in stable sorted order and always assigns to the
    currently least-used template, breaking ties by a stable hash of
    (qa_id, template_index). Reproducible and near-uniform.
    """
    if template_count <= 0:
        raise ValueError("template_count must be positive")
    ordered = sorted(str(q) for q in qa_ids)
    counts = [0] * template_count
    mapping: dict[str, int] = {}
    for qa_id in ordered:
        best = min(
            range(template_count),
            key=lambda t: (
                counts[t],
                stable_hash_int(qa_id, str(t), registry_hash, str(seed)),
            ),
        )
        mapping[qa_id] = best
        counts[best] += 1
    return {
        "template_count": template_count,
        "assigned": len(mapping),
        "usage_counts": {str(i): counts[i] for i in range(template_count)},
        "min_usage": min(counts) if counts else 0,
        "max_usage": max(counts) if counts else 0,
        "imbalance": (max(counts) - min(counts)) if counts else 0,
        "assignment": mapping,
    }


def template_balance_report(mapping: dict) -> dict:
    counts = Counter(mapping.values())
    values = list(counts.values()) or [0]
    return {
        "templates_used": len(counts),
        "min": min(values),
        "max": max(values),
        "imbalance": max(values) - min(values),
    }

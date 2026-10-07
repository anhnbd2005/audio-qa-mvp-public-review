"""Task-scoped sanitation + local-vs-systematic anomaly detection.

A source row may be eligible for one task and ineligible for another. Source
data is never edited: only task eligibility changes.

The detector refuses to reduce a coherent repeated semantic transformation to
"small local noise" — a known historical failure mode (VietMDD P3: 599
presentation-only mismatches that required a comparator repair, not dropping).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

ELIGIBLE = "ELIGIBLE"
DROP_LOCAL_ANOMALY = "DROP_LOCAL_ANOMALY"
BLOCKED_BY_CONTRACT = "BLOCKED_BY_CONTRACT"
KNOWLEDGE_UNRESOLVED = "KNOWLEDGE_UNRESOLVED"

ELIGIBILITY_STATUSES = (
    ELIGIBLE,
    DROP_LOCAL_ANOMALY,
    BLOCKED_BY_CONTRACT,
    KNOWLEDGE_UNRESOLVED,
)

LOCAL_HETEROGENEOUS_NOISE = "LOCAL_HETEROGENEOUS_NOISE"
SYSTEMATIC_CONTRACT_PATTERN = "SYSTEMATIC_CONTRACT_PATTERN"
UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class SanitationPolicy:
    local_noise_max_fraction: float = 0.02
    minimum_anomaly_count_for_pattern_detection: int = 5
    systematic_signature_threshold: float = 0.50
    action_local_noise: str = "AUTO_DROP_ROW"
    action_systematic_pattern: str = "BLOCKING"


class TaskEligibilityMap:
    """row_id -> task_id -> {status, reason}."""

    def __init__(self) -> None:
        self._map: dict[str, dict[str, dict[str, str]]] = {}

    def set(self, row_id: str, task_id: str, status: str, reason: str = "") -> None:
        if status not in ELIGIBILITY_STATUSES:
            raise ValueError(f"UNKNOWN_ELIGIBILITY_STATUS:{status}")
        self._map.setdefault(str(row_id), {})[task_id] = {
            "status": status,
            "reason": reason,
        }

    def status(self, row_id: str, task_id: str) -> str:
        return self._map.get(str(row_id), {}).get(task_id, {}).get("status", ELIGIBLE)

    def is_eligible(self, row_id: str, task_id: str) -> bool:
        return self.status(row_id, task_id) == ELIGIBLE

    def task_counts(self) -> dict[str, Counter]:
        result: dict[str, Counter] = {}
        for tasks in self._map.values():
            for task_id, info in tasks.items():
                result.setdefault(task_id, Counter())[info["status"]] += 1
        return result

    def to_dict(self) -> dict[str, Any]:
        return {row: dict(tasks) for row, tasks in self._map.items()}


def classify_anomalies(
    signatures: Iterable[str],
    total_rows: int,
    policy: SanitationPolicy | None = None,
) -> dict[str, Any]:
    """Classify an anomaly population as local noise or a systematic pattern.

    The percentage alone never decides: a single coherent signature that
    dominates the anomaly population is a systematic contract pattern even if
    its overall fraction is tiny.
    """
    policy = policy or SanitationPolicy()
    signatures = [str(s) for s in signatures]
    anomaly_count = len(signatures)
    fraction = anomaly_count / total_rows if total_rows else 0.0

    if anomaly_count == 0:
        return {
            "classification": "NONE",
            "severity": "PASS",
            "anomaly_count": 0,
            "anomaly_fraction": 0.0,
            "signature_distribution": {},
        }

    distribution = Counter(signatures)
    top_signature, top_count = distribution.most_common(1)[0]
    top_share = top_count / anomaly_count

    coherent_systematic = (
        top_signature != "HETEROGENEOUS"
        and top_share >= policy.systematic_signature_threshold
        and anomaly_count >= policy.minimum_anomaly_count_for_pattern_detection
    )
    if coherent_systematic:
        return {
            "classification": SYSTEMATIC_CONTRACT_PATTERN,
            "severity": policy.action_systematic_pattern,
            "anomaly_count": anomaly_count,
            "anomaly_fraction": round(fraction, 6),
            "dominant_signature": top_signature,
            "dominant_share": round(top_share, 6),
            "signature_distribution": dict(distribution.most_common()),
        }

    local_heterogeneous = fraction <= policy.local_noise_max_fraction
    if local_heterogeneous:
        return {
            "classification": LOCAL_HETEROGENEOUS_NOISE,
            "severity": policy.action_local_noise,
            "anomaly_count": anomaly_count,
            "anomaly_fraction": round(fraction, 6),
            "signature_distribution": dict(distribution.most_common()),
        }

    return {
        "classification": UNRESOLVED,
        "severity": "REVIEW",
        "anomaly_count": anomaly_count,
        "anomaly_fraction": round(fraction, 6),
        "signature_distribution": dict(distribution.most_common()),
    }


def task_scoped_eligibility(
    anomalies: list[dict[str, Any]],
    policy: SanitationPolicy | None = None,
) -> dict[str, Any]:
    """Build a task eligibility map from anomaly records.

    Each record: {row_id, task_id, signature}. If the anomaly population for a
    task is systematic, the whole task is BLOCKED_BY_CONTRACT rather than
    dropping individual rows; otherwise anomalous rows are dropped locally.
    """
    policy = policy or SanitationPolicy()
    by_task: dict[str, list[dict[str, Any]]] = {}
    for record in anomalies:
        by_task.setdefault(record["task_id"], []).append(record)

    elmap = TaskEligibilityMap()
    task_reports: dict[str, Any] = {}
    for task_id, records in by_task.items():
        total = max(1, len({r["row_id"] for r in records}))
        classification = classify_anomalies(
            [r.get("signature", "HETEROGENEOUS") for r in records], total, policy
        )
        task_reports[task_id] = classification
        for record in records:
            if classification["classification"] == SYSTEMATIC_CONTRACT_PATTERN:
                elmap.set(
                    record["row_id"],
                    task_id,
                    BLOCKED_BY_CONTRACT,
                    classification.get("dominant_signature", ""),
                )
            else:
                elmap.set(
                    record["row_id"], task_id, DROP_LOCAL_ANOMALY, "local_anomaly"
                )
    return {"eligibility": elmap.to_dict(), "tasks": task_reports}

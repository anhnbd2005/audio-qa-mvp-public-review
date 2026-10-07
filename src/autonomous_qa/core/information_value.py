"""Information-Value Gate.

Generic label-information metrics + a decision that considers both natural
imbalance AND usable (balanced) capacity. It deliberately avoids the naive
"majority > 80% -> reject" rule.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

from src.common.severity import max_severity

REASONABLE_MAJORITY_CEILING = 0.80
REASONABLE_NORMALIZED_ENTROPY_FLOOR = 0.50


def information_metrics(labels: Iterable) -> dict:
    counts = Counter(str(label) for label in labels)
    total = sum(counts.values())
    if total == 0:
        return {
            "counts": {},
            "total": 0,
            "class_count": 0,
            "majority_rate": 0.0,
            "minority_count": 0,
            "entropy": 0.0,
            "normalized_entropy": 0.0,
            "effective_class_count": 0.0,
            "balanced_capacity": 0,
        }
    class_count = len(counts)
    majority_rate = max(counts.values()) / total
    minority_count = min(counts.values())
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    max_entropy = math.log2(class_count) if class_count > 1 else 1.0
    normalized = entropy / max_entropy if max_entropy else 0.0
    return {
        "counts": dict(sorted(counts.items())),
        "total": total,
        "class_count": class_count,
        "majority_rate": round(majority_rate, 6),
        "minority_count": minority_count,
        "entropy": round(entropy, 6),
        "normalized_entropy": round(normalized, 6),
        "effective_class_count": round(2**entropy, 6),
        # A fully balanced subset can use minority_count per class.
        "balanced_capacity": minority_count * class_count,
    }


def information_value_decision(
    metrics: dict,
    *,
    target_size: int | None = None,
    min_minority_count: int = 50,
    min_balanced_capacity: int = 100,
) -> dict:
    """Decide information-value severity.

    - reasonable information -> PASS
    - high imbalance but a usable minority/balanced capacity -> PASS with
      planner balancing capability (the natural imbalance is not blindly
      rejected by the >80% majority rule)
    - high imbalance and genuinely tiny minority capacity -> AUTO_REJECT_TASK
    """
    severities: list[str] = []
    reasons: list[str] = []
    if metrics["total"] == 0 or metrics["class_count"] < 2:
        severities.append("AUTO_REJECT_TASK")
        reasons.append("NO_LABEL_VARIATION")

    reasonable = (
        metrics["majority_rate"] <= REASONABLE_MAJORITY_CEILING
        and metrics["normalized_entropy"] >= REASONABLE_NORMALIZED_ENTROPY_FLOOR
    )
    if severities:
        pass
    elif reasonable:
        severities.append("PASS")
        reasons.append("REASONABLE_INFORMATION")
    else:
        balanced_capacity = metrics["balanced_capacity"]
        if metrics["minority_count"] < min_minority_count:
            severities.append("AUTO_REJECT_TASK")
            reasons.append("INSUFFICIENT_MINORITY_CAPACITY")
        elif balanced_capacity < min_balanced_capacity:
            severities.append("AUTO_REJECT_TASK")
            reasons.append("BALANCED_CAPACITY_TOO_SMALL")
        else:
            severities.append("PASS")
            reasons.append("PASS_WITH_PLANNER_BALANCING_CAPABILITY")

    severity = max_severity(severities)
    return {
        "severity": severity,
        "reasons": reasons,
        "metrics": metrics,
        "target_size": target_size,
        "usable_balanced_capacity": metrics["balanced_capacity"],
    }

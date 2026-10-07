"""Generic severity taxonomy for guardrails and review outcomes."""

from __future__ import annotations

SEVERITIES: tuple[str, ...] = (
    "PASS",
    "REPORT_ONLY",
    "REVIEW",
    "AUTO_DROP_ROW",
    "AUTO_REJECT_TASK",
    "BLOCKING",
)

# Higher = more severe. Used to aggregate a guardrail report.
_SEVERITY_RANK = {
    "PASS": 0,
    "REPORT_ONLY": 1,
    "REVIEW": 2,
    "AUTO_DROP_ROW": 3,
    "AUTO_REJECT_TASK": 4,
    "BLOCKING": 5,
}

SEVERITY_BY_RANK = {rank: name for name, rank in _SEVERITY_RANK.items()}


def severity_rank(severity: str) -> int:
    if severity not in _SEVERITY_RANK:
        raise ValueError(f"UNKNOWN_SEVERITY:{severity}")
    return _SEVERITY_RANK[severity]


def max_severity(severities: list[str]) -> str:
    if not severities:
        return "PASS"
    return max(severities, key=severity_rank)


def is_blocking(severity: str) -> bool:
    return severity == "BLOCKING"


def rejects_task(severity: str) -> bool:
    return severity in ("AUTO_REJECT_TASK", "BLOCKING")

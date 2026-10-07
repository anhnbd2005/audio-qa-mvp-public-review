"""Intrinsic semantic tiers and gold-origin contracts."""

from __future__ import annotations

from typing import Literal

Tier = Literal[
    "T1_PERCEPTION",
    "T2_DERIVED",
    "T3_RELATIONAL",
]

GoldOrigin = Literal[
    "SOURCE",
    "DERIVED_SOURCE",
]

TIERS: tuple[str, ...] = (
    "T1_PERCEPTION",
    "T2_DERIVED",
    "T3_RELATIONAL",
)

GOLD_ORIGINS: tuple[str, ...] = (
    "SOURCE",
    "DERIVED_SOURCE",
)

_ALLOWED_GOLD_ORIGINS: dict[str, frozenset[str]] = {
    "T1_PERCEPTION": frozenset({"SOURCE"}),
    "T2_DERIVED": frozenset({"DERIVED_SOURCE", "SOURCE"}),
    "T3_RELATIONAL": frozenset({"DERIVED_SOURCE"}),
}


class TierError(ValueError):
    pass


def validate_tier(tier: str | None) -> str:
    if tier not in TIERS:
        raise TierError(f"UNKNOWN_TIER:{tier}")
    return tier


def validate_gold_origin(origin: str | None) -> str:
    if origin not in GOLD_ORIGINS:
        raise TierError(f"UNKNOWN_GOLD_ORIGIN:{origin}")
    return origin


def validate_pair(tier: str | None, gold_origin: str | None) -> list[str]:
    issues: list[str] = []

    if tier not in TIERS:
        issues.append(f"UNKNOWN_TIER:{tier}")

    if gold_origin not in GOLD_ORIGINS:
        issues.append(f"UNKNOWN_GOLD_ORIGIN:{gold_origin}")

    if not issues and gold_origin not in _ALLOWED_GOLD_ORIGINS[tier]:
        issues.append(
            f"GOLD_ORIGIN_NOT_ALLOWED_FOR_TIER:{tier}:{gold_origin}"
        )

    return issues

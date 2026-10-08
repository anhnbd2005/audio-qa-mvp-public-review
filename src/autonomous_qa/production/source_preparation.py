"""Canonical Source Preparation + Early Budget Authorization stage.

Establishes the lightweight preparation/budget stages that run BEFORE any LLM
semantic discovery or language authoring::

    SOURCE DISCOVERY
        -> SOURCE PREPARATION            (this module: prepare_source_inventory)
        -> EARLY BUDGET AUTHORIZATION    (this module: resolve_early_budget)
        -> LLM SEMANTIC DISCOVERY
        -> ... language authoring ...
        -> FINAL PER-TYPE BUDGET FREEZE  (this module: finalize_allocation)

Two distinct budget concepts are kept separate:

* :class:`EarlyBudget` — authoritative production intent, resolved from the
  source inventory BEFORE the first LLM proposal (``target_qa_per_audio`` /
  ``target_total_qa`` / ``coverage_mode`` / ``allowed_operator_families`` /
  ``shortfall_policy``).
* **Approved allocation** — the per-type quotas produced after semantic
  validation by the SAME production algorithms (``derive_full_split_budget`` /
  ``resolve_budget``). It may not exceed or silently revise the early target;
  any infeasibility yields explicit shortfall evidence.

This module owns no sampling, planning or rendering, and reuses the existing
budget algorithms (no duplicate budget system). It never loads the source JSONL
more than once per preparation workflow: callers that already hold ``rows`` pass
them in.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.autonomous_qa.language.language_quality import ProductionGenerationConfig
from src.autonomous_qa.language.template_contracts import TypeContract
from src.autonomous_qa.language.template_renderer import canonical_hash
from src.autonomous_qa.production.budget import (
    derive_full_split_budget,
    resolve_budget,
)


class BudgetAuthorityError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}:{detail}" if detail else code)


# ---------------------------------------------------------------------------
# SOURCE PREPARATION
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceInventory:
    """Lightweight, reproducible facts about one prepared source split."""

    dataset: str
    split: str
    path: str
    sha256: str
    row_count: int
    identity_field: str | None
    schema_fields: tuple[str, ...]
    eligible_rows: dict[str, int]
    label_groups: dict[str, int]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "path": self.path,
            "sha256": self.sha256,
            "row_count": self.row_count,
            "identity_field": self.identity_field,
            "schema_fields": list(self.schema_fields),
            "eligible_rows": dict(self.eligible_rows),
            "label_groups": dict(self.label_groups),
            "fingerprint": self.fingerprint,
        }


def _is_present(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and not value.strip())


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare_source_inventory(
    *,
    dataset: str,
    split: str,
    path: Path,
    rows: list[dict[str, Any]] | None = None,
    identity_field: str | None = None,
    eligibility_fields: list[str] | None = None,
    group_fields: list[str] | None = None,
    expected_sha256: str | None = None,
    expected_rows: int | None = None,
) -> SourceInventory:
    """Validate and inventory one source split (loads the file at most once).

    Minimal by design: identity + SHA256 + row count + schema + per-field
    eligibility counts + label-group counts. No EDA, no audio decode.
    """
    path = Path(path)
    if rows is None:
        if not path.exists():
            raise BudgetAuthorityError("SOURCE_MISSING", str(path))
        rows = load_jsonl_rows(path)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""

    if expected_sha256 is not None and sha256 != expected_sha256:
        raise BudgetAuthorityError(
            "SOURCE_SHA_MISMATCH", f"{sha256}!={expected_sha256}"
        )
    if expected_rows is not None and len(rows) != expected_rows:
        raise BudgetAuthorityError(
            "SOURCE_ROW_COUNT_MISMATCH", f"{len(rows)}!={expected_rows}"
        )

    schema_fields = tuple(sorted({key for row in rows for key in row}))

    eligibility_fields = list(eligibility_fields or [])
    eligible_rows = {
        field: sum(1 for row in rows if _is_present(row.get(field)))
        for field in eligibility_fields
    }

    label_groups: dict[str, int] = {}
    for field in group_fields or []:
        values = {
            canonical_hash(row.get(field))
            for row in rows
            if _is_present(row.get(field))
        }
        label_groups[field] = len(values)

    identity_unique = None
    if identity_field:
        ids = [row.get(identity_field) for row in rows if _is_present(row.get(identity_field))]
        identity_unique = len(ids) == len(set(ids))

    payload = {
        "dataset": dataset,
        "split": split,
        "path": path.as_posix(),
        "sha256": sha256,
        "row_count": len(rows),
        "identity_field": identity_field,
        "identity_unique": identity_unique,
        "schema_fields": list(schema_fields),
        "eligible_rows": eligible_rows,
        "label_groups": label_groups,
    }
    return SourceInventory(
        dataset=dataset,
        split=split,
        path=path.as_posix(),
        sha256=sha256,
        row_count=len(rows),
        identity_field=identity_field,
        schema_fields=schema_fields,
        eligible_rows=eligible_rows,
        label_groups=label_groups,
        fingerprint=canonical_hash(payload),
    )


# ---------------------------------------------------------------------------
# EARLY BUDGET AUTHORIZATION
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EarlyBudget:
    """Authoritative production intent, frozen before the first LLM call."""

    dataset: str
    split: str
    coverage_mode: str
    target_total_qa: int | None
    target_qa_per_audio: int | None
    allowed_operator_families: tuple[str, ...]
    shortfall_policy: str
    source_fingerprint: str
    status: str
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "coverage_mode": self.coverage_mode,
            "target_total_qa": self.target_total_qa,
            "target_qa_per_audio": self.target_qa_per_audio,
            "allowed_operator_families": list(self.allowed_operator_families),
            "shortfall_policy": self.shortfall_policy,
            "source_fingerprint": self.source_fingerprint,
            "status": self.status,
            "fingerprint": self.fingerprint,
        }


def resolve_early_budget(
    *,
    inventory: SourceInventory,
    config: ProductionGenerationConfig | None = None,
    exploratory: bool = False,
    eligible_anchors: int | None = None,
) -> EarlyBudget:
    """Resolve the early target budget from the prepared source + config.

    Exploratory authoring (no training-production intent) yields an explicit
    ``EXPLORATORY`` budget instead of fabricating a production target.
    """
    if exploratory or config is None:
        payload = {
            "dataset": inventory.dataset,
            "split": inventory.split,
            "coverage_mode": "exploratory",
            "target_total_qa": None,
            "target_qa_per_audio": None,
            "allowed_operator_families": [],
            "shortfall_policy": "report",
            "source_fingerprint": inventory.fingerprint,
        }
        return EarlyBudget(
            dataset=inventory.dataset,
            split=inventory.split,
            coverage_mode="exploratory",
            target_total_qa=None,
            target_qa_per_audio=None,
            allowed_operator_families=(),
            shortfall_policy="report",
            source_fingerprint=inventory.fingerprint,
            status="EXPLORATORY",
            fingerprint=canonical_hash(payload),
        )

    if eligible_anchors is None:
        eligible_anchors = max(inventory.eligible_rows.values(), default=0)

    coverage_mode = config.coverage_mode
    per_audio = config.target_qa_per_audio
    target_total: int | None = None
    if coverage_mode == "explicit":
        target_total = config.total_target_qa
    elif (
        coverage_mode == "full_split"
        and per_audio is not None
        and eligible_anchors > 0
    ):
        target_total = per_audio * eligible_anchors

    if target_total is not None or (
        coverage_mode == "explicit" and config.per_type_budget
    ):
        status = "AUTHORIZED"
    else:
        status = "READY_AWAITING_BUDGET"

    payload = {
        "dataset": inventory.dataset,
        "split": inventory.split,
        "coverage_mode": coverage_mode,
        "target_total_qa": target_total,
        "target_qa_per_audio": per_audio,
        "allowed_operator_families": list(config.allowed_operator_families),
        "shortfall_policy": config.shortfall_policy,
        "source_fingerprint": inventory.fingerprint,
    }
    return EarlyBudget(
        dataset=inventory.dataset,
        split=inventory.split,
        coverage_mode=coverage_mode,
        target_total_qa=target_total,
        target_qa_per_audio=per_audio,
        allowed_operator_families=tuple(config.allowed_operator_families),
        shortfall_policy=config.shortfall_policy,
        source_fingerprint=inventory.fingerprint,
        status=status,
        fingerprint=canonical_hash(payload),
    )


# ---------------------------------------------------------------------------
# FINAL PER-TYPE BUDGET FREEZE (approved allocation)
# ---------------------------------------------------------------------------


def assert_allocation_within_budget(
    early: EarlyBudget,
    allocation: dict[str, int],
    *,
    error_class: type[Exception] = BudgetAuthorityError,
) -> int | None:
    """Fail closed if an approved allocation exceeds the early target."""
    total = sum(int(v) for v in allocation.values())
    if early.target_total_qa is not None and total > early.target_total_qa:
        raise error_class(
            "ALLOCATION_EXCEEDS_EARLY_BUDGET", f"{total}>{early.target_total_qa}"
        )
    return None if early.target_total_qa is None else early.target_total_qa - total


def finalize_allocation(
    *,
    early: EarlyBudget,
    contracts: list[TypeContract],
    capacities: dict[str, dict[str, Any]],
    config: ProductionGenerationConfig,
    debug_cap: int | None = None,
    error_class: type[Exception] = BudgetAuthorityError,
) -> dict[str, Any]:
    """Freeze the approved per-type allocation via the shared budget algorithms.

    Returns the unchanged ``resolve_budget`` resolution (as ``budget``) plus the
    allocation, target, shortfall evidence and status. Raises when the
    allocation exceeds the early target or when ``shortfall_policy == "fail"``.
    """
    budget_config = config
    if config.coverage_mode == "full_split":
        derived = derive_full_split_budget(config, contracts, capacities)
        if derived is None:
            raise error_class(
                "FULL_SPLIT_COVERAGE_UNSUPPORTED",
                ",".join(sorted(c.type_id for c in contracts)),
            )
        budget_config = config.model_copy(update={"per_type_budget": derived})
    budget = resolve_budget(
        budget_config,
        contracts,
        capacities,
        debug_cap=debug_cap,
        error_class=error_class,
    )
    allocation = {str(k): int(v) for k, v in budget["per_type"].items()}
    shortfall = assert_allocation_within_budget(
        early, allocation, error_class=error_class
    )
    if shortfall and early.shortfall_policy == "fail":
        raise error_class("EARLY_BUDGET_SHORTFALL", str(shortfall))
    return {
        "budget": budget,
        "allocation": allocation,
        "target_total_qa": early.target_total_qa,
        "allocation_total": sum(allocation.values()),
        "shortfall": shortfall,
        "shortfall_policy": early.shortfall_policy,
        "status": budget["status"],
        "early_budget_fingerprint": early.fingerprint,
    }

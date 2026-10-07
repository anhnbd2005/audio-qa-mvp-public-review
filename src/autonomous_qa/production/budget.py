"""Production QA generation budget resolution and allocation.

Extracts budget resolution logic from production_qa.py to enforce single responsibility
and enable independent testing of production budget policies.
"""

from __future__ import annotations

from typing import Any

from src.autonomous_qa.certification.qa_sampling import feasible_semantic_capacity
from src.autonomous_qa.language.language_quality import ProductionGenerationConfig
from src.autonomous_qa.language.template_contracts import TypeContract


def resolve_budget(
    config: ProductionGenerationConfig,
    contracts: list[TypeContract],
    capacities: dict[str, dict[str, Any]],
    *,
    debug_cap: int | None = None,
    error_class: type[Exception] | None = None,
) -> dict[str, Any]:
    """Resolve generation budget per semantic type based on configuration policies.

    Supports DEBUG_SAMPLE_BUDGET, PER_TYPE_BUDGET, TOTAL_BUDGET_EQUAL_BY_TYPE,
    and handles capacity capping & strict budget infeasibility checks.
    """
    type_ids = sorted(contract.type_id for contract in contracts)
    positive_ratio = (
        config.boolean.positive_ratio
        if config.boolean is not None
        else config.positive_negative_ratio
    )
    if debug_cap is not None:
        base = debug_cap // len(type_ids)
        remainder = debug_cap % len(type_ids)
        per_type = {
            type_id: base + (1 if position < remainder else 0)
            for position, type_id in enumerate(type_ids)
        }
        status = "DEBUG_SAMPLE_BUDGET"
    elif config.per_type_budget:
        unknown = sorted(set(config.per_type_budget) - set(type_ids))
        if unknown:
            if error_class is not None:
                raise error_class("BUDGET_TYPE_UNKNOWN", ",".join(unknown))
            raise ValueError(f"BUDGET_TYPE_UNKNOWN: {','.join(unknown)}")
        per_type = {
            type_id: int(config.per_type_budget.get(type_id, 0)) for type_id in type_ids
        }
        status = "PER_TYPE_BUDGET"
    elif config.total_target_qa:
        base = config.total_target_qa // len(type_ids)
        remainder = config.total_target_qa % len(type_ids)
        per_type = {
            type_id: base + (1 if position < remainder else 0)
            for position, type_id in enumerate(type_ids)
        }
        status = "TOTAL_BUDGET_EQUAL_BY_TYPE"
    else:
        return {
            "status": "READY_AWAITING_BUDGET",
            "explicit_budget_present": False,
            "allocation_policy": config.allocation_policy,
            "strict_budget": config.strict_budget,
            "resolved_total": None,
            "per_type": {},
            "capacity_capping": [],
            "debug_sample": False,
        }
    capping: list[dict[str, Any]] = []
    shortfall_total = 0
    for type_id in type_ids:
        requested = per_type.get(type_id, 0)
        feasible = feasible_semantic_capacity(capacities[type_id], positive_ratio)
        if requested > feasible:
            shortfall = requested - feasible
            shortfall_total += shortfall
            per_type[type_id] = feasible
            capping.append(
                {
                    "type_id": type_id,
                    "requested": requested,
                    "feasible": feasible,
                    "shortfall": shortfall,
                }
            )
    resolution = {
        "status": status,
        "explicit_budget_present": True,
        "allocation_policy": config.allocation_policy,
        "strict_budget": config.strict_budget,
        "resolved_total": sum(per_type.values()),
        "requested_total": debug_cap
        if debug_cap is not None
        else (
            sum(int(config.per_type_budget.get(t, 0)) for t in type_ids)
            if config.per_type_budget
            else config.total_target_qa
        ),
        "per_type": per_type,
        "capacity_capping": capping,
        "shortfall_total": shortfall_total,
        "positive_ratio": positive_ratio,
        "debug_sample": debug_cap is not None,
    }
    if shortfall_total and config.strict_budget:
        resolution["status"] = "BUDGET_INFEASIBLE"
    return resolution

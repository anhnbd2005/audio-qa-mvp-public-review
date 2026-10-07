"""Contract Closure: the closed semantic contract a task must present before
Promotion. After closure, planner/renderer must not invent semantics.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.autonomous_qa.compiler.semantic_tiers import validate_pair


def _canonical(payload: Any) -> str:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


class ContractClosure(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    tier: str
    gold_origin: str
    proposition: str
    operator: str
    source_fields: tuple[str, ...] = ()
    visible_roles: tuple[str, ...] = ()
    hidden_roles: tuple[str, ...] = ()
    answer_schema: dict[str, Any] = Field(default_factory=dict)
    comparator_refs: tuple[str, ...] = ()
    invariances: tuple[str, ...] = ()
    non_invariances: tuple[str, ...] = ()
    row_eligibility_policy: str = "TASK_SCOPED"
    split_policy: dict[str, Any] = Field(default_factory=dict)
    negative_policy: str = "NONE"
    pair_policy: str = "NONE"
    information_value: dict[str, Any] = Field(default_factory=dict)
    capacity: dict[str, Any] = Field(default_factory=dict)
    guardrail_results: tuple[dict[str, Any], ...] = ()
    language_binding: str | None = None
    template_bank_ref: str | None = None
    contract_hash: str = ""

    def compute_hash(self) -> str:
        payload = self.model_dump(mode="json")
        payload.pop("contract_hash", None)
        return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


REQUIRED_NONEMPTY = (
    "task_id",
    "tier",
    "gold_origin",
    "proposition",
    "operator",
)


def close_contract(**fields: Any) -> ContractClosure:
    """Build and validate a closed contract; raises on incompleteness."""
    issues: list[str] = []
    for name in REQUIRED_NONEMPTY:
        if not fields.get(name):
            issues.append(f"MISSING_REQUIRED:{name}")
    issues.extend(validate_pair(fields.get("tier"), fields.get("gold_origin")))
    if issues:
        raise ValueError("CONTRACT_CLOSURE_INCOMPLETE:" + ",".join(issues))
    closure = ContractClosure(**fields)
    return closure.model_copy(update={"contract_hash": closure.compute_hash()})


def closure_complete(closure: ContractClosure) -> list[str]:
    """Return completeness issues for an existing closure."""
    issues: list[str] = []
    if not closure.contract_hash:
        issues.append("MISSING_CONTRACT_HASH")
    if closure.contract_hash and closure.compute_hash() != closure.contract_hash:
        issues.append("CONTRACT_HASH_MISMATCH")
    return issues

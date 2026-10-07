"""SemanticContract Schema Version 2.

Defines the complete semantic truth space (universe) for a task.
Decoupled completely from ProductionPlan sampling policies.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

SEMANTIC_CONTRACT_SCHEMA_VERSION = 2

FORBIDDEN_SAMPLING_FIELDS = frozenset(
    {
        "selected_count",
        "target_count",
        "target_qa_count",
        "ratio",
        "pos_neg_ratio",
        "sampling_seed",
        "random_seed",
        "max_pos_per_anchor",
        "max_neg_per_anchor",
        "sampling_strategy",
        "selected_positive_count",
        "selected_negative_count",
        "planned_qa_count",
    }
)


class InvalidSemanticContractError(ValueError):
    """Raised when a SemanticContract violates core schema rules or contains sampling fields."""


def _canonical(payload: Any) -> str:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


@dataclass(frozen=True)
class SemanticContract:
    schema_version: int = SEMANTIC_CONTRACT_SCHEMA_VERSION
    task_id: str = ""
    tier: str = ""
    topology: str = "ONE_TO_ONE"

    anchor_type: str = "audio_segment"
    candidate_type: str | None = None
    relation_type: str = "DIRECT"

    positive_universe_capacity: int = 0
    negative_universe_capacity: int = 0
    total_universe_capacity: int = 0

    concept_space: str | None = None  # e.g., 'LEXICAL_SURFACE_STRING', 'NORMALIZED_KG_CONCEPT'

    proposition: str = ""
    operator: str = ""
    gold_origin: str = ""
    answer_schema: dict[str, Any] = field(default_factory=dict)
    comparator_ref: str | None = None
    comparator_refs: tuple[str, ...] = ()
    source_roles: tuple[str, ...] = ()
    visible_roles: tuple[str, ...] = ()
    hidden_roles: tuple[str, ...] = ()
    invariances: tuple[str, ...] = ()
    non_invariances: tuple[str, ...] = ()
    row_eligibility_policy: str = "TASK_SCOPED"
    split_policy: dict[str, Any] = field(default_factory=dict)
    negative_policy: str = "NONE"
    pair_policy: str = "NONE"
    information_value: dict[str, Any] = field(default_factory=dict)
    guardrail_results: tuple[dict[str, Any], ...] = ()
    language_binding: str | None = None
    contract_hash: str = ""

    def __post_init__(self) -> None:
        # Enforce no sampling fields allowed in SemanticContract
        for key in self.__dict__:
            if key.lower() in FORBIDDEN_SAMPLING_FIELDS:
                raise InvalidSemanticContractError(
                    f"FORBIDDEN_SAMPLING_FIELD_IN_SEMANTIC_CONTRACT:{key}"
                )

    def compute_hash(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "tier": self.tier,
            "topology": self.topology,
            "anchor_type": self.anchor_type,
            "candidate_type": self.candidate_type,
            "relation_type": self.relation_type,
            "positive_universe_capacity": self.positive_universe_capacity,
            "negative_universe_capacity": self.negative_universe_capacity,
            "total_universe_capacity": self.total_universe_capacity,
            "concept_space": self.concept_space,
            "proposition": self.proposition,
            "operator": self.operator,
            "gold_origin": self.gold_origin,
            "answer_schema": self.answer_schema,
            "comparator_ref": self.comparator_ref,
            "comparator_refs": list(self.comparator_refs),
            "source_roles": list(self.source_roles),
            "visible_roles": list(self.visible_roles),
            "hidden_roles": list(self.hidden_roles),
            "invariances": list(self.invariances),
            "non_invariances": list(self.non_invariances),
            "row_eligibility_policy": self.row_eligibility_policy,
            "split_policy": self.split_policy,
            "negative_policy": self.negative_policy,
            "pair_policy": self.pair_policy,
            "information_value": self.information_value,
            "language_binding": self.language_binding,
        }
        return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()

    def model_dump(self) -> dict[str, Any]:
        d = {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "tier": self.tier,
            "topology": self.topology,
            "anchor_type": self.anchor_type,
            "candidate_type": self.candidate_type,
            "relation_type": self.relation_type,
            "positive_universe_capacity": self.positive_universe_capacity,
            "negative_universe_capacity": self.negative_universe_capacity,
            "total_universe_capacity": self.total_universe_capacity,
            "concept_space": self.concept_space,
            "proposition": self.proposition,
            "operator": self.operator,
            "gold_origin": self.gold_origin,
            "answer_schema": self.answer_schema,
            "comparator_ref": self.comparator_ref,
            "comparator_refs": list(self.comparator_refs),
            "source_roles": list(self.source_roles),
            "visible_roles": list(self.visible_roles),
            "hidden_roles": list(self.hidden_roles),
            "invariances": list(self.invariances),
            "non_invariances": list(self.non_invariances),
            "row_eligibility_policy": self.row_eligibility_policy,
            "split_policy": self.split_policy,
            "negative_policy": self.negative_policy,
            "pair_policy": self.pair_policy,
            "information_value": self.information_value,
            "guardrail_results": list(self.guardrail_results),
            "language_binding": self.language_binding,
            "contract_hash": self.contract_hash or self.compute_hash(),
        }
        return d


def create_semantic_contract(**kwargs: Any) -> SemanticContract:
    """Factory to build and validate a closed SemanticContract."""
    for forbidden in FORBIDDEN_SAMPLING_FIELDS:
        if forbidden in kwargs:
            raise InvalidSemanticContractError(
                f"FORBIDDEN_SAMPLING_FIELD_IN_SEMANTIC_CONTRACT:{forbidden}"
            )
    contract = SemanticContract(**kwargs)
    contract_hash = contract.compute_hash()
    return SemanticContract(**{**kwargs, "contract_hash": contract_hash})

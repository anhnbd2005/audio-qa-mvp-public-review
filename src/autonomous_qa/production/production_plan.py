"""ProductionPlan Schema Version 1 & FrozenInstanceManifest.

The ProductionPlan is the ONLY layer allowed to decide target counts, positive/negative
ratios, per-anchor caps, sampling strategies, and seeds.
It resolves valid instances from the SemanticContract truth space without redefining gold.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from src.autonomous_qa.compiler.semantic_contract import SemanticContract

PRODUCTION_PLAN_SCHEMA_VERSION = 1


class InvalidProductionPlanError(ValueError):
    """Raised when a ProductionPlan requests invalid parameters or exceeds universe bounds."""


def _canonical(payload: Any) -> str:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


@dataclass
class FrozenInstanceManifest:
    manifest_version: int = 1
    dataset_id: str = ""
    task_id: str = ""
    plan_hash: str = ""
    contract_hash: str = ""
    selected_positive_count: int = 0
    selected_negative_count: int = 0
    total_selected_count: int = 0
    instances: list[dict[str, Any]] = field(default_factory=list)

    def compute_hash(self) -> str:
        payload = {
            "manifest_version": self.manifest_version,
            "dataset_id": self.dataset_id,
            "task_id": self.task_id,
            "plan_hash": self.plan_hash,
            "contract_hash": self.contract_hash,
            "selected_positive_count": self.selected_positive_count,
            "selected_negative_count": self.selected_negative_count,
            "total_selected_count": self.total_selected_count,
            "instances": self.instances,
        }
        return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


@dataclass
class ProductionPlan:
    task_id: str
    target_qa_count: int
    pos_neg_ratio: float = 1.0
    max_pos_per_anchor: int = 1
    max_neg_per_anchor: int = 1
    candidate_sampling_strategy: str = "HARD_NEGATIVE_FIRST"
    random_seed: int = 42
    schema_version: int = PRODUCTION_PLAN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "target_qa_count": self.target_qa_count,
            "pos_neg_ratio": self.pos_neg_ratio,
            "max_pos_per_anchor": self.max_pos_per_anchor,
            "max_neg_per_anchor": self.max_neg_per_anchor,
            "candidate_sampling_strategy": self.candidate_sampling_strategy,
            "random_seed": self.random_seed,
            "plan_hash": self.compute_hash(),
        }

    def compute_hash(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "target_qa_count": self.target_qa_count,
            "pos_neg_ratio": self.pos_neg_ratio,
            "max_pos_per_anchor": self.max_pos_per_anchor,
            "max_neg_per_anchor": self.max_neg_per_anchor,
            "candidate_sampling_strategy": self.candidate_sampling_strategy,
            "random_seed": self.random_seed,
        }
        return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()

    def resolve(
        self,
        contract: SemanticContract,
        universe_data: dict[str, Any],
        dataset_id: str = "vimedcss",
    ) -> FrozenInstanceManifest:
        """Resolve a ProductionPlan against a SemanticContract and universe data."""
        if contract.task_id != self.task_id:
            raise InvalidProductionPlanError(
                f"TASK_ID_MISMATCH: plan={self.task_id} vs contract={contract.task_id}"
            )

        # Calculate requested positive / negative targets
        if self.pos_neg_ratio > 0:
            target_pos = round(self.target_qa_count * (self.pos_neg_ratio / (1.0 + self.pos_neg_ratio)))
            target_neg = self.target_qa_count - target_pos
        else:
            target_pos = self.target_qa_count
            target_neg = 0

        # Enforce capacity bounds
        if target_pos > contract.positive_universe_capacity:
            raise InvalidProductionPlanError(
                f"PLANNER_CAPACITY_EXCEEDED: requested pos={target_pos} > universe pos={contract.positive_universe_capacity}"
            )
        if target_neg > contract.negative_universe_capacity:
            raise InvalidProductionPlanError(
                f"PLANNER_CAPACITY_EXCEEDED: requested neg={target_neg} > universe neg={contract.negative_universe_capacity}"
            )

        # Deterministic sampling from universe_data
        pos_universe = universe_data.get("positives", [])
        neg_universe = universe_data.get("negatives", [])

        # Stable hash-based deterministic selection per seed
        def _seed_sort(item: dict[str, Any]) -> str:
            key = f"{self.random_seed}:{item.get('instance_id') or item.get('segment_id') or json.dumps(item)}"
            return hashlib.sha256(key.encode("utf-8")).hexdigest()

        sorted_pos = sorted(pos_universe, key=_seed_sort)
        sorted_neg = sorted(neg_universe, key=_seed_sort)

        selected_pos = sorted_pos[:target_pos]
        selected_neg = sorted_neg[:target_neg]

        instances = []
        for p in selected_pos:
            inst = dict(p)
            inst["semantic_relation"] = "POSITIVE"
            instances.append(inst)

        for n in selected_neg:
            inst = dict(n)
            inst["semantic_relation"] = "NEGATIVE"
            instances.append(inst)

        # Final sort by instance key for determinism
        instances.sort(key=lambda x: str(x.get("instance_id") or x.get("segment_id")))

        manifest = FrozenInstanceManifest(
            dataset_id=dataset_id,
            task_id=self.task_id,
            plan_hash=self.compute_hash(),
            contract_hash=contract.contract_hash,
            selected_positive_count=len(selected_pos),
            selected_negative_count=len(selected_neg),
            total_selected_count=len(instances),
            instances=instances,
        )
        return manifest

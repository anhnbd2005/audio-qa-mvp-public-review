"""ProductionContract dataclass and serializer for P3.1.

Defines the immutable production configuration and task-policy mapping
for autonomous production planning without embedding semantic gold.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

CONTRACT_SCHEMA_VERSION = "p3_1_v1"


def canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_hash(text_or_bytes: str | bytes) -> str:
    if isinstance(text_or_bytes, str):
        text_or_bytes = text_or_bytes.encode("utf-8")
    return hashlib.sha256(text_or_bytes).hexdigest()


@dataclass
class ProductionContract:
    dataset_id: str
    source_revision: str
    production_split: str = "train"
    planner_seed: int = 42
    source_group_field: str = "original_video_link"
    contract_schema_version: str = CONTRACT_SCHEMA_VERSION
    executable_contract_hashes: dict[str, str] = field(default_factory=dict)
    semantic_universe_hashes: dict[str, str] = field(default_factory=dict)
    planner_policy_registry_hash: str = ""
    task_policy_mapping: dict[str, str] = field(default_factory=dict)
    expected_task_targets: dict[str, int] = field(default_factory=dict)
    production_constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_schema_version": self.contract_schema_version,
            "dataset_id": self.dataset_id,
            "source_revision": self.source_revision,
            "production_split": self.production_split,
            "planner_seed": self.planner_seed,
            "source_group_field": self.source_group_field,
            "executable_contract_hashes": self.executable_contract_hashes,
            "semantic_universe_hashes": self.semantic_universe_hashes,
            "planner_policy_registry_hash": self.planner_policy_registry_hash,
            "task_policy_mapping": self.task_policy_mapping,
            "expected_task_targets": self.expected_task_targets,
            "production_constraints": self.production_constraints,
            "contract_hash": self.compute_hash(),
        }

    def compute_hash(self) -> str:
        payload = {
            "contract_schema_version": self.contract_schema_version,
            "dataset_id": self.dataset_id,
            "source_revision": self.source_revision,
            "production_split": self.production_split,
            "planner_seed": self.planner_seed,
            "source_group_field": self.source_group_field,
            "executable_contract_hashes": self.executable_contract_hashes,
            "semantic_universe_hashes": self.semantic_universe_hashes,
            "planner_policy_registry_hash": self.planner_policy_registry_hash,
            "task_policy_mapping": self.task_policy_mapping,
            "expected_task_targets": self.expected_task_targets,
            "production_constraints": self.production_constraints,
        }
        return sha256_hash(canonical_json(payload))

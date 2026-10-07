"""Canonical static-resource registry and generic contract resolvers.

Production resolves dataset contracts ONLY from ``resources/``. There is no
fallback to timestamped R&D outputs (fail fast on absence).

Dataset routing here is configuration (an ID -> path map), not semantic
branching.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from src.common.config import ROOT
from src.autonomous_qa.language.template_renderer import canonical_hash

RESOURCE_ROOT = ROOT / "resources"
DATASET_REGISTRY_PATH = RESOURCE_ROOT / "registry" / "datasets.json"


class CanonicalResourceError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}:{detail}" if detail else code)


class DatasetSpec(BaseModel):
    """Source facts production needs; NOT semantic task definitions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int
    dataset_id: str
    pretty_name: str | None = None
    source: dict[str, Any]
    allowed_splits: tuple[str, ...]
    split_policy: dict[str, Any] = {}
    record_identity: dict[str, Any] = {}
    source_schema: tuple[str, ...] = ()
    semantic_source_fields: dict[str, Any] = {}
    audio_reference_schema: dict[str, Any] = {}
    adapter_id: str
    production_constraints: dict[str, Any] = {}
    provenance: dict[str, Any] = {}

    def logical_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class ProductionContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int
    dataset_id: str
    dataset_spec_hash: str
    semantic_catalog_hash: str
    comparator_registry_hash: str
    comparator_referenced_hashes: dict[str, str]
    language_registry_hash: str
    active_semantic_types: tuple[str, ...]
    production_planner_policy: dict[str, Any] = {}
    promotion_evidence: dict[str, Any] = {}
    status: str = "PROMOTED"
    migration_source: str = "existing_canonical_final"
    promotion_fingerprint: str = ""

    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        payload.pop("promotion_fingerprint", None)
        return canonical_hash(payload)


class PromotionManifest(BaseModel):
    """Records EVIDENCE IDENTITIES; runtime does not load the evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int
    dataset_id: str
    status: str
    migration_source: str
    dataset_spec_hash: str
    semantic_catalog_hash: str
    comparator_registry_hash: str
    language_registry_hash: str
    production_contract_hash: str
    promotion_evidence: dict[str, Any] = {}
    promoted_semantic_types: tuple[str, ...] = ()
    source_revision: str | None = None
    allowed_splits: tuple[str, ...] = ()
    semantic_change_status: str = "NO_SEMANTIC_CHANGE"
    promotion_fingerprint: str = ""

    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        payload.pop("promotion_fingerprint", None)
        return canonical_hash(payload)


def _load_registry() -> dict[str, dict[str, str]]:
    if not DATASET_REGISTRY_PATH.exists():
        raise CanonicalResourceError(
            "CANONICAL_DATASET_REGISTRY_MISSING", str(DATASET_REGISTRY_PATH)
        )
    return json.loads(DATASET_REGISTRY_PATH.read_text(encoding="utf-8"))["datasets"]


def _entry(dataset_id: str) -> dict[str, str]:
    registry = _load_registry()
    if dataset_id not in registry:
        raise CanonicalResourceError("UNKNOWN_DATASET", dataset_id)
    return registry[dataset_id]


def _read_json(rel: str) -> Any:
    path = RESOURCE_ROOT.parent / rel
    if not path.exists():
        raise CanonicalResourceError("CANONICAL_RESOURCE_MISSING", rel)
    return json.loads(path.read_text(encoding="utf-8"))


def get_dataset_spec(dataset_id: str) -> DatasetSpec:
    entry = _entry(dataset_id)
    rel = entry.get("dataset_spec")
    if not rel:
        raise CanonicalResourceError("CANONICAL_DATASET_SPEC_MISSING", dataset_id)
    return DatasetSpec.model_validate(_read_json(rel))


def get_semantic_catalog_path(dataset_id: str) -> Path:
    entry = _entry(dataset_id)
    rel = entry.get("semantic_catalog")
    if not rel:
        raise CanonicalResourceError("CANONICAL_SEMANTIC_CATALOG_MISSING", dataset_id)
    path = RESOURCE_ROOT.parent / rel
    if not path.exists():
        raise CanonicalResourceError("CANONICAL_SEMANTIC_CATALOG_MISSING", rel)
    return path


def get_type_registry_path(dataset_id: str) -> Path | None:
    rel = _entry(dataset_id).get("type_registry")
    if not rel:
        return None
    path = RESOURCE_ROOT.parent / rel
    if not path.exists():
        raise CanonicalResourceError("CANONICAL_TYPE_REGISTRY_MISSING", rel)
    return path


def get_production_contract(dataset_id: str) -> ProductionContract:
    entry = _entry(dataset_id)
    rel = entry.get("production_contract")
    if not rel:
        raise CanonicalResourceError(
            "CANONICAL_PRODUCTION_CONTRACT_MISSING", dataset_id
        )
    return ProductionContract.model_validate(_read_json(rel))


def get_promotion_manifest(dataset_id: str) -> PromotionManifest:
    entry = _entry(dataset_id)
    rel = entry.get("promotion_manifest")
    if not rel:
        raise CanonicalResourceError("CANONICAL_PROMOTION_MANIFEST_MISSING", dataset_id)
    return PromotionManifest.model_validate(_read_json(rel))


def get_semantic_task(dataset_id: str, type_id: str):
    from src.autonomous_qa.compiler.semantic_task import load_semantic_catalog

    catalog = load_semantic_catalog(get_semantic_catalog_path(dataset_id))
    try:
        return catalog.by_type_id(type_id)
    except KeyError:
        raise CanonicalResourceError("UNKNOWN_SEMANTIC_TYPE", f"{dataset_id}:{type_id}")


def get_comparator(comparator_id: str):
    from src.autonomous_qa.compiler.semantic_comparators import load_comparator_registry

    try:
        return load_comparator_registry().by_id(comparator_id)
    except KeyError:
        raise CanonicalResourceError("UNKNOWN_COMPARATOR", comparator_id)


def registered_datasets() -> tuple[str, ...]:
    return tuple(sorted(_load_registry()))

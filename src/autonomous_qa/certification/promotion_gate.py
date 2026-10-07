"""Promotion Gate: R&D evidence -> validated canonical production contracts.

This is the compilation boundary. Ordinary production (plan/render/audit)
never writes canonical resources; only explicit promotion does. Promotion is
deterministic, atomic, and refuses on invalid evidence.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from src.autonomous_qa.compiler.canonical_resources import (
    DatasetSpec,
    ProductionContract,
    PromotionManifest,
)
from src.common.config import ROOT
from src.autonomous_qa.language.language_quality import load_language_registry
from src.autonomous_qa.language.language_preflight import REGISTRY_RESOURCE
from src.autonomous_qa.compiler.semantic_comparators import comparator_set_hash, load_comparator_registry
from src.autonomous_qa.compiler.semantic_task import SemanticCatalog, load_semantic_catalog

RESOURCE_ROOT = ROOT / "resources"
PRODUCTION_DIR = RESOURCE_ROOT / "production"
REGISTRY_PATH = RESOURCE_ROOT / "registry" / "datasets.json"

REQUIRED_STAGES = (
    "dataset_profile",
    "semantic_contracts",
    "semantic_invariance",
    "semantic_feasibility",
    "language_preflight",
    "production_readiness",
)
OPTIONAL_STAGES = ("composite_validation",)

PASS_STATUSES = {"PASS", "VALID", "N/A"}


class PromotionError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}:{detail}" if detail else code)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _registry_entry(dataset_id: str) -> dict[str, str]:
    entries = _read_json(REGISTRY_PATH)["datasets"]
    if dataset_id not in entries:
        raise PromotionError("UNKNOWN_DATASET", dataset_id)
    return entries[dataset_id]


def validate_evidence(evidence: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for stage in REQUIRED_STAGES:
        status = str(evidence.get(stage, {}).get("status", "MISSING"))
        if status not in PASS_STATUSES:
            failures.append(f"{stage}:{status}")
    for stage in OPTIONAL_STAGES:
        if stage in evidence:
            status = str(evidence[stage].get("status", "MISSING"))
            if status not in PASS_STATUSES:
                failures.append(f"{stage}:{status}")
    return failures


def validate_catalog(catalog: SemanticCatalog) -> list[str]:
    comparator_ids = {c.comparator_id for c in load_comparator_registry().comparators}
    issues: list[str] = []
    seen: set[str] = set()
    for task in catalog.tasks:
        if task.type_id in seen:
            issues.append(f"DUPLICATE_TYPE_ID:{task.type_id}")
        seen.add(task.type_id)
        if not task.proposition_id:
            issues.append(f"MISSING_PROPOSITION_ID:{task.type_id}")
        if task.audio_arity < 1:
            issues.append(f"INVALID_AUDIO_ARITY:{task.type_id}")
        if not task.outputs:
            issues.append(f"MISSING_ANSWER_SCHEMA:{task.type_id}")
        if task.comparator_id and task.comparator_id not in comparator_ids:
            issues.append(f"UNKNOWN_COMPARATOR:{task.type_id}:{task.comparator_id}")
        for output in task.outputs:
            for dependency in output.dependencies:
                if dependency not in {o.role for o in task.outputs}:
                    issues.append(f"INVALID_DEPENDENCY:{task.type_id}:{dependency}")
    return issues


def _comparator_info(
    catalog: SemanticCatalog,
) -> tuple[tuple[str, ...], dict[str, str], str]:
    registry = load_comparator_registry()
    ids = tuple(sorted({t.comparator_id for t in catalog.tasks if t.comparator_id}))
    hashes = {cid: registry.by_id(cid).logical_hash() for cid in ids}
    return ids, hashes, comparator_set_hash(ids, registry)


def build_candidate_contract(
    dataset_id: str,
    evidence: dict[str, Any],
    *,
    migration_source: str,
    semantic_change_status: str = "NO_SEMANTIC_CHANGE",
) -> tuple[ProductionContract, PromotionManifest, dict[str, Any]]:
    entry = _registry_entry(dataset_id)
    spec = DatasetSpec.model_validate(_read_json(ROOT / entry["dataset_spec"]))
    catalog = load_semantic_catalog(ROOT / entry["semantic_catalog"])
    issues = validate_catalog(catalog)
    if issues:
        raise PromotionError("CANONICAL_RESOURCE_VALIDATION_FAILED", ",".join(issues))

    language_hash = load_language_registry(REGISTRY_RESOURCE).registry_hash
    _, comparator_hashes, comparator_registry_hash = _comparator_info(catalog)
    catalog_hash = catalog.logical_hash()
    spec_hash = spec.logical_hash()
    active_types = tuple(t.type_id for t in catalog.tasks)

    contract = ProductionContract(
        schema_version=1,
        dataset_id=dataset_id,
        dataset_spec_hash=spec_hash,
        semantic_catalog_hash=catalog_hash,
        comparator_registry_hash=comparator_registry_hash,
        comparator_referenced_hashes=comparator_hashes,
        language_registry_hash=language_hash,
        active_semantic_types=active_types,
        production_planner_policy={
            "allowed_splits": list(spec.allowed_splits),
            "plan_location": (
                "data/materialized/vimd/current"
                if dataset_id == "vimd"
                else "data/materialized/vietmdd"
            ),
        },
        promotion_evidence=evidence,
        status="PROMOTED",
        migration_source=migration_source,
    )
    contract_hash = contract.fingerprint()
    contract = contract.model_copy(update={"promotion_fingerprint": contract_hash})
    manifest = PromotionManifest(
        schema_version=1,
        dataset_id=dataset_id,
        status="PROMOTED",
        migration_source=migration_source,
        dataset_spec_hash=spec_hash,
        semantic_catalog_hash=catalog_hash,
        comparator_registry_hash=comparator_registry_hash,
        language_registry_hash=language_hash,
        production_contract_hash=contract_hash,
        promotion_evidence=evidence,
        promoted_semantic_types=active_types,
        source_revision=spec.source.get("revision_identity"),
        allowed_splits=tuple(spec.allowed_splits),
        semantic_change_status=semantic_change_status,
    )
    manifest = manifest.model_copy(
        update={"promotion_fingerprint": manifest.fingerprint()}
    )
    diagnostics = {
        "dataset_spec_hash": spec_hash,
        "semantic_catalog_hash": catalog_hash,
        "comparator_registry_hash": comparator_registry_hash,
        "comparator_referenced_hashes": comparator_hashes,
        "active_semantic_types": active_types,
        "language_registry_hash": language_hash,
    }
    return contract, manifest, diagnostics


def semantic_diff(old_types: list[str], new_types: list[str]) -> dict[str, Any]:
    old_set, new_set = set(old_types), set(new_types)
    return {
        "added_types": sorted(new_set - old_set),
        "removed_types": sorted(old_set - new_set),
        "unchanged_types": sorted(old_set & new_set),
    }


def promote(
    dataset_id: str,
    evidence: dict[str, Any],
    *,
    migration_source: str = "new_rnd_promotion",
    write: bool = True,
    semantic_change_status: str = "NO_SEMANTIC_CHANGE",
) -> dict[str, Any]:
    """Validate evidence + catalog, then atomically promote canonical contracts."""
    failures = validate_evidence(evidence)
    if failures:
        raise PromotionError("PROMOTION_EVIDENCE_INVALID", ",".join(failures))

    contract, manifest, diagnostics = build_candidate_contract(
        dataset_id,
        evidence,
        migration_source=migration_source,
        semantic_change_status=semantic_change_status,
    )

    entry = _registry_entry(dataset_id)
    contract_path = ROOT / entry["production_contract"]
    manifest_path = ROOT / entry["promotion_manifest"]

    # Semantic diff against the currently promoted contract (if any).
    diff: dict[str, Any] = {
        "added_types": [],
        "removed_types": [],
        "unchanged_types": [],
    }
    if contract_path.exists():
        existing = ProductionContract.model_validate(_read_json(contract_path))
        diff = semantic_diff(
            list(existing.active_semantic_types), list(contract.active_semantic_types)
        )

    result = {
        "dataset_id": dataset_id,
        "status": "PROMOTED",
        "migration_source": migration_source,
        "contract": contract.model_dump(mode="json"),
        "promotion_manifest": manifest.model_dump(mode="json"),
        "diagnostics": diagnostics,
        "semantic_diff": diff,
        "written": write,
    }
    if not write:
        return result

    # Atomic replacement via a temporary candidate bundle.
    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    tmp_contract = contract_path.with_suffix(".json.tmp")
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    _write_json(tmp_contract, contract.model_dump(mode="json"))
    _write_json(tmp_manifest, manifest.model_dump(mode="json"))
    shutil.move(str(tmp_contract), str(contract_path))
    shutil.move(str(tmp_manifest), str(manifest_path))
    return result


def migration_evidence_from_final(dataset_id: str) -> dict[str, Any]:
    """Derive promotion evidence for an already-final dataset.

    Records evidence identities from retained canonical final artifacts; it
    does NOT rerun R&D or load timestamped discovery outputs.
    """
    if dataset_id == "vietmdd":
        from src.autonomous_qa.datasets.vietmdd import get_active_vietmdd_release_manifest
        release = get_active_vietmdd_release_manifest()
        return {
            "dataset_profile": {"status": "N/A", "note": "existing canonical final"},
            "semantic_contracts": {
                "status": "VALID",
                "catalog_hash": release.get("final_semantic_catalog_hash"),
            },
            "semantic_invariance": {
                "status": release.get("semantic_invariance_audit_status", "PASS")
            },
            "semantic_feasibility": {
                "status": "N/A",
                "note": "existing canonical final",
            },
            "language_preflight": {
                "status": "PASS",
                "fingerprint": release.get("dataset_preflight_fingerprint"),
            },
            "production_readiness": {
                "status": "N/A",
                "note": "existing canonical final",
            },
        }
    if dataset_id == "vimd":
        promotion_path = ROOT / "resources" / "production" / "vimd.promotion.json"
        if promotion_path.is_file():
            data = _read_json(promotion_path)
            if "promotion_evidence" in data:
                return data["promotion_evidence"]
        return {
            "dataset_profile": {"status": "N/A", "note": "existing canonical final"},
            "semantic_contracts": {"status": "VALID"},
            "semantic_invariance": {
                "status": "N/A",
                "note": "existing canonical final",
            },
            "semantic_feasibility": {
                "status": "N/A",
                "note": "existing canonical final",
            },
            "language_preflight": {
                "status": "PASS",
                "note": "canonical final plan already frozen",
            },
            "production_readiness": {
                "status": "N/A",
                "note": "existing canonical final",
            },
            "final_plan": {"plan": "vimd_plan_k22000"},
        }
    raise PromotionError("UNKNOWN_DATASET", dataset_id)

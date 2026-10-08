"""Promotion Gate: R&D evidence -> validated canonical production contracts.

This is the compilation boundary. Ordinary production (plan/render/audit)
never writes canonical resources; only explicit promotion does. Promotion is
deterministic, atomic, and refuses on invalid evidence.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.autonomous_qa.compiler.canonical_resources import (
    DatasetSpec,
    ProductionContract,
    PromotionManifest,
)
from src.autonomous_qa.compiler.semantic_comparators import (
    comparator_set_hash,
    load_comparator_registry,
)
from src.autonomous_qa.compiler.semantic_task import (
    SemanticCatalog,
    load_semantic_catalog,
)
from src.autonomous_qa.language.language_preflight import REGISTRY_RESOURCE
from src.autonomous_qa.language.language_quality import load_language_registry
from src.common.config import ROOT

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


def validate_catalog(
    catalog: SemanticCatalog, comparator_ids: set[str] | None = None
) -> list[str]:
    if comparator_ids is None:
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
            "plan_location": spec.production_constraints.get(
                "plan_location", f"data/materialized/{dataset_id}/current"
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
        from src.autonomous_qa.datasets.vietmdd import (
            get_active_vietmdd_release_manifest,
        )
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
            "final_plan": {"plan": "vimd_k22000"},
        }
    raise PromotionError("UNKNOWN_DATASET", dataset_id)


# ---------------------------------------------------------------------------
# DatasetSpec-hash reconciliation (governed rebind)
# ---------------------------------------------------------------------------
#
# A DatasetSpec may legitimately change in policy-only ways (for example the
# authorized four-split QA-generation policy) that do NOT change semantics,
# language templates or gold. Such a change still moves ``logical_hash()`` and
# therefore invalidates ``ProductionContract.dataset_spec_hash`` and the
# ``PromotionManifest`` binding. ``reconcile_dataset_spec_hash`` rebinds ONLY
# those spec-derived fields inside the promotion-gate transaction, atomically,
# with allowed-change enforcement and cross-resource re-validation. Semantic
# catalog, language registry, active types and promotion evidence are preserved
# byte-for-byte (never fabricated).

_SPEC_REBIND_CONTRACT_ALLOWED = frozenset(
    {"dataset_spec_hash", "production_planner_policy", "promotion_fingerprint"}
)
_SPEC_REBIND_MANIFEST_ALLOWED = frozenset(
    {
        "dataset_spec_hash",
        "allowed_splits",
        "production_contract_hash",
        "promotion_fingerprint",
    }
)


def _default_atomic_replace(src: str, dst: str) -> None:
    os.replace(src, dst)


def _write_canonical_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _spec_rebind_registry_entry(resource_root: Path, dataset_id: str) -> dict[str, str]:
    registry_path = resource_root / "registry" / "datasets.json"
    if not registry_path.exists():
        raise PromotionError("CANONICAL_DATASET_REGISTRY_MISSING", str(registry_path))
    entries = _read_json(registry_path)["datasets"]
    if dataset_id not in entries:
        raise PromotionError("UNKNOWN_DATASET", dataset_id)
    return entries[dataset_id]


def _spec_rebind_paths(
    resource_root: Path, dataset_id: str
) -> dict[str, Path]:
    entry = _spec_rebind_registry_entry(resource_root, dataset_id)
    for key in (
        "dataset_spec",
        "semantic_catalog",
        "production_contract",
        "promotion_manifest",
    ):
        if not entry.get(key):
            raise PromotionError("CANONICAL_REGISTRY_ENTRY_INCOMPLETE", f"{dataset_id}:{key}")
    return {
        key: resource_root.parent / entry[key]
        for key in (
            "dataset_spec",
            "semantic_catalog",
            "production_contract",
            "promotion_manifest",
        )
    }


def plan_dataset_spec_reconciliation(
    dataset_id: str,
    *,
    resource_root: Path = RESOURCE_ROOT,
) -> dict[str, Any]:
    """Build the rebind payloads for a policy-only DatasetSpec change (no writes)."""
    paths = _spec_rebind_paths(resource_root, dataset_id)
    spec = DatasetSpec.model_validate(_read_json(paths["dataset_spec"]))
    contract = ProductionContract.model_validate(_read_json(paths["production_contract"]))
    manifest = PromotionManifest.model_validate(_read_json(paths["promotion_manifest"]))
    catalog = load_semantic_catalog(paths["semantic_catalog"])
    registry = load_language_registry(resource_root / "language" / "production_registry.json")

    catalog_hash = catalog.logical_hash()
    catalog_types = tuple(sorted(t.type_id for t in catalog.tasks))
    if contract.semantic_catalog_hash != catalog_hash or manifest.semantic_catalog_hash != catalog_hash:
        raise PromotionError("SPEC_RECONCILE_CATALOG_HASH_MISMATCH", dataset_id)
    if contract.language_registry_hash != registry.registry_hash or manifest.language_registry_hash != registry.registry_hash:
        raise PromotionError("SPEC_RECONCILE_LANGUAGE_HASH_MISMATCH", dataset_id)
    if tuple(sorted(contract.active_semantic_types)) != catalog_types:
        raise PromotionError("SPEC_RECONCILE_CATALOG_TYPE_MISMATCH", dataset_id)
    if tuple(sorted(manifest.promoted_semantic_types)) != catalog_types:
        raise PromotionError("SPEC_RECONCILE_TYPE_SET_MISMATCH", dataset_id)

    spec_hash = spec.logical_hash()

    contract_payload = contract.model_dump(mode="json")
    planner = dict(contract_payload.get("production_planner_policy") or {})
    planner["allowed_splits"] = list(spec.allowed_splits)
    contract_payload["dataset_spec_hash"] = spec_hash
    contract_payload["production_planner_policy"] = planner
    contract_payload["promotion_fingerprint"] = ""
    new_contract = ProductionContract.model_validate(contract_payload)
    new_contract = new_contract.model_copy(
        update={"promotion_fingerprint": new_contract.fingerprint()}
    )

    manifest_payload = manifest.model_dump(mode="json")
    manifest_payload["dataset_spec_hash"] = spec_hash
    manifest_payload["allowed_splits"] = list(spec.allowed_splits)
    manifest_payload["production_contract_hash"] = new_contract.fingerprint()
    manifest_payload["promotion_fingerprint"] = ""
    new_manifest = PromotionManifest.model_validate(manifest_payload)
    new_manifest = new_manifest.model_copy(
        update={"promotion_fingerprint": new_manifest.fingerprint()}
    )

    old_contract_payload = contract.model_dump(mode="json")
    new_contract_payload = new_contract.model_dump(mode="json")
    contract_changed = tuple(
        sorted(k for k in new_contract_payload if old_contract_payload.get(k) != new_contract_payload.get(k))
    )
    if not set(contract_changed) <= _SPEC_REBIND_CONTRACT_ALLOWED:
        raise PromotionError(
            "SPEC_RECONCILE_DISALLOWED_CONTRACT_CHANGES", ",".join(contract_changed)
        )

    old_manifest_payload = manifest.model_dump(mode="json")
    new_manifest_payload = new_manifest.model_dump(mode="json")
    manifest_changed = tuple(
        sorted(k for k in new_manifest_payload if old_manifest_payload.get(k) != new_manifest_payload.get(k))
    )
    if not set(manifest_changed) <= _SPEC_REBIND_MANIFEST_ALLOWED:
        raise PromotionError(
            "SPEC_RECONCILE_DISALLOWED_MANIFEST_CHANGES", ",".join(manifest_changed)
        )

    return {
        "dataset_id": dataset_id,
        "paths": paths,
        "dataset_spec_hash_before": contract.dataset_spec_hash,
        "dataset_spec_hash_after": spec_hash,
        "contract_changed_fields": contract_changed,
        "manifest_changed_fields": manifest_changed,
        "preserved": {
            "semantic_catalog_hash": catalog_hash,
            "language_registry_hash": registry.registry_hash,
            "active_semantic_types": list(contract.active_semantic_types),
            "promotion_evidence": contract.promotion_evidence,
            "promotion_evidence_manifest": manifest.promotion_evidence,
        },
        "old_contract": old_contract_payload,
        "old_manifest": old_manifest_payload,
        "new_contract": new_contract,
        "new_manifest": new_manifest,
    }


def apply_dataset_spec_reconciliation(
    plan: dict[str, Any],
    *,
    resource_root: Path = RESOURCE_ROOT,
    replace_fn: Callable[[str, str], Any] = _default_atomic_replace,
    validation_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Atomically write a prepared spec-reconciliation plan with rollback."""
    dataset_id = plan["dataset_id"]
    paths: dict[str, Path] = plan["paths"]
    contract_path = paths["production_contract"]
    manifest_path = paths["promotion_manifest"]

    for dest in (contract_path, manifest_path):
        try:
            dest.resolve().relative_to(resource_root.resolve())
        except ValueError:
            raise PromotionError("SPEC_RECONCILE_DESTINATION_OUTSIDE_ROOT", str(dest))

    current_contract = _read_json(contract_path)
    current_manifest = _read_json(manifest_path)
    if current_contract != plan["old_contract"] or current_manifest != plan["old_manifest"]:
        raise PromotionError("SPEC_RECONCILE_BUNDLE_STALE", dataset_id)

    new_contract = plan["new_contract"]
    new_manifest = plan["new_manifest"]

    backups: dict[Path, bytes] = {}
    for dest in (contract_path, manifest_path):
        if dest.exists():
            backups[dest] = dest.read_bytes()

    txn = uuid.uuid4().hex
    temps = {dest: dest.parent / f".{dest.name}.{txn}.tmp" for dest in (contract_path, manifest_path)}
    try:
        _write_canonical_json(temps[contract_path], new_contract.model_dump(mode="json"))
        _write_canonical_json(temps[manifest_path], new_manifest.model_dump(mode="json"))
        replace_fn(str(temps[contract_path]), str(contract_path))
        replace_fn(str(temps[manifest_path]), str(manifest_path))

        if validation_hook is not None:
            validation_hook()

        reloaded_contract = ProductionContract.model_validate(_read_json(contract_path))
        reloaded_manifest = PromotionManifest.model_validate(_read_json(manifest_path))
        reloaded_spec = DatasetSpec.model_validate(_read_json(paths["dataset_spec"]))
        reloaded_catalog = load_semantic_catalog(paths["semantic_catalog"])
        reloaded_registry = load_language_registry(resource_root / "language" / "production_registry.json")

        if reloaded_contract.dataset_spec_hash != reloaded_spec.logical_hash():
            raise PromotionError("SPEC_RECONCILE_FINAL_VALIDATION_FAILED", "contract spec hash")
        if reloaded_manifest.dataset_spec_hash != reloaded_spec.logical_hash():
            raise PromotionError("SPEC_RECONCILE_FINAL_VALIDATION_FAILED", "manifest spec hash")
        if reloaded_contract.semantic_catalog_hash != reloaded_catalog.logical_hash():
            raise PromotionError("SPEC_RECONCILE_FINAL_VALIDATION_FAILED", "contract catalog hash")
        if reloaded_manifest.semantic_catalog_hash != reloaded_catalog.logical_hash():
            raise PromotionError("SPEC_RECONCILE_FINAL_VALIDATION_FAILED", "manifest catalog hash")
        if reloaded_contract.language_registry_hash != reloaded_registry.registry_hash:
            raise PromotionError("SPEC_RECONCILE_FINAL_VALIDATION_FAILED", "contract registry hash")
        if reloaded_manifest.language_registry_hash != reloaded_registry.registry_hash:
            raise PromotionError("SPEC_RECONCILE_FINAL_VALIDATION_FAILED", "manifest registry hash")
        if reloaded_manifest.production_contract_hash != reloaded_contract.fingerprint():
            raise PromotionError("SPEC_RECONCILE_FINAL_VALIDATION_FAILED", "manifest contract hash")
        if reloaded_contract.promotion_fingerprint != reloaded_contract.fingerprint():
            raise PromotionError("SPEC_RECONCILE_FINAL_VALIDATION_FAILED", "contract fingerprint")
        if reloaded_manifest.promotion_fingerprint != reloaded_manifest.fingerprint():
            raise PromotionError("SPEC_RECONCILE_FINAL_VALIDATION_FAILED", "manifest fingerprint")
    except Exception as exc:
        try:
            for dest, data in backups.items():
                dest.write_bytes(data)
            for temp in temps.values():
                if temp.exists():
                    temp.unlink()
        except Exception as rb_exc:
            raise PromotionError("SPEC_RECONCILE_ROLLBACK_FAILED", str(rb_exc)) from rb_exc
        raise PromotionError("SPEC_RECONCILE_TRANSACTION_FAILED_ROLLED_BACK", str(exc)) from exc

    return {
        "status": "DATASET_SPEC_HASH_RECONCILED",
        "dataset_id": dataset_id,
        "dataset_spec_hash_before": plan["dataset_spec_hash_before"],
        "dataset_spec_hash_after": plan["dataset_spec_hash_after"],
        "contract_changed_fields": list(plan["contract_changed_fields"]),
        "manifest_changed_fields": list(plan["manifest_changed_fields"]),
        "preserved": plan["preserved"],
    }


def reconcile_dataset_spec_hash(
    dataset_id: str,
    *,
    resource_root: Path = RESOURCE_ROOT,
    replace_fn: Callable[[str, str], Any] = _default_atomic_replace,
    validation_hook: Callable[[], None] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Plan and (optionally) apply a governed DatasetSpec-hash rebind."""
    plan = plan_dataset_spec_reconciliation(dataset_id, resource_root=resource_root)
    if not write:
        return {
            "dataset_id": dataset_id,
            "status": "PLANNED",
            "dataset_spec_hash_before": plan["dataset_spec_hash_before"],
            "dataset_spec_hash_after": plan["dataset_spec_hash_after"],
            "contract_changed_fields": list(plan["contract_changed_fields"]),
            "manifest_changed_fields": list(plan["manifest_changed_fields"]),
            "preserved": plan["preserved"],
        }
    return apply_dataset_spec_reconciliation(
        plan,
        resource_root=resource_root,
        replace_fn=replace_fn,
        validation_hook=validation_hook,
    )

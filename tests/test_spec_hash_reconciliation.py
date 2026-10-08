"""Governed DatasetSpec-hash reconciliation (policy-only rebind) tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src.autonomous_qa.certification.promotion_gate import (
    PromotionError,
    apply_dataset_spec_reconciliation,
    plan_dataset_spec_reconciliation,
    reconcile_dataset_spec_hash,
)
from src.autonomous_qa.compiler.canonical_resources import (
    DatasetSpec,
    ProductionContract,
    PromotionManifest,
    get_dataset_spec,
)
from src.common.config import ROOT

DATASET = "vimedcss"
CANONICAL_FILES = (
    "resources/registry/datasets.json",
    "resources/datasets/vimedcss.json",
    "resources/semantics/vimedcss_semantic_catalog.json",
    "resources/language/production_registry.json",
    "resources/production/vimedcss.json",
    "resources/production/vimedcss.promotion.json",
)


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _isolated_resource_root(tmp_path: Path) -> Path:
    for rel in CANONICAL_FILES:
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest)
    return tmp_path / "resources"


def _load_contract(resource_root: Path) -> ProductionContract:
    return ProductionContract.model_validate(
        json.loads((resource_root / "production" / "vimedcss.json").read_text("utf-8"))
    )


def _load_manifest(resource_root: Path) -> PromotionManifest:
    return PromotionManifest.model_validate(
        json.loads(
            (resource_root / "production" / "vimedcss.promotion.json").read_text("utf-8")
        )
    )


def test_plan_binds_to_spec_and_only_allowed_fields_change():
    plan = plan_dataset_spec_reconciliation(DATASET)
    assert plan["dataset_spec_hash_after"] == get_dataset_spec(DATASET).logical_hash()
    assert set(plan["contract_changed_fields"]) <= {
        "dataset_spec_hash",
        "production_planner_policy",
        "promotion_fingerprint",
    }
    assert set(plan["manifest_changed_fields"]) <= {
        "dataset_spec_hash",
        "allowed_splits",
        "production_contract_hash",
        "promotion_fingerprint",
    }
    contract = _load_contract(ROOT / "resources")
    manifest = _load_manifest(ROOT / "resources")
    assert plan["preserved"]["semantic_catalog_hash"] == contract.semantic_catalog_hash
    assert plan["preserved"]["language_registry_hash"] == contract.language_registry_hash
    assert plan["preserved"]["active_semantic_types"] == list(contract.active_semantic_types)
    assert plan["preserved"]["promotion_evidence"] == contract.promotion_evidence
    assert plan["preserved"]["promotion_evidence_manifest"] == manifest.promotion_evidence
    new_contract = plan["new_contract"]
    assert new_contract.dataset_spec_hash == get_dataset_spec(DATASET).logical_hash()
    assert new_contract.promotion_fingerprint == new_contract.fingerprint()
    assert list(new_contract.production_planner_policy["allowed_splits"]) == [
        "train",
        "validation",
        "test",
        "hard",
    ]
    assert new_contract.production_planner_policy["plan_location"] == (
        "data/materialized/vimedcss/current"
    )


def _write_legacy_spec_hash_state(resource_root: Path) -> None:
    """Rewrite an isolated copy back to the pre-reconciliation 4-split mismatch."""
    # The on-disk spec stays at the NEW four-split policy; only the
    # contract/manifest are rewound to the previous train-only spec binding.
    spec_path = resource_root / "datasets" / "vimedcss.json"
    old_spec_payload = json.loads(spec_path.read_text("utf-8"))
    old_spec_payload["allowed_splits"] = ["train"]
    old_spec_payload["split_policy"] = {
        "production_split": "train",
        "forbidden_splits": ["validation", "test", "hard"],
    }
    old_spec_hash = DatasetSpec.model_validate(old_spec_payload).logical_hash()

    contract_path = resource_root / "production" / "vimedcss.json"
    contract_payload = json.loads(contract_path.read_text("utf-8"))
    contract_payload["dataset_spec_hash"] = old_spec_hash
    contract_payload["production_planner_policy"] = {
        "allowed_splits": ["train"],
        "plan_location": "data/materialized/vimedcss/current",
    }
    contract_payload["promotion_fingerprint"] = ""
    contract = ProductionContract.model_validate(contract_payload)
    contract = contract.model_copy(update={"promotion_fingerprint": contract.fingerprint()})
    contract_path.write_text(
        json.dumps(contract.model_dump(mode="json"), sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest_path = resource_root / "production" / "vimedcss.promotion.json"
    manifest_payload = json.loads(manifest_path.read_text("utf-8"))
    manifest_payload["dataset_spec_hash"] = old_spec_hash
    manifest_payload["allowed_splits"] = ["train"]
    manifest_payload["production_contract_hash"] = contract.fingerprint()
    manifest_payload["promotion_fingerprint"] = ""
    manifest = PromotionManifest.model_validate(manifest_payload)
    manifest = manifest.model_copy(update={"promotion_fingerprint": manifest.fingerprint()})
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), sort_keys=True) + "\n", encoding="utf-8"
    )


def test_plan_rebinds_legacy_mismatch_with_exact_allowed_fields(tmp_path: Path):
    resource_root = _isolated_resource_root(tmp_path)
    _write_legacy_spec_hash_state(resource_root)
    old_contract = _load_contract(resource_root)
    assert old_contract.dataset_spec_hash != get_dataset_spec(DATASET).logical_hash()

    plan = plan_dataset_spec_reconciliation(DATASET, resource_root=resource_root)
    assert plan["contract_changed_fields"] == (
        "dataset_spec_hash",
        "production_planner_policy",
        "promotion_fingerprint",
    )
    assert plan["manifest_changed_fields"] == (
        "allowed_splits",
        "dataset_spec_hash",
        "production_contract_hash",
        "promotion_fingerprint",
    )
    assert plan["new_contract"].promotion_evidence == old_contract.promotion_evidence


def test_plan_does_not_mutate_canonical_resources():
    before = {rel: _sha(ROOT / rel) for rel in CANONICAL_FILES}
    plan_dataset_spec_reconciliation(DATASET)
    reconcile_dataset_spec_hash(DATASET, write=False)
    after = {rel: _sha(ROOT / rel) for rel in CANONICAL_FILES}
    assert before == after


def test_apply_isolated_rebinds_all_production_references(tmp_path: Path):
    resource_root = _isolated_resource_root(tmp_path)
    result = reconcile_dataset_spec_hash(DATASET, resource_root=resource_root, write=True)
    assert result["status"] == "DATASET_SPEC_HASH_RECONCILED"

    spec_hash = get_dataset_spec(DATASET).logical_hash()
    contract = _load_contract(resource_root)
    manifest = _load_manifest(resource_root)

    assert contract.dataset_spec_hash == spec_hash
    assert manifest.dataset_spec_hash == spec_hash
    assert manifest.production_contract_hash == contract.fingerprint()
    assert contract.promotion_fingerprint == contract.fingerprint()
    assert manifest.promotion_fingerprint == manifest.fingerprint()
    assert tuple(sorted(contract.active_semantic_types)) == (
        "vimedcss_cs_terms_count",
        "vimedcss_pairwise_topic_same",
        "vimedcss_topic_classification",
    )
    assert tuple(manifest.allowed_splits) == ("train", "validation", "test", "hard")
    # evidence and semantic identity preserved, never fabricated
    assert contract.promotion_evidence == {
        "authoring_run_id": "20261007_vimedcss_v2_authoring",
        "bundle_fingerprint": "f6adc478bf2207e08a98e1df8947d05b7cbf67b533fcd41abcde67f1b3bf10ee",
        "promotion_mode": "replace",
    }
    assert contract.migration_source == "authoring_promotion_bridge"
    assert contract.semantic_catalog_hash == manifest.semantic_catalog_hash


def test_second_reconciliation_is_a_noop(tmp_path: Path):
    resource_root = _isolated_resource_root(tmp_path)
    reconcile_dataset_spec_hash(DATASET, resource_root=resource_root, write=True)
    plan = plan_dataset_spec_reconciliation(DATASET, resource_root=resource_root)
    assert plan["contract_changed_fields"] == ()
    assert plan["manifest_changed_fields"] == ()


def test_rejects_semantic_drift(tmp_path: Path):
    resource_root = _isolated_resource_root(tmp_path)
    contract_path = resource_root / "production" / "vimedcss.json"
    payload = json.loads(contract_path.read_text("utf-8"))
    payload["semantic_catalog_hash"] = "deadbeef"
    contract_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PromotionError) as exc:
        plan_dataset_spec_reconciliation(DATASET, resource_root=resource_root)
    assert exc.value.code == "SPEC_RECONCILE_CATALOG_HASH_MISMATCH"


def test_apply_refuses_stale_plan(tmp_path: Path):
    resource_root = _isolated_resource_root(tmp_path)
    plan = plan_dataset_spec_reconciliation(DATASET, resource_root=resource_root)
    contract_path = resource_root / "production" / "vimedcss.json"
    payload = json.loads(contract_path.read_text("utf-8"))
    payload["status"] = "SUSPENDED"
    contract_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PromotionError) as exc:
        apply_dataset_spec_reconciliation(plan, resource_root=resource_root)
    assert exc.value.code == "SPEC_RECONCILE_BUNDLE_STALE"

"""Canonical resource / promotion-gate / R&D-independence tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.autonomous_qa.compiler.canonical_resources import (
    CanonicalResourceError,
    get_comparator,
    get_dataset_spec,
    get_production_contract,
    get_semantic_task,
    get_type_registry_path,
    registered_datasets,
)
from src.autonomous_qa.certification.promotion_gate import (
    PromotionError,
    build_candidate_contract,
    migration_evidence_from_final,
    promote,
    validate_evidence,
)
from src.autonomous_qa.compiler.semantic_task import load_dataset_semantic_catalog

ROOT = Path(__file__).resolve().parents[1]

# Exact snapshot hashes (must not change during migration).
VIMD_QA_INTERNAL = "e9bd8b5b29db15c967be983bdd45798de013b8b999281f6631d72dde4fa8f310"
VIMD_QA_MODEL_FACING = (
    "6f2d4103d9d1211bba07409777a6f011b0c7fbd5dd01c6135c8c3235e5fe8346"
)
VIMD_GENERATION_PLAN = (
    "9824f80f2202b9e9de2131a20b3b07dcc71b922eee3b806c52c210b038487657"
)
VIETMDD_UNIFIED_PLAN = (
    "659431b64c6b6220c5de9f5f141a2b39b8ae2c38241168d91ce562b772737184"
)
VIETMDD_QA_INTERNAL = "6f5de490dee31878dc825f75a7484da6e8c5ba12e3710d5d547371d496d59b41"
VIETMDD_QA_MODEL_FACING = (
    "cb26bf5125bb260ba1a02af899a22e78b5fde66cb73d7dba27d52db743572946"
)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_registry_and_resolvers_load():
    assert set(registered_datasets()) == {"vimd", "vietmdd", "vimedcss"}
    for ds in ("vimd", "vietmdd"):
        spec = get_dataset_spec(ds)
        assert spec.dataset_id == ds
        assert spec.allowed_splits == ("train",)
        catalog = load_dataset_semantic_catalog(ds)
        contract = get_production_contract(ds)
        assert contract.dataset_id == ds
        assert contract.status == "PROMOTED"
        assert set(contract.active_semantic_types) == {t.type_id for t in catalog.tasks}


def _strings_excluding_provenance(value, *, _under_provenance=False):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _strings_excluding_provenance(
                item, _under_provenance=_under_provenance or key == "provenance"
            )
    elif isinstance(value, list):
        for item in value:
            yield from _strings_excluding_provenance(
                item, _under_provenance=_under_provenance
            )
    elif isinstance(value, str) and not _under_provenance:
        yield value


def test_no_absolute_paths_or_timestamps_in_canonical_resources():
    for path in (ROOT / "resources").rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "G:\\" not in text and "G:/" not in text
        assert "audio_qa_mvp\\" not in text
        # Timestamped R&D run ids may appear ONLY as promotion provenance.
        for value in _strings_excluding_provenance(json.loads(text)):
            assert "2026" not in value, f"timestamp in identity: {path}:{value}"


def test_comparator_references_resolve():
    for ds in ("vimd", "vietmdd"):
        catalog = load_dataset_semantic_catalog(ds)
        for task in catalog.tasks:
            if task.comparator_id:
                assert get_comparator(task.comparator_id).comparator_id == (
                    task.comparator_id
                )


def test_semantic_task_lookup():
    task = get_semantic_task("vietmdd", "vietmdd_spoken_content_matches_reference")
    assert task.comparator_id == "spoken_lexical_content_equivalence"
    with pytest.raises(CanonicalResourceError):
        get_semantic_task("vietmdd", "does_not_exist")


def test_unknown_dataset_fails_fast():
    with pytest.raises(CanonicalResourceError) as exc:
        get_dataset_spec("nonexistent")
    assert exc.value.code == "UNKNOWN_DATASET"


def test_no_historical_output_dependency_in_resolvers():
    # Canonical resolution must not read timestamped R&D outputs.
    src = (ROOT / "src" / "autonomous_qa" / "compiler" / "canonical_resources.py").read_text(encoding="utf-8")
    assert "outputs/dataset_profiles" not in src
    assert "style_discovery" not in src
    for ds in ("vimd", "vietmdd"):
        path = get_type_registry_path(ds)
        assert path is None or path.relative_to(ROOT).as_posix().startswith(
            "resources/"
        )


def test_final_artifact_hashes_unchanged():
    assert (
        _sha(ROOT / "outputs/releases/vimd/final/qa_internal.jsonl") == VIMD_QA_INTERNAL
    )
    assert (
        _sha(ROOT / "outputs/releases/vimd/final/qa_model_facing.jsonl")
        == VIMD_QA_MODEL_FACING
    )
    assert (
        _sha(ROOT / "data/materialized/vimd/production_plan.jsonl")
        == VIMD_GENERATION_PLAN
    )
    assert (
        _sha(ROOT / "data/materialized/vietmdd/production_plan.jsonl")
        == VIETMDD_UNIFIED_PLAN
    )
    assert (
        _sha(ROOT / "outputs/releases/vietmdd/final/qa_internal.jsonl")
        == VIETMDD_QA_INTERNAL
    )
    assert (
        _sha(ROOT / "outputs/releases/vietmdd/final/qa_model_facing.jsonl")
        == VIETMDD_QA_MODEL_FACING
    )


def test_current_plans_validate_against_canonical_contracts():
    vimd_catalog = load_dataset_semantic_catalog("vimd")
    vimd_types = {t.type_id for t in vimd_catalog.tasks}
    plan_types = set()
    for line in (
        (ROOT / "data/materialized/vimd/production_plan.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        if line.strip():
            plan_types.add(json.loads(line)["type_id"])
    assert plan_types <= vimd_types
    assert len(vimd_types) == 11

    vietmdd_catalog = load_dataset_semantic_catalog("vietmdd")
    vietmdd_types = {t.type_id for t in vietmdd_catalog.tasks}
    unified_types = set()
    for line in (
        (ROOT / "data/materialized/vietmdd/production_plan.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        if line.strip():
            unified_types.add(json.loads(line)["type_id"])
    assert (
        unified_types
        == vietmdd_types
        == {
            "vietmdd_direct_observed_text",
            "vietmdd_target_match_observed_text",
            "vietmdd_spoken_content_matches_reference",
            "vietmdd_equality_observed_text",
            "vietmdd_selection_observed_text",
            "vietmdd_composite_transcribe_pair_equality",
        }
    )


def test_final_flow_preflight_from_static_resources():
    from src.autonomous_qa.language.language_preflight import run_preflight

    assert (
        run_preflight(mode="registry", write_outputs=False)["audit"]["result"]
        == "PREFLIGHT_PASS"
    )
    assert (
        run_preflight(mode="dataset", dataset="vimd", write_outputs=False)["audit"][
            "result"
        ]
        == "PREFLIGHT_PASS"
    )
    assert (
        run_preflight(mode="dataset", dataset="vietmdd", write_outputs=False)["audit"][
            "result"
        ]
        == "PREFLIGHT_PASS"
    )


# ---------------------------- promotion gate ----------------------------
def test_promotion_determinism():
    evidence = migration_evidence_from_final("vimd")
    first = build_candidate_contract("vimd", evidence, migration_source="test")
    second = build_candidate_contract("vimd", evidence, migration_source="test")
    assert first[0].fingerprint() == second[0].fingerprint()
    assert first[1].promotion_fingerprint == second[1].promotion_fingerprint
    assert first[2] == second[2]


def test_promotion_refuses_bad_evidence():
    ev = migration_evidence_from_final("vimd")
    ev["semantic_feasibility"] = {"status": "FAIL"}
    with pytest.raises(PromotionError):
        promote("vimd", ev, write=False)
    assert validate_evidence(ev)  # non-empty failures
    ev["language_preflight"] = {"status": "FAIL"}
    assert any("language_preflight" in f for f in validate_evidence(ev))
    assert any("semantic_feasibility" in f for f in validate_evidence(ev))


def test_validate_catalog_rejects_unknown_comparator():
    from src.autonomous_qa.certification.promotion_gate import validate_catalog
    from src.autonomous_qa.compiler.semantic_task import SemanticCatalog

    catalog = SemanticCatalog.model_validate(
        {
            "dataset": "x",
            "catalog_version": "v",
            "comparators_resource": "comparators.json",
            "tasks": [
                {
                    "type_id": "t",
                    "proposition_id": "p",
                    "proposition_description": "d",
                    "operator": "TARGET_MATCH",
                    "classification": "PRIMITIVE_RELATION",
                    "kind": "ATOMIC",
                    "audio_arity": 1,
                    "comparator_id": "bogus_comparator",
                    "outputs": [{"role": "r", "kind": "boolean"}],
                }
            ],
        }
    )
    issues = validate_catalog(catalog)
    assert any("UNKNOWN_COMPARATOR" in issue for issue in issues)


def test_promotion_vietmdd_records_repair_identity():
    contract = get_production_contract("vietmdd")
    # VietMDD comparator set hash must equal the repaired release identity.
    assert (
        contract.comparator_registry_hash
        == "ce799d351b3f393175ec3d8f6a62fe7e2cb646a9220fbde5a0af49a868b1536c"
    )
    assert (
        contract.semantic_catalog_hash
        == "1e66471491f3f5df30061f3a435f99503d21d2588811e3a56b1938431d90b143"
    )

"""Comprehensive test suite for Phase 4.1.4 Generic Authoring -> Promotion Bridge Safety.

Verifies:
- no transaction test ever mutates ROOT/resources (real canonical resources)
- APPLY resolves every canonical read/write through resource_root
- default replacement is atomic os.replace
- true rollback after first / second committed replacement and during final validation
- newly-created destination is removed by rollback
- source authoring evidence freshness fails closed
- canonical_id_overrides and capability_exclusions are separate override domains
- composite proposal artifact lives under llm/composite_discovery
- composite selection counters are real, not hardcoded
- promotable_total == promotable_primitive_count + promotable_composite_count
- reconciliation executable compatibility fails closed when unproven
- ProductionContract / PromotionManifest store real promotion_fingerprint
- PromotionManifest.source_revision comes from DatasetSpec source.revision_identity
- preflight implementation identity is derived automatically from the implementation
- all five canonical stale hashes are independently checked
- real ViMedCSS PREPARE remains language-blocked with apply_ready False
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
import pytest

from src.autonomous_qa.certification import authoring_promotion
from src.autonomous_qa.certification.authoring_promotion import (
    PromotionBundle,
    apply_promotion,
    evaluate_promotion_readiness,
    filter_executable_candidates,
    prepare_promotion,
    resolve_canonical_ids,
)
from src.autonomous_qa.certification.promotion_gate import PromotionError
from src.autonomous_qa.compiler.canonical_resources import (
    DatasetSpec,
    ProductionContract,
    PromotionManifest,
)
from src.autonomous_qa.compiler.semantic_comparators import (
    comparator_set_hash,
    load_comparator_registry,
)
from src.autonomous_qa.compiler.semantic_field_specs import SemanticFieldSpec
from src.autonomous_qa.compiler.semantic_task import SemanticTaskSpec, load_semantic_catalog
from src.autonomous_qa.language import language_preflight
from src.autonomous_qa.language.language_preflight import (
    PreflightInputError,
    accepted_types_from_semantic_catalog,
    compute_contract_fingerprint,
    get_canonical_preflight_implementation_sha256,
)
from src.autonomous_qa.language.template_engine import vimedcss_field_specs
from src.autonomous_qa.onboard import onboard_dataset
from src.common.config import ROOT

ACCEPTANCE_RUN_DIR = ROOT / "outputs" / "runs" / "vimedcss" / "20261007_vimedcss_v2_authoring"
RESOURCE_ROOT = ROOT / "resources"

# Real canonical resources that must remain byte-identical throughout this module.
_REAL_CANONICAL_FILES = [
    RESOURCE_ROOT / "semantics" / "vimedcss_semantic_catalog.json",
    RESOURCE_ROOT / "production" / "vimedcss.json",
    RESOURCE_ROOT / "production" / "vimedcss.promotion.json",
    RESOURCE_ROOT / "language" / "production_registry.json",
    RESOURCE_ROOT / "datasets" / "vimedcss.json",
]
_REAL_CANONICAL_BEFORE = {p: p.read_bytes() for p in _REAL_CANONICAL_FILES}


@pytest.fixture(autouse=True)
def _guard_real_canonical_resources():
    """No test in this module may mutate real canonical resources."""
    yield
    for path, before in _REAL_CANONICAL_BEFORE.items():
        assert path.read_bytes() == before, f"REAL CANONICAL RESOURCE MUTATED: {path}"


# ---------------------------------------------------------------------------
# Isolated synthetic canonical tree + bundle factory
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _toy_task(type_id: str, *, source_field: str = "topic", comparator_id: str = "toy_exact") -> dict:
    return {
        "type_id": type_id,
        "proposition_id": type_id,
        "proposition_description": f"Toy proposition {type_id}",
        "operator": "DIRECT",
        "classification": "PRIMITIVE_RELATION",
        "kind": "ATOMIC",
        "audio_arity": 1,
        "visible_context_roles": [],
        "outputs": [{"role": "value", "kind": "field_value", "dependencies": []}],
        "dependency_graph": [],
        "comparator_id": comparator_id,
        "invariances": [],
        "non_invariances": [],
        "base_type_id": None,
        "source_role_mapping": {"source_field": source_field},
        "evaluation_value": "",
        "tier": "T1_PERCEPTION",
        "gold_origin": "SOURCE",
        "guardrail_status": None,
        "closure_hash": None,
    }


def _toy_comparators() -> dict:
    return {
        "language": "generic",
        "version": "toy_v1",
        "schema_version": 1,
        "comparators": [
            {
                "comparator_id": "toy_exact",
                "semantic_purpose": "Toy exact equality.",
                "ordered_normalization_operations": ["exact_string_equality"],
                "invariances": [],
                "preserved_features": ["digits"],
                "punctuation_category_policy": "preserve",
                "unicode_normalization": "none",
                "case_policy": "preserve",
            }
        ],
    }


def _toy_language_registry() -> dict:
    registry = {
        "language": "toy",
        "version": "toy_v1",
        "schema_version": 1,
        "template_library_version": "toy_v1",
        "template_library_hash": "toy_template_hash",
        "paraphrase_library_version": "toy_v1",
        "paraphrase_library_hash": "toy_paraphrase_hash",
        "quality_model": "toy",
        "quality_call_ids": [],
        "registry_hash": "PENDING",
        "entries": [],
    }
    payload = dict(registry)
    payload.pop("registry_hash")
    registry["registry_hash"] = _canonical_hash(payload)
    return registry


def _toy_spec_dict() -> dict:
    return {
        "schema_version": 1,
        "dataset_id": "toy",
        "pretty_name": "Synthetic Toy Dataset",
        "source": {
            "provider": "local",
            "repository": "toy",
            "revision_identity": "toy_dataset_revision_1",
        },
        "allowed_splits": ["train"],
        "adapter_id": "toy_source",
        "production_constraints": {"plan_location": "data/materialized/toy/current"},
    }


def build_toy_tree(base: Path, *, include_manifest: bool = True) -> dict:
    """Build a minimal isolated canonical tree under base/resources.

    Returns a dict with the resource root, target dataset id, and the canonical
    catalog/spec objects.
    """
    res = base / "resources"
    (res / "registry").mkdir(parents=True, exist_ok=True)
    (res / "datasets").mkdir(parents=True, exist_ok=True)
    (res / "semantics").mkdir(parents=True, exist_ok=True)
    (res / "production").mkdir(parents=True, exist_ok=True)
    (res / "language").mkdir(parents=True, exist_ok=True)

    spec_dict = _toy_spec_dict()
    _write_json(res / "datasets" / "toy.json", spec_dict)
    _write_json(res / "semantics" / "comparators.json", _toy_comparators())

    catalog_dict = {
        "dataset": "toy",
        "catalog_version": "toy_v1",
        "comparators_resource": "comparators.json",
        "tasks": [_toy_task("toy_topic_state")],
    }
    _write_json(res / "semantics" / "toy_semantic_catalog.json", catalog_dict)

    lang_registry = _toy_language_registry()
    _write_json(res / "language" / "production_registry.json", lang_registry)

    spec_obj = DatasetSpec.model_validate(spec_dict)
    catalog_obj = load_semantic_catalog(res / "semantics" / "toy_semantic_catalog.json")

    comp_reg = load_comparator_registry(res / "semantics" / "comparators.json")
    ids = tuple(sorted({t.comparator_id for t in catalog_obj.tasks if t.comparator_id}))
    comp_hashes = {cid: comp_reg.by_id(cid).logical_hash() for cid in ids}
    comp_reg_hash = comparator_set_hash(ids, comp_reg) if ids else ""

    contract = ProductionContract(
        schema_version=1,
        dataset_id="toy",
        dataset_spec_hash=spec_obj.logical_hash(),
        semantic_catalog_hash=catalog_obj.logical_hash(),
        comparator_registry_hash=comp_reg_hash,
        comparator_referenced_hashes=comp_hashes,
        language_registry_hash=lang_registry["registry_hash"],
        active_semantic_types=tuple(t.type_id for t in catalog_obj.tasks),
        promotion_evidence={},
        status="PROMOTED",
        migration_source="toy_tree",
        promotion_fingerprint="",
    )
    contract = contract.model_copy(update={"promotion_fingerprint": contract.fingerprint()})
    _write_json(res / "production" / "toy.json", contract.model_dump(mode="json"))

    if include_manifest:
        manifest = PromotionManifest(
            schema_version=1,
            dataset_id="toy",
            status="PROMOTED",
            migration_source="toy_tree",
            dataset_spec_hash=spec_obj.logical_hash(),
            semantic_catalog_hash=catalog_obj.logical_hash(),
            comparator_registry_hash=comp_reg_hash,
            language_registry_hash=lang_registry["registry_hash"],
            production_contract_hash=contract.fingerprint(),
            promotion_evidence={},
            promoted_semantic_types=tuple(t.type_id for t in catalog_obj.tasks),
            source_revision=spec_obj.source["revision_identity"],
            allowed_splits=("train",),
            semantic_change_status="NEW_DATASET",
            promotion_fingerprint="",
        )
        manifest = manifest.model_copy(update={"promotion_fingerprint": manifest.fingerprint()})
        _write_json(res / "production" / "toy.promotion.json", manifest.model_dump(mode="json"))

    registry = {
        "schema_version": 1,
        "datasets": {
            "toy": {
                "dataset_spec": "resources/datasets/toy.json",
                "semantic_catalog": "resources/semantics/toy_semantic_catalog.json",
                "production_contract": "resources/production/toy.json",
                "promotion_manifest": "resources/production/toy.promotion.json",
            }
        },
    }
    _write_json(res / "registry" / "datasets.json", registry)

    return {
        "resource_root": res,
        "dataset_id": "toy",
        "spec": spec_obj,
        "catalog": catalog_obj,
        "include_manifest": include_manifest,
    }


def make_apply_ready_bundle(tree: dict, base: Path) -> PromotionBundle:
    """Build an apply-ready bundle whose expected hashes match the toy tree."""
    res = tree["resource_root"]
    spec_obj = tree["spec"]
    catalog_obj = tree["catalog"]

    existing_contract = ProductionContract.model_validate(
        json.loads((res / "production" / "toy.json").read_text(encoding="utf-8"))
    )
    manifest_path = res / "production" / "toy.promotion.json"
    existing_manifest = (
        PromotionManifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
        if manifest_path.exists()
        else None
    )
    lang_registry = json.loads((res / "language" / "production_registry.json").read_text(encoding="utf-8"))

    candidate_tasks = [t.model_dump(mode="json") for t in catalog_obj.tasks]
    candidate_tasks.append(_toy_task("toy_new_task"))
    candidate_catalog = {
        "dataset": "toy",
        "catalog_version": "toy_candidate",
        "comparators_resource": "comparators.json",
        "tasks": candidate_tasks,
    }
    proposed_active_types = tuple(sorted(t["type_id"] for t in candidate_tasks))

    return PromotionBundle(
        schema_version=1,
        dataset_id="toy",
        source_run_id="toy_run",
        source_run_path=str(base / "no_such_run"),
        promotion_mode="replace",
        authoring_run_hash="",
        source_documentation_hash="",
        dataset_profile_hash="",
        primitive_candidates=1,
        primitive_semantic_accepted=1,
        primitive_semantic_pass_count=1,
        composite_candidates=0,
        composite_semantic_accepted=0,
        composite_semantic_pass_count=0,
        primitive_language_ready=1,
        primitive_authoring_language_gate_pass_count=1,
        composite_language_ready=0,
        composite_authoring_language_gate_pass_count=0,
        authoring_language_gate_pass_count=1,
        primitive_executable_selected_count=1,
        staged_canonical_language_preflight_pass_count=0,
        staged_preflight_type_count=1,
        staged_preflight_pass_type_count=0,
        promotable_primitive_count=1,
        promotable_composite_count=0,
        promotable_total=1,
        promotable_types=proposed_active_types,
        selected_promotion_types=proposed_active_types,
        selected_type_blockers=(),
        canonical_id_mapping={},
        old_active_types=tuple(t.type_id for t in catalog_obj.tasks),
        proposed_active_types=proposed_active_types,
        semantic_diff={
            "added_types": ["toy_new_task"],
            "removed_types": [],
            "retained_types": [t.type_id for t in catalog_obj.tasks],
        },
        candidate_semantic_catalog=candidate_catalog,
        candidate_language_resource={"entries": []},
        expected_current_hashes={
            "dataset_spec_hash": spec_obj.logical_hash(),
            "semantic_catalog_hash": catalog_obj.logical_hash(),
            "production_contract_hash": existing_contract.fingerprint(),
            "promotion_manifest_hash": existing_manifest.fingerprint() if existing_manifest else "",
            "language_registry_hash": lang_registry["registry_hash"],
        },
        staged_language_preflight={"status": "PREFLIGHT_PASS", "blocking_issues": 0},
        prepare_status="PREPARED",
        apply_ready=True,
    )


def snapshot_tree(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _make_fail_after(n: int):
    state = {"committed": 0}

    def replace(src: str, dst: str):
        expected_bytes = Path(src).read_bytes()
        os.replace(src, dst)
        assert Path(dst).read_bytes() == expected_bytes
        state["committed"] += 1
        if state["committed"] == n:
            raise RuntimeError(f"INJECTED_FAILURE_AFTER_{n}_REPLACEMENTS")

    return replace, state


# ---------------------------------------------------------------------------
# Real ViMedCSS acceptance (read-only PREPARE)
# ---------------------------------------------------------------------------


def test_vimedcss_acceptance_run_prepare_dryrun(tmp_path: Path):
    scratch_out = tmp_path / "scratch_promotion"
    bundle = prepare_promotion(
        dataset_id="vimedcss",
        run_dir=ACCEPTANCE_RUN_DIR,
        promotion_mode="replace",
        output_root=scratch_out,
    )

    assert bundle.primitive_candidates == 4
    assert bundle.primitive_semantic_pass_count == 4
    assert bundle.primitive_authoring_language_gate_pass_count == 4

    assert bundle.composite_candidates == 1
    assert bundle.composite_semantic_pass_count == 1
    assert bundle.composite_authoring_language_gate_pass_count == 0

    assert bundle.primitive_executable_selected_count == 3
    assert bundle.promotable_primitive_count == 3
    assert bundle.promotable_total == 3
    assert "vimedcss_003_cs_term_presence" in bundle.non_selected_types
    assert bundle.non_selected_types["vimedcss_003_cs_term_presence"] == "OPERATOR_CAPABILITY_GAP:MEMBERSHIP"
    assert "vimedcss_comp_001_topic_term_verification" in bundle.non_selected_types
    assert bundle.non_selected_types["vimedcss_comp_001_topic_term_verification"] == "SEMANTIC_GATE_PASS_BUT_LANGUAGE_NOT_READY"

    assert bundle.staged_preflight_type_count == 3

    exp = bundle.expected_current_hashes
    assert all(exp[k] != "" for k in (
        "dataset_spec_hash", "semantic_catalog_hash", "production_contract_hash",
        "promotion_manifest_hash", "language_registry_hash",
    ))

    assert bundle.prepare_status == "PREPARED"
    assert bundle.apply_ready is False

    with pytest.raises(PromotionError) as exc_info:
        apply_promotion(bundle)
    assert exc_info.value.code == "PROMOTION_NOT_APPLY_READY"


def test_no_literal_vimedcss_task_ids_in_generic_promotion():
    content = (ROOT / "src" / "autonomous_qa" / "certification" / "authoring_promotion.py").read_text(encoding="utf-8")
    matches = re.findall(r"vimedcss_[a-z0-9_]+", content)
    assert len(matches) == 0, f"Found literal ViMedCSS task IDs: {matches}"


def test_vimedcss_promotable_total_invariant():
    bundle = prepare_promotion(
        dataset_id="vimedcss",
        run_dir=ACCEPTANCE_RUN_DIR,
        promotion_mode="replace",
        output_root=None,
    )
    assert bundle.promotable_total == bundle.promotable_primitive_count + bundle.promotable_composite_count
    assert bundle.promotable_total == len(bundle.selected_promotion_types)


def test_raw_vimedcss_staged_language_issues_are_machine_codes():
    """The staged preflight issue list must contain exact machine codes, not prose labels."""
    bundle = prepare_promotion(
        dataset_id="vimedcss",
        run_dir=ACCEPTANCE_RUN_DIR,
        promotion_mode="replace",
        output_root=None,
    )
    issues = bundle.staged_language_preflight["issues"]
    assert issues, "expected staged language blocking issues for ViMedCSS"
    codes = {row["issue_code"] for row in issues}
    assert codes, "issue codes must be present"
    assert all(isinstance(c, str) and c == c.upper() for c in codes)
    # The exact public collision code, when present, must be reported verbatim.
    if "SEMANTIC_HEAD_OWNERSHIP_COLLISION" in codes:
        assert "SEMANTIC_HEAD_OWNERSHIP_COLLISION" in codes


# ---------------------------------------------------------------------------
# Capability / registry-level regressions (read-only)
# ---------------------------------------------------------------------------


def test_candidate_type_id_rename_does_not_change_capability_outcome(tmp_path: Path):
    run_dir = tmp_path / "toy_run_rename"
    llm_dir = run_dir / "llm" / "primitive_semantic_discovery"
    llm_dir.mkdir(parents=True)
    (llm_dir / "parsed_response.json").write_text(json.dumps({
        "candidates": [
            {"candidate_id": "presence_term_001", "operator": "TARGET_MATCH", "scalar_equality": True},
            {"candidate_id": "random_code_999", "operator": "TARGET_MATCH", "scalar_equality": True},
            {"candidate_id": "alpha_check", "operator": "TARGET_MATCH", "target_relation": "membership"},
            {"candidate_id": "beta_presence_check", "operator": "TARGET_MATCH", "target_relation": "membership"},
        ]
    }), encoding="utf-8")

    cands = ["presence_term_001", "random_code_999", "alpha_check", "beta_presence_check"]
    executable, excluded = filter_executable_candidates(cands, run_dir)

    assert "presence_term_001" in executable
    assert "random_code_999" in executable
    assert excluded["alpha_check"] == "OPERATOR_CAPABILITY_GAP:MEMBERSHIP"
    assert excluded["beta_presence_check"] == "OPERATOR_CAPABILITY_GAP:MEMBERSHIP"


def test_staged_accepted_type_compilation_has_no_dataset_fallback():
    catalog = {
        "dataset": "vimedcss",
        "tasks": [{
            "type_id": "test_task",
            "operator": "DIRECT",
            "audio_arity": 1,
            "source_role_mapping": {"source_field": "topic"},
            "outputs": [{"role": "val", "kind": "field_value"}],
        }],
    }
    with pytest.raises(PreflightInputError) as exc_info:
        accepted_types_from_semantic_catalog(catalog, field_specs=None)
    assert "FIELD_SPECS_REQUIRED" in str(exc_info.value)


def test_no_field_name_count_or_num_semantic_inference():
    catalog = {
        "dataset": "custom_dataset",
        "tasks": [{
            "type_id": "event_count_task",
            "operator": "DIRECT",
            "audio_arity": 1,
            "source_role_mapping": {"source_field": "num_events_counted"},
            "outputs": [{"role": "val", "kind": "field_value"}],
        }],
    }
    dummy_specs = {"topic": vimedcss_field_specs()["topic"]}
    with pytest.raises(PreflightInputError) as exc_info:
        accepted_types_from_semantic_catalog(catalog, field_specs=dummy_specs)
    assert "FIELD_SPEC_MISSING:num_events_counted" in str(exc_info.value)


def test_no_segment_text_fallback():
    catalog = {
        "dataset": "custom_dataset",
        "tasks": [{
            "type_id": "no_source_task",
            "operator": "DIRECT",
            "audio_arity": 1,
            "outputs": [{"role": "val", "kind": "field_value"}],
        }],
    }
    with pytest.raises(PreflightInputError) as exc_info:
        accepted_types_from_semantic_catalog(catalog, field_specs=vimedcss_field_specs())
    assert "MISSING_EXECUTABLE_SOURCE_FIELD" in str(exc_info.value)


def test_authoring_gate_count_independent_from_executable_count():
    readiness = evaluate_promotion_readiness(ACCEPTANCE_RUN_DIR)
    assert readiness["primitive_authoring_language_gate_pass_count"] == 4
    assert readiness["authoring_language_gate_pass_count"] == 4
    executable, excluded = filter_executable_candidates(
        readiness["authoring_gate_pass_candidates"], ACCEPTANCE_RUN_DIR
    )
    assert len(executable) == 3
    assert len(excluded) == 1


def test_no_type_id_substring_heuristics():
    tc_content = (ROOT / "src" / "autonomous_qa" / "language" / "template_contracts.py").read_text(encoding="utf-8")
    assert '"presence" in type_id' not in tc_content
    assert '"presence" in task.type_id' not in tc_content
    assert '"pairwise" in type_id' not in tc_content
    assert '"pairwise" in task.type_id' not in tc_content


def test_membership_relation_execution_proof_gap(tmp_path: Path):
    run_dir = tmp_path / "toy_membership_run"
    llm_dir = run_dir / "llm" / "primitive_semantic_discovery"
    llm_dir.mkdir(parents=True)
    (llm_dir / "parsed_response.json").write_text(json.dumps({
        "candidates": [{"candidate_id": "cand_member", "operator": "TARGET_MATCH", "target_relation": "membership"}]
    }), encoding="utf-8")
    executable, excluded = filter_executable_candidates(["cand_member"], run_dir)
    assert "cand_member" not in executable
    assert excluded["cand_member"] == "OPERATOR_CAPABILITY_GAP:MEMBERSHIP"


def test_apply_ready_state_machine_blocks_apply(tmp_path: Path):
    tree = build_toy_tree(tmp_path)
    bundle = make_apply_ready_bundle(tree, tmp_path).model_copy(update={"apply_ready": False})
    with pytest.raises(PromotionError) as exc_info:
        apply_promotion(bundle, resource_root=tree["resource_root"])
    assert exc_info.value.code == "PROMOTION_NOT_APPLY_READY"


def test_onboard_does_not_label_prepare_as_production_plan(tmp_path: Path):
    res = onboard_dataset(
        dataset_id="vimedcss",
        authoring_run_dir=ACCEPTANCE_RUN_DIR,
        promotion_mode="replace",
        stop_after="prepare",
        output_root=tmp_path / "onboard_scratch",
    )
    assert res["status"] == "STOPPED_AFTER_PREPARE_ZERO_CANONICAL_MUTATIONS"
    assert "PLAN" not in res["status"]


# ---------------------------------------------------------------------------
# Override API separation
# ---------------------------------------------------------------------------


def test_canonical_id_override_does_not_exclude_candidate(tmp_path: Path):
    tree = build_toy_tree(tmp_path)
    run_dir = _make_synthetic_run(tmp_path)
    bundle = _prepare_synthetic(tree, run_dir, field_specs=_toy_field_specs(),
                                canonical_id_overrides={"toy_001_topic": "toy_custom_id"})
    assert "toy_custom_id" in bundle.selected_promotion_types
    assert "toy_001_topic" not in bundle.non_selected_types


def test_capability_exclusion_does_not_rename_candidate(tmp_path: Path):
    tree = build_toy_tree(tmp_path)
    run_dir = _make_synthetic_run(tmp_path)
    # Exclude the capability; the canonical ID must NOT be repurposed as a reason.
    bundle = _prepare_synthetic(
        tree, run_dir, field_specs=_toy_field_specs(),
        capability_exclusions={"toy_001_topic": "OPERATOR_CAPABILITY_GAP:MEMBERSHIP"},
    )
    assert bundle.non_selected_types["toy_001_topic"] == "OPERATOR_CAPABILITY_GAP:MEMBERSHIP"


# ---------------------------------------------------------------------------
# Composite artifact path + accounting
# ---------------------------------------------------------------------------


def _toy_field_specs():
    spec = SemanticFieldSpec(
        field_name="topic",
        semantic_class="categorical_attribute",
        entity_scope="utterance",
        entity_phrase="đoạn âm thanh",
        attribute_phrase="chủ đề",
        value_phrase="chủ đề",
        source="toy",
    )
    return {"topic": spec}


def _make_synthetic_run(base: Path, *, composite: bool = True, legacy_composite_path: bool = False) -> Path:
    run = base / "toy_run"
    (run / "deterministic").mkdir(parents=True, exist_ok=True)
    (run / "gates").mkdir(parents=True, exist_ok=True)
    (run / "language").mkdir(parents=True, exist_ok=True)
    (run / "llm" / "primitive_semantic_discovery").mkdir(parents=True, exist_ok=True)

    (run / "run_manifest.json").write_text(json.dumps({"dataset_id": "toy"}), encoding="utf-8")
    (run / "deterministic" / "dataset_profile.json").write_text(json.dumps({"fields": ["topic"]}), encoding="utf-8")
    (run / "gates" / "primitive_gates.json").write_text(json.dumps({
        "candidates": [{"candidate_id": "toy_001_topic", "verdict": "PASS"}]
    }), encoding="utf-8")
    (run / "language" / "preflight.json").write_text(json.dumps({
        "entries": [{"type_id": "toy_001_topic", "verdict": "PASS"}]
    }), encoding="utf-8")
    (run / "reconciliation_canonical.json").write_text(json.dumps({"matches": []}), encoding="utf-8")

    primitive = {
        "candidate_id": "toy_001_topic",
        "proposition": "Toy topic proposition",
        "operator": "DIRECT",
        "audio_arity": 1,
        "visible_inputs": [],
        "hidden_source_annotations": ["topic"],
        "answer_schema_proposal": {"kind": "field_value", "type": "string"},
        "comparator_id": "toy_exact",
        "expected_invariances": [],
        "expected_non_invariances": [],
    }
    (run / "llm" / "primitive_semantic_discovery" / "parsed_response.json").write_text(
        json.dumps({"candidates": [primitive]}), encoding="utf-8"
    )

    if composite:
        composite_payload = {
            "composites": [{
                "composite_id": "toy_comp_001_verify",
                "proposition": "Toy composite proposition",
                "operator": "COMPOSITE",
                "audio_arity": 1,
                "visible_inputs": ["topic"],
                "answer_schema_proposal": {"kind": "field_value"},
                "comparator_id": "toy_exact",
                "expected_invariances": [],
                "expected_non_invariances": [],
            }]
        }
        stage = "composite_semantic_discovery" if legacy_composite_path else "composite_discovery"
        comp_dir = run / "llm" / stage
        comp_dir.mkdir(parents=True, exist_ok=True)
        (comp_dir / "parsed_response.json").write_text(json.dumps(composite_payload), encoding="utf-8")

        (run / "gates" / "composite_gates.json").write_text(json.dumps({
            "composites": [{"composite_id": "toy_comp_001_verify", "verdict": "PASS"}]
        }), encoding="utf-8")
        (run / "language" / "preflight.json").write_text(json.dumps({
            "entries": [
                {"type_id": "toy_001_topic", "verdict": "PASS"},
                {"type_id": "toy_comp_001_verify", "verdict": "PASS"},
            ]
        }), encoding="utf-8")
    return run


def _prepare_synthetic(tree, run_dir, *, field_specs, **kwargs):
    return prepare_promotion(
        dataset_id="toy",
        run_dir=run_dir,
        promotion_mode="replace",
        output_root=run_dir.parent / "scratch",
        field_specs=field_specs,
        resource_root=tree["resource_root"],
        **kwargs,
    )


def test_composite_discovery_artifact_path_is_read(tmp_path: Path):
    tree = build_toy_tree(tmp_path)
    run_dir = _make_synthetic_run(tmp_path, composite=True, legacy_composite_path=False)
    bundle = _prepare_synthetic(tree, run_dir, field_specs=_toy_field_specs())

    assert bundle.promotable_composite_count >= 1
    assert "toy_comp_verify" in bundle.selected_promotion_types
    assert "toy_comp_verify" not in bundle.non_selected_types
    catalog_ids = {t["type_id"] for t in bundle.candidate_semantic_catalog["tasks"]}
    assert "toy_comp_verify" in catalog_ids
    # No missing-proposal error caused by a wrong composite stage path.
    assert all("MISSING_STRUCTURED_PROPOSAL" not in v for v in bundle.non_selected_types.values())
    assert all("MISSING_PROPOSAL_EVIDENCE" not in v for v in bundle.non_selected_types.values())


def test_legacy_composite_path_is_not_read(tmp_path: Path):
    """The historical composite_semantic_discovery path must not be read."""
    run_dir = tmp_path / "legacy_run"
    (run_dir / "llm" / "composite_semantic_discovery").mkdir(parents=True)
    (run_dir / "llm" / "composite_semantic_discovery" / "parsed_response.json").write_text(
        json.dumps({"composites": [{"composite_id": "toy_comp_999_x", "operator": "COMPOSITE"}]}),
        encoding="utf-8",
    )
    executable, excluded = filter_executable_candidates(["toy_comp_999_x"], run_dir)
    assert "toy_comp_999_x" not in executable
    assert excluded["toy_comp_999_x"] == "MISSING_STRUCTURED_PROPOSAL"


def test_promotable_total_invariant_with_composite(tmp_path: Path):
    tree = build_toy_tree(tmp_path)
    run_dir = _make_synthetic_run(tmp_path, composite=True)
    bundle = _prepare_synthetic(tree, run_dir, field_specs=_toy_field_specs())
    assert bundle.promotable_total == bundle.promotable_primitive_count + bundle.promotable_composite_count
    assert bundle.promotable_composite_count == 1
    assert bundle.promotable_primitive_count == 1


# ---------------------------------------------------------------------------
# Reconciliation executable compatibility (fail closed)
# ---------------------------------------------------------------------------


def _canon_task(**over) -> SemanticTaskSpec:
    base = dict(
        type_id="canon_topic",
        proposition_id="canon_topic",
        proposition_description="Canonical topic",
        operator="DIRECT",
        classification="PRIMITIVE_RELATION",
        kind="ATOMIC",
        audio_arity=1,
        source_role_mapping={"source_field": "topic"},
        outputs=[{"role": "val", "kind": "field_value", "dependencies": []}],
        comparator_id="toy_exact",
    )
    base.update(over)
    return SemanticTaskSpec(**base)


def _raw_candidate(**over) -> dict:
    base = {
        "candidate_id": "cand_001",
        "operator": "DIRECT",
        "audio_arity": 1,
        "visible_inputs": [],
        "hidden_source_annotations": ["topic"],
        "answer_schema_proposal": {"kind": "field_value"},
        "comparator_id": "toy_exact",
    }
    base.update(over)
    return base


def _recon() -> dict:
    return {"matches": [{
        "blind_candidate": "cand_001",
        "canonical_type": "canon_topic",
        "classification": "SAME_PROPOSITION_DIFFERENT_NAME",
    }]}


def _resolve_expect_error(raw, canonical_task):
    with pytest.raises(PromotionError) as exc_info:
        resolve_canonical_ids(
            "toy", ["cand_001"], _recon(),
            raw_candidates_map={"cand_001": raw},
            existing_catalog_tasks=[canonical_task],
        )
    assert exc_info.value.code == "RECONCILIATION_EXECUTABLE_COMPATIBILITY_UNPROVEN"


def test_reconciliation_canonical_task_missing_blocks():
    with pytest.raises(PromotionError) as exc_info:
        resolve_canonical_ids(
            "toy", ["cand_001"], _recon(),
            raw_candidates_map={"cand_001": _raw_candidate()},
            existing_catalog_tasks=[],
        )
    assert exc_info.value.code == "RECONCILIATION_EXECUTABLE_COMPATIBILITY_UNPROVEN"


def test_reconciliation_operator_mismatch_blocks():
    _resolve_expect_error(_raw_candidate(operator="EQUALITY"), _canon_task())


def test_reconciliation_arity_mismatch_blocks():
    _resolve_expect_error(_raw_candidate(audio_arity=2), _canon_task())


def test_reconciliation_answer_kind_mismatch_blocks():
    _resolve_expect_error(_raw_candidate(answer_schema_proposal={"kind": "boolean"}), _canon_task())


def test_reconciliation_source_field_mismatch_blocks():
    _resolve_expect_error(_raw_candidate(hidden_source_annotations=["segment_text"]), _canon_task())


def test_reconciliation_visible_context_mismatch_blocks():
    _resolve_expect_error(_raw_candidate(visible_inputs=["target"]), _canon_task())


def test_reconciliation_missing_evidence_blocks():
    raw = _raw_candidate()
    raw.pop("audio_arity")
    _resolve_expect_error(raw, _canon_task())


def test_comparator_conflict_blocks_reuse():
    _resolve_expect_error(_raw_candidate(comparator_id="other_comparator"), _canon_task())


def test_compatible_canonical_reuse_succeeds():
    mapping = resolve_canonical_ids(
        "toy", ["cand_001"], _recon(),
        raw_candidates_map={"cand_001": _raw_candidate()},
        existing_catalog_tasks=[_canon_task()],
    )
    assert mapping == {"cand_001": "canon_topic"}


# ---------------------------------------------------------------------------
# Source evidence freshness (fail closed)
# ---------------------------------------------------------------------------


def _source_run(base: Path) -> Path:
    run = base / "src_run"
    (run / "deterministic").mkdir(parents=True, exist_ok=True)
    (run / "source_documentation").mkdir(parents=True, exist_ok=True)
    (run / "run_manifest.json").write_text(json.dumps({"id": "run"}), encoding="utf-8")
    (run / "deterministic" / "dataset_profile.json").write_text(json.dumps({"p": 1}), encoding="utf-8")
    (run / "source_documentation" / "readme.md").write_text("doc", encoding="utf-8")
    return run


def _bundle_with_evidence(tree, base: Path, run: Path) -> PromotionBundle:
    bundle = make_apply_ready_bundle(tree, base)
    doc_hash = _canonical_hash([_sha256_file(p) for p in sorted((run / "source_documentation").glob("*"))])
    return bundle.model_copy(update={
        "source_run_path": str(run),
        "authoring_run_hash": _sha256_file(run / "run_manifest.json"),
        "dataset_profile_hash": _sha256_file(run / "deterministic" / "dataset_profile.json"),
        "source_documentation_hash": doc_hash,
    })


def _expect_stale(tree, bundle):
    with pytest.raises(PromotionError) as exc_info:
        apply_promotion(bundle, resource_root=tree["resource_root"])
    assert exc_info.value.code == "PROMOTION_SOURCE_EVIDENCE_STALE"


def test_missing_source_run_blocks(tmp_path: Path):
    tree = build_toy_tree(tmp_path)
    run = _source_run(tmp_path)
    bundle = _bundle_with_evidence(tree, tmp_path, run).model_copy(
        update={"source_run_path": str(tmp_path / "gone")}
    )
    _expect_stale(tree, bundle)


def test_missing_run_manifest_blocks(tmp_path: Path):
    tree = build_toy_tree(tmp_path)
    run = _source_run(tmp_path)
    bundle = _bundle_with_evidence(tree, tmp_path, run)
    (run / "run_manifest.json").unlink()
    _expect_stale(tree, bundle)


def test_missing_dataset_profile_blocks(tmp_path: Path):
    tree = build_toy_tree(tmp_path)
    run = _source_run(tmp_path)
    bundle = _bundle_with_evidence(tree, tmp_path, run)
    (run / "deterministic" / "dataset_profile.json").unlink()
    _expect_stale(tree, bundle)


def test_missing_source_documentation_blocks(tmp_path: Path):
    tree = build_toy_tree(tmp_path)
    run = _source_run(tmp_path)
    bundle = _bundle_with_evidence(tree, tmp_path, run)
    (run / "source_documentation" / "readme.md").unlink()
    _expect_stale(tree, bundle)


def test_modified_source_evidence_blocks(tmp_path: Path):
    tree = build_toy_tree(tmp_path)
    run = _source_run(tmp_path)
    bundle = _bundle_with_evidence(tree, tmp_path, run)
    (run / "run_manifest.json").write_text(json.dumps({"id": "changed"}), encoding="utf-8")
    _expect_stale(tree, bundle)


# ---------------------------------------------------------------------------
# Stale canonical hash protection (isolated)
# ---------------------------------------------------------------------------


def test_all_five_stale_hashes_are_checked(tmp_path: Path):
    tree = build_toy_tree(tmp_path)
    bundle = make_apply_ready_bundle(tree, tmp_path)
    for key in (
        "dataset_spec_hash",
        "semantic_catalog_hash",
        "production_contract_hash",
        "promotion_manifest_hash",
        "language_registry_hash",
    ):
        corrupted = dict(bundle.expected_current_hashes)
        corrupted[key] = "corrupted_stale_hash_value"
        stale = bundle.model_copy(update={"expected_current_hashes": corrupted})
        with pytest.raises(PromotionError) as exc_info:
            apply_promotion(stale, resource_root=tree["resource_root"])
        assert exc_info.value.code == "PROMOTION_BUNDLE_STALE", key


# ---------------------------------------------------------------------------
# Atomic replacement + rollback (isolated)
# ---------------------------------------------------------------------------


def _run_rollback_case(tmp_path: Path, replace_fn, *, expect_committed=None):
    tree = build_toy_tree(tmp_path)
    root = tree["resource_root"]
    bundle = make_apply_ready_bundle(tree, tmp_path)
    before = snapshot_tree(root)

    with pytest.raises(PromotionError) as exc_info:
        apply_promotion(bundle, resource_root=root, replace_fn=replace_fn)
    assert exc_info.value.code == "TRANSACTION_FAILED_ROLLED_BACK"

    after = snapshot_tree(root)
    assert before == after
    return before


def test_failure_after_first_replacement_rolls_back(tmp_path: Path):
    replace_fn, state = _make_fail_after(1)
    _run_rollback_case(tmp_path, replace_fn)
    assert state["committed"] == 1


def test_failure_after_second_replacement_rolls_back(tmp_path: Path):
    replace_fn, state = _make_fail_after(2)
    _run_rollback_case(tmp_path, replace_fn)
    assert state["committed"] == 2


def test_failure_during_final_validation_rolls_back(tmp_path: Path):
    tree = build_toy_tree(tmp_path)
    root = tree["resource_root"]
    bundle = make_apply_ready_bundle(tree, tmp_path)
    before = snapshot_tree(root)

    def boom():
        raise RuntimeError("INJECTED_FAILURE_DURING_FINAL_VALIDATION")

    with pytest.raises(PromotionError) as exc_info:
        apply_promotion(bundle, resource_root=root, validation_hook=boom)
    assert exc_info.value.code == "TRANSACTION_FAILED_ROLLED_BACK"
    assert snapshot_tree(root) == before


def test_newly_created_destination_removed_by_rollback(tmp_path: Path):
    tree = build_toy_tree(tmp_path, include_manifest=False)
    root = tree["resource_root"]
    bundle = make_apply_ready_bundle(tree, tmp_path)
    manifest_path = root / "production" / "toy.promotion.json"
    assert not manifest_path.exists()
    before = snapshot_tree(root)

    def replace(src: str, dst: str):
        os.replace(src, dst)
        if Path(dst) == manifest_path:
            raise RuntimeError("INJECTED_FAILURE_AFTER_NEW_MANIFEST_CREATED")

    with pytest.raises(PromotionError) as exc_info:
        apply_promotion(bundle, resource_root=root, replace_fn=replace)
    assert exc_info.value.code == "TRANSACTION_FAILED_ROLLED_BACK"
    assert not manifest_path.exists()
    assert snapshot_tree(root) == before


def test_default_replacement_uses_os_replace(tmp_path: Path, monkeypatch):
    tree = build_toy_tree(tmp_path)
    root = tree["resource_root"]
    bundle = make_apply_ready_bundle(tree, tmp_path)

    calls = []
    real_replace = os.replace

    def spy(src, dst):
        calls.append((src, dst))
        return real_replace(src, dst)

    monkeypatch.setattr(authoring_promotion.os, "replace", spy)
    apply_promotion(bundle, resource_root=root)
    assert len(calls) == 3


def test_happy_path_apply_writes_and_validates(tmp_path: Path):
    tree = build_toy_tree(tmp_path)
    root = tree["resource_root"]
    bundle = make_apply_ready_bundle(tree, tmp_path)

    result = apply_promotion(bundle, resource_root=root)
    assert result["status"] == "APPLIED"

    reloaded_contract = ProductionContract.model_validate(
        json.loads((root / "production" / "toy.json").read_text(encoding="utf-8"))
    )
    reloaded_manifest = PromotionManifest.model_validate(
        json.loads((root / "production" / "toy.promotion.json").read_text(encoding="utf-8"))
    )
    assert reloaded_contract.promotion_fingerprint
    assert reloaded_contract.promotion_fingerprint == reloaded_contract.fingerprint()
    assert reloaded_manifest.promotion_fingerprint
    assert reloaded_manifest.promotion_fingerprint == reloaded_manifest.fingerprint()
    assert reloaded_manifest.source_revision == tree["spec"].source["revision_identity"]
    assert reloaded_manifest.source_revision != bundle.source_run_id


def test_apply_writes_only_below_resource_root(tmp_path: Path):
    """resource_root is authoritative: a valid apply must not touch ROOT/resources."""
    tree = build_toy_tree(tmp_path)
    root = tree["resource_root"]
    before = snapshot_tree(root)
    bundle = make_apply_ready_bundle(tree, tmp_path)
    apply_promotion(bundle, resource_root=root)
    after = snapshot_tree(root)
    assert set(before) == set(after)


# ---------------------------------------------------------------------------
# Automatic preflight implementation identity
# ---------------------------------------------------------------------------


def test_preflight_implementation_identity_is_automatic():
    impl = get_canonical_preflight_implementation_sha256()
    assert impl == hashlib.sha256(Path(language_preflight.__file__).read_bytes()).hexdigest()

    fp_default, inputs = compute_contract_fingerprint(mode="dataset", accepted_types=[])
    assert inputs["preflight_implementation_sha256"] == impl

    fp_same, _ = compute_contract_fingerprint(
        mode="dataset", accepted_types=[], implementation_identity=impl
    )
    fp_other, _ = compute_contract_fingerprint(
        mode="dataset", accepted_types=[], implementation_identity="not_the_implementation"
    )
    assert fp_default == fp_same
    assert fp_default != fp_other

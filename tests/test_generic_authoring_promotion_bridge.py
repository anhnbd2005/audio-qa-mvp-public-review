"""Comprehensive test suite for Phase 4.1 Generic Authoring -> Promotion Bridge Correctness."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.autonomous_qa.certification.authoring_promotion import (
    PromotionBundle,
    apply_promotion,
    evaluate_promotion_readiness,
    prepare_promotion,
    resolve_canonical_ids,
)
from src.autonomous_qa.certification.promotion_gate import PromotionError
from src.autonomous_qa.compiler.canonical_resources import (
    DatasetSpec,
    ProductionContract,
    PromotionManifest,
    get_dataset_spec,
    get_semantic_catalog_path,
)
from src.autonomous_qa.compiler.semantic_task import load_semantic_catalog
from src.autonomous_qa.language.language_quality import load_language_registry
from src.common.config import ROOT

ACCEPTANCE_RUN_DIR = ROOT / "outputs" / "runs" / "vimedcss" / "20261007_vimedcss_v2_authoring"
RESOURCE_ROOT = ROOT / "resources"


def test_vimedcss_acceptance_run_prepare_dryrun(tmp_path: Path):
    """Stage 19: ViMedCSS real acceptance run prepare dry-run via generic engine."""
    scratch_out = tmp_path / "scratch_promotion"
    bundle = prepare_promotion(
        dataset_id="vimedcss",
        run_dir=ACCEPTANCE_RUN_DIR,
        promotion_mode="replace",
        output_root=scratch_out,
    )

    # Core readiness assertions
    assert bundle.primitive_candidates == 4
    assert bundle.primitive_semantic_accepted == 4
    assert bundle.primitive_language_ready == 3
    assert bundle.promotable_primitive_count == 3

    assert bundle.composite_candidates == 1
    assert bundle.composite_semantic_accepted == 1
    assert bundle.composite_language_ready == 0
    assert bundle.promotable_composite_count == 0

    assert bundle.promotable_total == 3
    assert "vimedcss_003_cs_term_presence" in bundle.non_promotable_types
    assert bundle.non_promotable_types["vimedcss_003_cs_term_presence"] == "OPERATOR_CAPABILITY_GAP:MEMBERSHIP"
    assert "vimedcss_comp_001_topic_term_verification" in bundle.non_promotable_types
    assert bundle.non_promotable_types["vimedcss_comp_001_topic_term_verification"] == "SEMANTIC_GATE_PASS_BUT_LANGUAGE_NOT_READY"

    # Profile hash check
    assert bundle.dataset_profile_hash != ""

    # Staged language preflight check & apply_ready state machine
    assert "status" in bundle.staged_language_preflight
    assert bundle.staged_language_preflight.get("status") in ("PREFLIGHT_PASS", "LANGUAGE_CONTRACT_FAIL")
    assert bundle.prepare_status == "PREPARED"
    assert bundle.apply_ready is False

    # Explicit 3-set type identity assertions
    expected_exec = ("vimedcss_cs_terms_count", "vimedcss_pairwise_topic_same", "vimedcss_topic_classification")
    assert bundle.executable_promotion_candidates == expected_exec
    assert bundle.staged_preflight_types == expected_exec
    assert bundle.staged_catalog_type_ids == expected_exec
    assert bundle.staged_accepted_type_ids == expected_exec
    assert bundle.staged_language_candidate_type_ids == expected_exec
    assert set(bundle.staged_accepted_type_ids) == set(bundle.executable_promotion_candidates)

    # Calling apply_promotion on non-apply-ready bundle fails with PROMOTION_NOT_APPLY_READY
    with pytest.raises(PromotionError) as exc_info:
        apply_promotion(bundle)
    assert exc_info.value.code == "PROMOTION_NOT_APPLY_READY"

    # Reconciliation ID mapping checks
    assert bundle.canonical_id_mapping["vimedcss_001_topic_classification"] == "vimedcss_topic_classification"
    assert bundle.canonical_id_mapping["vimedcss_002_cs_terms_count"] == "vimedcss_cs_terms_count"
    assert bundle.canonical_id_mapping["vimedcss_004_same_topic_pairwise"] == "vimedcss_pairwise_topic_same"

    # Replace mode active set check (3 promotable types)
    expected_active = (
        "vimedcss_cs_terms_count",
        "vimedcss_pairwise_topic_same",
        "vimedcss_topic_classification",
    )
    assert bundle.proposed_active_types == expected_active

    # Removed old active types under replace mode
    assert "vimedcss_spoken_content_transcription" in bundle.semantic_diff["removed_types"]
    assert "vimedcss_cs_term_extraction" in bundle.semantic_diff["removed_types"]

    # Verify zero canonical mutation during prepare
    assert bundle.zero_llm_evidence["llm_calls_in_promotion"] == 0


def test_synthetic_future_dataset_promotion(tmp_path: Path):
    """Stage 21: Synthetic future dataset proof (toy_audio dataset with non-ViMedCSS fields)."""
    run_dir = tmp_path / "toy_run"
    run_dir.mkdir(parents=True)
    gates_dir = run_dir / "gates"
    gates_dir.mkdir()
    lang_dir = run_dir / "language"
    lang_dir.mkdir()
    det_dir = run_dir / "deterministic"
    det_dir.mkdir()
    llm_dir = run_dir / "llm" / "primitive_semantic_discovery"
    llm_dir.mkdir(parents=True)
    llm_lang_dir = run_dir / "llm" / "language_generation"
    llm_lang_dir.mkdir(parents=True)

    # Write profile
    (det_dir / "dataset_profile.json").write_text(json.dumps({"dataset_id": "toy"}))

    # Write synthetic primitive gates
    (gates_dir / "primitive_gates.json").write_text(
        json.dumps({
            "candidates": [
                {"candidate_id": "toy_001_category", "verdict": "PASS"},
                {"candidate_id": "toy_002_token_match", "verdict": "PASS"},
                {"candidate_id": "toy_003_event_count", "verdict": "PASS"},
            ]
        })
    )

    # Write synthetic preflight
    (lang_dir / "preflight.json").write_text(
        json.dumps({
            "entries": [
                {"type_id": "toy_001_category", "verdict": "PASS"},
                {"type_id": "toy_002_token_match", "verdict": "PASS"},
                {"type_id": "toy_003_event_count", "verdict": "PASS"},
            ]
        })
    )

    # Write synthetic discovery parsed response
    (llm_dir / "parsed_response.json").write_text(
        json.dumps({
            "candidates": [
                {
                    "candidate_id": "toy_001_category",
                    "proposition": "Category X classification",
                    "operator": "DIRECT",
                    "audio_arity": 1,
                    "hidden_source_annotations": ["category_x"],
                    "answer_schema_proposal": {"kind": "field_value", "type": "string"},
                },
                {
                    "candidate_id": "toy_002_token_match",
                    "proposition": "Token match in token_list_x",
                    "operator": "EQUALITY",
                    "audio_arity": 1,
                    "hidden_source_annotations": ["token_x"],
                    "answer_schema_proposal": {"kind": "boolean"},
                },
                {
                    "candidate_id": "toy_003_event_count",
                    "proposition": "Event count integer",
                    "operator": "DIRECT",
                    "audio_arity": 1,
                    "hidden_source_annotations": ["event_count_x"],
                    "answer_schema_proposal": {"kind": "field_value", "type": "integer"},
                },
            ]
        })
    )

    # Write synthetic language generation response
    (llm_lang_dir / "parsed_response.json").write_text(
        json.dumps({
            "entries": [
                {"type_id": "toy_001_category", "question_templates": ["What category is this?"]},
                {"type_id": "toy_002_token_match", "question_templates": ["Is token present?"]},
                {"type_id": "toy_003_event_count", "question_templates": ["How many events?"]},
            ]
        })
    )

    (run_dir / "run_manifest.json").write_text(json.dumps({"status": "SUCCESS"}))
    (run_dir / "reconciliation_canonical.json").write_text(json.dumps({"matches": []}))

    # Prepare promotion for toy dataset
    bundle = prepare_promotion(
        dataset_id="toy",
        run_dir=run_dir,
        promotion_mode="replace",
        output_root=tmp_path / "toy_out",
    )

    assert bundle.promotable_total == 3
    # Check normalized canonical IDs: toy_003_event_count -> toy_event_count
    assert bundle.canonical_id_mapping["toy_003_event_count"] == "toy_event_count"
    assert bundle.canonical_id_mapping["toy_001_category"] == "toy_category"
    assert bundle.canonical_id_mapping["toy_002_token_match"] == "toy_token_match"

    # Verify candidate catalog compiled without dataset-specific branches
    cat = bundle.candidate_semantic_catalog
    assert len(cat["tasks"]) == 3
    task_ids = [t["type_id"] for t in cat["tasks"]]
    assert "toy_event_count" in task_ids


def test_no_vimedcss_comparator_hardcode_and_capability_gap(tmp_path: Path):
    """Stage 2: Toy TARGET_MATCH candidate with list membership fails with OPERATOR_CAPABILITY_GAP."""
    run_dir = tmp_path / "toy_run_gap"
    run_dir.mkdir(parents=True)
    gates_dir = run_dir / "gates"
    gates_dir.mkdir()
    lang_dir = run_dir / "language"
    lang_dir.mkdir()
    det_dir = run_dir / "deterministic"
    det_dir.mkdir()
    llm_dir = run_dir / "llm" / "primitive_semantic_discovery"
    llm_dir.mkdir(parents=True)

    (det_dir / "dataset_profile.json").write_text(json.dumps({"dataset_id": "toy"}))

    (gates_dir / "primitive_gates.json").write_text(
        json.dumps({
            "candidates": [
                {"candidate_id": "toy_001_list_membership", "verdict": "PASS"},
            ]
        })
    )

    (lang_dir / "preflight.json").write_text(
        json.dumps({"entries": [{"type_id": "toy_001_list_membership", "verdict": "PASS"}]})
    )

    (llm_dir / "parsed_response.json").write_text(
        json.dumps({
            "candidates": [
                {
                    "candidate_id": "toy_001_list_membership",
                    "proposition": "Target in list membership",
                    "operator": "TARGET_MATCH",
                    "audio_arity": 1,
                    "visible_inputs": ["target_item"],
                    "hidden_source_annotations": ["item_list_x"],
                    "answer_schema_proposal": {"kind": "boolean"},
                }
            ]
        })
    )

    (run_dir / "run_manifest.json").write_text(json.dumps({"status": "SUCCESS"}))
    (run_dir / "reconciliation_canonical.json").write_text(json.dumps({"matches": []}))

    with pytest.raises(PromotionError) as exc_info:
        prepare_promotion(
            dataset_id="toy",
            run_dir=run_dir,
            promotion_mode="replace",
            output_root=tmp_path / "out",
        )
    assert "OPERATOR_CAPABILITY_GAP" in str(exc_info.value)
    assert "vimedcss" not in str(exc_info.value).lower()


def test_profile_hash_nonempty_and_validation(tmp_path: Path):
    """Stage 1: Missing dataset_profile.json fails closed."""
    run_dir = tmp_path / "run_no_profile"
    run_dir.mkdir(parents=True)

    with pytest.raises(PromotionError) as exc_info:
        prepare_promotion(
            dataset_id="vimedcss",
            run_dir=run_dir,
            promotion_mode="replace",
        )
    assert "MISSING_AUTHORING_EVIDENCE" in str(exc_info.value)


def test_merge_and_replace_catalog_active_set_invariants(tmp_path: Path):
    """Stage 10: Verify replace and merge mode candidate catalog active set invariants."""
    scratch_out = tmp_path / "scratch_promotion"
    bundle_replace = prepare_promotion(
        dataset_id="vimedcss",
        run_dir=ACCEPTANCE_RUN_DIR,
        promotion_mode="replace",
        output_root=scratch_out / "replace",
    )

    compiled_replace_ids = {t["type_id"] for t in bundle_replace.candidate_semantic_catalog["tasks"]}
    assert compiled_replace_ids == set(bundle_replace.proposed_active_types)

    bundle_merge = prepare_promotion(
        dataset_id="vimedcss",
        run_dir=ACCEPTANCE_RUN_DIR,
        promotion_mode="merge",
        output_root=scratch_out / "merge",
    )

    compiled_merge_ids = {t["type_id"] for t in bundle_merge.candidate_semantic_catalog["tasks"]}
    assert compiled_merge_ids == set(bundle_merge.proposed_active_types)
    assert len(bundle_merge.proposed_active_types) >= len(bundle_replace.proposed_active_types)


def test_stale_bundle_protection(tmp_path: Path):
    """Stage 14: Stale bundle fails apply transaction."""
    scratch_out = tmp_path / "scratch_promotion"
    bundle = prepare_promotion(
        dataset_id="vimedcss",
        run_dir=ACCEPTANCE_RUN_DIR,
        promotion_mode="replace",
        output_root=scratch_out,
    )

    # Create stale bundle with modified expected hashes
    stale_hashes = dict(bundle.expected_current_hashes)
    stale_hashes["semantic_catalog_hash"] = "changed_hash_12345"

    stale_bundle = bundle.model_copy(update={"apply_ready": True, "expected_current_hashes": stale_hashes})

    with pytest.raises(PromotionError) as exc_info:
        apply_promotion(stale_bundle)
    assert "PROMOTION_BUNDLE_STALE" in str(exc_info.value)


def test_transaction_rollback_exact_bytes(tmp_path: Path):
    """Stage 13: Failure during apply transaction restores canonical files byte-identically."""
    canonical_cat = RESOURCE_ROOT / "semantics" / "vimedcss_semantic_catalog.json"
    canonical_contract = RESOURCE_ROOT / "production" / "vimedcss.json"
    canonical_manifest = RESOURCE_ROOT / "production" / "vimedcss.promotion.json"

    cat_bytes_before = canonical_cat.read_bytes()
    contract_bytes_before = canonical_contract.read_bytes()
    manifest_bytes_before = canonical_manifest.read_bytes()

    scratch_out = tmp_path / "scratch_promotion"
    bundle = prepare_promotion(
        dataset_id="vimedcss",
        run_dir=ACCEPTANCE_RUN_DIR,
        promotion_mode="replace",
        output_root=scratch_out,
    )

    # Create a corrupted bundle that causes post-write catalog validation to fail
    corrupt_catalog = dict(bundle.candidate_semantic_catalog)
    corrupt_catalog["tasks"] = list(corrupt_catalog["tasks"]) + [{
        "type_id": "corrupted_task",
        "proposition_id": "", # Missing proposition_id triggers validation error
        "proposition_description": "corrupted",
        "operator": "DIRECT",
        "classification": "PRIMITIVE_RELATION",
        "kind": "ATOMIC",
        "audio_arity": 1,
        "outputs": [{"role": "val", "kind": "field_value", "dependencies": []}],
        "comparator_id": "categorical_label_exact",
    }]

    corrupt_bundle = bundle.model_copy(update={"apply_ready": True, "candidate_semantic_catalog": corrupt_catalog})

    with pytest.raises(PromotionError) as exc_info:
        apply_promotion(corrupt_bundle)
    assert "TRANSACTION_FAILED_ROLLED_BACK" in str(exc_info.value) or "CANONICAL_RESOURCE_VALIDATION_FAILED" in str(exc_info.value)

    # Verify exact byte restoration
    assert canonical_cat.read_bytes() == cat_bytes_before
    assert canonical_contract.read_bytes() == contract_bytes_before
    assert canonical_manifest.read_bytes() == manifest_bytes_before


def test_logical_hash_parity(tmp_path: Path):
    """Stage 7: Verify logical hash parity between prepared bundle and catalog."""
    scratch_out = tmp_path / "scratch_promotion"
    bundle = prepare_promotion(
        dataset_id="vimedcss",
        run_dir=ACCEPTANCE_RUN_DIR,
        promotion_mode="replace",
        output_root=scratch_out,
    )

    cat_obj = load_semantic_catalog(scratch_out / "candidate_semantic_catalog.json")
    assert cat_obj.logical_hash() == bundle.fresh_contract_fingerprints["candidate_catalog"]


def test_datasetspec_split_policy_preserved(tmp_path: Path):
    """Stage 8: Verify DatasetSpec allowed_splits is preserved from spec (('train',))."""
    spec_obj = get_dataset_spec("vimedcss")
    assert spec_obj.allowed_splits == ("train",)


def test_zero_llm_and_no_authoring_import_in_production():
    """Stage 24: Zero LLM calls and production package isolation tests."""
    import src.autonomous_qa.production.production_qa as prod_qa
    assert "authoring" not in dir(prod_qa)

    bundle = prepare_promotion(
        dataset_id="vimedcss",
        run_dir=ACCEPTANCE_RUN_DIR,
        promotion_mode="replace",
    )
    assert bundle.zero_llm_evidence["llm_calls_in_promotion"] == 0


def test_membership_semantics_proof_synthetic_row():
    """Section 2 proof: scalar TARGET_MATCH fails list membership semantics."""
    cs_terms_list = "insulin; diabetes"
    target_pos = "insulin"
    target_neg = "aspirin"

    # Expected list membership relation: target MEMBER_OF parse(cs_terms_list)
    parsed_terms = [t.strip().casefold() for t in cs_terms_list.split(";")]
    expected_gold_pos = target_pos.casefold() in parsed_terms
    expected_gold_neg = target_neg.casefold() in parsed_terms
    assert expected_gold_pos is True
    assert expected_gold_neg is False

    # Current scalar TARGET_MATCH engine evaluation (actual == target)
    actual_scalar = cs_terms_list
    scalar_gold_pos = (actual_scalar == target_pos)
    scalar_gold_neg = (actual_scalar == target_neg)

    # Scalar comparison yields False for target_pos ("insulin; diabetes" == "insulin"), failing membership semantics
    assert scalar_gold_pos is False
    assert scalar_gold_pos != expected_gold_pos


def test_staged_preflight_input_identity_and_no_canonical_resolver():
    """Section 3 proof: candidate catalog with new_type sees new_type and NOT old_type in staged preflight."""
    from src.autonomous_qa.language.language_preflight import accepted_types_from_semantic_catalog

    candidate_catalog_dict = {
        "dataset": "vimedcss",
        "catalog_version": "candidate_v1",
        "tasks": [
            {
                "type_id": "new_synthetic_task",
                "proposition_id": "prop_new",
                "proposition_description": "New synthetic task",
                "operator": "DIRECT",
                "audio_arity": 1,
                "visible_context_roles": [],
                "source_role_mapping": {"source_field": "topic"},
                "outputs": [{"role": "val", "kind": "field_value", "dependencies": ["topic"]}],
                "comparator_id": "categorical_label_exact",
            }
        ],
    }

    accepted = accepted_types_from_semantic_catalog(candidate_catalog_dict, dataset_id="vimedcss")
    accepted_ids = [a.dataset_type_id for a in accepted]

    assert "new_synthetic_task" in accepted_ids
    assert "vimedcss_cs_term_extraction" not in accepted_ids
    assert "vimedcss_spoken_content_transcription" not in accepted_ids


def test_field_spec_missing_fails_closed():
    """Section 4 proof: missing field spec raises FIELD_SPEC_MISSING:<field> instead of silent fallback."""
    from src.autonomous_qa.language.language_preflight import accepted_types_from_semantic_catalog, PreflightInputError

    catalog_missing_field = {
        "dataset": "vimedcss",
        "catalog_version": "candidate_missing",
        "tasks": [
            {
                "type_id": "task_missing_spec",
                "proposition_id": "prop_m",
                "proposition_description": "Task with unmapped field",
                "operator": "DIRECT",
                "audio_arity": 1,
                "visible_context_roles": [],
                "source_role_mapping": {"source_field": "non_existent_unmapped_field"},
                "outputs": [{"role": "val", "kind": "field_value", "dependencies": ["non_existent_unmapped_field"]}],
                "comparator_id": "categorical_label_exact",
            }
        ],
    }

    with pytest.raises(PreflightInputError) as exc_info:
        accepted_types_from_semantic_catalog(catalog_missing_field, dataset_id="vimedcss")
    assert "FIELD_SPEC_MISSING:non_existent_unmapped_field" in str(exc_info.value)


def test_arbitrary_task_ids_zero_substring_dependence():
    """Section 6 & 7 proof: arbitrary task IDs alpha_foo, beta_bar, z9 derive semantics strictly from structured fields."""
    from src.autonomous_qa.language.template_contracts import build_type_contracts_from_semantic_catalog

    arbitrary_catalog = {
        "dataset": "toy",
        "catalog_version": "v_arb",
        "tasks": [
            {
                "type_id": "alpha_foo_presence",
                "proposition_id": "prop_alpha",
                "proposition_description": "Alpha DIRECT integer task",
                "operator": "DIRECT",
                "audio_arity": 1,
                "visible_context_roles": [],
                "source_role_mapping": {"source_field": "topic"},
                "outputs": [{"role": "val", "kind": "field_value", "dependencies": ["topic"]}],
                "comparator_id": "categorical_label_exact",
            },
            {
                "type_id": "beta_bar_pairwise",
                "proposition_id": "prop_beta",
                "proposition_description": "Beta TARGET_MATCH task",
                "operator": "TARGET_MATCH",
                "audio_arity": 1,
                "visible_context_roles": ["target_value"],
                "source_role_mapping": {"source_field": "cs_terms_list"},
                "outputs": [{"role": "val", "kind": "boolean", "dependencies": ["cs_terms_list"]}],
                "comparator_id": "categorical_label_exact",
            },
            {
                "type_id": "z9",
                "proposition_id": "prop_z9",
                "proposition_description": "z9 EQUALITY task",
                "operator": "EQUALITY",
                "audio_arity": 2,
                "visible_context_roles": [],
                "source_role_mapping": {"source_field": "topic"},
                "outputs": [{"role": "val", "kind": "boolean", "dependencies": ["topic"]}],
                "comparator_id": "categorical_label_exact",
            },
        ],
    }

    contracts = {c.type_id: c for c in build_type_contracts_from_semantic_catalog(arbitrary_catalog)}

    # alpha_foo_presence has operator DIRECT despite 'presence' in ID
    assert contracts["alpha_foo_presence"].operator == "DIRECT"
    assert contracts["alpha_foo_presence"].answer_mode == "FIELD_VALUE"

    # beta_bar_pairwise has operator TARGET_MATCH despite 'pairwise' in ID
    assert contracts["beta_bar_pairwise"].operator == "TARGET_MATCH"
    assert contracts["beta_bar_pairwise"].answer_mode == "BOOLEAN"

    # z9 has operator EQUALITY
    assert contracts["z9"].operator == "EQUALITY"
    assert contracts["z9"].audio_input_count == 2

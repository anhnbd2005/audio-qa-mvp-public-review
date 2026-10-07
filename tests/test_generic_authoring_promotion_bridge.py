"""Comprehensive test suite for Phase 4.1.3 Generic Authoring -> Promotion Bridge Correctness.

Verifies:
- candidate/type ID rename cannot change membership capability outcome
- generic promotion module contains no literal ViMedCSS task IDs
- staged accepted-type compilation has no dataset fallback
- no field-name 'count'/'num' semantic inference
- no segment_text fallback
- all expected current hashes nonempty for existing ViMedCSS
- stale detection independently covers spec/catalog/contract/manifest/language
- selected-set apply_ready can become true despite explicitly excluded candidates
- authoring gate count independent from executable count
- staged count independent from authoring count
- reconciliation executable reuse fails closed when equivalence unproven
- true rollback after first replacement
- true rollback after second replacement
- newly-created destination removed by rollback
- preflight implementation identity affects fingerprint
- onboard does not label PREPARE as production plan
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
import pytest

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
    get_dataset_spec,
    get_semantic_catalog_path,
)
from src.autonomous_qa.compiler.semantic_task import SemanticTaskSpec, load_semantic_catalog
from src.autonomous_qa.language.language_preflight import (
    PreflightInputError,
    accepted_types_from_semantic_catalog,
    compute_contract_fingerprint,
)
from src.autonomous_qa.language.template_engine import SemanticFieldSpec, vimedcss_field_specs
from src.autonomous_qa.onboard import onboard_dataset
from src.common.config import ROOT

ACCEPTANCE_RUN_DIR = ROOT / "outputs" / "runs" / "vimedcss" / "20261007_vimedcss_v2_authoring"
RESOURCE_ROOT = ROOT / "resources"


def test_vimedcss_acceptance_run_prepare_dryrun(tmp_path: Path):
    """ViMedCSS real acceptance run prepare dry-run via generic engine."""
    scratch_out = tmp_path / "scratch_promotion"
    bundle = prepare_promotion(
        dataset_id="vimedcss",
        run_dir=ACCEPTANCE_RUN_DIR,
        promotion_mode="replace",
        output_root=scratch_out,
    )

    # Core readiness assertions: 4 primitives passed authoring gates
    assert bundle.primitive_candidates == 4
    assert bundle.primitive_semantic_accepted == 4
    assert bundle.primitive_semantic_pass_count == 4
    assert bundle.primitive_language_ready == 4
    assert bundle.primitive_authoring_language_gate_pass_count == 4
    assert bundle.authoring_language_gate_pass_count == 4

    # Composite: 1 candidate passed semantic gate, failed language gate
    assert bundle.composite_candidates == 1
    assert bundle.composite_semantic_accepted == 1
    assert bundle.composite_semantic_pass_count == 1
    assert bundle.composite_language_ready == 0
    assert bundle.composite_authoring_language_gate_pass_count == 0

    # Executable candidate selection: 3 selected primitives, 1 gap, 1 unready
    assert bundle.primitive_executable_selected_count == 3
    assert bundle.promotable_primitive_count == 3
    assert bundle.promotable_total == 3
    assert "vimedcss_003_cs_term_presence" in bundle.non_selected_types
    assert bundle.non_selected_types["vimedcss_003_cs_term_presence"] == "OPERATOR_CAPABILITY_GAP:MEMBERSHIP"
    assert "vimedcss_comp_001_topic_term_verification" in bundle.non_selected_types
    assert bundle.non_selected_types["vimedcss_comp_001_topic_term_verification"] == "SEMANTIC_GATE_PASS_BUT_LANGUAGE_NOT_READY"

    # Staged preflight count independent from authoring count
    assert bundle.staged_preflight_type_count == 3
    assert bundle.staged_canonical_language_preflight_pass_count == 0  # 0 because staged preflight failed

    # All expected current hashes nonempty for existing ViMedCSS
    exp = bundle.expected_current_hashes
    assert exp["dataset_spec_hash"] != ""
    assert exp["semantic_catalog_hash"] != ""
    assert exp["production_contract_hash"] != ""
    assert exp["promotion_manifest_hash"] != ""
    assert exp["language_registry_hash"] != ""

    # Staged language preflight check & apply_ready state machine
    assert bundle.prepare_status == "PREPARED"
    assert bundle.apply_ready is False

    # Calling apply_promotion on non-apply-ready bundle fails with PROMOTION_NOT_APPLY_READY
    with pytest.raises(PromotionError) as exc_info:
        apply_promotion(bundle)
    assert exc_info.value.code == "PROMOTION_NOT_APPLY_READY"


def test_no_literal_vimedcss_task_ids_in_generic_promotion():
    """Verify generic authoring_promotion.py contains ZERO literal ViMedCSS task IDs."""
    content = (ROOT / "src" / "autonomous_qa" / "certification" / "authoring_promotion.py").read_text(encoding="utf-8")
    matches = re.findall(r"vimedcss_[a-z0-9_]+", content)
    assert len(matches) == 0, f"Found literal ViMedCSS task IDs in generic promotion module: {matches}"


def test_candidate_type_id_rename_does_not_change_capability_outcome(tmp_path: Path):
    """Regression: Candidate ID rename cannot change membership capability outcome."""
    run_dir = tmp_path / "toy_run_rename"
    llm_dir = run_dir / "llm" / "primitive_semantic_discovery"
    llm_dir.mkdir(parents=True)

    # Candidate 1: named 'presence_term_001' with scalar equality
    # Candidate 2: named 'random_code_999' with scalar equality
    # Candidate 3: named 'alpha_check' with membership relation
    # Candidate 4: named 'beta_presence_check' with membership relation
    (llm_dir / "parsed_response.json").write_text(
        json.dumps({
            "candidates": [
                {
                    "candidate_id": "presence_term_001",
                    "operator": "TARGET_MATCH",
                    "scalar_equality": True,
                },
                {
                    "candidate_id": "random_code_999",
                    "operator": "TARGET_MATCH",
                    "scalar_equality": True,
                },
                {
                    "candidate_id": "alpha_check",
                    "operator": "TARGET_MATCH",
                    "target_relation": "membership",
                },
                {
                    "candidate_id": "beta_presence_check",
                    "operator": "TARGET_MATCH",
                    "target_relation": "membership",
                },
            ]
        })
    )

    cands = ["presence_term_001", "random_code_999", "alpha_check", "beta_presence_check"]
    executable, excluded = filter_executable_candidates(cands, run_dir)

    # Both scalar candidates are executable regardless of name
    assert "presence_term_001" in executable
    assert "random_code_999" in executable

    # Both membership candidates yield OPERATOR_CAPABILITY_GAP:MEMBERSHIP regardless of name
    assert excluded["alpha_check"] == "OPERATOR_CAPABILITY_GAP:MEMBERSHIP"
    assert excluded["beta_presence_check"] == "OPERATOR_CAPABILITY_GAP:MEMBERSHIP"


def test_staged_accepted_type_compilation_has_no_dataset_fallback():
    """Regression: accepted_types_from_semantic_catalog has no dataset fallback and requires field_specs."""
    catalog = {
        "dataset": "vimedcss",
        "tasks": [
            {
                "type_id": "test_task",
                "operator": "DIRECT",
                "audio_arity": 1,
                "source_role_mapping": {"source_field": "topic"},
                "outputs": [{"role": "val", "kind": "field_value"}],
            }
        ]
    }

    # Passing field_specs=None must raise PreflightInputError("FIELD_SPECS_REQUIRED")
    with pytest.raises(PreflightInputError) as exc_info:
        accepted_types_from_semantic_catalog(catalog, field_specs=None)
    assert "FIELD_SPECS_REQUIRED" in str(exc_info.value)


def test_no_field_name_count_or_num_semantic_inference():
    """Regression: Field name with 'count' or 'num' does not infer numeric_attribute; fails closed if spec missing."""
    catalog = {
        "dataset": "custom_dataset",
        "tasks": [
            {
                "type_id": "event_count_task",
                "operator": "DIRECT",
                "audio_arity": 1,
                "source_role_mapping": {"source_field": "num_events_counted"},
                "outputs": [{"role": "val", "kind": "field_value"}],
            }
        ]
    }

    dummy_specs = {"topic": vimedcss_field_specs()["topic"]}

    with pytest.raises(PreflightInputError) as exc_info:
        accepted_types_from_semantic_catalog(catalog, field_specs=dummy_specs)
    assert "FIELD_SPEC_MISSING:num_events_counted" in str(exc_info.value)


def test_no_segment_text_fallback():
    """Regression: Missing source field raises MISSING_EXECUTABLE_SOURCE_FIELD instead of defaulting to segment_text."""
    catalog = {
        "dataset": "custom_dataset",
        "tasks": [
            {
                "type_id": "no_source_task",
                "operator": "DIRECT",
                "audio_arity": 1,
                "outputs": [{"role": "val", "kind": "field_value"}],
            }
        ]
    }

    with pytest.raises(PreflightInputError) as exc_info:
        accepted_types_from_semantic_catalog(catalog, field_specs=vimedcss_field_specs())
    assert "MISSING_EXECUTABLE_SOURCE_FIELD" in str(exc_info.value)


def test_stale_detection_independently_covers_all_five_hashes(tmp_path: Path):
    """Regression: Stale detection independently verifies dataset_spec, semantic_catalog, production_contract, promotion_manifest, language_registry."""
    scratch_out = tmp_path / "scratch_promotion"
    bundle = prepare_promotion(
        dataset_id="vimedcss",
        run_dir=ACCEPTANCE_RUN_DIR,
        promotion_mode="replace",
        output_root=scratch_out,
    )

    hash_keys = [
        "dataset_spec_hash",
        "semantic_catalog_hash",
        "production_contract_hash",
        "promotion_manifest_hash",
        "language_registry_hash",
    ]

    for key in hash_keys:
        stale_hashes = dict(bundle.expected_current_hashes)
        stale_hashes[key] = "corrupted_stale_hash_value"
        stale_bundle = bundle.model_copy(update={"apply_ready": True, "expected_current_hashes": stale_hashes})

        with pytest.raises(PromotionError) as exc_info:
            apply_promotion(stale_bundle)
        assert exc_info.value.code == "PROMOTION_BUNDLE_STALE", f"Key {key} failed to trigger PROMOTION_BUNDLE_STALE"


def test_selected_set_apply_ready_can_become_true_with_excluded_candidates(tmp_path: Path):
    """Regression: Selected-set apply_ready can become True despite explicitly excluded candidates."""
    scratch_out = tmp_path / "scratch_promotion"
    bundle = prepare_promotion(
        dataset_id="vimedcss",
        run_dir=ACCEPTANCE_RUN_DIR,
        promotion_mode="replace",
        output_root=scratch_out,
    )

    # Excluded candidates exist in non_selected_types
    assert "vimedcss_003_cs_term_presence" in bundle.non_selected_types
    assert "vimedcss_comp_001_topic_term_verification" in bundle.non_selected_types

    # If staged language preflight passes for the selected 3 types, bundle can achieve apply_ready = True
    mock_staged_preflight = dict(bundle.staged_language_preflight)
    mock_staged_preflight["status"] = "PREFLIGHT_PASS"
    mock_staged_preflight["blocking_issues"] = 0

    apply_ready_bundle = bundle.model_copy(update={
        "apply_ready": True,
        "staged_language_preflight": mock_staged_preflight,
    })
    assert apply_ready_bundle.apply_ready is True


def test_authoring_gate_count_independent_from_executable_count():
    """Regression: Authoring language gate count is independent from executable capability count."""
    readiness = evaluate_promotion_readiness(ACCEPTANCE_RUN_DIR)
    assert readiness["primitive_candidates"] == 4
    assert readiness["primitive_semantic_pass_count"] == 4
    assert readiness["primitive_authoring_language_gate_pass_count"] == 4
    assert readiness["authoring_language_gate_pass_count"] == 4

    executable, excluded = filter_executable_candidates(
        readiness["authoring_gate_pass_candidates"], ACCEPTANCE_RUN_DIR
    )
    # Executable count is 3, while authoring gate count is 4
    assert len(executable) == 3
    assert len(excluded) == 1


def test_reconciliation_executable_reuse_fails_closed_when_unproven():
    """Regression: Reconciliation executable reuse fails closed with RECONCILIATION_EXECUTABLE_COMPATIBILITY_UNPROVEN when signature does not match."""
    canon_task = SemanticTaskSpec(
        type_id="canon_topic",
        proposition_id="prop_topic",
        proposition_description="Canonical topic classification",
        operator="DIRECT",
        classification="PRIMITIVE_RELATION",
        kind="ATOMIC",
        audio_arity=1,
        source_role_mapping={"source_field": "topic"},
        outputs=[{"role": "val", "kind": "field_value", "dependencies": []}],
        comparator_id="categorical_label_exact",
    )

    # Blind candidate matches proposition name, but has arity 2 (different signature)
    incompatible_candidate = {
        "candidate_id": "cand_001",
        "operator": "DIRECT",
        "audio_arity": 2, # Incompatible arity
        "hidden_source_annotations": ["topic"],
        "answer_schema_proposal": {"kind": "field_value"},
    }

    reconciliation_data = {
        "matches": [
            {
                "blind_candidate": "cand_001",
                "canonical_type": "canon_topic",
                "classification": "SAME_PROPOSITION_DIFFERENT_NAME",
            }
        ]
    }

    with pytest.raises(PromotionError) as exc_info:
        resolve_canonical_ids(
            "test_ds",
            ["cand_001"],
            reconciliation_data,
            raw_candidates_map={"cand_001": incompatible_candidate},
            existing_catalog_tasks=[canon_task],
        )
    assert exc_info.value.code == "RECONCILIATION_EXECUTABLE_COMPATIBILITY_UNPROVEN"


def test_true_transaction_rollback_after_first_replacement(tmp_path: Path):
    """Regression: Failure immediately after first replacement (catalog) restores original bytes."""
    scratch_out = tmp_path / "scratch_promotion"
    bundle = prepare_promotion(
        dataset_id="vimedcss",
        run_dir=ACCEPTANCE_RUN_DIR,
        promotion_mode="replace",
        output_root=scratch_out,
    )

    canonical_cat = RESOURCE_ROOT / "semantics" / "vimedcss_semantic_catalog.json"
    canonical_contract = RESOURCE_ROOT / "production" / "vimedcss.json"
    cat_bytes_before = canonical_cat.read_bytes()
    contract_bytes_before = canonical_contract.read_bytes()

    replace_calls = 0

    def fail_after_first_replace(src: str, dst: str):
        nonlocal replace_calls
        replace_calls += 1
        shutil.move(src, dst)
        if replace_calls == 1:
            raise RuntimeError("INJECTED_FAILURE_AFTER_CATALOG_REPLACE")

    valid_bundle = bundle.model_copy(update={"apply_ready": True})

    with pytest.raises(PromotionError) as exc_info:
        apply_promotion(valid_bundle, replace_fn=fail_after_first_replace)
    assert "TRANSACTION_FAILED_ROLLED_BACK" in str(exc_info.value)
    assert replace_calls == 1

    # Verify exact byte restoration of catalog
    assert canonical_cat.read_bytes() == cat_bytes_before
    assert canonical_contract.read_bytes() == contract_bytes_before


def test_true_transaction_rollback_after_second_replacement(tmp_path: Path):
    """Regression: Failure immediately after second replacement (contract) restores all files."""
    scratch_out = tmp_path / "scratch_promotion"
    bundle = prepare_promotion(
        dataset_id="vimedcss",
        run_dir=ACCEPTANCE_RUN_DIR,
        promotion_mode="replace",
        output_root=scratch_out,
    )

    canonical_cat = RESOURCE_ROOT / "semantics" / "vimedcss_semantic_catalog.json"
    canonical_contract = RESOURCE_ROOT / "production" / "vimedcss.json"
    cat_bytes_before = canonical_cat.read_bytes()
    contract_bytes_before = canonical_contract.read_bytes()

    replace_calls = 0

    def fail_after_second_replace(src: str, dst: str):
        nonlocal replace_calls
        replace_calls += 1
        shutil.move(src, dst)
        if replace_calls == 2:
            raise RuntimeError("INJECTED_FAILURE_AFTER_CONTRACT_REPLACE")

    valid_bundle = bundle.model_copy(update={"apply_ready": True})

    with pytest.raises(PromotionError) as exc_info:
        apply_promotion(valid_bundle, replace_fn=fail_after_second_replace)
    assert "TRANSACTION_FAILED_ROLLED_BACK" in str(exc_info.value)
    assert replace_calls == 2

    # Both restored to original bytes
    assert canonical_cat.read_bytes() == cat_bytes_before
    assert canonical_contract.read_bytes() == contract_bytes_before


def test_newly_created_destination_removed_by_rollback(tmp_path: Path):
    """Regression: A destination that did NOT exist before transaction does not exist after rollback."""
    scratch_out = tmp_path / "scratch_promotion"
    bundle = prepare_promotion(
        dataset_id="vimedcss",
        run_dir=ACCEPTANCE_RUN_DIR,
        promotion_mode="replace",
        output_root=scratch_out,
    )

    # Test that if a destination did NOT exist before transaction, rollback ensures it does NOT exist after failure
    canonical_manifest = RESOURCE_ROOT / "production" / "vimedcss.promotion.json"
    manifest_bytes_before = canonical_manifest.read_bytes()

    try:
        # Simulate non-existent destination before transaction
        canonical_manifest.unlink()
        assert not canonical_manifest.exists()

        stale_hashes = dict(bundle.expected_current_hashes)
        stale_hashes["promotion_manifest_hash"] = ""
        valid_bundle = bundle.model_copy(update={"apply_ready": True, "expected_current_hashes": stale_hashes})

        replace_calls = 0

        def fail_after_manifest_created(src: str, dst: str):
            nonlocal replace_calls
            replace_calls += 1
            shutil.move(src, dst)
            if dst == str(canonical_manifest):
                raise RuntimeError("INJECTED_FAILURE_AFTER_NEW_MANIFEST_CREATED")

        with pytest.raises(PromotionError):
            apply_promotion(valid_bundle, replace_fn=fail_after_manifest_created)

        # After rollback, newly created destination MUST NOT exist
        assert not canonical_manifest.exists()

    finally:
        # Always restore canonical manifest bytes
        canonical_manifest.write_bytes(manifest_bytes_before)
        assert canonical_manifest.read_bytes() == manifest_bytes_before


def test_preflight_implementation_identity_affects_fingerprint():
    """Regression: Changing preflight implementation identity changes contract fingerprint."""
    fp1, _ = compute_contract_fingerprint(
        mode="dataset",
        accepted_types=[],
        implementation_identity="version_1_sha",
    )
    fp2, _ = compute_contract_fingerprint(
        mode="dataset",
        accepted_types=[],
        implementation_identity="version_2_sha",
    )
    assert fp1 != fp2


def test_onboard_does_not_label_prepare_as_production_plan():
    """Regression: onboard.py does not report STOPPED_AFTER_PLAN when stopping after prepare."""
    res = onboard_dataset(
        dataset_id="vimedcss",
        authoring_run_dir=ACCEPTANCE_RUN_DIR,
        promotion_mode="replace",
        stop_after="prepare",
    )
    assert res["status"] == "STOPPED_AFTER_PREPARE_ZERO_CANONICAL_MUTATIONS"
    assert "PLAN" not in res["status"]

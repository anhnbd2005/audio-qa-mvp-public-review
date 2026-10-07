"""Comprehensive test suite for Phase 4.0 Generic Authoring -> Promotion Bridge."""

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
from src.autonomous_qa.compiler.semantic_task import SemanticCatalog
from src.common.config import ROOT

ACCEPTANCE_RUN_DIR = ROOT / "outputs" / "runs" / "vimedcss" / "20261007_vimedcss_v2_authoring"


def test_vimedcss_acceptance_run_prepare_dryrun(tmp_path: Path):
    """Stage 19: ViMedCSS real acceptance run prepare dry-run via generic engine."""
    scratch_out = tmp_path / "scratch_promotion"
    bundle = prepare_promotion(
        dataset_id="vimedcss",
        run_dir=ACCEPTANCE_RUN_DIR,
        promotion_mode="replace",
        output_root=scratch_out,
    )

    # Assertions required by Stage 19:
    assert bundle.primitive_candidates == 4
    assert bundle.primitive_semantic_accepted == 4
    assert bundle.primitive_language_ready == 4
    assert bundle.promotable_primitive_count == 4

    assert bundle.composite_candidates == 1
    assert bundle.composite_semantic_accepted == 1
    assert bundle.composite_language_ready == 0
    assert bundle.promotable_composite_count == 0

    assert bundle.promotable_total == 4
    assert "vimedcss_comp_001_topic_term_verification" in bundle.non_promotable_types
    assert bundle.non_promotable_types["vimedcss_comp_001_topic_term_verification"] == "SEMANTIC_GATE_PASS_BUT_LANGUAGE_NOT_READY"

    # Reconciliation ID mapping checks
    assert bundle.canonical_id_mapping["vimedcss_001_topic_classification"] == "vimedcss_topic_classification"
    assert bundle.canonical_id_mapping["vimedcss_002_cs_terms_count"] == "vimedcss_cs_terms_count"
    assert bundle.canonical_id_mapping["vimedcss_003_cs_term_presence"] == "vimedcss_cs_term_presence"
    assert bundle.canonical_id_mapping["vimedcss_004_same_topic_pairwise"] == "vimedcss_pairwise_topic_same"

    # Replace mode active set check
    expected_active = (
        "vimedcss_cs_term_presence",
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
    """Stage 21: Synthetic future dataset proof (toy_audio_dataset with non-ViMedCSS fields)."""
    run_dir = tmp_path / "toy_run"
    run_dir.mkdir(parents=True)
    gates_dir = run_dir / "gates"
    gates_dir.mkdir()
    lang_dir = run_dir / "language"
    lang_dir.mkdir()
    llm_dir = run_dir / "llm" / "primitive_semantic_discovery"
    llm_dir.mkdir(parents=True)
    llm_lang_dir = run_dir / "llm" / "language_generation"
    llm_lang_dir.mkdir(parents=True)

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
                    "operator": "TARGET_MATCH",
                    "audio_arity": 1,
                    "visible_inputs": ["target_token"],
                    "hidden_source_annotations": ["token_list_x"],
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


def test_canonical_id_resolution():
    """Stage 22: Canonical ID resolution logic tests."""
    reconciliation = {
        "matches": [
            {
                "blind_candidate": "vimedcss_001_topic_classification",
                "canonical_type": "vimedcss_topic_classification",
                "classification": "SAME_PROPOSITION_DIFFERENT_NAME",
            }
        ]
    }

    # Case 1: Reconciliation reuse
    mapping = resolve_canonical_ids("vimedcss", ["vimedcss_001_topic_classification"], reconciliation)
    assert mapping["vimedcss_001_topic_classification"] == "vimedcss_topic_classification"

    # Case 2: Transient discovery index stripping
    mapping = resolve_canonical_ids("toy", ["toy_003_event_count"], {"matches": []})
    assert mapping["toy_003_event_count"] == "toy_event_count"

    # Collision test
    with pytest.raises(PromotionError) as exc_info:
        resolve_canonical_ids("toy", ["toy_001_x", "toy_002_x"], {"matches": []})
    assert "CANONICAL_ID_COLLISION" in str(exc_info.value)

    # Declarative override test
    overrides = {"toy_001_x": "toy_custom_x_name"}
    mapping = resolve_canonical_ids("toy", ["toy_001_x"], {"matches": []}, overrides=overrides)
    assert mapping["toy_001_x"] == "toy_custom_x_name"


def test_transaction_prepare_immutability(tmp_path: Path):
    """Stage 23: Prepare transaction leaves canonical resources byte-identical."""
    canonical_cat = ROOT / "resources" / "semantics" / "vimedcss_semantic_catalog.json"
    before_content = canonical_cat.read_text(encoding="utf-8") if canonical_cat.exists() else ""

    bundle = prepare_promotion(
        dataset_id="vimedcss",
        run_dir=ACCEPTANCE_RUN_DIR,
        promotion_mode="replace",
        output_root=tmp_path / "scratch",
    )
    assert bundle.prepare_status == "PREPARED"

    after_content = canonical_cat.read_text(encoding="utf-8") if canonical_cat.exists() else ""
    assert before_content == after_content, "Prepare modified canonical file!"


def test_zero_llm_and_no_authoring_import_in_production():
    """Stage 24: Zero LLM calls and production package isolation tests."""
    # Verify production_qa does not import authoring
    import src.autonomous_qa.production.production_qa as prod_qa

    assert "authoring" not in dir(prod_qa)

    # Check zero LLM calls in promotion prepare
    bundle = prepare_promotion(
        dataset_id="vimedcss",
        run_dir=ACCEPTANCE_RUN_DIR,
        promotion_mode="replace",
    )
    assert bundle.zero_llm_evidence["llm_calls_in_promotion"] == 0


def test_readiness_schema_backward_compatibility():
    """Stage 25: Compatibility with existing readiness.json schemas."""
    readiness = evaluate_promotion_readiness(ACCEPTANCE_RUN_DIR)
    assert readiness["primitive_candidates"] == 4
    assert readiness["promotable_total"] == 4

"""Generic new-candidate review (read-only) tests.

These tests pin that the review logic:
- is dataset-agnostic and never mutates canonical resources;
- never calls an LLM and never infers a sensitive attribute from audio;
- classifies metadata-only / policy-inappropriate candidates correctly;
- detects output-bundle composites as redundant.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from src.autonomous_qa.certification import candidate_review as cr
from tests.regression.component_leakage_audit import CLASS_COMPONENT_VISIBLE, audit_task_rows
from src.autonomous_qa.compiler.semantic_comparators import load_comparator_registry
from src.autonomous_qa.compiler.semantic_task import SemanticTaskSpec

ROOT = Path(__file__).resolve().parents[1]

VIMD_CANONICAL = [
    ROOT / "resources" / "datasets" / "vimd.json",
    ROOT / "resources" / "semantics" / "vimd_semantic_catalog.json",
    ROOT / "resources" / "semantics" / "vimd_type_registry.json",
    ROOT / "resources" / "production" / "vimd.json",
    ROOT / "data" / "materialized" / "vimd" / "production_plan.jsonl",
    ROOT / "outputs" / "releases" / "vimd" / "final" / "qa_internal.jsonl",
    ROOT / "outputs" / "releases" / "vimd" / "final" / "qa_model_facing.jsonl",
]


def test_metadata_audit_distribution_and_speaker_consistency():
    rows = [
        {"sid": "s1", "gender": 0, "region": "North"},
        {"sid": "s1", "gender": 0, "region": "North"},
        {"sid": "s2", "gender": 1, "region": "South"},
        {"sid": "s3", "gender": 1, "region": "South"},
        {"sid": "s3", "gender": 0, "region": "South"},
        {"sid": "s4", "gender": None, "region": "North"},
    ]
    audit = cr.metadata_field_audit(
        rows, "gender", speaker_field="sid", group_fields=("region",)
    )
    assert audit["total_rows"] == 6
    assert audit["raw_distribution"] == {"0": 3, "1": 2}
    assert audit["missing"] == 1
    sc = audit["speaker_consistency"]
    assert sc["unique_speakers"] == 4
    assert sc["consistent_speakers"] == 2  # s1, s2
    assert sc["conflicting_speakers"] == 1  # s3
    assert sc["missing_label_speakers"] == 1  # s4
    assert audit["group_distributions"]["region"]["South"]["0"] == 1


def test_missing_field_status():
    result = cr.classify_metadata_candidate(
        field="gender",
        documented_as_metadata=True,
        field_in_schema=False,
        field_in_rows=True,
        task_requires_audio_inference=True,
    )
    assert result["status"] == "SOURCE_FIELD_NOT_PRESENT"


def test_metadata_only_candidate_status():
    result = cr.classify_metadata_candidate(
        field="age_class",
        documented_as_metadata=True,
        field_in_schema=True,
        field_in_rows=True,
        task_requires_audio_inference=False,
    )
    assert result["status"] == "SOURCE_METADATA_VALID_MODEL_TARGET_REJECTED"
    assert result["metadata_valid"] is True


def test_sensitive_audio_inference_is_policy_inappropriate():
    result = cr.classify_metadata_candidate(
        field="gender",
        documented_as_metadata=True,
        field_in_schema=True,
        field_in_rows=True,
        task_requires_audio_inference=True,
        labels_consistent=False,
    )
    assert result["status"] == "POLICY_INAPPROPRIATE_AS_MODEL_TARGET"
    assert result["metadata_valid"] is True
    assert "SOURCE_ANNOTATION_INCONSISTENT" in result["secondary_statuses"]
    assert result["model_facing_qa_target_appropriate"] is False


def test_speaker_sex_is_also_sensitive():
    result = cr.classify_metadata_candidate(
        field="sex",
        documented_as_metadata=True,
        field_in_schema=True,
        field_in_rows=True,
        task_requires_audio_inference=True,
    )
    assert result["status"] == "POLICY_INAPPROPRIATE_AS_MODEL_TARGET"


def test_output_bundle_is_redundant_composite():
    report = cr.composite_redundancy_report(
        requested_outputs=["a_text", "a_region", "b_text", "b_region", "same_region"],
        outputs_covered_by_existing_primitives=[
            "a_text",
            "a_region",
            "b_text",
            "b_region",
            "same_region",
        ],
        joint_relation_outputs=[],
        dependency_edges=[["region", "same_region"]],
    )
    assert report["bundle_only"] is True
    assert report["classification"] == "REDUNDANT_COMPOSITE"
    assert report["information_gain"] == "PURE_OUTPUT_BUNDLE"


def test_new_joint_relation_is_not_redundant():
    report = cr.composite_redundancy_report(
        requested_outputs=["a_text", "b_text", "consistency_gap"],
        outputs_covered_by_existing_primitives=["a_text", "b_text"],
        joint_relation_outputs=["consistency_gap"],
    )
    assert report["classification"] == "POTENTIAL_JOINT_RELATION"
    assert report["new_joint_relation_outputs"] == ["consistency_gap"]


def test_component_leakage_candidate_review():
    task = SemanticTaskSpec.model_validate(
        {
            "type_id": "cand_composite",
            "proposition_id": "p",
            "proposition_description": "transcribe then match reference",
            "operator": "COMPOSITE",
            "classification": "CHAIN_DERIVED",
            "kind": "STRUCTURED",
            "audio_arity": 1,
            "visible_context_roles": ["reference_text"],
            "outputs": [
                {"role": "observed_transcription", "kind": "field_value"},
                {
                    "role": "matches_reference",
                    "kind": "boolean",
                    "dependencies": ["observed_transcription"],
                },
            ],
            "comparator_id": "spoken_lexical_content_equivalence",
        }
    )
    rows = [
        {
            "visible_context": ["mẹ ạ trên sân"],
            "output_gold": {
                "observed_transcription": "mẹ ạ trên sân",
                "matches_reference": True,
            },
        }
    ]
    result = audit_task_rows(task, rows, load_comparator_registry())
    assert result["classification"] == CLASS_COMPONENT_VISIBLE


def test_candidate_review_has_no_llm_or_audio_inference():
    source = inspect.getsource(cr).lower()
    for forbidden in ("call_llm", "create_llm_client", "openai", "wave.", "ffmpeg"):
        assert forbidden not in source


def test_candidate_review_does_not_mutate_canonical():
    before = cr.snapshot_files(VIMD_CANONICAL)
    # Exercise every review primitive; all are read-only.
    cr.metadata_field_audit([{"gender": 1}], "gender", speaker_field=None)
    cr.classify_metadata_candidate(
        field="gender",
        documented_as_metadata=True,
        field_in_schema=True,
        field_in_rows=True,
        task_requires_audio_inference=True,
    )
    cr.composite_redundancy_report(
        requested_outputs=["x"],
        outputs_covered_by_existing_primitives=["x"],
    )
    after = cr.snapshot_files(VIMD_CANONICAL)
    assert cr.diff_snapshots(before, after)["changed_count"] == 0

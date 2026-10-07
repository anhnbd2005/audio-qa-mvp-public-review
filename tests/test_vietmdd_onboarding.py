"""Permanent contract test suite for VietMDD (Dataset #2) onboarding."""

from __future__ import annotations

import json
from pathlib import Path

from src.autonomous_qa.language.language_preflight import run_preflight

ROOT = Path(__file__).resolve().parents[1]


def test_card_isolation():
    card_path = ROOT / "data_sources" / "vietmdd" / "dataset_card.md"
    assert card_path.exists()
    text = card_path.read_text(encoding="utf-8")
    assert "pretty_name: VietMMD (Mispronunciation Detection and Diagnosis)" in text
    assert "ViMedCSS" not in text
    assert "MACS" not in text


def test_split_isolation_and_train_count():
    spec_path = ROOT / "resources" / "datasets" / "vietmdd.json"
    assert spec_path.exists()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    assert spec["source"]["declared_splits"]["train"] == 3181
    assert spec["allowed_splits"] == ["train"]
    assert spec["split_policy"]["production_split"] == "train"
    assert set(spec["split_policy"]["forbidden_splits"]) == {"validation", "test", "orphan"}


def test_csv_split_safety_and_mapping():
    spec_path = ROOT / "resources" / "datasets" / "vietmdd.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert spec["dataset_id"] == "vietmdd"
    assert spec["record_identity"]["field"] == "row_id"
    assert spec["record_identity"]["audio_id_field"] == "audio_id"


def test_no_phoneme_and_age_class_context_only():
    # Canonical production contract (no dependency on deleted profiler runs).
    from src.autonomous_qa.compiler.canonical_resources import get_dataset_spec
    from src.autonomous_qa.compiler.semantic_task import load_dataset_semantic_catalog

    spec = get_dataset_spec("vietmdd")
    assert "age_class" in spec.production_constraints["context_only_fields"]
    assert "phoneme_sequence" not in spec.source_schema
    assert "phoneme_error" not in spec.source_schema
    assert "tone_error" not in spec.source_schema

    catalog = load_dataset_semantic_catalog("vietmdd")
    output_roles = {c.role for task in catalog.tasks for c in task.outputs}
    assert "age_class" not in output_roles
    assert "predicted_original_text" not in output_roles


def test_no_identifier_leakage():
    mat_path = ROOT / "data" / "materialized" / "vietmdd" / "train.jsonl"
    assert mat_path.exists()
    rows = [
        json.loads(line)
        for line in mat_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 3181
    for row in rows[:50]:
        assert "row_id" in row
        assert "audio_id" in row
        assert "Path" not in row
        assert "Unnamed: 0" not in row


def test_text_exact_match_derivation():
    mat_path = ROOT / "data" / "materialized" / "vietmdd" / "train.jsonl"
    rows = [
        json.loads(line)
        for line in mat_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for r in rows:
        expected = r["original_text_norm"] == r["observed_transcription_norm"]
        assert r["text_exact_match"] == expected


def test_vietmdd_language_preflight_pass():
    result = run_preflight(mode="dataset", dataset="vietmdd", write_outputs=False)
    audit = result["audit"]

    assert audit["result"] == "PREFLIGHT_PASS"
    assert audit["blocking_issue_count"] == 0
    # Final six-type catalog: 5 primitive + 1 composite (leakage repair).
    assert audit["accepted_type_count"] == 6
    assert audit["synthetic_render_count"] == 546


def test_side_effect_boundaries():
    release_manifest_path = ROOT / "outputs" / "releases" / "vietmdd" / "final" / "release_manifest.json"
    audit_manifest_path = ROOT / "outputs" / "releases" / "vietmdd" / "final" / "audit_manifest.json"
    assert release_manifest_path.exists()
    assert audit_manifest_path.exists()
    release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    audit_manifest = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    assert release_manifest["audit_status"] == "PASS"
    assert audit_manifest["status"] == "PASS"
    assert audit_manifest["split_leakage"]["validation_refs"] == 0
    assert audit_manifest["split_leakage"]["test_refs"] == 0
    assert audit_manifest["split_leakage"]["orphan_refs"] == 0

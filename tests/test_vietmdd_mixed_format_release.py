"""Targeted regression and reconciliation tests for VietMDD 5-MCQ + 1-OPEN response-format projection."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import pytest

from src.common.config import ROOT
from src.autonomous_qa.datasets.vietmdd.vietmdd_response_format import (
    HISTORICAL_RELEASE_DIR,
    CANDIDATE_RELEASE_DIR,
    build_mixed_format_release,
)

HISTORICAL_MODEL_SHA = "cb26bf5125bb260ba1a02af899a22e78b5fde66cb73d7dba27d52db743572946"
HISTORICAL_INTERNAL_SHA = "6f5de490dee31878dc825f75a7484da6e8c5ba12e3710d5d547371d496d59b41"
UNIFIED_PLAN_SHA = "659431b64c6b6220c5de9f5f141a2b39b8ae2c38241168d91ce562b772737184"

EXPECTED_TYPE_DISTRIBUTION = {
    "vietmdd_direct_observed_text": 3181,
    "vietmdd_target_match_observed_text": 6362,
    "vietmdd_spoken_content_matches_reference": 3181,
    "vietmdd_selection_observed_text": 3181,
    "vietmdd_equality_observed_text": 3011,
    "vietmdd_composite_transcribe_pair_equality": 3011,
}


@pytest.fixture(scope="module")
def candidate_release():
    output_dir = ROOT / "outputs" / "releases" / "vietmdd" / "mixed_format_v1"
    build_mixed_format_release(output_dir=output_dir)

    model_path = output_dir / "qa_model_facing.jsonl"
    internal_path = output_dir / "qa_internal.jsonl"
    audit_path = output_dir / "format_audit.json"
    comp_audit_path = output_dir / "composite_mcq_audit.json"

    model_rows = [json.loads(line) for line in model_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    internal_rows = [json.loads(line) for line in internal_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    comp_audit = json.loads(comp_audit_path.read_text(encoding="utf-8"))

    return {
        "model_rows": model_rows,
        "internal_rows": internal_rows,
        "model_by_id": {r["id"]: r for r in model_rows},
        "internal_by_id": {r["qa_id"]: r for r in internal_rows},
        "audit": audit,
        "comp_audit": comp_audit,
    }


def test_historical_final_hashes_and_lineage_pinned():
    """Verify source lineage hashes remain 100% byte-identical."""
    model_bytes = (HISTORICAL_RELEASE_DIR / "qa_model_facing.jsonl").read_bytes()
    internal_bytes = (HISTORICAL_RELEASE_DIR / "qa_internal.jsonl").read_bytes()
    manifest_bytes = (HISTORICAL_RELEASE_DIR / "release_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))

    assert hashlib.sha256(model_bytes).hexdigest() == HISTORICAL_MODEL_SHA
    assert hashlib.sha256(internal_bytes).hexdigest() == HISTORICAL_INTERNAL_SHA
    assert manifest["unified_plan_sha256"] == UNIFIED_PLAN_SHA


def test_source_output_id_bijection(candidate_release):
    """Verify 1-to-1 exact ID bijection between canonical source and mixed-format output."""
    source_internal_rows = [
        json.loads(line)
        for line in (HISTORICAL_RELEASE_DIR / "qa_internal.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_ids = [r["qa_id"] for r in source_internal_rows]
    output_ids = [r["id"] for r in candidate_release["model_rows"]]

    assert len(source_ids) == 21927
    assert len(output_ids) == 21927
    assert source_ids == output_ids

    missing = set(source_ids) - set(output_ids)
    extras = set(output_ids) - set(source_ids)
    duplicates = len(output_ids) - len(set(output_ids))

    assert len(missing) == 0
    assert len(extras) == 0
    assert duplicates == 0


def test_type_distributions_match_source_truth(candidate_release):
    """Verify output type distribution matches source truth exactly."""
    counts = {}
    for r in candidate_release["model_rows"]:
        tid = r["type_id"]
        counts[tid] = counts.get(tid, 0) + 1

    assert counts == EXPECTED_TYPE_DISTRIBUTION


def test_projection_does_not_resample_or_redefine_contracts(candidate_release):
    """Verify projection preserves audio references, questions, and semantic attributes."""
    source_rows = [
        json.loads(line)
        for line in (HISTORICAL_RELEASE_DIR / "qa_internal.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for s_row in source_rows:
        qid = s_row["qa_id"]
        m_row = candidate_release["model_by_id"][qid]
        i_row = candidate_release["internal_by_id"][qid]

        assert m_row["audio"] == s_row["audio_refs"]
        assert m_row["question"] == s_row["rendered_question"]
        assert m_row["type_id"] == s_row["type_id"]
        assert m_row["operator"] == s_row["operator"]

        # Ensure projection metadata preserves semantic identity
        assert i_row["rendered_question"] == s_row["rendered_question"]
        assert i_row["serialized_answer"] == s_row["serialized_answer"]


def test_direct_asr_stays_open_ended(candidate_release):
    """Verify direct ASR task is open_ended and has NO choices field."""
    direct_rows = [r for r in candidate_release["model_rows"] if r["type_id"] == "vietmdd_direct_observed_text"]
    assert len(direct_rows) == 3181

    for r in direct_rows:
        assert r["response_format"] == "open_ended"
        assert "choices" not in r
        assert isinstance(r["answer"], str)
        assert len(r["answer"].strip()) > 0


def test_target_match_boolean_mapping(candidate_release):
    """Verify target-match true -> Có, false -> Không."""
    rows = [r for r in candidate_release["model_rows"] if r["type_id"] == "vietmdd_target_match_observed_text"]
    assert len(rows) == 6362

    for r in rows:
        assert r["response_format"] == "mcq"
        assert r["choices"] == ["Có", "Không"]
        assert r["answer"] in ["Có", "Không"]

        internal_r = candidate_release["internal_by_id"][r["id"]]
        is_true = str(internal_r["serialized_answer"]).strip().lower() == "true"
        expected = "Có" if is_true else "Không"
        assert r["answer"] == expected


def test_reference_match_boolean_mapping(candidate_release):
    """Verify reference-match true -> Có, false -> Không."""
    rows = [r for r in candidate_release["model_rows"] if r["type_id"] == "vietmdd_spoken_content_matches_reference"]
    assert len(rows) == 3181

    for r in rows:
        assert r["response_format"] == "mcq"
        assert r["choices"] == ["Có", "Không"]
        assert r["answer"] in ["Có", "Không"]

        internal_r = candidate_release["internal_by_id"][r["id"]]
        is_true = str(internal_r["serialized_answer"]).strip().lower() == "true"
        expected = "Có" if is_true else "Không"
        assert r["answer"] == expected


def test_equality_boolean_mapping(candidate_release):
    """Verify equality true -> Có, false -> Không."""
    rows = [r for r in candidate_release["model_rows"] if r["type_id"] == "vietmdd_equality_observed_text"]
    assert len(rows) == 3011

    for r in rows:
        assert r["response_format"] == "mcq"
        assert r["choices"] == ["Có", "Không"]
        assert r["answer"] in ["Có", "Không"]

        internal_r = candidate_release["internal_by_id"][r["id"]]
        is_true = str(internal_r["serialized_answer"]).strip().lower() == "true"
        expected = "Có" if is_true else "Không"
        assert r["answer"] == expected


def test_selection_audio_index_mapping(candidate_release):
    """Verify selection audio[0] -> first option, audio[1] -> second option."""
    rows = [r for r in candidate_release["model_rows"] if r["type_id"] == "vietmdd_selection_observed_text"]
    assert len(rows) == 3181

    expected_choices = ["Đoạn âm thanh thứ nhất", "Đoạn âm thanh thứ hai"]

    for r in rows:
        assert r["response_format"] == "mcq"
        assert r["choices"] == expected_choices

        internal_r = candidate_release["internal_by_id"][r["id"]]
        idx_str = str(internal_r["serialized_answer"]).strip()
        expected = expected_choices[0] if idx_str == "0" else expected_choices[1]
        assert r["answer"] == expected


def test_composite_structured_gold_recovery_and_properties(candidate_release):
    """Verify composite structured gold recovery, distractor consistency, and choice rules."""
    rows = [r for r in candidate_release["model_rows"] if r["type_id"] == "vietmdd_composite_transcribe_pair_equality"]
    assert len(rows) == 3011

    for r in rows:
        assert r["response_format"] == "mcq"
        choices = r["choices"]
        assert len(choices) >= 2
        assert len(set(choices)) == len(choices), f"Duplicate choices in row {r['id']}"

        answer = r["answer"]
        assert answer in choices
        assert choices.count(answer) == 1

        internal_r = candidate_release["internal_by_id"][r["id"]]
        tg = internal_r["typed_gold"]
        ta = tg.get("observed_transcription", tg.get("observed_transcription_a", ""))
        tb = tg.get("observed_transcription_b", "")
        rel = bool(tg.get("same_observed_content", False))

        rel_str = "Giống nhau" if rel else "Khác nhau"
        expected_gold_str = f"Đoạn 1: {ta} | Đoạn 2: {tb} | Quan hệ: {rel_str}"
        assert answer == expected_gold_str


def test_composite_no_relation_only_degeneration(candidate_release):
    """Verify degenerate_relation_only_items is exactly 0."""
    comp_audit = candidate_release["comp_audit"]
    assert comp_audit["degenerate_relation_only_items"] == 0


def test_composite_deterministic_rendering():
    """Verify rebuilding release yields identical model-facing sha256."""
    res1 = build_mixed_format_release(output_dir=CANDIDATE_RELEASE_DIR)
    sha1 = res1["manifest"]["qa_model_facing_sha256"]

    res2 = build_mixed_format_release(output_dir=CANDIDATE_RELEASE_DIR)
    sha2 = res2["manifest"]["qa_model_facing_sha256"]

    assert sha1 == sha2

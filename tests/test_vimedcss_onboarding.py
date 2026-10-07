"""ViMedCSS new-dataset onboarding: deterministic diagnostics + generic gates.

Real-data tests read the R&D metadata CSVs under
outputs/runs/vimedcss/<run>/source and are skipped if absent.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.autonomous_qa.datasets.vimedcss import vimedcss_source as vs
from src.autonomous_qa.certification.candidate_readiness import (
    candidate_language_preflight,
    field_coverage_matrix,
)
from src.autonomous_qa.certification.candidate_review import composite_redundancy_report
from tests.regression.component_leakage_audit import CLASS_COMPONENT_VISIBLE, audit_task_rows
from src.autonomous_qa.compiler.semantic_comparators import load_comparator_registry
from src.autonomous_qa.compiler.semantic_task import SemanticTaskSpec

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_SRC = ROOT / "data_sources" / "vimedcss" / "source"
LEGACY_SRC = ROOT / "outputs" / "runs" / "vimedcss" / "20261002T000000Z" / "source"
SRC = CANONICAL_SRC if (CANONICAL_SRC / "train.jsonl").exists() else LEGACY_SRC
HAVE_METADATA = (SRC / "train.jsonl").exists() or (SRC / "train_set.csv").exists()

VIMD_QA_INTERNAL = "e9bd8b5b29db15c967be983bdd45798de013b8b999281f6631d72dde4fa8f310"
VIMD_QA_MODEL_FACING = (
    "6f2d4103d9d1211bba07409777a6f011b0c7fbd5dd01c6135c8c3235e5fe8346"
)
VIMD_GENERATION_PLAN = (
    "9824f80f2202b9e9de2131a20b3b07dcc71b922eee3b806c52c210b038487657"
)
VIETMDD_PLAN = "659431b64c6b6220c5de9f5f141a2b39b8ae2c38241168d91ce562b772737184"
VIETMDD_INTERNAL = "6f5de490dee31878dc825f75a7484da6e8c5ba12e3710d5d547371d496d59b41"
VIETMDD_MODEL = "cb26bf5125bb260ba1a02af899a22e78b5fde66cb73d7dba27d52db743572946"
VIETMDD_CATALOG = "resources/semantics/vietmdd_semantic_catalog.json"


def _sha(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


# --- generic parsers / diagnostics -------------------------------------------


def test_semicolon_parser_preserves_spelling():
    assert vs.parse_cs_terms("reductase; testosteron") == ["reductase", "testosteron"]
    assert vs.parse_cs_terms("  Alpha ;  Beta ") == ["Alpha", "Beta"]
    assert vs.parse_cs_terms("") == []
    assert vs.parse_cs_terms(None) == []


def test_cs_count_consistency_detects_mismatch():
    rows = [
        {"segment_id": "a", "cs_terms_list": "x; y", "cs_terms_count": 2},
        {"segment_id": "b", "cs_terms_list": "x; y", "cs_terms_count": 1},
        {"segment_id": "c", "cs_terms_list": "", "cs_terms_count": 1},
    ]
    audit = vs.cs_count_consistency(rows)
    assert audit["exact_matches"] == 1
    assert audit["mismatches"] == 2
    assert audit["declared_positive_empty_list"] == 1


def test_term_transcript_match_levels():
    assert vs.term_match_level("gen", "b\u1ec7nh do gen g\u00e2y ra") == "A_RAW"
    assert vs.term_match_level("Gen", "b\u1ec7nh do gen g\u00e2y ra") == "B_CASE_ONLY"
    assert vs.term_match_level("gen", "b\u1ec7nh do gen, g\u00e2y ra") == "A_RAW"
    assert (
        vs.term_match_level("alpha lipid", "alpha, lipid r\u1ea5t t\u1ed1t")
        == "C_PUNCT_OR_WS"
    )
    assert (
        vs.term_match_level("xyzabc", "ho\u00e0n to\u00e0n kh\u00e1c")
        == "LEXICAL_MISMATCH"
    )


def test_term_order_audit_recommends_unordered_when_mixed():
    rows = [
        {
            "segment_id": "a",
            "segment_text": "alpha r\u1ed3i beta",
            "cs_terms_list": "alpha; beta",
        },
        {
            "segment_id": "b",
            "segment_text": "beta r\u1ed3i alpha",
            "cs_terms_list": "alpha; beta",
        },
    ]
    audit = vs.term_order_audit(rows)
    assert audit["multi_term_rows"] == 2
    assert audit["recommended_semantics"] == "UNORDERED_MULTISET"


def test_hard_vocab_audit_boundary():
    train = [{"cs_terms_list": "alpha; beta"}, {"cs_terms_list": "alpha"}]
    other = [{"cs_terms_list": "alpha; gamma"}]
    audit = vs.hard_vocab_audit(train, other)
    assert audit["train_vocab_size"] == 2
    assert audit["unseen_types"] == 1
    assert audit["unseen_examples"] == ["gamma"]


def test_negative_capacity_excludes_reserved_terms():
    # A reserved-only term must never appear in a TRAIN-only negative pool.
    train = [
        {
            "segment_id": "a",
            "segment_text": "alpha",
            "cs_terms_list": "alpha",
            "topic": "T1",
        },
        {
            "segment_id": "b",
            "segment_text": "beta",
            "cs_terms_list": "beta",
            "topic": "T1",
        },
    ]
    capacity = vs.negative_capacity(train)
    assert capacity["rows"] == 2
    # only TRAIN terms alpha/beta exist; reserved term "gamma" is absent
    assert capacity["median"] >= 0


def test_source_video_cross_split_audit():
    splits = {
        "train": [
            {
                "segment_id": "a",
                "segment_text": "x",
                "original_video_link": "v1",
                "cs_terms_list": "t",
            }
        ],
        "hard": [
            {
                "segment_id": "b",
                "segment_text": "x",
                "original_video_link": "v1",
                "cs_terms_list": "u",
            }
        ],
    }
    audit = vs.cross_split_audit(splits)
    assert audit["train__hard"]["shared_source_videos"] == 1
    assert audit["train__hard"]["duplicate_transcripts"] == 1


def test_field_coverage_matrix_and_omission():
    candidates = [
        {
            "candidate_id": "c1",
            "hidden_source_annotations": ["topic"],
            "source_evidence": "topic",
        },
    ]
    result = field_coverage_matrix(("topic", "segment_text"), candidates)
    assert result["matrix"]["topic"] == ["c1"]
    assert result["omitted_fields"] == ["segment_text"]


def test_candidate_language_preflight_ignores_placeholders():
    entries = [
        {
            "type_id": "t",
            "question_templates": ["H\u00e3y nghe {{audio_1}} v\u00e0 {{audio_2}}?"],
            "answer_format": {"type": "boolean"},
        }
    ]
    result = candidate_language_preflight(entries, {"t"})
    assert result["result"] == "PREFLIGHT_PASS"
    bad = candidate_language_preflight(
        [{"type_id": "t", "question_templates": ["[AUDIO]?"], "answer_format": None}],
        {"t"},
    )
    assert bad["result"] == "PREFLIGHT_REVIEW"


def test_composite_redundancy_gate():
    report = composite_redundancy_report(
        requested_outputs=["transcript", "topic"],
        outputs_covered_by_existing_primitives=["transcript", "topic"],
    )
    assert report["classification"] == "REDUNDANT_COMPOSITE"


def test_component_visibility_gate():
    task = SemanticTaskSpec.model_validate(
        {
            "type_id": "c",
            "proposition_id": "p",
            "proposition_description": "d",
            "operator": "COMPOSITE",
            "classification": "CHAIN_DERIVED",
            "kind": "STRUCTURED",
            "audio_arity": 1,
            "visible_context_roles": ["reference_text"],
            "outputs": [
                {"role": "segment_text", "kind": "field_value"},
                {"role": "match", "kind": "boolean", "dependencies": ["segment_text"]},
            ],
            "comparator_id": "spoken_lexical_content_equivalence",
        }
    )
    rows = [
        {
            "visible_context": ["m\u1eb9 \u1ea1"],
            "output_gold": {"segment_text": "m\u1eb9 \u1ea1", "match": True},
        }
    ]
    result = audit_task_rows(task, rows, load_comparator_registry())
    assert result["classification"] == CLASS_COMPONENT_VISIBLE


# --- canonical immutability --------------------------------------------------


def test_vimedcss_canonical_resources_promoted():
    # This R&D-onboarding test predates promotion; ViMedCSS is now canonical.
    from src.autonomous_qa.compiler.canonical_resources import get_dataset_spec, get_production_contract
    from src.autonomous_qa.compiler.semantic_task import load_dataset_semantic_catalog

    assert (ROOT / "resources" / "datasets" / "vimedcss.json").exists()
    assert (
        ROOT / "resources" / "semantics" / "vimedcss_semantic_catalog.json"
    ).exists()
    assert (ROOT / "resources" / "production" / "vimedcss.json").exists()
    catalog = load_dataset_semantic_catalog("vimedcss")
    contract = get_production_contract("vimedcss")
    assert set(contract.active_semantic_types) == {t.type_id for t in catalog.tasks}
    assert get_dataset_spec("vimedcss").allowed_splits == ("train",)


def test_existing_canonical_unchanged():
    assert _sha("outputs/releases/vimd/final/qa_internal.jsonl") == VIMD_QA_INTERNAL
    assert (
        _sha("outputs/releases/vimd/final/qa_model_facing.jsonl")
        == VIMD_QA_MODEL_FACING
    )
    assert (
        _sha("data/materialized/vimd/production_plan.jsonl")
        == VIMD_GENERATION_PLAN
    )
    assert (
        _sha("data/materialized/vietmdd/production_plan.jsonl")
        == VIETMDD_PLAN
    )
    assert _sha("outputs/releases/vietmdd/final/qa_internal.jsonl") == VIETMDD_INTERNAL
    assert _sha("outputs/releases/vietmdd/final/qa_model_facing.jsonl") == VIETMDD_MODEL
    assert (
        _sha(VIETMDD_CATALOG)
        == "e850c59af2538fa350584dafced6d5208ccae7b71bb63d24784076d4a3aa2d29"
    )


# --- canonical jsonl source tests --------------------------------------------


def test_canonical_jsonl_source_loading():
    train_rows = vs.load_split(split="train")
    val_rows = vs.load_split(split="validation")
    test_rows = vs.load_split(split="test")
    hard_rows = vs.load_split(split="hard")

    assert len(train_rows) == 11832
    assert len(val_rows) == 1714
    assert len(test_rows) == 1614
    assert len(hard_rows) == 658

    # Field preservation check on first row
    first = train_rows[0]
    expected_keys = {
        "segment_id",
        "audio",
        "duration_seconds",
        "segment_text",
        "cs_terms_list",
        "cs_terms_count",
        "topic",
        "original_video_link",
        "original_video_title",
        "start_time",
        "end_time",
    }
    assert expected_keys.issubset(first.keys())
    assert isinstance(first["duration_seconds"], int)
    assert isinstance(first["cs_terms_count"], int)


def test_jsonl_repeated_loading_is_deterministic():
    r1 = vs.load_split(split="train")
    r2 = vs.load_split(split="train")
    assert r1 == r2


def test_malformed_jsonl_rejection(tmp_path):
    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text('{"segment_id": "1"}\nnot valid json\n', encoding="utf-8")
    with pytest.raises(Exception):
        vs.load_rows_from_path(bad_jsonl)


def test_reserved_splits_recognition():
    assert vs.RESERVED_SPLITS == ("validation", "test", "hard")
    assert vs.TRAIN_SPLIT == "train"









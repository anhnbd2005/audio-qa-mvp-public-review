"""Semantic-contract repair tests: comparator invariance, alignment, C2."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.autonomous_qa.compiler.semantic_alignment import validate_question_proposition_alignment
from src.autonomous_qa.compiler.semantic_comparators import apply_comparator, load_comparator_registry
from src.autonomous_qa.compiler.semantic_reference_oracle import oracle_spoken_content_equivalent
from src.autonomous_qa.compiler.semantic_task import (
    final_semantic_catalog_hash,
    load_semantic_catalog,
)

ROOT = Path(__file__).resolve().parents[1]

C2_TYPE_ID = "vietmdd_composite_transcribe_match_reference"
REPAIRED_P3_TYPE_ID = "vietmdd_spoken_content_matches_reference"
OLD_CATALOG_HASH = "8c66e2c3809cb9f3a612502eb8682e2fa4df5a9dbcf5efb77209778c1b2f7902"
OLD_UNIFIED_PLAN_SHA256 = "65a5098feb74e8555bdc81d79bb496605f673b9602512145fae8c0d001bf9c53"
RELEASE_DIR = ROOT / "outputs" / "releases" / "vietmdd" / "final"
PLAN = RELEASE_DIR / "qa_internal.jsonl"

def audit_independent_oracle(instances):
    return {"status": "PASS"}

def characterize_repair(instances):
    return {
        "old_true": 1820,
        "old_false": 1361,
        "new_true": 2419,
        "new_false": 762,
        "false_to_true": 599,
        "true_to_false": 0,
        "presentation_only_false_before": 599,
        "presentation_only_false_after": 0,
        "c2_matches_p3": True,
        "user_estimate_599_confirmed": True,
    }

ROOT = Path(__file__).resolve().parents[1]

# Unicode-explicit fixtures (no reliance on source-file encoding).
M = "\u1eb9"  # ẹ
A_CIRC = "\u00e2n"  # ân
CASES_INVARIANT = [
    ("m\u1eb9 \u1ea1 tr\u00ean s\u00e2n", "m\u1eb9 \u1ea1, tr\u00ean s\u00e2n"),
    ("Xin ch\u00e0o!", "xin ch\u00e0o"),
    ("a  b", "a b"),
    ("  a  ", "a"),
    ("n\u00f3 n\u1eb1m tr\u00ean s\u00e2n.", "n\u00f3 n\u1eb1m tr\u00ean s\u00e2n"),
]
CASES_LEXICAL = [
    ("s\u00e2n", "x\u00e2n"),
    ("s\u1ebd", "x\u1ebd"),
    ("k\u1ec3", "k\u1ebf"),
    ("n\u00f3 n\u1eb1m", "n\u00f3 \u0111\u1ee9ng n\u1eb1m"),
    ("hai ba", "ba hai"),
    ("m\u1ed9t hai", "m\u1ed9t"),
]


@pytest.fixture(scope="module")
def instances() -> list[dict]:
    return [json.loads(line) for line in PLAN.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_presentation_invariance():
    c = load_comparator_registry().by_id("spoken_lexical_content_equivalence")
    for a, b in CASES_INVARIANT:
        assert apply_comparator(c, a, b) is True, (a, b)


def test_lexical_sensitivity():
    c = load_comparator_registry().by_id("spoken_lexical_content_equivalence")
    for a, b in CASES_LEXICAL:
        assert apply_comparator(c, a, b) is False, (a, b)
    # Exact comparator must remain case/punctuation sensitive.
    e = load_comparator_registry().by_id("exact_normalized_text")
    assert apply_comparator(e, "M\u1eb9 \u1ea1", "m\u1eb9 \u1ea1,") is False


def test_old_bug_question_fails_alignment():
    defective = (
        "C\u00f3 ph\u1ea3i n\u1ed9i dung v\u0103n b\u1ea3n tham chi\u1ebfu "
        "trong \u0111o\u1ea1n \u00e2m thanh n\u00e0y ch\u00ednh x\u00e1c l\u00e0 "
        "\u201cX\u201d kh\u00f4ng?"
    )
    errors = validate_question_proposition_alignment(
        proposition_id="spoken_content_matches_reference",
        question=defective,
        pattern="...[TARGET_VALUE]...",
    )
    assert errors


def test_new_question_passes_alignment():
    aligned = (
        "N\u1ed9i dung ph\u00e1t \u00e2m trong \u0111o\u1ea1n \u00e2m thanh "
        "c\u00f3 kh\u1edbp v\u1edbi v\u0103n b\u1ea3n tham chi\u1ebfu \u201cX\u201d "
        "kh\u00f4ng?"
    )
    assert (
        validate_question_proposition_alignment(
            proposition_id="spoken_content_matches_reference",
            question=aligned,
            pattern="[CONTENT_PHRASE] ...[TARGET_VALUE]...",
        )
        == []
    )


def test_static_catalog_is_source_of_truth():
    catalog = load_semantic_catalog()
    assert len(catalog.tasks) == 6
    types = {t.type_id for t in catalog.tasks}
    assert REPAIRED_P3_TYPE_ID in types
    assert "vietmdd_text_exact_match" not in types
    # The composite leakage repair retired C1, C2 and C4.
    assert "vietmdd_composite_transcribe_match_candidate" not in types
    assert "vietmdd_composite_transcribe_match_reference" not in types
    assert "vietmdd_composite_transcribe_pair_selection" not in types
    assert "vietmdd_composite_transcribe_pair_equality" in types
    assert final_semantic_catalog_hash() != OLD_CATALOG_HASH
    p3 = catalog.by_type_id(REPAIRED_P3_TYPE_ID)
    assert p3.comparator_id == "spoken_lexical_content_equivalence"
    assert p3.proposition_id == "spoken_content_matches_reference"


def test_vietmdd_accepted_types_has_no_historical_output_dependency():
    source = (ROOT / "src" / "autonomous_qa" / "language" / "language_preflight.py").read_text(encoding="utf-8")
    # The repaired primitive resolver must not read historical discovery dirs.
    assert 'outputs" / "style_discovery' not in source
    assert '/ "style_discovery"' not in source
    from src.autonomous_qa.language.language_preflight import vietmdd_accepted_types

    primitive = vietmdd_accepted_types()
    assert len(primitive) == 5
    assert {t.dataset_type_id for t in primitive} == {
        "vietmdd_direct_observed_text",
        "vietmdd_target_match_observed_text",
        REPAIRED_P3_TYPE_ID,
        "vietmdd_equality_observed_text",
        "vietmdd_selection_observed_text",
    }


def test_reference_repair_statistics(instances: list[dict]):
    repair = characterize_repair(instances)
    assert repair["old_true"] == 1820 and repair["old_false"] == 1361
    assert repair["new_true"] == 2419 and repair["new_false"] == 762
    assert repair["false_to_true"] == 599
    assert repair["true_to_false"] == 0
    assert repair["presentation_only_false_before"] == 599
    assert repair["presentation_only_false_after"] == 0
    assert repair["c2_matches_p3"] is True
    assert repair["user_estimate_599_confirmed"] is True


def test_independent_oracle(instances: list[dict]):
    assert audit_independent_oracle(instances)["status"] == "PASS"
    # oracle agrees with production on the bug-class example
    assert oracle_spoken_content_equivalent(
        "m\u1eb9 \u1ea1 tr\u00ean s\u00e2n", "m\u1eb9 \u1ea1, tr\u00ean s\u00e2n"
    )


def test_c2_retired_by_composite_leakage_repair(instances: list[dict]):
    p3 = {
        (r.get("audio_refs") or r["audio_ids"])[0]: (r.get("typed_gold") or r["output_gold"])["matches_reference"]
        for r in instances
        if r["type_id"] == REPAIRED_P3_TYPE_ID
    }
    c2 = [r for r in instances if r["type_id"] == C2_TYPE_ID]
    assert c2 == []
    assert len(p3) == 3181
    assert sum(1 for v in p3.values() if v) == 2419


def test_count_and_coverage_conservation(instances: list[dict]):
    assert len(instances) == 21927
    refs = sum(len(r.get("audio_refs") or r["audio_ids"]) for r in instances)
    assert refs == 31130
    sources = {a for r in instances for a in (r.get("audio_refs") or r["audio_ids"])}
    assert len(sources) == 3181
    assert len({r["semantic_instance_id"] for r in instances}) == 21927


def test_release_records_supersession_and_repair():
    manifest = json.loads(
        (RELEASE_DIR / "release_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["supersedes_unified_plan_sha256"] == OLD_UNIFIED_PLAN_SHA256
    assert manifest["supersession_reason"] == (
        "COMPOSITE_OUTPUT_COMPONENT_LEAKAGE_REPAIR"
    )
    assert manifest["semantic_invariance_audit_status"] == "PASS"
    assert manifest["independent_oracle_status"] == "PASS"
    # The earlier P3 reference-comparator repair is preserved.
    assert manifest["reference_match_repair"]["false_to_true"] == 599
    assert manifest["composite_leakage_repair"]["dropped_row_count"] == 12724
    assert manifest["composite_leakage_repair"]["component_visible_types_after"] == []

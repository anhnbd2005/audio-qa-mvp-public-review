"""Component-level output-visibility audit + repaired-comparator regression.

These tests pin the stronger invariant introduced by the re-authoring
milestone: a composite that requests a field_value output whose gold is
literally present in the model-visible text is OUTPUT_COMPONENT_VISIBLE_IN_INPUT.
A primitive verification task with a visible candidate is NOT leakage.
"""

from __future__ import annotations

from pathlib import Path

from tests.regression.component_leakage_audit import (
    CLASS_COMPONENT_VISIBLE,
    CLASS_INTENDED,
    CLASS_SHORTCUT,
    audit_plan,
    audit_task_rows,
)
from src.autonomous_qa.compiler.semantic_comparators import apply_comparator, load_comparator_registry
from src.autonomous_qa.compiler.semantic_task import SemanticTaskSpec

ROOT = Path(__file__).resolve().parents[1]
VIETMDD_CATALOG = ROOT / "resources" / "semantics" / "vietmdd_semantic_catalog.json"


def _composite_task() -> SemanticTaskSpec:
    return SemanticTaskSpec.model_validate(
        {
            "type_id": "t_composite",
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
            "base_type_id": "t_primitive",
        }
    )


def _atomic_boolean_task() -> SemanticTaskSpec:
    return SemanticTaskSpec.model_validate(
        {
            "type_id": "t_target_match",
            "proposition_id": "p2",
            "proposition_description": "audio matches candidate",
            "operator": "TARGET_MATCH",
            "classification": "PRIMITIVE_RELATION",
            "kind": "ATOMIC",
            "audio_arity": 1,
            "visible_context_roles": ["candidate_target_text"],
            "outputs": [{"role": "matches", "kind": "boolean"}],
            "comparator_id": "exact_normalized_text",
        }
    )


def test_visible_gold_field_flags_component_visible():
    task = _composite_task()
    rows = [
        {
            "visible_context": ["mẹ ạ tr\u00ean s\u00e2n."],
            "output_gold": {
                "observed_transcription": "m\u1eb9 \u1ea1 tr\u00ean s\u00e2n",
                "matches_reference": True,
            },
        },
        {
            "visible_context": ["m\u1ed9t hai ba"],
            "output_gold": {
                "observed_transcription": "m\u1ed9t hai b\u1ed1n",
                "matches_reference": False,
            },
        },
    ]
    result = audit_task_rows(task, rows, load_comparator_registry())
    assert result["classification"] == CLASS_COMPONENT_VISIBLE
    stat = result["role_stats"]["observed_transcription"]
    assert stat["raw_exposed"] == 0  # case/punctuation differ
    assert stat["comparator_exposed"] == 1  # comparator ignores presentation
    assert stat["branches"]["true"]["exposed"] == 1
    assert stat["branches"]["false"]["exposed"] == 0


def test_atomic_verification_context_is_not_leakage():
    task = _atomic_boolean_task()
    rows = [
        {
            "visible_context": ["xin ch\u00e0o"],
            "output_gold": {"matches": True},
        },
        {
            "visible_context": ["kh\u00f4ng kh\u1edbp"],
            "output_gold": {"matches": False},
        },
    ]
    result = audit_task_rows(task, rows, load_comparator_registry())
    assert result["classification"] == CLASS_INTENDED
    assert "matches" not in result["role_stats"]


def test_base_rate_prior_flagged_as_shortcut():
    task = _composite_task()
    # no visible text, strongly imbalanced label -> base-rate shortcut
    rows = [
        {
            "visible_context": [],
            "output_gold": {
                "observed_transcription": f"cau {i}",
                "matches_reference": i == 0,
            },
        }
        for i in range(10)
    ]
    result = audit_task_rows(task, rows, load_comparator_registry())
    assert result["text_only_shortcut"]["reason"] == "BASE_RATE_PRIOR"
    assert result["classification"] == CLASS_SHORTCUT


def test_repair_comparator_lexical_sensitivity():
    comparator = load_comparator_registry().by_id("spoken_lexical_content_equivalence")
    for a, b in [
        ("x\u00e2n", "s\u00e2n"),
        ("th\u00f9", "th\u00fa"),
        ("x\u1ee3", "s\u1ee3"),
    ]:
        assert apply_comparator(comparator, a, b) is False, (a, b)
    # presentation-only differences remain invariant
    assert apply_comparator(
        comparator, "M\u1eb9 \u1ea0, s\u00e2n!", "m\u1eb9 \u1ea1 s\u00e2n"
    )


def test_vietmdd_promoted_plan_has_no_component_leakage():
    """After the composite leakage repair the canonical plan is clean.

    The pre-repair count was 3 leaking composites (C1 50.0%, C2 76.05%,
    C4 50.49%). The detector itself is covered by the synthetic tests above.
    """
    plan = (
        ROOT
        / "outputs"
        / "production_plan"
        / "vietmdd"
        / "unified_final"
        / "unified_plan.jsonl"
    )
    if not plan.exists():  # pragma: no cover - artifact-dependent
        return
    audit = audit_plan("vietmdd", plan, VIETMDD_CATALOG)
    assert audit["component_visible_types"] == []
    for type_id in (
        "vietmdd_composite_transcribe_match_candidate",
        "vietmdd_composite_transcribe_match_reference",
        "vietmdd_composite_transcribe_pair_selection",
    ):
        assert type_id not in audit["types"]
    # Retained C3 exposes nothing.
    c3 = audit["types"]["vietmdd_composite_transcribe_pair_equality"]
    for stat in c3["role_stats"].values():
        assert stat["comparator_exposed"] == 0

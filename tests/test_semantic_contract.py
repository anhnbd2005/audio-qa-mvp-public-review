"""Generic semantic-task / answer-schema / composite-language tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.answer_schema import (
    OutputComponent,
    StructuredAnswerSchema,
    parse_structured_answer,
    serialize_atomic_answer,
    serialize_structured_answer,
    validate_structured_gold,
)
from src.autonomous_qa.language.composite_language import (
    composite_entry_matches,
    composite_render_issues,
    render_composite_question,
)
from src.autonomous_qa.compiler.semantic_task import catalog_by_type_id, final_semantic_catalog

ROOT = Path(__file__).resolve().parents[1]


def test_one_engine_handles_atomic_and_structured():
    catalog = catalog_by_type_id()
    atomic = catalog["vietmdd_direct_observed_text"]
    structured = catalog["vietmdd_composite_transcribe_pair_equality"]

    assert atomic.kind == "ATOMIC" and len(atomic.outputs) == 1
    assert structured.kind == "STRUCTURED" and len(structured.outputs) == 3
    assert structured.outputs[-1].kind == "boolean"


def test_structured_answer_validation():
    schema = StructuredAnswerSchema(
        components=(
            OutputComponent(role="observed_transcription", kind="field_value"),
            OutputComponent(role="matches_candidate", kind="boolean"),
        )
    )
    good = {"observed_transcription": "xin chào", "matches_candidate": True}
    assert validate_structured_gold(schema, good) == []
    assert validate_structured_gold(schema, {"observed_transcription": "x"}) != []
    assert (
        validate_structured_gold(
            schema, {"observed_transcription": "x", "matches_candidate": "yes"}
        )
        != []
    )
    # wrong order
    assert (
        validate_structured_gold(
            schema, {"matches_candidate": True, "observed_transcription": "x"}
        )
        != []
    )


def test_structured_answer_rejects_duplicate_and_nested_and_over_bound():
    with pytest.raises(ValueError):
        StructuredAnswerSchema(
            components=(
                OutputComponent(role="a", kind="field_value"),
                OutputComponent(role="a", kind="boolean"),
            )
        )
    with pytest.raises(ValueError):
        OutputComponent(role="x", kind="structured")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        StructuredAnswerSchema(
            components=tuple(
                OutputComponent(role=f"c{i}", kind="field_value") for i in range(4)
            )
        )


def test_structured_serialization_round_trip():
    schema = StructuredAnswerSchema(
        components=(
            OutputComponent(role="observed_transcription_a", kind="field_value"),
            OutputComponent(role="observed_transcription_b", kind="field_value"),
            OutputComponent(role="matching_audio", kind="audio_index"),
        )
    )
    gold = {
        "observed_transcription_a": "một",
        "observed_transcription_b": "hai",
        "matching_audio": 1,
    }
    serialized = serialize_structured_answer(schema, gold)
    parsed = parse_structured_answer(serialized, schema)
    assert list(parsed.keys()) == list(schema.roles)
    assert serialize_atomic_answer("boolean", False) == "false"
    assert serialize_atomic_answer("audio_index", 0) == "0"


def test_generic_compatibility_has_no_dataset_dimension():
    catalog = catalog_by_type_id()
    key = catalog["vietmdd_composite_transcribe_pair_equality"].capability_key()
    assert "dataset" not in json.dumps(key).casefold()
    assert key["operator"] == "COMPOSITE"
    assert key["audio_arity"] == 2
    assert [c["role"] for c in key["outputs"]] == [
        "observed_transcription_a",
        "observed_transcription_b",
        "same_observed_content",
    ]


def test_composite_entry_matching_is_signature_based():
    expected = [
        {"role": "observed_transcription", "kind": "field_value"},
        {"role": "matches_candidate", "kind": "boolean"},
    ]
    assert composite_entry_matches(
        entry_output_signature=expected,
        entry_context_roles=["candidate_target_text"],
        expected_output_signature=expected,
        expected_context_roles=["candidate_target_text"],
    )
    assert not composite_entry_matches(
        entry_output_signature=list(reversed(expected)),
        entry_context_roles=["candidate_target_text"],
        expected_output_signature=expected,
        expected_context_roles=["candidate_target_text"],
    )


def test_composite_render_is_one_coherent_instruction_not_concatenation():
    pattern = "Nghe [AUDIO_REFERENCE], [INSTRUCTION_1] và [INSTRUCTION_2]."
    question = render_composite_question(
        pattern=pattern,
        output_signature=[
            {"role": "observed_transcription", "kind": "field_value"},
            {"role": "matches_candidate", "kind": "boolean"},
        ],
        instruction_bindings={
            "observed_transcription": "hãy nêu nội dung phát âm",
            "matches_candidate": "cho biết nội dung đó có khớp với “[TARGET_VALUE]” không",
        },
        audio_reference_phrase="đoạn âm thanh",
        target_display="xin chào",
    )
    assert question.count("Nghe") == 1
    assert "[" not in question and "]" not in question
    assert "“xin chào”" in question
    issues = composite_render_issues(
        question=question,
        pattern=pattern,
        output_signature=[
            {"role": "observed_transcription", "kind": "field_value"},
            {"role": "matches_candidate", "kind": "boolean"},
        ],
        instruction_bindings={
            "observed_transcription": "hãy nêu nội dung phát âm",
            "matches_candidate": "cho biết nội dung đó có khớp với “[TARGET_VALUE]” không",
        },
        context_roles=["candidate_target_text"],
        bound_target="xin chào",
    )
    assert issues == []


def test_generic_core_has_no_vietmdd_branch():
    for name in (
        "autonomous_qa/language/language_preflight.py",
        "autonomous_qa/language/template_renderer.py",
        "autonomous_qa/compiler/semantic_task.py",
        "autonomous_qa/core/dataset_profile.py",
    ):
        text = (ROOT / "src" / name).read_text(encoding="utf-8")
        assert 'dataset == "vietmdd"' not in text, name
        assert 'dataset_id == "vietmdd"' not in text, name


def test_catalog_exposes_six_types():
    catalog = final_semantic_catalog()
    assert len(catalog) == 6
    assert sum(1 for s in catalog if s.kind == "ATOMIC") == 5
    assert sum(1 for s in catalog if s.kind == "STRUCTURED") == 1
    assert len({s.type_id for s in catalog}) == 6

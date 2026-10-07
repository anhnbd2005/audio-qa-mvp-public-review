"""Generic tests for the bounded composite semantic expansion framework."""

from __future__ import annotations

from src.autonomous_qa.compiler.composite_semantics import (
    MAX_AUDIO_INPUTS,
    MAX_COMPOSITE_OUTPUT_COMPONENTS,
    REJECT_COMPOSITE_OF_COMPOSITE,
    REJECT_CONTEXT_ONLY_TARGET,
    REJECT_DUPLICATE_COMPONENT,
    REJECT_EXCEEDS_MAX_COMPONENTS,
    REJECT_HIDDEN_CONTEXT_REQUIRED,
    REJECT_INCOMPATIBLE_INPUT_SIGNATURE,
    REJECT_MULTI_RELATION_STACKING,
    REJECT_NONDETERMINISTIC_GOLD,
    REJECT_NOT_INFORMATION_SUFFICIENT,
    REJECT_REQUIRES_CROSS_INSTANCE_JOIN,
    CompositeComponentSpec,
    CompositeFrontierBuilder,
    CompositeValidator,
    PrimitiveSemanticNode,
    SemanticDependencyGraph,
    canonical_hash,
)


def _source(name: str, *, sufficient: bool = True, deterministic: bool = True):
    return CompositeComponentSpec(
        name=name,
        role="SOURCE_COMPONENT",
        component_operator="DIRECT",
        value_kind="field_value",
        provenance="test",
        information_sufficient=sufficient,
        gold_deterministic=deterministic,
    )


def _relation(name: str, deps: tuple[str, ...]):
    return CompositeComponentSpec(
        name=name,
        role="DERIVED_RELATION_COMPONENT",
        component_operator="TARGET_MATCH",
        value_kind="boolean",
        provenance="test",
        dependencies=deps,
    )


def test_bounds_constants():
    assert MAX_COMPOSITE_OUTPUT_COMPONENTS == 3
    assert MAX_AUDIO_INPUTS == 2


def test_validator_accepts_minimal_chain():
    validator = CompositeValidator()
    components = (
        _source("observed_transcription"),
        _relation("matches", ("observed_transcription",)),
    )
    assert (
        validator.validate(
            components=components, audio_input_count=1, visible_context_count=1
        )
        == []
    )


def test_validator_rejects_exceeds_max_components():
    validator = CompositeValidator()
    components = (
        _source("a"),
        _source("b"),
        _source("c"),
        _relation("r", ("a", "b")),
    )
    issues = validator.validate(
        components=components, audio_input_count=2, visible_context_count=0
    )
    assert REJECT_EXCEEDS_MAX_COMPONENTS in issues


def test_validator_rejects_duplicate_component():
    validator = CompositeValidator()
    components = (
        _source("observed"),
        _source("observed"),
        _relation("r", ("observed",)),
    )
    issues = validator.validate(
        components=components, audio_input_count=1, visible_context_count=0
    )
    assert REJECT_DUPLICATE_COMPONENT in issues


def test_validator_rejects_audio_arity_over_bound():
    validator = CompositeValidator()
    components = (_source("a"), _relation("r", ("a",)))
    issues = validator.validate(
        components=components, audio_input_count=3, visible_context_count=0
    )
    assert REJECT_INCOMPATIBLE_INPUT_SIGNATURE in issues


def test_validator_rejects_context_only_and_hidden_outputs():
    validator = CompositeValidator(
        context_only_fields=("age_class",),
        forbidden_output_fields=("predicted_original_text",),
    )
    issues = validator.validate(
        components=(_source("age_class"), _relation("r", ("age_class",))),
        audio_input_count=1,
        visible_context_count=0,
    )
    assert REJECT_CONTEXT_ONLY_TARGET in issues
    hidden = validator.validate(
        components=(
            _source("predicted_original_text"),
            _relation("r", ("predicted_original_text",)),
        ),
        audio_input_count=1,
        visible_context_count=0,
    )
    assert REJECT_HIDDEN_CONTEXT_REQUIRED in hidden


def test_validator_rejects_non_sufficient_and_nondeterministic():
    validator = CompositeValidator()
    insufficient = validator.validate(
        components=(_source("x", sufficient=False), _relation("r", ("x",))),
        audio_input_count=1,
        visible_context_count=0,
    )
    assert REJECT_NOT_INFORMATION_SUFFICIENT in insufficient
    nondeterministic = validator.validate(
        components=(_source("x", deterministic=False), _relation("r", ("x",))),
        audio_input_count=1,
        visible_context_count=0,
    )
    assert REJECT_NONDETERMINISTIC_GOLD in nondeterministic


def test_validator_rejects_cross_instance_and_composite_of_composite():
    validator = CompositeValidator()
    components = (_source("a"), _relation("r", ("a",)))
    cross = validator.validate(
        components=components,
        audio_input_count=1,
        visible_context_count=0,
        cross_instance_join=True,
    )
    assert REJECT_REQUIRES_CROSS_INSTANCE_JOIN in cross
    nested = validator.validate(
        components=components,
        audio_input_count=1,
        visible_context_count=0,
        composite_of_composite=True,
    )
    assert REJECT_COMPOSITE_OF_COMPOSITE in nested


def test_validator_chain_consistency_detects_disagreement():
    validator = CompositeValidator()
    assert (
        validator.validate_chain_consistency(
            consistency_contract="c1", checks={"matches": True}
        )
        == []
    )
    violations = validator.validate_chain_consistency(
        consistency_contract="c1", checks={"matches": False}
    )
    assert violations and "CHAIN_INCONSISTENT" in violations[0]


def test_dependency_graph_cycle_and_unknown_edges():
    acyclic = SemanticDependencyGraph.from_components(
        (_source("a"), _relation("b", ("a",)))
    )
    assert not acyclic.has_cycle()
    assert acyclic.dependencies_of("b") == ("a",)
    cyclic = SemanticDependencyGraph(nodes=("a", "b"), edges=(("a", "b"), ("b", "a")))
    assert cyclic.has_cycle()
    unknown = SemanticDependencyGraph(nodes=("a",), edges=(("x", "a"),))
    assert unknown.unknown_edges() == [("x", "a")]


def test_frontier_builder_accepts_chain_and_rejects_stacking():
    node = PrimitiveSemanticNode(
        type_id="ds_direct",
        operator="DIRECT",
        semantic_class="text_content",
        output_kind="field_value",
        audio_input_count=1,
        perception_outputs=(_source("observed_transcription"),),
    )
    rel = PrimitiveSemanticNode(
        type_id="ds_match",
        operator="TARGET_MATCH",
        semantic_class="text_content",
        output_kind="boolean",
        audio_input_count=1,
        logical_context_inputs=1,
        context_kind="candidate",
        perception_outputs=(_source("observed_transcription"),),
        relation_component=_relation("matches", ("observed_transcription",)),
    )
    rel2 = PrimitiveSemanticNode(
        type_id="ds_match_2",
        operator="TARGET_MATCH",
        semantic_class="text_content",
        output_kind="boolean",
        audio_input_count=1,
        logical_context_inputs=1,
        context_kind="reference",
        perception_outputs=(_source("observed_transcription"),),
        relation_component=_relation("matches_reference", ("observed_transcription",)),
    )
    builder = CompositeFrontierBuilder(
        (node, rel, rel2),
        composite_type_ids={
            "ds_match": "ds_composite_match",
            "ds_match_2": "ds_composite_ref",
        },
    )
    result = builder.build()
    accepted = {c.composite_type_id for c in result.accepted}
    assert accepted == {"ds_composite_match", "ds_composite_ref"}
    reasons = {c.reason for c in result.rejected}
    assert REJECT_MULTI_RELATION_STACKING in reasons


def test_candidate_ids_and_hash_are_deterministic():
    components = (
        _source("observed_transcription"),
        _relation("matches", ("observed_transcription",)),
    )
    first = canonical_hash([c.model_dump() for c in components])
    second = canonical_hash([c.model_dump() for c in components])
    assert first == second
    assert len(first) == 64

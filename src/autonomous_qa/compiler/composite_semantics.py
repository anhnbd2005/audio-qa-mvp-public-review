"""Generic bounded composite semantic expansion framework.

A composite semantic instance is a NEW semantic contract that combines one
already-frozen relational primitive with the perception output(s) its gold
depends on. Two conceptual classes are supported:

- ``CHAIN_DERIVED``: one output is a perception/intermediate result and
  another output is a deterministic relation derived from that result plus
  optional visible context. It intentionally tests perception + reasoning +
  self-consistency in one response.
- ``JOINT_INDEPENDENT``: multiple non-redundant source-supported targets on
  the same model-visible input.

This module is dataset-agnostic. Dataset adapters declare
:class:`PrimitiveSemanticNode` records (perception outputs, relation
component, input signature) and the composition logic here stays generic.
No dataset names are hardcoded.

The expansion is deliberately bounded: no powerset enumeration, no
composite-of-composite nesting, no cross-instance joins, no new sampling.
"""

from __future__ import annotations

import hashlib
import json
from itertools import permutations
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

# --------------------------------------------------------------------------
# Bounded expansion limits (Sections 8, 20, 85)
# --------------------------------------------------------------------------
MAX_COMPOSITE_OUTPUT_COMPONENTS = 3
MAX_AUDIO_INPUTS = 2
MAX_VISIBLE_TEXT_CONTEXTS = 1

ComponentRole = Literal["SOURCE_COMPONENT", "DERIVED_RELATION_COMPONENT"]
CompositeClassification = Literal["CHAIN_DERIVED", "JOINT_INDEPENDENT"]
CandidateStatus = Literal["ACCEPTED", "REJECTED"]

# Rejection vocabulary (Section 35). Stable machine codes, not prose.
REJECT_INCOMPATIBLE_INPUT_SIGNATURE = "INCOMPATIBLE_INPUT_SIGNATURE"
REJECT_REQUIRES_CROSS_INSTANCE_JOIN = "REQUIRES_CROSS_INSTANCE_JOIN"
REJECT_HIDDEN_CONTEXT_REQUIRED = "HIDDEN_CONTEXT_REQUIRED"
REJECT_CONTEXT_ONLY_TARGET = "CONTEXT_ONLY_TARGET"
REJECT_SEMANTIC_REDUNDANCY = "SEMANTIC_REDUNDANCY_WITHOUT_CHAIN_VALUE"
REJECT_EXCEEDS_MAX_COMPONENTS = "EXCEEDS_MAX_COMPONENTS"
REJECT_MULTI_RELATION_STACKING = "MULTI_RELATION_STACKING"
REJECT_NO_DISTINCT_EVALUATION_VALUE = "NO_DISTINCT_EVALUATION_VALUE"
REJECT_NOT_INFORMATION_SUFFICIENT = "NOT_INFORMATION_SUFFICIENT"
REJECT_NONDETERMINISTIC_GOLD = "NONDETERMINISTIC_GOLD"
REJECT_DUPLICATE_COMPONENT = "DUPLICATE_OUTPUT_COMPONENT"
REJECT_COMPOSITE_OF_COMPOSITE = "COMPOSITE_OF_COMPOSITE"
REJECT_REQUIRES_NEW_SAMPLING = "REQUIRES_NEW_SAMPLING"


def canonical_hash(value: Any) -> str:
    """Canonical JSON SHA256, matching the project hashing convention."""
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class CompositeComponentSpec(BaseModel):
    """One typed output component of a composite semantic contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    role: ComponentRole
    component_operator: str
    value_kind: str
    provenance: str
    derivation: str | None = None
    dependencies: tuple[str, ...] = ()
    information_sufficient: bool = True
    gold_deterministic: bool = True


class SemanticDependencyGraph(BaseModel):
    """Explicit dependency graph over composite output components."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_components(
        cls, components: tuple[CompositeComponentSpec, ...]
    ) -> SemanticDependencyGraph:
        names = tuple(c.name for c in components)
        edges: list[tuple[str, str]] = []
        for component in components:
            for dependency in component.dependencies:
                edges.append((dependency, component.name))
        return cls(nodes=names, edges=tuple(edges))

    def dependencies_of(self, name: str) -> tuple[str, ...]:
        return tuple(src for src, dst in self.edges if dst == name)

    def unknown_edges(self) -> list[tuple[str, str]]:
        known = set(self.nodes)
        return [
            (src, dst)
            for src, dst in self.edges
            if src not in known or dst not in known
        ]

    def has_cycle(self) -> bool:
        adjacency: dict[str, list[str]] = {name: [] for name in self.nodes}
        for src, dst in self.edges:
            if src in adjacency and dst in adjacency:
                adjacency[src].append(dst)
        state: dict[str, int] = {}

        def visit(node: str) -> bool:
            state[node] = 1
            for neighbour in adjacency.get(node, []):
                if state.get(neighbour) == 1:
                    return True
                if state.get(neighbour) != 2 and visit(neighbour):
                    return True
            state[node] = 2
            return False

        return any(state.get(node, 0) == 0 and visit(node) for node in self.nodes)


class CompositeSemanticSpec(BaseModel):
    """A new semantic contract built from one frozen relational primitive."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    composite_type_id: str
    classification: CompositeClassification
    base_primitive_type_id: str
    operator: str
    semantic_class: str
    answer_kind: str
    audio_input_count: int
    visible_context_kinds: tuple[str, ...] = ()
    components: tuple[CompositeComponentSpec, ...]
    chain_derived: bool
    evaluation_value: str

    @property
    def output_component_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.components)

    def dependency_graph(self) -> SemanticDependencyGraph:
        return SemanticDependencyGraph.from_components(self.components)


class PrimitiveSemanticNode(BaseModel):
    """Dataset-declared signature of one frozen primitive semantic type.

    ``perception_outputs`` are the source/observation outputs the relation
    depends on. ``relation_component`` is present only for relational
    primitives (TARGET_MATCH / EQUALITY / PAIRWISE_SELECTION style); a
    primitive without one cannot seed a chain-derived composite.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type_id: str
    operator: str
    semantic_class: str
    output_kind: str
    audio_input_count: int
    logical_context_inputs: int = 0
    perception_outputs: tuple[CompositeComponentSpec, ...] = ()
    relation_component: CompositeComponentSpec | None = None
    context_kind: str | None = None

    @property
    def is_relational(self) -> bool:
        return self.relation_component is not None


class CompositeCandidate(BaseModel):
    """One enumerated candidate (accepted or rejected) with its audit trail."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    composite_type_id: str | None
    base_primitive_type_id: str | None
    input_signature: dict[str, Any]
    visible_context: tuple[str, ...]
    output_components: tuple[str, ...]
    classification: str | None
    information_sufficient: bool
    gold_deterministic: bool
    requires_new_sampling: bool
    requires_cross_instance_join: bool
    semantic_value: str
    status: CandidateStatus
    reason: str


class CompositeValidator:
    """Generic structural / semantic validator for composite components."""

    def __init__(
        self,
        *,
        context_only_fields: tuple[str, ...] = (),
        forbidden_output_fields: tuple[str, ...] = (),
        max_components: int = MAX_COMPOSITE_OUTPUT_COMPONENTS,
        max_audio_inputs: int = MAX_AUDIO_INPUTS,
        max_visible_contexts: int = MAX_VISIBLE_TEXT_CONTEXTS,
    ) -> None:
        self.context_only_fields = tuple(context_only_fields)
        self.forbidden_output_fields = tuple(forbidden_output_fields)
        self.max_components = max_components
        self.max_audio_inputs = max_audio_inputs
        self.max_visible_contexts = max_visible_contexts

    def validate(
        self,
        *,
        components: tuple[CompositeComponentSpec, ...],
        audio_input_count: int,
        visible_context_count: int,
        cross_instance_join: bool = False,
        composite_of_composite: bool = False,
        requires_new_sampling: bool = False,
    ) -> list[str]:
        issues: list[str] = []
        names = [c.name for c in components]
        if len(components) < 2:
            issues.append(REJECT_NO_DISTINCT_EVALUATION_VALUE)
        if len(components) > self.max_components:
            issues.append(REJECT_EXCEEDS_MAX_COMPONENTS)
        if len(set(names)) != len(names):
            issues.append(REJECT_DUPLICATE_COMPONENT)
        if audio_input_count > self.max_audio_inputs:
            issues.append(REJECT_INCOMPATIBLE_INPUT_SIGNATURE)
        if visible_context_count > self.max_visible_contexts:
            issues.append(REJECT_INCOMPATIBLE_INPUT_SIGNATURE)
        if cross_instance_join:
            issues.append(REJECT_REQUIRES_CROSS_INSTANCE_JOIN)
        if composite_of_composite:
            issues.append(REJECT_COMPOSITE_OF_COMPOSITE)
        if requires_new_sampling:
            issues.append(REJECT_REQUIRES_NEW_SAMPLING)
        for component in components:
            if component.name in self.forbidden_output_fields:
                issues.append(REJECT_HIDDEN_CONTEXT_REQUIRED)
            if component.name in self.context_only_fields:
                issues.append(REJECT_CONTEXT_ONLY_TARGET)
            if not component.information_sufficient:
                issues.append(REJECT_NOT_INFORMATION_SUFFICIENT)
            if not component.gold_deterministic:
                issues.append(REJECT_NONDETERMINISTIC_GOLD)
            if (
                component.role == "DERIVED_RELATION_COMPONENT"
                and not component.dependencies
            ):
                issues.append(REJECT_NO_DISTINCT_EVALUATION_VALUE)
        graph = SemanticDependencyGraph.from_components(components)
        if graph.unknown_edges() or graph.has_cycle():
            issues.append(REJECT_INCOMPATIBLE_INPUT_SIGNATURE)
        # Deterministic component order: perception outputs first, relation last.
        derived_positions = [
            index
            for index, component in enumerate(components)
            if component.role == "DERIVED_RELATION_COMPONENT"
        ]
        if derived_positions and min(derived_positions) != len(components) - len(
            derived_positions
        ):
            issues.append(REJECT_INCOMPATIBLE_INPUT_SIGNATURE)
        # de-duplicate while preserving order
        return list(dict.fromkeys(issues))

    def validate_chain_consistency(
        self,
        *,
        consistency_contract: str,
        checks: dict[str, bool],
    ) -> list[str]:
        """Reject a candidate whose derived relation disagrees with gold.

        Adapters supply a mapping of ``relation name -> agreement flag``.
        Any ``False`` flag is a BLOCKING semantic inconsistency.
        """
        violations = [name for name, ok in checks.items() if not ok]
        if violations:
            return [
                f"CHAIN_INCONSISTENT:{consistency_contract}:{','.join(sorted(violations))}"
            ]
        return []


class FrontierResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[CompositeCandidate]
    accepted: list[CompositeCandidate]
    rejected: list[CompositeCandidate]


class CompositeFrontierBuilder:
    """Bounded first-order composite frontier enumeration.

    Algorithm (Section 12 / 20 / 35):

    1. For each relational primitive node, the candidate is exactly the
       node's perception dependencies plus its relation component. This is
       the only source of ACCEPTED chain-derived composites.
    2. For each relational node, adding an extra perception output is
       considered and rejected (duplicate component / hidden context /
       incompatible signature).
    3. Ordered pairs of relational nodes are considered and rejected as
       multi-relation stacking or cross-instance joins.
    4. Adapter-declared "considered" combinations (context-only targets,
       removed hidden targets, over-bound combinations, composite-of-
       composite) are appended with explicit rejection reasons.
    """

    def __init__(
        self,
        primitive_nodes: tuple[PrimitiveSemanticNode, ...],
        *,
        validator: CompositeValidator | None = None,
        considered_rejections: tuple[dict[str, Any], ...] = (),
        composite_type_ids: dict[str, str] | None = None,
    ) -> None:
        self.nodes = primitive_nodes
        self.validator = validator or CompositeValidator()
        self.considered_rejections = tuple(considered_rejections)
        self.composite_type_ids = dict(composite_type_ids or {})

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _candidate_id(
        components: tuple[CompositeComponentSpec, ...], base_type_id: str | None
    ) -> str:
        payload = {
            "base": base_type_id,
            "components": [
                {"name": c.name, "role": c.role, "operator": c.component_operator}
                for c in components
            ],
        }
        return "cc_" + canonical_hash(payload)[:16]

    def _make_candidate(
        self,
        *,
        base_type_id: str | None,
        composite_type_id: str | None,
        components: tuple[CompositeComponentSpec, ...],
        classification: str | None,
        audio_input_count: int,
        visible_context: tuple[str, ...],
        issues: list[str],
        reason: str | None = None,
        semantic_value: str = "",
        cross_instance_join: bool = False,
        requires_new_sampling: bool = False,
    ) -> CompositeCandidate:
        rejected = bool(issues)
        return CompositeCandidate(
            candidate_id=self._candidate_id(components, base_type_id),
            composite_type_id=composite_type_id,
            base_primitive_type_id=base_type_id,
            input_signature={
                "audio_input_count": audio_input_count,
                "visible_context": list(visible_context),
            },
            visible_context=visible_context,
            output_components=tuple(c.name for c in components),
            classification=classification,
            information_sufficient=all(c.information_sufficient for c in components),
            gold_deterministic=all(c.gold_deterministic for c in components),
            requires_new_sampling=requires_new_sampling,
            requires_cross_instance_join=cross_instance_join,
            semantic_value=semantic_value,
            status="REJECTED" if rejected else "ACCEPTED",
            reason=(reason or (issues[0] if issues else "ACCEPTED")),
        )

    # -- public API --------------------------------------------------------
    def build(self) -> FrontierResult:
        candidates: list[CompositeCandidate] = []

        relational = [node for node in self.nodes if node.is_relational]
        perception = [node for node in self.nodes if not node.is_relational]

        # 1. Chain-derived acceptance frontier.
        for node in relational:
            assert node.relation_component is not None
            components = tuple(node.perception_outputs) + (node.relation_component,)
            issues = self.validator.validate(
                components=components,
                audio_input_count=node.audio_input_count,
                visible_context_count=node.logical_context_inputs,
            )
            candidates.append(
                self._make_candidate(
                    base_type_id=node.type_id,
                    composite_type_id=self._composite_type_for(node),
                    components=components,
                    classification="CHAIN_DERIVED",
                    audio_input_count=node.audio_input_count,
                    visible_context=((node.context_kind,) if node.context_kind else ()),
                    issues=issues,
                    semantic_value=(
                        "perception + relation reasoning + internal consistency"
                    ),
                )
            )

        # 2. Adding an extra perception output to a relational node.
        for node in relational:
            assert node.relation_component is not None
            base_components = tuple(node.perception_outputs) + (
                node.relation_component,
            )
            base_names = {c.name for c in base_components}
            for extra_node in perception:
                for extra in extra_node.perception_outputs:
                    components = base_components + (extra,)
                    if extra.name in base_names:
                        issues = [REJECT_DUPLICATE_COMPONENT]
                    else:
                        issues = self.validator.validate(
                            components=components,
                            audio_input_count=node.audio_input_count,
                            visible_context_count=node.logical_context_inputs,
                        )
                        if not issues:
                            issues = [REJECT_SEMANTIC_REDUNDANCY]
                    candidates.append(
                        self._make_candidate(
                            base_type_id=node.type_id,
                            composite_type_id=None,
                            components=components,
                            classification="JOINT_INDEPENDENT",
                            audio_input_count=node.audio_input_count,
                            visible_context=(
                                (node.context_kind,) if node.context_kind else ()
                            ),
                            issues=issues,
                            semantic_value="extra perception output",
                        )
                    )

        # 3. Ordered pairs of relational nodes.
        for left, right in permutations(relational, 2):
            assert left.relation_component is not None
            assert right.relation_component is not None
            components = (
                tuple(left.perception_outputs)
                + (left.relation_component,)
                + (right.relation_component,)
            )
            shared_audio = left.audio_input_count >= 1 and right.audio_input_count >= 1
            issues = self.validator.validate(
                components=components,
                audio_input_count=max(left.audio_input_count, right.audio_input_count),
                visible_context_count=left.logical_context_inputs
                + right.logical_context_inputs,
                cross_instance_join=shared_audio,
            )
            if REJECT_EXCEEDS_MAX_COMPONENTS not in issues:
                issues.append(REJECT_MULTI_RELATION_STACKING)
            candidates.append(
                self._make_candidate(
                    base_type_id=left.type_id,
                    composite_type_id=None,
                    components=components,
                    classification="JOINT_INDEPENDENT",
                    audio_input_count=max(
                        left.audio_input_count, right.audio_input_count
                    ),
                    visible_context=(),
                    issues=issues,
                    reason=REJECT_MULTI_RELATION_STACKING,
                    semantic_value="two stacked relations",
                    cross_instance_join=shared_audio,
                )
            )

        # 4. Adapter-declared considered rejections (Section 35 exhaustiveness).
        for row in self.considered_rejections:
            components = tuple(row.get("components", ()))
            candidates.append(
                self._make_candidate(
                    base_type_id=row.get("base_primitive_type_id"),
                    composite_type_id=None,
                    components=components,
                    classification=row.get("classification"),
                    audio_input_count=row.get("audio_input_count", 1),
                    visible_context=tuple(row.get("visible_context", ())),
                    issues=[row["reason"]],
                    semantic_value=row.get("semantic_value", ""),
                    cross_instance_join=row.get("cross_instance_join", False),
                    requires_new_sampling=row.get("requires_new_sampling", False),
                )
            )

        accepted = [c for c in candidates if c.status == "ACCEPTED"]
        rejected = [c for c in candidates if c.status == "REJECTED"]
        return FrontierResult(
            candidates=candidates, accepted=accepted, rejected=rejected
        )

    def _composite_type_for(self, node: PrimitiveSemanticNode) -> str:
        return self.composite_type_ids.get(node.type_id, f"composite::{node.type_id}")

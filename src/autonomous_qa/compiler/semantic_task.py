"""Generic semantic task contract and the canonical semantic catalog loader.

The final semantic type set is a STATIC canonical resource under
``resources/semantics/``. Runtime code must never resolve semantic types from
historical generated discovery outputs.

One contract represents both primitive (one output component) and composite
(several output components) tasks; ``classification`` is audit metadata only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from src.common.answer_schema import (
    AtomicAnswerSchema,
    OutputComponent,
    StructuredAnswerSchema,
)
from src.common.config import ROOT
from src.autonomous_qa.language.template_renderer import canonical_hash

SEMANTIC_CATALOG_PATH = (
    ROOT / "resources" / "semantics" / "vietmdd_semantic_catalog.json"
)

Classification = Literal["PRIMITIVE_RELATION", "CHAIN_DERIVED"]
TaskKind = Literal["ATOMIC", "STRUCTURED"]

COMPOSITE_OPERATOR = "COMPOSITE"


class SemanticTaskSpec(BaseModel):
    """One final semantic type (atomic or structured)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type_id: str
    proposition_id: str
    proposition_description: str
    operator: str
    classification: Classification
    kind: TaskKind
    audio_arity: int
    visible_context_roles: tuple[str, ...] = ()
    outputs: tuple[OutputComponent, ...]
    dependency_graph: tuple[tuple[str, str], ...] = ()
    comparator_id: str | None = None
    invariances: tuple[str, ...] = ()
    non_invariances: tuple[str, ...] = ()
    base_type_id: str | None = None
    source_role_mapping: dict[str, Any] = {}
    evaluation_value: str = ""
    # Architecture metadata (training-data compiler). These are NOT part of the
    # T1-T3 semantic identity: they are excluded from logical_hash so adding
    # them does not change an existing frozen plan/release identity.
    tier: str | None = None
    gold_origin: str | None = None
    guardrail_status: str | None = None
    closure_hash: str | None = None

    @property
    def is_composite(self) -> bool:
        return self.kind == "STRUCTURED"

    @property
    def output_roles(self) -> tuple[str, ...]:
        return tuple(c.role for c in self.outputs)

    def answer_schema(self) -> AtomicAnswerSchema | StructuredAnswerSchema:
        if self.kind == "ATOMIC":
            return AtomicAnswerSchema(kind=self.outputs[0].kind)
        return StructuredAnswerSchema(components=self.outputs)

    def capability_key(self) -> dict[str, Any]:
        """Dataset-agnostic language compatibility key."""
        return {
            "operator": self.operator,
            "classification": self.classification,
            "audio_arity": self.audio_arity,
            "visible_context": list(self.visible_context_roles),
            "outputs": [{"role": c.role, "kind": c.kind} for c in self.outputs],
            "answer_schema": "STRUCTURED" if self.kind == "STRUCTURED" else "ATOMIC",
        }

    def capability_signature(self) -> str:
        return canonical_hash(self.capability_key())


class SemanticCatalog(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: str
    catalog_version: str
    comparators_resource: str
    tasks: tuple[SemanticTaskSpec, ...]

    def by_type_id(self, type_id: str) -> SemanticTaskSpec:
        for task in self.tasks:
            if task.type_id == type_id:
                return task
        raise KeyError(f"UNKNOWN_SEMANTIC_TYPE:{type_id}")

    # Architecture-only metadata excluded from T1-T3 semantic identity.
    IDENTITY_EXCLUDED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"tier", "gold_origin", "guardrail_status", "closure_hash"}
    )

    def logical_hash(self) -> str:
        tasks = []
        for task in self.tasks:
            payload = task.model_dump(mode="json")
            for field in self.IDENTITY_EXCLUDED_FIELDS:
                payload.pop(field, None)
            tasks.append(payload)
        return canonical_hash(
            {
                "dataset": self.dataset,
                "catalog_version": self.catalog_version,
                "comparators_resource": self.comparators_resource,
                "tasks": tasks,
            }
        )


def load_semantic_catalog(path: Path = SEMANTIC_CATALOG_PATH) -> SemanticCatalog:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return SemanticCatalog.model_validate(raw)


def load_dataset_semantic_catalog(dataset_id: str) -> SemanticCatalog:
    """Resolve a dataset's catalog from the canonical resource registry."""
    from src.autonomous_qa.compiler.canonical_resources import get_semantic_catalog_path

    return load_semantic_catalog(get_semantic_catalog_path(dataset_id))


def final_semantic_catalog() -> tuple[SemanticTaskSpec, ...]:
    return load_semantic_catalog().tasks


def catalog_by_type_id() -> dict[str, SemanticTaskSpec]:
    return {spec.type_id: spec for spec in final_semantic_catalog()}


def final_semantic_catalog_hash() -> str:
    return load_semantic_catalog().logical_hash()

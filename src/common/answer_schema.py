"""Generic atomic / structured answer-schema contract.

Primitive tasks have one atomic output component; composite tasks have an
ordered list of typed components. The contract is dataset-agnostic and gives
the renderer, gold audit and language layer one validated representation.

Structured answers are depth-1 only: a structured component may not itself be
structured.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

AtomicKind = Literal["field_value", "boolean", "audio_index"]
ATOMIC_KINDS: tuple[str, ...] = ("field_value", "boolean", "audio_index")

# Canonical model-facing serialization for atomic answers.
BOOLEAN_MODEL_FACING = {"true": True, "false": False}
AUDIO_INDEX_MODEL_FACING = ("0", "1")

MAX_STRUCTURED_COMPONENTS = 3


class AtomicAnswerSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: AtomicKind


class OutputComponent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str
    kind: AtomicKind
    dependencies: tuple[str, ...] = ()


class StructuredAnswerSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    components: tuple[OutputComponent, ...]
    kind: Literal["structured"] = "structured"

    @model_validator(mode="after")
    def validate_structure(self) -> StructuredAnswerSchema:
        if not self.components:
            raise ValueError("EMPTY_STRUCTURED_SCHEMA")
        if len(self.components) > MAX_STRUCTURED_COMPONENTS:
            raise ValueError("EXCEEDS_MAX_STRUCTURED_COMPONENTS")
        roles = [c.role for c in self.components]
        if len(set(roles)) != len(roles):
            raise ValueError("DUPLICATE_STRUCTURED_ROLE")
        for component in self.components:
            for dependency in component.dependencies:
                if dependency not in roles:
                    raise ValueError(f"UNKNOWN_COMPONENT_DEPENDENCY:{dependency}")
        return self

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(c.role for c in self.components)

    def kind_for(self, role: str) -> str:
        for component in self.components:
            if component.role == role:
                return component.kind
        raise KeyError(role)


class AnswerSchemaError(ValueError):
    pass


def validate_structured_gold(
    schema: StructuredAnswerSchema, gold: dict[str, Any]
) -> list[str]:
    """Return validation errors for a typed structured gold object."""
    errors: list[str] = []
    expected_roles = list(schema.roles)
    actual_roles = list(gold.keys())
    if actual_roles != expected_roles:
        errors.append(f"COMPONENT_ORDER_OR_SET:{actual_roles}!={expected_roles}")
    for component in schema.components:
        if component.role not in gold:
            errors.append(f"MISSING_COMPONENT:{component.role}")
            continue
        value = gold[component.role]
        if component.kind == "boolean" and not isinstance(value, bool):
            errors.append(f"INVALID_BOOLEAN:{component.role}")
        elif component.kind == "audio_index" and value not in (0, 1):
            errors.append(f"INVALID_AUDIO_INDEX:{component.role}")
        elif component.kind == "field_value" and not isinstance(value, str):
            errors.append(f"INVALID_FIELD_VALUE:{component.role}")
    return errors


def serialize_atomic_answer(kind: str, value: Any) -> str:
    """Canonical model-facing atomic answer serialization."""
    if kind == "boolean":
        return "true" if bool(value) else "false"
    return str(value)


def serialize_structured_answer(
    schema: StructuredAnswerSchema, gold: dict[str, Any]
) -> str:
    """Canonical deterministic, parseable, human-readable structured answer.

    Format: ``role=value`` pairs joined by `` | `` in schema order, with each
    value routed through the canonical atomic serializer. Role display labels
    are supplied by the language layer; this keeps the wire format stable and
    machine-parseable.
    """
    parts = []
    for component in schema.components:
        parts.append(
            f"{component.role}={serialize_atomic_answer(component.kind, gold[component.role])}"
        )
    return " | ".join(parts)


def parse_structured_answer(
    text: str, schema: StructuredAnswerSchema
) -> dict[str, str]:
    """Inverse of :func:`serialize_structured_answer` (validation helper)."""
    pieces = [p.strip() for p in text.split("|")]
    result: dict[str, str] = {}
    for piece in pieces:
        if "=" not in piece:
            raise AnswerSchemaError(f"MALFORMED_STRUCTURED_ANSWER:{piece}")
        role, value = piece.split("=", 1)
        result[role.strip()] = value.strip()
    if list(result.keys()) != list(schema.roles):
        raise AnswerSchemaError("STRUCTURED_ANSWER_ROLE_MISMATCH")
    return result

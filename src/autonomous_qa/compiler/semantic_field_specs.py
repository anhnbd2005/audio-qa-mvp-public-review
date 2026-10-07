"""Semantic field specifications for deterministic global template rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SemanticClass = Literal[
    "categorical_attribute",
    "numeric_attribute",
    "text_content",
    "ordinal_attribute",
    "boolean_attribute",
    "multi_label_attribute",
    "identifier_relation",
]


class ValuePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    normalization: str = "identity"
    match_policy: Literal[
        "exact", "bucket", "tolerance", "set_exact", "set_overlap"
    ] = "exact"
    comparison_policy: str | None = None
    bins: list[float] | None = None
    tolerance: float | None = None

    @model_validator(mode="after")
    def policy_parameters_are_explicit(self) -> ValuePolicy:
        if self.match_policy == "bucket" and not self.bins:
            raise ValueError("BUCKET_POLICY_REQUIRES_BINS")
        if self.match_policy == "tolerance" and self.tolerance is None:
            raise ValueError("TOLERANCE_POLICY_REQUIRES_TOLERANCE")
        return self


class RenderingPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_quote_style: Literal["plain", "vietnamese_quotes"] = "plain"
    allow_unit_omission: bool = False


class SemanticFieldSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field_name: str
    semantic_class: SemanticClass
    entity_scope: str
    entity_phrase: str
    attribute_phrase: str | None = None
    content_phrase: str | None = None
    value_phrase: str | None = None
    unit: str | None = None
    value_policy: ValuePolicy = Field(default_factory=ValuePolicy)
    rendering: RenderingPolicy = Field(default_factory=RenderingPolicy)
    semantic_status: str = "SUPPORTED"
    source: str = "profile_native"

    @model_validator(mode="after")
    def required_verbalization_exists(self) -> SemanticFieldSpec:
        if not self.entity_phrase.strip():
            raise ValueError("ENTITY_PHRASE_MISSING")
        if self.semantic_class in {
            "categorical_attribute",
            "numeric_attribute",
            "ordinal_attribute",
            "boolean_attribute",
            "multi_label_attribute",
        } and not (self.attribute_phrase and self.attribute_phrase.strip()):
            raise ValueError("ATTRIBUTE_PHRASE_MISSING")
        if self.semantic_class == "text_content" and not (
            self.content_phrase and self.content_phrase.strip()
        ):
            raise ValueError("CONTENT_PHRASE_MISSING")
        return self

    def slot_values(self) -> dict[str, str | None]:
        return {
            "[ENTITY_PHRASE]": self.entity_phrase,
            "[ATTRIBUTE_PHRASE]": self.attribute_phrase,
            "[CONTENT_PHRASE]": self.content_phrase,
            "[VALUE_PHRASE]": self.value_phrase,
            "[UNIT]": self.unit,
        }


def field_spec_from_profile(
    field_name: str, profile_field: Any
) -> SemanticFieldSpec | None:
    """Resolve only profile-native hints; never humanize a raw field name."""
    semantic_class = getattr(profile_field, "semantic_class", None)
    verbalization = getattr(profile_field, "verbalization", None)
    if not semantic_class or verbalization is None:
        return None
    policy = getattr(profile_field, "value_policy", None)
    payload = (
        verbalization.model_dump()
        if hasattr(verbalization, "model_dump")
        else dict(verbalization)
    )
    return SemanticFieldSpec(
        field_name=field_name,
        semantic_class=semantic_class,
        entity_scope=getattr(profile_field, "entity_scope", None) or "unknown",
        **payload,
        value_policy=(policy.model_dump() if hasattr(policy, "model_dump") else policy)
        or {},
        source="profile_native",
    )


def load_migration_sidecar(path: Path) -> dict[str, SemanticFieldSpec]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("field_specs", data)
    specs = [SemanticFieldSpec.model_validate(item) for item in rows]
    return {spec.field_name: spec for spec in specs}


def resolve_field_spec(
    field_name: str,
    profile_field: Any | None,
    migration_specs: dict[str, SemanticFieldSpec] | None = None,
) -> SemanticFieldSpec:
    native = (
        field_spec_from_profile(field_name, profile_field)
        if profile_field is not None
        else None
    )
    if native is not None:
        return native
    if migration_specs and field_name in migration_specs:
        return migration_specs[field_name]
    raise ValueError(f"FIELD_SPEC_MISSING:{field_name}")

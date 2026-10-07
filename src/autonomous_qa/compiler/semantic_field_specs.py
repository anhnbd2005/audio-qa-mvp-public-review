"""Semantic field specifications for deterministic global template rendering.

Phase 4.2: field specs are resolved DECLARATIVELY. A DatasetProfile's
per-field optional semantics (semantic_class / verbalization / value_policy /
entity_scope) are the primary source; an explicit migration sidecar supplies
semantics only when the profile predates them. Raw field names NEVER determine
semantic class. No dataset-ID branch exists in this module.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

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


def _field_attr(profile_field: Any, name: str, default: Any = None) -> Any:
    if isinstance(profile_field, Mapping):
        return profile_field.get(name, default)
    return getattr(profile_field, name, default)


def field_spec_from_profile(
    field_name: str, profile_field: Any
) -> SemanticFieldSpec | None:
    """Resolve only profile-native hints; never humanize a raw field name.

    Accepts either a validated ``ProfileField`` object or a raw mapping (the
    authoring pipeline persists its profile as JSON). A field with no explicit
    semantic_class or verbalization yields ``None`` (migration required).
    """
    semantic_class = _field_attr(profile_field, "semantic_class")
    verbalization = _field_attr(profile_field, "verbalization")
    if not semantic_class or verbalization is None:
        return None
    policy = _field_attr(profile_field, "value_policy")
    payload = (
        verbalization.model_dump()
        if hasattr(verbalization, "model_dump")
        else dict(verbalization)
    )
    return SemanticFieldSpec(
        field_name=field_name,
        semantic_class=semantic_class,
        entity_scope=_field_attr(profile_field, "entity_scope") or "unknown",
        **payload,
        value_policy=(policy.model_dump() if hasattr(policy, "model_dump") else policy)
        or {},
        source="profile_native",
    )


def load_migration_sidecar(
    path: Path, *, expected_dataset_id: str | None = None
) -> dict[str, SemanticFieldSpec]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if expected_dataset_id is not None and isinstance(data, Mapping):
        declared = data.get("dataset_id")
        if declared is not None and declared != expected_dataset_id:
            raise ValueError(
                f"FIELD_SPEC_DATASET_MISMATCH:{declared}!={expected_dataset_id}"
            )
    rows = data.get("field_specs", data)
    specs = [SemanticFieldSpec.model_validate(item) for item in rows]
    return {spec.field_name: spec for spec in specs}


class SemanticFieldSpecBundle(BaseModel):
    """Deterministic, hashable identity for a compiled field-spec map."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    dataset_id: str
    source: str
    field_specs: dict[str, SemanticFieldSpec] = Field(default_factory=dict)

    def logical_hash(self) -> str:
        return _canonical_hash(self.model_dump(mode="json"))


def _profile_fields(profile: Any) -> dict[str, Any]:
    if isinstance(profile, Mapping):
        fields = profile.get("fields")
    else:
        fields = getattr(profile, "fields", None)
    if not isinstance(fields, Mapping):
        raise ValueError("PROFILE_FIELDS_MISSING")
    return dict(fields)


def _profile_dataset_id(profile: Any) -> str:
    if isinstance(profile, Mapping):
        return str(profile.get("dataset") or profile.get("dataset_id") or "")
    return str(getattr(profile, "dataset", "") or "")


def compile_semantic_field_specs(
    profile: Any,
    *,
    required_fields: set[str] | None = None,
    migration_specs: dict[str, SemanticFieldSpec] | None = None,
    dataset_id: str | None = None,
) -> SemanticFieldSpecBundle:
    """Compile a declarative ``dict[str, SemanticFieldSpec]`` from a profile.

    Resolution order per field:
      1. profile-native semantic metadata (explicit semantic_class +
         verbalization) wins;
      2. an explicitly supplied migration sidecar;
      3. otherwise, if the field is required -> ``FIELD_SPEC_MISSING:<field>``.

    Raw field names are NEVER inspected to infer semantic class or wording.
    """
    fields = _profile_fields(profile)
    resolved: dict[str, SemanticFieldSpec] = {}
    sources: set[str] = set()

    for name in sorted(fields):
        native = field_spec_from_profile(name, fields[name])
        if native is not None:
            resolved[name] = native
            sources.add("profile_native")
            continue
        if migration_specs and name in migration_specs:
            spec = migration_specs[name]
            if spec.field_name != name:
                raise ValueError(f"MIGRATION_SIDECAR_FIELD_MISMATCH:{name}")
            resolved[name] = spec
            sources.add("migration_sidecar")
            continue
        if required_fields is not None and name in required_fields:
            raise ValueError(f"FIELD_SPEC_MISSING:{name}")

    if required_fields is not None:
        for name in sorted(required_fields):
            if name not in resolved:
                raise ValueError(f"FIELD_SPEC_MISSING:{name}")

    source = (
        "mixed"
        if len(sources) > 1
        else (next(iter(sources)) if sources else "empty")
    )
    return SemanticFieldSpecBundle(
        schema_version=1,
        dataset_id=dataset_id or _profile_dataset_id(profile),
        source=source,
        field_specs=resolved,
    )


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

"""Zero-LLM global Operator × SemanticClass blueprint renderer."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.autonomous_qa.compiler.semantic_field_specs import SemanticClass, SemanticFieldSpec
from src.autonomous_qa.language.template_contracts import QuestionTemplateSpec, TypeContract

OperatorId = Literal[
    "DIRECT", "EQUALITY", "PAIRWISE_SELECTION", "TARGET_MATCH", "COMPOSITE"
]
SLOT_RE = re.compile(r"\[[A-Z][A-Z0-9_]*\]")
ALLOWED_SLOTS = frozenset(
    {
        "[ENTITY_PHRASE]",
        "[ATTRIBUTE_PHRASE]",
        "[CONTENT_PHRASE]",
        "[VALUE_PHRASE]",
        "[TARGET_VALUE]",
        "[UNIT]",
        "[AUDIO_REFERENCE]",
        "[INSTRUCTION_1]",
        "[INSTRUCTION_2]",
        "[INSTRUCTION_3]",
    }
)


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class TemplateBlueprint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    blueprint_id: str
    language: str = "vi"
    operator: OperatorId
    semantic_class: SemanticClass
    pattern: str
    required_slots: list[str]
    optional_slots: list[str] = Field(default_factory=list)
    answer_kind: str
    version: str = "canonical"
    unit_policy: Literal["any", "required", "forbidden"] = "any"
    match_policies: list[str] = Field(default_factory=lambda: ["exact"])

    @model_validator(mode="after")
    def slots_are_valid(self) -> TemplateBlueprint:
        declared = set(self.required_slots) | set(self.optional_slots)
        used = set(SLOT_RE.findall(self.pattern))
        if not declared.issubset(ALLOWED_SLOTS) or not used.issubset(ALLOWED_SLOTS):
            raise ValueError("UNKNOWN_BLUEPRINT_SLOT")
        if not set(self.required_slots).issubset(used):
            raise ValueError("REQUIRED_SLOT_NOT_IN_PATTERN")
        if not used.issubset(declared):
            raise ValueError("UNDECLARED_BLUEPRINT_SLOT")
        return self


class TemplateLibrary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    language: str
    version: str
    schema_version: int
    created_from: str
    library_hash: str
    blueprints: list[TemplateBlueprint]

    @model_validator(mode="after")
    def unique_blueprint_ids(self) -> TemplateLibrary:
        ids = [b.blueprint_id for b in self.blueprints]
        if len(ids) != len(set(ids)):
            raise ValueError("DUPLICATE_BLUEPRINT_ID")
        return self

    def computed_hash(self) -> str:
        payload = self.model_dump()
        payload.pop("library_hash", None)
        return canonical_hash(payload)

    def lookup(self, operator: str, semantic_class: str) -> list[TemplateBlueprint]:
        return sorted(
            [
                item
                for item in self.blueprints
                if item.operator == operator and item.semantic_class == semantic_class
            ],
            key=lambda item: item.blueprint_id,
        )


class RenderedTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rendered_template_id: str
    blueprint_id: str
    blueprint_hash: str
    type_id: str
    field_name: str
    operator: OperatorId
    semantic_class: SemanticClass
    question_pattern: str
    target_binding_mode: Literal["NONE", "PER_INSTANCE"]
    answer_kind: str
    answer_mode: str
    library_version: str
    semantic_status: str
    template_status: Literal["PASS", "REVIEW", "REJECTED"] = "PASS"
    preview_capacity_status: Literal[
        "SUFFICIENT",
        "INSUFFICIENT_POSITIVE_CAPACITY",
        "INSUFFICIENT_NEGATIVE_CAPACITY",
        "PAIR_DATA_REQUIRED",
        "NOT_APPLICABLE",
        "UNKNOWN",
    ] = "UNKNOWN"
    full_train_capacity_status: Literal[
        "SUFFICIENT",
        "INSUFFICIENT_POSITIVE_CAPACITY",
        "INSUFFICIENT_NEGATIVE_CAPACITY",
        "PAIR_DATA_REQUIRED",
        "NOT_AUDITED",
        "NOT_APPLICABLE",
        "UNKNOWN",
    ] = "NOT_AUDITED"
    generation_status: Literal[
        "PRODUCTION_ELIGIBLE",
        "WAITING_FULL_TRAIN_CAPACITY_AUDIT",
        "DEFERRED",
        "DEFERRED_DATA_CAPACITY",
        "BLOCKED",
    ] = "WAITING_FULL_TRAIN_CAPACITY_AUDIT"
    validation_reasons: list[str] = Field(default_factory=list)

    def as_legacy_template(self) -> QuestionTemplateSpec:
        target = "[TARGET_VALUE]" in self.question_pattern
        return QuestionTemplateSpec(
            template_id=self.rendered_template_id,
            type_id=self.type_id,
            operator=self.operator,
            question_text=self.question_pattern,
            language="vi",
            answer_mode=self.answer_mode,
            audio_reference_style="attached_audio_order",
            required_placeholders=["[TARGET_VALUE]"] if target else [],
            optional_placeholders=[],
            forbidden_placeholders=[] if target else ["[TARGET_VALUE]"],
            allowed_answer_forms=[],
            notes=f"Rendered from global blueprint {self.blueprint_id}",
        )


def load_library(path: Path) -> TemplateLibrary:
    library = TemplateLibrary.model_validate_json(path.read_text(encoding="utf-8"))
    if library.library_hash != library.computed_hash():
        raise ValueError("TEMPLATE_LIBRARY_HASH_MISMATCH")
    return library


def _normalize_rendered_pattern(text: str) -> str:
    text = " ".join(text.split())
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    if text:
        text = text[0].upper() + text[1:]
    return text


def render_blueprint(
    blueprint: TemplateBlueprint,
    field_spec: SemanticFieldSpec,
    type_contract: TypeContract,
    library_version: str = "canonical",
) -> RenderedTemplate:
    reasons: list[str] = []
    if blueprint.operator != type_contract.operator:
        reasons.append("OPERATOR_MISMATCH")
    if blueprint.semantic_class != field_spec.semantic_class:
        reasons.append("SEMANTIC_CLASS_MISMATCH")
    if type_contract.semantic_field != field_spec.field_name:
        reasons.append("FIELD_BINDING_MISMATCH")
    if blueprint.answer_kind != type_contract.answer_kind:
        reasons.append("ANSWER_KIND_MISMATCH")
    if (
        type_contract.source_status != "SUPPORTED"
        or field_spec.semantic_status != "SUPPORTED"
    ):
        reasons.append("SEMANTIC_FIELD_NOT_SUPPORTED")
    if field_spec.value_policy.match_policy not in blueprint.match_policies:
        reasons.append("MATCH_POLICY_UNSUPPORTED")
    if blueprint.unit_policy == "required" and not field_spec.unit:
        reasons.append("UNIT_REQUIRED")
    if blueprint.unit_policy == "forbidden" and field_spec.unit:
        reasons.append("UNIT_MUST_BE_ABSENT")
    if reasons:
        raise ValueError(";".join(reasons))
    rendered = blueprint.pattern
    slots = field_spec.slot_values()
    for slot in blueprint.required_slots:
        if slot == "[TARGET_VALUE]":
            continue
        if not slots.get(slot):
            raise ValueError(f"REQUIRED_SLOT_MISSING:{slot}")
    for slot, value in slots.items():
        if value is not None:
            rendered = rendered.replace(slot, value)
    unresolved = set(SLOT_RE.findall(rendered)) - {"[TARGET_VALUE]"}
    if unresolved:
        raise ValueError(f"UNRESOLVED_SLOTS:{sorted(unresolved)}")
    rendered = _normalize_rendered_pattern(rendered)
    suffix = hashlib.sha256(
        f"{type_contract.type_id}|{blueprint.blueprint_id}".encode()
    ).hexdigest()[:10]
    return RenderedTemplate(
        rendered_template_id=f"rt_{suffix}",
        blueprint_id=blueprint.blueprint_id,
        blueprint_hash=canonical_hash(blueprint.model_dump()),
        type_id=type_contract.type_id,
        field_name=field_spec.field_name,
        operator=type_contract.operator,
        semantic_class=field_spec.semantic_class,
        question_pattern=rendered,
        target_binding_mode="PER_INSTANCE" if "[TARGET_VALUE]" in rendered else "NONE",
        answer_kind=type_contract.answer_kind,
        answer_mode=type_contract.answer_mode,
        library_version=library_version,
        semantic_status=type_contract.source_status,
    )


def render_type_contract(
    library: TemplateLibrary,
    field_spec: SemanticFieldSpec,
    type_contract: TypeContract,
) -> list[RenderedTemplate]:
    matches = library.lookup(type_contract.operator, field_spec.semantic_class)
    if not matches:
        raise ValueError(
            f"BLUEPRINT_MISSING:{type_contract.operator}:{field_spec.semantic_class}"
        )
    rendered = []
    incompatibilities = []
    for blueprint in matches:
        try:
            rendered.append(
                render_blueprint(blueprint, field_spec, type_contract, library.version)
            )
        except ValueError as exc:
            incompatibilities.append(str(exc))
    if not rendered:
        raise ValueError(
            f"BLUEPRINT_INCOMPATIBLE:{type_contract.type_id}:{incompatibilities}"
        )
    return rendered

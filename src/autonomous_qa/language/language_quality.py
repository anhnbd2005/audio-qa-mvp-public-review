"""Canonical language-quality contracts and production language registry."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.common.config import ROOT, resolve_llm_base_url, resolve_llm_model
from src.autonomous_qa.language.paraphrase_engine import (
    ParaphraseBlueprint,
    ParaphraseLibrary,
    normalized_lexical,
    semantic_contract_hash,
)
from src.autonomous_qa.language.template_engine import (
    hash_tree,
    sha256_file,
    vimd_field_specs,
)
from src.autonomous_qa.authoring.llm_client import GenerateContentConfig, ThinkingConfig, create_llm_client
from src.autonomous_qa.compiler.semantic_field_specs import SemanticFieldSpec
from src.autonomous_qa.language.template_contracts import (
    TypeContract,
    build_type_contracts,
    operator_contracts,
    write_json,
)
from src.autonomous_qa.language.template_renderer import (
    RenderedTemplate,
    TemplateBlueprint,
    TemplateLibrary,
    canonical_hash,
    load_library,
    render_blueprint,
)

OperatorId = Literal[
    "DIRECT", "EQUALITY", "PAIRWISE_SELECTION", "TARGET_MATCH", "COMPOSITE"
]
QualityDecision = Literal["PASS", "REVIEW", "REJECT"]
DimensionDecision = Literal["PASS", "REVIEW", "FAIL"]
ProductionStatus = Literal["PRODUCTION_PASS", "PRODUCTION_REVIEW", "PRODUCTION_REJECT"]

CANONICAL_INTERNAL_HASH = (
    "02365605fc88a63be423d9a4701320905a26030a2408dd88f45b34fb8bdfeaa9"
)
PARAPHRASE_INTERNAL_HASH = (
    "c68393212dda8c4a76fb2d58540fa46b2f5a34bf952c1ad7f758b39055257070"
)
PARAPHRASE_FILE_SHA256 = (
    "2d339ea32bcd047c3cd7b24a6afd454634641985588292d914fd341b40a4d443"
)
OPERATORS: tuple[OperatorId, ...] = (
    "DIRECT",
    "EQUALITY",
    "PAIRWISE_SELECTION",
    "TARGET_MATCH",
    "COMPOSITE",
)


class LanguageQualityPair(BaseModel):
    model_config = ConfigDict(extra="forbid")
    canonical_id: str
    paraphrase_id: str
    operator: OperatorId
    semantic_class: str
    canonical_pattern: str
    paraphrase_pattern: str
    answer_kind: str
    required_audio_input_count: int
    logical_context_roles: list[str]
    required_slots: list[str]
    optional_slots: list[str] = Field(default_factory=list)
    match_policies: list[str]
    semantic_contract_hash: str


class LanguageQualityDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    canonical_id: str
    paraphrase_id: str
    decision: QualityDecision
    naturalness: DimensionDecision
    semantic_faithfulness: DimensionDecision
    ambiguity: DimensionDecision
    slot_bindability: DimensionDecision
    operator_preservation: Literal["PASS", "FAIL"]
    match_policy_preservation: Literal["PASS", "FAIL"]
    reason_codes: list[str] = Field(default_factory=list)
    short_reason: str

    @model_validator(mode="after")
    def internally_consistent(self) -> LanguageQualityDecision:
        dimensions = (
            self.naturalness,
            self.semantic_faithfulness,
            self.ambiguity,
            self.slot_bindability,
        )
        hard_fail = (
            "FAIL" in dimensions
            or self.operator_preservation == "FAIL"
            or self.match_policy_preservation == "FAIL"
        )
        review = "REVIEW" in dimensions
        expected = "REJECT" if hard_fail else "REVIEW" if review else "PASS"
        if self.decision != expected:
            raise ValueError(f"INCONSISTENT_QUALITY_DECISION:expected={expected}")
        return self


class OperatorLanguageQualityOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operator: OperatorId
    evaluations: list[LanguageQualityDecision]


class LanguageRegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    language_entry_id: str
    source_kind: Literal["CANONICAL", "PARAPHRASE", "CANDIDATE"]
    source_id: str
    canonical_blueprint_id: str
    operator: OperatorId
    semantic_class: str
    pattern: str
    answer_kind: str
    required_slots: list[str]
    optional_slots: list[str] = Field(default_factory=list)
    unit_policy: str
    match_policy: list[str]
    semantic_contract_hash: str
    quality_status: ProductionStatus
    quality_reason_codes: list[str] = Field(default_factory=list)
    enabled: bool
    template_library_hash: str
    paraphrase_library_hash: str | None = None
    registry_version: str
    # Structured (COMPOSITE) capability metadata. Defaults keep every
    # primitive entry valid without migration.
    output_signature: list[dict[str, Any]] = Field(default_factory=list)
    instruction_bindings: dict[str, str] = Field(default_factory=dict)
    context_roles: list[str] = Field(default_factory=list)
    audio_reference_phrase: str | None = None
    proposition_id: str | None = None
    # Phase 4.2 entity-composition capability metadata. Optional so every
    # historical entry keeps loading unchanged; None means "classify
    # structurally, never silently wildcard".
    entity_scopes: list[str] | None = None
    entity_reference_owner: Literal["slot", "literal", "none"] | None = None


class ProductionLanguageRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    language: str
    version: str
    schema_version: int
    template_library_version: str
    template_library_hash: str
    paraphrase_library_version: str
    paraphrase_library_hash: str
    quality_model: str
    quality_call_ids: list[str]
    registry_hash: str
    entries: list[LanguageRegistryEntry]

    @model_validator(mode="after")
    def unique_ids(self) -> ProductionLanguageRegistry:
        ids = [item.language_entry_id for item in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("DUPLICATE_LANGUAGE_ENTRY_ID")
        if any(
            item.enabled != (item.quality_status == "PRODUCTION_PASS")
            for item in self.entries
        ):
            raise ValueError("LANGUAGE_ENTRY_ENABLEMENT_MISMATCH")
        return self

    def computed_hash(self) -> str:
        payload = self.model_dump()
        payload.pop("registry_hash", None)
        # Phase 4.2 capability fields are identity-bearing ONLY when declared.
        # Popping them when None keeps every historical registry hash stable.
        for entry in payload.get("entries", []):
            for key in ("entity_scopes", "entity_reference_owner"):
                if entry.get(key) is None:
                    entry.pop(key, None)
        return canonical_hash(payload)

    def active(self, operator: str, semantic_class: str) -> list[LanguageRegistryEntry]:
        return sorted(
            [
                item
                for item in self.entries
                if item.enabled
                and item.operator == operator
                and item.semantic_class == semantic_class
            ],
            key=lambda item: item.language_entry_id,
        )


class SamplingPolicyConfig(BaseModel):
    """Bounded sampling behaviour of the zero-LLM production generator."""

    model_config = ConfigDict(extra="forbid")
    value_sampling: Literal["uniform_over_rows", "uniform_over_values"] = (
        "uniform_over_values"
    )
    max_source_row_reuse: int | None = Field(default=None, ge=1)
    max_source_row_reuse_per_type: int | None = Field(default=None, ge=1)
    prefer_distinct_speaker: bool = True
    hidden_identifier_field: str | None = "speakerID"
    max_sampling_attempts: int = Field(default=64, ge=1)


class BooleanConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    positive_ratio: float | None = Field(default=None, ge=0, le=1)


class SelectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    randomized_position: bool = True


class TextNegativeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    length_bucket_match: bool = True


class LanguageConfig(BaseModel):
    """One approved wording per semantic instance; v1 forbids multiplication."""

    model_config = ConfigDict(extra="forbid")
    variants_per_semantic_instance: int = Field(default=1, ge=1)
    selection: Literal["deterministic_sha256"] = "deterministic_sha256"

    @model_validator(mode="after")
    def v1_is_single_variant(self) -> LanguageConfig:
        if self.variants_per_semantic_instance != 1:
            raise ValueError("UNSUPPORTED_VARIANTS_PER_SEMANTIC_INSTANCE")
        return self


class AudioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    materialize: bool = False
    local_root: str | None = None


class ProductionGenerationConfig(BaseModel):
    """Schema only; release-specific budgets remain intentionally unset."""

    model_config = ConfigDict(extra="forbid")
    seed: int | None = None
    total_target_qa: int | None = Field(default=None, ge=1)
    per_type_budget: dict[str, int] | None = None
    per_language_pattern_budget: dict[str, int] | None = None
    max_audio_reuse: int | None = Field(default=None, ge=1)
    max_source_row_reuse: int | None = Field(default=None, ge=1)
    target_value_frequency_cap: int | None = Field(default=None, ge=1)
    positive_negative_ratio: float | None = Field(default=None, ge=0, le=1)
    selection_a_b_ratio: float | None = Field(default=None, ge=0, le=1)
    dedup_policy: str | None = None
    language_preflight_artifact: str | None = None
    strict_budget: bool = True
    allocation_policy: Literal["equal_by_type"] = "equal_by_type"
    sampling: SamplingPolicyConfig = Field(default_factory=SamplingPolicyConfig)
    boolean: BooleanConfig = Field(default_factory=BooleanConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    text_negative: TextNegativeConfig = Field(default_factory=TextNegativeConfig)
    language: LanguageConfig = Field(default_factory=LanguageConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)

    @model_validator(mode="after")
    def budget_policy_is_supported(self) -> ProductionGenerationConfig:
        if self.per_language_pattern_budget:
            raise ValueError("UNSUPPORTED_LANGUAGE_PATTERN_BUDGET")
        return self


def strict_parse(text: str) -> OperatorLanguageQualityOutput:
    value = text.strip()
    if value.startswith("```json") and value.endswith("```"):
        value = value[7:-3].strip()
    elif value.startswith("```") and value.endswith("```"):
        value = value[3:-3].strip()
    return OperatorLanguageQualityOutput.model_validate(json.loads(value))


def load_paraphrase_library(path: Path) -> ParaphraseLibrary:
    library = ParaphraseLibrary.model_validate_json(path.read_text(encoding="utf-8"))
    if library.library_hash != library.computed_hash():
        raise ValueError("PARAPHRASE_LIBRARY_HASH_MISMATCH")
    return library


def deterministic_language_reasons(
    source: TemplateBlueprint, para: ParaphraseBlueprint
) -> list[str]:
    """Conservative structural/policy gate before subjective LLM judgment."""
    text = para.pattern.casefold()
    reasons: list[str] = []
    if (
        source.semantic_class == "text_content"
        and "exact" in source.match_policies
        and any(term in text for term in ("chứa", "từ khóa", "nhắc đến"))
    ):
        reasons.append("TEXT_EXACT_CONTAINMENT_DRIFT")
    if (
        source.semantic_class == "numeric_attribute"
        and "exact" in source.match_policies
        and any(
            term in text for term in ("khoảng", "xấp xỉ", "tầm", "gần", "ước chừng")
        )
    ):
        reasons.append("NUMERIC_EXACT_POLICY_DRIFT")
    if source.operator == "PAIRWISE_SELECTION" and "exact" in source.match_policies:
        if any(term in text for term in ("gần nhất", "giống hơn", "phù hợp hơn")):
            reasons.append("SELECTION_SIMILARITY_DRIFT")
        if not (re.search(r"\ba\b", text) and re.search(r"\bb\b", text)):
            reasons.append("SELECTION_AUDIO_ARITY_DRIFT")
        if "nào" not in text and "chọn" not in text:
            reasons.append("SELECTION_ANSWER_KIND_DRIFT")
    if source.operator == "EQUALITY":
        if "đoạn nào" in text or "hãy chọn" in text:
            reasons.append("EQUALITY_TO_SELECTION_DRIFT")
        if "hai đoạn" not in text:
            reasons.append("EQUALITY_AUDIO_ARITY_DRIFT")
    if source.operator == "TARGET_MATCH":
        if "[target_value]" not in text:
            reasons.append("TARGET_VISIBILITY_DRIFT")
        if not any(term in text for term in ("không?", "hay không", "đúng không")):
            reasons.append("TARGET_MATCH_BOOLEAN_DRIFT")
    if source.operator == "DIRECT" and (
        "[target_value]" in text
        or any(term in text for term in ("có phải", "đúng không"))
    ):
        reasons.append("DIRECT_TO_TARGET_MATCH_DRIFT")
    return reasons


def build_quality_pairs(
    canonical: TemplateLibrary, paraphrases: ParaphraseLibrary
) -> tuple[list[LanguageQualityPair], dict[str, Any]]:
    canonical_by_id = {item.blueprint_id: item for item in canonical.blueprints}
    pairs: list[LanguageQualityPair] = []
    rejected: list[dict[str, Any]] = []
    seen_patterns: set[tuple[str, str, str]] = set()
    for para in paraphrases.paraphrases:
        source = canonical_by_id.get(para.source_blueprint_id)
        reasons: list[str] = []
        if source is None:
            reasons.append("UNKNOWN_CANONICAL_ID")
        else:
            if para.operator != source.operator:
                reasons.append("OPERATOR_MISMATCH")
            if para.semantic_class != source.semantic_class:
                reasons.append("SEMANTIC_CLASS_MISMATCH")
            if para.answer_kind != source.answer_kind:
                reasons.append("ANSWER_KIND_MISMATCH")
            if set(para.required_slots) != set(source.required_slots):
                reasons.append("REQUIRED_SLOT_MISMATCH")
            if set(para.optional_slots) != set(source.optional_slots):
                reasons.append("OPTIONAL_SLOT_MISMATCH")
            if para.semantic_contract_hash != semantic_contract_hash(source):
                reasons.append("SEMANTIC_CONTRACT_HASH_MISMATCH")
            op = operator_contracts()[source.operator]
            if op.audio_input_count not in {1, 2}:
                reasons.append("AUDIO_ARITY_INVALID")
            key = (para.operator, para.semantic_class, normalized_lexical(para.pattern))
            if key in seen_patterns:
                reasons.append("DUPLICATE_PARAPHRASE")
            seen_patterns.add(key)
            if normalized_lexical(para.pattern) == normalized_lexical(source.pattern):
                reasons.append("DUPLICATE_CANONICAL")
            if para.paraphrase_status != "PASS":
                reasons.append("PARAPHRASE_NOT_DETERMINISTIC_PASS")
            reasons.extend(deterministic_language_reasons(source, para))
        if reasons:
            rejected.append(
                {"paraphrase_id": para.paraphrase_id, "reason_codes": reasons}
            )
            continue
        assert source is not None
        op = operator_contracts()[source.operator]
        pairs.append(
            LanguageQualityPair(
                canonical_id=source.blueprint_id,
                paraphrase_id=para.paraphrase_id,
                operator=source.operator,
                semantic_class=source.semantic_class,
                canonical_pattern=source.pattern,
                paraphrase_pattern=para.pattern,
                answer_kind=source.answer_kind,
                required_audio_input_count=op.audio_input_count,
                logical_context_roles=op.context_roles,
                required_slots=sorted(source.required_slots),
                optional_slots=sorted(source.optional_slots),
                match_policies=source.match_policies,
                semantic_contract_hash=semantic_contract_hash(source),
            )
        )
    return pairs, {
        "input_paraphrases": len(paraphrases.paraphrases),
        "passed": len(pairs),
        "rejected": len(rejected),
        "rejections": rejected,
        "semantic_types_created": 0,
        "field_eligibility_changes": 0,
    }


def build_prompt(operator: str, pairs: list[LanguageQualityPair]) -> str:
    schema = json.dumps(
        OperatorLanguageQualityOutput.model_json_schema(), ensure_ascii=False, indent=2
    )
    records = json.dumps(
        [item.model_dump() for item in pairs], ensure_ascii=False, indent=2
    )
    return f"""You are performing GLOBAL LANGUAGE QUALITY judgment on reusable Vietnamese
blueprints. This is not semantic type discovery and not QA-instance generation.

Evaluate every canonical/paraphrase pair exactly once. Return judgments only.
Never rewrite, repair, or propose replacement wording. Do not add any pattern field.
The decision applies to the paraphrase; a rejected paraphrase does not invalidate its
structurally valid canonical source.

Judge natural Vietnamese, clarity, semantic faithfulness, ambiguity, generic slot
bindability, operator preservation, and exact match-policy preservation.

Operator rules:
- DIRECT: one audio -> field/content value; no target, yes/no, or pairwise drift.
- EQUALITY: two audios -> Boolean same/different; no A/B selection.
- TARGET_MATCH: one visible target + one audio -> Boolean exact decision.
- PAIRWISE_SELECTION: one visible target + audio A + audio B -> A/B exact selection.
- Whole-text exact matching must not become substring or keyword containment.
- Numeric exact matching must not introduce khoảng, xấp xỉ, tầm, gần, or ước chừng.
- Categorical exact matching must not become comparative similarity or ranking.

Use PASS only when all dimensions pass. Any REVIEW dimension requires decision REVIEW.
Any FAIL dimension, operator FAIL, or match-policy FAIL requires decision REJECT.
Copy canonical_id and paraphrase_id exactly. operator must be {operator}.
No Markdown and no text outside JSON.

GLOBAL PAIRS:
{records}

JSON SCHEMA (authoritative final block):
{schema}"""


def mock_output(
    operator: str, pairs: list[LanguageQualityPair]
) -> OperatorLanguageQualityOutput:
    evaluations = []
    for index, pair in enumerate(pairs):
        if index == 1:
            evaluations.append(
                LanguageQualityDecision(
                    canonical_id=pair.canonical_id,
                    paraphrase_id=pair.paraphrase_id,
                    decision="REVIEW",
                    naturalness="REVIEW",
                    semantic_faithfulness="PASS",
                    ambiguity="PASS",
                    slot_bindability="PASS",
                    operator_preservation="PASS",
                    match_policy_preservation="PASS",
                    reason_codes=["MOCK_NATURALNESS_REVIEW"],
                    short_reason="Mock review case.",
                )
            )
        elif index == 2:
            evaluations.append(
                LanguageQualityDecision(
                    canonical_id=pair.canonical_id,
                    paraphrase_id=pair.paraphrase_id,
                    decision="REJECT",
                    naturalness="PASS",
                    semantic_faithfulness="FAIL",
                    ambiguity="PASS",
                    slot_bindability="PASS",
                    operator_preservation="FAIL",
                    match_policy_preservation="PASS",
                    reason_codes=["MOCK_OPERATOR_DRIFT"],
                    short_reason="Mock rejection case.",
                )
            )
        else:
            evaluations.append(
                LanguageQualityDecision(
                    canonical_id=pair.canonical_id,
                    paraphrase_id=pair.paraphrase_id,
                    decision="PASS",
                    naturalness="PASS",
                    semantic_faithfulness="PASS",
                    ambiguity="PASS",
                    slot_bindability="PASS",
                    operator_preservation="PASS",
                    match_policy_preservation="PASS",
                    reason_codes=[],
                    short_reason="Mock pass case.",
                )
            )
    return OperatorLanguageQualityOutput(operator=operator, evaluations=evaluations)


def validate_accounting(
    output: OperatorLanguageQualityOutput,
    operator: str,
    pairs: list[LanguageQualityPair],
) -> dict[str, Any]:
    expected = {(item.canonical_id, item.paraphrase_id) for item in pairs}
    returned = [(item.canonical_id, item.paraphrase_id) for item in output.evaluations]
    counts = Counter(returned)
    actual = set(returned)
    audit = {
        "operator": operator,
        "expected": [list(item) for item in sorted(expected)],
        "returned": [list(item) for item in sorted(actual)],
        "missing": [list(item) for item in sorted(expected - actual)],
        "unknown": [list(item) for item in sorted(actual - expected)],
        "duplicates": [
            list(item) for item, count in sorted(counts.items()) if count > 1
        ],
        "empty_nonempty_input": bool(pairs and not output.evaluations),
        "operator_match": output.operator == operator,
    }
    audit["complete"] = (
        not audit["missing"]
        and not audit["unknown"]
        and not audit["duplicates"]
        and not audit["empty_nonempty_input"]
        and audit["operator_match"]
    )
    if not audit["complete"]:
        raise ValueError(f"LANGUAGE_QUALITY_ACCOUNTING_FAILURE:{audit}")
    return audit


def _production_status(decision: str) -> ProductionStatus:
    return {
        "PASS": "PRODUCTION_PASS",
        "REVIEW": "PRODUCTION_REVIEW",
        "REJECT": "PRODUCTION_REJECT",
    }[decision]  # type: ignore[return-value]


def build_registry(
    canonical: TemplateLibrary,
    paraphrases: ParaphraseLibrary,
    decisions: list[LanguageQualityDecision],
    quality_model: str,
    call_ids: list[str],
) -> tuple[ProductionLanguageRegistry, list[dict[str, Any]]]:
    para_by_id = {item.paraphrase_id: item for item in paraphrases.paraphrases}
    entries: list[LanguageRegistryEntry] = []
    for source in canonical.blueprints:
        entries.append(
            LanguageRegistryEntry(
                language_entry_id=f"lang_can_{source.blueprint_id}",
                source_kind="CANONICAL",
                source_id=source.blueprint_id,
                canonical_blueprint_id=source.blueprint_id,
                operator=source.operator,
                semantic_class=source.semantic_class,
                pattern=source.pattern,
                answer_kind=source.answer_kind,
                required_slots=source.required_slots,
                optional_slots=source.optional_slots,
                unit_policy=source.unit_policy,
                match_policy=source.match_policies,
                semantic_contract_hash=semantic_contract_hash(source),
                quality_status="PRODUCTION_PASS",
                quality_reason_codes=[],
                enabled=True,
                template_library_hash=canonical.library_hash,
                registry_version="canonical",
            )
        )
    rejected_audit = []
    for decision in decisions:
        para = para_by_id[decision.paraphrase_id]
        status = _production_status(decision.decision)
        if status == "PRODUCTION_REJECT":
            rejected_audit.append(
                {
                    "source_id": para.paraphrase_id,
                    "canonical_blueprint_id": para.source_blueprint_id,
                    "quality_status": status,
                    "quality_reason_codes": decision.reason_codes,
                    "short_reason": decision.short_reason,
                }
            )
            continue
        source = next(
            item
            for item in canonical.blueprints
            if item.blueprint_id == para.source_blueprint_id
        )
        entries.append(
            LanguageRegistryEntry(
                language_entry_id=f"lang_para_{para.paraphrase_id}",
                source_kind="PARAPHRASE",
                source_id=para.paraphrase_id,
                canonical_blueprint_id=para.source_blueprint_id,
                operator=para.operator,
                semantic_class=para.semantic_class,
                pattern=para.pattern,
                answer_kind=para.answer_kind,
                required_slots=para.required_slots,
                optional_slots=para.optional_slots,
                unit_policy=source.unit_policy,
                match_policy=source.match_policies,
                semantic_contract_hash=para.semantic_contract_hash,
                quality_status=status,
                quality_reason_codes=decision.reason_codes,
                enabled=status == "PRODUCTION_PASS",
                template_library_hash=canonical.library_hash,
                paraphrase_library_hash=paraphrases.library_hash,
                registry_version="canonical",
            )
        )
    registry = ProductionLanguageRegistry(
        language="vi",
        version="canonical",
        schema_version=1,
        template_library_version=canonical.version,
        template_library_hash=canonical.library_hash,
        paraphrase_library_version=paraphrases.version,
        paraphrase_library_hash=paraphrases.library_hash,
        quality_model=quality_model,
        quality_call_ids=call_ids,
        registry_hash="PENDING",
        entries=entries,
    )
    registry.registry_hash = registry.computed_hash()
    return registry, rejected_audit


def load_language_registry(path: Path) -> ProductionLanguageRegistry:
    registry = ProductionLanguageRegistry.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if registry.registry_hash != registry.computed_hash():
        raise ValueError("LANGUAGE_REGISTRY_HASH_MISMATCH")
    return registry


def approved_patterns_for_contract(
    registry: ProductionLanguageRegistry,
    field_spec: SemanticFieldSpec,
    contract: TypeContract,
) -> list[LanguageRegistryEntry]:
    entries = registry.active(contract.operator, field_spec.semantic_class)
    if not entries:
        raise ValueError(
            f"LANGUAGE_LIBRARY_COVERAGE_MISSING:{contract.operator}:{field_spec.semantic_class}"
        )
    return [
        item
        for item in entries
        if field_spec.value_policy.match_policy in item.match_policy
        and item.answer_kind == contract.answer_kind
    ]


def render_approved_patterns(
    registry: ProductionLanguageRegistry,
    field_spec: SemanticFieldSpec,
    contract: TypeContract,
) -> list[tuple[str, RenderedTemplate]]:
    rendered = []
    for entry in approved_patterns_for_contract(registry, field_spec, contract):
        blueprint = TemplateBlueprint(
            blueprint_id=entry.language_entry_id,
            operator=entry.operator,
            semantic_class=entry.semantic_class,
            pattern=entry.pattern,
            required_slots=entry.required_slots,
            optional_slots=entry.optional_slots,
            answer_kind=entry.answer_kind,
            version=registry.version,
            unit_policy=entry.unit_policy,
            match_policies=entry.match_policy,
        )
        try:
            rendered.append(
                (
                    entry.language_entry_id,
                    render_blueprint(blueprint, field_spec, contract, registry.version),
                )
            )
        except ValueError:
            continue
    if not rendered:
        raise ValueError(
            f"LANGUAGE_LIBRARY_COVERAGE_MISSING:{contract.operator}:{field_spec.semantic_class}"
        )
    return rendered


def semantic_instance_identity(
    *, type_id: str, operator: str, audio_ids: list[str], target: Any, gold: Any
) -> str:
    """Identity of QA semantics; intentionally excludes wording realization."""
    return canonical_hash(
        {
            "type_id": type_id,
            "operator": operator,
            "audio_ids": audio_ids,
            "target": target,
            "gold": gold,
        }
    )


def language_realization_identity(
    semantic_instance_id: str, language_entry_id: str
) -> str:
    """Identity of one approved wording applied to a semantic instance."""
    return canonical_hash(
        {
            "semantic_instance_id": semantic_instance_id,
            "language_entry_id": language_entry_id,
        }
    )


def check_runtime_coverage(
    registry: ProductionLanguageRegistry,
    contracts: list[TypeContract],
    specs: dict[str, SemanticFieldSpec],
) -> dict[str, Any]:
    rows = []
    for contract in contracts:
        field_spec = specs[contract.semantic_field]
        matches = approved_patterns_for_contract(registry, field_spec, contract)
        rows.append(
            {
                "type_id": contract.type_id,
                "operator": contract.operator,
                "semantic_class": field_spec.semantic_class,
                "active_language_entries": len(matches),
                "covered": bool(matches),
            }
        )
    return {
        "supported_semantic_types": len(contracts),
        "covered": sum(item["covered"] for item in rows),
        "missing": [item["type_id"] for item in rows if not item["covered"]],
        "types": rows,
        "llm_calls": 0,
        "qa_instantiated": 0,
    }


def future_coverage_demos(registry: ProductionLanguageRegistry) -> list[dict[str, Any]]:
    demos = (
        ("gender", "categorical_attribute"),
        ("emotion", "categorical_attribute"),
        ("language", "categorical_attribute"),
        ("age", "numeric_attribute"),
    )
    return [
        {
            "field": field,
            "operator": "DIRECT",
            "semantic_class": semantic_class,
            "active_language_entries": len(registry.active("DIRECT", semantic_class)),
            "language_coverage": bool(registry.active("DIRECT", semantic_class)),
            "qa_eligibility_claimed": False,
            "llm_calls": 0,
        }
        for field, semantic_class in demos
    ]


# ---------------------------------------------------------------------------
# Phase 4.2: entity-composition capability model.
#
# Compatibility is structural: a semantic role has exactly ONE presentation
# owner. An entry either owns the entity reference through the [ENTITY_PHRASE]
# slot (owner="slot"), owns it literally in the pattern (owner="literal"), or
# does not reference a single entity (owner="none"). Binding a phrase that the
# pattern literal already contains would give the same semantic head two
# owners, so such an entry is filtered BEFORE rendering.
# ---------------------------------------------------------------------------

_ENTITY_SLOT_RE = re.compile(r"\[[A-Z][A-Z0-9_]*\]")

# Lexicon is used ONLY to classify a legacy entry's owner; it never grants a
# wildcard capability and never appears in a generic language entry.
ENTITY_HEAD_LEXICON = (
    "đoạn âm thanh",
    "file âm thanh",
    "người nói",
    "cuộc hội thoại",
    "bản ghi",
)

# Canonical entity heads per scope. A slot-owned entry whose literal already
# names the field's entity head gives that role two owners.
SCOPE_ENTITY_HEADS: dict[str, tuple[str, ...]] = {
    "utterance": ("đoạn âm thanh", "file âm thanh", "âm thanh"),
    "speaker": ("người nói",),
    "recording": ("bản ghi", "file âm thanh"),
    "conversation": ("cuộc hội thoại",),
}


class LanguageCapabilityUnresolved(RuntimeError):
    """A language entry's entity-composition capability cannot be classified."""


class EntryCapability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_reference_owner: Literal["slot", "literal", "none"]
    entity_scopes: tuple[str, ...] = ()
    literal_entity_heads: tuple[str, ...] = ()


def pattern_literal(pattern: str) -> str:
    return _ENTITY_SLOT_RE.sub(" ", pattern)


def resolve_entry_capability(entry: LanguageRegistryEntry) -> EntryCapability:
    """Deterministic classification of an entry's entity-composition capability.

    Declared metadata wins. Historical entries (no declared metadata) are
    classified from STRUCTURAL pattern features only. An internally
    inconsistent declaration fails closed with LANGUAGE_ENTRY_CAPABILITY_UNRESOLVED.
    """
    has_entity_slot = "[ENTITY_PHRASE]" in entry.pattern
    literal = pattern_literal(entry.pattern).casefold()
    heads = tuple(h for h in ENTITY_HEAD_LEXICON if h.casefold() in literal)

    owner = entry.entity_reference_owner
    if owner is None:
        if has_entity_slot:
            owner = "slot"
        elif heads:
            owner = "literal"
        else:
            owner = "none"

    if owner == "slot" and not has_entity_slot:
        raise LanguageCapabilityUnresolved(
            f"LANGUAGE_ENTRY_CAPABILITY_UNRESOLVED:{entry.language_entry_id}:"
            "owner_slot_but_no_entity_slot"
        )
    if owner == "literal" and has_entity_slot:
        raise LanguageCapabilityUnresolved(
            f"LANGUAGE_ENTRY_CAPABILITY_UNRESOLVED:{entry.language_entry_id}:"
            "owner_literal_but_entity_slot_present"
        )

    if entry.entity_scopes is not None and not entry.entity_scopes:
        raise LanguageCapabilityUnresolved(
            f"LANGUAGE_ENTRY_CAPABILITY_UNRESOLVED:{entry.language_entry_id}:"
            "empty_entity_scopes"
        )

    return EntryCapability(
        entity_reference_owner=owner,
        entity_scopes=tuple(entry.entity_scopes) if entry.entity_scopes else (),
        literal_entity_heads=heads,
    )


def slot_value_map(bindings: dict[str, Any]) -> dict[str, str]:
    """Map phrase-bindings to their slot tokens for ownership checking."""
    mapping: dict[str, str] = {}
    for binding_key, slot in (
        ("entity_phrase", "[ENTITY_PHRASE]"),
        ("attribute_phrase", "[ATTRIBUTE_PHRASE]"),
        ("content_phrase", "[CONTENT_PHRASE]"),
        ("value_phrase", "[VALUE_PHRASE]"),
        ("unit", "[UNIT]"),
    ):
        value = bindings.get(binding_key)
        if value:
            mapping[slot] = str(value)
    return mapping


def pattern_ownership_conflicts(
    pattern: str, slot_values: dict[str, str]
) -> list[tuple[str, str]]:
    """Slots whose bound phrase the pattern literal already contains.

    Only slots actually present in the pattern are considered: a literal-owned
    entry deliberately omits [ENTITY_PHRASE].
    """
    literal = pattern_literal(pattern).casefold()
    conflicts: list[tuple[str, str]] = []
    for slot, phrase in slot_values.items():
        if slot == "[UNIT]":
            # Unit is a measurement label, not an entity/attribute head; a
            # pattern may legitimately name the label "đơn vị" and bind [UNIT].
            continue
        if not phrase or slot not in pattern:
            continue
        if phrase.casefold() in literal:
            conflicts.append((slot, phrase))
    return conflicts


def entry_capability_compatible(
    entry: LanguageRegistryEntry,
    *,
    entity_scope: str | None,
    unit: str | None,
    slot_values: dict[str, str],
) -> tuple[bool, str | None]:
    """Fail-closed compatibility of an entry with a field's composition needs."""
    caps = resolve_entry_capability(entry)

    if caps.entity_scopes:
        if not entity_scope or (
            entity_scope not in caps.entity_scopes
            and "*" not in caps.entity_scopes
        ):
            return False, "ENTITY_SCOPE_INCOMPATIBLE"

    if unit is None and entry.unit_policy == "required":
        return False, "UNIT_REQUIRED_BUT_ABSENT"
    if unit is not None and entry.unit_policy == "forbidden":
        return False, "UNIT_FORBIDDEN_BUT_PRESENT"

    if pattern_ownership_conflicts(entry.pattern, slot_values):
        return False, "SEMANTIC_OWNERSHIP_CONFLICT"

    # A slot-owned entry already binding [ENTITY_PHRASE] must not also name the
    # field's entity head literally, or the entity role would have two owners.
    if caps.entity_reference_owner == "slot" and entity_scope:
        literal = pattern_literal(entry.pattern).casefold()
        if any(
            head.casefold() in literal
            for head in SCOPE_ENTITY_HEADS.get(entity_scope, ())
        ):
            return False, "SEMANTIC_OWNERSHIP_CONFLICT"

    return True, None









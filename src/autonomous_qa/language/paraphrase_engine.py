"""Canonical paraphrase models, validation, and rendering helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.common.config import ROOT, resolve_llm_base_url, resolve_llm_model
from src.autonomous_qa.language.template_engine import (
    sha256_file,
    vimd_field_specs,
)
from src.autonomous_qa.authoring.llm_client import GenerateContentConfig, ThinkingConfig, create_llm_client
from src.autonomous_qa.language.template_contracts import (
    QuestionTemplateSpec,
    TypeContract,
    build_type_contracts,
    instantiate_type,
    load_train_rows,
    model_facing,
    normalize_value,
    operator_contracts,
    validate_instance,
    write_json,
    write_jsonl,
)
from src.autonomous_qa.language.template_renderer import (
    ALLOWED_SLOTS,
    SLOT_RE,
    TemplateBlueprint,
    canonical_hash,
    load_library,
    render_blueprint,
)

OperatorId = Literal["DIRECT", "EQUALITY", "PAIRWISE_SELECTION", "TARGET_MATCH"]


class ParaphraseCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_blueprint_id: str
    operator: OperatorId
    semantic_class: str
    paraphrase_pattern: str
    required_slots: list[str]
    optional_slots: list[str] = Field(default_factory=list)
    answer_kind: str
    notes: str | None = None


class OperatorParaphraseOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operator: OperatorId
    paraphrases: list[ParaphraseCandidate]


class GenerationProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    llm_call_id: str
    prompt_hash: str


class ParaphraseBlueprint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paraphrase_id: str
    source_blueprint_id: str
    language: str = "vi"
    operator: OperatorId
    semantic_class: str
    pattern: str
    required_slots: list[str]
    optional_slots: list[str] = Field(default_factory=list)
    answer_kind: str
    source_library_version: str
    semantic_contract_hash: str
    generation_provenance: GenerationProvenance
    paraphrase_status: Literal["PASS", "REVIEW", "REJECTED"]
    reason_codes: list[str] = Field(default_factory=list)

    def as_template_blueprint(self) -> TemplateBlueprint:
        return TemplateBlueprint(
            blueprint_id=self.paraphrase_id,
            language=self.language,
            operator=self.operator,
            semantic_class=self.semantic_class,
            pattern=self.pattern,
            required_slots=self.required_slots,
            optional_slots=self.optional_slots,
            answer_kind=self.answer_kind,
            version="canonical",
        )


class ParaphraseLibrary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    language: str
    version: str
    schema_version: int
    source_template_library: str
    source_template_library_sha256: str
    generation_model: str
    generation_endpoint_class: str
    created_from_calls: list[str]
    library_hash: str
    paraphrases: list[ParaphraseBlueprint]

    def computed_hash(self) -> str:
        payload = self.model_dump()
        payload.pop("library_hash", None)
        return canonical_hash(payload)


def semantic_contract(blueprint: TemplateBlueprint) -> dict[str, Any]:
    op = operator_contracts()[blueprint.operator]
    return {
        "operator": blueprint.operator,
        "semantic_class": blueprint.semantic_class,
        "answer_kind": blueprint.answer_kind,
        "audio_input_count": op.audio_input_count,
        "context_roles": op.context_roles,
        "match_policies": blueprint.match_policies,
        "required_slots": sorted(blueprint.required_slots),
        "optional_slots": sorted(blueprint.optional_slots),
        "unit_policy": blueprint.unit_policy,
    }


def semantic_contract_hash(blueprint: TemplateBlueprint) -> str:
    return canonical_hash(semantic_contract(blueprint))


def strict_parse(text: str) -> OperatorParaphraseOutput:
    value = text.strip()
    if value.startswith("```json") and value.endswith("```"):
        value = value[7:-3].strip()
    elif value.startswith("```") and value.endswith("```"):
        value = value[3:-3].strip()
    return OperatorParaphraseOutput.model_validate(json.loads(value))


def normalized_lexical(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    value = re.sub(r"[^\w\[\]]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def build_prompt(operator: str, sources: list[TemplateBlueprint]) -> str:
    records = [
        {
            "blueprint_id": item.blueprint_id,
            "operator": item.operator,
            "semantic_class": item.semantic_class,
            "pattern": item.pattern,
            "required_slots": item.required_slots,
            "optional_slots": item.optional_slots,
            "answer_kind": item.answer_kind,
            "match_policies": item.match_policies,
            "semantic_contract_hash": semantic_contract_hash(item),
        }
        for item in sources
    ]
    schema = json.dumps(
        OperatorParaphraseOutput.model_json_schema(), ensure_ascii=False, indent=2
    )
    return f"""You are NOT generating dataset questions.
You are rewriting reusable Vietnamese language blueprints.
Preserve every semantic placeholder exactly.
Do not insert concrete values. Do not add or remove logical inputs.
Do not change answer type. Do not change exact matching into approximate or containment semantics.

Return exactly one paraphrase for every source blueprint. Copy source_blueprint_id,
operator, semantic_class, required_slots, optional_slots, and answer_kind exactly.
Change surface wording meaningfully, while retaining natural Vietnamese grammar.
No Markdown and no text outside JSON.

Operator contract:
{json.dumps(operator_contracts()[operator].model_dump(), ensure_ascii=False, indent=2)}

Semantic preservation rules:
- DIRECT remains one-audio field/content retrieval and has no target.
- EQUALITY remains a Boolean same-value relation over two audio examples.
- TARGET_MATCH retains one visible [TARGET_VALUE], one audio, and a Boolean decision.
- PAIRWISE_SELECTION retains [TARGET_VALUE], symmetric A/B candidates, and an A/B answer.
- Exact numeric wording must not introduce khoảng/xấp xỉ/gần/tầm/ước chừng.
- Exact text wording must mean equality of the whole normalized content, not substring containment.
- Exact categorical selection must not become similarity or ranking.

Canonical global sources:
{json.dumps(records, ensure_ascii=False, indent=2)}

JSON SCHEMA (authoritative final block):
{schema}"""


def mock_output(
    operator: str, sources: list[TemplateBlueprint]
) -> OperatorParaphraseOutput:
    def safe_pattern(item: TemplateBlueprint) -> str:
        pattern = item.pattern
        if item.semantic_class == "text_content" and "exact" in item.match_policies:
            pattern = pattern.replace(
                "chứa nội dung này", "nói đúng nội dung mục tiêu này"
            )
            pattern = pattern.replace("chứa", "nói đúng")
        return f"Vui lòng trả lời câu hỏi sau: {pattern}"

    return OperatorParaphraseOutput(
        operator=operator,
        paraphrases=[
            ParaphraseCandidate(
                source_blueprint_id=item.blueprint_id,
                operator=item.operator,
                semantic_class=item.semantic_class,
                paraphrase_pattern=safe_pattern(item),
                required_slots=item.required_slots,
                optional_slots=item.optional_slots,
                answer_kind=item.answer_kind,
                notes="deterministic mock",
            )
            for item in sources
        ],
    )


def validate_output(
    output: OperatorParaphraseOutput,
    sources: list[TemplateBlueprint],
    guard: dict[str, Any],
    call_id: str,
    prompt_hash: str,
) -> tuple[list[ParaphraseBlueprint], dict[str, Any]]:
    source_map = {item.blueprint_id: item for item in sources}
    returned = [item.source_blueprint_id for item in output.paraphrases]
    expected = sorted(source_map)
    accounting = {
        "expected": expected,
        "returned": sorted(returned),
        "missing": sorted(set(expected) - set(returned)),
        "unknown": sorted(set(returned) - set(expected)),
        "duplicates": sorted(
            key for key, count in Counter(returned).items() if count > 1
        ),
    }
    accounting["complete"] = (
        not any(accounting[key] for key in ("missing", "unknown", "duplicates"))
        and output.operator == sources[0].operator
    )
    if not accounting["complete"]:
        raise ValueError(f"PARAPHRASE_ACCOUNTING_FAILURE:{accounting}")
    results = []
    for index, candidate in enumerate(output.paraphrases, 1):
        source = source_map[candidate.source_blueprint_id]
        reasons = []
        if candidate.operator != source.operator or output.operator != source.operator:
            reasons.append("OPERATOR_UNCHANGED")
        if candidate.semantic_class != source.semantic_class:
            reasons.append("SEMANTIC_CLASS_UNCHANGED")
        if candidate.answer_kind != source.answer_kind:
            reasons.append("ANSWER_KIND_UNCHANGED")
        if set(candidate.required_slots) != set(source.required_slots):
            reasons.append("REQUIRED_SLOT_SET_PRESERVED")
        if set(candidate.optional_slots) != set(source.optional_slots):
            reasons.append("OPTIONAL_SLOT_SET_PRESERVED")
        used = set(SLOT_RE.findall(candidate.paraphrase_pattern))
        if not used.issubset(ALLOWED_SLOTS):
            reasons.append("NO_UNKNOWN_SLOT")
        if used != set(source.required_slots) | set(source.optional_slots):
            reasons.append("SLOT_OCCURRENCE_PRESERVED")
        lower = candidate.paraphrase_pattern.casefold()
        if any(token in lower for token in guard["dataset_specific_literals"]):
            reasons.append("NO_DATASET_SPECIFIC_LITERAL")
        if normalized_lexical(candidate.paraphrase_pattern) == normalized_lexical(
            source.pattern
        ):
            reasons.append("LEXICAL_DUPLICATE_SOURCE")
        if (
            source.semantic_class == "numeric_attribute"
            and "exact" in source.match_policies
            and any(term in lower for term in guard["numeric_exact_forbidden"])
        ):
            reasons.append("NUMERIC_EXACT_POLICY_DRIFT")
        if (
            source.semantic_class == "text_content"
            and "exact" in source.match_policies
            and any(term in lower for term in guard["text_exact_forbidden"])
        ):
            reasons.append("TEXT_EXACT_CONTAINMENT_DRIFT")
        if source.operator == "PAIRWISE_SELECTION":
            has_ab = bool(re.search(r"\ba\b", lower)) and bool(
                re.search(r"\bb\b", lower)
            )
            if "[TARGET_VALUE]" not in used or not has_ab:
                reasons.append("PAIRWISE_SELECTION_CONTEXT_DRIFT")
            if any(term in lower for term in guard["selection_exact_forbidden"]):
                reasons.append("SELECTION_MATCH_POLICY_DRIFT")
            if any(term in lower for term in guard["position_leak_forbidden"]):
                reasons.append("POSITION_LEAK")
        elif source.operator == "TARGET_MATCH":
            if "[TARGET_VALUE]" not in used or not any(
                x in lower for x in ("không?", "hay không", "đúng không")
            ):
                reasons.append("TARGET_MATCH_OPERATOR_DRIFT")
            if bool(re.search(r"\ba\b", lower)) and bool(re.search(r"\bb\b", lower)):
                reasons.append("AUDIO_ARITY_DRIFT")
        elif source.operator == "EQUALITY":
            if "hai đoạn" not in lower or not any(
                x in lower for x in ("không?", "hay không", "giống nhau")
            ):
                reasons.append("EQUALITY_OPERATOR_DRIFT")
            if "[TARGET_VALUE]" in used:
                reasons.append("LOGICAL_CONTEXT_DRIFT")
        elif source.operator == "DIRECT":
            if "[TARGET_VALUE]" in used or any(
                x in lower for x in ("có phải", "đúng không")
            ):
                reasons.append("DIRECT_OPERATOR_DRIFT")
        para_id = f"vp1_{source.blueprint_id}_{index:02d}"
        results.append(
            ParaphraseBlueprint(
                paraphrase_id=para_id,
                source_blueprint_id=source.blueprint_id,
                operator=source.operator,
                semantic_class=source.semantic_class,
                pattern=candidate.paraphrase_pattern,
                required_slots=candidate.required_slots,
                optional_slots=candidate.optional_slots,
                answer_kind=candidate.answer_kind,
                source_library_version=source.version,
                semantic_contract_hash=semantic_contract_hash(source),
                generation_provenance={
                    "llm_call_id": call_id,
                    "prompt_hash": prompt_hash,
                },
                paraphrase_status="PASS" if not reasons else "REJECTED",
                reason_codes=sorted(set(reasons)),
            )
        )
    # Deterministic cross-paraphrase duplicate resolution within family.
    seen: dict[tuple[str, str, str], str] = {}
    for item in sorted(results, key=lambda value: value.paraphrase_id):
        key = (item.operator, item.semantic_class, normalized_lexical(item.pattern))
        if key in seen and item.paraphrase_status == "PASS":
            item.paraphrase_status = "REJECTED"
            item.reason_codes.append("LEXICAL_DUPLICATE")
        else:
            seen[key] = item.paraphrase_id
    return results, accounting


def load_flat_metadata(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit_capacity(
    contracts: list[TypeContract], metadata_path: Path
) -> dict[str, Any]:
    rows = load_flat_metadata(metadata_path)
    complete = len(rows) == 15023
    results = []
    for contract in contracts:
        field = contract.semantic_field
        counts = Counter(
            normalize_value(row.get(field), field)
            for row in rows
            if row.get(field) not in (None, "")
        )
        valid_rows = sum(counts.values())
        positive_groups = sum(count >= 2 for count in counts.values())
        distinct = len(counts)
        positive = valid_rows >= 1
        negative = distinct >= 2
        selection = positive and negative
        if contract.operator == "EQUALITY":
            positive = positive_groups >= 1
            status = (
                "SUFFICIENT"
                if positive and negative
                else (
                    "INSUFFICIENT_POSITIVE_CAPACITY"
                    if not positive
                    else "INSUFFICIENT_NEGATIVE_CAPACITY"
                )
            )
        elif contract.operator == "DIRECT":
            status = "SUFFICIENT" if positive else "UNKNOWN"
        else:
            status = "SUFFICIENT" if selection else "INSUFFICIENT_NEGATIVE_CAPACITY"
        if not complete:
            status = "NOT_AUDITED"
        results.append(
            {
                "type_id": contract.type_id,
                "operator": contract.operator,
                "field": field,
                "metadata_rows_examined": len(rows),
                "source_scope": "COMPLETE_LOCAL_TRAIN"
                if complete
                else "PARTIAL_LOCAL_ONLY",
                "positive_capacity": positive,
                "negative_capacity": negative
                if contract.operator != "DIRECT"
                else None,
                "selection_capacity": selection
                if contract.operator == "PAIRWISE_SELECTION"
                else None,
                "full_train_capacity_status": status,
                "semantic_status": contract.source_status,
                "evidence_summary": {
                    "valid_rows": valid_rows,
                    "unique_values": distinct,
                    "groups_with_count_ge_2": positive_groups,
                    "maximum_group_size": max(counts.values(), default=0),
                    "negative_value_availability": negative,
                    "pair_enumeration": "NOT_PERFORMED",
                },
            }
        )
    return {
        "metadata_source": str(metadata_path),
        "metadata_rows": len(rows),
        "complete_official_train": complete,
        "complexity": "O(N) grouping; no O(N^2) pair enumeration",
        "types": results,
    }









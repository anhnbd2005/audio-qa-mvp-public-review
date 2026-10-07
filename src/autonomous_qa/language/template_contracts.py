"""Production question-template design and deterministic TRAIN instantiation.

The LLM is restricted to language design.  Every data binding, pair/target
choice, gold derivation, validation decision and preview record is produced by
Python.  Operator implementations consume TypeContract.semantic_field and are
therefore dataset agnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from src.common.config import ROOT, resolve_llm_base_url, resolve_llm_model
from src.autonomous_qa.authoring.llm_client import (
    GenerateContentConfig,
    ThinkingConfig,
    create_llm_client,
)

OperatorId = Literal[
    "DIRECT", "EQUALITY", "PAIRWISE_SELECTION", "TARGET_MATCH", "COMPOSITE"
]
PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z0-9_]*\]")
ALLOWED_PLACEHOLDERS = frozenset(
    {
        "[TARGET_VALUE]",
        "[AUDIO_A]",
        "[AUDIO_B]",
        "[AUDIO_REFERENCE]",
        "[INSTRUCTION_1]",
        "[INSTRUCTION_2]",
        "[INSTRUCTION_3]",
    }
)
MIN_TEMPLATES_PER_TYPE = 2
MAX_TEMPLATES_PER_TYPE = 3


class OperatorContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operator_id: OperatorId
    audio_input_count: int
    context_roles: list[str]
    answer_kind: str
    target_role: str | None = None
    gold_derivation: str
    required_constraints: list[str] = Field(default_factory=list)
    allowed_template_placeholders: list[str] = Field(default_factory=list)
    forbidden_placeholders: list[str] = Field(default_factory=list)


class TypeContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type_id: str
    operator: OperatorId
    semantic_field: str
    semantic_description: str
    semantic_phrase_vi: str
    entity_scope: str
    audio_input_count: int
    condition_fields: list[str]
    gold_source_fields: list[str]
    answer_kind: str
    answer_mode: str
    template_status: str
    template_policy: dict[str, Any]
    instantiation_policy: dict[str, Any]
    source_status: str
    source_status_reason: str | None = None


class QuestionTemplateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    template_id: str
    type_id: str
    operator: OperatorId
    question_text: str
    language: str = "vi"
    answer_mode: str
    audio_reference_style: str
    required_placeholders: list[str] = Field(default_factory=list)
    optional_placeholders: list[str] = Field(default_factory=list)
    forbidden_placeholders: list[str] = Field(default_factory=list)
    allowed_answer_forms: list[str] = Field(default_factory=list)
    notes: str | None = None


class OperatorTemplateOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operator: OperatorId
    templates: list[QuestionTemplateSpec]


class GoldRecord(BaseModel):
    kind: str
    value: Any


class QARecord(BaseModel):
    qa_id: str
    dataset: str
    type_id: str
    template_id: str
    operator: OperatorId
    audio_ids: list[str]
    visible_context: dict[str, Any]
    question: str
    gold: GoldRecord
    source_rows: list[str]
    generation: dict[str, Any]
    derivation_evidence: dict[str, Any]


def operator_contracts() -> dict[str, OperatorContract]:
    """The only four execution families; contains no dataset field names."""
    return {
        "DIRECT": OperatorContract(
            operator_id="DIRECT",
            audio_input_count=1,
            context_roles=[],
            answer_kind="field_value",
            gold_derivation="hidden_field_value",
            forbidden_placeholders=["[TARGET_VALUE]"],
        ),
        "EQUALITY": OperatorContract(
            operator_id="EQUALITY",
            audio_input_count=2,
            context_roles=[],
            answer_kind="boolean",
            gold_derivation="field(A) == field(B)",
            required_constraints=["different_audio_ids"],
            forbidden_placeholders=["[TARGET_VALUE]"],
        ),
        "PAIRWISE_SELECTION": OperatorContract(
            operator_id="PAIRWISE_SELECTION",
            audio_input_count=2,
            context_roles=["target_value"],
            answer_kind="audio_index",
            target_role="target_value",
            gold_derivation="match(A,target) XOR match(B,target)",
            required_constraints=["exactly_one_candidate_matches_target"],
            allowed_template_placeholders=["[TARGET_VALUE]", "[AUDIO_A]", "[AUDIO_B]"],
        ),
        "TARGET_MATCH": OperatorContract(
            operator_id="TARGET_MATCH",
            audio_input_count=1,
            context_roles=["target_value"],
            answer_kind="boolean",
            target_role="target_value",
            gold_derivation="field(audio) == target_value",
            allowed_template_placeholders=["[TARGET_VALUE]"],
        ),
        "COMPOSITE": OperatorContract(
            operator_id="COMPOSITE",
            audio_input_count=0,
            context_roles=[],
            answer_kind="structured",
            gold_derivation="ordered_structured_components",
            allowed_template_placeholders=[
                "[TARGET_VALUE]",
                "[AUDIO_REFERENCE]",
                "[INSTRUCTION_1]",
                "[INSTRUCTION_2]",
                "[INSTRUCTION_3]",
            ],
        ),
    }


FIELD_SEMANTICS = {
    "text": ("utterance", "spoken content transcription", "nội dung lời nói"),
    "region": ("speaker", "broad macro-dialect region", "vùng phương ngữ"),
    "province_name": (
        "speaker",
        "fine-grained provincial dialect category",
        "phương ngữ tỉnh/thành",
    ),
    "speakerID": ("speaker", "speaker identity", "danh tính người nói"),
}


def _canonical_operator(entry: dict[str, Any]) -> str:
    kind = entry["operation"]["kind"]
    return "PAIRWISE_SELECTION" if kind == "SELECTION" else kind


def build_type_contracts_from_legacy_registry(registry: dict[str, Any]) -> list[TypeContract]:
    """Migration compatibility path for legacy production_registry.json."""
    contracts: list[TypeContract] = []
    entries = registry.get("entries", [])
    for entry in entries:
        status = entry["status"]
        if status not in {"SUPPORTED", "REVIEW_REQUIRED"}:
            continue
        field = entry["gold_source_fields"][0]
        scope, description, phrase = FIELD_SEMANTICS.get(
            field, ("unknown", "dataset semantic attribute", "thuộc tính ngữ nghĩa")
        )
        operator = _canonical_operator(entry)
        answer_mode = {
            "DIRECT": "FIELD_VALUE",
            "EQUALITY": "BOOLEAN",
            "PAIRWISE_SELECTION": "A_B_SELECTION",
            "TARGET_MATCH": "BOOLEAN",
            "COMPOSITE": "STRUCTURED",
        }[operator]
        deferred = status == "REVIEW_REQUIRED"
        contracts.append(
            TypeContract(
                type_id=entry["internal_type_id"],
                operator=operator,
                semantic_field=field,
                semantic_description=description,
                semantic_phrase_vi=phrase,
                entity_scope=scope,
                audio_input_count=entry["audio_input_count"],
                condition_fields=list(entry.get("visible_context_fields", [])),
                gold_source_fields=list(entry["gold_source_fields"]),
                answer_kind=operator_contracts()[operator].answer_kind,
                answer_mode=answer_mode,
                template_status="DEFERRED_DATA_CAPACITY" if deferred else "ELIGIBLE",
                template_policy={
                    "templates_min": MIN_TEMPLATES_PER_TYPE,
                    "templates_max": MAX_TEMPLATES_PER_TYPE,
                    "visible_target": operator in {"PAIRWISE_SELECTION", "TARGET_MATCH"},
                },
                instantiation_policy={
                    "split": "train",
                    "normalize_text": field == "text",
                    "avoid_same_speaker": operator == "EQUALITY"
                    and field in {"region", "province_name"},
                    "text_negative_length_bucket": field == "text"
                    and operator in {"PAIRWISE_SELECTION", "TARGET_MATCH"},
                },
                source_status=status,
                source_status_reason=entry.get("status_reason"),
            )
        )
    return contracts


def build_type_contracts_from_semantic_catalog(
    catalog: Any,
    field_specs: dict[str, Any] | None = None,
) -> list[TypeContract]:
    """Clean generic canonical/staged path from SemanticCatalog with zero type-id heuristics."""
    tasks = catalog.get("tasks", []) if isinstance(catalog, dict) else catalog.tasks
    contracts: list[TypeContract] = []

    for task_raw in tasks:
        task = task_raw.model_dump(mode="json") if hasattr(task_raw, "model_dump") else task_raw
        if "operator" not in task:
            raise ValueError("MISSING_STRUCTURED_OPERATOR")
        operator = task["operator"]

        if operator in {"TARGET_MATCH", "EQUALITY"}:
            answer_mode = "BOOLEAN"
        elif operator == "PAIRWISE_SELECTION":
            answer_mode = "A_B_SELECTION"
        elif operator == "COMPOSITE":
            answer_mode = "STRUCTURED"
        else:
            answer_mode = "FIELD_VALUE"

        source_role = task.get("source_role_mapping", {})
        outputs = task.get("outputs", [])
        source_field = source_role.get("source_field") or (
            outputs[0]["dependencies"][0] if outputs and outputs[0].get("dependencies") else None
        )
        if not source_field:
            raise ValueError("MISSING_STRUCTURED_SOURCE_FIELD")

        if "audio_arity" not in task:
            raise ValueError("MISSING_STRUCTURED_AUDIO_ARITY")
        audio_arity = task["audio_arity"]

        if not outputs or not outputs[0].get("kind"):
            raise ValueError("MISSING_STRUCTURED_OUTPUTS")
        answer_kind = outputs[0]["kind"]

        condition_fields = ["target_value"] if operator in {"TARGET_MATCH", "PAIRWISE_SELECTION"} else []
        type_id = task["type_id"]

        normalize_text = False
        if field_specs and source_field in field_specs:
            spec = field_specs[source_field]
            val_policy = getattr(spec, "value_policy", None) or (spec.get("value_policy", {}) if isinstance(spec, dict) else None)
            match_norm = getattr(val_policy, "normalization", None) or (val_policy.get("normalization") if isinstance(val_policy, dict) else None)
            if match_norm and match_norm != "identity":
                normalize_text = True

        contracts.append(
            TypeContract(
                type_id=type_id,
                operator=operator,
                semantic_field=source_field,
                semantic_description=task.get("proposition_description", ""),
                semantic_phrase_vi="",
                entity_scope="utterance",
                audio_input_count=audio_arity,
                condition_fields=condition_fields,
                gold_source_fields=[source_field],
                answer_kind=answer_kind,
                answer_mode=answer_mode,
                template_status="ELIGIBLE",
                template_policy={
                    "templates_min": MIN_TEMPLATES_PER_TYPE,
                    "templates_max": MAX_TEMPLATES_PER_TYPE,
                    "visible_target": operator in {"PAIRWISE_SELECTION", "TARGET_MATCH"},
                },
                instantiation_policy={
                    "split": "train",
                    "normalize_text": normalize_text,
                },
                source_status="SUPPORTED",
            )
        )
    return contracts


def build_type_contracts(registry_or_catalog: dict[str, Any]) -> list[TypeContract]:
    """Dispatcher for TypeContract compilation: SemanticCatalog vs Legacy Registry."""
    if "tasks" in registry_or_catalog:
        return build_type_contracts_from_semantic_catalog(registry_or_catalog)
    return build_type_contracts_from_legacy_registry(registry_or_catalog)


def schema_json() -> str:
    return json.dumps(
        OperatorTemplateOutput.model_json_schema(), ensure_ascii=False, indent=2
    )


def build_operator_prompt(operator: str, contracts: list[TypeContract]) -> str:
    safe = [
        {
            "type_id": c.type_id,
            "operator": c.operator,
            "semantic_description": c.semantic_description,
            "semantic_phrase_vi": c.semantic_phrase_vi,
            "entity_scope": c.entity_scope,
            "answer_kind": c.answer_kind,
            "answer_mode": c.answer_mode,
            "audio_input_count": c.audio_input_count,
        }
        for c in contracts
    ]
    contract = operator_contracts()[operator].model_dump()
    return (
        "Bạn thiết kế câu hỏi chuẩn bằng tiếng Việt cho dữ liệu âm thanh. "
        "Chỉ tạo wording/template, tuyệt đối không tạo gold, lựa chọn, nhãn dữ liệu, "
        "sample ID, filename, speaker ID hay giá trị metadata thực.\n"
        "Trả về JSON duy nhất, không Markdown, không văn bản ngoài JSON.\n"
        "Mỗi type_id phải có đúng 3 template khác nhau có ích; không thêm type_id. "
        "template_id phải duy nhất. Giữ nguyên operator và answer_mode.\n"
        "Chỉ dùng placeholder cố định [TARGET_VALUE], [AUDIO_A], [AUDIO_B]. "
        "[TARGET_VALUE] bắt buộc đúng một lần cho PAIRWISE_SELECTION/TARGET_MATCH "
        "và bị cấm cho DIRECT/EQUALITY. Audio placeholders là tùy chọn.\n"
        "PAIRWISE_SELECTION phải hỏi chọn A hoặc B đối xứng. EQUALITY phải nói rõ hai "
        "đoạn âm thanh. TARGET_MATCH phải là câu hỏi đúng/sai về target. DIRECT phải "
        "hỏi giá trị ngữ nghĩa từ một audio và không lộ đáp án.\n"
        f"OPERATOR CONTRACT:\n{json.dumps(contract, ensure_ascii=False, indent=2)}\n"
        f"TYPE CONTRACTS (không có giá trị dữ liệu):\n{json.dumps(safe, ensure_ascii=False, indent=2)}\n"
        "JSON SCHEMA (authoritative, final prompt section):\n" + schema_json()
    )


def normalize_lexical(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("audio", "âm thanh")
    text = re.sub(r"\[[A-Z][A-Z0-9_]*\]", " placeholder ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def validate_operator_output(
    output: OperatorTemplateOutput,
    contracts: list[TypeContract],
    raw_metadata_values: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {c.type_id: c for c in contracts}
    errors: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    ids = [t.template_id for t in output.templates]
    if len(ids) != len(set(ids)):
        errors.append({"code": "DUPLICATE_TEMPLATE_ID", "scope": output.operator})
    returned = Counter(t.type_id for t in output.templates)
    for unknown in sorted(set(returned) - set(by_id)):
        errors.append({"code": "UNKNOWN_TYPE", "type_id": unknown})
    for type_id in sorted(by_id):
        count = returned[type_id]
        if count == 0:
            errors.append({"code": "MISSING_TYPE", "type_id": type_id})
        elif not MIN_TEMPLATES_PER_TYPE <= count <= MAX_TEMPLATES_PER_TYPE:
            errors.append(
                {"code": "TEMPLATE_COUNT_INVALID", "type_id": type_id, "count": count}
            )
    fingerprints: dict[tuple[str, str], str] = {}
    for template in output.templates:
        local: list[str] = []
        contract = by_id.get(template.type_id)
        if contract is None:
            local.append("TYPE_EXISTS")
        else:
            if (
                template.operator != contract.operator
                or output.operator != contract.operator
            ):
                local.append("OPERATOR_MATCH")
            if template.answer_mode != contract.answer_mode:
                local.append("ANSWER_MODE_MATCHES_OPERATOR")
            placeholders = PLACEHOLDER_RE.findall(template.question_text)
            declared = template.required_placeholders + template.optional_placeholders
            if any(p not in ALLOWED_PLACEHOLDERS for p in placeholders + declared):
                local.append("NO_UNKNOWN_PLACEHOLDER")
            required = set(template.required_placeholders)
            allowed_declared = required | set(template.optional_placeholders)
            used = set(placeholders)
            if not required.issubset(used) or not used.issubset(allowed_declared):
                local.append("PLACEHOLDER_SET_VALID")
            target_count = placeholders.count("[TARGET_VALUE]")
            if contract.operator in {"DIRECT", "EQUALITY"} and target_count:
                local.append("TARGET_PLACEHOLDER_FORBIDDEN")
            if (
                contract.operator in {"PAIRWISE_SELECTION", "TARGET_MATCH"}
                and target_count != 1
            ):
                local.append("TARGET_PLACEHOLDER_REQUIRED")
            q = template.question_text.strip()
            qlow = q.lower()
            if not q:
                local.append("QUESTION_NONEMPTY")
            if template.language.lower() not in {"vi", "vi-vn"}:
                local.append("LANGUAGE_VALID")
            forbidden = (
                "speakerid",
                "province_code",
                "filename",
                "sample_id",
                "audio_path",
            )
            if any(x in qlow for x in forbidden):
                local.append("NO_HIDDEN_IDENTIFIER_OR_PROVENANCE")
            if re.search(r"\b[\w-]+\.wav\b", qlow):
                local.append("NO_FILENAME")
            if raw_metadata_values and any(
                v and v.casefold() in q.casefold() for v in raw_metadata_values
            ):
                local.append("NO_RAW_METADATA_LEAK")
            explicit_ab = ("[AUDIO_A]" in q and "[AUDIO_B]" in q) or bool(
                re.search(r"\bA\b.+\bB\b", q, flags=re.DOTALL)
            )
            two_audio = explicit_ab or any(
                x in qlow
                for x in (
                    "hai đoạn",
                    "a và b",
                    "a hoặc b",
                    "đoạn a",
                    "đoạn âm thanh nào",
                )
            )
            boolean = any(x in qlow for x in ("không?", "hay không", "đúng hay sai"))
            drift = (
                (contract.operator == "DIRECT" and two_audio)
                or (
                    contract.operator in {"EQUALITY", "PAIRWISE_SELECTION"}
                    and not two_audio
                )
                or (contract.operator == "TARGET_MATCH" and not boolean)
            )
            if drift:
                local.append("OPERATOR_SEMANTIC_DRIFT")
            fp = normalize_lexical(q)
            key = (template.type_id, fp)
            if key in fingerprints:
                local.append("NEAR_DUPLICATE_TEMPLATE")
            fingerprints[key] = template.template_id
        result = template.model_dump()
        result["validation_status"] = "PASS" if not local else "REJECT"
        result["reasons"] = sorted(set(local))
        results.append(result)
    return results, errors


def normalize_value(value: Any, field: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.split()).casefold() if field == "text" else text


def load_train_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError("NON_TRAIN_ROW")
            rows.append(row)
    return rows


class IndexedSampler:
    """O(N) indexes followed by bounded O(K) sampling; never lists all pairs."""

    def __init__(self, rows: list[dict[str, Any]], field: str, seed: int):
        self.rows = rows
        self.field = field
        self.rng = random.Random(seed)
        self.groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.display: dict[str, Any] = {}
        for row in rows:
            raw = row["metadata"].get(field)
            key = normalize_value(raw, field)
            if key:
                self.groups[key].append(row)
                self.display.setdefault(key, raw)
        self.keys = sorted(self.groups)
        self.positive_keys = [k for k in self.keys if len(self.groups[k]) >= 2]

    def _different_key(self, key: str, text_compatible: bool = False) -> str:
        candidates = [k for k in self.keys if k != key]
        if text_compatible:
            size = max(1, len(key))
            close = [k for k in candidates if 0.6 <= len(k) / size <= 1.67]
            if close:
                candidates = close
        if not candidates:
            raise ValueError("NO_NEGATIVE_CAPACITY")
        return candidates[self.rng.randrange(len(candidates))]

    def direct(self) -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
        row = self.rows[self.rng.randrange(len(self.rows))]
        value = row["metadata"].get(self.field)
        if value in {None, ""}:
            raise ValueError("EMPTY_TEXT")
        return [row], value, {}

    def equality(
        self, positive: bool, avoid_same_speaker: bool
    ) -> tuple[list[dict[str, Any]], bool, dict[str, Any]]:
        for _ in range(100):
            if positive:
                if not self.positive_keys:
                    raise ValueError("NO_POSITIVE_CAPACITY")
                key = self.rng.choice(self.positive_keys)
                a, b = self.rng.sample(self.groups[key], 2)
            else:
                ka, kb = self.rng.sample(self.keys, 2)
                a, b = (
                    self.rng.choice(self.groups[ka]),
                    self.rng.choice(self.groups[kb]),
                )
            if a["sample_id"] == b["sample_id"]:
                continue
            if avoid_same_speaker and a["metadata"].get("speakerID") == b[
                "metadata"
            ].get("speakerID"):
                continue
            gold = normalize_value(
                a["metadata"][self.field], self.field
            ) == normalize_value(b["metadata"][self.field], self.field)
            return [a, b], gold, {"positive": gold}
        raise ValueError("PAIR_SAMPLER_EXHAUSTED")

    def target_match(
        self, positive: bool, text_compatible: bool
    ) -> tuple[list[dict[str, Any]], bool, dict[str, Any]]:
        row = self.rng.choice(self.rows)
        actual_key = normalize_value(row["metadata"].get(self.field), self.field)
        target_key = (
            actual_key if positive else self._different_key(actual_key, text_compatible)
        )
        target = self.display[target_key]
        gold = actual_key == normalize_value(target, self.field)
        return [row], gold, {"target_value": target, "positive": gold}

    def selection(
        self, answer_position: str, text_compatible: bool
    ) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
        target_key = self.rng.choice(self.keys)
        negative_key = self._different_key(target_key, text_compatible)
        match = self.rng.choice(self.groups[target_key])
        nonmatch = self.rng.choice(self.groups[negative_key])
        rows = [match, nonmatch] if answer_position == "A" else [nonmatch, match]
        target = self.display[target_key]
        matches = [
            normalize_value(r["metadata"][self.field], self.field) == target_key
            for r in rows
        ]
        if sum(matches) != 1:
            raise ValueError("INVALID_SELECTION_CARDINALITY")
        return rows, answer_position, {"target_value": target, "matches": matches}


def _stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256((str(seed) + "|" + "|".join(parts)).encode()).hexdigest()
    return int(digest[:16], 16)


def instantiate_type(
    type_contract: TypeContract,
    template: QuestionTemplateSpec | dict[str, Any],
    train_metadata: list[dict[str, Any]],
    sampler_config: dict[str, Any],
) -> list[QARecord]:
    if isinstance(template, dict):
        template = QuestionTemplateSpec.model_validate(
            {
                k: v
                for k, v in template.items()
                if k in QuestionTemplateSpec.model_fields
            }
        )
    count = int(sampler_config["instances"])
    seed = int(sampler_config["seed"])
    sampler = IndexedSampler(
        train_metadata,
        type_contract.semantic_field,
        _stable_seed(seed, type_contract.type_id, template.template_id),
    )
    records: list[QARecord] = []
    template_hash = hashlib.sha256(template.question_text.encode("utf-8")).hexdigest()
    for index in range(count):
        operator = type_contract.operator
        if operator == "DIRECT":
            rows, gold, context = sampler.direct()
        elif operator == "EQUALITY":
            rows, gold, context = sampler.equality(
                positive=index % 2 == 0,
                avoid_same_speaker=bool(
                    type_contract.instantiation_policy.get("avoid_same_speaker")
                ),
            )
        elif operator == "TARGET_MATCH":
            rows, gold, context = sampler.target_match(
                positive=index % 2 == 0,
                text_compatible=bool(
                    type_contract.instantiation_policy.get(
                        "text_negative_length_bucket"
                    )
                ),
            )
        elif operator == "PAIRWISE_SELECTION":
            rows, gold, context = sampler.selection(
                answer_position="A" if index % 2 == 0 else "B",
                text_compatible=bool(
                    type_contract.instantiation_policy.get(
                        "text_negative_length_bucket"
                    )
                ),
            )
        else:
            raise ValueError(f"UNKNOWN_OPERATOR:{operator}")
        question = template.question_text
        if "target_value" in context:
            question = question.replace("[TARGET_VALUE]", str(context["target_value"]))
        question = question.replace("[AUDIO_A]", "đoạn âm thanh A").replace(
            "[AUDIO_B]", "đoạn âm thanh B"
        )
        record = QARecord(
            qa_id=f"{type_contract.type_id}:{template.template_id}:{index + 1:03d}",
            dataset="vimd",
            type_id=type_contract.type_id,
            template_id=template.template_id,
            operator=operator,
            audio_ids=[r["audio_path"] for r in rows],
            visible_context={k: v for k, v in context.items() if k == "target_value"},
            question=question,
            gold=GoldRecord(kind=type_contract.answer_kind, value=gold),
            source_rows=[r["sample_id"] for r in rows],
            generation={
                "seed": seed,
                "sampler": "indexed_bounded_v1",
                "template_hash": template_hash,
            },
            derivation_evidence={
                "field": type_contract.semantic_field,
                "hidden_values": [
                    r["metadata"].get(type_contract.semantic_field) for r in rows
                ],
                **context,
            },
        )
        records.append(record)
    return records


def validate_instance(record: QARecord) -> list[str]:
    reasons: list[str] = []
    q = record.question.casefold()
    evidence = record.derivation_evidence
    if any(
        x in q
        for x in ("speakerid", "province_code", "filename", "sample_id", "audio_path")
    ):
        reasons.append("HIDDEN_IDENTIFIER_LEAK")
    if re.search(r"\b[\w-]+\.wav\b", q):
        reasons.append("FILENAME_LEAK")
    if any(not path or not Path(path).exists() for path in record.audio_ids):
        reasons.append("AUDIO_PATH_MISSING")
    if len(record.audio_ids) != len(set(record.audio_ids)):
        reasons.append("DUPLICATE_AUDIO_PAIR")
    if record.operator == "PAIRWISE_SELECTION":
        matches = evidence.get("matches", [])
        if sum(bool(x) for x in matches) != 1:
            reasons.append("INVALID_SELECTION_CARDINALITY")
        expected = "A" if matches == [True, False] else "B"
        if record.gold.value != expected:
            reasons.append("NONDETERMINISTIC_GOLD")
    elif record.operator == "TARGET_MATCH":
        actual = normalize_value(evidence["hidden_values"][0], evidence["field"])
        target = normalize_value(evidence.get("target_value"), evidence["field"])
        if record.gold.value != (actual == target):
            reasons.append("NONDETERMINISTIC_GOLD")
        if not record.gold.value and actual == target:
            reasons.append("INVALID_NEGATIVE_TARGET")
    elif record.operator == "EQUALITY":
        values = [
            normalize_value(v, evidence["field"]) for v in evidence["hidden_values"]
        ]
        if record.gold.value != (values[0] == values[1]):
            reasons.append("NONDETERMINISTIC_GOLD")
    elif record.operator == "DIRECT":
        hidden = str(evidence["hidden_values"][0])
        if hidden and hidden.casefold() in q:
            reasons.append("VISIBLE_GOLD_LEAK")
    if not record.question.strip():
        reasons.append("EMPTY_TEXT")
    return sorted(set(reasons))


def model_facing(record: QARecord) -> dict[str, Any]:
    return {
        "qa_id": record.qa_id,
        "dataset": record.dataset,
        "type_id": record.type_id,
        "template_id": record.template_id,
        "operator": record.operator,
        "audio": record.audio_ids,
        "visible_context": record.visible_context,
        "question": record.question,
        "answer": record.gold.model_dump(),
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def parse_llm_json(text: str) -> OperatorTemplateOutput:
    # Deliberately no fence stripping or semantic salvage.
    return OperatorTemplateOutput.model_validate(json.loads(text))


def call_template_llm(client: Any, prompt: str) -> tuple[str, Any]:
    response = client.models.generate_content(
        model=resolve_llm_model(),
        contents=prompt,
        config=GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=8192,
            response_mime_type="application/json",
            response_schema=OperatorTemplateOutput,
            thinking_config=ThinkingConfig(thinking_budget=1536),
        ),
    )
    return response.text or "", response







"""Deterministic Python audit for generated QA instances (zero-LLM)."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from pydantic import BaseModel, ConfigDict

from src.autonomous_qa.language.language_quality import ProductionLanguageRegistry
from src.autonomous_qa.certification.qa_sampling import TrainIndex, meta_of, normalize_field_value
from src.autonomous_qa.compiler.semantic_field_specs import SemanticFieldSpec
from src.autonomous_qa.language.template_contracts import TypeContract, operator_contracts
from src.autonomous_qa.language.template_renderer import SLOT_RE

FILENAME_RE = re.compile(r"\b[\w-]+\.wav\b", re.IGNORECASE)
PROVENANCE_TOKENS = (
    ".wav",
    ".parquet",
    "_shard",
    "speakerid",
    "province_code",
    "filename",
    "sample_id",
    "train-",
)

AUDIT_CHECKS = (
    "KNOWN_TYPE",
    "TYPE_SUPPORTED",
    "TYPE_FULL_TRAIN_CAPACITY_SUFFICIENT",
    "KNOWN_LANGUAGE_ENTRY",
    "LANGUAGE_ENTRY_ACTIVE",
    "LANGUAGE_COMPATIBLE",
    "OPERATOR_MATCH",
    "SEMANTIC_CLASS_MATCH",
    "ANSWER_KIND_MATCH",
    "AUDIO_COUNT_MATCH",
    "NO_UNRESOLVED_SLOT",
    "TARGET_ALLOWED",
    "GOLD_DETERMINISTIC",
    "NEGATIVE_TARGET_IS_NEGATIVE",
    "EQUALITY_RELATION_CORRECT",
    "PAIRWISE_SELECTION_XOR",
    "NO_IDENTICAL_AUDIO_PAIR",
    "NO_HIDDEN_IDENTIFIER_LEAK",
    "NO_FILENAME_LEAK",
    "NO_PROVENANCE_LEAK",
    "NO_VISIBLE_GOLD_LEAK",
    "SEMANTIC_INSTANCE_UNIQUE",
    "QA_ID_UNIQUE",
    "SCHEMA_VALID",
)


class GoldModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str
    value: Any


class InternalProvenanceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_row_ids: list[str]
    hidden_source_field: str
    hidden_values: list[Any]
    target_normalized: str | None = None


class InternalQAModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    qa_id: str
    semantic_instance_id: str
    dataset: str
    split: str
    dataset_revision: str
    type_id: str
    operator: str
    semantic_class: str
    language_entry_id: str
    audio_ids: list[str]
    visible_context: dict[str, Any]
    question: str
    gold: GoldModel
    internal: InternalProvenanceModel
    generation: dict[str, Any]


class ModelFacingQAModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    audio: list[str]
    question: str
    answer: str
    type_id: str
    operator: str


def _fold(text: Any) -> str:
    return " ".join(str(text).casefold().split())


def _visible_target(record: dict[str, Any]) -> Any:
    return record["visible_context"].get("target_value")


def _gold_is_deterministic(record: dict[str, Any], spec: SemanticFieldSpec) -> bool:
    operator = record["operator"]
    hidden = record["internal"]["hidden_values"]
    policy = spec.value_policy
    gold = record["gold"]["value"]
    if operator == "DIRECT":
        return gold == hidden[0]
    if operator == "EQUALITY":
        left = normalize_field_value(hidden[0], policy)
        right = normalize_field_value(hidden[1], policy)
        return bool(gold) is (left == right)
    if operator == "TARGET_MATCH":
        target = record["internal"].get("target_normalized")
        actual = normalize_field_value(hidden[0], policy)
        return bool(gold) is (actual == target)
    if operator == "PAIRWISE_SELECTION":
        target = record["internal"].get("target_normalized")
        matches = [normalize_field_value(value, policy) == target for value in hidden]
        if sum(bool(m) for m in matches) != 1:
            return False
        expected = "A" if matches[0] else "B"
        return gold == expected
    return False


def _visible_gold_leak(record: dict[str, Any], spec: SemanticFieldSpec) -> bool:
    question = _fold(record["question"])
    hidden = record["internal"]["hidden_values"]
    policy = spec.value_policy
    target = record["internal"].get("target_normalized")
    operator = record["operator"]
    if operator in {"DIRECT", "EQUALITY"}:
        return any(_fold(v) and _fold(v) in question for v in hidden)
    # TARGET_MATCH / PAIRWISE_SELECTION: the visible target may legitimately
    # equal the matching candidate label; any other hidden label is a leak.
    return any(
        _fold(v) and _fold(v) in question and normalize_field_value(v, policy) != target
        for v in hidden
    )


def audit_record(
    record: dict[str, Any],
    *,
    contracts: dict[str, TypeContract],
    specs: dict[str, SemanticFieldSpec],
    registry: ProductionLanguageRegistry,
    compatible_by_type: dict[str, set[str]],
    capacity_status: dict[str, str],
    index: TrainIndex,
    model_record: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    type_id = record.get("type_id")
    contract = contracts.get(type_id)
    if contract is None:
        return ["KNOWN_TYPE"]
    spec = specs[contract.semantic_field]
    entry = next(
        (
            item
            for item in registry.entries
            if item.language_entry_id == record.get("language_entry_id")
        ),
        None,
    )
    if entry is None:
        failures.append("KNOWN_LANGUAGE_ENTRY")
    if contract.source_status != "SUPPORTED":
        failures.append("TYPE_SUPPORTED")
    if capacity_status.get(type_id) != "SUFFICIENT":
        failures.append("TYPE_FULL_TRAIN_CAPACITY_SUFFICIENT")
    if entry is not None:
        if not entry.enabled:
            failures.append("LANGUAGE_ENTRY_ACTIVE")
        if type_id not in compatible_by_type or (
            entry.language_entry_id not in compatible_by_type[type_id]
        ):
            failures.append("LANGUAGE_COMPATIBLE")
        if entry.operator != contract.operator:
            failures.append("OPERATOR_MATCH")
        if entry.semantic_class != spec.semantic_class:
            failures.append("SEMANTIC_CLASS_MATCH")
        if entry.answer_kind != contract.answer_kind:
            failures.append("ANSWER_KIND_MATCH")
    if len(record["audio_ids"]) != contract.audio_input_count:
        failures.append("AUDIO_COUNT_MATCH")
    if SLOT_RE.search(record["question"]):
        failures.append("NO_UNRESOLVED_SLOT")
    operator_contract = operator_contracts()[contract.operator]
    wants_target = "target_value" in operator_contract.context_roles
    visible_keys = set(record["visible_context"])
    if not visible_keys.issubset({"target_value"}):
        failures.append("TARGET_ALLOWED")
    if wants_target:
        target_display = record["visible_context"].get("target_value")
        if target_display is None or _fold(target_display) not in _fold(
            record["question"]
        ):
            failures.append("TARGET_ALLOWED")
    elif visible_keys:
        failures.append("TARGET_ALLOWED")
    if not _gold_is_deterministic(record, spec):
        failures.append("GOLD_DETERMINISTIC")
    if contract.operator == "TARGET_MATCH" and record["gold"]["value"] is False:
        target = record["internal"].get("target_normalized")
        actual = normalize_field_value(
            record["internal"]["hidden_values"][0], spec.value_policy
        )
        if target is None or actual == target:
            failures.append("NEGATIVE_TARGET_IS_NEGATIVE")
    if contract.operator == "EQUALITY":
        left = normalize_field_value(
            record["internal"]["hidden_values"][0], spec.value_policy
        )
        right = normalize_field_value(
            record["internal"]["hidden_values"][1], spec.value_policy
        )
        if bool(record["gold"]["value"]) is not (left == right):
            failures.append("EQUALITY_RELATION_CORRECT")
    if contract.operator == "PAIRWISE_SELECTION":
        target = record["internal"].get("target_normalized")
        matches = [
            normalize_field_value(value, spec.value_policy) == target
            for value in record["internal"]["hidden_values"]
        ]
        if sum(bool(m) for m in matches) != 1:
            failures.append("PAIRWISE_SELECTION_XOR")
    if len(set(record["audio_ids"])) != len(record["audio_ids"]):
        failures.append("NO_IDENTICAL_AUDIO_PAIR")
    question_folded = _fold(record["question"])
    hidden_identifiers = [
        index.hidden_value_by_row.get(row_id)
        for row_id in record["internal"]["source_row_ids"]
    ]
    if any(value and _fold(value) in question_folded for value in hidden_identifiers):
        failures.append("NO_HIDDEN_IDENTIFIER_LEAK")
    if FILENAME_RE.search(record["question"]) or any(
        row_id and _fold(row_id) in question_folded
        for row_id in record["internal"]["source_row_ids"]
    ):
        failures.append("NO_FILENAME_LEAK")
    if any(token in question_folded for token in PROVENANCE_TOKENS):
        failures.append("NO_PROVENANCE_LEAK")
    if _visible_gold_leak(record, spec):
        failures.append("NO_VISIBLE_GOLD_LEAK")
        failures.append("ANSWER_WITHOUT_AUDIO")
    try:
        InternalQAModel.model_validate(record)
        ModelFacingQAModel.model_validate(model_record)
    except Exception:  # noqa: BLE001 - any validation error is an audit failure
        failures.append("SCHEMA_VALID")
    return sorted(set(failures))


def audit_qa_records(
    internal_records: list[dict[str, Any]],
    model_records: list[dict[str, Any]],
    *,
    contracts: dict[str, TypeContract],
    specs: dict[str, SemanticFieldSpec],
    registry: ProductionLanguageRegistry,
    compatible_by_type: dict[str, set[str]],
    capacity_status: dict[str, str],
    index: TrainIndex,
) -> dict[str, Any]:
    per_check = {name: {"pass": 0, "fail": 0} for name in AUDIT_CHECKS}
    per_check["ANSWER_WITHOUT_AUDIO"] = {"pass": 0, "fail": 0}
    failures: list[dict[str, Any]] = []
    semantic_ids = Counter(r["semantic_instance_id"] for r in internal_records)
    qa_ids = Counter(r["qa_id"] for r in internal_records)
    for record, model_record in zip(internal_records, model_records, strict=True):
        failed = audit_record(
            record,
            contracts=contracts,
            specs=specs,
            registry=registry,
            compatible_by_type=compatible_by_type,
            capacity_status=capacity_status,
            index=index,
            model_record=model_record,
        )
        if semantic_ids[record["semantic_instance_id"]] > 1:
            failed.append("SEMANTIC_INSTANCE_UNIQUE")
        if qa_ids[record["qa_id"]] > 1:
            failed.append("QA_ID_UNIQUE")
        for name, counter in per_check.items():
            if name in failed:
                counter["fail"] += 1
            else:
                counter["pass"] += 1
        if failed:
            failures.append({"qa_id": record["qa_id"], "failures": sorted(set(failed))})
    total = len(internal_records)
    passed = total - len(failures)
    return {
        "records": total,
        "records_passed": passed,
        "records_failed": len(failures),
        "all_passed": not failures,
        "checks": per_check,
        "failures": failures[:100],
        "unique_semantic_instances": len(semantic_ids),
        "unique_qa_ids": len(qa_ids),
    }


def boolean_label_audit(
    internal_records: list[dict[str, Any]], configured_ratio: float | None
) -> dict[str, Any]:
    per_type: dict[str, dict[str, Any]] = {}
    for record in internal_records:
        if record["operator"] not in {"EQUALITY", "TARGET_MATCH"}:
            continue
        bucket = per_type.setdefault(record["type_id"], {"positive": 0, "negative": 0})
        if record["gold"]["value"] is True:
            bucket["positive"] += 1
        else:
            bucket["negative"] += 1
    for bucket in per_type.values():
        total = bucket["positive"] + bucket["negative"]
        ratio = bucket["positive"] / total if total else 0.0
        bucket["positive_ratio"] = round(ratio, 6)
        if configured_ratio is not None and total:
            bucket["configured_positive_ratio"] = configured_ratio
            bucket["ratio_within_rounding"] = (
                abs(ratio - configured_ratio) <= (1.0 / total) + 1e-9
            )
        else:
            bucket["configured_positive_ratio"] = configured_ratio
            bucket["ratio_within_rounding"] = True
    return per_type


def selection_position_audit(internal_records: list[dict[str, Any]]) -> dict[str, Any]:
    per_type: dict[str, dict[str, Any]] = {}
    for record in internal_records:
        if record["operator"] != "PAIRWISE_SELECTION":
            continue
        bucket = per_type.setdefault(record["type_id"], {"gold_a": 0, "gold_b": 0})
        if record["gold"]["value"] == "A":
            bucket["gold_a"] += 1
        else:
            bucket["gold_b"] += 1
    for bucket in per_type.values():
        total = bucket["gold_a"] + bucket["gold_b"]
        bucket["total"] = total
        bucket["severe_position_bias"] = (
            total >= 2 and min(bucket["gold_a"], bucket["gold_b"]) == 0
        )
    return per_type


def language_usage_audit(
    internal_records: list[dict[str, Any]],
    registry: ProductionLanguageRegistry,
    compatible_by_type: dict[str, set[str]],
) -> dict[str, Any]:
    counts = Counter(r["language_entry_id"] for r in internal_records)
    kinds = {entry.language_entry_id: entry.source_kind for entry in registry.entries}
    canonical = sum(
        count
        for entry_id, count in counts.items()
        if kinds.get(entry_id) == "CANONICAL"
    )
    paraphrase = sum(
        count
        for entry_id, count in counts.items()
        if kinds.get(entry_id) == "PARAPHRASE"
    )
    compatible_sizes = {
        type_id: len(ids) for type_id, ids in sorted(compatible_by_type.items())
    }
    return {
        "language_entry_counts": dict(sorted(counts.items())),
        "canonical_entries_used": len(
            [e for e in counts if kinds.get(e) == "CANONICAL"]
        ),
        "paraphrase_entries_used": len(
            [e for e in counts if kinds.get(e) == "PARAPHRASE"]
        ),
        "canonical_rows": canonical,
        "paraphrase_rows": paraphrase,
        "distinct_entries_used": len(counts),
        "compatible_entries_per_type": compatible_sizes,
        "zero_variation": len(counts) <= 1 and len(internal_records) >= 1,
        "quota_policy": "none; informational only",
    }


def target_distribution_audit(internal_records: list[dict[str, Any]]) -> dict[str, Any]:
    per_type: dict[str, dict[str, Any]] = {}
    for record in internal_records:
        if record["operator"] not in {"TARGET_MATCH", "PAIRWISE_SELECTION"}:
            continue
        target = record["visible_context"].get("target_value")
        bucket = per_type.setdefault(
            record["type_id"],
            {"field": record["internal"]["hidden_source_field"], "frequencies": {}},
        )
        key = str(target)
        bucket["frequencies"][key] = bucket["frequencies"].get(key, 0) + 1
    for bucket in per_type.values():
        frequencies = bucket["frequencies"]
        bucket["distinct_targets"] = len(frequencies)
        bucket["frequencies"] = dict(
            sorted(frequencies.items(), key=lambda kv: (-kv[1], kv[0]))
        )
    return per_type


def source_reuse_per_type(internal_records: list[dict[str, Any]]) -> dict[str, Any]:
    per_type: dict[str, dict[str, Any]] = {}
    for record in internal_records:
        bucket = per_type.setdefault(
            record["type_id"], {"references": 0, "unique_rows": set()}
        )
        bucket["references"] += len(record["internal"]["source_row_ids"])
        bucket["unique_rows"].update(record["internal"]["source_row_ids"])
    return {
        type_id: {
            "references": bucket["references"],
            "unique_source_rows": len(bucket["unique_rows"]),
        }
        for type_id, bucket in sorted(per_type.items())
    }


def build_distribution_audit(
    internal_records: list[dict[str, Any]],
    *,
    registry: ProductionLanguageRegistry,
    compatible_by_type: dict[str, set[str]],
    configured_positive_ratio: float | None,
    reuse_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "boolean_labels": boolean_label_audit(
            internal_records, configured_positive_ratio
        ),
        "selection_position": selection_position_audit(internal_records),
        "language_usage": language_usage_audit(
            internal_records, registry, compatible_by_type
        ),
        "target_distribution": target_distribution_audit(internal_records),
        "source_reuse": reuse_audit,
        "source_reuse_per_type": source_reuse_per_type(internal_records),
    }


def flat_row_values(index: TrainIndex, row_id: str, field_name: str) -> Any:
    return meta_of(index.row_by_id[row_id]).get(field_name)


def groupby_hidden(rows: list[dict[str, Any]], field_name: str) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for position, row in enumerate(rows):
        value = meta_of(row).get(field_name)
        if value not in (None, ""):
            groups[str(value)].append(position)
    return groups

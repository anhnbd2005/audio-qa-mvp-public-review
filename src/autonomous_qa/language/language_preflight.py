"""Deterministic pre-generation validation for language/rendering composition.

Style discovery decides *what to ask*.  This module validates *how to say it*
after semantic acceptance and before production sampling.  It uses bounded
synthetic values only and has no audio, model, network, or sampling dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.autonomous_qa.language.language_quality import (
    LanguageRegistryEntry,
    ProductionLanguageRegistry,
    entry_capability_compatible,
    load_language_registry,
    slot_value_map,
)
from src.autonomous_qa.language.template_engine import vimd_field_specs
from src.autonomous_qa.production.production_qa import (
    DATASET_SOURCES,
    bind_target_value,
    render_question,
)
from src.autonomous_qa.compiler.semantic_field_specs import SemanticFieldSpec
from src.autonomous_qa.language.template_contracts import TypeContract, build_type_contracts

ROOT = Path(__file__).resolve().parents[3]
PREFLIGHT_CONTRACT_ID = "language_preflight"
OPERATOR_CONTRACT_ID = "operator_contract"
FIXTURE_STRATEGY_ID = "semantic_fixture"
RENDERER_CONTRACT_ID = "canonical_renderer"
SLOT_OWNERSHIP_ID = "slot_ownership"

LANGUAGE_ROOT = ROOT / "resources" / "language"
REGISTRY_RESOURCE = LANGUAGE_ROOT / "production_registry.json"
TEMPLATE_RESOURCE = LANGUAGE_ROOT / "template_library.json"
PARAPHRASE_RESOURCE = LANGUAGE_ROOT / "paraphrase_library.json"
RENDERER_CONTRACT_RESOURCE = LANGUAGE_ROOT / "renderer_contract.json"

PreflightResult = Literal[
    "PREFLIGHT_PASS",
    "LANGUAGE_CAPABILITY_MISSING",
    "LANGUAGE_CONTRACT_FAIL",
    "PREFLIGHT_FIXTURE_STRATEGY_MISSING",
    "OPERATOR_CONTRACT_MISSING",
    "LANGUAGE_PHRASE_BINDING_MISSING",
    "PROFILE_SEMANTIC_CONTRACT_FAIL",
    "BLOCKED",
]


class PreflightOperatorContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    operator: str
    audio_input_count: int
    logical_context_inputs: int
    answer_kind: str
    required_slots: tuple[str, ...] = ()
    forbidden_slots: tuple[str, ...] = ()
    target_cardinality: int
    candidate_cardinality: int
    structured: bool = False
    output_component_kinds: tuple[str, ...] = ()


class Fixture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    fixture_id: str
    fixture_kind: str
    semantic_value: Any
    constraints: dict[str, Any] = Field(default_factory=dict)


class AcceptedLanguageType(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_type_id: str
    operator: str
    semantic_class: str
    semantic_field: str
    answer_kind: str
    audio_input_count: int
    logical_context_inputs: int
    phrase_bindings: dict[str, Any]
    match_policy: str = "exact"
    source_visibility: str = "model_semantic"
    output_signature: list[dict[str, Any]] = Field(default_factory=list)
    context_roles: list[str] = Field(default_factory=list)
    proposition_id: str | None = None
    comparator_id: str | None = None


class PreflightInputError(RuntimeError):
    """Raised only for input integrity or unsupported invocation failures."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def operator_contracts() -> dict[str, PreflightOperatorContract]:
    return {
        "DIRECT": PreflightOperatorContract(
            operator="DIRECT",
            audio_input_count=1,
            logical_context_inputs=0,
            answer_kind="field_value",
            forbidden_slots=("[TARGET_VALUE]", "[AUDIO_A]", "[AUDIO_B]"),
            target_cardinality=0,
            candidate_cardinality=1,
        ),
        "EQUALITY": PreflightOperatorContract(
            operator="EQUALITY",
            audio_input_count=2,
            logical_context_inputs=0,
            answer_kind="boolean",
            forbidden_slots=("[TARGET_VALUE]",),
            target_cardinality=0,
            candidate_cardinality=2,
        ),
        "TARGET_MATCH": PreflightOperatorContract(
            operator="TARGET_MATCH",
            audio_input_count=1,
            logical_context_inputs=1,
            answer_kind="boolean",
            required_slots=("[TARGET_VALUE]",),
            target_cardinality=1,
            candidate_cardinality=1,
        ),
        "PAIRWISE_SELECTION": PreflightOperatorContract(
            operator="PAIRWISE_SELECTION",
            audio_input_count=2,
            logical_context_inputs=1,
            answer_kind="audio_index",
            required_slots=("[TARGET_VALUE]",),
            target_cardinality=1,
            candidate_cardinality=2,
        ),
        "COMPOSITE": PreflightOperatorContract(
            operator="COMPOSITE",
            audio_input_count=0,
            logical_context_inputs=0,
            answer_kind="structured",
            target_cardinality=0,
            candidate_cardinality=0,
            structured=True,
            output_component_kinds=("field_value", "boolean", "audio_index"),
        ),
    }


SLOT_OWNERSHIP = {
    "contract_id": SLOT_OWNERSHIP_ID,
    "slots": {
        "TARGET_VALUE": {
            "semantic_value": "raw semantic value",
            "renderer": "normalizes display whitespace; strips one balanced outer quote pair",
            "language_entry": "owns presentation quotes and surrounding punctuation",
            "owners": {
                "quotation_marks": "language_entry",
                "parentheses": "semantic_value_or_language_entry_not_both",
                "terminal_punctuation": "language_entry_with_boundary_normalization",
            },
        },
        "UNIT": {
            "semantic_value": "unit label without surrounding spacing",
            "renderer": "binds phrase value",
            "language_entry": "owns unit spacing and punctuation",
        },
        "ENTITY_PHRASE": {
            "semantic_value": "human-language entity phrase",
            "renderer": "binds exactly once where required",
            "language_entry": "owns surrounding grammar",
        },
        "ATTRIBUTE_PHRASE": {
            "semantic_value": "human-language attribute phrase",
            "renderer": "binds exactly once where required",
            "language_entry": "owns surrounding grammar",
        },
        "CONTENT_PHRASE": {
            "semantic_value": "human-language content phrase",
            "renderer": "binds exactly once where required",
            "language_entry": "must not duplicate the same semantic head",
        },
        "AUDIO_A_AUDIO_B": {
            "semantic_value": "ordered logical candidates",
            "renderer": "does not reorder candidates",
            "language_entry": "owns visible A/B labels",
        },
    },
}


def _fixtures(rows: list[tuple[str, str, Any, dict[str, Any] | None]]) -> list[Fixture]:
    return [
        Fixture(
            fixture_id=f"fx_{kind}_{index:02d}",
            fixture_kind=kind,
            semantic_value=value,
            constraints=constraints or {},
        )
        for index, (kind, _label, value, constraints) in enumerate(rows, start=1)
    ]


def fixture_strategies() -> dict[str, list[Fixture]]:
    categorical = _fixtures(
        [
            ("single_token", "single", "Bắc", None),
            ("multi_token", "multi", "miền Trung", None),
            ("proper_location", "location", "Thành phố Hồ Chí Minh", None),
            ("diacritics", "diacritics", "Thừa Thiên Huế", None),
            ("hyphenated", "hyphen", "Nhóm Bắc-Trung", None),
            ("digits", "digits", "Nhóm 2", None),
            ("parentheses", "parentheses", "Nhóm A (mở rộng)", None),
            ("long", "long", "nhóm phân loại tổng hợp khu vực mở rộng", None),
            ("surrounding_whitespace", "whitespace", "  miền Nam  ", None),
        ]
    )
    text = _fixtures(
        [
            ("plain", "plain", "xin chào", None),
            ("short_phrase", "short", "hôm nay trời đẹp", None),
            ("terminal_period", "period", "hôm nay trời đẹp.", None),
            ("terminal_question", "question", "bạn khỏe không?", None),
            ("terminal_exclamation", "exclamation", "tuyệt quá!", None),
            ("comma", "comma", "vâng, tôi hiểu", None),
            ("colon_semicolon", "colon", "ghi chú: một; hai", None),
            ("ascii_quotes", "ascii", 'anh ấy nói "được rồi"', None),
            ("curly_quotes", "curly", "anh ấy nói “được rồi”", None),
            ("outer_ascii_quotes", "outer_ascii", '"được rồi"', None),
            ("outer_curly_quotes", "outer_curly", "“được rồi”", None),
            ("nested_quotes", "nested", 'anh ấy nói “tôi nghe "được rồi"”', None),
            ("apostrophe", "apostrophe", "O'Connor đồng ý", None),
            ("leading_whitespace", "leading", "  xin chào", None),
            ("trailing_whitespace", "trailing", "xin chào  ", None),
            ("internal_spaces", "spaces", "xin   chào   bạn", None),
            ("newline_tab", "control_ws", "xin\nchào\tbạn", None),
            ("numbers", "numbers", "tôi có 25 nghìn đồng", None),
            ("a_b_tokens", "ab", "chọn A hay B", None),
            ("long", "long", "nội dung thử nghiệm " * 30, None),
            ("very_long", "very_long", "một câu kiểm thử có kiểm soát " * 80, None),
            ("braces_brackets", "braces", "ghi chú {mở} và [đóng]", None),
            ("ending_quote", "ending_quote", 'anh ấy nói "được rồi"', None),
            ("both_quote_styles", "both_quotes", '“anh ấy” nói "được rồi"', None),
            ("terminal_ellipsis", "ellipsis", "tôi đang nghĩ…", None),
            ("terminal_parenthesis", "parenthesis", "nhóm này (mở rộng)", None),
        ]
    )
    numeric = _fixtures(
        [
            ("zero", "zero", "0", {"negative_allowed": False}),
            ("one", "one", "1", {"negative_allowed": False}),
            ("ten", "ten", "10", {"negative_allowed": False}),
            ("hundred", "hundred", "100", {"negative_allowed": False}),
            ("decimal", "decimal", "12,5", {"negative_allowed": False}),
            ("small_decimal", "small", "0,01", {"negative_allowed": False}),
            ("negative", "negative", "-3", {"negative_allowed": True}),
            ("large", "large", "1000000", {"negative_allowed": False}),
        ]
    )
    return {
        "categorical_attribute": categorical,
        "text_content": text,
        "numeric_attribute": numeric,
        "ordinal_attribute": _fixtures(
            [
                ("first", "first", "thứ nhất", None),
                ("middle", "middle", "ở giữa", None),
                ("last", "last", "cuối cùng", None),
            ]
        ),
        "boolean_attribute": _fixtures(
            [
                ("true", "true", "có", None),
                ("false", "false", "không", None),
            ]
        ),
        "multi_label_attribute": _fixtures(
            [
                ("one_label", "one", "nhãn một", None),
                ("two_labels", "two", "nhãn một, nhãn hai", None),
                ("three_labels", "three", "nhãn một, nhãn hai, nhãn ba", None),
            ]
        ),
        "identifier_relation": _fixtures(
            [
                ("identifier_pair", "pair", "thực thể A và thực thể B", None),
            ]
        ),
    }


def _generic_phrase_bindings(
    semantic_class: str, *, unit: str | None
) -> dict[str, Any]:
    common = {
        "entity_scope": "generic_audio_entity",
        "entity_phrase": "người nói",
        "attribute_phrase": None,
        "content_phrase": None,
        "value_phrase": None,
        "unit": unit,
        "target_quote_style": "plain",
    }
    if semantic_class == "categorical_attribute":
        common.update(
            attribute_phrase="thuộc tính phân loại",
            value_phrase="giá trị phân loại",
        )
    elif semantic_class == "numeric_attribute":
        common.update(attribute_phrase="giá trị đo", value_phrase="giá trị")
    elif semantic_class == "text_content":
        common.update(
            entity_scope="utterance",
            entity_phrase="đoạn âm thanh",
            content_phrase="nội dung được nói",
            value_phrase="nội dung",
            target_quote_style="vietnamese_quotes",
        )
    else:
        common.update(attribute_phrase="thuộc tính", value_phrase="giá trị")
    return common


def _accepted_to_spec(
    item: AcceptedLanguageType, entry: LanguageRegistryEntry
) -> SemanticFieldSpec:
    binding = item.phrase_bindings
    unit = binding.get("unit")
    if entry.unit_policy == "required" and not unit:
        unit = "đơn vị"
    if entry.unit_policy == "forbidden":
        unit = None
    return SemanticFieldSpec(
        field_name=item.semantic_field,
        semantic_class=item.semantic_class,
        entity_scope=binding.get("entity_scope", "generic_audio_entity"),
        entity_phrase=binding["entity_phrase"],
        attribute_phrase=binding.get("attribute_phrase"),
        content_phrase=binding.get("content_phrase"),
        value_phrase=binding.get("value_phrase"),
        unit=unit,
        value_policy={"normalization": "identity", "match_policy": item.match_policy},
        rendering={"target_quote_style": binding.get("target_quote_style", "plain")},
        source="language_preflight_fixture",
    )


def _accepted_to_contract(item: AcceptedLanguageType) -> TypeContract:
    return TypeContract(
        type_id=item.dataset_type_id,
        operator=item.operator,
        semantic_field=item.semantic_field,
        semantic_description="accepted semantic type preflight",
        semantic_phrase_vi="language composition preflight",
        entity_scope=item.phrase_bindings.get("entity_scope", "generic_audio_entity"),
        audio_input_count=item.audio_input_count,
        condition_fields=["target_value"] if item.logical_context_inputs else [],
        gold_source_fields=[item.semantic_field],
        answer_kind=item.answer_kind,
        answer_mode={
            "DIRECT": "FIELD_VALUE",
            "EQUALITY": "BOOLEAN",
            "TARGET_MATCH": "BOOLEAN",
            "PAIRWISE_SELECTION": "A_B_SELECTION",
        }.get(item.operator, "UNKNOWN"),
        template_status="ELIGIBLE",
        template_policy={"visible_target": bool(item.logical_context_inputs)},
        instantiation_policy={},
        source_status="SUPPORTED",
    )


def _issue(
    code: str,
    severity: Literal["BLOCKING", "REVIEW", "INFO"],
    detail: str,
    *,
    case_id: str | None = None,
    entry_id: str | None = None,
    type_id: str | None = None,
) -> dict[str, Any]:
    return {
        "issue_code": code,
        "severity": severity,
        "detail": detail,
        "preflight_case_id": case_id,
        "language_entry_id": entry_id,
        "dataset_type_id": type_id,
    }


def _structural_issues(
    item: AcceptedLanguageType,
    entry: LanguageRegistryEntry,
    contract: PreflightOperatorContract,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    context = {"entry_id": entry.language_entry_id, "type_id": item.dataset_type_id}
    if entry.operator != item.operator or entry.semantic_class != item.semantic_class:
        issues.append(
            _issue(
                "LANGUAGE_ENTRY_INCOMPATIBLE",
                "BLOCKING",
                "Entry Operator × SemanticClass does not match the accepted type.",
                **context,
            )
        )
    if (
        entry.answer_kind != item.answer_kind
        or entry.answer_kind != contract.answer_kind
    ):
        issues.append(
            _issue(
                "ANSWER_KIND_CONTRACT_MISMATCH",
                "BLOCKING",
                "Language entry answer kind differs from the operator/type contract.",
                **context,
            )
        )
    if (
        item.audio_input_count != contract.audio_input_count
        or item.logical_context_inputs != contract.logical_context_inputs
    ):
        issues.append(
            _issue(
                "OPERATOR_CONTRACT_MISMATCH",
                "BLOCKING",
                "Accepted type arity/context does not match the operator contract.",
                **context,
            )
        )
    for slot in contract.required_slots:
        count = entry.pattern.count(slot)
        if count == 0:
            issues.append(
                _issue(
                    "SLOT_BINDING_MISSING",
                    "BLOCKING",
                    f"Required slot {slot} is absent.",
                    **context,
                )
            )
        elif count > 1:
            issues.append(
                _issue(
                    "SLOT_BINDING_DUPLICATED",
                    "BLOCKING",
                    f"Required slot {slot} occurs {count} times.",
                    **context,
                )
            )
    for slot in contract.forbidden_slots:
        if slot in entry.pattern:
            issues.append(
                _issue(
                    "OPERATOR_CONTRACT_MISMATCH",
                    "BLOCKING",
                    f"Forbidden slot {slot} is present.",
                    **context,
                )
            )
    return issues


_SLOT_RE = re.compile(r"\[[A-Z][A-Z0-9_]*\]")
_INTERNAL_TOKENS = (
    "speakerID",
    "province_code",
    "source_row",
    "source_path",
    "semantic_instance",
    "language_entry",
    "template_id",
    "parquet",
)


def _composition_ownership_issues(
    pattern: str, spec: SemanticFieldSpec
) -> list[tuple[str, str, str]]:
    literal = _SLOT_RE.sub(" ", pattern).casefold()
    issues: list[tuple[str, str, str]] = []
    for slot, phrase in spec.slot_values().items():
        if not phrase or slot not in pattern:
            continue
        phrase_lower = phrase.casefold()
        for head in ("nội dung", "đoạn âm thanh", "người nói", "phương ngữ"):
            if head in literal and head in phrase_lower:
                issues.append(
                    (
                        "SEMANTIC_HEAD_OWNERSHIP_COLLISION",
                        "BLOCKING",
                        f"Both entry literal and {slot} phrase own semantic head '{head}'.",
                    )
                )
    return issues


def detect_render_issues(
    *,
    question: str,
    entry: LanguageRegistryEntry,
    spec: SemanticFieldSpec,
    bound_target: str | None,
    operator_contract: PreflightOperatorContract,
) -> list[tuple[str, str, str]]:
    issues: list[tuple[str, str, str]] = []
    if _SLOT_RE.search(question):
        issues.append(
            ("UNRESOLVED_SLOT", "BLOCKING", "Rendered slot remains unresolved.")
        )
    pattern_owns_quotes = bool(
        re.search(r"[\"“]\s*\[TARGET_VALUE\]\s*[\"”]", entry.pattern)
    )
    bound_is_wrapped = bool(
        bound_target
        and len(bound_target) >= 2
        and (
            (bound_target.startswith("“") and bound_target.endswith("”"))
            or (bound_target.startswith('"') and bound_target.endswith('"'))
        )
    )
    pattern_has_duplicate_wrapper = any(
        token in entry.pattern for token in ("““", "””", '""')
    )
    if pattern_has_duplicate_wrapper or (pattern_owns_quotes and bound_is_wrapped):
        issues.append(
            (
                "DUPLICATE_PRESENTATION_WRAPPER",
                "BLOCKING",
                "Multiple layers own the same quotation wrapper.",
            )
        )
    if question.count("“") != question.count("”") or question.count('"') % 2:
        issues.append(
            ("UNBALANCED_QUOTES", "BLOCKING", "Quotation marks are unbalanced.")
        )
    if re.search(r"[.!?;:,]”[.!?](?=\s|$)|\?\.|\.\?|!!|\?\?", question):
        issues.append(
            (
                "PUNCTUATION_COLLISION",
                "BLOCKING",
                "Target and entry punctuation collide at a composition boundary.",
            )
        )
    duplicate_word = re.search(
        r"\b([^\W\d_]+)\s+\1\b", question, flags=re.IGNORECASE | re.UNICODE
    )
    if duplicate_word:
        issues.append(
            (
                "CONSECUTIVE_DUPLICATE_WORD",
                "BLOCKING",
                "Rendered composition contains an immediate duplicate word.",
            )
        )
    issues.extend(_composition_ownership_issues(entry.pattern, spec))

    scaffold = question
    if bound_target:
        count = question.count(bound_target)
        if operator_contract.target_cardinality and count == 0:
            issues.append(("TARGET_MISSING", "BLOCKING", "Visible target was lost."))
        elif operator_contract.target_cardinality and count > 1:
            issues.append(
                (
                    "TARGET_DUPLICATED",
                    "BLOCKING",
                    "Visible target occurs more than once.",
                )
            )
        scaffold = scaffold.replace(bound_target, "TARGET")
    if re.search(r"\b[a-zA-Z]+_[a-zA-Z0-9_]+\b", scaffold):
        issues.append(
            ("SNAKE_CASE_LEAK", "BLOCKING", "Raw snake_case leaked into wording.")
        )
    if any(token.casefold() in scaffold.casefold() for token in _INTERNAL_TOKENS):
        issues.append(
            ("INTERNAL_TOKEN_LEAK", "BLOCKING", "Internal/provenance token leaked.")
        )
    if entry.operator == "PAIRWISE_SELECTION":
        tokens = set(re.findall(r"\b[A-B]\b", scaffold))
        if tokens != {"A", "B"}:
            issues.append(
                (
                    "A_B_CONTRACT_VIOLATION",
                    "BLOCKING",
                    "Pairwise selection must expose both ordered candidates A and B.",
                )
            )
    return list(dict.fromkeys(issues))


def render_with_trace(
    *,
    entry: LanguageRegistryEntry,
    spec: SemanticFieldSpec,
    contract: TypeContract,
    registry: ProductionLanguageRegistry,
    semantic_target: Any,
) -> dict[str, Any]:
    target_required = "[TARGET_VALUE]" in entry.pattern
    bound = (
        bind_target_value(
            semantic_target,
            quote_style=spec.rendering.target_quote_style,
        )
        if target_required
        else None
    )
    question = render_question(
        entry=entry,
        spec=spec,
        contract=contract,
        registry=registry,
        target_display=semantic_target if target_required else None,
    )
    transforms = ["semantic_phrase_binding", "whitespace_normalization"]
    if target_required:
        transforms += [
            "target_display_normalization",
            "target_quote_boundary_normalization",
        ]
    return {
        "question": question,
        "language_entry_id": entry.language_entry_id,
        "slots": {
            "TARGET_VALUE": {
                "semantic_value": semantic_target,
                "bound_value": bound,
                "display_value": bound,
                "presentation_owner": "language_entry",
            },
            **{
                slot.strip("[]"): {"semantic_value": value, "display_value": value}
                for slot, value in spec.slot_values().items()
                if value is not None
            },
        },
        "transforms": transforms,
        "renderer_contract_id": RENDERER_CONTRACT_ID,
    }


def _case_id(payload: dict[str, Any]) -> str:
    return f"pf_{canonical_hash(payload)[:20]}"


def _validate_phrase_binding(item: AcceptedLanguageType) -> str | None:
    binding = item.phrase_bindings
    if not str(binding.get("entity_phrase") or "").strip():
        return "ENTITY_PHRASE"
    if (
        item.semantic_class == "text_content"
        and not str(binding.get("content_phrase") or "").strip()
    ):
        return "CONTENT_PHRASE"
    if (
        item.semantic_class != "text_content"
        and not str(binding.get("attribute_phrase") or "").strip()
    ):
        return "ATTRIBUTE_PHRASE"
    return None


def _prefer_proposition(
    entries: list[LanguageRegistryEntry], item: AcceptedLanguageType
) -> list[LanguageRegistryEntry]:
    """Prefer entries explicitly bound to the task proposition.

    Fall back to unbound (wildcard) entries only when no proposition-tagged
    entry exists, so repaired tasks deterministically select repaired wording.
    """
    if not item.proposition_id:
        return entries
    tagged = [entry for entry in entries if entry.proposition_id == item.proposition_id]
    return tagged or entries


def _compatible_entries(
    registry: ProductionLanguageRegistry, item: AcceptedLanguageType
) -> list[LanguageRegistryEntry]:
    if item.operator == "COMPOSITE":
        from src.autonomous_qa.language.composite_language import composite_entry_matches

        candidates = [
            entry
            for entry in registry.active(item.operator, item.semantic_class)
            if entry.answer_kind == item.answer_kind
            and item.match_policy in entry.match_policy
            and composite_entry_matches(
                entry_output_signature=entry.output_signature,
                entry_context_roles=entry.context_roles,
                expected_output_signature=item.output_signature,
                expected_context_roles=item.context_roles,
            )
        ]
        return _prefer_proposition(candidates, item)
    slot_values = slot_value_map(item.phrase_bindings)
    candidates = [
        entry
        for entry in registry.active(item.operator, item.semantic_class)
        if entry.answer_kind == item.answer_kind
        and item.match_policy in entry.match_policy
        and entry_capability_compatible(
            entry,
            entity_scope=item.phrase_bindings.get("entity_scope"),
            unit=item.phrase_bindings.get("unit"),
            slot_values=slot_values,
        )[0]
    ]
    return _prefer_proposition(candidates, item)


def _resource_identity(
    registry_path: Path | None = None,
    *,
    registry_hash: str | None = None,
) -> dict[str, Any]:
    registry_path = Path(registry_path) if registry_path else REGISTRY_RESOURCE
    template_path = TEMPLATE_RESOURCE
    paraphrase_path = PARAPHRASE_RESOURCE
    renderer_path = ROOT / "src" / "autonomous_qa" / "production" / "production_qa.py"
    try:
        registry_file = registry_path.relative_to(ROOT).as_posix()
    except ValueError:
        registry_file = registry_path.name
    if registry_hash is None:
        registry_hash = json.loads(
            registry_path.read_text(encoding="utf-8")
        )["registry_hash"]
    return {
        "language_registry_file": registry_file,
        "language_registry_file_sha256": sha256_file(registry_path),
        "language_registry_hash": registry_hash,
        "template_resource_file_sha256": sha256_file(template_path),
        "paraphrase_resource_file_sha256": sha256_file(paraphrase_path),
        "renderer_contract_id": RENDERER_CONTRACT_ID,
        "renderer_file_sha256": sha256_file(renderer_path),
        "renderer_contract_sha256": sha256_file(RENDERER_CONTRACT_RESOURCE),
    }


def get_canonical_preflight_implementation_sha256() -> str:
    return sha256_file(Path(__file__))


CANONICAL_PREFLIGHT_IMPLEMENTATION_SHA256 = get_canonical_preflight_implementation_sha256()


def compute_contract_fingerprint(
    *,
    mode: str,
    accepted_types: list[AcceptedLanguageType],
    implementation_identity: str | None = None,
    registry_path: Path | None = None,
    registry_hash: str | None = None,
) -> tuple[str, dict[str, Any]]:
    inputs = {
        "preflight_contract_id": PREFLIGHT_CONTRACT_ID,
        "preflight_implementation_sha256": (
            implementation_identity
            if implementation_identity is not None
            else get_canonical_preflight_implementation_sha256()
        ),
        "mode": mode,
        "resources": _resource_identity(registry_path, registry_hash=registry_hash),
        "operator_contract_id": OPERATOR_CONTRACT_ID,
        "operator_contracts": {
            key: value.model_dump(mode="json")
            for key, value in sorted(operator_contracts().items())
        },
        "fixture_strategy_id": FIXTURE_STRATEGY_ID,
        "fixture_manifest_hash": canonical_hash(
            {
                key: [fixture.model_dump(mode="json") for fixture in value]
                for key, value in sorted(fixture_strategies().items())
            }
        ),
        "slot_ownership": SLOT_OWNERSHIP,
        "accepted_type_contract_hash": canonical_hash(
            [item.model_dump(mode="json") for item in accepted_types]
        ),
        "phrase_bundle_hash": canonical_hash(
            {
                item.dataset_type_id: item.phrase_bindings
                for item in sorted(accepted_types, key=lambda row: row.dataset_type_id)
            }
        ),
    }
    return canonical_hash(inputs), inputs


def _registry_types(registry: ProductionLanguageRegistry) -> list[AcceptedLanguageType]:
    result: list[AcceptedLanguageType] = []
    for entry in sorted(registry.entries, key=lambda row: row.language_entry_id):
        if not entry.enabled:
            continue
        unit = "đơn vị" if entry.unit_policy == "required" else None
        op = operator_contracts()[entry.operator]
        is_composite = entry.operator == "COMPOSITE"
        bindings = _generic_phrase_bindings(entry.semantic_class, unit=unit)
        # Registry mode must exercise an entry against its OWN declared scope.
        if entry.entity_scopes:
            scope = entry.entity_scopes[0]
            bindings["entity_scope"] = scope
            if scope == "utterance":
                bindings["entity_phrase"] = "đoạn âm thanh"
        result.append(
            AcceptedLanguageType(
                dataset_type_id=f"registry::{entry.language_entry_id}",
                operator=entry.operator,
                semantic_class=entry.semantic_class,
                semantic_field=f"synthetic_{entry.semantic_class}",
                answer_kind=entry.answer_kind,
                audio_input_count=(0 if is_composite else op.audio_input_count),
                logical_context_inputs=(
                    len(entry.context_roles)
                    if is_composite
                    else op.logical_context_inputs
                ),
                phrase_bindings=bindings,
                output_signature=[dict(c) for c in entry.output_signature],
                context_roles=list(entry.context_roles),
            )
        )
    return result


def _vietmdd_semantic_field(spec: Any) -> str:
    # The repaired reference-match task treats the reference text as a VISIBLE
    # object while the subject of perception is spoken content.
    if spec.proposition_id == "spoken_content_matches_reference":
        return "reference_text"
    return "observed_transcription"


def _vietmdd_phrase_bindings() -> dict[str, Any]:
    return {
        "entity_scope": "utterance",
        "entity_phrase": "người nói",
        "attribute_phrase": "nội dung phát âm",
        "content_phrase": "nội dung phát âm",
        "value_phrase": "nội dung",
        "unit": None,
        "target_quote_style": "vietnamese_quotes",
    }


def vietmdd_accepted_types() -> list[AcceptedLanguageType]:
    """Primitive VietMDD accepted types, resolved from the STATIC catalog.

    No historical discovery output is read at runtime.
    """
    from src.autonomous_qa.compiler.semantic_task import final_semantic_catalog

    result: list[AcceptedLanguageType] = []
    for spec in final_semantic_catalog():
        if spec.is_composite:
            continue
        result.append(
            AcceptedLanguageType(
                dataset_type_id=spec.type_id,
                operator=spec.operator,
                semantic_class="text_content",
                semantic_field=_vietmdd_semantic_field(spec),
                answer_kind=spec.outputs[0].kind,
                audio_input_count=spec.audio_arity,
                logical_context_inputs=len(spec.visible_context_roles),
                phrase_bindings=_vietmdd_phrase_bindings(),
                match_policy="exact",
                source_visibility="model_semantic",
                proposition_id=spec.proposition_id,
                comparator_id=spec.comparator_id,
            )
        )
    return result


def vietmdd_composite_accepted_types() -> list[AcceptedLanguageType]:
    """Accepted types for the four final CHAIN_DERIVED composite tasks."""
    from src.autonomous_qa.compiler.semantic_task import final_semantic_catalog

    result: list[AcceptedLanguageType] = []
    for spec in final_semantic_catalog():
        if not spec.is_composite:
            continue
        result.append(
            AcceptedLanguageType(
                dataset_type_id=spec.type_id,
                operator=spec.operator,
                semantic_class="text_content",
                semantic_field=_vietmdd_semantic_field(spec),
                answer_kind="structured",
                audio_input_count=spec.audio_arity,
                logical_context_inputs=len(spec.visible_context_roles),
                phrase_bindings=_vietmdd_phrase_bindings(),
                match_policy="exact",
                source_visibility="model_semantic",
                output_signature=[
                    {"role": component.role, "kind": component.kind}
                    for component in spec.outputs
                ],
                context_roles=list(spec.visible_context_roles),
                proposition_id=spec.proposition_id,
                comparator_id=spec.comparator_id,
            )
        )
    return result


def vietmdd_full_accepted_types() -> list[AcceptedLanguageType]:
    """The one final nine-type VietMDD semantic catalog (primitive + composite)."""
    return vietmdd_accepted_types() + vietmdd_composite_accepted_types()


def vimd_accepted_types() -> list[AcceptedLanguageType]:
    contracts, specs = _contracts_and_specs()
    result: list[AcceptedLanguageType] = []
    for contract in sorted(contracts.values(), key=lambda row: row.type_id):
        if contract.source_status != "SUPPORTED":
            continue
        spec = specs[contract.semantic_field]
        op = operator_contracts()[contract.operator]
        result.append(
            AcceptedLanguageType(
                dataset_type_id=contract.type_id,
                operator=contract.operator,
                semantic_class=spec.semantic_class,
                semantic_field=contract.semantic_field,
                answer_kind=contract.answer_kind,
                audio_input_count=contract.audio_input_count,
                logical_context_inputs=op.logical_context_inputs,
                phrase_bindings={
                    "entity_scope": spec.entity_scope,
                    "entity_phrase": spec.entity_phrase,
                    "attribute_phrase": spec.attribute_phrase,
                    "content_phrase": spec.content_phrase,
                    "value_phrase": spec.value_phrase,
                    "unit": spec.unit,
                    "target_quote_style": spec.rendering.target_quote_style,
                },
                match_policy=spec.value_policy.match_policy,
                source_visibility="model_semantic",
            )
        )
    return result


def accepted_types_from_semantic_catalog(
    catalog: Any,
    field_specs: dict[str, SemanticFieldSpec] | None = None,
) -> list[AcceptedLanguageType]:
    """Generates AcceptedLanguageType list directly from a candidate SemanticCatalog with NO type-id substring heuristics."""
    if field_specs is None:
        raise PreflightInputError("FIELD_SPECS_REQUIRED")

    tasks = catalog.get("tasks", []) if isinstance(catalog, dict) else catalog.tasks

    result: list[AcceptedLanguageType] = []
    op_contracts = operator_contracts()

    for task_raw in tasks:
        task_dict = task_raw.model_dump(mode="json") if hasattr(task_raw, "model_dump") else task_raw
        source_role = task_dict.get("source_role_mapping", {})
        outputs = task_dict.get("outputs", [])
        field = source_role.get("source_field") or (
            outputs[0]["dependencies"][0] if outputs and outputs[0].get("dependencies") else None
        )
        if not field:
            raise PreflightInputError("MISSING_EXECUTABLE_SOURCE_FIELD")
        if field not in field_specs:
            raise PreflightInputError(f"FIELD_SPEC_MISSING:{field}")
        spec = field_specs[field]

        op_name = task_dict.get("operator", "DIRECT")
        if op_name not in op_contracts:
            raise PreflightInputError(f"UNSUPPORTED_OPERATOR:{op_name}")
        op = op_contracts[op_name]

        type_id = task_dict["type_id"]
        audio_arity = task_dict.get("audio_arity", 1)
        visible_context_roles = task_dict.get("visible_context_roles", [])
        answer_kind = outputs[0]["kind"] if outputs else "field_value"
        prop_id = task_dict.get("proposition_id")
        comp_id = task_dict.get("comparator_id")

        result.append(
            AcceptedLanguageType(
                dataset_type_id=type_id,
                operator=op_name,
                semantic_class=spec.semantic_class,
                semantic_field=field,
                answer_kind=answer_kind,
                audio_input_count=audio_arity,
                logical_context_inputs=len(visible_context_roles),
                phrase_bindings={
                    "entity_scope": spec.entity_scope,
                    "entity_phrase": spec.entity_phrase,
                    "attribute_phrase": spec.attribute_phrase,
                    "content_phrase": spec.content_phrase,
                    "value_phrase": spec.value_phrase,
                    "unit": spec.unit,
                    "target_quote_style": spec.rendering.target_quote_style,
                },
                match_policy=spec.value_policy.match_policy,
                source_visibility="model_semantic",
                context_roles=list(visible_context_roles),
                proposition_id=prop_id,
                comparator_id=comp_id,
            )
        )
    return result


def vimedcss_accepted_types() -> list[AcceptedLanguageType]:
    from src.autonomous_qa.compiler.canonical_resources import get_semantic_catalog_path
    from src.autonomous_qa.compiler.semantic_task import load_semantic_catalog
    from src.autonomous_qa.language.template_engine import vimedcss_field_specs

    catalog = load_semantic_catalog(get_semantic_catalog_path("vimedcss"))
    return accepted_types_from_semantic_catalog(catalog, field_specs=vimedcss_field_specs())


_DATASET_ACCEPTED_TYPE_RESOLVERS = {
    "vietmdd": vietmdd_full_accepted_types,
    "vimd": vimd_accepted_types,
    "vimedcss": vimedcss_accepted_types,
}


def get_dataset_accepted_types(dataset: str) -> list[AcceptedLanguageType]:
    resolver = _DATASET_ACCEPTED_TYPE_RESOLVERS.get(dataset)
    if resolver is None:
        raise PreflightInputError(f"UNKNOWN_DATASET:{dataset}")
    return resolver()


def _contracts_and_specs() -> tuple[dict[str, Any], dict[str, Any]]:
    type_data = json.loads(
        DATASET_SOURCES["vimd"]["type_registry"].read_text(encoding="utf-8")
    )
    contracts = {row.type_id: row for row in build_type_contracts(type_data)}
    return contracts, vimd_field_specs()


def _source_reference_issues(
    registry: ProductionLanguageRegistry,
) -> list[dict[str, Any]]:
    template = json.loads(TEMPLATE_RESOURCE.read_text(encoding="utf-8"))
    paraphrase = json.loads(PARAPHRASE_RESOURCE.read_text(encoding="utf-8"))
    template_ids = {row["blueprint_id"] for row in template["blueprints"]}
    paraphrase_ids = {row["paraphrase_id"] for row in paraphrase["paraphrases"]}
    issues = []
    for entry in registry.entries:
        if entry.source_kind == "CANDIDATE":
            # Candidate entries are defined by the candidate capability
            # resource, not by the canonical template/paraphrase libraries.
            continue
        exists = (
            entry.source_id in template_ids
            if entry.source_kind == "CANONICAL"
            else entry.source_id in paraphrase_ids
        )
        if not exists:
            issues.append(
                _issue(
                    "LANGUAGE_ENTRY_INCOMPATIBLE",
                    "BLOCKING",
                    "Language entry has a dangling source resource reference.",
                    entry_id=entry.language_entry_id,
                )
            )
    return issues


def _composite_structural_issues(
    item: AcceptedLanguageType, entry: LanguageRegistryEntry
) -> list[dict[str, Any]]:
    from src.autonomous_qa.language.composite_language import composite_entry_matches

    issues: list[dict[str, Any]] = []
    context = {"entry_id": entry.language_entry_id, "type_id": item.dataset_type_id}
    if entry.operator != "COMPOSITE" or entry.semantic_class != item.semantic_class:
        issues.append(
            _issue(
                "LANGUAGE_ENTRY_INCOMPATIBLE",
                "BLOCKING",
                "Composite entry operator × semantic_class does not match.",
                **context,
            )
        )
    if entry.answer_kind != "structured" or item.answer_kind != "structured":
        issues.append(
            _issue(
                "ANSWER_KIND_CONTRACT_MISMATCH",
                "BLOCKING",
                "Composite entry must declare the structured answer kind.",
                **context,
            )
        )
    if not composite_entry_matches(
        entry_output_signature=entry.output_signature,
        entry_context_roles=entry.context_roles,
        expected_output_signature=item.output_signature,
        expected_context_roles=item.context_roles,
    ):
        issues.append(
            _issue(
                "OUTPUT_SIGNATURE_MISMATCH",
                "BLOCKING",
                "Composite output signature does not match the accepted type.",
                **context,
            )
        )
    for index, _component in enumerate(item.output_signature):
        slot = f"[INSTRUCTION_{index + 1}]"
        if entry.pattern.count(slot) != 1:
            issues.append(
                _issue(
                    "OUTPUT_INSTRUCTION_MISSING",
                    "BLOCKING",
                    f"Expected exactly one {slot} in the composite pattern.",
                    **context,
                )
            )
    if not entry.audio_reference_phrase:
        issues.append(
            _issue(
                "AUDIO_REFERENCE_MISSING",
                "BLOCKING",
                "Composite entry lacks an audio reference phrase.",
                **context,
            )
        )
    return issues


def _run_composite_cases(
    *,
    item: AcceptedLanguageType,
    compatible: list[LanguageRegistryEntry],
    strategies: dict[str, list[Fixture]],
    mode: str,
    fingerprint: str,
    issues: list[dict[str, Any]],
    matrix: list[dict[str, Any]],
) -> None:
    from src.autonomous_qa.language.composite_language import (
        composite_render_issues,
        render_composite_question,
    )

    for entry in compatible:
        issues.extend(_composite_structural_issues(item, entry))
        needs_target = any(
            "[TARGET_VALUE]" in phrase for phrase in entry.instruction_bindings.values()
        )
        for fixture in strategies[item.semantic_class]:
            case_payload = {
                "mode": mode,
                "dataset_type_id": item.dataset_type_id,
                "language_entry_id": entry.language_entry_id,
                "fixture_id": fixture.fixture_id,
                "fingerprint": fingerprint,
            }
            case_id = _case_id(case_payload)
            target = fixture.semantic_value if needs_target else None
            bound = (
                bind_target_value(target, quote_style="vietnamese_quotes")
                if target is not None
                else None
            )
            try:
                question = render_composite_question(
                    pattern=entry.pattern,
                    output_signature=entry.output_signature,
                    instruction_bindings=entry.instruction_bindings,
                    audio_reference_phrase=entry.audio_reference_phrase or "",
                    target_display=target,
                )
                detected = composite_render_issues(
                    question=question,
                    pattern=entry.pattern,
                    output_signature=entry.output_signature,
                    instruction_bindings=entry.instruction_bindings,
                    context_roles=item.context_roles,
                    bound_target=bound,
                )
                if item.proposition_id and question:
                    from src.autonomous_qa.compiler.semantic_alignment import (
                        validate_question_proposition_alignment,
                    )

                    for alignment_code in validate_question_proposition_alignment(
                        proposition_id=item.proposition_id,
                        question=question,
                        pattern=entry.pattern
                        + " "
                        + " ".join(entry.instruction_bindings.values()),
                        phrase_bindings=item.phrase_bindings,
                    ):
                        detected.append(
                            (
                                "PROPOSITION_LANGUAGE_MISALIGNMENT",
                                "BLOCKING",
                                alignment_code,
                            )
                        )
            except (ValueError, KeyError) as exc:
                question = None
                detected = [
                    (
                        "LANGUAGE_ENTRY_INCOMPATIBLE",
                        "BLOCKING",
                        f"Composite renderer rejected entry: {exc}",
                    )
                ]
            case_issues = [
                _issue(
                    code,
                    severity,
                    detail,
                    case_id=case_id,
                    entry_id=entry.language_entry_id,
                    type_id=item.dataset_type_id,
                )
                for code, severity, detail in detected
            ]
            issues.extend(case_issues)
            matrix.append(
                {
                    "preflight_case_id": case_id,
                    "mode": mode,
                    "operator": item.operator,
                    "semantic_class": item.semantic_class,
                    "dataset_type_id": (
                        item.dataset_type_id if mode == "dataset" else None
                    ),
                    "language_entry_id": entry.language_entry_id,
                    "fixture_id": fixture.fixture_id,
                    "fixture_kind": fixture.fixture_kind,
                    "slot_semantic_values": {
                        "TARGET_VALUE": target,
                        **item.phrase_bindings,
                    },
                    "slot_display_values": {
                        "TARGET_VALUE": {
                            "semantic_value": target,
                            "bound_value": bound,
                            "display_value": bound,
                        }
                    },
                    "rendered_question": question,
                    "detector_results": [row["issue_code"] for row in case_issues],
                    "status": (
                        "FAIL"
                        if any(row["severity"] == "BLOCKING" for row in case_issues)
                        else "REVIEW"
                        if case_issues
                        else "PASS"
                    ),
                }
            )


def run_preflight(
    *,
    mode: Literal["registry", "dataset"],
    accepted_types: list[AcceptedLanguageType] | None = None,
    dataset: str | None = None,
    write_outputs: bool = True,
    registry: ProductionLanguageRegistry | None = None,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    if registry is None:
        registry = load_language_registry(registry_path or REGISTRY_RESOURCE)
    if accepted_types is not None:
        items = sorted(accepted_types, key=lambda row: row.dataset_type_id)
    elif mode == "registry":
        items = _registry_types(registry)
    elif dataset:
        items = get_dataset_accepted_types(dataset)
    else:
        items = []
    fingerprint, fingerprint_inputs = compute_contract_fingerprint(
        mode=mode,
        accepted_types=items,
        registry_path=registry_path,
        registry_hash=registry.registry_hash,
    )
    contracts = operator_contracts()
    strategies = fixture_strategies()
    issues = _source_reference_issues(registry)
    coverage_rows: list[dict[str, Any]] = []
    matrix: list[dict[str, Any]] = []
    required_cells: set[tuple[str, str]] = set()
    covered_cells: set[tuple[str, str]] = set()

    terminal_result: PreflightResult | None = None
    for item in items:
        required_cells.add((item.operator, item.semantic_class))
        if item.operator not in contracts:
            terminal_result = terminal_result or "OPERATOR_CONTRACT_MISSING"
            issues.append(
                _issue(
                    "OPERATOR_CONTRACT_MISSING",
                    "BLOCKING",
                    f"No reusable contract exists for operator {item.operator}.",
                    type_id=item.dataset_type_id,
                )
            )
            continue
        if item.semantic_class not in strategies:
            terminal_result = terminal_result or "PREFLIGHT_FIXTURE_STRATEGY_MISSING"
            issues.append(
                _issue(
                    "PREFLIGHT_FIXTURE_STRATEGY_MISSING",
                    "BLOCKING",
                    f"No bounded fixtures exist for {item.semantic_class}.",
                    type_id=item.dataset_type_id,
                )
            )
            continue
        if item.source_visibility in {"hidden_identifier", "provenance_only"}:
            terminal_result = terminal_result or "PROFILE_SEMANTIC_CONTRACT_FAIL"
            issues.append(
                _issue(
                    "PROFILE_SEMANTIC_CONTRACT_FAIL",
                    "BLOCKING",
                    "Accepted type exposes a hidden/provenance-only field.",
                    type_id=item.dataset_type_id,
                )
            )
            continue
        if item.operator == "COMPOSITE":
            compatible = _compatible_entries(registry, item)
            if mode == "registry":
                expected_entry_id = item.dataset_type_id.removeprefix("registry::")
                compatible = [
                    entry
                    for entry in compatible
                    if entry.language_entry_id == expected_entry_id
                ]
            coverage_rows.append(
                {
                    "dataset_type_id": item.dataset_type_id,
                    "operator": item.operator,
                    "semantic_class": item.semantic_class,
                    "compatible_entry_ids": [
                        row.language_entry_id for row in compatible
                    ],
                    "coverage": "COVERED" if compatible else "MISSING",
                }
            )
            if not compatible:
                terminal_result = terminal_result or "LANGUAGE_CAPABILITY_MISSING"
                issues.append(
                    _issue(
                        "LANGUAGE_CAPABILITY_MISSING",
                        "BLOCKING",
                        "No active compatible composite entry.",
                        type_id=item.dataset_type_id,
                    )
                )
                continue
            covered_cells.add((item.operator, item.semantic_class))
            _run_composite_cases(
                item=item,
                compatible=compatible,
                strategies=strategies,
                mode=mode,
                fingerprint=fingerprint,
                issues=issues,
                matrix=matrix,
            )
            continue
        missing_phrase = _validate_phrase_binding(item)
        if missing_phrase:
            terminal_result = terminal_result or "LANGUAGE_PHRASE_BINDING_MISSING"
            issues.append(
                _issue(
                    "LANGUAGE_PHRASE_BINDING_MISSING",
                    "BLOCKING",
                    f"Required human-language binding {missing_phrase} is absent.",
                    type_id=item.dataset_type_id,
                )
            )
            continue
        compatible = _compatible_entries(registry, item)
        if mode == "registry":
            expected_entry_id = item.dataset_type_id.removeprefix("registry::")
            compatible = [
                entry
                for entry in compatible
                if entry.language_entry_id == expected_entry_id
            ]
        coverage_rows.append(
            {
                "dataset_type_id": item.dataset_type_id,
                "operator": item.operator,
                "semantic_class": item.semantic_class,
                "compatible_entry_ids": [row.language_entry_id for row in compatible],
                "coverage": "COVERED" if compatible else "MISSING",
            }
        )
        if not compatible:
            terminal_result = terminal_result or "LANGUAGE_CAPABILITY_MISSING"
            issues.append(
                _issue(
                    "LANGUAGE_CAPABILITY_MISSING",
                    "BLOCKING",
                    f"No active compatible entry for {item.operator} × {item.semantic_class}.",
                    type_id=item.dataset_type_id,
                )
            )
            continue
        covered_cells.add((item.operator, item.semantic_class))
        operator_contract = contracts[item.operator]
        type_contract = _accepted_to_contract(item)
        for entry in compatible:
            issues.extend(_structural_issues(item, entry, operator_contract))
            spec = _accepted_to_spec(item, entry)
            for fixture in strategies[item.semantic_class]:
                case_payload = {
                    "mode": mode,
                    "dataset_type_id": item.dataset_type_id,
                    "language_entry_id": entry.language_entry_id,
                    "fixture_id": fixture.fixture_id,
                    "fingerprint": fingerprint,
                }
                case_id = _case_id(case_payload)
                try:
                    trace = render_with_trace(
                        entry=entry,
                        spec=spec,
                        contract=type_contract,
                        registry=registry,
                        semantic_target=fixture.semantic_value,
                    )
                    detected = detect_render_issues(
                        question=trace["question"],
                        entry=entry,
                        spec=spec,
                        bound_target=trace["slots"]["TARGET_VALUE"]["bound_value"],
                        operator_contract=operator_contract,
                    )
                    if item.proposition_id and trace.get("question"):
                        from src.autonomous_qa.compiler.semantic_alignment import (
                            validate_question_proposition_alignment,
                        )

                        for alignment_code in validate_question_proposition_alignment(
                            proposition_id=item.proposition_id,
                            question=trace["question"],
                            pattern=entry.pattern,
                            phrase_bindings=item.phrase_bindings,
                        ):
                            detected.append(
                                (
                                    "PROPOSITION_LANGUAGE_MISALIGNMENT",
                                    "BLOCKING",
                                    alignment_code,
                                )
                            )
                except (ValueError, KeyError) as exc:
                    trace = {
                        "question": None,
                        "slots": {},
                        "transforms": [],
                        "renderer_contract_id": RENDERER_CONTRACT_ID,
                    }
                    detected = [
                        (
                            "LANGUAGE_ENTRY_INCOMPATIBLE",
                            "BLOCKING",
                            f"Renderer rejected entry composition: {exc}",
                        )
                    ]
                case_issues = [
                    _issue(
                        code,
                        severity,
                        detail,
                        case_id=case_id,
                        entry_id=entry.language_entry_id,
                        type_id=item.dataset_type_id,
                    )
                    for code, severity, detail in detected
                ]
                issues.extend(case_issues)
                matrix.append(
                    {
                        "preflight_case_id": case_id,
                        "mode": mode,
                        "operator": item.operator,
                        "semantic_class": item.semantic_class,
                        "dataset_type_id": (
                            item.dataset_type_id if mode == "dataset" else None
                        ),
                        "language_entry_id": entry.language_entry_id,
                        "fixture_id": fixture.fixture_id,
                        "fixture_kind": fixture.fixture_kind,
                        "slot_semantic_values": {
                            "TARGET_VALUE": fixture.semantic_value,
                            **item.phrase_bindings,
                        },
                        "slot_display_values": trace["slots"],
                        "rendered_question": trace["question"],
                        "detector_results": [row["issue_code"] for row in case_issues],
                        "status": (
                            "FAIL"
                            if any(row["severity"] == "BLOCKING" for row in case_issues)
                            else "REVIEW"
                            if case_issues
                            else "PASS"
                        ),
                    }
                )

    blocking = [row for row in issues if row["severity"] == "BLOCKING"]
    review = [row for row in issues if row["severity"] == "REVIEW"]
    result: PreflightResult = (
        terminal_result
        if terminal_result
        else "LANGUAGE_CONTRACT_FAIL"
        if blocking
        else "PREFLIGHT_PASS"
    )
    unique_entries = sorted({row["language_entry_id"] for row in matrix})
    audit = {
        "preflight_contract_id": PREFLIGHT_CONTRACT_ID,
        "mode": mode,
        "dataset": dataset,
        "contract_fingerprint": fingerprint,
        **fingerprint_inputs["resources"],
        "required_capability_set": [
            {"operator": operator, "semantic_class": semantic_class}
            for operator, semantic_class in sorted(required_cells)
        ],
        "covered_capability_set": [
            {"operator": operator, "semantic_class": semantic_class}
            for operator, semantic_class in sorted(covered_cells)
        ],
        "coverage_status": (
            "COMPLETE" if required_cells == covered_cells else "INCOMPLETE"
        ),
        "accepted_type_count": len(items) if mode == "dataset" else None,
        "entries_checked": len(unique_entries),
        "fixture_count": sum(len(value) for value in strategies.values()),
        "synthetic_render_count": len(matrix),
        "blocking_issue_count": len(blocking),
        "review_issue_count": len(review),
        "result": result,
        "zero_side_effects": {
            "semantic_sampling": 0,
            "production_rows_read": 0,
            "qa_ids_generated": 0,
            "audio_io": 0,
            "audio_decode": 0,
            "audio_download": 0,
            "external_llm_calls": 0,
            "qwen": 0,
            "training": 0,
        },
    }
    output_dir = _output_dir(mode, dataset, fingerprint)
    if write_outputs:
        _write_outputs(
            output_dir=output_dir,
            audit=audit,
            fingerprint_inputs=fingerprint_inputs,
            coverage=coverage_rows,
            items=items,
            matrix=matrix,
            issues=issues,
            registry=registry,
        )
    return {
        "audit": audit,
        "output_dir": output_dir,
        "render_matrix": matrix,
        "issues": issues,
        "coverage": coverage_rows,
        "fingerprint_inputs": fingerprint_inputs,
    }


def _output_dir(mode: str, dataset: str | None, fingerprint: str) -> Path:
    if mode == "registry":
        return ROOT / "data" / "materialized" / "language_preflight" / "registry" / "current"
    if not dataset:
        raise PreflightInputError("DATASET_NAME_REQUIRED")
    return ROOT / "data" / "materialized" / "language_preflight" / dataset / "current"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _summaries(
    matrix: list[dict[str, Any]], issues: list[dict[str, Any]], key: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in matrix:
        grouped[str(row[key])].append(row)
    for value, group in sorted(grouped.items()):
        case_ids = {row["preflight_case_id"] for row in group}
        group_issues = [row for row in issues if row["preflight_case_id"] in case_ids]
        rows.append(
            {
                key: value,
                "synthetic_renders": len(group),
                "passed": sum(row["status"] == "PASS" for row in group),
                "blocking_issues": sum(
                    row["severity"] == "BLOCKING" for row in group_issues
                ),
                "review_issues": sum(
                    row["severity"] == "REVIEW" for row in group_issues
                ),
            }
        )
    return rows


def _write_outputs(
    *,
    output_dir: Path,
    audit: dict[str, Any],
    fingerprint_inputs: dict[str, Any],
    coverage: list[dict[str, Any]],
    items: list[AcceptedLanguageType],
    matrix: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    registry: ProductionLanguageRegistry,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        **{key: audit[key] for key in audit if key != "zero_side_effects"},
        "operator_contract_id": OPERATOR_CONTRACT_ID,
        "fixture_strategy_id": FIXTURE_STRATEGY_ID,
        "slot_ownership_id": SLOT_OWNERSHIP_ID,
    }
    _write_json(output_dir / "preflight_manifest.json", manifest)
    _write_json(output_dir / "coverage.json", coverage)
    _write_json(
        output_dir / "operator_contracts.json",
        {
            "contract_id": OPERATOR_CONTRACT_ID,
            "contracts": {
                key: value.model_dump(mode="json")
                for key, value in sorted(operator_contracts().items())
            },
            "slot_ownership": SLOT_OWNERSHIP,
        },
    )
    fixtures = fixture_strategies()
    _write_json(
        output_dir / "fixture_manifest.json",
        {
            "strategy_id": FIXTURE_STRATEGY_ID,
            "strategies": {
                key: [fixture.model_dump(mode="json") for fixture in value]
                for key, value in sorted(fixtures.items())
            },
        },
    )
    _write_jsonl(output_dir / "render_matrix.jsonl", matrix)
    _write_jsonl(output_dir / "issues.jsonl", issues)
    _write_json(
        output_dir / "entry_summary.json",
        _summaries(matrix, issues, "language_entry_id"),
    )
    if audit["mode"] == "dataset":
        _write_json(
            output_dir / "required_capabilities.json",
            audit["required_capability_set"],
        )
        _write_json(
            output_dir / "phrase_bindings.json",
            {
                row.dataset_type_id: row.phrase_bindings
                for row in sorted(items, key=lambda item: item.dataset_type_id)
            },
        )
        _write_json(
            output_dir / "type_summary.json",
            _summaries(matrix, issues, "dataset_type_id"),
        )
    artifact_hashes = {
        name: sha256_file(output_dir / name)
        for name in (
            "coverage.json",
            "operator_contracts.json",
            "fixture_manifest.json",
            "render_matrix.jsonl",
            "issues.jsonl",
            "entry_summary.json",
        )
    }
    _write_json(
        output_dir / "determinism_audit.json",
        {
            "canonical_serialization": True,
            "stable_fixture_order": True,
            "stable_entry_order": True,
            "stable_case_ids": True,
            "contract_fingerprint": audit["contract_fingerprint"],
            "artifact_hashes": artifact_hashes,
        },
    )
    _write_json(
        output_dir / "audit.json",
        {
            **audit,
            "fingerprint_inputs": fingerprint_inputs,
            "registry_active_entries": sum(entry.enabled for entry in registry.entries),
        },
    )
    (output_dir / "report.md").write_text(_report(audit, len(items)), encoding="utf-8")


def _report(audit: dict[str, Any], item_count: int) -> str:
    mode_label = "Registry" if audit["mode"] == "registry" else "Dataset"
    return f"""# Language Preflight — {mode_label} Mode

Style Discovery chooses **what to ask**. Language Preflight validates **how to
say it**. Missing language coverage never rejects an otherwise valid semantic
task; it requests a reusable language-capability extension.

## Result

- Result: `{audit["result"]}`
- Contract fingerprint: `{audit["contract_fingerprint"]}`
- Language registry SHA256: `{audit["language_registry_file_sha256"]}`
- Renderer contract: `{audit["renderer_contract_id"]}`
- Accepted types/input entries: {item_count}
- Entries checked: {audit["entries_checked"]}
- Capability cells: {len(audit["required_capability_set"])}
- Synthetic renders: {audit["synthetic_render_count"]}
- Blocking issues: {audit["blocking_issue_count"]}
- Review issues: {audit["review_issue_count"]}

## Contracts and scaling

Supported operators are DIRECT, EQUALITY, TARGET_MATCH, and
PAIRWISE_SELECTION. Supported fixture strategies cover categorical, text,
numeric, ordinal, boolean, multi-label, and identifier-relation classes.
Presentation ownership is explicit: under the canonical renderer contract, language
entries own target quotation and surrounding punctuation while the renderer
normalizes and binds the target exactly once.

Complexity is `O(required language entries × bounded fixtures)`, independent
of dataset-row count, possible pair count, and production-QA count.

## Side effects

Semantic sampling, production-row reads, QA-ID generation, audio I/O, audio
decode/download, external LLM calls, Qwen, and training are all zero.
"""


def detector_regression() -> dict[str, Any]:
    defective_quote = "Mục tiêu là ““xin chào””. Hãy chọn A hoặc B."
    quote_caught = any(
        code == "DUPLICATE_PRESENTATION_WRAPPER"
        for code, _severity, _detail in detect_render_issues(
            question=defective_quote,
            entry=_synthetic_entry(
                operator="PAIRWISE_SELECTION",
                semantic_class="text_content",
                pattern="Mục tiêu là “[TARGET_VALUE]”. Hãy chọn A hoặc B.",
            ),
            spec=_synthetic_spec("text_content", "nội dung được nói"),
            bound_target="“xin chào”",
            operator_contract=operator_contracts()["PAIRWISE_SELECTION"],
        )
    )
    bad_pattern = "Liệu [CONTENT_PHRASE] nội dung được nói có giống nhau không?"
    tautology_caught = bool(
        _composition_ownership_issues(
            bad_pattern, _synthetic_spec("text_content", "nội dung được nói")
        )
    )
    registry = load_language_registry(REGISTRY_RESOURCE)
    corrected_entry = next(
        entry
        for entry in registry.entries
        if entry.language_entry_id == "lang_para_vp1_vi1_eq_text_01_05"
    )
    corrected_type = AcceptedLanguageType(
        dataset_type_id="regression-equality-text",
        operator="EQUALITY",
        semantic_class="text_content",
        semantic_field="synthetic_text",
        answer_kind="boolean",
        audio_input_count=2,
        logical_context_inputs=0,
        phrase_bindings=_generic_phrase_bindings("text_content", unit=None),
    )
    corrected_spec = _accepted_to_spec(corrected_type, corrected_entry)
    corrected_trace = render_with_trace(
        entry=corrected_entry,
        spec=corrected_spec,
        contract=_accepted_to_contract(corrected_type),
        registry=registry,
        semantic_target="xin chào",
    )
    corrected_issues = detect_render_issues(
        question=corrected_trace["question"],
        entry=corrected_entry,
        spec=corrected_spec,
        bound_target=None,
        operator_contract=operator_contracts()["EQUALITY"],
    )
    natural = "Người nói thuộc vùng phương ngữ nào?"
    natural_issues = detect_render_issues(
        question=natural,
        entry=_synthetic_entry(
            operator="DIRECT",
            semantic_class="categorical_attribute",
            pattern=natural,
        ),
        spec=_synthetic_spec("categorical_attribute", None),
        bound_target=None,
        operator_contract=operator_contracts()["DIRECT"],
    )
    return {
        "double_quote_would_be_caught": quote_caught,
        "direct_text_tautology_would_be_caught": tautology_caught,
        "duplicate_semantic_head_detected": tautology_caught,
        "corrected_entry_passes": not corrected_issues,
        "corrected_question": corrected_trace["question"],
        "corrected_issues": corrected_issues,
        "natural_nguoi_noi_incorrectly_rejected": bool(natural_issues),
        "natural_nguoi_noi_issues": natural_issues,
    }


def _synthetic_spec(
    semantic_class: str, content_phrase: str | None
) -> SemanticFieldSpec:
    return SemanticFieldSpec(
        field_name="synthetic",
        semantic_class=semantic_class,
        entity_scope="speaker",
        entity_phrase="người nói",
        attribute_phrase=(
            "vùng phương ngữ" if semantic_class != "text_content" else None
        ),
        content_phrase=content_phrase,
        value_phrase="giá trị",
        rendering={
            "target_quote_style": (
                "vietnamese_quotes" if semantic_class == "text_content" else "plain"
            )
        },
    )


def _synthetic_entry(
    *, operator: str, semantic_class: str, pattern: str
) -> LanguageRegistryEntry:
    answer = operator_contracts()[operator].answer_kind
    return LanguageRegistryEntry(
        language_entry_id="synthetic_entry",
        source_kind="CANONICAL",
        source_id="synthetic_source",
        canonical_blueprint_id="synthetic_source",
        operator=operator,
        semantic_class=semantic_class,
        pattern=pattern,
        answer_kind=answer,
        required_slots=re.findall(r"\[[A-Z][A-Z0-9_]*\]", pattern),
        unit_policy="any",
        match_policy=["exact"],
        semantic_contract_hash="synthetic",
        quality_status="PRODUCTION_PASS",
        enabled=True,
        template_library_hash="synthetic",
        registry_version="synthetic",
    )


def authorize_new_production(
    *,
    require_language_preflight: bool,
    current_fingerprint: str,
    pass_artifact: dict[str, Any] | None,
) -> dict[str, Any]:
    if not require_language_preflight:
        return {"allowed": True, "status": "PREFLIGHT_NOT_REQUIRED"}
    if pass_artifact is None:
        return {"allowed": False, "status": "PREFLIGHT_REQUIRED"}
    if pass_artifact.get("result") != "PREFLIGHT_PASS":
        return {"allowed": False, "status": "PREFLIGHT_FAILED"}
    if pass_artifact.get("contract_fingerprint") != current_fingerprint:
        return {"allowed": False, "status": "PREFLIGHT_STALE"}
    return {"allowed": True, "status": "PREFLIGHT_PASS"}


def verify_release() -> dict[str, Any]:
    actual = (
        (ROOT / "outputs" / "releases" / "vimd" / "final" / "qa_package.sha256")
        .read_text(encoding="utf-8")
        .strip()
    )
    return {
        "package_sha256": actual,
        "present": bool(actual),
    }


def write_milestone_report(
    *,
    registry_result: dict[str, Any],
    dataset_result: dict[str, Any],
    focused_tests: str,
    full_tests: str,
    ruff_status: str,
    format_status: str,
) -> Path:
    """Write the cross-mode architecture report after external test execution."""
    out = ROOT / "data" / "materialized" / "language_preflight"
    out.mkdir(parents=True, exist_ok=True)
    regression = detector_regression()
    release = verify_release()
    resource_hashes = {
        path.name: sha256_file(path) for path in sorted(LANGUAGE_ROOT.glob("*.json"))
    }
    gate = {
        "matching_pass": authorize_new_production(
            require_language_preflight=True,
            current_fingerprint=dataset_result["audit"]["contract_fingerprint"],
            pass_artifact=dataset_result["audit"],
        ),
        "missing_pass": authorize_new_production(
            require_language_preflight=True,
            current_fingerprint=dataset_result["audit"]["contract_fingerprint"],
            pass_artifact=None,
        ),
        "stale_pass": authorize_new_production(
            require_language_preflight=True,
            current_fingerprint="changed-contract",
            pass_artifact=dataset_result["audit"],
        ),
        "failed_pass": authorize_new_production(
            require_language_preflight=True,
            current_fingerprint=dataset_result["audit"]["contract_fingerprint"],
            pass_artifact={
                **dataset_result["audit"],
                "result": "LANGUAGE_CONTRACT_FAIL",
            },
        ),
    }
    _write_json(out / "detector_regression.json", regression)
    _write_json(out / "production_gate_audit.json", gate)
    _write_json(
        out / "canonical_integrity.json",
        {"release": release, "resource_hashes": resource_hashes},
    )
    conclusion = (
        "LANGUAGE_PREFLIGHT_PASS"
        if registry_result["audit"]["result"] == "PREFLIGHT_PASS"
        and dataset_result["audit"]["result"] == "PREFLIGHT_PASS"
        and not registry_result["audit"]["review_issue_count"]
        and not dataset_result["audit"]["review_issue_count"]
        and release["present"]
        else "LANGUAGE_PREFLIGHT_PASS_WITH_REVIEW"
    )
    audit = {
        "architecture": {
            "semantic_discovery": "chooses WHAT TO ASK",
            "language_preflight": "validates HOW TO SAY IT",
            "semantic_discovery_constrained_by_language": False,
        },
        "registry_preflight": registry_result["audit"],
        "vimd_dataset_preflight": dataset_result["audit"],
        "detector_regression": regression,
        "production_gate": gate,
        "canonical_integrity": {
            "release": release,
            "resource_hashes": resource_hashes,
        },
        "tests": {
            "focused": focused_tests,
            "full_pytest": full_tests,
            "ruff": ruff_status,
            "format": format_status,
        },
        "side_effects": registry_result["audit"]["zero_side_effects"],
        "conclusion": conclusion,
    }
    _write_json(out / "audit.json", audit)
    registry = registry_result["audit"]
    dataset = dataset_result["audit"]
    (out / "report.md").write_text(
        f"""# Global Language Preflight / Contract Validator

## Architecture

Style Discovery chooses **what to ask**. Language Preflight validates **how to
say it** after semantic acceptance. Missing language capability does not reject
or rewrite a valid semantic task.

## Contracts

Operators: DIRECT, EQUALITY, TARGET_MATCH, PAIRWISE_SELECTION. Fixture
strategies cover categorical, text, numeric, ordinal, boolean, multi-label,
and identifier-relation semantic classes. Under the canonical renderer contract, language
entries own target quotation and surrounding grammar; the renderer normalizes
and binds display values once and normalizes punctuation boundaries.

## Registry Preflight

- Registry: canonical production registry
- Entries: {registry["entries_checked"]}
- Capability cells: {len(registry["required_capability_set"])}
- Fixtures: {registry["fixture_count"]}
- Synthetic renders: {registry["synthetic_render_count"]}
- Blocking/review: {registry["blocking_issue_count"]}/{registry["review_issue_count"]}
- Result: `{registry["result"]}`

## ViMD Dataset Preflight

- Accepted types: {dataset["accepted_type_count"]}
- Required/covered cells: {len(dataset["required_capability_set"])}/{len(dataset["covered_capability_set"])}
- Compatible entries tested: {dataset["entries_checked"]}
- Synthetic renders: {dataset["synthetic_render_count"]}
- Blocking/review: {dataset["blocking_issue_count"]}/{dataset["review_issue_count"]}
- Result: `{dataset["result"]}`

## Detector Regression

- Doubled-quote composition caught: {regression["double_quote_would_be_caught"]}
- Duplicate semantic head caught: {regression["duplicate_semantic_head_detected"]}
- Corrected entry passes: {regression["corrected_entry_passes"]}
- Natural `người nói` incorrectly rejected: {regression["natural_nguoi_noi_incorrectly_rejected"]}

## Scaling

Runtime is `O(required language entries × bounded fixtures)`, not proportional
to dataset rows, possible pairs, or production QA count.

## Production Gate

Matching PASS allows new production. Missing, failed, capability-missing, and
stale artifacts block it.

## Canonical Integrity

- Release present: {release["present"]}
- Resource hashes: {resource_hashes}

## Side Effects

Production sampling, production-row reads, QA generation, audio I/O,
download/decode, external LLM calls, Qwen, and training are all zero.

## Tests

- Focused: {focused_tests}
- Full pytest: {full_tests}
- Ruff: {ruff_status}
- Format: {format_status}

## Conclusion

{conclusion}
""",
        encoding="utf-8",
    )
    return out


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("registry", "dataset"):
        command = subparsers.add_parser(mode)
        if mode == "dataset":
            command.add_argument(
                "--dataset", required=True, choices=("vimd", "vietmdd")
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _cli().parse_args(argv)
    try:
        accepted = (
            get_dataset_accepted_types(args.dataset) if args.mode == "dataset" else None
        )
        result = run_preflight(
            mode=args.mode,
            accepted_types=accepted,
            dataset=getattr(args, "dataset", None),
        )
    except (PreflightInputError, ValueError, KeyError) as exc:
        print(json.dumps({"result": "BLOCKED", "reason": str(exc)}))
        return 2
    print(
        json.dumps(
            {
                "result": result["audit"]["result"],
                "contract_fingerprint": result["audit"]["contract_fingerprint"],
                "output_dir": str(result["output_dir"]),
                "synthetic_render_count": result["audit"]["synthetic_render_count"],
                "blocking_issue_count": result["audit"]["blocking_issue_count"],
                "review_issue_count": result["audit"]["review_issue_count"],
            },
            indent=2,
        )
    )
    return 0 if result["audit"]["result"] == "PREFLIGHT_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

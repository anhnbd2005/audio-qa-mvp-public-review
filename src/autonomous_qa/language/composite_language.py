"""Generic COMPOSITE (structured multi-output) language rendering.

The composite language entry owns every model-facing phrase. The pattern is
one coherent instruction using ordered output-instruction slots; it is NOT a
concatenation of primitive wordings.

Slots owned by the entry:
  [AUDIO_REFERENCE]       e.g. "đoạn âm thanh" / "hai đoạn âm thanh A và B"
  [INSTRUCTION_1..3]      one phrase per ordered output component
  [TARGET_VALUE]          optional visible context value (bound by renderer)
"""

from __future__ import annotations

import re
from typing import Any

from src.common.answer_schema import (
    StructuredAnswerSchema,
    serialize_structured_answer,
    validate_structured_gold,
)
from src.autonomous_qa.production.production_qa import (
    _normalize_rendered_pattern,
    bind_target_value,
    normalize_target_quote_boundary,
)

SLOT_RE = re.compile(r"\[[A-Z][A-Z0-9_]*\]")
INSTRUCTION_SLOTS = ("[INSTRUCTION_1]", "[INSTRUCTION_2]", "[INSTRUCTION_3]")


class CompositeLanguageError(ValueError):
    pass


def composite_slots_used(pattern: str) -> set[str]:
    return set(SLOT_RE.findall(pattern))


def render_composite_question(
    *,
    pattern: str,
    output_signature: list[dict[str, Any]],
    instruction_bindings: dict[str, str],
    audio_reference_phrase: str,
    target_display: Any = None,
    quote_style: str = "vietnamese_quotes",
) -> str:
    """Render one coherent composite question from a language entry."""
    question = pattern
    for index, component in enumerate(output_signature):
        slot = f"[INSTRUCTION_{index + 1}]"
        if slot not in question:
            continue
        phrase = instruction_bindings.get(component["role"])
        if not phrase:
            raise CompositeLanguageError(
                f"INSTRUCTION_BINDING_MISSING:{component['role']}"
            )
        question = question.replace(slot, phrase)
    question = question.replace("[AUDIO_REFERENCE]", audio_reference_phrase)
    if "[TARGET_VALUE]" in question:
        if target_display is None:
            raise CompositeLanguageError("TARGET_DISPLAY_MISSING")
        bound = bind_target_value(target_display, quote_style=quote_style)
        question = _normalize_rendered_pattern(
            question.replace("[TARGET_VALUE]", bound)
        )
        question = normalize_target_quote_boundary(question)
    unresolved = SLOT_RE.findall(question)
    # [TARGET_VALUE] may legitimately remain only if the caller supplied none.
    unresolved = [slot for slot in unresolved if slot != "[TARGET_VALUE]"]
    if unresolved:
        raise CompositeLanguageError(f"UNRESOLVED_SLOT:{sorted(set(unresolved))}")
    return question


def composite_render_issues(
    *,
    question: str,
    pattern: str,
    output_signature: list[dict[str, Any]],
    instruction_bindings: dict[str, str],
    context_roles: list[str],
    bound_target: str | None,
) -> list[tuple[str, str, str]]:
    """Deterministic composite wording validators (Sections 39, 75)."""
    issues: list[tuple[str, str, str]] = []
    if not question.strip():
        issues.append(("EMPTY_QUESTION", "BLOCKING", "Rendered question is empty."))
    if SLOT_RE.search(question):
        issues.append(("UNRESOLVED_SLOT", "BLOCKING", "Slot remains unresolved."))
    # Every declared output must have exactly one instruction slot + binding.
    for index, component in enumerate(output_signature):
        slot = f"[INSTRUCTION_{index + 1}]"
        if pattern.count(slot) != 1:
            issues.append(
                (
                    "OUTPUT_INSTRUCTION_MISSING",
                    "BLOCKING",
                    f"Expected exactly one {slot}.",
                )
            )
        if not instruction_bindings.get(component["role"]):
            issues.append(
                (
                    "INSTRUCTION_BINDING_MISSING",
                    "BLOCKING",
                    f"No binding for role {component['role']}.",
                )
            )
    extra = [
        slot for slot in INSTRUCTION_SLOTS[len(output_signature) :] if slot in pattern
    ]
    if extra:
        issues.append(
            ("DUPLICATE_OUTPUT_INSTRUCTION", "BLOCKING", f"Extra slots {extra}.")
        )
    if question.count("“") != question.count("”") or question.count('"') % 2:
        issues.append(("UNBALANCED_QUOTES", "BLOCKING", "Unbalanced quotes."))
    if any(token in pattern for token in ("““", "””", '""')):
        issues.append(
            ("DUPLICATE_PRESENTATION_WRAPPER", "BLOCKING", "Double quote wrapper.")
        )
    if re.search(r"[.!?;:,]”[.!?](?=\s|$)|\?\.|\.\?|!!|\?\?", question):
        issues.append(("PUNCTUATION_COLLISION", "BLOCKING", "Punctuation collision."))
    # context presence
    needs_target = any(
        "[TARGET_VALUE]" in phrase for phrase in instruction_bindings.values()
    )
    if needs_target and bound_target is None:
        issues.append(("TARGET_MISSING", "BLOCKING", "Visible target was lost."))
    if bound_target is not None and question.count(bound_target) != 1:
        issues.append(
            (
                "TARGET_DUPLICATED"
                if question.count(bound_target) > 1
                else "TARGET_MISSING",
                "BLOCKING",
                "Visible target must appear exactly once.",
            )
        )
    if context_roles and not needs_target:
        pass  # context may be a reference label without a bound target
    if re.search(r"\b[a-zA-Z]+_[a-zA-Z0-9_]+\b", question):
        issues.append(("SNAKE_CASE_LEAK", "BLOCKING", "snake_case leaked."))
    for token in ("semantic_instance", "language_entry", "source_row", "speakerID"):
        if token.casefold() in question.casefold():
            issues.append(("INTERNAL_TOKEN_LEAK", "BLOCKING", f"{token} leaked."))
    return list(dict.fromkeys(issues))


def composite_entry_matches(
    *,
    entry_output_signature: list[dict[str, Any]],
    entry_context_roles: list[str],
    expected_output_signature: list[dict[str, Any]],
    expected_context_roles: list[str],
) -> bool:
    def _sig(sig: list[dict[str, Any]]) -> list[tuple[str, str]]:
        return [(c.get("role"), c.get("kind")) for c in sig]

    return _sig(entry_output_signature) == _sig(expected_output_signature) and list(
        entry_context_roles
    ) == list(expected_context_roles)


def serialize_composite_gold(
    schema: StructuredAnswerSchema, gold: dict[str, Any]
) -> str:
    errors = validate_structured_gold(schema, gold)
    if errors:
        raise CompositeLanguageError(";".join(errors))
    return serialize_structured_answer(schema, gold)

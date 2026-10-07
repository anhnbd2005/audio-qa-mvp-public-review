"""Shared operation-family base template machinery (§1-§20, §25, §29, §L-§O).

Enables delexicalized operation-level base templates (e.g. ("equality", "semantic", entity_scope))
to be shared across multiple concrete QA types of compatible entity scope,
bound to type-specific [KEY] realizations before Paraphrase, and persisted
across rounds in family_template_bank.
"""

from __future__ import annotations

import re
from typing import Any

from src.autonomous_qa.core.validity import (
    template_family_signature,
    validate_family_base_template,
)

FAMILY_EQUALITY_SEMANTIC_SPEAKER: tuple[str, str, str] = ("equality", "semantic", "speaker")
FAMILY_EQUALITY_SEMANTIC_UTTERANCE: tuple[str, str, str] = ("equality", "semantic", "utterance")
FAMILY_EQUALITY_SEMANTIC: tuple[str, str] = ("equality", "semantic")
FAMILY_KEY_DELIMITER = ":"


def serialize_family_signature(sig: tuple[str, ...]) -> str:
    """Convert tuple signature to string key for JSON serialization."""
    return FAMILY_KEY_DELIMITER.join(sig)


def deserialize_family_signature(key: str) -> tuple[str, ...]:
    """Convert string key back to tuple signature."""
    parts = key.split(FAMILY_KEY_DELIMITER)
    return tuple(parts)


def bind_family_template(
    base: dict,
    type_id: str,
    key_realization: str,
    family_sig: tuple[str, ...] | None = None,
) -> dict:
    """Bind a shared family base template to a concrete type's key realization.

    Produces a fully lexicalized type-specific base template before Paraphrase
    with complete lineage metadata (§15, §29).
    """
    base_id = base.get("base_template_id") or base.get("template_id") or "BF_UNKNOWN"
    base_text = base.get("text", "")
    bound_text = base_text.replace("[KEY]", key_realization)
    bound_id = f"BT_{type_id}__{base_id}"
    sig = family_sig
    if sig is None and "family_signature" in base and base["family_signature"]:
        sig = tuple(base["family_signature"]) if isinstance(base["family_signature"], list) else base["family_signature"]
    return {
        "template_id": bound_id,
        "question_type_id": type_id,
        "text": bound_text,
        "family_signature": list(sig) if sig else list(FAMILY_EQUALITY_SEMANTIC),
        "base_template_id": base_id,
        "base_template_text": base_text,
        "key_realization": key_realization,
        "bound_base_text": bound_text,
    }


def bind_round_family_templates(
    family_bases: list[dict],
    family_types: list[dict],
    key_realizations: dict[str, str],
    family_sig: tuple[str, ...] | None = None,
) -> list[dict]:
    """Bind shared family bases to all current-round family member types.

    Runs in Python BEFORE the deterministic Gate (§25).
    """
    bound_templates: list[dict] = []
    for qtype in family_types:
        tid = qtype.get("id")
        if not tid:
            continue
        phrase = key_realizations.get(tid)
        if not isinstance(phrase, str) or not phrase.strip():
            continue
        for base in family_bases:
            # Validate structural skeleton
            errs = validate_family_base_template(base.get("text", ""))
            if errs:
                continue
            bound = bind_family_template(base, tid, phrase.strip(), family_sig=family_sig)
            bound_templates.append(bound)
    return bound_templates


def build_family_bank_entry(
    base_template_id: str,
    text: str,
    round_created: int,
) -> dict:
    """Create a structured family template bank item."""
    return {
        "base_template_id": base_template_id,
        "text": text,
        "round_created": round_created,
    }

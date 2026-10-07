"""Generic new-dataset candidate readiness helpers.

Dataset-agnostic and read-only:
- field-to-semantic coverage matrix (prevents stochastic LLM omission);
- deterministic candidate language preflight.

No LLM calls, no canonical writes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

_SLOT_RE = re.compile(r"\[[A-Z][A-Z0-9_]*\]")
_PLACEHOLDER_RE = re.compile(r"\{\{[^}]*\}\}")
_SNAKE_RE = re.compile(r"\b[a-z]+_[a-z0-9_]+\b")


def field_coverage_matrix(
    semantic_fields: Iterable[str], candidates: Iterable[dict]
) -> dict[str, Any]:
    """Map each semantic-bearing field to the candidates that reference it."""
    matrix: dict[str, list[str]] = {}
    for field in semantic_fields:
        users: list[str] = []
        for cand in candidates:
            blob = json.dumps(
                {
                    "hidden": cand.get("hidden_source_annotations"),
                    "visible": cand.get("visible_inputs"),
                    "evidence": cand.get("source_evidence"),
                    "proposition": cand.get("proposition"),
                },
                ensure_ascii=False,
            ).lower()
            if field.lower() in blob:
                users.append(cand.get("candidate_id"))
        matrix[field] = users
    omitted = [field for field, users in matrix.items() if not users]
    return {"matrix": matrix, "omitted_fields": omitted}


def candidate_language_preflight(
    entries: Iterable[dict], accepted_ids: set[str]
) -> dict[str, Any]:
    """Deterministic language checks for candidate entries (no LLM)."""
    entries = list(entries)
    blocking = 0
    details: list[dict[str, Any]] = []
    for entry in entries:
        issues: list[str] = []
        if entry.get("type_id") not in accepted_ids:
            issues.append("UNKNOWN_TYPE_ID")
        templates = entry.get("question_templates") or []
        if not templates:
            issues.append("NO_TEMPLATES")
        for template in templates:
            stripped = _PLACEHOLDER_RE.sub(" ", str(template))
            if _SLOT_RE.search(stripped):
                issues.append("UNRESOLVED_SLOT")
            if _SNAKE_RE.search(stripped):
                issues.append("SNAKE_CASE_LEAK")
        if not entry.get("answer_format"):
            issues.append("NO_ANSWER_FORMAT")
        if issues:
            blocking += 1
        details.append({"type_id": entry.get("type_id"), "issues": issues})
    return {
        "entries": len(entries),
        "blocking_issue_count": blocking,
        "result": "PREFLIGHT_PASS" if blocking == 0 else "PREFLIGHT_REVIEW",
        "details": details,
    }

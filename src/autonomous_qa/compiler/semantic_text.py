"""Single canonical semantic text normalization for VietMDD relations.

The frozen materialized TRAIN fields (``*_norm``) were produced by exactly:

    1. strip leading/trailing Unicode whitespace
    2. collapse every whitespace run to one ASCII space

It is deliberately NOT Unicode-NFKC normalized (NFKC rewrites U+2026
"…" to "...") and NOT casefolded (casefold changes 170 reference rows).
Empirical reconciliation over all 3,181 TRAIN rows shows this function
reproduces both ``original_text_norm`` and ``observed_transcription_norm``
with 0 mismatches.

Every relation implementation (materialized ``text_exact_match``, primitive
P3 audit, composite C2 derivation, composite C3 equality, target matching and
the final QA audit) MUST route through this one function.
"""

from __future__ import annotations

from typing import Any


def normalize_semantic_text(value: Any) -> str:
    """Canonical semantic-text normalization: strip + collapse whitespace."""
    return " ".join(str(value or "").split())


def semantic_text_equal(left: Any, right: Any) -> bool:
    """Canonical exact relation used by all text equality/match contracts."""
    return normalize_semantic_text(left) == normalize_semantic_text(right)

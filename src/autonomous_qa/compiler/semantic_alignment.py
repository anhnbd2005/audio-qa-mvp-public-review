"""Deterministic question ↔ proposition alignment validation.

Distinguishes IMPLEMENTATION CONSISTENCY from SEMANTIC VALIDITY. A rendered
question is accepted only if it asks the same proposition as the declared
semantic task.

Dataset-agnostic: keys on ``proposition_id`` and declared semantic roles.
"""

from __future__ import annotations

from typing import Any

SPOKEN_CONTENT_MARKERS = (
    "nội dung phát âm",
    "nội dung được nói",
    "nội dung lời nói",
    "nội dung đọc",
    "nội dung được đọc",
)

# Phrasings that wrongly place the reference text inside the audio, or ask
# whether the displayed reference equals itself.
FORBIDDEN_REFERENCE_SUBJECT_PHRASES = (
    "nội dung văn bản tham chiếu",
    "văn bản tham chiếu trong đoạn âm thanh",
    "văn bản tham chiếu của đoạn âm thanh",
    "văn bản tham chiếu trong âm thanh",
)

_SPOKEN_SUBJECT_PROPOSITIONS = {
    "spoken_content_matches_reference",
    "spoken_content_matches_candidate",
    "structured_transcribe_then_match_reference",
    "structured_transcribe_then_match_candidate",
}


def _norm(text: Any) -> str:
    return " ".join(str(text or "").casefold().split())


def validate_question_proposition_alignment(
    *,
    proposition_id: str | None,
    question: str,
    pattern: str | None = None,
    phrase_bindings: dict[str, Any] | None = None,
) -> list[str]:
    """Return alignment error codes for one rendered question."""
    if not proposition_id or proposition_id not in _SPOKEN_SUBJECT_PROPOSITIONS:
        return []
    q = _norm(question)
    errors: list[str] = []
    if not any(marker in q for marker in SPOKEN_CONTENT_MARKERS):
        errors.append("SPOKEN_CONTENT_SUBJECT_MISSING")
    # The visible object (reference/candidate) must be asked about.
    if pattern is not None and "[TARGET_VALUE]" not in pattern:
        errors.append("VISIBLE_OBJECT_MISSING")
    for bad in FORBIDDEN_REFERENCE_SUBJECT_PHRASES:
        if bad in q:
            errors.append(f"FORBIDDEN_REFERENCE_SUBJECT:{bad}")
    return list(dict.fromkeys(errors))

"""Independent oracle + characterization for the reference-match repair.

The oracle deliberately does NOT call the production comparator: it implements
spoken-lexical equivalence through a separate regex-based code path so a
production regression cannot silently pass.

This module is dataset-agnostic; callers supply rows.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

# Independent implementation: delete every character that is not a Unicode
# word character or whitespace (a different mechanism than the production
# comparator's unicodedata category check), then casefold + collapse + strip.
_ORACLE_NONLEXICAL_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def oracle_spoken_content_equivalent(left: Any, right: Any) -> bool:
    def _norm(value: Any) -> str:
        text = "" if value is None else str(value)
        text = _ORACLE_NONLEXICAL_RE.sub(" ", text)
        text = " ".join(text.split()).casefold()
        return text

    return _norm(left) == _norm(right)


def _classify(observed: str, reference: str) -> str:
    """Attribute a difference between observed and reference text."""
    if observed == reference:
        return "RAW_EQUAL"
    o_ws = " ".join(observed.split())
    r_ws = " ".join(reference.split())
    if o_ws == r_ws:
        return "WHITESPACE_ONLY"
    if o_ws.casefold() == r_ws.casefold():
        return "CASE_ONLY"
    o_lex = _ORACLE_NONLEXICAL_RE.sub(" ", o_ws).casefold()
    r_lex = _ORACLE_NONLEXICAL_RE.sub(" ", r_ws).casefold()
    o_lex = " ".join(o_lex.split())
    r_lex = " ".join(r_lex.split())
    if o_lex == r_lex:
        if o_ws.casefold() != r_ws.casefold() and (
            o_lex != o_ws.casefold() or r_lex != r_ws.casefold()
        ):
            return "PUNCTUATION_OR_CASE_AND_PUNCTUATION_ONLY"
        return "PUNCTUATION_ONLY"
    return "LEXICAL_OR_MIXED"


def characterize_reference_repair(
    train_rows: list[dict[str, Any]],
    old_p3_gold: dict[str, bool],
    new_p3_gold: dict[str, bool],
) -> dict[str, Any]:
    """Full defect/flip characterization for the reference-match contract.

    ``old_p3_gold`` / ``new_p3_gold`` map audio_id -> boolean.
    """
    classes: Counter = Counter()
    old_true = old_false = new_true = new_false = 0
    f2t = t2f = unchanged_true = unchanged_false = 0
    punctuation_chars: Counter = Counter()
    examples: dict[str, list[dict[str, str]]] = {}
    presentation_only_false_before = 0
    presentation_only_false_after = 0

    for row in train_rows:
        audio_id = row["audio_id"]
        observed = row["observed_transcription_norm"]
        reference = row["original_text_norm"]
        old = bool(old_p3_gold[audio_id])
        new = bool(new_p3_gold[audio_id])
        old_true += 1 if old else 0
        old_false += 0 if old else 1
        new_true += 1 if new else 0
        new_false += 0 if new else 1
        if old and new:
            unchanged_true += 1
        elif not old and not new:
            unchanged_false += 1
        elif not old and new:
            f2t += 1
        else:
            t2f += 1
        if observed != reference:
            cls = _classify(observed, reference)
            classes[cls] += 1
            for ch in set(observed) ^ set(reference):
                if unicodedata.category(ch).startswith("P"):
                    punctuation_chars[ch] += 1
            examples.setdefault(cls, [])
            if len(examples[cls]) < 5 and cls != "LEXICAL_OR_MIXED":
                examples[cls].append({"observed": observed, "reference": reference})
        if not old and observed != reference:
            cls = _classify(observed, reference)
            if cls in (
                "PUNCTUATION_ONLY",
                "PUNCTUATION_OR_CASE_AND_PUNCTUATION_ONLY",
                "CASE_ONLY",
                "WHITESPACE_ONLY",
            ):
                presentation_only_false_before += 1
        if not new and observed != reference:
            cls = _classify(observed, reference)
            if cls in (
                "PUNCTUATION_ONLY",
                "PUNCTUATION_OR_CASE_AND_PUNCTUATION_ONLY",
                "CASE_ONLY",
                "WHITESPACE_ONLY",
            ):
                presentation_only_false_after += 1

    return {
        "reference_rows": len(train_rows),
        "old_true": old_true,
        "old_false": old_false,
        "new_true": new_true,
        "new_false": new_false,
        "false_to_true": f2t,
        "true_to_false": t2f,
        "unchanged_true": unchanged_true,
        "unchanged_false": unchanged_false,
        "difference_classes": dict(sorted(classes.items())),
        "punctuation_characters": {
            ch: {"category": unicodedata.category(ch), "count": n}
            for ch, n in sorted(punctuation_chars.items(), key=lambda kv: -kv[1])
        },
        "examples": examples,
        "presentation_only_false_before": presentation_only_false_before,
        "presentation_only_false_after": presentation_only_false_after,
        "user_estimate_599_confirmed": (presentation_only_false_before == 599),
    }

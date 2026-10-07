"""Evaluation metrics and constrained candidate next-token logit scoring."""

from __future__ import annotations

import unicodedata
from typing import Any, Sequence

UNPARSEABLE = "UNPARSEABLE"
_PUNCT = " \t\r\n\"'`.,;:!?()[]{}<>-\u2013\u2014_*/\\|+="


def normalise_answer_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    return text.strip(_PUNCT).strip()


def parse_answer(raw: str, choices: Sequence[str]) -> tuple[str | None, str]:
    choice_list = [str(c) for c in (choices or [])]
    if not choice_list:
        return None, UNPARSEABLE
    norm_choices = [normalise_answer_text(c) for c in choice_list]
    norm_raw = normalise_answer_text(raw)
    if not norm_raw:
        return None, UNPARSEABLE

    exact = [i for i, c in enumerate(norm_choices) if c and c == norm_raw]
    if len(exact) == 1:
        return choice_list[exact[0]], "exact"
    if len(exact) > 1:
        return None, UNPARSEABLE

    contained = [i for i, c in enumerate(norm_choices) if c and c in norm_raw]
    if len(contained) == 1:
        return choice_list[contained[0]], "contained"
    if len(contained) > 1:
        return None, UNPARSEABLE

    if len(norm_raw) == 1 and "a" <= norm_raw <= "z":
        idx = ord(norm_raw.upper()) - ord("A")
        if 0 <= idx < len(choice_list):
            return choice_list[idx], "letter"

    return None, UNPARSEABLE


def is_correct(parsed: str | None, gold: str | None) -> bool:
    if parsed is None or gold is None:
        return False
    return normalise_answer_text(parsed) == normalise_answer_text(gold)


def evaluate_predictions(
    records: Sequence[dict[str, Any]],
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    parser_failures = 0
    for record in records:
        choices = list(record.get("choices") or [])
        gold = record.get("gold_answer")
        raw = record.get("raw_prediction")
        if raw is None:
            raw = record.get("raw_output", "")
        parsed, status = parse_answer(raw, choices)
        correct = is_correct(parsed, gold)
        if parsed is None:
            parser_failures += 1
        results.append({
            **record,
            "gold_answer": gold,
            "raw_prediction": raw,
            "parsed_prediction": parsed,
            "parse_status": status,
            "correct": correct,
        })

    total = len(results)
    correct_count = sum(1 for row in results if row["correct"])
    accuracy = correct_count / total if total else 0.0

    label_list = list(labels) if labels else sorted(
        {str(row["gold_answer"]) for row in results if row.get("gold_answer") is not None}
    )

    per_class: dict[str, dict[str, float]] = {}
    for label in label_list:
        tp = sum(1 for row in results
                 if is_correct(row["parsed_prediction"], label)
                 and normalise_answer_text(row["gold_answer"]) == normalise_answer_text(label))
        fp = sum(1 for row in results
                 if is_correct(row["parsed_prediction"], label)
                 and normalise_answer_text(row["gold_answer"]) != normalise_answer_text(label))
        fn = sum(1 for row in results
                 if not is_correct(row["parsed_prediction"], label)
                 and normalise_answer_text(row["gold_answer"]) == normalise_answer_text(label))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    macro_f1 = (sum(v["f1"] for v in per_class.values()) / len(per_class)) if per_class else 0.0

    confusion: dict[str, dict[str, int]] = {}
    for row in results:
        gold = str(row.get("gold_answer"))
        pred = row["parsed_prediction"] if row["parsed_prediction"] is not None else UNPARSEABLE
        confusion.setdefault(gold, {})
        confusion[gold][str(pred)] = confusion[gold].get(str(pred), 0) + 1

    return {
        "rows": total,
        "correct": correct_count,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "parser_failures": parser_failures,
        "parser_failure_rate": (parser_failures / total) if total else 0.0,
        "records": results,
    }

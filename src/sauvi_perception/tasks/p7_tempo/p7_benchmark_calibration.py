"""Benchmark-calibrated P7 lyric-density rule inference from SAUVI P7 gold.

This module intentionally derives 3 ordered WPM thresholds from the 89 SAUVI P7
benchmark examples (``music-ld-*``) so that future training construction can use
a benchmark-calibrated density rule.

PROVENANCE WARNING: the resulting policy is
``BENCHMARK_DERIVED_SEMANTIC_POLICY`` — it is NOT the original SAUVI
construction rule. Evaluating a model trained with these derived labels on the
same SAUVI P7 examples is not an untouched held-out evaluation.

No audio, no LLM, no per-ID overrides; the hypothesis family is strictly
3 ordered WPM thresholds -> 4 ordered density classes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from decimal import Decimal, localcontext
from itertools import pairwise
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from src.common.config import ROOT
from src.sauvi_perception.datasets.vietlyrics.vietlyrics_sanitation import (
    extract_sauvi_asset_id,
    load_sauvi_vietlyrics_rows,
    parse_wpm_decimal,
    resolve_sauvi_manifest,
)
from src.sauvi_perception.datasets.vietlyrics.vietlyrics_source import (
    REVISION,
    extract_zing_id,
)

TASK = "P7"
SEMANTIC_TYPE = "lyric_density_classification"
POLICY_ORIGIN = "BENCHMARK_DERIVED_SEMANTIC_POLICY"
P7_SUBTASK = "music tempo detection"
CLASS_ORDER = ("Thưa", "Vừa", "Dày", "Rất dày")
CLASS_RANK = {label: index for index, label in enumerate(CLASS_ORDER)}

POLICY_PATH = ROOT / "resources" / "semantics" / "p7_lyric_density_policy.json"
CALIBRATION_DIR = (
    ROOT
    / "outputs"
    / "source_audits"
    / "vietlyrics"
    / REVISION
    / "p7_benchmark_calibration"
)
CLEAN_TRAIN_ASSETS = (
    ROOT
    / "outputs"
    / "source_audits"
    / "vietlyrics"
    / REVISION
    / "asset_sanitation"
    / "clean_train_assets.jsonl"
)


class CalibrationError(RuntimeError):
    """Raised when calibration cannot proceed under the contract."""


# ---------------------------------------------------------------------------
# Extraction + mapping
# ---------------------------------------------------------------------------


def extract_p7_rows(manifest_path: Path | None = None) -> list[dict[str, Any]]:
    rows = load_sauvi_vietlyrics_rows(resolve_sauvi_manifest(manifest_path))
    p7 = [row for row in rows if str(row.get("sub-category")) == P7_SUBTASK]
    result = []
    for row in p7:
        result.append(
            {
                "qa_id": str(row.get("id")),
                "benchmark_audio_id": str(row.get("audio_id")),
                "zing_id": extract_sauvi_asset_id(row.get("audio_id")),
                "answer": str(row.get("answer")),
            }
        )
    return result


def build_source_wpm_index(
    train_rows: Sequence[dict[str, str]], val_rows: Sequence[dict[str, str]]
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"members": [], "membership": set()}
    )
    for split, rows in (("train", train_rows), ("val", val_rows)):
        for row_index, row in enumerate(rows):
            asset_id = extract_zing_id(row.get("link"))
            if not asset_id:
                continue
            index[asset_id]["members"].append(
                {"split": split, "row_index": row_index, "wpm_raw": row.get("wpm", "")}
            )
            index[asset_id]["membership"].add(split)
    return index


def resolve_calibration(
    p7_rows: Sequence[dict[str, Any]], index: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    usable: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    not_found: list[dict[str, Any]] = []
    for row in p7_rows:
        entry = index.get(row["zing_id"])
        if not entry:
            not_found.append(row)
            continue
        decimals = {parse_wpm_decimal(member["wpm_raw"]) for member in entry["members"]}
        membership = sorted(entry["membership"])
        if len(decimals) == 1 and None not in decimals:
            usable.append(
                {
                    "qa_id": row["qa_id"],
                    "zing_id": row["zing_id"],
                    "wpm": next(iter(decimals)),
                    "gold_label": row["answer"],
                    "source_membership": membership,
                }
            )
        else:
            ambiguous.append(
                {
                    "qa_id": row["qa_id"],
                    "zing_id": row["zing_id"],
                    "gold_label": row["answer"],
                    "source_membership": membership,
                    "candidate_wpm": sorted(
                        str(value) for value in decimals if value is not None
                    ),
                }
            )
    usable.sort(key=lambda item: (item["wpm"], CLASS_RANK[item["gold_label"]]))
    return {"usable": usable, "ambiguous": ambiguous, "not_found": not_found}


# ---------------------------------------------------------------------------
# Monotonicity + statistics
# ---------------------------------------------------------------------------


def monotonicity(usable: Sequence[dict[str, Any]]) -> dict[str, Any]:
    sequence = [CLASS_RANK[item["gold_label"]] for item in usable]
    inversions = sum(1 for a, b in pairwise(sequence) if a > b)
    by_wpm: dict[Decimal, set[str]] = defaultdict(set)
    for item in usable:
        by_wpm[item["wpm"]].add(item["gold_label"])
    conflicts = {
        str(wpm): sorted(labels) for wpm, labels in by_wpm.items() if len(labels) > 1
    }
    return {
        "sorted_ascending": sequence == sorted(sequence),
        "adjacent_inversions": inversions,
        "same_wpm_conflicting_labels": conflicts,
        "monotonic": inversions == 0 and not conflicts,
    }


def _percentile(sorted_values: Sequence[Decimal], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    low = float(sorted_values[lower])
    high = float(sorted_values[upper])
    return low * (1 - weight) + high * weight


def class_statistics(usable: Sequence[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for label in CLASS_ORDER:
        values = sorted(item["wpm"] for item in usable if item["gold_label"] == label)
        if not values:
            stats[label] = {"n": 0}
            continue
        stats[label] = {
            "n": len(values),
            "min": round(float(values[0]), 6),
            "p05": round(_percentile(values, 0.05), 6),
            "p25": round(_percentile(values, 0.25), 6),
            "median": round(_percentile(values, 0.50), 6),
            "mean": round(mean(float(v) for v in values), 6),
            "p75": round(_percentile(values, 0.75), 6),
            "p95": round(_percentile(values, 0.95), 6),
            "max": round(float(values[-1]), 6),
            "std": round(pstdev(float(v) for v in values), 6)
            if len(values) > 1
            else 0.0,
        }
    return stats


def feasible_boundaries(usable: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    boundaries = []
    for lower_label, upper_label in pairwise(CLASS_ORDER):
        lower = [item["wpm"] for item in usable if item["gold_label"] == lower_label]
        upper = [item["wpm"] for item in usable if item["gold_label"] == upper_label]
        lower_max = max(lower) if lower else None
        upper_min = min(upper) if upper else None
        gap = (
            (upper_min - lower_max)
            if lower_max is not None and upper_min is not None
            else None
        )
        boundaries.append(
            {
                "boundary": f"{lower_label} -> {upper_label}",
                "lower_label": lower_label,
                "upper_label": upper_label,
                "lower_max": str(lower_max) if lower_max is not None else None,
                "upper_min": str(upper_min) if upper_min is not None else None,
                "gap": str(gap) if gap is not None else None,
                "clean": gap is not None and gap > 0,
            }
        )
    return boundaries


def derive_thresholds(boundaries: Sequence[dict[str, Any]]) -> dict[str, Decimal]:
    if not all(boundary["clean"] for boundary in boundaries):
        raise CalibrationError("boundaries_not_clean")
    with localcontext() as context:
        context.prec = 60
        thresholds = {}
        for index, boundary in enumerate(boundaries, start=1):
            lower_max = Decimal(boundary["lower_max"])
            upper_min = Decimal(boundary["upper_min"])
            thresholds[f"t{index}"] = (lower_max + upper_min) / 2
        return thresholds


def predict(wpm: Decimal, thresholds: dict[str, Decimal]) -> str:
    if wpm < thresholds["t1"]:
        return "Thưa"
    if wpm < thresholds["t2"]:
        return "Vừa"
    if wpm < thresholds["t3"]:
        return "Dày"
    return "Rất dày"


def reproduce(
    usable: Sequence[dict[str, Any]], thresholds: dict[str, Decimal]
) -> dict[str, Any]:
    mismatches = []
    correct = 0
    for item in usable:
        predicted = predict(item["wpm"], thresholds)
        if predicted == item["gold_label"]:
            correct += 1
        else:
            mismatches.append(
                {
                    "qa_id": item["qa_id"],
                    "zing_id": item["zing_id"],
                    "wpm": str(item["wpm"]),
                    "gold_label": item["gold_label"],
                    "predicted_label": predicted,
                }
            )
    return {
        "correct": correct,
        "total": len(usable),
        "accuracy": round(correct / len(usable), 6) if usable else 0.0,
        "mismatches": mismatches,
    }


# ---------------------------------------------------------------------------
# Exhaustive best monotonic search (diagnostic)
# ---------------------------------------------------------------------------


def best_monotonic_search(usable: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Exhaustively search 3 ordered thresholds over WPM gaps (no ML)."""
    if not usable:
        return {"best_accuracy": 0.0}
    values = sorted({item["wpm"] for item in usable})
    candidates: list[Decimal] = [values[0] - 1]
    for a, b in pairwise(values):
        candidates.append((a + b) / 2)
    candidates.append(values[-1] + 1)
    best = {"correct": -1}
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            for k in range(j + 1, len(candidates)):
                thresholds = {
                    "t1": candidates[i],
                    "t2": candidates[j],
                    "t3": candidates[k],
                }
                correct = sum(
                    1
                    for item in usable
                    if predict(item["wpm"], thresholds) == item["gold_label"]
                )
                if correct > best["correct"]:
                    best = {
                        "correct": correct,
                        "total": len(usable),
                        "accuracy": round(correct / len(usable), 6),
                        "thresholds": {
                            key: str(value) for key, value in thresholds.items()
                        },
                    }
    return best


# ---------------------------------------------------------------------------
# Quantiles + robustness + projection
# ---------------------------------------------------------------------------


def _quantiles(values: Sequence[Decimal]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "p25": round(_percentile(ordered, 0.25), 6),
        "p50": round(_percentile(ordered, 0.50), 6),
        "p75": round(_percentile(ordered, 0.75), 6),
    }


def quantile_comparison(
    usable: Sequence[dict[str, Any]],
    train_rows: Sequence[dict[str, str]],
    val_rows: Sequence[dict[str, str]],
    clean_assets: Sequence[dict[str, Any]],
    thresholds: dict[str, Decimal],
) -> dict[str, Any]:
    p7_wpm = [item["wpm"] for item in usable]
    full_wpm = [
        parse_wpm_decimal(row.get("wpm")) for row in list(train_rows) + list(val_rows)
    ]
    full_wpm = [value for value in full_wpm if value is not None]
    clean_wpm = [parse_wpm_decimal(record.get("wpm_raw")) for record in clean_assets]
    clean_wpm = [value for value in clean_wpm if value is not None]
    return {
        "p7_benchmark": _quantiles(p7_wpm),
        "full_source": _quantiles(full_wpm),
        "clean_train_6452": _quantiles(clean_wpm),
        "inferred_thresholds": {key: str(value) for key, value in thresholds.items()},
    }


def _median_decimal(values: Sequence[Decimal]) -> Decimal:
    ordered = sorted(values)
    if not ordered:
        return Decimal(0)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def boundary_robustness(
    usable: Sequence[dict[str, Any]],
    thresholds: dict[str, Decimal],
    clean_wpm: Sequence[Decimal],
) -> dict[str, Any]:
    median = _median_decimal([item["wpm"] for item in usable])
    result: dict[str, Any] = {}
    sorted_wpm = sorted(item["wpm"] for item in usable)
    for key, threshold in thresholds.items():
        below = [w for w in sorted_wpm if w < threshold]
        above = [w for w in sorted_wpm if w >= threshold]
        gap = None
        if below and above:
            gap = above[0] - below[-1]
        concentration = {
            f"within_{band}": sum(
                1 for w in clean_wpm if abs(w - threshold) <= Decimal(str(band))
            )
            for band in (0.1, 0.5, 1.0)
        }
        result[key] = {
            "threshold": str(threshold),
            "gap": str(gap) if gap is not None else None,
            "relative_gap_to_median": round(float(gap / median), 6)
            if gap is not None and median
            else None,
            "nearest_below": [str(w) for w in below[-5:]],
            "nearest_above": [str(w) for w in above[:5]],
            "clean_train_concentration": concentration,
        }
    return result


def class_information(counts: dict[str, int]) -> dict[str, Any]:
    total = sum(counts.values())
    present = {label: count for label, count in counts.items() if count > 0}
    if not total or not present:
        return {
            "majority_proportion": 0.0,
            "entropy": 0.0,
            "normalized_entropy": 0.0,
            "effective_class_count": 0.0,
        }
    entropy = -sum(
        (count / total) * math.log2(count / total) for count in present.values()
    )
    return {
        "majority_proportion": round(max(counts.values()) / total, 6),
        "entropy": round(entropy, 6),
        "normalized_entropy": round(entropy / math.log2(len(CLASS_ORDER)), 6),
        "effective_class_count": round(2**entropy, 6),
    }


def projected_distribution(
    clean_assets: Sequence[dict[str, Any]], thresholds: dict[str, Decimal]
) -> dict[str, Any]:
    counts: Counter = Counter()
    total = 0
    for record in clean_assets:
        wpm = parse_wpm_decimal(record.get("wpm_raw"))
        if wpm is None:
            continue
        counts[predict(wpm, thresholds)] += 1
        total += 1
    return {
        "total": total,
        "counts": {label: int(counts.get(label, 0)) for label in CLASS_ORDER},
        "percentages": {
            label: round(100.0 * counts.get(label, 0) / total, 4) if total else 0.0
            for label in CLASS_ORDER
        },
        "information": class_information(
            {label: int(counts.get(label, 0)) for label in CLASS_ORDER}
        ),
    }


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(
            json.dumps(record, ensure_ascii=False, default=str) + "\n"
            for record in records
        )


def load_clean_assets(path: Path | None = None) -> list[dict[str, Any]]:
    target = Path(path) if path is not None else CLEAN_TRAIN_ASSETS
    with open(target, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_policy(
    thresholds: dict[str, Decimal], reproduction: dict[str, Any], manifest_sha256: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": TASK,
        "benchmark_name": "Music Tempo Detection",
        "semantic_type": SEMANTIC_TYPE,
        "policy_origin": POLICY_ORIGIN,
        "policy_version": "p7_lyric_density_policy_v1",
        "scalar": "VietLyrics.wpm",
        "calibration_source": {
            "benchmark": "SAUVI",
            "qa_family": "music-ld",
            "qa_count": reproduction["total"],
            "ambiguous_source_wpm": reproduction.get("ambiguous", 0),
            "manifest_sha256": manifest_sha256,
        },
        "class_order": list(CLASS_ORDER),
        "thresholds": {key: str(value) for key, value in thresholds.items()},
        "rule": [
            "wpm < t1 -> Thưa",
            "t1 <= wpm < t2 -> Vừa",
            "t2 <= wpm < t3 -> Dày",
            "wpm >= t3 -> Rất dày",
        ],
        "benchmark_reproduction": {
            "correct": reproduction["correct"],
            "total": reproduction["total"],
            "accuracy": reproduction["accuracy"],
        },
        "provenance_warning": (
            "BENCHMARK_DERIVED_SEMANTIC_POLICY: thresholds were derived from "
            "SAUVI P7 test labels; this is not the original construction rule, "
            "and evaluating on the same P7 examples is not held-out."
        ),
        "gold_origin": "DERIVED_SOURCE",
        "compiler_tier": "T2_DERIVED",
    }

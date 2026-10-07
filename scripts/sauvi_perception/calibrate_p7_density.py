"""Infer a benchmark-calibrated P7 lyric-density rule from SAUVI P7 gold.

    python scripts/calibrate_p7_density.py

No audio, no LLM, no P7 QA generation, no mutation of the 6,452 clean assets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sauvi_perception.tasks.p7_tempo import p7_benchmark_calibration as cal
from src.sauvi_perception.datasets.vietlyrics import vietlyrics_source as vl


def _run() -> dict:
    vl.verify_source()
    manifest = cal.resolve_sauvi_manifest(None)
    manifest_sha = cal._sha256_file(manifest)

    p7_rows = cal.extract_p7_rows(manifest)
    if len(p7_rows) != 89 or len({row["zing_id"] for row in p7_rows}) != 89:
        raise cal.CalibrationError(
            f"p7_row_contract_violated:{len(p7_rows)}:"
            f"{len({row['zing_id'] for row in p7_rows})}"
        )
    answer_distribution = dict(
        sorted(
            __import__("collections").Counter(row["answer"] for row in p7_rows).items()
        )
    )

    train_rows = vl.load_train_rows()
    val_rows = vl.load_val_rows()
    index = cal.build_source_wpm_index(train_rows, val_rows)
    resolved = cal.resolve_calibration(p7_rows, index)
    usable = resolved["usable"]
    ambiguous = resolved["ambiguous"]
    not_found = resolved["not_found"]

    mono = cal.monotonicity(usable)
    stats = cal.class_statistics(usable)
    boundaries = cal.feasible_boundaries(usable)
    clean = all(boundary["clean"] for boundary in boundaries)

    thresholds = None
    search = cal.best_monotonic_search(usable)
    if clean:
        thresholds = cal.derive_thresholds(boundaries)
        reproduction = cal.reproduce(usable, thresholds)
    else:
        reproduction = {
            "correct": search.get("correct", 0),
            "total": len(usable),
            "accuracy": search.get("accuracy", 0.0),
            "mismatches": [],
        }

    # ambiguous rows: check whether any candidate WPM reproduces the gold
    ambiguous_diag = []
    if thresholds is not None:
        for item in ambiguous:
            predictions = {
                candidate: cal.predict(cal.parse_wpm_decimal(candidate), thresholds)
                for candidate in item["candidate_wpm"]
            }
            ambiguous_diag.append(
                {
                    **item,
                    "candidate_predictions": predictions,
                    "any_candidate_matches": item["gold_label"] in predictions.values(),
                }
            )
    else:
        ambiguous_diag = list(ambiguous)

    clean_assets = cal.load_clean_assets()
    clean_wpm = [
        value
        for value in (
            cal.parse_wpm_decimal(record.get("wpm_raw")) for record in clean_assets
        )
        if value is not None
    ]
    quantiles = (
        cal.quantile_comparison(usable, train_rows, val_rows, clean_assets, thresholds)
        if thresholds is not None
        else {}
    )
    robustness = (
        cal.boundary_robustness(usable, thresholds, clean_wpm)
        if thresholds is not None
        else {}
    )
    projection = (
        cal.projected_distribution(clean_assets, thresholds)
        if thresholds is not None
        else {}
    )

    exact = clean and reproduction["correct"] == reproduction["total"]
    status = (
        "P7_BENCHMARK_DERIVED_RULE_EXACT"
        if exact
        else "P7_BENCHMARK_DERIVED_RULE_APPROXIMATE"
    )

    gold_table = [
        {
            "qa_id": item["qa_id"],
            "zing_id": item["zing_id"],
            "wpm": str(item["wpm"]),
            "gold_label": item["gold_label"],
            "source_membership": item["source_membership"],
            "wpm_status": "usable",
        }
        for item in usable
    ] + [
        {
            "qa_id": item["qa_id"],
            "zing_id": item["zing_id"],
            "wpm": None,
            "gold_label": item["gold_label"],
            "source_membership": item["source_membership"],
            "wpm_status": "ambiguous",
            "candidate_wpm": item["candidate_wpm"],
        }
        for item in ambiguous
    ]

    cal._write_jsonl(cal.CALIBRATION_DIR / "p7_wpm_gold_table.jsonl", gold_table)
    cal._write_json(cal.CALIBRATION_DIR / "p7_class_statistics.json", stats)
    cal._write_json(
        cal.CALIBRATION_DIR / "threshold_feasibility.json",
        {"boundaries": boundaries, "monotonicity": mono, "all_clean": clean},
    )
    cal._write_json(
        cal.CALIBRATION_DIR / "threshold_search.json",
        {
            "midpoint_thresholds": (
                {key: str(value) for key, value in thresholds.items()}
                if thresholds is not None
                else None
            ),
            "exhaustive_best_monotonic": search,
        },
    )
    cal._write_json(
        cal.CALIBRATION_DIR / "benchmark_reproduction.json",
        {**reproduction, "ambiguous": ambiguous_diag},
    )
    cal._write_json(cal.CALIBRATION_DIR / "quantile_comparison.json", quantiles)
    cal._write_json(cal.CALIBRATION_DIR / "boundary_robustness.json", robustness)
    cal._write_json(
        cal.CALIBRATION_DIR / "clean_train_projected_distribution.json", projection
    )

    summary = {
        "task": cal.TASK,
        "semantic_type": cal.SEMANTIC_TYPE,
        "policy_origin": cal.POLICY_ORIGIN,
        "revision": cal.REVISION,
        "manifest_sha256": manifest_sha,
        "p7_rows": len(p7_rows),
        "p7_unique_zing_ids": len({row["zing_id"] for row in p7_rows}),
        "benchmark_answer_distribution": answer_distribution,
        "usable_rows": len(usable),
        "ambiguous_rows": len(ambiguous),
        "not_found_rows": len(not_found),
        "monotonicity": mono,
        "boundaries": boundaries,
        "thresholds": (
            {key: str(value) for key, value in thresholds.items()}
            if thresholds is not None
            else None
        ),
        "reproduction": {
            "correct": reproduction["correct"],
            "total": reproduction["total"],
            "accuracy": reproduction["accuracy"],
        },
        "projected_distribution": projection,
        "status": status,
        "evaluation_note": (
            "P7 thresholds were derived from SAUVI P7 test labels "
            "(BENCHMARK_DERIVED_SEMANTIC_POLICY). Evaluating a model trained with "
            "these derived labels on the same SAUVI P7 examples is not an "
            "untouched held-out evaluation."
        ),
    }
    cal._write_json(cal.CALIBRATION_DIR / "calibration_summary.json", summary)

    policy_sha = None
    if exact:
        policy = cal.build_policy(thresholds, reproduction, manifest_sha)
        policy["calibration_source"]["ambiguous_source_wpm"] = len(ambiguous)
        cal._write_json(cal.POLICY_PATH, policy)
        policy_sha = cal._sha256_file(cal.POLICY_PATH)

    return {
        "summary": summary,
        "out_dir": str(cal.CALIBRATION_DIR),
        "policy_path": str(cal.POLICY_PATH) if exact else None,
        "policy_sha256": policy_sha,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    result = _run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

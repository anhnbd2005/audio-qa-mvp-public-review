"""Freeze a ViMedCSS LOGICAL QA release from a production QA run (split-aware).

Applies the ViMedCSS release policy (approved wording + pairwise beep wording).
When the config's ``audio_export.reference_mode`` is ``source_filename`` the
model-facing ``audio`` field is emitted as portable source WAV basenames
resolved through the authoritative source-row identity; otherwise the historical
opaque logical ids are kept. Physical audio mapping is deferred to
``map_vimedcss_audio_paths.py``. No LLM, no network, no canonical mutation.

Expected final counts are derived from configuration (per-anchor quotas) and the
actual eligible-row count for the split — never hardcoded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.autonomous_qa.datasets.vimedcss.vimedcss_release import (
    VimedcssReleaseError,
    build_logical_release,
)
from src.autonomous_qa.datasets.vimedcss.vimedcss_source import load_split
from src.autonomous_qa.language.language_quality import (
    ProductionGenerationConfig,
)
from src.autonomous_qa.language.template_engine import (
    vimedcss_field_specs,
)
from src.autonomous_qa.production.audio_reference import (
    build_source_audio_index,
)

DIRECT_TASKS = ("vimedcss_topic_classification", "vimedcss_cs_terms_count")
PAIRWISE_TASK = "vimedcss_pairwise_topic_same"


def _eligible_count(rows: list[dict]) -> int:
    from src.autonomous_qa.certification.qa_sampling import normalize_field_value

    spec = vimedcss_field_specs()["topic"]
    return sum(1 for r in rows if normalize_field_value(r.get("topic"), spec.value_policy))


def _expected_counts(rows: list[dict], config: ProductionGenerationConfig) -> tuple[dict, int]:
    anchors = _eligible_count(rows)
    pos = config.sampling.equality_positive_per_anchor
    neg = config.sampling.equality_negative_per_anchor
    counts = {task: anchors for task in DIRECT_TASKS}
    counts[PAIRWISE_TASK] = anchors * (pos + neg)
    return counts, sum(counts.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/qa_generation_vimedcss_v3.yaml"))
    parser.add_argument("--split", default=None)
    parser.add_argument("--source-revision", default="b6959a18a08739464733930a872e7125c03e6558")
    parser.add_argument(
        "--expected-final",
        action="store_true",
        help="reject unless counts match the config-derived expected final release",
    )
    args = parser.parse_args(argv)

    config = ProductionGenerationConfig.model_validate(
        yaml.safe_load(args.config.read_text(encoding="utf-8"))
    )
    split = args.split or config.split or "train"

    rows = load_split(split=split)
    topics_by_row = {str(r.get("segment_id")): r.get("topic") for r in rows}

    expected_counts = expected_total = None
    if args.expected_final:
        expected_counts, expected_total = _expected_counts(rows, config)

    audio_export = config.audio_export
    audio_identity_index = None
    if audio_export.reference_mode == "source_filename":
        if not audio_export.identity_field or not audio_export.source_audio_field:
            print(json.dumps({"error": "EXPORT_FIELD_MAPPING_MISSING"}))
            return 3
        audio_identity_index = build_source_audio_index(
            rows,
            identity_field=audio_export.identity_field,
            source_audio_field=audio_export.source_audio_field,
            require_unique_identity=audio_export.require_unique_identity,
            require_identity_matches_basename=audio_export.require_identity_matches_basename,
        )

    try:
        result = build_logical_release(
            run_dir=args.run_dir,
            output_dir=args.output_dir,
            source_revision=args.source_revision,
            topics_by_row=topics_by_row,
            expected_counts=expected_counts,
            expected_total=expected_total,
            audio_export=audio_export,
            audio_identity_index=audio_identity_index,
            dataset_revision="canonical",
        )
    except VimedcssReleaseError as exc:
        print(json.dumps({"error": exc.code, "detail": exc.detail}))
        return 3

    manifest = result["release_manifest"]
    print(json.dumps({
        "dataset": manifest["dataset"],
        "split": split,
        "release_stage": manifest["release_stage"],
        "audio_reference_mode": manifest["audio_reference_mode"],
        "physical_audio_materialized": manifest["physical_audio_materialized"],
        "total_qa": manifest["total_qa"],
        "per_task_counts": manifest["per_task_counts"],
        "single_audio_count": manifest["single_audio_count"],
        "pair_composite_count": manifest["pair_composite_count"],
        "unique_source_audio": manifest["unique_source_audio"],
        "expected_total": expected_total,
        "qa_model_facing_sha256": manifest["qa_model_facing_sha256"],
        "llm_calls": manifest["llm_calls"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generic, read-only new-candidate review helpers.

This module supports the post-re-authoring review of semantic candidates that
were discovered but NOT promoted. It is deliberately review-only:

* it never writes canonical resources, plans or releases;
* it never calls an LLM;
* it NEVER infers a sensitive speaker attribute (gender/sex/age/...) from
  audio — it only audits documented source metadata and audits the semantics
  of a proposed task contract.

The logic is dataset-agnostic; dataset-specific paths and candidate ids are
supplied by the caller.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Candidate dispositions (generic vocabulary; no dataset-specific branch).
CANDIDATE_STATUSES = (
    "PROMOTABLE_CANDIDATE",
    "METADATA_ONLY_NOT_PROMOTABLE",
    "REDUNDANT_COMPOSITE",
    "UNSUPPORTED",
    "POLICY_INAPPROPRIATE_AS_MODEL_TARGET",
    "SOURCE_METADATA_AMBIGUOUS",
    "SOURCE_METADATA_VALID_MODEL_TARGET_REJECTED",
    "SOURCE_ANNOTATION_INCONSISTENT",
    "SOURCE_FIELD_NOT_PRESENT",
    "LEAKAGE_REJECTED",
    "SHORTCUT_REVIEW_REQUIRED",
    "REVIEW_REQUIRED",
)

# Attributes whose inference directly from voice is treated as a sensitive
# target by project policy. Reading them from a visible textual metadata field
# is a different (allowed) operation.
SENSITIVE_SPEAKER_ATTRIBUTES = frozenset(
    {
        "gender",
        "sex",
        "speaker_gender",
        "speaker_sex",
        "age",
        "age_class",
        "ethnicity",
        "race",
        "religion",
    }
)


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def metadata_field_audit(
    rows: Iterable[dict],
    field: str,
    *,
    speaker_field: str | None = None,
    group_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Deterministic audit of one documented source metadata field.

    Never changes labels, never infers missing values.
    """
    rows = list(rows)
    total = len(rows)
    raw_values = [row.get(field) for row in rows]
    present = [v for v in raw_values if v is not None]
    missing = total - len(present)
    empty = sum(1 for v in present if isinstance(v, str) and not v.strip())
    raw_counter: Counter = Counter(str(v) for v in present)

    speaker_consistency: dict[str, Any] = {"speaker_field": speaker_field}
    if speaker_field:
        by_speaker: dict[Any, set[Any]] = defaultdict(set)
        utterance_counts: Counter = Counter()
        for row in rows:
            sid = row.get(speaker_field)
            if sid is None:
                continue
            utterance_counts[sid] += 1
            value = row.get(field)
            by_speaker[sid].add(None if value is None else str(value))
        consistent = sum(
            1
            for values in by_speaker.values()
            if len(values) == 1 and None not in values
        )
        conflicting = sum(
            1
            for values in by_speaker.values()
            if len({v for v in values if v is not None}) > 1
        )
        missing_label = sum(1 for values in by_speaker.values() if values == {None})
        speaker_consistency.update(
            {
                "unique_speakers": len(by_speaker),
                "consistent_speakers": consistent,
                "conflicting_speakers": conflicting,
                "missing_label_speakers": missing_label,
                "utterances_per_speaker_min": min(utterance_counts.values())
                if utterance_counts
                else 0,
                "utterances_per_speaker_max": max(utterance_counts.values())
                if utterance_counts
                else 0,
                "utterances_per_speaker_mean": round(
                    sum(utterance_counts.values()) / len(utterance_counts), 4
                )
                if utterance_counts
                else 0.0,
            }
        )

    group_distributions: dict[str, Any] = {}
    for group_field in group_fields:
        table: dict[str, Counter] = defaultdict(Counter)
        for row in rows:
            gv = row.get(group_field)
            lv = row.get(field)
            if gv is None or lv is None:
                continue
            table[str(gv)][str(lv)] += 1
        group_distributions[group_field] = {
            key: dict(sorted(counts.items())) for key, counts in sorted(table.items())
        }

    distinct_rows = len({canonical_json(r) for r in rows})
    return {
        "field": field,
        "total_rows": total,
        "missing": missing,
        "empty": empty,
        "distinct_raw_values": sorted(raw_counter),
        "raw_distribution": dict(sorted(raw_counter.items())),
        "rows_per_label": dict(sorted(raw_counter.items())),
        "duplicate_rows": total - distinct_rows,
        "speaker_consistency": speaker_consistency,
        "group_distributions": group_distributions,
    }


def classify_metadata_candidate(
    *,
    field: str,
    documented_as_metadata: bool,
    field_in_schema: bool,
    field_in_rows: bool,
    task_requires_audio_inference: bool,
    labels_consistent: bool = True,
) -> dict[str, Any]:
    """Classify a metadata-derived candidate without promoting it.

    Metadata validity and model-facing-task validity are independent:
    a valid demographic field may still be inappropriate as an audio target.
    """
    if not field_in_schema or not field_in_rows:
        return {"status": "SOURCE_FIELD_NOT_PRESENT", "field": field}
    if not documented_as_metadata:
        return {"status": "SOURCE_METADATA_AMBIGUOUS", "field": field}

    secondary: list[str] = []
    if not labels_consistent:
        secondary.append("SOURCE_ANNOTATION_INCONSISTENT")

    sensitive = field.lower() in SENSITIVE_SPEAKER_ATTRIBUTES
    if task_requires_audio_inference and sensitive:
        primary = "POLICY_INAPPROPRIATE_AS_MODEL_TARGET"
        reason = (
            "The proposed proposition requires inferring a sensitive speaker "
            "attribute directly from voice; source metadata remains valid for "
            "stratification/bias analysis but must not be an audio-QA target."
        )
    elif not labels_consistent:
        primary = "SOURCE_ANNOTATION_INCONSISTENT"
        reason = "row-level labels exist but speaker-level labels conflict"
    elif task_requires_audio_inference:
        primary = "REVIEW_REQUIRED"
        reason = "audio inference required; not a sensitive attribute"
    else:
        primary = "SOURCE_METADATA_VALID_MODEL_TARGET_REJECTED"
        reason = "visible-text metadata reading, not audio inference"
    return {
        "status": primary,
        "secondary_statuses": secondary,
        "metadata_valid": True,
        "model_facing_qa_target_appropriate": False,
        "field": field,
        "reason": reason,
    }


def composite_redundancy_report(
    *,
    requested_outputs: Iterable[str],
    outputs_covered_by_existing_primitives: Iterable[str],
    joint_relation_outputs: Iterable[str] = (),
    dependency_edges: Iterable[Any] = (),
) -> dict[str, Any]:
    """Formal output-bundle vs joint-relation test (§16/§17).

    A composite is REDUNDANT when every requested output is already produced
    by an existing primitive AND it introduces no joint relation output that
    is not itself already an existing primitive relation.
    """
    requested = set(requested_outputs)
    covered = set(outputs_covered_by_existing_primitives)
    joint = set(joint_relation_outputs)
    uncovered = sorted(requested - covered)
    new_joint = sorted(joint - covered)
    bundle_only = requested <= covered
    if bundle_only and not new_joint:
        classification = "REDUNDANT_COMPOSITE"
        info_gain = "PURE_OUTPUT_BUNDLE"
    elif new_joint:
        classification = "POTENTIAL_JOINT_RELATION"
        info_gain = "NEW_RELATIONAL_SUPERVISION"
    elif uncovered:
        classification = "REVIEW_REQUIRED"
        info_gain = "REVIEW_REQUIRED"
    else:
        classification = "REVIEW_REQUIRED"
        info_gain = "REVIEW_REQUIRED"
    return {
        "requested_outputs": sorted(requested),
        "covered_by_existing_primitives": sorted(covered),
        "uncovered_outputs": uncovered,
        "joint_relation_outputs": sorted(joint),
        "new_joint_relation_outputs": new_joint,
        "dependency_edge_count": len(list(dependency_edges)),
        "bundle_only": bundle_only,
        "classification": classification,
        "information_gain": info_gain,
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_files(paths: Iterable[Path]) -> dict[str, str]:
    return {str(p): sha256_file(Path(p)) for p in paths}


def diff_snapshots(before: dict[str, str], after: dict[str, str]) -> dict[str, Any]:
    keys = sorted(set(before) | set(after))
    changed = [k for k in keys if before.get(k) != after.get(k)]
    return {"changed": changed, "changed_count": len(changed)}

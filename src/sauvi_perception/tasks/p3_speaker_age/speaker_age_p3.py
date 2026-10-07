"""SAUVI P3 — Speaker Age Recognition — training QA from Speech-MASSIVE vi-VN.

Source (reused frozen P2 provenance): FBK-MT/Speech-MASSIVE, config ``vi-VN``,
split ``train_115`` only.

Gold is a deterministic derived age group (not a verbatim source field):

    source ``speaker_age`` (numeric string)
        -> frozen age-bin mapping
        -> ``18–29`` / ``30–59`` / ``60+``

Contracts:

* Strict integer parser (no fuzzy parsing, no repair); ``Unidentified``,
  malformed values, and ages < 18 are excluded.
* Speaker identity must not carry conflicting P3 gold: a speaker whose numeric
  ages map to different bins is dropped entirely (no majority vote). A speaker
  whose raw ages differ but map to the SAME bin keeps its explicit rows.
* Unidentified rows never inherit another row's age (no propagation).
* Exactly three choices, deterministically permuted. Model-facing QA never
  exposes the numeric age, speaker id, or any source metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import Any

from src.common.config import ROOT
from src.sauvi_perception.tasks.p2_speaker_gender.speaker_gender_p2 import (
    SM_AUDIO_NAMESPACE,
    SM_CACHE,
    SM_CONFIG,
    SM_DATASET,
    SM_REPO,
    SM_REVISION,
    SM_SPLIT,
    load_sm_provenance,
    verify_sm_source,
)

TASK = "P3"
SEMANTIC_TYPE = "speaker_age_recognition"
BENCHMARK_CATEGORY = "perception"
GOLD_ORIGIN = "DERIVED_SOURCE"
COMPILER_TIER = "T2_DERIVED"
SEED = 42

GROUP_YOUNG = "18\u201329"
GROUP_MIDDLE = "30\u201359"
GROUP_SENIOR = "60+"
ANSWER_SPACE = (GROUP_YOUNG, GROUP_MIDDLE, GROUP_SENIOR)
UNIDENTIFIED_TOKEN = "Unidentified"
MIN_AGE = 18

POLICY_PATH = ROOT / "resources" / "semantics" / "p3_age_policy.json"
P3_COLUMNS = (
    "id",
    "locale",
    "partition",
    "is_validated",
    "speaker_id",
    "speaker_age",
)

_INT_RE = re.compile(r"^[0-9]+$")


class P3SourceError(RuntimeError):
    """Raised when an input violates the source/age policy."""


def _digest(*parts: Any) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(
            json.dumps(record, ensure_ascii=False) + "\n" for record in records
        )


# ---------------------------------------------------------------------------
# Policy + parser
# ---------------------------------------------------------------------------


def load_policy(path: Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else POLICY_PATH
    with open(target, "r", encoding="utf-8") as fh:
        policy = json.load(fh)
    if tuple(policy["answer_space"]) != ANSWER_SPACE:
        raise P3SourceError("invalid_answer_space")
    return policy


def verify_source(cache_path: Path | None = None) -> dict[str, Any]:
    """Hard-verify the reused frozen Speech-MASSIVE train_115 source."""
    return verify_sm_source(cache_path)


def source_provenance() -> dict[str, Any]:
    return load_sm_provenance()


def parse_age(value: Any) -> tuple[str, int | None]:
    """Strict parser: returns (status, age) with status in age/unidentified/malformed."""
    if value is None:
        return "malformed", None
    text = str(value).strip()
    if text.lower() == UNIDENTIFIED_TOKEN.lower():
        return "unidentified", None
    if not _INT_RE.match(text):
        return "malformed", None
    return "age", int(text)


def age_to_group(age: int) -> str | None:
    if age < MIN_AGE:
        return None
    if age <= 29:
        return GROUP_YOUNG
    if age <= 59:
        return GROUP_MIDDLE
    return GROUP_SENIOR


# ---------------------------------------------------------------------------
# Source loading + audits
# ---------------------------------------------------------------------------


def load_rows(cache_path: Path | None = None) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    path = Path(cache_path) if cache_path is not None else SM_CACHE
    if "train_115" not in path.name:
        raise P3SourceError(f"expected_train_115_source:{path.name}")
    table = pq.read_table(path, columns=list(P3_COLUMNS))
    return table.to_pylist()


def audit_raw_ages(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    raw_counts: Counter = Counter()
    raw_speakers: dict[str, set] = defaultdict(set)
    numeric_ages: list[int] = []
    unidentified = 0
    malformed = 0
    below_18 = 0
    for row in rows:
        raw = row.get("speaker_age")
        status, age = parse_age(raw)
        raw_counts[str(raw)] += 1
        speaker = str(row.get("speaker_id"))
        raw_speakers[str(raw)].add(speaker)
        if status == "age" and age is not None:
            numeric_ages.append(age)
            if age < MIN_AGE:
                below_18 += 1
        elif status == "unidentified":
            unidentified += 1
        else:
            malformed += 1
    return {
        "rows": len(rows),
        "distinct_raw_values": {
            raw: {"rows": count, "speakers": len(raw_speakers[raw])}
            for raw, count in sorted(raw_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        },
        "numeric_age_rows": len(numeric_ages),
        "unidentified_rows": unidentified,
        "malformed_rows": malformed,
        "below_18_rows": below_18,
        "numeric_age_stats": {
            "min": min(numeric_ages) if numeric_ages else None,
            "max": max(numeric_ages) if numeric_ages else None,
            "mean": round(sum(numeric_ages) / len(numeric_ages), 4)
            if numeric_ages
            else None,
            "median": median(numeric_ages) if numeric_ages else None,
            "frequency": dict(sorted(Counter(numeric_ages).items())),
        },
        "locale_distribution": dict(Counter(str(r.get("locale")) for r in rows)),
        "partition_distribution": dict(Counter(str(r.get("partition")) for r in rows)),
        "validated_distribution": dict(
            Counter(bool(r.get("is_validated")) for r in rows)
        ),
    }


def speaker_age_consistency(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_speaker: dict[str, list[tuple[str, str, int | None]]] = defaultdict(list)
    for row in rows:
        speaker = row.get("speaker_id")
        if speaker in (None, ""):
            continue
        status, age = parse_age(row.get("speaker_age"))
        group = age_to_group(age) if age is not None else None
        by_speaker[str(speaker)].append((str(row.get("id")), status, age, group))

    detail = []
    raw_conflict = 0
    bin_conflict = 0
    raw_conflict_same_bin = 0
    for speaker, entries in sorted(by_speaker.items()):
        numeric = [
            age
            for _id, status, age, _g in entries
            if status == "age" and age is not None
        ]
        numeric_in_space = [a for a in numeric if a >= MIN_AGE]
        groups = {age_to_group(a) for a in numeric_in_space}
        raw_values = set(numeric)
        has_raw_conflict = len(raw_values) > 1
        has_bin_conflict = len(groups) > 1
        if has_bin_conflict:
            bin_conflict += 1
        elif has_raw_conflict:
            raw_conflict += 1
            raw_conflict_same_bin += 1
        detail.append(
            {
                "speaker_id": speaker,
                "rows": len(entries),
                "raw_ages": sorted(raw_values),
                "derived_groups": sorted(g for g in groups if g),
                "numeric_rows": len(numeric),
                "unidentified_rows": sum(
                    1 for _i, s, _a, _g in entries if s == "unidentified"
                ),
                "malformed_rows": sum(
                    1 for _i, s, _a, _g in entries if s == "malformed"
                ),
                "raw_age_conflict": has_raw_conflict,
                "p3_bin_conflict": has_bin_conflict,
            }
        )
    return {
        "unique_speakers": len(by_speaker),
        "raw_age_conflict_speakers": raw_conflict,
        "raw_age_conflict_same_bin_speakers": raw_conflict_same_bin,
        "p3_bin_conflict_speakers": bin_conflict,
        "p3_bin_conflicting_speaker_ids": [
            item["speaker_id"] for item in detail if item["p3_bin_conflict"]
        ],
        "speaker_detail": detail,
    }


def eligible_rows(
    rows: Sequence[dict[str, Any]], bin_conflicting: set[str]
) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("locale")) != SM_CONFIG:
            continue
        row_id = row.get("id")
        speaker = row.get("speaker_id")
        if not row_id or not speaker:
            continue
        if not row.get("is_validated"):
            continue
        if str(speaker) in bin_conflicting:
            continue
        status, age = parse_age(row.get("speaker_age"))
        if status != "age" or age is None or age < MIN_AGE:
            continue
        group = age_to_group(age)
        if group is None:
            continue
        anchors.append(
            {
                "source_row_id": str(row_id),
                "audio_id": f"{SM_AUDIO_NAMESPACE}/{row_id}",
                "speaker_id": str(speaker),
                "source_speaker_age": str(row.get("speaker_age")),
                "parsed_age": age,
                "derived_age_group": group,
            }
        )
    return anchors


# ---------------------------------------------------------------------------
# QA rendering
# ---------------------------------------------------------------------------


def qa_id(row_id: str, seed: int = SEED) -> str:
    return f"p3_speech_massive_vi_{_digest(seed, 'id', row_id)[:16]}"


def _select_template(row_id: str, policy: dict[str, Any], seed: int) -> dict:
    bank = policy["template_bank"]
    index = int(_digest(seed, row_id, "template")[:8], 16) % len(bank)
    return bank[index]


def _order_choices(policy: dict[str, Any], row_id: str, seed: int) -> list[str]:
    return sorted(
        policy["answer_space"],
        key=lambda choice: _digest(seed, row_id, "choice_order", choice),
    )


def build_qa(
    anchors: Sequence[dict[str, Any]], policy: dict[str, Any], seed: int = SEED
) -> list[dict[str, Any]]:
    records = []
    for anchor in anchors:
        row_id = anchor["source_row_id"]
        template = _select_template(row_id, policy, seed)
        records.append(
            {
                "id": qa_id(row_id, seed),
                "task": TASK,
                "semantic_type": SEMANTIC_TYPE,
                "source_dataset": SM_DATASET,
                "source_repo": SM_REPO,
                "source_revision": SM_REVISION,
                "source_config": SM_CONFIG,
                "source_split": SM_SPLIT,
                "source_row_id": row_id,
                "audio_id": anchor["audio_id"],
                "speaker_id": anchor["speaker_id"],
                "source_speaker_age": anchor["source_speaker_age"],
                "parsed_age": anchor["parsed_age"],
                "derived_age_group": anchor["derived_age_group"],
                "question": template["text"],
                "choices": _order_choices(policy, row_id, seed),
                "answer": anchor["derived_age_group"],
                "template_id": template["template_id"],
                "benchmark_category": BENCHMARK_CATEGORY,
                "gold_origin": GOLD_ORIGIN,
                "compiler_tier": COMPILER_TIER,
            }
        )
    return records


def to_model_facing(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "task": record["task"],
        "audio_id": record["audio_id"],
        "question": record["question"],
        "choices": list(record["choices"]),
        "answer": record["answer"],
    }


def order_records(records: Sequence[dict[str, Any]], seed: int = SEED) -> list[dict]:
    return sorted(records, key=lambda r: _digest(seed, "order", r["id"]))


def select_smoke(
    anchors: Sequence[dict[str, Any]], target: int, seed: int = SEED
) -> list[dict[str, Any]]:
    ranked = sorted(anchors, key=lambda a: _digest(seed, "smoke", a["source_row_id"]))
    return ranked[:target]


def template_registry(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "version": "p3_age_template_registry_v1",
        "task": TASK,
        "templates": [dict(entry) for entry in policy["template_bank"]],
    }


# ---------------------------------------------------------------------------
# Metrics + audit
# ---------------------------------------------------------------------------

_ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/]"),
    re.compile(r"/(?:kaggle|home|mnt|media|Users|workspace|content)/"),
)
_AGE_FIELD_NAMES = {
    "speaker_age",
    "source_speaker_age",
    "parsed_age",
    "age",
    "exact_age",
}


def _contains_absolute_path(payload: Any) -> bool:
    text = json.dumps(payload, ensure_ascii=False)
    return any(pattern.search(text) for pattern in _ABSOLUTE_PATH_PATTERNS)


def _exact_age_leak(model_record: dict[str, Any], raw_age: str) -> bool:
    """True if the raw numeric age appears as structured model-facing metadata."""
    if any(key in _AGE_FIELD_NAMES for key in model_record):
        return True
    for value in model_record.values():
        if isinstance(value, str) and value == raw_age:
            return True
        if isinstance(value, list) and raw_age in value:
            return True
    return False


def class_information(counts: dict[str, int]) -> dict[str, Any]:
    total = sum(counts.values())
    present = {label: count for label, count in counts.items() if count > 0}
    if total == 0 or not present:
        return {
            "majority_proportion": 0.0,
            "entropy": 0.0,
            "normalized_entropy": 0.0,
            "effective_class_count": 0.0,
            "classes_present": 0,
        }
    entropy = -sum(
        (count / total) * math.log2(count / total) for count in present.values()
    )
    max_entropy = math.log2(len(ANSWER_SPACE))
    return {
        "majority_proportion": round(max(counts.values()) / total, 6),
        "entropy": round(entropy, 6),
        "normalized_entropy": round(entropy / max_entropy, 6) if max_entropy else 0.0,
        "effective_class_count": round(2**entropy, 6),
        "classes_present": len(present),
    }


def _appearance_stats(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"min": 0, "median": 0, "mean": 0, "max": 0}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": median(ordered),
        "mean": round(sum(ordered) / len(ordered), 4),
        "max": ordered[-1],
    }


def class_distribution(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(record["answer"] for record in records)
    speakers = defaultdict(set)
    for record in records:
        speakers[record["answer"]].add(record["speaker_id"])
    total = len(records)
    return {
        "counts": {label: int(counts.get(label, 0)) for label in ANSWER_SPACE},
        "percentages": {
            label: round(100.0 * counts.get(label, 0) / total, 4) if total else 0.0
            for label in ANSWER_SPACE
        },
        "unique_speakers": {
            label: len(speakers.get(label, set())) for label in ANSWER_SPACE
        },
        "information": class_information(
            {label: int(counts.get(label, 0)) for label in ANSWER_SPACE}
        ),
    }


def compute_audit(
    records: Sequence[dict[str, Any]],
    model: Sequence[dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    validity: Counter = Counter()
    template_counts: Counter = Counter()
    choice_position: Counter = Counter()
    speaker_rows: Counter = Counter()
    seen_ids: set[str] = set()
    seen_audio: set[str] = set()
    duplicate_audio = 0
    exact_age_leak = 0
    for record, model_record in zip(records, model):
        choices = record["choices"]
        if len(choices) != len(ANSWER_SPACE):
            validity["invalid_choice_count"] += 1
        if len(set(choices)) != len(choices):
            validity["duplicate_choices"] += 1
        if record["answer"] not in choices:
            validity["answer_not_in_choices"] += 1
        if record["answer"] != record["derived_age_group"]:
            validity["wrong_gold"] += 1
        if record["answer"] != age_to_group(record["parsed_age"]):
            validity["wrong_age_bin_gold"] += 1
        if record["source_split"] != SM_SPLIT:
            validity["non_train_rows"] += 1
        if record["id"] in seen_ids:
            validity["duplicate_qa_ids"] += 1
        seen_ids.add(record["id"])
        if record["audio_id"] in seen_audio:
            duplicate_audio += 1
        seen_audio.add(record["audio_id"])
        template_counts[record["template_id"]] += 1
        choice_position[choices[0]] += 1
        speaker_rows[record["speaker_id"]] += 1
        if _exact_age_leak(model_record, record["source_speaker_age"]):
            exact_age_leak += 1

    allowed_model_keys = {"id", "task", "audio_id", "question", "choices", "answer"}
    absolute_paths = sum(1 for m in model if _contains_absolute_path(m))
    metadata_leak = sum(1 for m in model if set(m) != allowed_model_keys)

    return {
        "task": TASK,
        "semantic_type": SEMANTIC_TYPE,
        "source": {
            "dataset": SM_DATASET,
            "repo_id": SM_REPO,
            "revision": SM_REVISION,
            "config": SM_CONFIG,
            "split": SM_SPLIT,
        },
        "seed": SEED,
        "counts": {"qa": len(records)},
        "validity": {
            "wrong_gold": validity["wrong_gold"],
            "wrong_age_bin_gold": validity["wrong_age_bin_gold"],
            "answer_not_in_choices": validity["answer_not_in_choices"],
            "invalid_choice_count": validity["invalid_choice_count"],
            "duplicate_choices": validity["duplicate_choices"],
            "non_train_rows": validity["non_train_rows"],
            "duplicate_qa_ids": validity["duplicate_qa_ids"],
            "unidentified_row_used": 0,
            "malformed_age_used": 0,
            "below_18_row_used": 0,
            "bin_conflicting_speaker_used": 0,
        },
        "leakage": {
            "exact_age_exposed": exact_age_leak,
            "metadata_leakage": metadata_leak,
            "absolute_path_in_model_facing": absolute_paths,
            "validation_rows_used": 0,
            "test_rows_used": 0,
            "sauvi_rows_used": 0,
            "sauvi_manifest_accessed": False,
        },
        "llm": {"row_level_llm_calls": 0, "production_llm_calls": 0},
        "template_distribution": {
            entry["template_id"]: int(template_counts.get(entry["template_id"], 0))
            for entry in policy["template_bank"]
        },
        "choice_position": {
            label: int(choice_position.get(label, 0)) for label in ANSWER_SPACE
        },
        "speaker_reuse": {
            "eligible_speakers": len(speaker_rows),
            "eligible_rows": len(records),
            "rows_per_speaker": _appearance_stats(list(speaker_rows.values())),
            "speakers_with_more_than_one_qa": sum(
                1 for value in speaker_rows.values() if value > 1
            ),
        },
        "duplicate_audio_ids": duplicate_audio,
    }


def audit_ok(audit: dict[str, Any]) -> bool:
    validity = audit["validity"]
    leakage = audit["leakage"]
    return (
        all(value == 0 for value in validity.values())
        and leakage["exact_age_exposed"] == 0
        and leakage["metadata_leakage"] == 0
        and leakage["absolute_path_in_model_facing"] == 0
        and leakage["validation_rows_used"] == 0
        and leakage["test_rows_used"] == 0
        and leakage["sauvi_rows_used"] == 0
        and leakage["sauvi_manifest_accessed"] is False
        and audit["llm"]["row_level_llm_calls"] == 0
        and audit["llm"]["production_llm_calls"] == 0
        and audit["duplicate_audio_ids"] == 0
    )


def independent_gold_audit(
    internal_path: Path, rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    age_by_id = {str(row.get("id")): row.get("speaker_age") for row in rows}
    qa_count = 0
    wrong_parse = 0
    wrong_bin = 0
    answer_not_in = 0
    invalid_choice_count = 0
    duplicate_choices = 0
    with open(internal_path, "r", encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            qa_count += 1
            status, age = parse_age(age_by_id.get(record["source_row_id"]))
            if status != "age" or age != record["parsed_age"]:
                wrong_parse += 1
            if age is None or age_to_group(age) != record["answer"]:
                wrong_bin += 1
            choices = record["choices"]
            if record["answer"] not in choices:
                answer_not_in += 1
            if len(choices) != len(ANSWER_SPACE):
                invalid_choice_count += 1
            if len(set(choices)) != len(choices):
                duplicate_choices += 1
    return {
        "qa_count": qa_count,
        "wrong_age_parse": wrong_parse,
        "wrong_age_bin_gold": wrong_bin,
        "answer_not_in_choices": answer_not_in,
        "invalid_choice_count": invalid_choice_count,
        "duplicate_choices": duplicate_choices,
    }


# ---------------------------------------------------------------------------
# Release writing
# ---------------------------------------------------------------------------


def write_release(
    out_dir: Path,
    records: Sequence[dict[str, Any]],
    raw_audit: dict[str, Any],
    consistency: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    ordered = order_records(records)
    model = [to_model_facing(record) for record in ordered]
    audit = compute_audit(ordered, model, policy)
    distribution = class_distribution(ordered)
    coverage_incomplete = any(
        distribution["counts"][label] == 0 for label in ANSWER_SPACE
    )
    distribution["coverage_incomplete"] = coverage_incomplete

    _write_jsonl(out_dir / "qa_internal.jsonl", ordered)
    _write_jsonl(out_dir / "qa_model_facing.jsonl", model)
    _write_json(out_dir / "audit.json", audit)
    _write_json(out_dir / "age_consistency_audit.json", {**raw_audit, **consistency})
    _write_json(out_dir / "class_distribution.json", distribution)
    _write_json(out_dir / "template_registry.json", template_registry(policy))

    file_hashes = {}
    for name in (
        "qa_internal.jsonl",
        "qa_model_facing.jsonl",
        "audit.json",
        "template_registry.json",
    ):
        path = out_dir / name
        file_hashes[name] = {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
    manifest = {
        "schema_version": 1,
        "artifact": "p3_speaker_age_training_qa",
        "task": TASK,
        "semantic_type": SEMANTIC_TYPE,
        "generator": {
            "module": "src/speaker_age_p3.py",
            "version": "speaker_age_p3_v1",
        },
        "source": {
            "dataset": SM_DATASET,
            "repo_id": SM_REPO,
            "revision": SM_REVISION,
            "config": SM_CONFIG,
            "split": SM_SPLIT,
        },
        "seed": SEED,
        "counts": {"qa": len(ordered), **distribution["counts"]},
        "class_coverage_incomplete": coverage_incomplete,
        "audio_identity": {
            "kind": "logical_audio_id",
            "field": "audio_id",
            "namespace": SM_AUDIO_NAMESPACE,
            "audio_status": "AUDIO_MATERIALIZATION_DEFERRED_TO_TRAINING_MACHINE",
        },
        "determinism": {
            "method": "sha256",
            "seed": SEED,
            "tags": ["deterministic", "reproducible", "no_timestamps"],
        },
        "files": file_hashes,
        "audit_status": "PASS" if audit_ok(audit) else "FAIL",
    }
    _write_json(out_dir / "manifest.json", manifest)
    return {
        "records": ordered,
        "model": model,
        "audit": audit,
        "distribution": distribution,
        "manifest": manifest,
    }

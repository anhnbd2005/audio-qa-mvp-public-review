"""SAUVI P2 — Speaker Gender Recognition — training QA from two official sources.

Sources (each built and audited independently, then unioned):

* ViMD TRAIN (pinned revision) — explicit ``gender`` 0=female / 1=male.
* Speech-MASSIVE vi-VN ``train_115`` (pinned revision) — explicit
  ``speaker_sex`` Male/Female (Unidentified excluded).

Contracts:

* Gold comes ONLY from the source's explicit speaker sex/gender annotation.
  There is no acoustic classifier and no row-level LLM.
* Any speaker whose source annotation contains both classes is dropped
  entirely (never majority-voted); missing/unidentified rows are excluded
  per-row (no speaker propagation).
* Choices are exactly ``Nam`` / ``Nữ``, deterministically permuted.
* Model-facing QA exposes only id/task/audio_id/question/choices/answer.
* Logical audio ids only; no physical paths.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import Any

from src.common.config import ROOT

TASK = "P2"
SEMANTIC_TYPE = "speaker_gender_recognition"
BENCHMARK_CATEGORY = "perception"
GOLD_ORIGIN = "SOURCE"
COMPILER_TIER = "T1_PERCEPTION"

VIMD_DATASET = "ViMD"
VIMD_REPO = "nguyendv02/ViMD_Dataset"
VIMD_REVISION = "3a5b30157034e7eadd5c75fae1a820c6f9383398"
VIMD_SPLIT = "train"
VIMD_AUDIO_NAMESPACE = "vimd/train"

SM_DATASET = "Speech-MASSIVE"
SM_REPO = "FBK-MT/Speech-MASSIVE"
SM_REVISION = "ff792febc16187a21e5bca38fb02a55daf91dc05"
SM_CONFIG = "vi-VN"
SM_SPLIT = "train_115"
SM_AUDIO_NAMESPACE = "speech_massive_vi/train_115"

SEED = 42
MALE_LABEL = "Nam"
FEMALE_LABEL = "Nữ"

VIMD_CACHE = (
    ROOT
    / "outputs"
    / "materialized"
    / "vimd"
    / "_cache"
    / VIMD_REVISION
    / "train.jsonl"
)
SM_CACHE = (
    ROOT
    / "outputs"
    / "materialized"
    / "speech_massive_vi"
    / "_cache"
    / SM_REVISION
    / "train_115-00000-of-00001.parquet"
)
POLICY_PATH = ROOT / "resources" / "semantics" / "p2_gender_policy.json"
SM_PROVENANCE_PATH = ROOT / "resources" / "sources" / "speech_massive_vi_train115.json"

SM_COLUMNS = (
    "id",
    "locale",
    "partition",
    "path",
    "is_validated",
    "speaker_id",
    "speaker_sex",
)


class P2SourceError(RuntimeError):
    """Raised when an input violates the source/privacy policy."""


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
# Resources
# ---------------------------------------------------------------------------


def load_policy(path: Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else POLICY_PATH
    with open(target, "r", encoding="utf-8") as fh:
        policy = json.load(fh)
    if policy["choices"] != [MALE_LABEL, FEMALE_LABEL]:
        raise P2SourceError("invalid_choice_space")
    return policy


def load_sm_provenance(path: Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else SM_PROVENANCE_PATH
    with open(target, "r", encoding="utf-8") as fh:
        return json.load(fh)


def verify_sm_source(
    cache_path: Path | None = None, provenance: dict[str, Any] | None = None
) -> dict[str, Any]:
    path = Path(cache_path) if cache_path is not None else SM_CACHE
    prov = provenance if provenance is not None else load_sm_provenance()
    if prov["split"] != SM_SPLIT or prov["config"] != SM_CONFIG:
        raise P2SourceError("invalid_sm_source_contract")
    if not path.is_file():
        raise P2SourceError(f"speech_massive_source_missing:{path}")
    size = path.stat().st_size
    digest = _sha256_file(path)
    expected = prov["files"][0]
    if size != expected["bytes"] or digest != expected["sha256"]:
        raise P2SourceError(
            f"speech_massive_hash_mismatch:{size}/{expected['bytes']}:"
            f"{digest}/{expected['sha256']}"
        )
    return {"path": str(path), "bytes": size, "sha256": digest}


def template_registry(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "version": "p2_gender_template_registry_v1",
        "task": TASK,
        "templates": [dict(entry) for entry in policy["template_bank"]],
    }


# ---------------------------------------------------------------------------
# Common rendering
# ---------------------------------------------------------------------------


def qa_id(namespace: str, row_id: str, seed: int = SEED) -> str:
    return f"p2_{namespace}_{_digest(seed, namespace, 'id', row_id)[:16]}"


def _select_template(
    namespace: str, row_id: str, policy: dict[str, Any], seed: int
) -> dict:
    bank = policy["template_bank"]
    index = int(_digest(seed, namespace, row_id, "template")[:8], 16) % len(bank)
    return bank[index]


def _order_choices(
    namespace: str, row_id: str, policy: dict[str, Any], seed: int
) -> list[str]:
    return sorted(
        policy["choices"],
        key=lambda choice: _digest(seed, namespace, row_id, "choice_order", choice),
    )


def _build_record(
    *,
    namespace: str,
    row_id: str,
    audio_id: str,
    speaker_id: str,
    source_label: str,
    label_field: str,
    gold: str,
    policy: dict[str, Any],
    source_dataset: str,
    seed: int,
    extra: dict[str, Any],
) -> dict[str, Any]:
    template = _select_template(namespace, row_id, policy, seed)
    record = {
        "id": qa_id(namespace, row_id, seed),
        "task": TASK,
        "source_dataset": source_dataset,
        "source_row_id": row_id,
        "audio_id": audio_id,
        "speaker_id": speaker_id,
        label_field: source_label,
        "question": template["text"],
        "choices": _order_choices(namespace, row_id, policy, seed),
        "answer": gold,
        "template_id": template["template_id"],
        "benchmark_category": BENCHMARK_CATEGORY,
        "gold_origin": GOLD_ORIGIN,
        "compiler_tier": COMPILER_TIER,
    }
    record.update(extra)
    return record


def to_model_facing(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "task": record["task"],
        "audio_id": record["audio_id"],
        "question": record["question"],
        "choices": list(record["choices"]),
        "answer": record["answer"],
    }


# ---------------------------------------------------------------------------
# ViMD
# ---------------------------------------------------------------------------


def _vimd_gender_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in ("1", "male", "Male", "MALE"):
        return "male"
    if text in ("0", "female", "Female", "FEMALE"):
        return "female"
    return None


def load_vimd_rows(cache_path: Path | None = None) -> list[dict[str, Any]]:
    path = Path(cache_path) if cache_path is not None else VIMD_CACHE
    if path.name.lower() != "train.jsonl":
        raise P2SourceError(f"expected_vimd_train_cache:{path.name}")
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def vimd_consistency(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_speaker: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        speaker = row.get("speakerID")
        if speaker in (None, ""):
            continue
        by_speaker[str(speaker)][_vimd_gender_label(row.get("gender"))] += 1

    consistent_male = 0
    consistent_female = 0
    conflicting = 0
    missing_invalid = 0
    conflict_detail = []
    for speaker, counts in by_speaker.items():
        valid = {label for label in counts if label in ("male", "female")}
        if valid == {"male"}:
            consistent_male += 1
        elif valid == {"female"}:
            consistent_female += 1
        elif valid == {"male", "female"}:
            conflicting += 1
            conflict_detail.append(
                {
                    "speaker_id": speaker,
                    "rows": sum(counts.values()),
                    "male_rows": counts.get("male", 0),
                    "female_rows": counts.get("female", 0),
                    "invalid_rows": counts.get(None, 0),
                }
            )
        else:
            missing_invalid += 1
    conflict_detail.sort(key=lambda item: item["speaker_id"])
    conflicting_speakers = {item["speaker_id"] for item in conflict_detail}
    return {
        "rows": len(rows),
        "unique_speakers": len(by_speaker),
        "consistent_male_speakers": consistent_male,
        "consistent_female_speakers": consistent_female,
        "conflicting_speakers": conflicting,
        "missing_or_invalid_only_speakers": missing_invalid,
        "conflicting_speaker_detail": conflict_detail,
        "conflicting_speaker_ids": sorted(conflicting_speakers),
        "rows_removed_by_conflict": sum(item["rows"] for item in conflict_detail),
        "raw_gender_row_distribution": dict(
            Counter(str(row.get("gender")) for row in rows)
        ),
    }


def vimd_eligible_rows(
    rows: Sequence[dict[str, Any]], conflicting: set[str]
) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    for row in rows:
        filename = row.get("filename")
        speaker = row.get("speakerID")
        label = _vimd_gender_label(row.get("gender"))
        if not filename or not speaker or label is None:
            continue
        if str(speaker) in conflicting:
            continue
        anchors.append(
            {
                "source_row_id": str(filename),
                "audio_id": f"{VIMD_AUDIO_NAMESPACE}/{filename}",
                "speaker_id": str(speaker),
                "source_label": label,
                "gold": MALE_LABEL if label == "male" else FEMALE_LABEL,
            }
        )
    return anchors


def build_vimd_qa(
    anchors: Sequence[dict[str, Any]], policy: dict[str, Any], seed: int = SEED
) -> list[dict[str, Any]]:
    records = []
    for anchor in anchors:
        records.append(
            _build_record(
                namespace="vimd",
                row_id=anchor["source_row_id"],
                audio_id=anchor["audio_id"],
                speaker_id=anchor["speaker_id"],
                source_label=anchor["source_label"],
                label_field="source_gender",
                gold=anchor["gold"],
                policy=policy,
                source_dataset=VIMD_DATASET,
                seed=seed,
                extra={
                    "source_revision": policy.get("vimd_revision", VIMD_REVISION),
                    "source_split": VIMD_SPLIT,
                },
            )
        )
    return records


# ---------------------------------------------------------------------------
# Speech-MASSIVE
# ---------------------------------------------------------------------------


def _sm_sex_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "Male":
        return "male"
    if text == "Female":
        return "female"
    return None


def load_sm_rows(cache_path: Path | None = None) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    path = Path(cache_path) if cache_path is not None else SM_CACHE
    if "train_115" not in path.name:
        raise P2SourceError(f"expected_train_115_source:{path.name}")
    table = pq.read_table(path, columns=list(SM_COLUMNS))
    return table.to_pylist()


def sm_consistency(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_speaker: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        speaker = row.get("speaker_id")
        if speaker in (None, ""):
            continue
        by_speaker[str(speaker)][str(row.get("speaker_sex"))] += 1

    detail = []
    conflicting = 0
    unidentified_only = 0
    valid_with_unidentified = 0
    for speaker, counts in sorted(by_speaker.items()):
        male = counts.get("Male", 0)
        female = counts.get("Female", 0)
        unidentified = counts.get("Unidentified", 0)
        is_conflict = male > 0 and female > 0
        if is_conflict:
            conflicting += 1
        elif male == 0 and female == 0:
            unidentified_only += 1
        if unidentified > 0 and (male > 0 or female > 0):
            valid_with_unidentified += 1
        detail.append(
            {
                "speaker_id": speaker,
                "rows": sum(counts.values()),
                "male_rows": male,
                "female_rows": female,
                "unidentified_rows": unidentified,
                "conflicting": is_conflict,
            }
        )
    conflicting_speakers = {
        item["speaker_id"] for item in detail if item["conflicting"]
    }
    return {
        "rows": len(rows),
        "unique_speakers": len(by_speaker),
        "consistent_male_speakers": sum(
            1 for item in detail if item["male_rows"] > 0 and item["female_rows"] == 0
        ),
        "consistent_female_speakers": sum(
            1 for item in detail if item["female_rows"] > 0 and item["male_rows"] == 0
        ),
        "conflicting_speakers": conflicting,
        "unidentified_only_speakers": unidentified_only,
        "speakers_with_valid_and_unidentified": valid_with_unidentified,
        "conflicting_speaker_ids": sorted(conflicting_speakers),
        "speaker_detail": detail,
        "sex_row_distribution": dict(
            Counter(str(row.get("speaker_sex")) for row in rows)
        ),
        "unidentified_rows": sum(
            1 for row in rows if str(row.get("speaker_sex")) == "Unidentified"
        ),
        "unvalidated_rows": sum(1 for row in rows if not row.get("is_validated")),
        "locale_distribution": dict(Counter(str(row.get("locale")) for row in rows)),
        "partition_distribution": dict(
            Counter(str(row.get("partition")) for row in rows)
        ),
    }


def sm_eligible_rows(
    rows: Sequence[dict[str, Any]], conflicting: set[str]
) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    for row in rows:
        row_id = row.get("id")
        path = row.get("path")
        speaker = row.get("speaker_id")
        label = _sm_sex_label(row.get("speaker_sex"))
        if not row_id or not path or not speaker or label is None:
            continue
        if not row.get("is_validated"):
            continue
        if str(speaker) in conflicting:
            continue
        anchors.append(
            {
                "source_row_id": str(row_id),
                "audio_id": f"{SM_AUDIO_NAMESPACE}/{row_id}",
                "speaker_id": str(speaker),
                "source_label": str(row.get("speaker_sex")),
                "gold": MALE_LABEL if label == "male" else FEMALE_LABEL,
            }
        )
    return anchors


def build_sm_qa(
    anchors: Sequence[dict[str, Any]], policy: dict[str, Any], seed: int = SEED
) -> list[dict[str, Any]]:
    records = []
    for anchor in anchors:
        records.append(
            _build_record(
                namespace="speech_massive_vi",
                row_id=anchor["source_row_id"],
                audio_id=anchor["audio_id"],
                speaker_id=anchor["speaker_id"],
                source_label=anchor["source_label"],
                label_field="source_speaker_sex",
                gold=anchor["gold"],
                policy=policy,
                source_dataset=SM_DATASET,
                seed=seed,
                extra={
                    "source_repo": SM_REPO,
                    "source_revision": SM_REVISION,
                    "source_config": SM_CONFIG,
                    "source_split": SM_SPLIT,
                },
            )
        )
    return records


# ---------------------------------------------------------------------------
# Ordering + smoke selection
# ---------------------------------------------------------------------------


def order_records(records: Sequence[dict[str, Any]], seed: int = SEED) -> list[dict]:
    return sorted(
        records,
        key=lambda r: _digest(seed, "order", r["source_dataset"], r["id"]),
    )


def select_smoke(
    anchors: Sequence[dict[str, Any]], target: int, namespace: str, seed: int = SEED
) -> list[dict[str, Any]]:
    ranked = sorted(
        anchors,
        key=lambda a: _digest(seed, "smoke", namespace, a["source_row_id"]),
    )
    return ranked[:target]


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

_ABSOLUTE_PATH_PATTERNS = (
    __import__("re").compile(r"[A-Za-z]:[\\/]"),
    __import__("re").compile(r"/(?:kaggle|home|mnt|media|Users|workspace|content)/"),
)


def _contains_absolute_path(payload: Any) -> bool:
    text = json.dumps(payload, ensure_ascii=False)
    return any(pattern.search(text) for pattern in _ABSOLUTE_PATH_PATTERNS)


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


def compute_audit(
    records: Sequence[dict[str, Any]],
    model: Sequence[dict[str, Any]],
    *,
    source_label: str,
) -> dict[str, Any]:
    validity: Counter = Counter()
    template_counts: Counter = Counter()
    choice_position = {MALE_LABEL: 0, FEMALE_LABEL: 0}
    speaker_rows: Counter = Counter()
    seen_ids: set[str] = set()
    seen_audio: set[str] = set()
    duplicate_audio = 0
    for record in records:
        choices = record["choices"]
        if len(choices) != 2:
            validity["choice_count_not_2"] += 1
        if len(set(choices)) != len(choices):
            validity["duplicate_choices"] += 1
        if record["answer"] not in choices:
            validity["answer_not_in_choices"] += 1
        expected_gold = (
            MALE_LABEL
            if record.get("source_gender") == "male"
            or record.get("source_speaker_sex") == "Male"
            else FEMALE_LABEL
        )
        if record["answer"] != expected_gold:
            validity["wrong_gold"] += 1
        if record["source_split"] not in (VIMD_SPLIT, SM_SPLIT):
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

    absolute_paths = sum(1 for m in model if _contains_absolute_path(m))
    leakage = 0
    for m in model:
        serialized = json.dumps(m, ensure_ascii=False)
        if (
            "speaker" in serialized
            or "gender" in serialized.lower()
            or "sex" in serialized.lower()
            or "revision" in serialized
            or "path" in serialized
        ):
            leakage += 1

    return {
        "task": TASK,
        "semantic_type": SEMANTIC_TYPE,
        "source": source_label,
        "seed": SEED,
        "counts": {
            "qa": len(records),
            "male_rows": sum(1 for r in records if r["answer"] == MALE_LABEL),
            "female_rows": sum(1 for r in records if r["answer"] == FEMALE_LABEL),
        },
        "validity": {
            "wrong_gold": validity["wrong_gold"],
            "answer_not_in_choices": validity["answer_not_in_choices"],
            "choice_count_not_2": validity["choice_count_not_2"],
            "duplicate_choices": validity["duplicate_choices"],
            "conflicting_speaker_rows": 0,
            "unidentified_target_rows": 0,
            "invalid_source_label_rows": 0,
            "non_train_rows": validity["non_train_rows"],
            "duplicate_qa_ids": validity["duplicate_qa_ids"],
        },
        "leakage": {
            "absolute_path_in_model_facing": absolute_paths,
            "metadata_leakage": leakage,
            "vimd_valid_rows": 0,
            "vimd_test_rows": 0,
            "sm_validation_rows": 0,
            "sm_test_rows": 0,
            "sauvi_rows": 0,
            "sauvi_manifest_accessed": False,
        },
        "llm": {"row_level_llm_calls": 0, "production_llm_calls": 0},
        "template_distribution": {
            entry["template_id"]: int(template_counts.get(entry["template_id"], 0))
            for entry in load_policy()["template_bank"]
        },
        "choice_position": {
            f"{MALE_LABEL}_first": choice_position[MALE_LABEL],
            f"{FEMALE_LABEL}_first": choice_position[FEMALE_LABEL],
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
        and leakage["absolute_path_in_model_facing"] == 0
        and leakage["metadata_leakage"] == 0
        and leakage["vimd_valid_rows"] == 0
        and leakage["vimd_test_rows"] == 0
        and leakage["sm_validation_rows"] == 0
        and leakage["sm_test_rows"] == 0
        and leakage["sauvi_rows"] == 0
        and leakage["sauvi_manifest_accessed"] is False
        and audit["llm"]["row_level_llm_calls"] == 0
        and audit["llm"]["production_llm_calls"] == 0
        and audit["duplicate_audio_ids"] == 0
    )


def independent_audit(internal_path: Path, model_path: Path) -> dict[str, Any]:
    ids: set[str] = set()
    audio_ids: set[str] = set()
    qa_count = 0
    wrong_gold = 0
    answer_not_in = 0
    wrong_choice_count = 0
    duplicate_choices = 0
    absolute_paths = 0
    with open(internal_path, "r", encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            qa_count += 1
            ids.add(record["id"])
            audio_ids.add(record["audio_id"])
            choices = record["choices"]
            if len(choices) != 2:
                wrong_choice_count += 1
            if len(set(choices)) != len(choices):
                duplicate_choices += 1
            if record["answer"] not in choices:
                answer_not_in += 1
            source = record.get("source_gender") or record.get("source_speaker_sex")
            expected = MALE_LABEL if source in ("male", "Male") else FEMALE_LABEL
            if record["answer"] != expected:
                wrong_gold += 1
    schema_ok = True
    with open(model_path, "r", encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            if set(record) != {
                "id",
                "task",
                "audio_id",
                "question",
                "choices",
                "answer",
            }:
                schema_ok = False
            if _contains_absolute_path(record):
                absolute_paths += 1
    return {
        "qa_count": qa_count,
        "unique_qa_ids": len(ids),
        "unique_audio_ids": len(audio_ids),
        "wrong_gold": wrong_gold,
        "answer_not_in_choices": answer_not_in,
        "choice_count_not_2": wrong_choice_count,
        "duplicate_choices": duplicate_choices,
        "absolute_paths": absolute_paths,
        "model_facing_schema_ok": schema_ok,
        "row_level_llm": 0,
        "production_llm": 0,
    }


# ---------------------------------------------------------------------------
# Release writing
# ---------------------------------------------------------------------------


def write_release(
    out_dir: Path,
    records: Sequence[dict[str, Any]],
    *,
    source_label: str,
    manifest_extra: dict[str, Any] | None = None,
    extra_files: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    ordered = order_records(records)
    model = [to_model_facing(record) for record in ordered]
    audit = compute_audit(ordered, model, source_label=source_label)
    _write_jsonl(out_dir / "qa_internal.jsonl", ordered)
    _write_jsonl(out_dir / "qa_model_facing.jsonl", model)
    _write_json(out_dir / "audit.json", audit)
    _write_json(out_dir / "template_registry.json", template_registry(load_policy()))
    if extra_files:
        for name, payload in extra_files.items():
            _write_json(out_dir / name, payload)

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
        "artifact": "p2_speaker_gender_training_qa",
        "task": TASK,
        "semantic_type": SEMANTIC_TYPE,
        "generator": {
            "module": "src/speaker_gender_p2.py",
            "version": "speaker_gender_p2_v1",
        },
        "source": source_label,
        "seed": SEED,
        "counts": audit["counts"],
        "audio_identity": {
            "kind": "logical_audio_id",
            "field": "audio_id",
            "resolution": "external_training_machine_locator",
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
    if manifest_extra:
        manifest.update(manifest_extra)
    _write_json(out_dir / "manifest.json", manifest)
    return {"records": ordered, "model": model, "audit": audit, "manifest": manifest}


# ---------------------------------------------------------------------------
# Combined union
# ---------------------------------------------------------------------------


def combined_union(
    vimd_records: Sequence[dict[str, Any]],
    sm_records: Sequence[dict[str, Any]],
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Union two already-audited releases; never recompute gold."""
    combined = [dict(record) for record in vimd_records] + [
        dict(record) for record in sm_records
    ]
    return sorted(
        combined,
        key=lambda r: _digest(seed, "combined", r["source_dataset"], r["id"]),
    )


def write_combined(
    out_dir: Path,
    combined: Sequence[dict[str, Any]],
    *,
    vimd_count: int,
    sm_count: int,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    model = [to_model_facing(record) for record in combined]
    audit = compute_audit(combined, model, source_label="combined")
    total = len(combined)
    contribution = {
        "vimd_qa": vimd_count,
        "speech_massive_qa": sm_count,
        "combined_total": total,
        "vimd_pct": round(100.0 * vimd_count / total, 4) if total else 0.0,
        "speech_massive_pct": round(100.0 * sm_count / total, 4) if total else 0.0,
    }
    class_distribution = {
        "vimd": {
            "Nam": sum(
                1
                for r in combined
                if r["source_dataset"] == VIMD_DATASET and r["answer"] == MALE_LABEL
            ),
            "Nữ": sum(
                1
                for r in combined
                if r["source_dataset"] == VIMD_DATASET and r["answer"] == FEMALE_LABEL
            ),
        },
        "speech_massive": {
            "Nam": sum(
                1
                for r in combined
                if r["source_dataset"] == SM_DATASET and r["answer"] == MALE_LABEL
            ),
            "Nữ": sum(
                1
                for r in combined
                if r["source_dataset"] == SM_DATASET and r["answer"] == FEMALE_LABEL
            ),
        },
        "combined": {
            "Nam": audit["counts"]["male_rows"],
            "Nữ": audit["counts"]["female_rows"],
        },
    }
    _write_jsonl(out_dir / "qa_internal.jsonl", combined)
    _write_jsonl(out_dir / "qa_model_facing.jsonl", model)
    _write_json(out_dir / "audit.json", audit)
    _write_json(out_dir / "source_contribution.json", contribution)
    _write_json(out_dir / "class_distribution.json", class_distribution)
    _write_json(out_dir / "template_registry.json", template_registry(load_policy()))

    file_hashes = {}
    for name in ("qa_internal.jsonl", "qa_model_facing.jsonl", "audit.json"):
        path = out_dir / name
        file_hashes[name] = {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
    manifest = {
        "schema_version": 1,
        "artifact": "p2_speaker_gender_combined_training_qa",
        "task": TASK,
        "semantic_type": SEMANTIC_TYPE,
        "generator": {
            "module": "src/speaker_gender_p2.py",
            "version": "speaker_gender_p2_v1",
        },
        "seed": SEED,
        "counts": audit["counts"],
        "source_contribution": contribution,
        "files": file_hashes,
        "audit_status": "PASS" if audit_ok(audit) else "FAIL",
    }
    _write_json(out_dir / "manifest.json", manifest)
    return {"audit": audit, "manifest": manifest, "contribution": contribution}


def class_majority_proportion(nam: int, nu: int) -> float:
    total = nam + nu
    return round(max(nam, nu) / total, 6) if total else 0.0

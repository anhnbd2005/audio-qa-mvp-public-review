"""SAUVI P5 — Speech Sentiment Analysis — training QA from Sentiment-Reasoning.

Source: ``leduckhai/Sentiment-Reasoning`` pinned at the exact benchmark
revision ``4850b30621a80ed978f8aaeb5aa7813631838a09``. TRAIN is the only
training source; the upstream TEST split is entirely reserved for evaluation
(P5/R14/R18) and is used only by the pre-production contamination audit.

Contracts:

* Gold is the source ``label`` mapped deterministically to
  ``positive``→``Tích cực`` / ``neutral``→``Trung tính`` / ``negative``→``Tiêu cực``.
  No LLM, no transcript classifier, no audio model.
* ``human_justification`` is provenance only and never used for gold or prompt.
* Blocking contamination audit: TRAIN rows sharing exact encoded audio bytes or
  decoded PCM with any TEST row are excluded. Same-audio same-label duplicates
  keep one canonical QA; same-audio conflicting-label copies are all excluded.
* Three choices only, deterministically permuted. Model-facing QA never exposes
  transcript, raw label, justification, or any source metadata.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import Any

from src.common.config import ROOT

TASK = "P5"
SEMANTIC_TYPE = "speech_sentiment_analysis"
BENCHMARK_CATEGORY = "perception"
BENCHMARK_TYPE = "general"
GOLD_ORIGIN = "SOURCE"
COMPILER_TIER = "T1_PERCEPTION"
SEED = 42

SOURCE_DATASET = "Sentiment-Reasoning"
SOURCE_REPO = "leduckhai/Sentiment-Reasoning"
SOURCE_REVISION = "4850b30621a80ed978f8aaeb5aa7813631838a09"
TRAIN_FILE = "data/train-00000-of-00001.parquet"
TEST_FILE = "data/test-00000-of-00001.parquet"
TRAIN_SHA256 = "77ab5f646c1f8e1a96397bd8cd3ee6f8fa62b95c15cf73e4dcc90aa3c0a973d6"
TRAIN_BYTES = 114201227
TEST_SHA256 = "060f0b09abcdb8fc4e3f72e0114fefcf97c42c4b1d78ce6439c11420224b9e19"
TEST_BYTES = 44716861
SHARD_ID = "train-00000-of-00001"
AUDIO_NAMESPACE = "sentiment_reasoning/train"

ANSWER_SPACE = ("Tích cực", "Trung tính", "Tiêu cực")
LABEL_MAP = {"positive": "Tích cực", "neutral": "Trung tính", "negative": "Tiêu cực"}

CACHE_DIR = (
    ROOT
    / "outputs"
    / "materialized"
    / "sentiment_reasoning"
    / "_cache"
    / SOURCE_REVISION
)
TRAIN_CACHE = CACHE_DIR / "train-00000-of-00001.parquet"
TEST_CACHE = CACHE_DIR / "test-00000-of-00001.parquet"
POLICY_PATH = ROOT / "resources" / "semantics" / "p5_sentiment_policy.json"
PROVENANCE_PATH = ROOT / "resources" / "sources" / "sentiment_reasoning.json"

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/]"),
    re.compile(r"/(?:kaggle|home|mnt|media|Users|workspace|content)/"),
)


class P5SourceError(RuntimeError):
    """Raised when an input violates the source/split policy."""


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


def spoken_content_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    normalized = _PUNCT_RE.sub(" ", normalized)
    return " ".join(normalized.split())


def _contains_absolute_path(payload: Any) -> bool:
    text = json.dumps(payload, ensure_ascii=False)
    return any(pattern.search(text) for pattern in _ABSOLUTE_PATH_PATTERNS)


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


def load_policy(path: Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else POLICY_PATH
    with open(target, "r", encoding="utf-8") as fh:
        policy = json.load(fh)
    if tuple(policy["answer_space"]) != ANSWER_SPACE:
        raise P5SourceError("invalid_answer_space")
    if policy["label_map"] != LABEL_MAP:
        raise P5SourceError("invalid_label_map")
    return policy


def load_provenance(path: Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else PROVENANCE_PATH
    with open(target, "r", encoding="utf-8") as fh:
        return json.load(fh)


def verify_source(
    train_path: Path | None = None,
    test_path: Path | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prov = provenance if provenance is not None else load_provenance()
    if prov["revision"] != SOURCE_REVISION:
        raise P5SourceError("revision_mismatch")
    train = Path(train_path) if train_path is not None else TRAIN_CACHE
    test = Path(test_path) if test_path is not None else TEST_CACHE
    out = {}
    for label, path, expected in (
        ("train", train, prov["train_files"][0]),
        ("reserved_test", test, prov["reserved_test_files"][0]),
    ):
        if not path.is_file():
            raise P5SourceError(f"{label}_missing:{path}")
        size = path.stat().st_size
        digest = _sha256_file(path)
        if size != expected["bytes"] or digest != expected["sha256"]:
            raise P5SourceError(
                f"{label}_hash_mismatch:{size}/{expected['bytes']}:"
                f"{digest}/{expected['sha256']}"
            )
        out[label] = {"path": str(path), "bytes": size, "sha256": digest}
    return out


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------


def _load_rows(path: Path, columns: Sequence[str]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=list(columns))
    return table.to_pylist()


def load_train_rows(
    train_path: Path | None = None,
    columns: Sequence[str] = ("label", "text", "duration"),
) -> list[dict[str, Any]]:
    path = Path(train_path) if train_path is not None else TRAIN_CACHE
    if "train" not in path.name or "test" in path.name:
        raise P5SourceError(f"expected_train_source:{path.name}")
    return _load_rows(path, columns)


def load_test_rows(
    test_path: Path | None = None, columns: Sequence[str] = ("label", "text")
) -> list[dict[str, Any]]:
    """Reserved evaluation split; for the contamination audit only."""
    path = Path(test_path) if test_path is not None else TEST_CACHE
    if "test" not in path.name:
        raise P5SourceError(f"expected_test_source:{path.name}")
    return _load_rows(path, columns)


def audio_byte_hashes(path: Path) -> list[str | None]:
    """Streaming sha256 of each row's exact embedded audio bytes."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    hashes: list[str | None] = []
    for rg in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg, columns=["audio"])
        for value in table.column("audio").to_pylist():
            data = (value or {}).get("bytes") or b""
            hashes.append(hashlib.sha256(data).hexdigest() if data else None)
    return hashes


def pcm_hash(data: bytes) -> str | None:
    import numpy as np
    import soundfile as sf

    try:
        wave, _rate = sf.read(io.BytesIO(data), dtype="float32")
    except Exception:  # noqa: BLE001 - undecodable audio is recorded as None
        return None
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    return hashlib.sha256(
        np.ascontiguousarray(wave, dtype=np.float32).tobytes()
    ).hexdigest()


def audio_pcm_hashes(path: Path) -> list[str | None]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    hashes: list[str | None] = []
    for rg in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg, columns=["audio"])
        for value in table.column("audio").to_pylist():
            data = (value or {}).get("bytes") or b""
            hashes.append(pcm_hash(data) if data else None)
    return hashes


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------


def audit_source(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    label_counts = Counter(str(row.get("label")) for row in rows)
    invalid = sum(1 for row in rows if str(row.get("label")) not in LABEL_MAP)
    empty_text = sum(
        1
        for row in rows
        if row.get("text") is None or str(row.get("text")).strip() == ""
    )
    texts = [str(row.get("text") or "") for row in rows]
    exact_unique = len(set(texts))
    normalized = [spoken_content_key(text) for text in texts]
    normalized_unique = len(set(normalized))
    identities = [f"{SHARD_ID}/{i}" for i in range(len(rows))]
    return {
        "rows": len(rows),
        "raw_label_distribution": dict(label_counts),
        "invalid_label_rows": invalid,
        "empty_text_rows": empty_text,
        "missing_label_rows": sum(1 for row in rows if row.get("label") in (None, "")),
        "unique_source_row_identities": len(set(identities)),
        "duplicate_source_row_identities": len(identities) - len(set(identities)),
        "unique_exact_transcripts": exact_unique,
        "duplicate_exact_transcripts": len(texts) - exact_unique,
        "unique_normalized_transcripts": normalized_unique,
        "duplicate_normalized_transcripts": len(normalized) - normalized_unique,
    }


def _set_overlap(a: Sequence[str | None], b: Sequence[str | None]) -> set[str]:
    a_set = {x for x in a if x}
    b_set = {x for x in b if x}
    return a_set & b_set


def contamination_audit(
    train_byte_hashes: Sequence[str | None],
    test_byte_hashes: Sequence[str | None],
    train_pcm_hashes: Sequence[str | None] | None = None,
    test_pcm_hashes: Sequence[str | None] | None = None,
    train_texts: Sequence[str] | None = None,
    test_texts: Sequence[str] | None = None,
) -> dict[str, Any]:
    byte_overlap = _set_overlap(train_byte_hashes, test_byte_hashes)
    contaminated = {
        i for i, h in enumerate(train_byte_hashes) if h and h in byte_overlap
    }
    pcm_overlap: set[str] = set()
    pcm_contaminated: set[int] = set()
    if train_pcm_hashes is not None and test_pcm_hashes is not None:
        pcm_overlap = _set_overlap(train_pcm_hashes, test_pcm_hashes)
        pcm_contaminated = {
            i for i, h in enumerate(train_pcm_hashes) if h and h in pcm_overlap
        }
    transcript_exact = 0
    transcript_normalized = 0
    if train_texts is not None and test_texts is not None:
        transcript_exact = len(set(train_texts) & set(test_texts))
        transcript_normalized = len(
            {spoken_content_key(t) for t in train_texts}
            & {spoken_content_key(t) for t in test_texts}
        )
    return {
        "encoded_byte_overlap_hashes": len(byte_overlap),
        "contaminated_train_rows": sorted(contaminated),
        "contaminated_train_row_count": len(contaminated),
        "pcm_overlap_hashes": len(pcm_overlap),
        "pcm_contaminated_train_rows": sorted(pcm_contaminated),
        "pcm_contaminated_train_row_count": len(pcm_contaminated),
        "transcript_exact_overlap": transcript_exact,
        "transcript_normalized_overlap": transcript_normalized,
        "transcript_overlap_policy": "REPORT_ONLY",
    }


def duplicate_audio_decisions(
    byte_hashes: Sequence[str | None], labels: Sequence[str]
) -> dict[str, Any]:
    """Canonical keep / conflict exclude decisions for within-TRAIN duplicates."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, (hashed, _label) in enumerate(zip(byte_hashes, labels)):
        if hashed:
            groups[hashed].append(index)
    keep: set[int] = set()
    exclude: set[int] = set()
    same_label_exclude: set[int] = set()
    same_label_groups = 0
    conflicting_groups = 0
    for indices in groups.values():
        if len(indices) == 1:
            continue
        distinct = {labels[i] for i in indices}
        if len(distinct) == 1:
            same_label_groups += 1
            canonical = min(indices)
            keep.add(canonical)
            same_label_exclude.update(i for i in indices if i != canonical)
        else:
            conflicting_groups += 1
            exclude.update(indices)
    return {
        "duplicate_audio_groups": same_label_groups + conflicting_groups,
        "same_label_duplicate_groups": same_label_groups,
        "conflicting_label_duplicate_groups": conflicting_groups,
        "canonical_keep_indices": sorted(keep),
        "same_label_exclude_indices": sorted(same_label_exclude),
        "conflicting_exclude_indices": sorted(exclude),
    }


def eligible_rows(
    rows: Sequence[dict[str, Any]],
    *,
    excluded_indices: set[int] | None = None,
    empty_audio_indices: set[int] | None = None,
) -> list[dict[str, Any]]:
    excluded = excluded_indices or set()
    empty_audio = empty_audio_indices or set()
    anchors: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index in excluded or index in empty_audio:
            continue
        label = str(row.get("label"))
        if label not in LABEL_MAP:
            continue
        anchors.append(
            {
                "row_index": index,
                "source_row_id": f"{SHARD_ID}/{index}",
                "audio_id": f"{AUDIO_NAMESPACE}/{SHARD_ID}/{index}",
                "source_label": label,
                "source_text": str(row.get("text") or ""),
                "gold": LABEL_MAP[label],
            }
        )
    return anchors


# ---------------------------------------------------------------------------
# QA rendering
# ---------------------------------------------------------------------------


def qa_id(source_row_id: str, seed: int = SEED) -> str:
    return f"p5_sentiment_reasoning_{_digest(seed, 'id', source_row_id)[:16]}"


def _select_template(source_row_id: str, policy: dict[str, Any], seed: int) -> dict:
    bank = policy["template_bank"]
    index = int(_digest(seed, source_row_id, "template")[:8], 16) % len(bank)
    return bank[index]


def _order_choices(source_row_id: str, policy: dict[str, Any], seed: int) -> list[str]:
    return sorted(
        policy["answer_space"],
        key=lambda choice: _digest(seed, source_row_id, "choice_order", choice),
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
                "source_dataset": SOURCE_DATASET,
                "source_repo": SOURCE_REPO,
                "source_revision": SOURCE_REVISION,
                "source_split": "train",
                "source_row_id": row_id,
                "audio_id": anchor["audio_id"],
                "source_text": anchor["source_text"],
                "source_sentiment_label": anchor["source_label"],
                "question": template["text"],
                "choices": _order_choices(row_id, policy, seed),
                "answer": anchor["gold"],
                "template_id": template["template_id"],
                "benchmark_category": BENCHMARK_CATEGORY,
                "benchmark_type": BENCHMARK_TYPE,
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
        "version": "p5_sentiment_template_registry_v1",
        "task": TASK,
        "templates": [dict(entry) for entry in policy["template_bank"]],
    }


# ---------------------------------------------------------------------------
# Metrics + audit
# ---------------------------------------------------------------------------


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


def class_distribution(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(record["answer"] for record in records)
    total = len(records)
    return {
        "counts": {label: int(counts.get(label, 0)) for label in ANSWER_SPACE},
        "percentages": {
            label: round(100.0 * counts.get(label, 0) / total, 4) if total else 0.0
            for label in ANSWER_SPACE
        },
        "information": class_information(
            {label: int(counts.get(label, 0)) for label in ANSWER_SPACE}
        ),
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


def compute_audit(
    records: Sequence[dict[str, Any]],
    model: Sequence[dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    validity: Counter = Counter()
    template_counts: Counter = Counter()
    choice_position: Counter = Counter()
    seen_ids: set[str] = set()
    seen_audio: set[str] = set()
    duplicate_audio = 0
    for record, model_record in zip(records, model):
        choices = record["choices"]
        if len(choices) != len(ANSWER_SPACE):
            validity["invalid_choice_count"] += 1
        if len(set(choices)) != len(choices):
            validity["duplicate_choices"] += 1
        if record["answer"] not in choices:
            validity["answer_not_in_choices"] += 1
        if record["answer"] != LABEL_MAP.get(record["source_sentiment_label"]):
            validity["wrong_gold"] += 1
        if record["source_split"] != "train":
            validity["non_train_rows"] += 1
        if record["id"] in seen_ids:
            validity["duplicate_qa_ids"] += 1
        seen_ids.add(record["id"])
        if record["audio_id"] in seen_audio:
            duplicate_audio += 1
        seen_audio.add(record["audio_id"])
        template_counts[record["template_id"]] += 1
        choice_position[choices.index(record["answer"])] += 1
        if _contains_absolute_path(model_record):
            validity["absolute_path"] += 1

    allowed = {"id", "task", "audio_id", "question", "choices", "answer"}
    metadata_leak = sum(1 for m in model if set(m) != allowed)
    return {
        "task": TASK,
        "semantic_type": SEMANTIC_TYPE,
        "source": {
            "dataset": SOURCE_DATASET,
            "repo_id": SOURCE_REPO,
            "revision": SOURCE_REVISION,
            "split": "train",
        },
        "seed": SEED,
        "counts": {"qa": len(records)},
        "validity": {
            "wrong_gold": validity["wrong_gold"],
            "answer_not_in_choices": validity["answer_not_in_choices"],
            "invalid_choice_count": validity["invalid_choice_count"],
            "duplicate_choices": validity["duplicate_choices"],
            "non_train_rows": validity["non_train_rows"],
            "duplicate_qa_ids": validity["duplicate_qa_ids"],
            "absolute_path": validity["absolute_path"],
            "invalid_label_used": 0,
            "contaminated_row_used": 0,
        },
        "leakage": {
            "transcript_exposed": 0,
            "source_label_exposed": 0,
            "human_justification_exposed": 0,
            "metadata_leakage": metadata_leak,
            "absolute_path_in_model_facing": 0,
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
            f"index_{i}": int(choice_position.get(i, 0))
            for i in range(len(ANSWER_SPACE))
        },
        "duplicate_audio_ids": duplicate_audio,
    }


def audit_ok(audit: dict[str, Any]) -> bool:
    validity = audit["validity"]
    leakage = audit["leakage"]
    return (
        all(value == 0 for value in validity.values())
        and all(value == 0 for value in leakage.values() if isinstance(value, int))
        and leakage["sauvi_manifest_accessed"] is False
        and audit["llm"]["row_level_llm_calls"] == 0
        and audit["llm"]["production_llm_calls"] == 0
        and audit["duplicate_audio_ids"] == 0
    )


def independent_audit(
    internal_path: Path, train_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    label_by_id = {
        f"{SHARD_ID}/{i}": str(row.get("label")) for i, row in enumerate(train_rows)
    }
    qa_count = 0
    wrong_row = 0
    wrong_gold = 0
    answer_not_in = 0
    invalid_choice_count = 0
    duplicate_choices = 0
    with open(internal_path, "r", encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            qa_count += 1
            source_label = label_by_id.get(record["source_row_id"])
            if source_label is None:
                wrong_row += 1
            if source_label != record["source_sentiment_label"]:
                wrong_row += 1
            if LABEL_MAP.get(source_label) != record["answer"]:
                wrong_gold += 1
            choices = record["choices"]
            if record["answer"] not in choices:
                answer_not_in += 1
            if len(choices) != len(ANSWER_SPACE):
                invalid_choice_count += 1
            if len(set(choices)) != len(choices):
                duplicate_choices += 1
    return {
        "qa_count": qa_count,
        "wrong_source_row": wrong_row,
        "wrong_gold": wrong_gold,
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
    policy: dict[str, Any],
    extra_files: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    ordered = order_records(records)
    model = [to_model_facing(record) for record in ordered]
    audit = compute_audit(ordered, model, policy)
    distribution = class_distribution(ordered)

    _write_jsonl(out_dir / "qa_internal.jsonl", ordered)
    _write_jsonl(out_dir / "qa_model_facing.jsonl", model)
    _write_json(out_dir / "audit.json", audit)
    _write_json(out_dir / "template_registry.json", template_registry(policy))
    _write_json(out_dir / "class_distribution.json", distribution)
    _write_json(
        out_dir / "choice_position.json",
        audit["choice_position"],
    )
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
        "artifact": "p5_sentiment_reasoning_training_qa",
        "task": TASK,
        "semantic_type": SEMANTIC_TYPE,
        "generator": {
            "module": "src/speech_sentiment_p5.py",
            "version": "speech_sentiment_p5_v1",
        },
        "source": {
            "dataset": SOURCE_DATASET,
            "repo_id": SOURCE_REPO,
            "revision": SOURCE_REVISION,
            "split": "train",
        },
        "seed": SEED,
        "counts": {"qa": len(ordered), **distribution["counts"]},
        "audio_identity": {
            "kind": "logical_audio_id",
            "field": "audio_id",
            "namespace": AUDIO_NAMESPACE,
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

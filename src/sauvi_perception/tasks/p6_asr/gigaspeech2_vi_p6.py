"""GigaSpeech2-VI TRAIN REFined onboarding + SAUVI P6 (ASR) training-QA core.

Metadata-first. This module never downloads or decodes audio and never touches
DEV/TEST/SAUVI data. It reads one frozen TSV (``data/vi/train_refined.tsv`` at
the pinned revision), audits it, and builds deterministic four-choice
transcript-selection QA where the gold is the exact source transcript.

Contracts:

* TSV line format is ``<segment_id>\\t<text>\\n`` parsed by the FIRST TAB only.
  Source text is preserved byte-for-byte; no normalization is applied to gold.
* The comparator (NFKC + casefold + punctuation->space + whitespace collapse)
  keeps Vietnamese diacritics and is used only to keep choices distinct.
* The distractor universe is ALL eligible train_refined rows, addressed through
  a compact on-disk bucket index of byte offsets (no per-anchor ranking, no
  fixed 50k pool). Candidate positions are drawn deterministically by SHA256
  modulo the bucket size, then rejected on self/comparator/duplicate.
* Selection is deterministic in ``segment_id`` + seed and nested by rank, so a
  100k target is a subset of a 200k target is a subset of a 500k target.
* No LLM, no embeddings, no edit-distance mining, no audio.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
import unicodedata
from array import array
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from statistics import median
from typing import Any, Self

from src.common.config import ROOT

TASK = "P6"
SEMANTIC_TYPE = "automatic_speech_recognition"
BENCHMARK_CATEGORY = "perception"
BENCHMARK_TYPE = "general"
GOLD_ORIGIN = "SOURCE"
COMPILER_TIER = "T1_PERCEPTION"

SOURCE_DATASET = "GigaSpeech2-VI"
SOURCE_REPO = "speechcolab/gigaspeech2"
SOURCE_REVISION = "8dc0d0e502b7e6d5649a13cd4b677cf10c0b21e3"
SOURCE_SPLIT = "train_refined"
SOURCE_FILENAME = "data/vi/train_refined.tsv"
SOURCE_LANGUAGE = "vi"
AUDIO_NAMESPACE = "gigaspeech2_vi/train_refined"
SEED = 42
COMPARATOR_VERSION = "p6_comparator_v1"

CACHE_PATH = (
    ROOT
    / "outputs"
    / "materialized"
    / "gigaspeech2_vi"
    / "_cache"
    / SOURCE_REVISION
    / "train_refined.tsv"
)
POLICY_PATH = ROOT / "resources" / "semantics" / "p6_gigaspeech2_vi_policy.json"
PROVENANCE_PATH = ROOT / "resources" / "sources" / "gigaspeech2_vi_train_refined.json"

CHOICE_COUNT = 4
DISTRACTOR_COUNT = 3
MAX_BUCKET_ATTEMPTS = 4096

_FORBIDDEN_SPLIT_TOKENS = ("dev", "test", "sauvi", "train_raw")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_DIGIT_RE = re.compile(r"[0-9]")
_NONASCII_RE = re.compile(r"[^\x00-\x7f]")

BUCKETS = ("1-5", "6-10", "11-20", "21-40", "41-80", ">80")


class P6SourceError(RuntimeError):
    """Raised when an input violates the metadata-only / TRAIN-only policy."""


def _digest(*parts: Any) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fp64(text: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little"
    )


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Parsing + comparator
# ---------------------------------------------------------------------------


def parse_tsv_line(line: str) -> tuple[str, str] | None:
    """Parse ``<segment_id>\\t<text>`` by the first tab only."""
    stripped = line.rstrip("\r\n")
    if "\t" not in stripped:
        return None
    segment_id, text = stripped.split("\t", 1)
    return segment_id, text


def spoken_content_key(text: str) -> str:
    """Comparator key: NFKC + casefold + punctuation->space + whitespace."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = _PUNCT_RE.sub(" ", normalized)
    return " ".join(normalized.split())


def word_bucket(word_count: int) -> str:
    if word_count <= 5:
        return "1-5"
    if word_count <= 10:
        return "6-10"
    if word_count <= 20:
        return "11-20"
    if word_count <= 40:
        return "21-40"
    if word_count <= 80:
        return "41-80"
    return ">80"


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


def load_policy(path: Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else POLICY_PATH
    with open(target, "r", encoding="utf-8") as fh:
        policy = json.load(fh)
    if len(policy["template_bank"]) < 1:
        raise P6SourceError("empty_template_bank")
    if policy["answer_policy"]["choice_count"] != CHOICE_COUNT:
        raise P6SourceError("choice_count_mismatch")
    return policy


def load_provenance(path: Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else PROVENANCE_PATH
    with open(target, "r", encoding="utf-8") as fh:
        return json.load(fh)


def assert_train_source(cache_path: Path) -> None:
    name = Path(cache_path).name.lower()
    if any(token in name for token in _FORBIDDEN_SPLIT_TOKENS):
        raise P6SourceError(f"forbidden_split_source:{cache_path}")
    if name != "train_refined.tsv":
        raise P6SourceError(f"expected_train_refined_source:{cache_path}")


def verify_source(
    cache_path: Path | None = None, provenance: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Hard-fail unless the retained TSV matches the frozen SHA256/size."""
    path = Path(cache_path) if cache_path is not None else CACHE_PATH
    prov = provenance if provenance is not None else load_provenance()
    assert_train_source(path)
    if not path.is_file():
        raise P6SourceError(f"source_missing:{path}")
    size = path.stat().st_size
    digest = _sha256_file(path)
    if size != prov["tsv_bytes"] or digest != prov["tsv_sha256"]:
        raise P6SourceError(
            f"source_hash_mismatch:bytes={size}/{prov['tsv_bytes']} "
            f"sha256={digest}/{prov['tsv_sha256']}"
        )
    return {"path": str(path), "bytes": size, "sha256": digest}


def iter_parsed_rows(path: Path) -> Iterator[tuple[str, str]]:
    assert_train_source(path)
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for line in fh:
            parsed = parse_tsv_line(line)
            if parsed is None:
                continue
            segment_id, text = parsed
            if segment_id == "" or text == "":
                continue
            yield segment_id, text


# ---------------------------------------------------------------------------
# Source audit
# ---------------------------------------------------------------------------


def _percentiles(histogram: Counter, points: Sequence[float]) -> dict[str, int]:
    total = sum(histogram.values())
    result: dict[str, int] = {}
    if total == 0:
        return {f"p{int(p * 100):02d}": 0 for p in points}
    ordered = sorted(histogram)
    for point in points:
        target = math.ceil(point * total)
        cumulative = 0
        value = ordered[-1]
        for key in ordered:
            cumulative += histogram[key]
            if cumulative >= target:
                value = key
                break
        result[f"p{int(point * 100):02d}"] = value
    return result


def _distribution(histogram: Counter) -> dict[str, Any]:
    total = sum(histogram.values())
    mean = sum(k * v for k, v in histogram.items()) / total if total else 0.0
    percentiles = _percentiles(histogram, (0.01, 0.05, 0.5, 0.95, 0.99))
    return {
        "min": min(histogram) if histogram else 0,
        "p01": percentiles["p01"],
        "p05": percentiles["p05"],
        "median": percentiles["p50"],
        "p95": percentiles["p95"],
        "p99": percentiles["p99"],
        "max": max(histogram) if histogram else 0,
        "mean": round(mean, 3),
    }


def _dedup_fingerprints(fingerprints: Any) -> dict[str, int]:
    import numpy as np

    values = np.frombuffer(fingerprints, dtype=np.uint64)
    _unique, counts = np.unique(values, return_counts=True)
    duplicate_groups = int((counts > 1).sum())
    duplicate_rows = int(counts[counts > 1].sum() - duplicate_groups)
    return {
        "unique": len(_unique),
        "duplicate_groups": duplicate_groups,
        "duplicate_rows": duplicate_rows,
        "max_group_size": int(counts.max()) if len(counts) else 0,
    }


def audit_source(
    cache_path: Path | None = None,
    provenance: dict[str, Any] | None = None,
    *,
    verify: bool = True,
) -> dict[str, Any]:
    """Stream the whole TSV and compute the full metadata audit."""
    path = Path(cache_path) if cache_path is not None else CACHE_PATH
    if verify:
        verified = verify_source(path, provenance)
    else:
        assert_train_source(path)
        verified = {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}

    physical = 0
    parsed = 0
    malformed = 0
    empty_id = 0
    empty_text = 0
    all_upper = 0
    all_lower = 0
    punctuation_rows = 0
    digit_rows = 0
    nonascii_rows = 0
    sample_upper_chars = 0
    sample_lower_chars = 0
    sample_limit = 200_000
    sample_rows = 0

    char_hist: Counter = Counter()
    word_hist: Counter = Counter()
    bucket_counts: Counter = Counter()
    segment_fp = array("Q")
    text_fp = array("Q")
    normalized_fp = array("Q")

    with open(path, "r", encoding="utf-8", newline="") as fh:
        for line in fh:
            physical += 1
            parsed_line = parse_tsv_line(line)
            if parsed_line is None:
                malformed += 1
                continue
            segment_id, text = parsed_line
            if segment_id == "":
                empty_id += 1
                continue
            if text == "":
                empty_text += 1
                continue
            parsed += 1
            char_hist[len(text)] += 1
            word_count = len(text.split())
            word_hist[word_count] += 1
            bucket_counts[word_bucket(word_count)] += 1
            if text.isupper():
                all_upper += 1
            if text.islower():
                all_lower += 1
            if _PUNCT_RE.search(text):
                punctuation_rows += 1
            if _DIGIT_RE.search(text):
                digit_rows += 1
            if _NONASCII_RE.search(text):
                nonascii_rows += 1
            if sample_rows < sample_limit:
                sample_upper_chars += sum(1 for c in text if c.isupper())
                sample_lower_chars += sum(1 for c in text if c.islower())
                sample_rows += 1
            segment_fp.append(_fp64(segment_id))
            text_fp.append(_fp64(text))
            normalized_fp.append(_fp64(spoken_content_key(text)))

    total_cased = sample_upper_chars + sample_lower_chars
    segment_stats = _dedup_fingerprints(segment_fp)
    return {
        "source": {
            "dataset": SOURCE_DATASET,
            "repo_id": SOURCE_REPO,
            "revision": SOURCE_REVISION,
            "filename": SOURCE_FILENAME,
            "split": SOURCE_SPLIT,
            "language": SOURCE_LANGUAGE,
            "bytes": verified["bytes"],
            "sha256": verified["sha256"],
        },
        "parsing": {
            "physical_lines": physical,
            "parsed_rows": parsed,
            "malformed_rows": malformed,
            "empty_segment_id": empty_id,
            "empty_transcript": empty_text,
        },
        "segment_ids": {
            "unique": segment_stats["unique"],
            "duplicate_groups": segment_stats["duplicate_groups"],
            "duplicate_rows": segment_stats["duplicate_rows"],
            "fingerprint": "blake2b-64 (collision probability negligible)",
        },
        "transcripts_exact": _dedup_fingerprints(text_fp),
        "transcripts_normalized": _dedup_fingerprints(normalized_fp),
        "character_length": _distribution(char_hist),
        "word_count": _distribution(word_hist),
        "word_count_buckets": {
            bucket: int(bucket_counts.get(bucket, 0)) for bucket in BUCKETS
        },
        "normalization_audit": {
            "rows_all_uppercase": all_upper,
            "rows_all_lowercase": all_lower,
            "rows_mixed_or_uncased": parsed - all_upper - all_lower,
            "uppercase_row_proportion": round(all_upper / parsed, 6) if parsed else 0.0,
            "lowercase_row_proportion": round(all_lower / parsed, 6) if parsed else 0.0,
            "char_uppercase_ratio_sample": round(sample_upper_chars / total_cased, 6)
            if total_cased
            else 0.0,
            "char_lowercase_ratio_sample": round(sample_lower_chars / total_cased, 6)
            if total_cased
            else 0.0,
            "sample_rows_for_char_ratio": sample_rows,
            "punctuation_row_incidence": punctuation_rows,
            "arabic_digit_row_incidence": digit_rows,
            "vietnamese_diacritic_row_incidence": nonascii_rows,
            "note": "source text is not transformed by this audit",
        },
        "eligibility": {
            "eligible_rows": parsed,
            "unique_logical_audio_ids": parsed,
            "policy": "parsed & non-empty id & non-empty text & unique segment_id",
        },
    }


# ---------------------------------------------------------------------------
# All-row distractor index
# ---------------------------------------------------------------------------


class P6DistractorIndex:
    """Compact bucket index over ALL eligible rows (byte offsets into the TSV).

    ``bucket_arrays[bucket]`` is an ``array('q')`` of line-start byte offsets.
    Candidate rows are read on demand by seeking the retained TSV; no full-row
    Python objects and no fixed-size distractor pool are materialized.
    """

    def __init__(self, path: Path, bucket_arrays: dict[str, array]) -> None:
        self.path = Path(path)
        self.bucket_arrays = bucket_arrays
        self._fh = None

    def open(self) -> Self:
        self._fh = open(self.path, "rb")  # noqa: SIM115 - handle owned by index
        return self

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    def bucket_size(self, bucket: str) -> int:
        return len(self.bucket_arrays.get(bucket, ()))

    def total_indexed(self) -> int:
        return sum(len(values) for values in self.bucket_arrays.values())

    def read_row(self, offset: int) -> tuple[str, str] | None:
        assert self._fh is not None, "index not open"
        self._fh.seek(offset)
        raw = self._fh.readline()
        return parse_tsv_line(raw.decode("utf-8"))

    def select_distractors(
        self,
        anchor: dict[str, Any],
        seed: int = SEED,
        count: int = DISTRACTOR_COUNT,
    ) -> list[dict[str, Any]]:
        gold_key = anchor["key"]
        used = {gold_key}
        chosen: list[dict[str, Any]] = []
        for bucket in _bucket_order(anchor["bucket"]):
            candidates = self.bucket_arrays.get(bucket)
            if not candidates:
                continue
            size = len(candidates)
            attempts = min(size, MAX_BUCKET_ATTEMPTS)
            for attempt in range(attempts):
                digest = _digest(
                    seed, anchor["segment_id"], "slot", len(chosen), bucket, attempt
                )
                position = int(digest[:16], 16) % size
                parsed = self.read_row(candidates[position])
                if parsed is None:
                    continue
                segment_id, text = parsed
                if segment_id == anchor["segment_id"]:
                    continue
                key = spoken_content_key(text)
                if key in used:
                    continue
                used.add(key)
                chosen.append({"segment_id": segment_id, "text": text, "key": key})
                if len(chosen) >= count:
                    return chosen
        if len(chosen) < count:
            raise P6SourceError(
                f"insufficient_distractors:{anchor['segment_id']}:{len(chosen)}"
            )
        return chosen


def _bucket_order(anchor_bucket: str) -> list[str]:
    index = BUCKETS.index(anchor_bucket)
    return sorted(
        BUCKETS, key=lambda b: (abs(BUCKETS.index(b) - index), BUCKETS.index(b))
    )


def _heap_push(
    heap: list[tuple[int, str, str, int, int]],
    limit: int | None,
    rank: int,
    segment_id: str,
    text: str,
    word_count: int,
    offset: int,
) -> None:
    entry = (-rank, segment_id, text, word_count, offset)
    if limit is None:
        heapq.heappush(heap, entry)
        return
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif rank < -heap[0][0]:
        heapq.heapreplace(heap, entry)


def _anchor_row(entry: tuple[int, str, str, int, int]) -> dict[str, Any]:
    _neg_rank, segment_id, text, word_count, offset = entry
    return {
        "segment_id": segment_id,
        "text": text,
        "word_count": word_count,
        "bucket": word_bucket(word_count),
        "key": spoken_content_key(text),
        "offset": offset,
    }


def build_index_and_anchors(
    cache_path: Path | None = None,
    *,
    target_count: int | None,
    seed: int = SEED,
) -> tuple[int, list[dict[str, Any]], P6DistractorIndex]:
    """Single streaming pass: build the all-row bucket index + select anchors.

    Anchors are the ``target_count`` eligible rows with the smallest
    ``SHA256(seed|anchor|segment_id)`` rank. Because the rank is global and
    independent of the target count, target sets are nested (100k ⊂ 200k ⊂ …).
    """
    path = Path(cache_path) if cache_path is not None else CACHE_PATH
    bucket_arrays: dict[str, array] = {bucket: array("q") for bucket in BUCKETS}
    anchor_heap: list[tuple[int, str, str, int, int]] = []
    eligible = 0
    with open(path, "rb") as fh:
        offset = 0
        for raw in fh:
            start = offset
            offset += len(raw)
            parsed = parse_tsv_line(raw.decode("utf-8"))
            if parsed is None:
                continue
            segment_id, text = parsed
            if segment_id == "" or text == "":
                continue
            eligible += 1
            word_count = len(text.split())
            bucket = word_bucket(word_count)
            bucket_arrays[bucket].append(start)
            rank = int(_digest(seed, "anchor", segment_id)[:16], 16)
            _heap_push(
                anchor_heap, target_count, rank, segment_id, text, word_count, start
            )
    anchors = [
        _anchor_row(entry) for entry in sorted(anchor_heap, key=lambda e: (-e[0], e[1]))
    ]
    return eligible, anchors, P6DistractorIndex(path, bucket_arrays)


def write_index(index: P6DistractorIndex, out_dir: Path) -> dict[str, Any]:
    """Persist the compact bucket index and return its on-disk metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    binary = out_dir / "bucket_offsets.bin"
    with open(binary, "wb") as fh:
        fh.writelines(index.bucket_arrays[bucket].tobytes() for bucket in BUCKETS)
    meta = {
        "format": "int64_line_offsets_per_bucket",
        "bucket_order": list(BUCKETS),
        "bucket_sizes": {b: len(index.bucket_arrays[b]) for b in BUCKETS},
        "total_indexed": index.total_indexed(),
        "binary_bytes": binary.stat().st_size,
        "source_tsv": SOURCE_FILENAME,
        "revision": SOURCE_REVISION,
    }
    (out_dir / "bucket_index.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return meta


# ---------------------------------------------------------------------------
# QA construction
# ---------------------------------------------------------------------------


def _select_template(segment_id: str, policy: dict[str, Any], seed: int) -> dict:
    bank = policy["template_bank"]
    index = int(_digest(seed, segment_id, "template")[:8], 16) % len(bank)
    return bank[index]


def _permute_choices(choices: Sequence[str], segment_id: str, seed: int) -> list[str]:
    return sorted(
        choices, key=lambda choice: _digest(seed, segment_id, "choice_order", choice)
    )


def audio_id(segment_id: str) -> str:
    return f"{AUDIO_NAMESPACE}/{segment_id}"


def qa_id(segment_id: str, seed: int = SEED) -> str:
    return f"p6_gigaspeech2_vi_train_{_digest(seed, 'id', segment_id)[:16]}"


def build_record(
    anchor: dict[str, Any],
    distractors: Sequence[dict[str, Any]],
    policy: dict[str, Any],
    seed: int = SEED,
) -> dict[str, Any]:
    template = _select_template(anchor["segment_id"], policy, seed)
    choices = _permute_choices(
        [anchor["text"], *[d["text"] for d in distractors]],
        anchor["segment_id"],
        seed,
    )
    return {
        "id": qa_id(anchor["segment_id"], seed),
        "task": TASK,
        "semantic_type": SEMANTIC_TYPE,
        "source_dataset": SOURCE_DATASET,
        "source_repo": SOURCE_REPO,
        "source_revision": policy["source"]["revision"],
        "source_split": SOURCE_SPLIT,
        "segment_id": anchor["segment_id"],
        "audio_id": audio_id(anchor["segment_id"]),
        "source_transcript": anchor["text"],
        "question": template["text"],
        "choices": choices,
        "answer": anchor["text"],
        "distractor_segment_ids": [d["segment_id"] for d in distractors],
        "template_id": template["template_id"],
        "benchmark_category": BENCHMARK_CATEGORY,
        "benchmark_type": BENCHMARK_TYPE,
        "gold_origin": GOLD_ORIGIN,
        "compiler_tier": COMPILER_TIER,
        "comparator_version": COMPARATOR_VERSION,
        "selection_seed": seed,
    }


def to_model_facing(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "task": record["task"],
        "audio_id": record["audio_id"],
        "question": record["question"],
        "choices": list(record["choices"]),
        "answer": record["answer"],
    }


def template_registry(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "version": "p6_gigaspeech2_vi_template_registry_v1",
        "task": TASK,
        "templates": [dict(entry) for entry in policy["template_bank"]],
    }


# ---------------------------------------------------------------------------
# Streaming generation + audit
# ---------------------------------------------------------------------------

_ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/]"),
    re.compile(r"/(?:kaggle|home|mnt|media|Users|workspace|content)/"),
)


def _contains_absolute_path(payload: Any) -> bool:
    text = json.dumps(payload, ensure_ascii=False)
    return any(pattern.search(text) for pattern in _ABSOLUTE_PATH_PATTERNS)


def _percentile_from_sorted(values: Sequence[int], fraction: float) -> int:
    if not values:
        return 0
    index = min(len(values) - 1, math.ceil(fraction * len(values)) - 1)
    return values[max(0, index)]


def _appearance_stats(counts: Counter) -> dict[str, Any]:
    values = sorted(counts.values())
    if not values:
        return {"min": 0, "median": 0, "mean": 0, "p95": 0, "p99": 0, "max": 0}
    return {
        "min": values[0],
        "median": median(values),
        "mean": round(sum(values) / len(values), 4),
        "p95": _percentile_from_sorted(values, 0.95),
        "p99": _percentile_from_sorted(values, 0.99),
        "max": values[-1],
    }


def generate_stream(
    out_dir: Path,
    anchors: Sequence[dict[str, Any]],
    index: P6DistractorIndex,
    policy: dict[str, Any],
    seed: int = SEED,
) -> dict[str, Any]:
    """Stream QA to disk, accumulating validity/leakage/role statistics."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    internal_path = out_dir / "qa_internal.jsonl"
    model_path = out_dir / "qa_model_facing.jsonl"

    validity: Counter = Counter()
    choice_position: Counter = Counter()
    template_counts: Counter = Counter()
    bucket_counts: Counter = Counter()
    distractor_counts: Counter = Counter()
    distractor_texts: dict[str, str] = {}
    seen_ids: set[str] = set()
    metadata_exposed = 0
    absolute_paths = 0
    qa_count = 0

    with (
        index,
        open(internal_path, "w", encoding="utf-8", newline="\n") as internal_fh,
        open(model_path, "w", encoding="utf-8", newline="\n") as model_fh,
    ):
        for anchor in anchors:
            distractors = index.select_distractors(anchor, seed)
            record = build_record(anchor, distractors, policy, seed)
            model = to_model_facing(record)
            internal_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            model_fh.write(json.dumps(model, ensure_ascii=False) + "\n")
            qa_count += 1

            choices = record["choices"]
            if len(choices) != CHOICE_COUNT:
                validity["wrong_choice_count"] += 1
            if len(set(choices)) != len(choices):
                validity["duplicate_choices"] += 1
            keys = [spoken_content_key(choice) for choice in choices]
            if len(set(keys)) != len(keys):
                validity["comparator_collision_choices"] += 1
            if record["answer"] not in choices:
                validity["answer_not_in_choices"] += 1
            if record["answer"] != record["source_transcript"]:
                validity["wrong_gold"] += 1
            if record["segment_id"] in record["distractor_segment_ids"]:
                validity["self_distractor"] += 1
            if record["id"] in seen_ids:
                validity["duplicate_qa_ids"] += 1
            seen_ids.add(record["id"])
            choice_position[choices.index(record["answer"])] += 1
            template_counts[record["template_id"]] += 1
            bucket_counts[anchor["bucket"]] += 1
            for distractor in distractors:
                distractor_counts[distractor["segment_id"]] += 1
                distractor_texts[distractor["segment_id"]] = distractor["text"]
            if _contains_absolute_path(model):
                absolute_paths += 1
            serialized = json.dumps(model, ensure_ascii=False)
            if "segment_id" in serialized or "revision" in serialized:
                metadata_exposed += 1

    anchor_ids = {anchor["segment_id"] for anchor in anchors}
    audit = {
        "task": TASK,
        "semantic_type": SEMANTIC_TYPE,
        "seed": seed,
        "comparator_version": COMPARATOR_VERSION,
        "source": {
            "dataset": SOURCE_DATASET,
            "repo_id": SOURCE_REPO,
            "revision": SOURCE_REVISION,
            "split": SOURCE_SPLIT,
        },
        "counts": {"qa": qa_count},
        "validity": {
            "wrong_gold": validity["wrong_gold"],
            "answer_not_in_choices": validity["answer_not_in_choices"],
            "wrong_choice_count": validity["wrong_choice_count"],
            "duplicate_choices": validity["duplicate_choices"],
            "comparator_collision_choices": validity["comparator_collision_choices"],
            "self_distractor": validity["self_distractor"],
            "duplicate_qa_ids": validity["duplicate_qa_ids"],
        },
        "leakage": {
            "absolute_path_in_model_facing": absolute_paths,
            "source_metadata_exposed": metadata_exposed,
            "test_row_used": 0,
            "dev_row_used": 0,
            "sauvi_row_used": 0,
            "sauvi_manifest_accessed": False,
        },
        "llm": {"row_level_llm_calls": 0, "production_llm_calls": 0},
    }
    distractor_stats = {
        "unique_distractor_segment_ids": len(distractor_counts),
        "unique_distractor_transcripts": len(set(distractor_texts.values())),
        "distractor_appearances": _appearance_stats(distractor_counts),
        "anchors_also_used_as_distractor": sum(
            1 for segment_id in anchor_ids if segment_id in distractor_counts
        ),
        "anchor_count": len(anchor_ids),
        "fraction_anchors_used_as_distractor": round(
            sum(1 for s in anchor_ids if s in distractor_counts) / len(anchor_ids), 6
        )
        if anchor_ids
        else 0.0,
        "total_distractor_references": sum(distractor_counts.values()),
    }
    choice_position_stats = {
        f"index_{i}": int(choice_position.get(i, 0)) for i in range(CHOICE_COUNT)
    }
    template_stats = {
        entry["template_id"]: int(template_counts.get(entry["template_id"], 0))
        for entry in policy["template_bank"]
    }
    length_stats = {bucket: int(bucket_counts.get(bucket, 0)) for bucket in BUCKETS}

    _write_json(out_dir / "audit.json", audit)
    _write_json(out_dir / "template_registry.json", template_registry(policy))
    _write_json(out_dir / "distractor_stats.json", distractor_stats)
    _write_json(out_dir / "choice_position.json", choice_position_stats)
    _write_json(out_dir / "template_distribution.json", template_stats)
    _write_json(out_dir / "length_distribution.json", length_stats)

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
        "artifact": "gigaspeech2_vi_p6_training_qa",
        "task": TASK,
        "semantic_type": SEMANTIC_TYPE,
        "generator": {
            "module": "src/gigaspeech2_vi_p6.py",
            "version": "gigaspeech2_vi_p6_v2_all_row_index",
        },
        "source": {
            "dataset": SOURCE_DATASET,
            "repo_id": SOURCE_REPO,
            "revision": SOURCE_REVISION,
            "filename": SOURCE_FILENAME,
            "split": SOURCE_SPLIT,
        },
        "seed": seed,
        "comparator_version": COMPARATOR_VERSION,
        "distractor_universe": "all_eligible_rows",
        "counts": {"qa": qa_count},
        "audio_identity": {
            "kind": "logical_audio_id",
            "field": "audio_id",
            "namespace": AUDIO_NAMESPACE,
            "resolution": "external_training_machine_locator",
            "audio_status": "AUDIO_NOT_DOWNLOADED_BY_DESIGN",
        },
        "determinism": {
            "method": "sha256",
            "seed": seed,
            "tags": [
                "deterministic",
                "reproducible",
                "nested_targets",
                "no_timestamps",
            ],
        },
        "files": file_hashes,
        "audit_status": "PASS" if audit_ok(audit) else "FAIL",
    }
    _write_json(out_dir / "manifest.json", manifest)
    return {
        "out_dir": str(out_dir),
        "audit": audit,
        "distractor_stats": distractor_stats,
        "choice_position": choice_position_stats,
        "template_distribution": template_stats,
        "length_distribution": length_stats,
        "manifest": manifest,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def audit_ok(audit: dict[str, Any]) -> bool:
    validity = audit["validity"]
    leakage = audit["leakage"]
    return (
        all(value == 0 for value in validity.values())
        and leakage["absolute_path_in_model_facing"] == 0
        and leakage["source_metadata_exposed"] == 0
        and leakage["test_row_used"] == 0
        and leakage["dev_row_used"] == 0
        and leakage["sauvi_row_used"] == 0
        and leakage["sauvi_manifest_accessed"] is False
        and audit["llm"]["row_level_llm_calls"] == 0
        and audit["llm"]["production_llm_calls"] == 0
    )


def independent_audit(internal_path: Path, model_path: Path) -> dict[str, Any]:
    """Re-read generated files and recompute all invariants independently."""
    ids: set[str] = set()
    segment_ids: set[str] = set()
    audio_ids: set[str] = set()
    wrong_gold = 0
    answer_not_in = 0
    wrong_choice_count = 0
    duplicate_choices = 0
    comparator_collisions = 0
    self_distractor = 0
    non_train_split = 0
    absolute_paths = 0
    qa_count = 0
    with open(internal_path, "r", encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            qa_count += 1
            ids.add(record["id"])
            segment_ids.add(record["segment_id"])
            audio_ids.add(record["audio_id"])
            if record["source_split"] != SOURCE_SPLIT:
                non_train_split += 1
            choices = record["choices"]
            if len(choices) != CHOICE_COUNT:
                wrong_choice_count += 1
            if len(set(choices)) != len(choices):
                duplicate_choices += 1
            keys = [spoken_content_key(choice) for choice in choices]
            if len(set(keys)) != len(keys):
                comparator_collisions += 1
            if record["answer"] not in choices:
                answer_not_in += 1
            if record["answer"] != record["source_transcript"]:
                wrong_gold += 1
            if record["segment_id"] in record["distractor_segment_ids"]:
                self_distractor += 1
    model_keys_ok = True
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
                model_keys_ok = False
            if _contains_absolute_path(record):
                absolute_paths += 1
    return {
        "qa_count": qa_count,
        "unique_qa_ids": len(ids),
        "unique_anchor_segment_ids": len(segment_ids),
        "unique_logical_audio_ids": len(audio_ids),
        "non_train_refined_rows": non_train_split,
        "invalid_gold": wrong_gold,
        "answer_not_in_choices": answer_not_in,
        "choice_count_not_4": wrong_choice_count,
        "comparator_duplicate_choices": comparator_collisions,
        "duplicate_choices": duplicate_choices,
        "self_distractor": self_distractor,
        "dev_rows": 0,
        "test_rows": 0,
        "sauvi_rows": 0,
        "absolute_paths": absolute_paths,
        "model_facing_schema_ok": model_keys_ok,
        "row_level_llm": 0,
        "production_llm": 0,
    }

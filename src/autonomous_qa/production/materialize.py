"""Deterministic ViMD TRAIN/VALID subset selection (pure logic, no I/O).

Everything here is dataset-shape aware only through RAW SOURCE FIELD NAMES
(region, province_name, filename, text, speakerID, gender, ...) and never
renames or re-encodes them. No network, no audio decoding, no LLM.

Selection contract (§E):
- only the `train` and `valid` splits are accepted; anything else (notably
  `test`) is rejected before any row is touched;
- one row per province_name per split;
- ranking key = SHA256 over `split` + `filename` (never Python's built-in
  hash(), never model/LLM output);
- unusable selected audio advances deterministically to the next ranked row
  of the same province, with the rejection recorded.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

ALLOWED_SPLITS = ("train", "valid")
FORBIDDEN_SPLITS = ("test",)

PROVINCE_FIELD = "province_name"
SPLIT_FIELD = "split"
FILENAME_FIELD = "filename"
SPEAKER_FIELD = "speakerID"
REGION_FIELD = "region"
TEXT_FIELD = "text"
GENDER_FIELD = "gender"

RAW_METADATA_FIELDS = (
    REGION_FIELD,
    "province_code",
    PROVINCE_FIELD,
    FILENAME_FIELD,
    TEXT_FIELD,
    SPEAKER_FIELD,
    GENDER_FIELD,
)


class UnsupportedSplitError(ValueError):
    """Raised when a forbidden (e.g. TEST) split/path reaches the materializer."""


def assert_supported_split(split: str) -> str:
    """TEST and any unknown split are refused loudly and early."""
    name = str(split).strip().lower()
    if name in FORBIDDEN_SPLITS or name not in ALLOWED_SPLITS:
        raise UnsupportedSplitError(
            f"unsupported_split:{split}: only {list(ALLOWED_SPLITS)} are "
            "materialized in this phase"
        )
    return name


def selection_key(split: str, filename: str) -> str:
    """Stable SHA256 ranking key over split + filename (hex)."""
    payload = f"{split}::{filename}".encode()
    return hashlib.sha256(payload).hexdigest()


def build_sample_id(split: str, filename: str) -> str:
    """Pipeline bookkeeping id. Never semantic QA content."""
    return f"vimd:{split}:{filename}"


def row_value(row: dict, field: str):
    """Read a raw field from either a flat row or a canonical envelope."""
    if field in row:
        return row.get(field)
    meta = row.get("metadata")
    if isinstance(meta, dict):
        return meta.get(field)
    return None


def group_by_province(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        province = row_value(row, PROVINCE_FIELD)
        if province is None:
            continue
        grouped[str(province)].append(row)
    return dict(grouped)


def ranked_candidates(rows: list[dict], split: str) -> dict[str, list[dict]]:
    """Per-province candidate lists in deterministic SHA256 rank order."""
    split = assert_supported_split(split)
    out: dict[str, list[dict]] = {}
    for province, group in group_by_province(rows).items():
        out[province] = sorted(
            group,
            key=lambda r: (
                selection_key(split, str(row_value(r, FILENAME_FIELD) or "")),
                str(row_value(r, FILENAME_FIELD) or ""),
            ),
        )
    return out


def select_first_usable(
    candidates: dict[str, list[dict]],
    is_usable,
    *,
    blocked_speakers: set[str] | None = None,
    blocked_filenames: set[str] | None = None,
    rejections: list[dict] | None = None,
) -> tuple[dict[str, dict], dict[str, int]]:
    """First usable row per province, advancing deterministically.

    `is_usable(row)` decides audio/decoding acceptance. Speaker and filename
    blocks are checked here so cross-split overlap can be resolved by
    advancing the candidate list only.
    Returns (selection, advances) where `advances` counts how many ranked
    rows were skipped per province (0 = first choice was taken).
    """
    blocked_speakers = blocked_speakers or set()
    blocked_filenames = blocked_filenames or set()
    selection: dict[str, dict] = {}
    advances: dict[str, int] = {}
    for province in sorted(candidates):
        chosen = None
        skipped = 0
        for rank, row in enumerate(candidates[province]):
            filename = str(row_value(row, FILENAME_FIELD) or "")
            speaker = row_value(row, SPEAKER_FIELD)
            if filename in blocked_filenames:
                skipped += 1
                if rejections is not None:
                    rejections.append({
                        "province_name": province,
                        "filename": filename,
                        "rank": rank,
                        "reason": "filename_overlap",
                    })
                continue
            if speaker is not None and str(speaker) in blocked_speakers:
                skipped += 1
                if rejections is not None:
                    rejections.append({
                        "province_name": province,
                        "filename": filename,
                        "rank": rank,
                        "reason": "speaker_overlap",
                    })
                continue
            if not is_usable(row):
                skipped += 1
                if rejections is not None:
                    rejections.append({
                        "province_name": province,
                        "filename": filename,
                        "rank": rank,
                        "reason": "audio_unusable",
                    })
                continue
            chosen = row
            break
        if chosen is None:
            raise RuntimeError(f"no_usable_candidate:{province}")
        selection[province] = chosen
        advances[province] = skipped
    return selection, advances


def materialize_selection(
    train_rows: list[dict],
    valid_rows: list[dict],
    is_usable,
) -> dict:
    """Full §E + §F selection: 63/63, no filename or speaker overlap.

    TRAIN is selected first (independent of VALID). VALID then advances past
    every TRAIN speaker and filename. TEST is never read here.
    """
    train_ranked = ranked_candidates(train_rows, "train")
    valid_ranked = ranked_candidates(valid_rows, "valid")

    rejections: list[dict] = []
    train_sel, train_adv = select_first_usable(
        train_ranked, is_usable, rejections=rejections)

    train_speakers = {str(row_value(r, SPEAKER_FIELD))
                      for r in train_sel.values()
                      if row_value(r, SPEAKER_FIELD) is not None}
    train_files = {str(row_value(r, FILENAME_FIELD))
                   for r in train_sel.values()}

    valid_sel, valid_adv = select_first_usable(
        valid_ranked, is_usable,
        blocked_speakers=train_speakers,
        blocked_filenames=train_files,
        rejections=rejections,
    )

    return {
        "train": [train_sel[p] for p in sorted(train_sel)],
        "valid": [valid_sel[p] for p in sorted(valid_sel)],
        "train_advances": train_adv,
        "valid_advances": valid_adv,
        "rejections": rejections,
        "train_ranked": train_ranked,
        "valid_ranked": valid_ranked,
    }


def overlap_report(train_rows: list[dict], valid_rows: list[dict]) -> dict:
    """§F verification numbers."""
    train_files = [str(row_value(r, FILENAME_FIELD)) for r in train_rows]
    valid_files = [str(row_value(r, FILENAME_FIELD)) for r in valid_rows]
    train_speakers = {str(row_value(r, SPEAKER_FIELD)) for r in train_rows
                      if row_value(r, SPEAKER_FIELD) is not None}
    valid_speakers = {str(row_value(r, SPEAKER_FIELD)) for r in valid_rows
                      if row_value(r, SPEAKER_FIELD) is not None}
    train_provinces = {str(row_value(r, PROVINCE_FIELD)) for r in train_rows}
    valid_provinces = {str(row_value(r, PROVINCE_FIELD)) for r in valid_rows}
    return {
        "train_count": len(train_rows),
        "valid_count": len(valid_rows),
        "train_unique_provinces": len(train_provinces),
        "valid_unique_provinces": len(valid_provinces),
        "train_unique_speakers": len(train_speakers),
        "valid_unique_speakers": len(valid_speakers),
        "filename_overlap": len(set(train_files) & set(valid_files)),
        "speaker_overlap": len(train_speakers & valid_speakers),
        "train_duplicate_filenames": _dupes(train_files),
        "valid_duplicate_filenames": _dupes(valid_files),
        "province_symmetric_difference": sorted(
            train_provinces ^ valid_provinces),
    }


def _dupes(values: list[str]) -> list[str]:
    return sorted(v for v, n in Counter(values).items() if n > 1)


# -- §K discovery sample ---------------------------------------------------
def _text_len(row: dict) -> int:
    text = row_value(row, TEXT_FIELD)
    return len(str(text or ""))


def select_discovery_sample(
    train_rows: list[dict],
    target: int = 24,
    split: str = "train",
) -> list[dict]:
    """Compact, deterministic, TRAIN-only design sample.

    Guarantees (in priority order, ties broken by SHA256 rank):
    - every region present in TRAIN is represented;
    - as many distinct province_name values as possible;
    - both raw gender codes when available in TRAIN;
    - spread across text-length buckets;
    - no repeated speakerID where a fresh speaker is still available.
    """
    split = assert_supported_split(split)
    if split != "train":
        raise UnsupportedSplitError(
            f"discovery_sample_split_must_be_train:{split}")
    pool = list(train_rows)
    if not pool:
        return []
    target = min(int(target), len(pool))

    ordered = sorted(
        pool,
        key=lambda r: (
            selection_key(split, str(row_value(r, FILENAME_FIELD) or "")),
            str(row_value(r, FILENAME_FIELD) or ""),
        ),
    )

    regions = sorted({str(row_value(r, REGION_FIELD) or "") for r in ordered})
    lengths = sorted(_text_len(r) for r in ordered)
    q1 = lengths[len(lengths) // 3]
    q2 = lengths[(2 * len(lengths)) // 3]

    def length_bucket(row: dict) -> int:
        n = _text_len(row)
        return 0 if n <= q1 else (1 if n <= q2 else 2)

    quota: dict[str, int] = {}
    if regions:
        base = target // len(regions)
        rest = target - base * len(regions)
        counts = Counter(str(row_value(r, REGION_FIELD) or "")
                         for r in ordered)
        for i, region in enumerate(sorted(counts, key=lambda x: (-counts[x], x))):
            quota[region] = base + (1 if i < rest else 0)

    chosen: list[dict] = []
    chosen_ids: set[int] = set()
    seen_province: set[str] = set()
    seen_speaker: set[str] = set()
    seen_gender: Counter = Counter()
    seen_bucket: Counter = Counter()
    region_fill: Counter = Counter()

    def score(row: dict, region: str) -> tuple:
        province = str(row_value(row, PROVINCE_FIELD) or "")
        speaker = row_value(row, SPEAKER_FIELD)
        gender = row_value(row, GENDER_FIELD)
        return (
            region_fill[region] < quota.get(region, 0),   # fill region quota
            province not in seen_province,                 # province spread
            0 if speaker is None else int(str(speaker) not in seen_speaker),
            (seen_gender[str(gender)] == 0
             if gender is not None else 0),                # gender spread
            seen_bucket[length_bucket(row)] == 0,          # length spread
        )

    for _round in range(target):
        room_left = any(region_fill[r] < quota.get(r, 0) for r in quota)
        pool = [
            row for row in ordered
            if id(row) not in chosen_ids and (
                not room_left
                or region_fill[str(row_value(row, REGION_FIELD) or "")]
                < quota.get(str(row_value(row, REGION_FIELD) or ""), 0)
            )
        ]
        if not pool:
            pool = [row for row in ordered if id(row) not in chosen_ids]
        if not pool:
            break
        best = None
        best_key = None
        for row in pool:
            region = str(row_value(row, REGION_FIELD) or "")
            rank = selection_key(split,
                                 str(row_value(row, FILENAME_FIELD) or ""))
            # every component: larger is better -> negate for min()-ordering
            full_key = tuple(-int(x) for x in score(row, region)) + (rank,)
            if best_key is None or full_key < best_key:
                best_key = full_key
                best = row
        if best is None:
            break
        chosen.append(best)
        chosen_ids.add(id(best))
        region = str(row_value(best, REGION_FIELD) or "")
        region_fill[region] += 1
        seen_province.add(str(row_value(best, PROVINCE_FIELD) or ""))
        speaker = row_value(best, SPEAKER_FIELD)
        if speaker is not None:
            seen_speaker.add(str(speaker))
        gender = row_value(best, GENDER_FIELD)
        if gender is not None:
            seen_gender[str(gender)] += 1
        seen_bucket[length_bucket(best)] += 1

    chosen.sort(key=lambda r: selection_key(
        split, str(row_value(r, FILENAME_FIELD) or "")))
    return chosen


# -- §G/H canonical envelope ----------------------------------------------
def canonical_row(
    dataset: str,
    split: str,
    row: dict,
    audio_path: str,
    *,
    sample_id: str | None = None,
) -> dict:
    """Canonical envelope with EXACT raw source keys inside `metadata`."""
    split = assert_supported_split(split)
    filename = str(row_value(row, FILENAME_FIELD) or "")
    meta = {}
    for field in RAW_METADATA_FIELDS:
        value = row_value(row, field)
        if value is not None:
            meta[field] = value
    return {
        "dataset": dataset,
        "split": split,
        "sample_id": sample_id or build_sample_id(split, filename),
        "audio_path": audio_path,
        "metadata": meta,
    }


def sample_stats(rows: list[dict]) -> dict:
    """Report numbers for a materialized split (§F / §S)."""
    provinces = Counter(str(row_value(r, PROVINCE_FIELD)) for r in rows)
    regions = Counter(str(row_value(r, REGION_FIELD)) for r in rows)
    genders = Counter(
        str(row_value(r, GENDER_FIELD)) for r in rows
        if row_value(r, GENDER_FIELD) is not None)
    speakers = Counter(
        str(row_value(r, SPEAKER_FIELD)) for r in rows
        if row_value(r, SPEAKER_FIELD) is not None)
    lengths = sorted(_text_len(r) for r in rows)
    return {
        "rows": len(rows),
        "unique_provinces": len(provinces),
        "region_distribution": dict(sorted(regions.items())),
        "raw_gender_code_distribution": dict(sorted(genders.items())),
        "unique_speakers": len(speakers),
        "duplicate_speakers": sorted(s for s, n in speakers.items() if n > 1),
        "text_length": {
            "min": lengths[0] if lengths else 0,
            "max": lengths[-1] if lengths else 0,
            "mean": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        },
    }


__all__ = [
    "ALLOWED_SPLITS",
    "FORBIDDEN_SPLITS",
    "RAW_METADATA_FIELDS",
    "UnsupportedSplitError",
    "assert_supported_split",
    "build_sample_id",
    "canonical_row",
    "group_by_province",
    "materialize_selection",
    "overlap_report",
    "ranked_candidates",
    "row_value",
    "sample_stats",
    "select_discovery_sample",
    "select_first_usable",
    "selection_key",
]

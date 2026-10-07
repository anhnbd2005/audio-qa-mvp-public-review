"""VietLyrics source sanitation + split isolation + SAUVI contamination audit.

Builds a clean, canonical, TRAIN-eligible asset pool for future P7 generation.
Asset identity is the Zing ID extracted from the canonical source link; one
clean asset yields at most one future P7 QA.

Rules:

* VAL has evaluation precedence: any asset_id present in VAL is
  ``SOURCE_VAL_RESERVED`` and every TRAIN row for it is not TRAIN-eligible.
* TRAIN-only assets: singleton -> unique; duplicates with numerically equal
  (Decimal) WPM -> canonicalize to one; duplicates with differing WPM ->
  ``TRAIN_WPM_CONFLICT`` (entire asset excluded; no vote/average).
* Metadata disagreement (title/artist/song/genre) with equal WPM is report-only.
* SAUVI: every VietLyrics-backed benchmark asset (all subtasks) is reserved;
  clean TRAIN assets matching it are excluded.

This module never derives P7 density classes and never reads audio or an LLM.
The SAUVI manifest is used only to build a frozen reserved-ID artifact.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from src.common.config import ROOT
from src.sauvi_perception.datasets.vietlyrics.vietlyrics_source import (
    REVISION,
    extract_zing_id,
)

TASK_ASSET_NAMESPACE = "vietlyrics/train"

SAUVI_MANIFEST_CANDIDATES = (
    ROOT / "outputs" / "source_manifests" / "sauvi" / "sauvi_all.jsonl",
    Path(r"G:\VietnameseSpeechQABenchmark\QAofSAUVI\sauvi_all.jsonl"),
)

SAUVI_ZING_RE = re.compile(r"zing_([A-Za-z0-9]{8})")
METADATA_FIELDS = ("title", "artist", "song", "genre")


class SanitationError(RuntimeError):
    """Raised when the source or SAUVI manifest violates the contract."""


# ---------------------------------------------------------------------------
# WPM (Decimal equality)
# ---------------------------------------------------------------------------


def parse_wpm_decimal(value: Any) -> Decimal | None:
    """Strict finite positive WPM as Decimal (numeric, not textual, equality)."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    if not number.is_finite() or number <= 0:
        return None
    return number


# ---------------------------------------------------------------------------
# SAUVI reserved set
# ---------------------------------------------------------------------------


def resolve_sauvi_manifest(path: Path | None = None) -> Path:
    if path is not None:
        if not Path(path).is_file():
            raise SanitationError(f"sauvi_manifest_missing:{path}")
        return Path(path)
    for candidate in SAUVI_MANIFEST_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise SanitationError("sauvi_manifest_not_found")


def extract_sauvi_asset_id(audio_id: Any) -> str | None:
    if audio_id is None:
        return None
    match = SAUVI_ZING_RE.search(str(audio_id))
    return match.group(1) if match else None


def load_sauvi_vietlyrics_rows(manifest_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("dataset")) == "VietLyrics" or extract_sauvi_asset_id(
                row.get("audio_id")
            ):
                rows.append(row)
    return rows


def build_sauvi_reserved(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_subtask: dict[str, set[str]] = defaultdict(set)
    ids: set[str] = set()
    for row in rows:
        asset_id = extract_sauvi_asset_id(row.get("audio_id"))
        if not asset_id:
            continue
        ids.add(asset_id)
        by_subtask[str(row.get("sub-category"))].add(asset_id)
    return {
        "total_rows": len(rows),
        "unique_asset_ids": len(ids),
        "reserved_asset_ids": sorted(ids),
        "by_subtask": {
            name: {
                "rows": sum(1 for r in rows if str(r.get("sub-category")) == name),
                "unique_asset_ids": len(values),
                "asset_ids": sorted(values),
            }
            for name, values in sorted(by_subtask.items())
        },
    }


# ---------------------------------------------------------------------------
# Asset inventory
# ---------------------------------------------------------------------------


def _member(split: str, row_index: int, row: dict[str, str]) -> dict[str, Any]:
    number = parse_wpm_decimal(row.get("wpm"))
    return {
        "source_split": split,
        "source_row_index": row_index,
        "link": row.get("link", ""),
        "title": row.get("title", ""),
        "artist": row.get("artist", ""),
        "song": row.get("song", ""),
        "genre": row.get("genre", ""),
        "wpm_raw": row.get("wpm", ""),
        "wpm_decimal": str(number) if number is not None else None,
        "wpm_value": float(number) if number is not None else None,
        "duration_mins": row.get("duration_mins", ""),
        "token_count": row.get("token_count", ""),
    }


def build_asset_inventory(
    train_rows: Sequence[dict[str, str]], val_rows: Sequence[dict[str, str]]
) -> tuple[dict[str, dict[str, list[dict]]], list[dict], list[dict]]:
    inventory: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: {"train": [], "val": []}
    )
    unassigned_train: list[dict] = []
    unassigned_val: list[dict] = []
    for index, row in enumerate(train_rows):
        asset_id = extract_zing_id(row.get("link"))
        if asset_id:
            inventory[asset_id]["train"].append(_member("train", index, row))
        else:
            unassigned_train.append(
                {"source_row_index": index, "link": row.get("link")}
            )
    for index, row in enumerate(val_rows):
        asset_id = extract_zing_id(row.get("link"))
        if asset_id:
            inventory[asset_id]["val"].append(_member("val", index, row))
        else:
            unassigned_val.append({"source_row_index": index, "link": row.get("link")})
    return inventory, unassigned_train, unassigned_val


def _metadata_inconsistent(members: Sequence[dict]) -> bool:
    for field in METADATA_FIELDS:
        if len({str(member.get(field, "")) for member in members}) > 1:
            return True
    return False


def classify_asset(
    asset_id: str, entry: dict[str, list[dict]], sauvi_ids: set[str]
) -> dict[str, Any]:
    train_members = entry["train"]
    val_members = entry["val"]
    appears_train = bool(train_members)
    appears_val = bool(val_members)
    appears_sauvi = asset_id in sauvi_ids

    train_decimals = [parse_wpm_decimal(m.get("wpm_raw")) for m in train_members]
    wpm_valid = appears_train and all(value is not None for value in train_decimals)
    wpm_conflict = appears_train and (not wpm_valid or len(set(train_decimals)) > 1)
    metadata_inconsistent = (
        appears_train
        and not appears_val
        and len(train_members) > 1
        and _metadata_inconsistent(train_members)
    )

    if appears_val:
        terminal = "VAL_RESERVED"
    elif appears_sauvi:
        terminal = "SAUVI_RESERVED"
    elif wpm_conflict:
        terminal = "TRAIN_WPM_CONFLICT"
    else:
        terminal = "TRAIN_CLEAN"

    return {
        "asset_id": asset_id,
        "audio_id": f"{TASK_ASSET_NAMESPACE}/{asset_id}",
        "appears_train": appears_train,
        "appears_val": appears_val,
        "appears_sauvi": appears_sauvi,
        "wpm_valid": wpm_valid,
        "wpm_conflict": wpm_conflict,
        "metadata_inconsistency_report_only": metadata_inconsistent,
        "train_member_count": len(train_members),
        "val_member_count": len(val_members),
        "terminal_status": terminal,
        "canonical_row": canonical_row(train_members) if appears_train else None,
    }


def canonical_row(train_members: Sequence[dict]) -> dict[str, Any] | None:
    """Deterministic provenance representative: lowest TRAIN logical row index."""
    if not train_members:
        return None
    ordered = sorted(train_members, key=lambda m: int(m["source_row_index"]))
    return ordered[0]


def classify_all(
    inventory: dict[str, dict[str, list[dict]]], sauvi_ids: set[str]
) -> dict[str, dict[str, Any]]:
    return {
        asset_id: classify_asset(asset_id, entry, sauvi_ids)
        for asset_id, entry in inventory.items()
    }


# ---------------------------------------------------------------------------
# Clean pools + exclusion audit
# ---------------------------------------------------------------------------


def _asset_record(asset_id: str, classification: dict[str, Any]) -> dict[str, Any]:
    member = classification["canonical_row"]
    return {
        "asset_id": asset_id,
        "audio_id": classification["audio_id"],
        "source_split": "train",
        "source_row_index": member["source_row_index"] if member else None,
        "wpm_raw": member["wpm_raw"] if member else None,
        "wpm_decimal": member["wpm_decimal"] if member else None,
        "wpm_value": member["wpm_value"] if member else None,
        "title": member["title"] if member else None,
        "artist": member["artist"] if member else None,
        "song": member["song"] if member else None,
        "genre": member["genre"] if member else None,
        "duration_mins": member["duration_mins"] if member else None,
        "token_count": member["token_count"] if member else None,
        "link": member["link"] if member else None,
        "train_member_count": classification["train_member_count"],
        "metadata_inconsistency_report_only": classification[
            "metadata_inconsistency_report_only"
        ],
    }


def build_clean_assets(
    classifications: dict[str, dict[str, Any]],
) -> tuple[list[dict], list[dict]]:
    source_only = [
        _asset_record(asset_id, cls)
        for asset_id, cls in sorted(classifications.items())
        if cls["appears_train"] and not cls["appears_val"] and not cls["wpm_conflict"]
    ]
    final = [
        record
        for record in source_only
        if not classifications[record["asset_id"]]["appears_sauvi"]
    ]
    return source_only, final


def build_exclusion_audit(
    inventory: dict[str, dict[str, list[dict]]],
    classifications: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Complete row-level accounting of every TRAIN row's disposition."""
    records: list[dict[str, Any]] = []
    for asset_id, entry in sorted(inventory.items()):
        classification = classifications[asset_id]
        terminal = classification["terminal_status"]
        canonical = canonical_row(entry["train"])
        canonical_index = (
            int(canonical["source_row_index"]) if canonical is not None else None
        )
        for member in entry["train"]:
            index = int(member["source_row_index"])
            if terminal == "VAL_RESERVED":
                disposition = "EXCLUDED_VAL_RESERVED"
            elif terminal == "SAUVI_RESERVED":
                disposition = "EXCLUDED_SAUVI_RESERVED"
            elif terminal == "TRAIN_WPM_CONFLICT":
                disposition = "EXCLUDED_WPM_CONFLICT"
            elif index == canonical_index:
                disposition = "KEPT_CANONICAL"
            else:
                disposition = "COLLAPSED_DUPLICATE"
            if disposition == "KEPT_CANONICAL":
                continue
            records.append(
                {
                    "source_split": "train",
                    "source_row_index": index,
                    "asset_id": asset_id,
                    "terminal_status": terminal,
                    "disposition": disposition,
                    "wpm_raw": member["wpm_raw"],
                }
            )
    return records


# ---------------------------------------------------------------------------
# Summary + independent audit
# ---------------------------------------------------------------------------


def _appearance_stats(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"min": 0, "median": 0, "mean": 0, "max": 0}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "mean": round(sum(ordered) / len(ordered), 4),
        "max": ordered[-1],
    }


def sanitation_summary(
    train_rows: Sequence[dict],
    val_rows: Sequence[dict],
    inventory: dict[str, dict[str, list[dict]]],
    classifications: dict[str, dict[str, Any]],
    sauvi: dict[str, Any],
) -> dict[str, Any]:
    train_ids = {aid for aid, e in inventory.items() if e["train"]}
    val_ids = {aid for aid, e in inventory.items() if e["val"]}
    train_only = train_ids - val_ids
    train_only_singleton = sum(
        1 for aid in train_only if len(inventory[aid]["train"]) == 1
    )
    train_only_duplicate = len(train_only) - train_only_singleton
    wpm_consistent = sum(
        1
        for aid in train_only
        if not classifications[aid]["wpm_conflict"] and len(inventory[aid]["train"]) > 1
    )
    wpm_conflicting = sum(
        1 for aid in train_only if classifications[aid]["wpm_conflict"]
    )

    source_only, final = build_clean_assets(classifications)
    sauvi_ids = set(sauvi["reserved_asset_ids"])

    sauvi_mapping = {"train_only": 0, "val_only": 0, "both": 0, "not_found": 0}
    for asset_id in sauvi_ids:
        in_train = asset_id in train_ids
        in_val = asset_id in val_ids
        if in_train and in_val:
            sauvi_mapping["both"] += 1
        elif in_train:
            sauvi_mapping["train_only"] += 1
        elif in_val:
            sauvi_mapping["val_only"] += 1
        else:
            sauvi_mapping["not_found"] += 1

    val_reserved_rows = sum(len(inventory[aid]["train"]) for aid in train_ids & val_ids)
    wpm_conflict_rows = sum(
        len(inventory[aid]["train"])
        for aid in train_only
        if classifications[aid]["wpm_conflict"]
    )
    collapsed_rows = sum(
        max(0, len(inventory[aid]["train"]) - 1)
        for aid in train_only
        if not classifications[aid]["wpm_conflict"]
    )
    sauvi_excluded_rows = sum(
        len(inventory[aid]["train"])
        for aid in train_only
        if classifications[aid]["appears_sauvi"]
    )

    return {
        "revision": REVISION,
        "raw_rows": {"train": len(train_rows), "val": len(val_rows)},
        "asset_level": {
            "unique_train_ids": len(train_ids),
            "unique_val_ids": len(val_ids),
            "train_val_overlap_ids": len(train_ids & val_ids),
            "total_unique_assets": len(inventory),
        },
        "train_only_assets": {
            "total": len(train_only),
            "singleton": train_only_singleton,
            "duplicate": train_only_duplicate,
            "duplicate_wpm_consistent": wpm_consistent,
            "duplicate_wpm_conflicting": wpm_conflicting,
        },
        "rows_affected": {
            "val_reservation": val_reserved_rows,
            "wpm_conflict": wpm_conflict_rows,
            "canonical_duplicate_collapse": collapsed_rows,
        },
        "sauvi": {
            "total_vietlyrics_rows": sauvi["total_rows"],
            "unique_vietlyrics_ids": sauvi["unique_asset_ids"],
            "by_subtask": {
                name: {"rows": info["rows"], "unique_ids": info["unique_asset_ids"]}
                for name, info in sauvi["by_subtask"].items()
            },
            "source_mapping": sauvi_mapping,
        },
        "final": {
            "source_clean_train_assets": len(source_only),
            "sauvi_exclusions": len(source_only) - len(final),
            "final_clean_train_assets": len(final),
            "sauvi_excluded_train_rows": sauvi_excluded_rows,
        },
    }


def independent_audit(
    train_rows: Sequence[dict],
    val_rows: Sequence[dict],
    classifications: dict[str, dict[str, Any]],
    inventory: dict[str, dict[str, list[dict]]],
    sauvi: dict[str, Any],
) -> dict[str, Any]:
    """Re-derive from raw rows and verify accounting + disjointness."""
    val_ids = {aid for aid, e in inventory.items() if e["val"]}
    sauvi_ids = set(sauvi["reserved_asset_ids"])
    _source_only, final = build_clean_assets(classifications)
    final_ids = {record["asset_id"] for record in final}

    train_rows_accounted = sum(len(e["train"]) for e in inventory.values())
    val_rows_accounted = sum(len(e["val"]) for e in inventory.values())

    return {
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_rows_accounted": train_rows_accounted,
        "val_rows_accounted": val_rows_accounted,
        "all_train_rows_accounted": train_rows_accounted == len(train_rows),
        "all_val_rows_accounted": val_rows_accounted == len(val_rows),
        "final_clean_train_ids": len(final_ids),
        "final_clean_intersect_val": sorted(final_ids & val_ids),
        "final_clean_intersect_sauvi": sorted(final_ids & sauvi_ids),
        "no_val_asset_survives": not (final_ids & val_ids),
        "no_sauvi_asset_survives": not (final_ids & sauvi_ids),
        "no_wpm_conflict_survives": all(
            not classifications[aid]["wpm_conflict"] for aid in final_ids
        ),
        "terminal_status_distribution": dict(
            Counter(cls["terminal_status"] for cls in classifications.values())
        ),
    }


# ---------------------------------------------------------------------------
# IO helpers
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
            json.dumps(record, ensure_ascii=False) + "\n" for record in records
        )


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

"""VietLyrics official metadata source onboarding (source acquisition + audit).

Scope: pin the official ``BatmanofZuhandArrgh/VietLyrics`` revision, retain the
two metadata CSVs, and audit them. This module deliberately does NOT define any
P7 lyric-density class or threshold; the original SAUVI WPM -> density rule has
not been recovered.

Contracts:

* Revision is pinned; retrieval uses ``raw.githubusercontent.com`` at the exact
  SHA. SHA256 of the raw downloaded bytes is the production integrity authority;
  the GitHub object hash is retained as ``git_blob_sha1`` provenance only.
* Only the two metadata CSVs are fetched. No audio, no yt-dlp, no conversion,
  no LLM.
* The historical curated subset ``vietlyrics_selected_clean_manifest.csv`` is
  never used as source.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import unicodedata
import urllib.request
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from src.common.config import ROOT

SOURCE_NAME = "VietLyrics"
REPO = "BatmanofZuhandArrgh/VietLyrics"
REVISION = "d52817ad9f63c4e1bdb7f77991e122a919d5dcf6"
TRAIN_FILE = "data/train_7k_metadata_authors.csv"
VAL_FILE = "data/val_1k_metadata_authors.csv"
TRAIN_BYTES = 1747425
VAL_BYTES = 236077
TRAIN_SHA256 = "bdcbe18eb995ae83cbb8406fb781feec26c32e13d8da8066977ed4c7e13bd0fb"
VAL_SHA256 = "77b9027d35df817a404ab23f3698108c48672e0828a5b79ec86543fa574603e0"
TRAIN_GIT_BLOB_SHA1 = "3179a6ebaa7dfa72a4616c14ff056928a46a6607"
VAL_GIT_BLOB_SHA1 = "e274a0ed215e56e4fdb3cb759627d63c1f7ed378"
PARSER_VERSION = "vietlyrics_source_parser_v1"

EXPECTED_HEADERS = [
    "song",
    "link",
    "prefix",
    "title",
    "genre",
    "artist",
    "token_count",
    "wpm",
    "duration_mins",
    "loibaihat_authors",
    "lyricvn_authors",
]

RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{REVISION}/"
CACHE_DIR = ROOT / "outputs" / "materialized" / "vietlyrics" / "_cache" / REVISION
TRAIN_CACHE = CACHE_DIR / "train_7k_metadata_authors.csv"
VAL_CACHE = CACHE_DIR / "val_1k_metadata_authors.csv"
PROVENANCE_PATH = ROOT / "resources" / "sources" / "vietlyrics.json"

ZING_ID_RE = re.compile(r"/([A-Za-z0-9]{8})\.html/?$")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


class VietLyricsSourceError(RuntimeError):
    """Raised when the retained source violates the pinned-source contract."""


# ---------------------------------------------------------------------------
# Provenance + fetch
# ---------------------------------------------------------------------------


def load_provenance(path: Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else PROVENANCE_PATH
    with open(target, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def fetch_source(cache_dir: Path | None = None) -> dict[str, Any]:
    """Download the two metadata CSVs at the pinned revision and verify bytes."""
    target_dir = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    expected = {
        TRAIN_FILE: (TRAIN_CACHE.name, TRAIN_BYTES, TRAIN_SHA256),
        VAL_FILE: (VAL_CACHE.name, VAL_BYTES, VAL_SHA256),
    }
    results = {}
    for source_path, (name, size, sha) in expected.items():
        url = RAW_BASE + source_path
        with urllib.request.urlopen(url, timeout=120) as response:
            if response.status != 200:
                raise VietLyricsSourceError(f"http_status:{response.status}:{url}")
            data = response.read()
        if len(data) != size:
            raise VietLyricsSourceError(f"size_mismatch:{name}:{len(data)}:{size}")
        digest = _sha256_bytes(data)
        if digest != sha:
            raise VietLyricsSourceError(f"sha256_mismatch:{name}:{digest}:{sha}")
        (target_dir / name).write_bytes(data)
        results[name] = {"bytes": len(data), "sha256": digest}
    return results


def verify_source(
    train_path: Path | None = None,
    val_path: Path | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Hard-fail unless both retained CSVs match the frozen bytes/SHA256."""
    prov = provenance if provenance is not None else load_provenance()
    if prov["revision"] != REVISION:
        raise VietLyricsSourceError("revision_mismatch")
    out: dict[str, Any] = {}
    for key, default_path, expected in (
        ("train", TRAIN_CACHE, prov["train"]),
        ("validation", VAL_CACHE, prov["validation"]),
    ):
        path = Path(default_path)
        if train_path is not None and key == "train":
            path = Path(train_path)
        if val_path is not None and key == "validation":
            path = Path(val_path)
        if not path.is_file():
            raise VietLyricsSourceError(f"{key}_missing:{path}")
        size = path.stat().st_size
        digest = _sha256_file(path)
        if size != expected["bytes"] or digest != expected["sha256"]:
            raise VietLyricsSourceError(
                f"{key}_hash_mismatch:{size}/{expected['bytes']}:"
                f"{digest}/{expected['sha256']}"
            )
        out[key] = {"path": str(path), "bytes": size, "sha256": digest}
    return out


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def load_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != EXPECTED_HEADERS:
            raise VietLyricsSourceError(f"unexpected_header:{reader.fieldnames}")
        for row in reader:
            rows.append(dict(row))
    return rows


def physical_line_count(path: Path) -> int:
    """Raw newline count (differs from CSV logical records if fields are quoted)."""
    data = path.read_bytes()
    return data.count(b"\n") + (0 if data.endswith(b"\n") else 1)


def load_train_rows(path: Path | None = None) -> list[dict[str, str]]:
    target = Path(path) if path is not None else TRAIN_CACHE
    if "train_7k" not in target.name:
        raise VietLyricsSourceError(f"expected_train_source:{target.name}")
    return load_rows(target)


def load_val_rows(path: Path | None = None) -> list[dict[str, str]]:
    target = Path(path) if path is not None else VAL_CACHE
    if "val_1k" not in target.name:
        raise VietLyricsSourceError(f"expected_validation_source:{target.name}")
    return load_rows(target)


def parse_wpm(value: Any) -> tuple[str, float | None]:
    """Strict WPM parser: only finite positive floats are valid."""
    if value is None:
        return "missing", None
    text = str(value).strip()
    if text == "":
        return "missing", None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return "malformed", None
    if math.isnan(number) or math.isinf(number):
        return "malformed", None
    if number <= 0:
        return "nonpositive", None
    return "valid", number


def extract_zing_id(link: Any) -> str | None:
    """Strict Zing ID: the final URL path component before ``.html``."""
    if link is None:
        return None
    text = str(link).strip()
    match = ZING_ID_RE.search(text)
    return match.group(1) if match else None


def normalize_title_artist(title: Any, artist: Any) -> str:
    combined = f"{title or ''}|{artist or ''}"
    normalized = unicodedata.normalize("NFKC", combined).casefold()
    normalized = _PUNCT_RE.sub(" ", normalized)
    return " ".join(normalized.split())


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
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
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 6),
        "p01": round(_percentile(ordered, 0.01), 6),
        "p05": round(_percentile(ordered, 0.05), 6),
        "p25": round(_percentile(ordered, 0.25), 6),
        "median": round(_percentile(ordered, 0.50), 6),
        "p75": round(_percentile(ordered, 0.75), 6),
        "p95": round(_percentile(ordered, 0.95), 6),
        "p99": round(_percentile(ordered, 0.99), 6),
        "max": round(ordered[-1], 6),
        "mean": round(mean(ordered), 6),
        "std": round(pstdev(ordered), 6) if len(ordered) > 1 else 0.0,
    }


def audit_wpm(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    status_counts: Counter = Counter()
    valid: list[float] = []
    for row in rows:
        status, number = parse_wpm(row.get("wpm"))
        status_counts[status] += 1
        if status == "valid" and number is not None:
            valid.append(number)
    return {
        "total_rows": len(rows),
        "valid_wpm": status_counts["valid"],
        "missing_wpm": status_counts["missing"],
        "malformed_wpm": status_counts["malformed"],
        "nonpositive_wpm": status_counts["nonpositive"],
        "distribution": _distribution(valid),
    }


def audit_zing_ids(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    parsed = 0
    missing_link = 0
    malformed_link = 0
    ids: list[str] = []
    for row in rows:
        link = row.get("link")
        if link is None or str(link).strip() == "":
            missing_link += 1
            continue
        zing_id = extract_zing_id(link)
        if zing_id is None:
            malformed_link += 1
            continue
        parsed += 1
        ids.append(zing_id)
    counts = Counter(ids)
    duplicates = {key: value for key, value in counts.items() if value > 1}
    duplicate_rows = sum(value for value in duplicates.values())
    return {
        "total_rows": len(rows),
        "parsed_ids": parsed,
        "missing_link": missing_link,
        "malformed_link": malformed_link,
        "unique_ids": len(counts),
        "duplicate_ids": len(duplicates),
        "duplicate_rows": duplicate_rows,
        "duplicate_id_groups": dict(sorted(duplicates.items())),
    }


def cross_split_audit(
    train_rows: Sequence[dict[str, str]], val_rows: Sequence[dict[str, str]]
) -> dict[str, Any]:
    train_ids = {
        zid for zid in (extract_zing_id(row.get("link")) for row in train_rows) if zid
    }
    val_ids = {
        zid for zid in (extract_zing_id(row.get("link")) for row in val_rows) if zid
    }
    train_links = {
        str(row.get("link")).strip() for row in train_rows if row.get("link")
    }
    val_links = {str(row.get("link")).strip() for row in val_rows if row.get("link")}
    train_titles = {
        normalize_title_artist(row.get("title"), row.get("artist"))
        for row in train_rows
    }
    val_titles = {
        normalize_title_artist(row.get("title"), row.get("artist")) for row in val_rows
    }
    id_overlap = sorted(train_ids & val_ids)
    overlap_set = set(id_overlap)
    return {
        "zing_id_overlap": id_overlap,
        "zing_id_overlap_count": len(id_overlap),
        "train_rows_with_overlap_id": sum(
            1 for row in train_rows if extract_zing_id(row.get("link")) in overlap_set
        ),
        "val_rows_with_overlap_id": sum(
            1 for row in val_rows if extract_zing_id(row.get("link")) in overlap_set
        ),
        "exact_link_overlap": len(train_links & val_links),
        "normalized_title_artist_overlap": len(train_titles & val_titles),
        "blocking": len(id_overlap) > 0,
    }


def audit_source() -> dict[str, Any]:
    """Verify the retained source and produce the full metadata audit."""
    verified = verify_source()
    train_rows = load_train_rows()
    val_rows = load_val_rows()
    train_wpm = audit_wpm(train_rows)
    val_wpm = audit_wpm(val_rows)
    return {
        "source_name": SOURCE_NAME,
        "repo": REPO,
        "revision": REVISION,
        "parser_version": PARSER_VERSION,
        "integrity_authority": "sha256",
        "files": {
            "train": {
                "source_path": TRAIN_FILE,
                "bytes": verified["train"]["bytes"],
                "sha256": verified["train"]["sha256"],
                "git_blob_sha1": TRAIN_GIT_BLOB_SHA1,
                "physical_lines": physical_line_count(TRAIN_CACHE),
                "logical_records": len(train_rows),
            },
            "validation": {
                "source_path": VAL_FILE,
                "bytes": verified["validation"]["bytes"],
                "sha256": verified["validation"]["sha256"],
                "git_blob_sha1": VAL_GIT_BLOB_SHA1,
                "physical_lines": physical_line_count(VAL_CACHE),
                "logical_records": len(val_rows),
            },
        },
        "schema": EXPECTED_HEADERS,
        "wpm": {"train": train_wpm, "validation": val_wpm},
        "zing_ids": {
            "train": audit_zing_ids(train_rows),
            "validation": audit_zing_ids(val_rows),
        },
        "cross_split": cross_split_audit(train_rows, val_rows),
    }

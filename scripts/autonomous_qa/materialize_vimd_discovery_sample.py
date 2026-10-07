"""Materialize a small deterministic ViMD TRAIN/VALID subset (bootstrap).

Source of rows + audio: the pinned Hugging Face revision of
`nguyendv02/ViMD_Dataset`. Only the `train` and `valid` splits are ever
touched; `test` is refused before any row is read.

    python scripts/autonomous_qa/materialize_vimd_discovery_sample.py --stage scan
    python scripts/autonomous_qa/materialize_vimd_discovery_sample.py --stage build
    python scripts/autonomous_qa/materialize_vimd_discovery_sample.py --stage sample

Writes a NEW isolated workspace data/materialized/vimd/<run_id>/ and
never touches legacy data/vimd/.

Selection is deterministic: SHA256(split + filename) ranking, one row per
province_name per split, speaker/filename overlap resolved by advancing the
VALID candidate list only. No Python built-in hash(), no model output.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.autonomous_qa.production.materialize import (
    ALLOWED_SPLITS,
    UnsupportedSplitError,
    assert_supported_split,
    build_sample_id,
    canonical_row,
    materialize_selection,
    overlap_report,
    ranked_candidates,
    row_value,
    sample_stats,
    select_discovery_sample,
)

REPO_ID = "nguyendv02/ViMD_Dataset"
REVISION = "3a5b30157034e7eadd5c75fae1a820c6f9383398"
METADATA_COLUMNS = [
    "region", "province_code", "province_name", "filename",
    "text", "speakerID", "gender",
]
CACHE_DIR = ROOT / "data" / "materialized" / "vimd" / "_cache" / REVISION

TARGET_TRAIN = 63
_TARGET_VALID = 63
DISCOVERY_TARGET = 24
CANDIDATE_DEPTH = 3
AUDIO_WORKERS = 3
SCAN_WORKERS = 6
AUDIO_BLOCK_SIZE = 32 << 20
SCAN_BLOCK_SIZE = 4 << 20


def _log(msg: str) -> None:
    print(msg, flush=True)


def list_split_files() -> dict[str, list[str]]:
    """Parquet shard names per split at the pinned revision.

    Only `train-*.parquet` and `valid-*.parquet` are returned; `test-*`
    is filtered out here (and never requested).
    """
    import urllib.request

    url = (f"https://huggingface.co/api/datasets/{REPO_ID}/tree/{REVISION}"
           "?recursive=true")
    with urllib.request.urlopen(url, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    out: dict[str, list[str]] = {s: [] for s in ALLOWED_SPLITS}
    for item in payload:
        if item.get("type") != "file":
            continue
        path = str(item.get("path") or "")
        if not path.startswith("data/") or not path.endswith(".parquet"):
            continue
        name = path.split("/", 1)[1]
        split = name.split("-", 1)[0]
        if split in out:
            out[split].append(name)
        elif split == "test":
            _log(f"ignored_test_shard:{name}")
    for split in ALLOWED_SPLITS:
        out[split].sort()
        if not out[split]:
            raise RuntimeError(f"no_parquet_shards_for_split:{split}")
    return out


def shard_url(name: str) -> str:
    return (f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}"
            f"/data/{name}")


def _scan_shard(name: str) -> list[dict]:
    """Metadata-only read of one shard (audio column never touched)."""
    import fsspec
    import pyarrow.parquet as pq

    assert_supported_split(name.split("-", 1)[0])
    if name.startswith("test-"):
        raise UnsupportedSplitError(f"test_shard_refused:{name}")

    rows: list[dict] = []
    with fsspec.open(shard_url(name), "rb", block_size=SCAN_BLOCK_SIZE,
                     cache_type="readahead") as fh:
        pf = pq.ParquetFile(fh)
        for rg_idx in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=METADATA_COLUMNS)
            for i, rec in enumerate(table.to_pylist()):
                rec["_shard"] = name
                rec["_row_group"] = rg_idx
                rec["_row_index"] = i
                rows.append(rec)
    return rows


def scan_split(split: str, shards: list[str]) -> list[dict]:
    """All metadata rows of one allowed split, cached to a local JSONL."""
    split = assert_supported_split(split)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{split}.jsonl"
    if cache.is_file() and cache.stat().st_size > 0:
        rows = [json.loads(line) for line in
                cache.read_text(encoding="utf-8").splitlines() if line.strip()]
        if rows and all(row_value(r, "filename") is not None for r in rows):
            _log(f"cache_hit:{split} rows={len(rows)}")
            return rows

    t0 = time.time()
    rows: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=SCAN_WORKERS) as pool:
        for part in pool.map(_scan_shard, shards):
            rows.extend(part)
    rows.sort(key=lambda r: (r["_shard"], r["_row_group"], r["_row_index"]))
    with open(cache, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(rec, ensure_ascii=False) + "\n" for rec in rows)
    _log(f"scanned:{split} rows={len(rows)} shards={len(shards)} "
         f"elapsed={time.time() - t0:.1f}s")
    return rows


def decode_ok(data: bytes) -> tuple[bool, str, dict]:
    """Full decode check + source audio properties (no normalization)."""
    import soundfile as sf

    try:
        info = sf.info(io.BytesIO(data))
        # Full decode: proves the file is readable end to end.
        sf.read(io.BytesIO(data), dtype="float32")
    except Exception as exc:  # noqa: BLE001 - record every decode failure
        return False, f"decode_failed:{type(exc).__name__}:{exc}", {}
    if info.frames <= 0:
        return False, "decode_failed:zero_frames", {}
    return True, "ok", {
        "samplerate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_seconds": round(info.frames / float(info.samplerate), 3),
        "format": str(info.format),
        "subtype": str(info.subtype),
    }


def audio_extension(data: bytes) -> str:
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return ".wav"
    if data[:4] == b"fLaC":
        return ".flac"
    if data[:3] == b"ID3" or (data[:1] == b"\xff" and data[1] & 0xE0 == 0xE0):
        return ".mp3"
    return ".bin"


class AudioFetcher:
    """Row-group level audio fetch with caching and usability checks.

    One parquet row group is downloaded at most once; every candidate row
    inside it (top-CANDIDATE_DEPTH ranked rows per province) is extracted
    from that single download. Rows outside the prefetched set trigger one
    extra row-group load on demand.
    """

    def __init__(self,
                 member_index: dict[tuple, set[int]] | None = None) -> None:
        self.member_index = member_index or {}
        self.cache: dict[tuple, dict] = {}
        self.loaded: set[tuple] = set()
        self.rg_fetches = 0
        self.bytes_downloaded = 0
        self.decode_checks = 0

    @staticmethod
    def rg_key(row: dict) -> tuple:
        return (row.get("_shard"), row.get("_row_group"))

    @staticmethod
    def key(row: dict) -> tuple:
        return (row.get("_shard"), row.get("_row_group"),
                row.get("_row_index"))

    def usable(self, row: dict) -> bool:
        record = self.ensure(row)
        return bool(record and record.get("ok"))

    def ensure(self, row: dict) -> dict | None:
        key = self.rg_key(row)
        if key not in self.loaded:
            self._load(key, {int(row["_row_index"])})
        return self.cache.get(self.key(row))

    def prefetch(self, rows: list[dict], workers: int = AUDIO_WORKERS
                 ) -> None:
        """Load every distinct row group of `rows` once, in parallel."""
        groups: dict[tuple, set[int]] = {}
        for row in rows:
            key = self.rg_key(row)
            if key in self.loaded:
                continue
            groups.setdefault(key, set()).add(int(row["_row_index"]))
        if not groups:
            return
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = []
            for key, extra in groups.items():
                members = set(self.member_index.get(key, ())) | extra
                futures.append(pool.submit(self._load, key, members))
            for fut in futures:
                fut.result()
        _log(f"prefetched_row_groups={len(futures)} "
             f"elapsed={time.time() - t0:.1f}s")

    def _load(self, key: tuple, member_indices: set[int]) -> None:
        shard, row_group = key
        if key in self.loaded:
            return
        if str(shard).startswith("test-"):
            raise UnsupportedSplitError(f"test_shard_refused:{shard}")
        assert_supported_split(str(shard).split("-", 1)[0])

        import fsspec
        import pyarrow.parquet as pq

        chunk_bytes = 0
        with fsspec.open(shard_url(str(shard)), "rb",
                         block_size=AUDIO_BLOCK_SIZE,
                         cache_type="readahead") as fh:
            pf = pq.ParquetFile(fh)
            table = pf.read_row_group(int(row_group), columns=["audio"])
            column = table.column("audio")
            for idx in sorted(member_indices):
                value = column[int(idx)].as_py()
                data = (value or {}).get("bytes") or b""
                source_name = (value or {}).get("path") or ""
                chunk_bytes += len(data)
                record: dict = {"ok": False, "source_name": source_name}
                if not data:
                    record["reason"] = "empty_audio_bytes"
                else:
                    ok, reason, props = decode_ok(data)
                    self.decode_checks += 1
                    record.update(ok=ok, reason=reason, props=props,
                                  bytes=data)
                self.cache[(str(shard), int(row_group), int(idx))] = record
        self.loaded.add(key)
        self.rg_fetches += 1
        self.bytes_downloaded += chunk_bytes
        _log(f"row_group_loaded={shard}#rg{row_group} "
             f"members={len(member_indices)} audio_bytes={chunk_bytes}")


def audio_out_name(filename: str, detected_ext: str) -> str:
    """Destination file name: never double the extension."""
    if filename.lower().endswith((".wav", ".flac", ".mp3", ".m4a", ".ogg")):
        return filename
    return filename + detected_ext


def build_run_dir(run_id: str | None) -> Path:
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "data" / "materialized" / "vimd" / run_id
    if out.exists() and (out / "materialization_report.json").is_file():
        raise FileExistsError(f"refusing_to_overwrite:{out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def stage_scan() -> dict[str, list[dict]]:
    shards = list_split_files()
    _log(f"shards train={len(shards['train'])} valid={len(shards['valid'])}")
    rows = {}
    for split in ALLOWED_SPLITS:
        rows[split] = scan_split(split, shards[split])
    return rows


def stage_build(run_dir: Path, rows: dict[str, list[dict]]) -> dict:
    t0 = time.time()
    train_rows, valid_rows = rows["train"], rows["valid"]
    fetcher = AudioFetcher()

    # Top-K ranked candidates per province share row groups: index them so
    # one row-group download serves every candidate inside it.
    ranked: dict[str, dict[str, list[dict]]] = {
        "train": ranked_candidates(train_rows, "train"),
        "valid": ranked_candidates(valid_rows, "valid"),
    }
    member_index: dict[tuple, set[int]] = {}
    for split, by_province in ranked.items():
        for candidates in by_province.values():
            for row in candidates[:CANDIDATE_DEPTH]:
                member_index.setdefault(
                    (row["_shard"], row["_row_group"]), set()).add(
                        int(row["_row_index"]))
    fetcher.member_index = member_index

    first_choices = [by_province[p][0]
                     for by_province in ranked.values()
                     for p in sorted(by_province)]
    fetcher.prefetch(first_choices, workers=AUDIO_WORKERS)

    selection = materialize_selection(
        train_rows, valid_rows, is_usable=fetcher.usable)

    report_rows = {}
    manifests = {}
    audio_written = []
    total_bytes = 0
    decode_props = {}
    for split in ("train", "valid"):
        (run_dir / "audio" / split).mkdir(parents=True, exist_ok=True)
        out_rows = []
        for row in selection[split]:
            rec = fetcher.cache[AudioFetcher.key(row)]
            filename = str(row_value(row, "filename"))
            ext = audio_extension(rec["bytes"])
            dest = (run_dir / "audio" / split
                    / audio_out_name(filename, ext))
            dest.write_bytes(rec["bytes"])
            audio_written.append(str(dest))
            total_bytes += len(rec["bytes"])
            decode_props[f"{split}:{filename}"] = rec["props"]
            out_rows.append(canonical_row(
                "vimd", split, row, str(dest.resolve()),
                sample_id=build_sample_id(split, filename),
            ))
        manifests[split] = out_rows
        report_rows[split] = sample_stats(out_rows)

    for split, path in (("train", "manifest_train.jsonl"),
                        ("valid", "manifest_valid.jsonl")):
        with open(run_dir / path, "w", encoding="utf-8") as fh:
            for rec in manifests[split]:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    overlap = overlap_report(manifests["train"], manifests["valid"])
    elapsed = time.time() - t0

    report = {
        "dataset": "vimd",
        "source": {
            "hf_dataset": REPO_ID,
            "revision": REVISION,
            "splits_accessed": list(ALLOWED_SPLITS),
            "test_accessed": False,
            "shard_count": {s: len({r["_shard"] for r in rows[s]})
                            for s in ALLOWED_SPLITS},
            "scanned_rows": {s: len(rows[s]) for s in ALLOWED_SPLITS},
        },
        "selection": {
            "target_per_split": TARGET_TRAIN,
            "ranking": "sha256(split + filename)",
            "train_selected": len(manifests["train"]),
            "valid_selected": len(manifests["valid"]),
            "train_advances": selection["train_advances"],
            "valid_advances": selection["valid_advances"],
            "rejections": selection["rejections"],
        },
        "overlap": overlap,
        "split_stats": report_rows,
        "audio": {
            "files_written": len(audio_written),
            "bytes_written": total_bytes,
            "duration_seconds_total": round(
                sum(p.get("duration_seconds", 0)
                    for p in decode_props.values()), 3),
            "decode_success": len(decode_props),
            "decode_failures": sum(
                1 for r in selection["rejections"]
                if r.get("reason") == "audio_unusable"),
            "properties_sample": dict(list(decode_props.items())[:4]),
            "source_row_group_fetches": fetcher.rg_fetches,
            "source_decode_checks": fetcher.decode_checks,
            "source_member_audio_bytes_extracted": fetcher.bytes_downloaded,
            "normalized": False,
            "trimmed": False,
            "vad": False,
        },
        "materialization_elapsed_seconds": round(elapsed, 2),
        "counts": {
            "train": len(manifests["train"]),
            "valid": len(manifests["valid"]),
            "total": len(manifests["train"]) + len(manifests["valid"]),
        },
    }

    # §F hard guards
    problems = []
    if overlap["train_count"] != TARGET_TRAIN:
        problems.append(f"train_count:{overlap['train_count']}")
    if overlap["valid_count"] != _TARGET_VALID:
        problems.append(f"valid_count:{overlap['valid_count']}")
    if overlap["train_unique_provinces"] != TARGET_TRAIN:
        problems.append(
            f"train_provinces:{overlap['train_unique_provinces']}")
    if overlap["valid_unique_provinces"] != _TARGET_VALID:
        problems.append(
            f"valid_provinces:{overlap['valid_unique_provinces']}")
    if overlap["filename_overlap"]:
        problems.append(f"filename_overlap:{overlap['filename_overlap']}")
    if overlap["speaker_overlap"]:
        problems.append(f"speaker_overlap:{overlap['speaker_overlap']}")
    if overlap["province_symmetric_difference"]:
        problems.append(
            "province_mismatch:"
            + ",".join(overlap["province_symmetric_difference"]))
    report["guard_problems"] = problems
    report["guard_ok"] = not problems

    (run_dir / "materialization_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    source_manifest = {
        "hf_dataset": REPO_ID,
        "revision": REVISION,
        "splits": list(ALLOWED_SPLITS),
        "test_excluded": True,
        "shards": {s: sorted({r["_shard"] for r in rows[s]})
                   for s in ALLOWED_SPLITS},
        "metadata_columns": METADATA_COLUMNS,
        "manifest_train_sha256_input": "raw source rows (unmodified)",
        "row_counts": {s: len(rows[s]) for s in ALLOWED_SPLITS},
    }
    (run_dir / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8")
    return report


def stage_sample(run_dir: Path) -> dict:
    """§K: 24-row TRAIN-only discovery sample."""
    path = run_dir / "manifest_train.jsonl"
    rows = [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]
    sample = select_discovery_sample(rows, target=DISCOVERY_TARGET,
                                     split="train")
    out = run_dir / "sample_for_discovery.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(rec, ensure_ascii=False) + "\n" for rec in sample)
    stats = sample_stats(sample)
    _log(f"discovery_sample rows={stats['rows']} "
         f"provinces={stats['unique_provinces']} "
         f"regions={stats['region_distribution']}")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all",
                        choices=["scan", "build", "sample", "all"])
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    rows = None
    if args.stage in ("scan", "build", "all"):
        rows = stage_scan()
    if args.stage in ("build", "all"):
        run_dir = build_run_dir(args.run_id)
        _log(f"run_dir={run_dir}")
        report = stage_build(run_dir, rows)
        _log(json.dumps({k: report[k] for k in
                         ("counts", "guard_ok", "guard_problems",
                          "materialization_elapsed_seconds")},
                        indent=2))
        if not report["guard_ok"]:
            _log("GUARD FAILED")
            return 1
        stats = stage_sample(run_dir)
        _log(json.dumps(stats, indent=2))
    elif args.stage == "sample":
        if args.run_id is None:
            raise SystemExit("--run-id required with --stage sample")
        run_dir = ROOT / "data" / "materialized" / "vimd" / args.run_id
        _log(json.dumps(stage_sample(run_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Regenerate data/vimd/sample.jsonl from the source ViMD parquet.

ViMD raw gender encoding (confirmed, NOT an assumption):
    0 = female
    1 = male

Local only: reads parquet, writes sample.jsonl. No network, no LLM.

Usage:
    python scripts/sauvi_perception/prepare_vimd.py
"""

from __future__ import annotations

import glob
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path("G:/VietnameseSpeechQABenchmark/data/vimd/data")
MANIFEST_PATH = ROOT / "data" / "vimd" / "manifest.jsonl"
OUT_PATH = ROOT / "data" / "vimd" / "sample.jsonl"

GENDER_MAP = {
    0: "female",
    1: "male",
}


def normalize_vimd_gender(value):
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid ViMD gender value: {value!r}")

    if value not in GENDER_MAP:
        raise ValueError(f"Unknown ViMD gender value: {value!r}")

    return GENDER_MAP[value]


def main() -> int:
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(DATA_DIR / "*.parquet")))
    if not files:
        print(f"No parquet files found in {DATA_DIR}")
        return 1

    # Raw distribution diagnostic BEFORE normalizing. Only 0/1 are known.
    raw_values: Counter = Counter()
    rows: list[dict] = []
    for fp in files:
        table = pq.read_table(
            fp,
            columns=["region", "province_name", "filename", "text", "speakerID", "gender", "audio"],
        )
        for r in table.to_pylist():
            raw_values[r["gender"]] += 1
            audio_path = (r.get("audio") or {}).get("path") or r["filename"]
            rows.append(
                {
                    "audio": f"G:/VietnameseSpeechQABenchmark/data/vimd/{audio_path}",
                    "text": r["text"],
                    "gender": normalize_vimd_gender(r["gender"]),
                    "region": r["region"],
                    "province": r["province_name"],
                    "speakerID": r["speakerID"],
                }
            )

    print("raw gender distribution:", dict(sorted(raw_values.items())))
    unknown = set(raw_values) - set(GENDER_MAP)
    if unknown:
        print(f"STOP: unexpected raw gender labels: {sorted(unknown, key=repr)}")
        return 1

    print(f"total rows: {len(rows)}")
    print("regions:", dict(Counter(r["region"] for r in rows)))
    print("genders:", dict(Counter(r["gender"] for r in rows)))
    print("provinces:", len(set(r["province"] for r in rows)))

    by_speaker: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_speaker[r["speakerID"]].append(i)
    multi = {k: v for k, v in by_speaker.items() if len(v) >= 2}
    print(f"speakers with >=2 utterances: {len(multi)}")

    rng = random.Random(42)
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        buckets[(r["region"], r["gender"])].append(i)

    picked: list[int] = []
    per_bucket = 4
    for key in sorted(buckets):
        pool = buckets[key][:]
        rng.shuffle(pool)
        seen_prov: set[str] = set()
        for i in pool:
            picked.append(i)
            seen_prov.add(rows[i]["province"])
            if len([p for p in picked if (rows[p]["region"], rows[p]["gender"]) == key]) >= per_bucket:
                break

    # Guarantee a same-speaker pair.
    pair_spk = sorted(multi)[0]
    for i in multi[pair_spk][:2]:
        if i not in picked:
            picked.append(i)

    picked = picked[:26]
    sample = [rows[i] for i in picked]
    print(f"sampled: {len(sample)}")
    spk_in_sample = Counter(r["speakerID"] for r in sample)
    print("same-speaker pair in sample:", any(c >= 2 for c in spk_in_sample.values()))

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {MANIFEST_PATH} ({len(rows)} rows)")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {OUT_PATH} ({len(sample)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

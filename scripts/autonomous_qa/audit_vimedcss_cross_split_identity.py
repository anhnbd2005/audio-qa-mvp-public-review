"""Cross-split identity diagnostic for ViMedCSS (non-blocking, no mutation).

Reports whether the same ``segment_id`` (and audio identity) leaks across the
train/validation/test/hard source splits. This is a DIAGNOSTIC only: it never
dedupes, reassigns, or blocks planning/generation. Exits 0 always.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SPLITS = ("train", "validation", "test", "hard")
ROOT = Path(__file__).resolve().parents[2]


def _segment_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.append(json.loads(line)["segment_id"])
    return ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="vimedcss")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    source_dir = ROOT / "data_sources" / args.dataset / "source"
    per_split: dict[str, list[str]] = {}
    for split in SPLITS:
        path = source_dir / f"{split}.jsonl"
        if path.exists():
            per_split[split] = _segment_ids(path)

    within: dict[str, dict[str, int]] = {}
    for split, ids in per_split.items():
        within[split] = {
            "rows": len(ids),
            "unique_segment_ids": len(set(ids)),
            "duplicate_segment_ids": len(ids) - len(set(ids)),
        }

    overlaps: list[dict] = []
    for a, b in combinations(per_split, 2):
        shared = set(per_split[a]) & set(per_split[b])
        if shared:
            overlaps.append(
                {
                    "splits": [a, b],
                    "shared_segment_id_count": len(shared),
                    "sample": sorted(shared)[:10],
                }
            )

    report = {
        "dataset": args.dataset,
        "splits": list(per_split),
        "within_split": within,
        "cross_split_overlaps": overlaps,
        "total_shared_segment_ids": sum(
            row["shared_segment_id_count"] for row in overlaps
        ),
        "blocking": False,
        "note": "Diagnostic only; leakage is reported, never auto-corrected.",
    }
    out = args.output or (
        ROOT / "outputs" / "_scratch" / "vimedcss_v3" / "cross_split_identity_report.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

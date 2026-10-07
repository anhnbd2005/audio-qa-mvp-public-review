"""Onboard the official VietLyrics metadata source (acquisition + audit only).

    python scripts/sauvi_perception/onboard_vietlyrics.py fetch
    python scripts/sauvi_perception/onboard_vietlyrics.py verify
    python scripts/sauvi_perception/onboard_vietlyrics.py audit
    python scripts/sauvi_perception/onboard_vietlyrics.py all

No audio, no P7 QA, no density thresholds, no LLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sauvi_perception.datasets.vietlyrics import vietlyrics_source as vl

AUDIT_DIR = ROOT / "outputs" / "source_audits" / "vietlyrics" / vl.REVISION


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cmd_fetch() -> int:
    result = vl.fetch_source()
    print(json.dumps({"mode": "fetch", "files": result}, ensure_ascii=False, indent=2))
    return 0


def cmd_verify() -> int:
    result = vl.verify_source()
    print(json.dumps({"mode": "verify", "files": result}, ensure_ascii=False, indent=2))
    return 0


def cmd_audit() -> int:
    audit = vl.audit_source()
    train_rows = vl.load_train_rows()
    val_rows = vl.load_val_rows()
    full_wpm = vl.audit_wpm(list(train_rows) + list(val_rows))

    _write_json(AUDIT_DIR / "source_audit.json", audit)
    _write_json(
        AUDIT_DIR / "wpm_audit.json",
        {
            "note": "FULL TRAIN+VAL statistics are report-only; no density thresholds",
            "train": audit["wpm"]["train"],
            "validation": audit["wpm"]["validation"],
            "full": full_wpm,
        },
    )
    _write_json(AUDIT_DIR / "zing_id_audit.json", audit["zing_ids"])
    _write_json(AUDIT_DIR / "split_overlap_audit.json", audit["cross_split"])
    _write_json(AUDIT_DIR / "provenance_snapshot.json", vl.load_provenance())

    summary = {
        "mode": "audit",
        "out_dir": str(AUDIT_DIR),
        "revision": audit["revision"],
        "train_rows": audit["files"]["train"]["logical_records"],
        "val_rows": audit["files"]["validation"]["logical_records"],
        "train_wpm_valid": audit["wpm"]["train"]["valid_wpm"],
        "val_wpm_valid": audit["wpm"]["validation"]["valid_wpm"],
        "train_zing_duplicates": audit["zing_ids"]["train"]["duplicate_ids"],
        "val_zing_duplicates": audit["zing_ids"]["validation"]["duplicate_ids"],
        "zing_id_overlap_count": audit["cross_split"]["zing_id_overlap_count"],
        "blocking": audit["cross_split"]["blocking"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fetch", "verify", "audit", "all"))
    args = parser.parse_args()
    if args.mode == "fetch":
        return cmd_fetch()
    if args.mode == "verify":
        return cmd_verify()
    if args.mode == "audit":
        return cmd_audit()
    rc = cmd_fetch()
    rc |= cmd_verify()
    rc |= cmd_audit()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

"""Sanitize VietLyrics assets: split isolation + SAUVI contamination audit.

    python scripts/sauvi_perception/sanitize_vietlyrics_assets.py
    python scripts/sauvi_perception/sanitize_vietlyrics_assets.py --sauvi-manifest <path>

No P7 QA, no density thresholds, no audio, no LLM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sauvi_perception.datasets.vietlyrics import vietlyrics_sanitation as vs
from src.sauvi_perception.datasets.vietlyrics import vietlyrics_source as vl

OUT_DIR = (
    ROOT / "outputs" / "source_audits" / "vietlyrics" / vl.REVISION / "asset_sanitation"
)
CANONICAL_DIR = (
    ROOT / "data" / "materialized" / "vietlyrics" / vl.REVISION / "asset_sanitation"
)


def _run(sauvi_manifest: Path | None) -> dict:
    vl.verify_source()
    train_rows = vl.load_train_rows()
    val_rows = vl.load_val_rows()

    manifest_path = vs.resolve_sauvi_manifest(sauvi_manifest)
    sauvi_rows = vs.load_sauvi_vietlyrics_rows(manifest_path)
    sauvi = vs.build_sauvi_reserved(sauvi_rows)
    sauvi["manifest_path"] = manifest_path.name
    sauvi["manifest_sha256"] = vs._sha256_file(manifest_path)

    inventory, unassigned_train, unassigned_val = vs.build_asset_inventory(
        train_rows, val_rows
    )
    classifications = vs.classify_all(inventory, set(sauvi["reserved_asset_ids"]))
    source_only, final = vs.build_clean_assets(classifications)
    excluded = vs.build_exclusion_audit(inventory, classifications)
    summary = vs.sanitation_summary(
        train_rows, val_rows, inventory, classifications, sauvi
    )
    independent = vs.independent_audit(
        train_rows, val_rows, classifications, inventory, sauvi
    )

    train_ids = {aid for aid, e in inventory.items() if e["train"]}
    val_ids = {aid for aid, e in inventory.items() if e["val"]}

    # inventory jsonl (one record per unique asset, full members)
    inventory_records = []
    for asset_id, entry in sorted(inventory.items()):
        record = dict(classifications[asset_id])
        record["train_members"] = entry["train"]
        record["val_members"] = entry["val"]
        inventory_records.append(record)

    duplicate_assets = []
    wpm_conflict_assets = []
    for asset_id, cls in sorted(classifications.items()):
        entry = inventory[asset_id]
        if cls["appears_val"] or len(entry["train"]) < 2:
            continue
        record = {
            "asset_id": asset_id,
            "train_member_count": len(entry["train"]),
            "wpm_values": [m["wpm_raw"] for m in entry["train"]],
            "terminal_status": cls["terminal_status"],
            "wpm_conflict": cls["wpm_conflict"],
            "metadata_inconsistency_report_only": cls[
                "metadata_inconsistency_report_only"
            ],
            "members": entry["train"],
        }
        duplicate_assets.append(record)
        if cls["wpm_conflict"]:
            wpm_conflict_assets.append(record)

    source_split_audit = {
        "unique_train_ids": len(train_ids),
        "unique_val_ids": len(val_ids),
        "train_val_overlap_ids": sorted(train_ids & val_ids),
        "train_val_overlap_count": len(train_ids & val_ids),
        "train_only_count": len(train_ids - val_ids),
        "val_only_count": len(val_ids - train_ids),
        "unassigned_train_rows": unassigned_train,
        "unassigned_val_rows": unassigned_val,
    }

    sauvi_overlap = {
        "reserved_unique_ids": sauvi["unique_asset_ids"],
        "source_mapping": summary["sauvi"]["source_mapping"],
        "reserved_ids_in_train": sorted(set(sauvi["reserved_asset_ids"]) & train_ids),
        "reserved_ids_in_val": sorted(set(sauvi["reserved_asset_ids"]) & val_ids),
        "reserved_ids_in_source_only_clean": sorted(
            {r["asset_id"] for r in source_only} & set(sauvi["reserved_asset_ids"])
        ),
    }

    vs._write_jsonl(OUT_DIR / "asset_inventory.jsonl", inventory_records)
    vs._write_jsonl(CANONICAL_DIR / "asset_inventory.jsonl", inventory_records)
    vs._write_json(
        OUT_DIR / "duplicate_asset_audit.json",
        {
            "duplicate_asset_count": len(duplicate_assets),
            "wpm_consistent_count": sum(
                1 for a in duplicate_assets if not a["wpm_conflict"]
            ),
            "wpm_conflict_count": len(wpm_conflict_assets),
            "assets": duplicate_assets,
        },
    )
    vs._write_json(
        OUT_DIR / "wpm_conflict_audit.json",
        {"conflict_count": len(wpm_conflict_assets), "assets": wpm_conflict_assets},
    )
    vs._write_json(OUT_DIR / "source_split_asset_audit.json", source_split_audit)
    vs._write_json(OUT_DIR / "sauvi_vietlyrics_reserved_ids.json", sauvi)
    vs._write_json(CANONICAL_DIR / "sauvi_vietlyrics_reserved_ids.json", sauvi)
    vs._write_json(OUT_DIR / "sauvi_overlap_audit.json", sauvi_overlap)
    vs._write_jsonl(OUT_DIR / "clean_train_assets_source_only.jsonl", source_only)
    vs._write_jsonl(OUT_DIR / "clean_train_assets.jsonl", final)
    vs._write_jsonl(CANONICAL_DIR / "clean_train_assets.jsonl", final)
    vs._write_jsonl(OUT_DIR / "excluded_train_rows.jsonl", excluded)
    vs._write_json(OUT_DIR / "sanitation_summary.json", summary)
    vs._write_json(OUT_DIR / "independent_audit.json", independent)
    return {"summary": summary, "independent": independent, "out_dir": str(OUT_DIR), "canonical_dir": str(CANONICAL_DIR)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sauvi-manifest", default=None)
    args = parser.parse_args()
    result = _run(Path(args.sauvi_manifest) if args.sauvi_manifest else None)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    ok = (
        result["independent"]["all_train_rows_accounted"]
        and result["independent"]["all_val_rows_accounted"]
        and result["independent"]["no_val_asset_survives"]
        and result["independent"]["no_sauvi_asset_survives"]
        and result["independent"]["no_wpm_conflict_survives"]
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

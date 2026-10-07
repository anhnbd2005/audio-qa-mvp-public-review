"""Build SAUVI P3 (speaker age recognition) training QA from Speech-MASSIVE vi-VN.

    python scripts/sauvi_perception/generate_p3_training_qa.py smoke
    python scripts/sauvi_perception/generate_p3_training_qa.py full
    python scripts/sauvi_perception/generate_p3_training_qa.py all

Source: frozen FBK-MT/Speech-MASSIVE vi-VN ``train_115`` only. Gold is a
deterministic derived age group. No audio is materialized, no validation/test
data is touched, and no LLM is used. Generation runs twice and must be
byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sauvi_perception.tasks.p3_speaker_age import speaker_age_p3 as p3

P3_OUT = ROOT / "outputs" / "training_qa" / "speech_massive_vi" / "p3" / "current"
SCRATCH = ROOT / "outputs" / "training_qa" / "_p3_scratch"
COMPARE_FILES = (
    "qa_internal.jsonl",
    "qa_model_facing.jsonl",
    "audit.json",
    "template_registry.json",
    "manifest.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _prepare() -> tuple[dict, list, list, dict, dict, set]:
    p3.verify_source()
    policy = p3.load_policy()
    rows = p3.load_rows()
    raw_audit = p3.audit_raw_ages(rows)
    consistency = p3.speaker_age_consistency(rows)
    conflicting = set(consistency["p3_bin_conflicting_speaker_ids"])
    anchors = p3.eligible_rows(rows, conflicting)
    return policy, rows, anchors, raw_audit, consistency, conflicting


def _write_pair(
    records: list[dict],
    raw_audit: dict,
    consistency: dict,
    policy: dict,
    out_dir: Path,
) -> dict:
    dir_a = SCRATCH / out_dir.name / "run_a"
    dir_b = SCRATCH / out_dir.name / "run_b"
    result = p3.write_release(dir_a, records, raw_audit, consistency, policy)
    p3.write_release(dir_b, records, raw_audit, consistency, policy)
    hashes_a = {name: _sha256(dir_a / name) for name in COMPARE_FILES}
    hashes_b = {name: _sha256(dir_b / name) for name in COMPARE_FILES}
    mismatches = [name for name in COMPARE_FILES if hashes_a[name] != hashes_b[name]]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(dir_a, out_dir)
    return {
        "audit": result["audit"],
        "distribution": result["distribution"],
        "manifest": result["manifest"],
        "hashes": hashes_a,
        "byte_identical": not mismatches,
        "mismatches": mismatches,
    }


def cmd_smoke() -> int:
    policy, _rows, anchors, raw_audit, consistency, _conf = _prepare()
    selected = p3.select_smoke(anchors, min(100, len(anchors)))
    records = p3.build_qa(selected, policy)
    dir_a = SCRATCH / "smoke" / "run_a"
    dir_b = SCRATCH / "smoke" / "run_b"
    result = p3.write_release(dir_a, records, raw_audit, consistency, policy)
    p3.write_release(dir_b, records, raw_audit, consistency, policy)
    hashes_a = {name: _sha256(dir_a / name) for name in COMPARE_FILES}
    hashes_b = {name: _sha256(dir_b / name) for name in COMPARE_FILES}
    mismatches = [name for name in COMPARE_FILES if hashes_a[name] != hashes_b[name]]
    print(
        json.dumps(
            {
                "mode": "smoke",
                "rows": len(records),
                "audit_ok": p3.audit_ok(result["audit"]),
                "byte_identical": not mismatches,
                "mismatches": mismatches,
                "result": "PASS"
                if p3.audit_ok(result["audit"]) and not mismatches
                else "FAIL",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if p3.audit_ok(result["audit"]) and not mismatches else 1


def cmd_full() -> int:
    policy, rows, anchors, raw_audit, consistency, _conf = _prepare()
    records = p3.build_qa(anchors, policy)
    info = _write_pair(records, raw_audit, consistency, policy, P3_OUT)
    independent = p3.independent_gold_audit(P3_OUT / "qa_internal.jsonl", rows)
    _write_json(P3_OUT / "independent_gold_audit.json", independent)
    distribution = info["distribution"]
    independent_ok = (
        independent["qa_count"] == len(records)
        and independent["wrong_age_parse"] == 0
        and independent["wrong_age_bin_gold"] == 0
        and independent["answer_not_in_choices"] == 0
        and independent["invalid_choice_count"] == 0
        and independent["duplicate_choices"] == 0
    )
    status = (
        "PASS"
        if info["manifest"]["audit_status"] == "PASS" and independent_ok
        else "FAIL"
    )
    print(
        json.dumps(
            {
                "mode": "full",
                "out_dir": str(P3_OUT),
                "counts": info["manifest"]["counts"],
                "class_distribution": distribution,
                "class_coverage_incomplete": distribution["coverage_incomplete"],
                "audit_status": info["manifest"]["audit_status"],
                "independent_gold_audit_ok": independent_ok,
                "byte_identical": info["byte_identical"],
                "mismatches": info["mismatches"],
                "hashes": info["hashes"],
                "result": status,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if status == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "full", "all"))
    args = parser.parse_args()
    if args.mode == "smoke":
        return cmd_smoke()
    if args.mode == "full":
        return cmd_full()
    rc = cmd_smoke()
    rc |= cmd_full()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

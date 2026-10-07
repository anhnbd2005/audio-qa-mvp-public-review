"""Generate the final SAUVI P7 training-QA release from frozen inputs.

    python scripts/sauvi_perception/generate_p7_training_qa.py smoke
    python scripts/sauvi_perception/generate_p7_training_qa.py full
    python scripts/sauvi_perception/generate_p7_training_qa.py repro
    python scripts/sauvi_perception/generate_p7_training_qa.py all

No audio, no LLM, no benchmark access at production runtime, no threshold
recomputation.
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

from src.sauvi_perception.tasks.p7_tempo import p7_production as prod

OUT = ROOT / "outputs" / "training_qa" / "vietlyrics" / "p7" / "current"
SCRATCH = ROOT / "outputs" / "training_qa" / "_p7_scratch"
COMPARE_FILES = (
    "qa_internal.jsonl",
    "qa_model_facing.jsonl",
    "template_registry.json",
    "class_distribution.json",
    "boundary_audit.json",
    "manifest.json",
    "audit.json",
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


def _prepare() -> dict:
    verified = prod.verify_inputs()
    policy = prod.load_policy()
    assets = prod.load_clean_assets()
    isolation = prod.load_isolation_sets()
    records = prod.generate_records(assets, policy, verified["policy_sha256"])
    distribution = prod.class_distribution(records)
    if distribution["counts"] != prod.EXPECTED_CLASS_COUNTS:
        raise prod.P7ProductionError(
            f"P7_FROZEN_POLICY_REPRODUCTION_MISMATCH:{distribution['counts']}"
        )
    return {
        "verified": verified,
        "policy": policy,
        "assets": assets,
        "isolation": isolation,
        "records": records,
        "distribution": distribution,
    }


def _write_pair(data: dict, out_dir: Path) -> dict:
    scratch = SCRATCH / f"{out_dir.name}_work"
    dir_a = scratch / "run_a"
    dir_b = scratch / "run_b"
    result = prod.write_release(
        dir_a,
        data["records"],
        data["assets"],
        data["policy"],
        data["verified"]["policy_sha256"],
        data["isolation"],
    )
    prod.write_release(
        dir_b,
        data["records"],
        data["assets"],
        data["policy"],
        data["verified"]["policy_sha256"],
        data["isolation"],
    )
    hashes_a = {name: _sha256(dir_a / name) for name in COMPARE_FILES}
    hashes_b = {name: _sha256(dir_b / name) for name in COMPARE_FILES}
    mismatches = [name for name in COMPARE_FILES if hashes_a[name] != hashes_b[name]]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(dir_a, out_dir)
    return {
        "result": result,
        "hashes": hashes_a,
        "byte_identical": not mismatches,
        "mismatches": mismatches,
    }


def cmd_smoke() -> int:
    data = _prepare()
    selected = prod.select_smoke(data["assets"], 100)
    subset = prod.generate_records(
        selected, data["policy"], data["verified"]["policy_sha256"]
    )
    sub_data = {**data, "records": subset}
    info = _write_pair(sub_data, SCRATCH / "smoke_current")
    audit = info["result"]["audit"]
    valid = (
        len(subset) == 100
        and prod.audit_ok(audit)
        and info["byte_identical"]
        and audit["isolation"]["val_overlap"] == 0
        and audit["isolation"]["sauvi_overlap"] == 0
        and audit["isolation"]["wpm_conflict_overlap"] == 0
    )
    print(
        json.dumps(
            {
                "mode": "smoke",
                "rows": len(subset),
                "audit_ok": prod.audit_ok(audit),
                "byte_identical": info["byte_identical"],
                "mismatches": info["mismatches"],
                "result": "PASS" if valid else "FAIL",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if valid else 1


def cmd_full() -> int:
    data = _prepare()
    info = _write_pair(data, OUT)
    independent = prod.independent_gold_audit(
        data["records"], data["assets"], data["policy"]
    )
    _write_json(OUT / "independent_audit.json", independent)
    independent_ok = all(
        independent[key] == 0
        for key in (
            "wrong_source_asset",
            "wrong_wpm",
            "wrong_derived_class",
            "answer_not_in_choices",
            "invalid_choice_count",
        )
    )
    status = (
        "PASS"
        if info["result"]["manifest"]["audit_status"] == "PASS" and independent_ok
        else "FAIL"
    )
    print(
        json.dumps(
            {
                "mode": "full",
                "out_dir": str(OUT),
                "counts": info["result"]["manifest"]["counts"],
                "class_distribution": info["result"]["distribution"],
                "choice_position": prod.choice_position(data["records"]),
                "template_distribution": prod.template_distribution(data["records"]),
                "audit_status": info["result"]["manifest"]["audit_status"],
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


def cmd_repro() -> int:
    data = _prepare()
    info = _write_pair(data, SCRATCH / "repro_current")
    print(
        json.dumps(
            {
                "mode": "repro",
                "byte_identical": info["byte_identical"],
                "mismatches": info["mismatches"],
                "hashes": info["hashes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if info["byte_identical"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "full", "repro", "all"))
    args = parser.parse_args()
    if args.mode == "smoke":
        return cmd_smoke()
    if args.mode == "full":
        return cmd_full()
    if args.mode == "repro":
        return cmd_repro()
    rc = cmd_smoke()
    rc |= cmd_full()
    rc |= cmd_repro()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate SAUVI P8 TRAINING QA from ViMD TRAIN metadata.

    python scripts/sauvi_perception/generate_p8_training_qa.py smoke
    python scripts/sauvi_perception/generate_p8_training_qa.py full
    python scripts/sauvi_perception/generate_p8_training_qa.py repro

No LLM, no audio I/O, no network, TRAIN only. All output is deterministic:
running the same mode twice yields byte-identical files.
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

from src.sauvi_perception.tasks.p8_dialect.vimd_p8_training_qa import (
    audit_ok,
    generate_artifacts,
    load_ontology,
    load_train_rows,
    select_smoke_rows,
)

BASE = ROOT / "outputs" / "training_qa" / "vimd" / "p8"
CURRENT = BASE / "current"
ARTIFACT_FILES = (
    "qa_internal.jsonl",
    "qa_model_facing.jsonl",
    "p8_ontology.json",
    "template_registry.json",
    "audit.json",
    "manifest.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hashes(out_dir: Path) -> dict[str, str]:
    return {name: _sha256(out_dir / name) for name in ARTIFACT_FILES}


def _run(rows, ontology, out_dir: Path) -> dict:
    result = generate_artifacts(rows, ontology, out_dir)
    return {
        "out_dir": str(out_dir),
        "counts": result["manifest"]["counts"],
        "audit_status": result["manifest"]["audit_status"],
        "audit": result["audit"],
    }


def _compare(dir_a: Path, dir_b: Path) -> dict:
    hashes_a = _hashes(dir_a)
    hashes_b = _hashes(dir_b)
    mismatches = [name for name in ARTIFACT_FILES if hashes_a[name] != hashes_b[name]]
    return {
        "run_a": hashes_a,
        "run_b": hashes_b,
        "byte_identical": not mismatches,
        "mismatches": mismatches,
    }


def smoke(rows, ontology) -> int:
    sample = select_smoke_rows(rows, 100)
    dir_a = BASE / "_smoke" / "run_a"
    dir_b = BASE / "_smoke" / "run_b"
    result = _run(sample, ontology, dir_a)
    _run(sample, ontology, dir_b)
    comparison = _compare(dir_a, dir_b)
    counts = result["counts"]
    valid = (
        counts == {"l1": 100, "l2": 100, "l3": 100, "total": 300}
        and audit_ok(result["audit"])
        and comparison["byte_identical"]
    )
    summary = {
        "mode": "smoke",
        "rows": len(sample),
        "counts": counts,
        "audit_status": result["audit_status"],
        "byte_identical": comparison["byte_identical"],
        "mismatches": comparison["mismatches"],
        "result": "PASS" if valid else "FAIL",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if valid else 1


def full(rows, ontology) -> int:
    result = _run(rows, ontology, CURRENT)
    print(
        json.dumps(
            {
                "mode": "full",
                "out_dir": result["out_dir"],
                "counts": result["counts"],
                "audit_status": result["audit_status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["audit_status"] == "PASS" else 1


def repro(rows, ontology) -> int:
    dir_a = BASE / "_repro" / "run_a"
    dir_b = BASE / "_repro" / "run_b"
    _run(rows, ontology, dir_a)
    _run(rows, ontology, dir_b)
    comparison = _compare(dir_a, dir_b)
    payload = {
        "mode": "repro",
        "files": comparison["run_a"],
        "run_b": comparison["run_b"],
        "byte_identical": comparison["byte_identical"],
        "mismatches": comparison["mismatches"],
    }
    (BASE / "_repro").mkdir(parents=True, exist_ok=True)
    (BASE / "_repro" / "reproducibility.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if comparison["byte_identical"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "full", "repro"))
    args = parser.parse_args()

    ontology = load_ontology()
    rows = load_train_rows()

    if args.mode == "smoke":
        return smoke(rows, ontology)
    if args.mode == "full":
        return full(rows, ontology)
    return repro(rows, ontology)


if __name__ == "__main__":
    raise SystemExit(main())

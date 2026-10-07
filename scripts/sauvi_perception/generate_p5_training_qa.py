"""Build SAUVI P5 (speech sentiment) training QA from Sentiment-Reasoning.

    python scripts/sauvi_perception/generate_p5_training_qa.py audit
    python scripts/sauvi_perception/generate_p5_training_qa.py smoke
    python scripts/sauvi_perception/generate_p5_training_qa.py full
    python scripts/sauvi_perception/generate_p5_training_qa.py all

TRAIN only; the upstream TEST split is reserved and used solely by the
pre-production contamination audit. No LLM. Generation runs twice and must be
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

from src.sauvi_perception.tasks.p5_sentiment import speech_sentiment_p5 as p5

OUT = ROOT / "outputs" / "training_qa" / "sentiment_reasoning" / "p5" / "current"
SCRATCH = ROOT / "outputs" / "training_qa" / "_p5_scratch"
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


def _prepare() -> dict:
    p5.verify_source()
    policy = p5.load_policy()
    train_rows = p5.load_train_rows()
    source_audit = p5.audit_source(train_rows)
    train_bytes = p5.audio_byte_hashes(p5.TRAIN_CACHE)
    test_bytes = p5.audio_byte_hashes(p5.TEST_CACHE)
    train_pcm = p5.audio_pcm_hashes(p5.TRAIN_CACHE)
    test_pcm = p5.audio_pcm_hashes(p5.TEST_CACHE)
    test_rows = p5.load_test_rows()
    labels = [str(row.get("label")) for row in train_rows]
    texts = [str(row.get("text") or "") for row in train_rows]
    test_texts = [str(row.get("text") or "") for row in test_rows]
    contamination = p5.contamination_audit(
        train_bytes, test_bytes, train_pcm, test_pcm, texts, test_texts
    )
    duplicate = p5.duplicate_audio_decisions(train_bytes, labels)
    empty_audio = {i for i, hashed in enumerate(train_bytes) if hashed is None}
    excluded = (
        set(contamination["contaminated_train_rows"])
        | set(contamination["pcm_contaminated_train_rows"])
        | set(duplicate["same_label_exclude_indices"])
        | set(duplicate["conflicting_exclude_indices"])
    )
    anchors = p5.eligible_rows(
        train_rows, excluded_indices=excluded, empty_audio_indices=empty_audio
    )
    return {
        "policy": policy,
        "train_rows": train_rows,
        "source_audit": source_audit,
        "contamination": contamination,
        "duplicate": duplicate,
        "empty_audio_indices": sorted(empty_audio),
        "excluded_indices": sorted(excluded),
        "anchors": anchors,
    }


def _run_pair(records, policy, out_dir: Path, extra_files: dict) -> dict:
    dir_a = SCRATCH / out_dir.name / "run_a"
    dir_b = SCRATCH / out_dir.name / "run_b"
    result = p5.write_release(dir_a, records, policy, extra_files=extra_files)
    p5.write_release(dir_b, records, policy, extra_files=extra_files)
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


def cmd_audit() -> int:
    data = _prepare()
    OUT.mkdir(parents=True, exist_ok=True)
    _write_json(OUT / "source_audit.json", data["source_audit"])
    _write_json(OUT / "split_overlap_audit.json", data["contamination"])
    _write_json(OUT / "duplicate_audio_audit.json", data["duplicate"])
    print(
        json.dumps(
            {
                "mode": "audit",
                "train_rows": data["source_audit"]["rows"],
                "raw_label_distribution": data["source_audit"][
                    "raw_label_distribution"
                ],
                "contaminated_train_rows": data["contamination"][
                    "contaminated_train_row_count"
                ],
                "pcm_contaminated_train_rows": data["contamination"][
                    "pcm_contaminated_train_row_count"
                ],
                "duplicate_audio_groups": data["duplicate"]["duplicate_audio_groups"],
                "conflicting_label_duplicate_groups": data["duplicate"][
                    "conflicting_label_duplicate_groups"
                ],
                "eligible_rows": len(data["anchors"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_smoke() -> int:
    data = _prepare()
    selected = p5.select_smoke(data["anchors"], min(100, len(data["anchors"])))
    records = p5.build_qa(selected, data["policy"])
    dir_a = SCRATCH / "smoke" / "run_a"
    dir_b = SCRATCH / "smoke" / "run_b"
    result = p5.write_release(dir_a, records, data["policy"])
    p5.write_release(dir_b, records, data["policy"])
    hashes_a = {name: _sha256(dir_a / name) for name in COMPARE_FILES}
    hashes_b = {name: _sha256(dir_b / name) for name in COMPARE_FILES}
    mismatches = [name for name in COMPARE_FILES if hashes_a[name] != hashes_b[name]]
    print(
        json.dumps(
            {
                "mode": "smoke",
                "rows": len(records),
                "audit_ok": p5.audit_ok(result["audit"]),
                "byte_identical": not mismatches,
                "mismatches": mismatches,
                "result": "PASS"
                if p5.audit_ok(result["audit"]) and not mismatches
                else "FAIL",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if p5.audit_ok(result["audit"]) and not mismatches else 1


def cmd_full() -> int:
    data = _prepare()
    records = p5.build_qa(data["anchors"], data["policy"])
    extra = {
        "source_audit.json": data["source_audit"],
        "split_overlap_audit.json": data["contamination"],
        "duplicate_audio_audit.json": data["duplicate"],
    }
    info = _run_pair(records, data["policy"], OUT, extra)
    independent = p5.independent_audit(OUT / "qa_internal.jsonl", data["train_rows"])
    _write_json(OUT / "independent_audit.json", independent)
    independent_ok = (
        independent["qa_count"] == len(records)
        and independent["wrong_source_row"] == 0
        and independent["wrong_gold"] == 0
        and independent["answer_not_in_choices"] == 0
        and independent["invalid_choice_count"] == 0
        and independent["duplicate_choices"] == 0
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
                "audit_status": info["result"]["manifest"]["audit_status"],
                "independent_audit_ok": independent_ok,
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
    parser.add_argument("mode", choices=("audit", "smoke", "full", "all"))
    args = parser.parse_args()
    if args.mode == "audit":
        return cmd_audit()
    if args.mode == "smoke":
        return cmd_smoke()
    if args.mode == "full":
        return cmd_full()
    rc = cmd_audit()
    rc |= cmd_smoke()
    rc |= cmd_full()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

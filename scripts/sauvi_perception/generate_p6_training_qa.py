"""GigaSpeech2-VI P6 (ASR) training QA: onboarding audit + production generation.

    python scripts/sauvi_perception/generate_p6_training_qa.py verify
    python scripts/sauvi_perception/generate_p6_training_qa.py audit
    python scripts/sauvi_perception/generate_p6_training_qa.py smoke
    python scripts/sauvi_perception/generate_p6_training_qa.py production --count 100000

Metadata-first: no audio, no DEV/TEST/SAUVI, no LLM. The distractor universe is
ALL eligible rows via a compact bucket index; selection is deterministic and
nested by rank. The same run generated twice is byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sauvi_perception.tasks.p6_asr.gigaspeech2_vi_p6 import (
    CACHE_PATH,
    COMPARATOR_VERSION,
    SEED,
    build_index_and_anchors,
    generate_stream,
    independent_audit,
    load_policy,
    load_provenance,
    template_registry,
    verify_source,
    write_index,
)

BASE = ROOT / "outputs" / "training_qa" / "gigaspeech2_vi" / "p6"
CURRENT = BASE / "current"
INDEX_DIR = BASE / "_index"
COMPARE_FILES = ("qa_internal.jsonl", "qa_model_facing.jsonl")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _peak_memory_mb() -> float | None:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except Exception:  # noqa: BLE001 - diagnostics only
        return None


def _namespace(target_count: int) -> str:
    if target_count % 1000 == 0:
        return f"{target_count // 1000}k"
    return str(target_count)


def cmd_verify() -> int:
    verified = verify_source(CACHE_PATH, load_provenance())
    print(json.dumps(verified, ensure_ascii=False, indent=2))
    return 0


def cmd_audit() -> int:
    from src.sauvi_perception.tasks.p6_asr.gigaspeech2_vi_p6 import audit_source

    result = audit_source(CACHE_PATH, load_provenance())
    CURRENT.mkdir(parents=True, exist_ok=True)
    _write_json(CURRENT / "source_audit.json", result)
    eligibility = result["eligibility"]
    _write_json(
        CURRENT / "capacity_report.json",
        {
            "eligible_rows": eligibility["eligible_rows"],
            "potential_p6_qa": eligibility["eligible_rows"],
            "unique_logical_audio_ids": eligibility["unique_logical_audio_ids"],
            "unit": "one source audio -> one transcript-selection QA",
            "word_count_buckets": result["word_count_buckets"],
            "character_length": result["character_length"],
            "word_count": result["word_count"],
            "note": "target_count is chosen by the user; no cap applied",
        },
    )
    print(
        json.dumps(
            {
                "mode": "audit",
                "eligible_rows": eligibility["eligible_rows"],
                "duplicate_segment_ids": result["segment_ids"]["duplicate_rows"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _require_clean_audit() -> None:
    audit_path = CURRENT / "source_audit.json"
    if not audit_path.is_file():
        raise SystemExit("source_audit.json missing; run `audit` first")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["segment_ids"]["duplicate_rows"] != 0:
        raise SystemExit(
            "BLOCKED: duplicate segment IDs in source:"
            f"{audit['segment_ids']['duplicate_rows']}"
        )


def _generate_pair(name: str, target_count: int, policy) -> dict:
    _require_clean_audit()
    t0 = time.time()
    eligible, anchors, index = build_index_and_anchors(
        CACHE_PATH, target_count=target_count
    )
    index_build_seconds = time.time() - t0
    index_meta = write_index(index, INDEX_DIR / _namespace(target_count))

    scratch = BASE / f"_{name}"
    dir_a = scratch / "run_a"
    dir_b = scratch / "run_b"
    t1 = time.time()
    result_a = generate_stream(dir_a, anchors, index, policy)
    generation_seconds = time.time() - t1
    generate_stream(dir_b, anchors, index, policy)

    hashes_a = {n: _sha256(dir_a / n) for n in COMPARE_FILES}
    hashes_b = {n: _sha256(dir_b / n) for n in COMPARE_FILES}
    mismatches = [n for n in COMPARE_FILES if hashes_a[n] != hashes_b[n]]

    dest = CURRENT / name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(dir_a, dest)

    return {
        "eligible": eligible,
        "anchors": len(anchors),
        "index_meta": index_meta,
        "index_build_seconds": round(index_build_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "qa_per_second": round(len(anchors) / generation_seconds, 2)
        if generation_seconds
        else 0.0,
        "peak_memory_mb": _peak_memory_mb(),
        "hashes": hashes_a,
        "run_b_hashes": hashes_b,
        "byte_identical": not mismatches,
        "mismatches": mismatches,
        "result": result_a,
    }


def cmd_smoke() -> int:
    policy = load_policy()
    info = _generate_pair("smoke", 100, policy)
    valid = (
        info["result"]["audit"]["counts"]["qa"] == 100
        and info["byte_identical"]
        and info["result"]["manifest"]["audit_status"] == "PASS"
    )
    print(
        json.dumps(
            {
                "mode": "smoke",
                "counts": info["result"]["audit"]["counts"],
                "audit_status": info["result"]["manifest"]["audit_status"],
                "byte_identical": info["byte_identical"],
                "mismatches": info["mismatches"],
                "result": "PASS" if valid else "FAIL",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if valid else 1


def cmd_production(count: int) -> int:
    policy = load_policy()
    name = _namespace(count)
    info = _generate_pair(name, count, policy)
    dest = CURRENT / name
    result = info["result"]

    source_audit = json.loads(
        (CURRENT / "source_audit.json").read_text(encoding="utf-8")
    )
    full_buckets = source_audit["word_count_buckets"]
    selected_buckets = result["length_distribution"]
    length_comparison = {}
    full_total = sum(full_buckets.values())
    selected_total = sum(selected_buckets.values())
    for bucket, full_count in full_buckets.items():
        selected_count = selected_buckets.get(bucket, 0)
        length_comparison[bucket] = {
            "full_count": full_count,
            "full_pct": round(100.0 * full_count / full_total, 4)
            if full_total
            else 0.0,
            "selected_count": selected_count,
            "selected_pct": round(100.0 * selected_count / selected_total, 4)
            if selected_total
            else 0.0,
        }

    independent = independent_audit(
        dest / "qa_internal.jsonl", dest / "qa_model_facing.jsonl"
    )
    _write_json(dest / "independent_audit.json", independent)
    _write_json(dest / "length_distribution_comparison.json", length_comparison)
    _write_json(dest / "source_provenance.json", load_provenance())
    _write_json(
        dest / "generation_policy.json",
        {
            "policy_version": policy["version"],
            "task": policy["task"],
            "semantic_type": policy["semantic_type"],
            "benchmark_category": policy["benchmark_category"],
            "gold_origin": policy["gold_origin"],
            "compiler_tier": policy["compiler_tier"],
            "seed": SEED,
            "comparator_version": COMPARATOR_VERSION,
            "distractor_universe": "all_eligible_rows",
            "distractor_index": "int64_line_offset_buckets",
            "candidate_selection": "sha256(seed|anchor|slot|bucket|attempt) mod bucket_size",
            "length_bucket_policy": "same bucket -> adjacent -> global",
            "target_count": count,
            "nested_targets": True,
        },
    )
    _write_json(
        dest / "performance.json",
        {
            "eligible_rows": info["eligible"],
            "target_count": count,
            "index_build_seconds": info["index_build_seconds"],
            "generation_seconds": info["generation_seconds"],
            "qa_per_second": info["qa_per_second"],
            "peak_memory_mb": info["peak_memory_mb"],
            "index": info["index_meta"],
        },
    )
    manifest = dict(result["manifest"])
    manifest["counts"] = {"qa": count}
    _write_json(dest / "manifest.json", manifest)

    independent_ok = (
        independent["qa_count"] == count
        and independent["unique_qa_ids"] == count
        and independent["unique_anchor_segment_ids"] == count
        and independent["unique_logical_audio_ids"] == count
        and all(
            independent[key] == 0
            for key in (
                "non_train_refined_rows",
                "invalid_gold",
                "answer_not_in_choices",
                "choice_count_not_4",
                "comparator_duplicate_choices",
                "duplicate_choices",
                "self_distractor",
                "absolute_paths",
                "row_level_llm",
                "production_llm",
            )
        )
        and independent["model_facing_schema_ok"]
    )
    print(
        json.dumps(
            {
                "mode": "production",
                "target_count": count,
                "out_dir": str(dest),
                "audit_status": result["manifest"]["audit_status"],
                "byte_identical": info["byte_identical"],
                "mismatches": info["mismatches"],
                "independent_audit_ok": independent_ok,
                "distractor_stats": result["distractor_stats"],
                "choice_position": result["choice_position"],
                "template_distribution": result["template_distribution"],
                "performance": {
                    "index_build_seconds": info["index_build_seconds"],
                    "generation_seconds": info["generation_seconds"],
                    "qa_per_second": info["qa_per_second"],
                    "peak_memory_mb": info["peak_memory_mb"],
                    "index_bytes": info["index_meta"]["binary_bytes"],
                },
                "result": "PASS"
                if (result["manifest"]["audit_status"] == "PASS" and independent_ok)
                else "FAIL",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["manifest"]["audit_status"] == "PASS" and independent_ok else 1


def cmd_manifest() -> int:
    policy = load_policy()
    CURRENT.mkdir(parents=True, exist_ok=True)
    _write_json(CURRENT / "template_registry.json", template_registry(policy))
    _write_json(
        CURRENT / "manifest.json",
        {
            "schema_version": 1,
            "artifact": "gigaspeech2_vi_p6_training_qa_namespace",
            "task": "P6",
            "source": {
                "repo_id": policy["source"]["repo_id"],
                "revision": policy["source"]["revision"],
                "filename": policy["source"]["filename"],
                "split": policy["source"]["split"],
            },
            "policy_version": policy["version"],
            "contents": [
                "smoke",
                "source_audit.json",
                "capacity_report.json",
                "100k",
            ],
            "audio_status": "AUDIO_NOT_DOWNLOADED_BY_DESIGN",
        },
    )
    print(json.dumps({"mode": "manifest", "out": str(CURRENT)}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("verify", "audit", "smoke", "production", "manifest")
    )
    parser.add_argument("--count", type=int, default=100_000)
    args = parser.parse_args()
    if args.mode == "verify":
        return cmd_verify()
    if args.mode == "audit":
        return cmd_audit()
    if args.mode == "smoke":
        return cmd_smoke()
    if args.mode == "production":
        return cmd_production(args.count)
    return cmd_manifest()


if __name__ == "__main__":
    raise SystemExit(main())

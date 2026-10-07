"""Generate SAUVI P1 TRAINING QA (speaker verification) from ViMD TRAIN.

    python scripts/sauvi_perception/generate_p1_training_qa.py smoke
    python scripts/sauvi_perception/generate_p1_training_qa.py full
    python scripts/sauvi_perception/generate_p1_training_qa.py repro
    python scripts/sauvi_perception/generate_p1_training_qa.py audio-smoke

No LLM, no network, TRAIN only. Pair construction is metadata-first and
deterministic; the same mode run twice yields byte-identical artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sauvi_perception.tasks.p1_speaker_verification.vimd_p1_training_qa import (
    audit_ok,
    build_pair_plan,
    build_positive_pairs,
    generate_artifacts,
    load_policy,
    load_recipe,
    load_train_rows,
    select_smoke_positives,
)

BASE = ROOT / "outputs" / "training_qa" / "vimd" / "p1"
CURRENT = BASE / "current"
ARTIFACT_FILES = (
    "p1_pair_plan.jsonl",
    "qa_internal.jsonl",
    "qa_model_facing.jsonl",
    "composite_audio_manifest.jsonl",
    "template_registry.json",
    "p1_audio_recipe.json",
    "audit.json",
    "manifest.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hashes(out_dir: Path) -> dict[str, str]:
    return {name: _sha256(out_dir / name) for name in ARTIFACT_FILES}


def _run(rows, policy, recipe, out_dir: Path, plan) -> dict:
    result = generate_artifacts(rows, policy, recipe, out_dir, plan=plan)
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


def smoke(rows, policy, recipe) -> int:
    positives = select_smoke_positives(build_positive_pairs(rows), 50)
    plan = build_pair_plan(rows, positives=positives, answers=policy["answers"])
    dir_a = BASE / "_smoke" / "run_a"
    dir_b = BASE / "_smoke" / "run_b"
    result = _run(rows, policy, recipe, dir_a, plan)
    _run(rows, policy, recipe, dir_b, plan)
    comparison = _compare(dir_a, dir_b)
    counts = result["counts"]
    valid = (
        counts == {"same": 50, "different": 50, "total": 100}
        and audit_ok(result["audit"])
        and comparison["byte_identical"]
    )
    print(
        json.dumps(
            {
                "mode": "smoke",
                "counts": counts,
                "audit_status": result["audit_status"],
                "negative_tiers": result["audit"]["negative_tiers"],
                "byte_identical": comparison["byte_identical"],
                "mismatches": comparison["mismatches"],
                "result": "PASS" if valid else "FAIL",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if valid else 1


def full(rows, policy, recipe) -> int:
    plan = build_pair_plan(rows, answers=policy["answers"])
    result = _run(rows, policy, recipe, CURRENT, plan)
    capacity_met = result["audit"]["pairs"]["capacity_met"]
    print(
        json.dumps(
            {
                "mode": "full",
                "out_dir": result["out_dir"],
                "counts": result["counts"],
                "positive_capacity": result["audit"]["pairs"]["positive_capacity"],
                "capacity_met": capacity_met,
                "audit_status": result["audit_status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["audit_status"] == "PASS" and capacity_met else 1


def repro(rows, policy, recipe) -> int:
    plan = build_pair_plan(rows, answers=policy["answers"])
    dir_a = BASE / "_repro" / "run_a"
    dir_b = BASE / "_repro" / "run_b"
    _run(rows, policy, recipe, dir_a, plan)
    _run(rows, policy, recipe, dir_b, plan)
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


def _resolve_audio_root() -> Path | None:
    for candidate in (
        os.environ.get("VIMD_AUDIO_ROOT"),
        str(ROOT / "data" / "vimd" / "audio_wav"),
        r"G:\VietnameseSpeechQABenchmark\data\vimd\audio_wav",
    ):
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    return None


def audio_smoke(rows, policy, recipe) -> int:
    """Render 10 composites (5 SAME, 5 DIFFERENT) if source audio is local."""
    import numpy as np
    import soundfile as sf

    from src.sauvi_perception.tasks.p1_speaker_verification.vimd_p1_audio_recipe import compose, verify_composite

    audio_root = _resolve_audio_root()
    if audio_root is None:
        print(
            json.dumps(
                {
                    "mode": "audio-smoke",
                    "result": "AUDIO_RENDER_SMOKE_DEFERRED_TO_KAGGLE",
                    "reason": "source audio not resolvable locally",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    plan = build_pair_plan(rows, answers=policy["answers"])
    selected = [e for e in plan if e["pair_class"] == "same"][:5] + [
        e for e in plan if e["pair_class"] == "different"
    ][:5]
    checks = []
    for entry in selected:
        first, second = entry["rendered_component_order"]
        path_a = next(audio_root.rglob(first), None)
        path_b = next(audio_root.rglob(second), None)
        if path_a is None or path_b is None:
            checks.append({"pair": entry["canonical_pair_id"], "resolved": False})
            continue
        wave_a, sr_a = sf.read(path_a, dtype="float32")
        wave_b, sr_b = sf.read(path_b, dtype="float32")
        composite, sr = compose(wave_a, sr_a, wave_b, sr_b, recipe)
        report = verify_composite(composite, sr, recipe, len(wave_a), len(wave_b))
        report.update(
            {
                "pair": entry["canonical_pair_id"],
                "pair_class": entry["pair_class"],
                "resolved": True,
                "deterministic": bool(
                    np.array_equal(
                        composite, compose(wave_a, sr_a, wave_b, sr_b, recipe)[0]
                    )
                ),
            }
        )
        checks.append(report)
    payload = {"mode": "audio-smoke", "checked": checks}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "full", "repro", "audio-smoke"))
    args = parser.parse_args()

    policy = load_policy()
    recipe = load_recipe()
    rows = load_train_rows()

    if args.mode == "smoke":
        return smoke(rows, policy, recipe)
    if args.mode == "full":
        return full(rows, policy, recipe)
    if args.mode == "repro":
        return repro(rows, policy, recipe)
    return audio_smoke(rows, policy, recipe)


if __name__ == "__main__":
    raise SystemExit(main())

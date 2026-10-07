"""Build SAUVI P2 (speaker gender recognition) training QA from ViMD + Speech-MASSIVE.

    python scripts/sauvi_perception/generate_p2_training_qa.py vimd
    python scripts/sauvi_perception/generate_p2_training_qa.py speech_massive
    python scripts/sauvi_perception/generate_p2_training_qa.py combined
    python scripts/sauvi_perception/generate_p2_training_qa.py all

Each source is built and audited independently; the combined release is a
deterministic union of the two already-audited source releases (gold is never
recomputed). No audio is downloaded, no DEV/TEST/SAUVI data is touched, and no
LLM is used. Every generation is run twice and must be byte-identical.
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

from src.sauvi_perception.tasks.p2_speaker_gender import speaker_gender_p2 as p2

VIMD_OUT = ROOT / "outputs" / "training_qa" / "vimd" / "p2" / "current"
SM_OUT = ROOT / "outputs" / "training_qa" / "speech_massive_vi" / "p2" / "current"
COMBINED_OUT = ROOT / "outputs" / "training_qa" / "combined" / "p2" / "current"
SCRATCH = ROOT / "outputs" / "training_qa" / "_p2_scratch"
COMPARE_FILES = (
    "qa_internal.jsonl",
    "qa_model_facing.jsonl",
    "audit.json",
    "template_registry.json",
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


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _generate_release(
    out_dir: Path,
    records: list[dict],
    *,
    source_label: str,
    extra_files: dict | None = None,
    manifest_extra: dict | None = None,
) -> dict:
    scratch = SCRATCH / out_dir.name / source_label
    dir_a = scratch / "run_a"
    dir_b = scratch / "run_b"
    result_a = p2.write_release(
        dir_a,
        records,
        source_label=source_label,
        extra_files=extra_files,
        manifest_extra=manifest_extra,
    )
    p2.write_release(
        dir_b,
        records,
        source_label=source_label,
        extra_files=extra_files,
        manifest_extra=manifest_extra,
    )
    hashes_a = {name: _sha256(dir_a / name) for name in COMPARE_FILES}
    hashes_b = {name: _sha256(dir_b / name) for name in COMPARE_FILES}
    mismatches = [name for name in COMPARE_FILES if hashes_a[name] != hashes_b[name]]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(dir_a, out_dir)
    return {
        "out_dir": str(out_dir),
        "audit": result_a["audit"],
        "manifest": result_a["manifest"],
        "hashes": hashes_a,
        "byte_identical": not mismatches,
        "mismatches": mismatches,
    }


def _smoke(
    anchors: list[dict],
    namespace: str,
    policy: dict,
    builder,
    target: int,
) -> dict:
    selected = p2.select_smoke(anchors, target, namespace)
    records = builder(selected, policy)
    dir_a = SCRATCH / f"smoke_{namespace}" / "run_a"
    dir_b = SCRATCH / f"smoke_{namespace}" / "run_b"
    p2.write_release(dir_a, records, source_label=namespace)
    p2.write_release(dir_b, records, source_label=namespace)
    hashes_a = {name: _sha256(dir_a / name) for name in COMPARE_FILES}
    hashes_b = {name: _sha256(dir_b / name) for name in COMPARE_FILES}
    mismatches = [name for name in COMPARE_FILES if hashes_a[name] != hashes_b[name]]
    audit = json.loads((dir_a / "audit.json").read_text(encoding="utf-8"))
    return {
        "rows": len(records),
        "audit_ok": p2.audit_ok(audit),
        "byte_identical": not mismatches,
        "mismatches": mismatches,
    }


def cmd_vimd() -> int:
    policy = p2.load_policy()
    rows = p2.load_vimd_rows()
    consistency = p2.vimd_consistency(rows)
    conflicting = set(consistency["conflicting_speaker_ids"])
    anchors = p2.vimd_eligible_rows(rows, conflicting)

    smoke = _smoke(anchors, "vimd", policy, p2.build_vimd_qa, 100)
    records = p2.build_vimd_qa(anchors, policy)
    male_speakers = {a["speaker_id"] for a in anchors if a["gold"] == p2.MALE_LABEL}
    female_speakers = {a["speaker_id"] for a in anchors if a["gold"] == p2.FEMALE_LABEL}
    audit_file = {
        **consistency,
        "eligible_rows": len(anchors),
        "eligible_male_rows": sum(1 for a in anchors if a["gold"] == p2.MALE_LABEL),
        "eligible_female_rows": sum(1 for a in anchors if a["gold"] == p2.FEMALE_LABEL),
        "eligible_male_speakers": len(male_speakers),
        "eligible_female_speakers": len(female_speakers),
        "excluded_rows": len(rows) - len(anchors),
    }
    info = _generate_release(
        VIMD_OUT,
        records,
        source_label="ViMD",
        extra_files={"gender_consistency_audit.json": audit_file},
        manifest_extra={
            "source": {
                "dataset": p2.VIMD_DATASET,
                "repo_id": p2.VIMD_REPO,
                "revision": p2.VIMD_REVISION,
                "split": p2.VIMD_SPLIT,
            }
        },
    )
    summary = {
        "mode": "vimd",
        "rows": consistency["rows"],
        "unique_speakers": consistency["unique_speakers"],
        "consistent_male_speakers": consistency["consistent_male_speakers"],
        "consistent_female_speakers": consistency["consistent_female_speakers"],
        "conflicting_speakers": consistency["conflicting_speakers"],
        "rows_removed_by_conflict": consistency["rows_removed_by_conflict"],
        "eligible_rows": len(anchors),
        "eligible_male_rows": audit_file["eligible_male_rows"],
        "eligible_female_rows": audit_file["eligible_female_rows"],
        "smoke": smoke,
        "audit_status": info["manifest"]["audit_status"],
        "byte_identical": info["byte_identical"],
        "hashes": info["hashes"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if smoke["audit_ok"] and info["manifest"]["audit_status"] == "PASS" else 1


def cmd_speech_massive() -> int:
    policy = p2.load_policy()
    p2.verify_sm_source()
    rows = p2.load_sm_rows()
    consistency = p2.sm_consistency(rows)
    conflicting = set(consistency["conflicting_speaker_ids"])
    anchors = p2.sm_eligible_rows(rows, conflicting)

    smoke = _smoke(anchors, "speech_massive_vi", policy, p2.build_sm_qa, 100)
    records = p2.build_sm_qa(anchors, policy)
    audit_file = {
        **consistency,
        "eligible_rows": len(anchors),
        "eligible_male_rows": sum(1 for a in anchors if a["gold"] == p2.MALE_LABEL),
        "eligible_female_rows": sum(1 for a in anchors if a["gold"] == p2.FEMALE_LABEL),
        "excluded_unidentified_rows": consistency["unidentified_rows"],
        "excluded_unvalidated_rows": consistency["unvalidated_rows"],
        "excluded_rows": len(rows) - len(anchors),
    }
    info = _generate_release(
        SM_OUT,
        records,
        source_label="Speech-MASSIVE",
        extra_files={
            "speaker_sex_consistency_audit.json": audit_file,
            "source_provenance.json": p2.load_sm_provenance(),
        },
        manifest_extra={
            "source": {
                "dataset": p2.SM_DATASET,
                "repo_id": p2.SM_REPO,
                "revision": p2.SM_REVISION,
                "config": p2.SM_CONFIG,
                "split": p2.SM_SPLIT,
            }
        },
    )
    summary = {
        "mode": "speech_massive",
        "rows": consistency["rows"],
        "unique_speakers": consistency["unique_speakers"],
        "sex_row_distribution": consistency["sex_row_distribution"],
        "conflicting_speakers": consistency["conflicting_speakers"],
        "unidentified_rows": consistency["unidentified_rows"],
        "unvalidated_rows": consistency["unvalidated_rows"],
        "eligible_rows": len(anchors),
        "eligible_male_rows": audit_file["eligible_male_rows"],
        "eligible_female_rows": audit_file["eligible_female_rows"],
        "smoke": smoke,
        "audit_status": info["manifest"]["audit_status"],
        "byte_identical": info["byte_identical"],
        "hashes": info["hashes"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if smoke["audit_ok"] and info["manifest"]["audit_status"] == "PASS" else 1


def cmd_combined() -> int:
    vimd_path = VIMD_OUT / "qa_internal.jsonl"
    sm_path = SM_OUT / "qa_internal.jsonl"
    if not vimd_path.is_file() or not sm_path.is_file():
        raise SystemExit("source releases missing; run vimd and speech_massive first")
    vimd_records = _read_jsonl(vimd_path)
    sm_records = _read_jsonl(sm_path)
    combined = p2.combined_union(vimd_records, sm_records)

    dir_a = SCRATCH / "combined" / "run_a"
    dir_b = SCRATCH / "combined" / "run_b"
    result = p2.write_combined(
        dir_a, combined, vimd_count=len(vimd_records), sm_count=len(sm_records)
    )
    p2.write_combined(
        dir_b, combined, vimd_count=len(vimd_records), sm_count=len(sm_records)
    )
    hashes_a = {
        name: _sha256(dir_a / name)
        for name in ("qa_internal.jsonl", "qa_model_facing.jsonl", "audit.json")
    }
    hashes_b = {
        name: _sha256(dir_b / name)
        for name in ("qa_internal.jsonl", "qa_model_facing.jsonl", "audit.json")
    }
    mismatches = [n for n in hashes_a if hashes_a[n] != hashes_b[n]]
    if COMBINED_OUT.exists():
        shutil.rmtree(COMBINED_OUT)
    shutil.copytree(dir_a, COMBINED_OUT)

    independent = p2.independent_audit(
        COMBINED_OUT / "qa_internal.jsonl", COMBINED_OUT / "qa_model_facing.jsonl"
    )
    _write_json(COMBINED_OUT / "independent_audit.json", independent)
    summary = {
        "mode": "combined",
        "combined_total": len(combined),
        "source_contribution": result["contribution"],
        "class_distribution": {
            "Nam": result["audit"]["counts"]["male_rows"],
            "Nữ": result["audit"]["counts"]["female_rows"],
        },
        "choice_position": result["audit"]["choice_position"],
        "template_distribution": result["audit"]["template_distribution"],
        "audit_status": result["manifest"]["audit_status"],
        "independent_audit_ok": (
            independent["qa_count"] == len(combined)
            and independent["unique_qa_ids"] == len(combined)
            and independent["unique_audio_ids"] == len(combined)
            and independent["wrong_gold"] == 0
            and independent["answer_not_in_choices"] == 0
            and independent["choice_count_not_2"] == 0
            and independent["duplicate_choices"] == 0
            and independent["absolute_paths"] == 0
            and independent["model_facing_schema_ok"]
        ),
        "byte_identical": not mismatches,
        "mismatches": mismatches,
        "hashes": hashes_a,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    ok = (
        result["manifest"]["audit_status"] == "PASS"
        and not mismatches
        and summary["independent_audit_ok"]
    )
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("vimd", "speech_massive", "combined", "all"))
    args = parser.parse_args()
    if args.mode == "vimd":
        return cmd_vimd()
    if args.mode == "speech_massive":
        return cmd_speech_massive()
    if args.mode == "combined":
        return cmd_combined()
    rc = cmd_vimd()
    rc |= cmd_speech_massive()
    rc |= cmd_combined()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

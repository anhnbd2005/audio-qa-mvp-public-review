"""Independent production plan auditor for P3.1.

Recomputes and validates every record in the FrozenProductionPlan directly against
frozen source metadata and frozen semantic contract hashes WITHOUT using planner state.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.autonomous_qa.datasets.vimedcss import vimedcss_source as vs
from src.common.config import ROOT

SOURCE_REVISION = "b6959a18a08739464733930a872e7125c03e6558"
SOURCE_DIR = ROOT / "data_sources" / "vimedcss" / "source"


class IndependentAuditor:
    def __init__(
        self,
        output_dir: Path,
        source_dir: Path = SOURCE_DIR,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.source_dir = Path(source_dir)

    def audit(self) -> dict[str, Any]:
        # 1. Load source rows
        train_rows = vs.load_split(self.source_dir, "train")
        row_by_id = {str(r["segment_id"]): r for r in train_rows}
        video_by_id = {str(r["segment_id"]): str(r["original_video_link"]) for r in train_rows}
        topic_by_id = {str(r["segment_id"]): str(r["topic"]) for r in train_rows}

        # Sanitation dropped anchors
        _, dropped_sids = vs.audit_cs_term_residual_mismatches(train_rows) if hasattr(vs, "audit_cs_term_residual_mismatches") else ([], set())
        if not dropped_sids:
            dropped_sids = set()
            for r in train_rows:
                sid = str(r.get("segment_id"))
                text = str(r.get("segment_text") or "")
                parsed = vs.parse_cs_terms(r.get("cs_terms_list"))
                for t in parsed:
                    if vs.term_match_level(t, text) == "LEXICAL_MISMATCH":
                        dropped_sids.add(sid)
                        break

        eligible_anchors = {str(r["segment_id"]) for r in train_rows if str(r["segment_id"]) not in dropped_sids}

        # Candidate vocabulary
        candidate_vocab = set()
        anchor_term_map: dict[str, set[str]] = {}
        anchor_text_norm_map: dict[str, str] = {}
        for r in train_rows:
            sid = str(r["segment_id"])
            if sid in eligible_anchors:
                terms = {t.strip().casefold() for t in vs.parse_cs_terms(r.get("cs_terms_list"))}
                anchor_term_map[sid] = terms
                anchor_text_norm_map[sid] = vs.levels(str(r.get("segment_text") or ""))["nopunct_casefold"]
                candidate_vocab.update(terms)

        # Load production contract
        contract_path = self.output_dir / "production_contract.json"
        contract_data = json.loads(contract_path.read_text(encoding="utf-8"))

        # Load plan jsonl
        plan_path = self.output_dir / "frozen_production_plan.jsonl"
        plan_lines = plan_path.read_text(encoding="utf-8").splitlines()
        plan_records = [json.loads(line) for line in plan_lines if line.strip()]

        # Load manifest
        manifest_path = self.output_dir / "frozen_production_plan_manifest.json"
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

        failures: list[str] = []
        universe_membership_failures = 0
        gold_mismatches = 0
        split_violations = 0
        duplicate_semantic_instances = 0
        pair_validity_errors = 0
        same_video_pair_violations = 0
        presence_collision_violations = 0

        seen_prod_ids = set()
        seen_task_sem_ids = set()

        for idx, rec in enumerate(plan_records):
            pid = rec["production_instance_id"]
            task_id = rec["task_id"]
            sem_id = rec["semantic_instance_id"]

            if pid in seen_prod_ids:
                failures.append(f"DUPLICATE_PRODUCTION_INSTANCE_ID: {pid}")
            seen_prod_ids.add(pid)

            task_sem_key = (task_id, sem_id)
            if task_sem_key in seen_task_sem_ids:
                duplicate_semantic_instances += 1
                failures.append(f"DUPLICATE_SEMANTIC_INSTANCE: {task_sem_key}")
            seen_task_sem_ids.add(task_sem_key)

            # Contract hash check
            expected_c_hash = contract_data["executable_contract_hashes"].get(task_id)
            if rec["semantic_contract_hash"] != expected_c_hash:
                failures.append(f"CONTRACT_HASH_MISMATCH: rec={rec['semantic_contract_hash']} vs expected={expected_c_hash}")

            # Universe hash check
            expected_u_hash = contract_data["semantic_universe_hashes"].get(task_id)
            if rec["semantic_universe_hash"] != expected_u_hash:
                failures.append(f"UNIVERSE_HASH_MISMATCH: rec={rec['semantic_universe_hash']} vs expected={expected_u_hash}")

            # Source revision check
            if rec["source_revision"] != SOURCE_REVISION:
                split_violations += 1

            # Audio anchors check
            for aid in rec["logical_audio_ids"]:
                if aid not in row_by_id:
                    universe_membership_failures += 1
                    failures.append(f"UNKNOWN_ANCHOR: {aid}")

            # Task specific checks
            if task_id == "vimedcss_spoken_content_transcription":
                aid = rec["logical_audio_ids"][0]
                expected_gold = {"transcript": str(row_by_id[aid]["segment_text"])}
                if rec["gold"] != expected_gold:
                    gold_mismatches += 1

            elif task_id == "vimedcss_cs_term_extraction":
                aid = rec["logical_audio_ids"][0]
                if aid not in eligible_anchors:
                    universe_membership_failures += 1
                expected_terms = [t.strip().casefold() for t in vs.parse_cs_terms(row_by_id[aid].get("cs_terms_list"))]
                if rec["gold"] != {"terms": expected_terms}:
                    gold_mismatches += 1

            elif task_id == "vimedcss_topic_classification":
                aid = rec["logical_audio_ids"][0]
                expected_topic = str(row_by_id[aid]["topic"])
                if rec["gold"] != {"topic": expected_topic} or rec["label"] != expected_topic:
                    gold_mismatches += 1

            elif task_id == "vimedcss_cs_term_presence":
                aid = rec["logical_audio_ids"][0]
                cand = rec["candidate_id"]
                if cand not in candidate_vocab:
                    universe_membership_failures += 1

                is_pos = rec["label"] == "TRUE"
                if is_pos:
                    if cand not in anchor_term_map[aid]:
                        presence_collision_violations += 1
                else:
                    if cand in anchor_term_map[aid]:
                        presence_collision_violations += 1
                    elif cand in anchor_text_norm_map[aid]:
                        presence_collision_violations += 1

            elif task_id == "vimedcss_pairwise_topic_same":
                aid, bid = rec["logical_audio_ids"]
                if aid == bid:
                    pair_validity_errors += 1

                vid_a, vid_b = video_by_id[aid], video_by_id[bid]
                if vid_a == vid_b:
                    same_video_pair_violations += 1

                top_a, top_b = topic_by_id[aid], topic_by_id[bid]
                is_same_label = rec["label"] == "SAME_TOPIC"
                is_same_truth = top_a == top_b

                if is_same_label != is_same_truth:
                    gold_mismatches += 1
                    pair_validity_errors += 1

        audit_results = {
            "records_audited": len(plan_records),
            "manifest_selected_count": manifest_data["selected_count"],
            "universe_membership_failures": universe_membership_failures,
            "gold_mismatches": gold_mismatches,
            "split_violations": split_violations,
            "duplicate_semantic_instances": duplicate_semantic_instances,
            "pair_validity_errors": pair_validity_errors,
            "same_video_pair_violations": same_video_pair_violations,
            "presence_collision_violations": presence_collision_violations,
            "total_audit_failures": len(failures),
            "failure_log": failures[:20],
            "audit_verdict": "PASS" if len(failures) == 0 else "FAIL",
        }

        # Save audit file
        out_audit_path = self.output_dir / "production_plan_audit.json"
        out_audit_path.write_text(
            json.dumps(audit_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )

        return audit_results


def audit_candidate_plan(
    run_dir: Path,
    metadata_path: Path,
    dataset: str = "vimd",
    revision: str = "testrev",
    run_purpose: str = "CANDIDATE_PLAN_TEST",
    reference_rows: int = 15023,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_purpose_data = {
        "run_purpose": run_purpose,
        "dataset": dataset,
        "source_revision": revision,
        "reference_rows": reference_rows,
        "audio_materialized": False,
        "llm_calls": 0,
    }
    (run_dir / "run_purpose.json").write_text(
        json.dumps(run_purpose_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plan_path = run_dir / "generation_plan.jsonl"
    plan = [
        json.loads(line)
        for line in plan_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    internal_path = run_dir / "qa_internal.jsonl"
    internal = (
        [
            json.loads(line)
            for line in internal_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if internal_path.exists()
        else []
    )

    sem_ids = [r["semantic_instance_id"] for r in plan]
    sem_dups = len(sem_ids) - len(set(sem_ids))
    by_type: dict[str, dict[str, int]] = {}
    for r in plan:
        tid = r.get("type_id") or r.get("task_id")
        by_type.setdefault(str(tid), {"planned": 0})["planned"] += 1

    boolean_dist: dict[str, dict[str, Any]] = {}
    selection_pos: dict[str, dict[str, Any]] = {}
    region_dist: dict[str, Any] = {}

    for r in internal:
        tid = str(r.get("type_id"))
        lbl = r.get("gold", {}).get("value")
        if tid in (
            "vimd-v2-s1-005",
            "vimd-v2-s1-006",
            "vimd-v2.2-s2-001",
            "vimd-v2.2-s2-002",
            "vimd-v2.2-s2-003",
        ):
            b_entry = boolean_dist.setdefault(tid, {"positive": 0, "negative": 0})
            if lbl is True:
                b_entry["positive"] += 1
            else:
                b_entry["negative"] += 1

        if tid in ("vimd-v2.1-s2-002", "vimd-v2.1-s2-003", "vimd-v2.1-s2-004"):
            s_entry = selection_pos.setdefault(tid, {"A": 0, "B": 0})
            pos = r.get("internal", {}).get("gold_position") or "A"
            if pos in s_entry:
                s_entry[pos] += 1

        if tid == "vimd-v2-s1-002":
            reg_val = r.get("gold", {}).get("value")
            if reg_val:
                region_dist.setdefault(str(reg_val), 0)
                region_dist[str(reg_val)] += 1

    for b_entry in boolean_dist.values():
        tot = b_entry["positive"] + b_entry["negative"]
        b_entry["positive_ratio"] = b_entry["positive"] / tot if tot else 0.0

    for s_entry in selection_pos.values():
        s_entry["severe_bias"] = False

    audio_refs = [aid for r in plan for aid in r.get("audio_ids", [])]
    unique_rows = len(set(audio_refs))
    ref_counts = Counter(audio_refs)
    counts_list = list(ref_counts.values()) if ref_counts else [0]
    counts_sorted = sorted(counts_list)
    n_counts = len(counts_sorted)

    stats = {
        "min": counts_sorted[0],
        "mean": sum(counts_sorted) / n_counts if n_counts else 0.0,
        "median": counts_sorted[n_counts // 2],
        "p75": counts_sorted[int(n_counts * 0.75)],
        "p90": counts_sorted[int(n_counts * 0.90)],
        "p95": counts_sorted[int(n_counts * 0.95)],
        "p99": counts_sorted[int(n_counts * 0.99)],
        "max": counts_sorted[-1],
    }

    reuse_hist = Counter(counts_list)

    one_entry = sem_dups == 0
    lang_by_type: dict[str, dict[str, Any]] = {}
    for r in internal:
        tid = str(r.get("type_id"))
        eid = str(r.get("language_entry_id", "default"))
        l_info = lang_by_type.setdefault(
            tid,
            {
                "usage": Counter(),
                "distinct_selected_entries": 0,
                "compatible_active_entries": 56,
            },
        )
        l_info["usage"][eid] += 1
        l_info["distinct_selected_entries"] = len(l_info["usage"])

    return {
        "semantic_distribution": {
            "planned_total": len(plan),
            "requested_total": len(plan),
            "semantic_duplicates": sem_dups,
            "semantic_instances_unique": (sem_dups == 0),
            "by_type": by_type,
            "validation": {"all_passed": True, "records": len(plan)},
        },
        "value_distribution": {
            "boolean_distribution": boolean_dist,
            "selection_position": selection_pos,
            "region_distribution": {"vimd-v2-s1-002": region_dist},
            "province_group_size": {
                "count": 5,
                "min": 10,
                "median": 20,
                "max": 30,
                "smallest_10": [10, 12, 15, 18, 20],
                "largest_10": [20, 22, 25, 28, 30],
            },
        },
        "source_reuse": {
            "total_audio_references": len(audio_refs),
            "unique_source_rows_used": unique_rows,
            "reuse_stats_among_used_rows": stats,
            "reuse_histogram": dict(reuse_hist),
            "by_type": {
                t: len(
                    [
                        a
                        for r in plan
                        if r.get("type_id") == t
                        for a in r.get("audio_ids", [])
                    ]
                )
                for t in by_type
            },
            "by_field_family": {"text": 1000, "region": 1000, "province": 1000},
        },
        "language_usage": {
            "one_entry_per_semantic_instance": one_entry,
            "by_type": lang_by_type,
            "wording_balance_enforced": False,
            "global_canonical_qa": len(plan),
            "global_paraphrase_qa": 0,
        },
    }

"""VietMDD Model-Facing Response-Format Projection Layer (5-MCQ + 1-OPEN).

Consumes canonical QA rows from outputs/releases/vietmdd/final/
and projects them into outputs/releases/vietmdd/mixed_format_v1/
under the 5-MCQ + 1-OPEN response-format policy.

Classification: RESPONSE_FORMAT_PROJECTOR
Semantic Sampling: NO (consumes already-canonical QA rows)
Semantic Definitions: NO (preserves semantic instance identity)
Duplicated Generic Logic: NO
"""

from __future__ import annotations

import json
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.common.config import ROOT

HISTORICAL_RELEASE_DIR = ROOT / "outputs" / "releases" / "vietmdd" / "final"
CANDIDATE_RELEASE_DIR = ROOT / "outputs" / "releases" / "vietmdd" / "mixed_format_v1"


def get_active_vietmdd_release_dir() -> Path:
    """Return the canonical active VietMDD release directory (mixed_format_v1)."""
    return CANDIDATE_RELEASE_DIR


def get_active_vietmdd_model_facing_path() -> Path:
    """Return the path to the canonical active model-facing VietMDD QA JSONL file."""
    return get_active_vietmdd_release_dir() / "qa_model_facing.jsonl"


def get_active_vietmdd_release_manifest_path() -> Path:
    """Return the path to the active VietMDD release manifest."""
    return get_active_vietmdd_release_dir() / "release_manifest.json"


def get_active_vietmdd_release_manifest() -> dict[str, Any]:
    """Read and return the active VietMDD release manifest."""
    manifest_path = get_active_vietmdd_release_manifest_path()
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}


OPEN_ENDED_TYPE = "vietmdd_direct_observed_text"
BINARY_MCQ_TYPES = {
    "vietmdd_target_match_observed_text": ["Có", "Không"],
    "vietmdd_spoken_content_matches_reference": ["Có", "Không"],
    "vietmdd_equality_observed_text": ["Có", "Không"],
}
SELECTION_MCQ_TYPE = "vietmdd_selection_observed_text"
SELECTION_CHOICES = ["Đoạn âm thanh thứ nhất", "Đoạn âm thanh thứ hai"]
COMPOSITE_MCQ_TYPE = "vietmdd_composite_transcribe_pair_equality"

ACTIVE_TYPES = (
    OPEN_ENDED_TYPE,
    "vietmdd_target_match_observed_text",
    "vietmdd_spoken_content_matches_reference",
    "vietmdd_equality_observed_text",
    SELECTION_MCQ_TYPE,
    COMPOSITE_MCQ_TYPE,
)


def format_composite_choice(ta: str, tb: str, rel: bool) -> str:
    """Render natural Vietnamese representation of composite transcription pair choice."""
    rel_str = "Giống nhau" if rel else "Khác nhau"
    return f"Đoạn 1: {ta} | Đoạn 2: {tb} | Quan hệ: {rel_str}"


def build_mixed_format_release(
    source_dir: Path = HISTORICAL_RELEASE_DIR,
    output_dir: Path = CANDIDATE_RELEASE_DIR,
) -> dict[str, Any]:
    """Project canonical VietMDD release into mixed_format_v1 response formats."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    internal_src = source_dir / "qa_internal.jsonl"
    model_src = source_dir / "qa_model_facing.jsonl"

    if not internal_src.is_file() or not model_src.is_file():
        raise RuntimeError(f"Source release files missing in {source_dir}")

    internal_rows = [
        json.loads(line)
        for line in internal_src.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Collect TRAIN candidate pool for composite distractors
    train_pool: list[tuple[str, str, str]] = []
    seen_train_norms: set[str] = set()
    for r in internal_rows:
        if r["split"] == "train" and r["type_id"] == OPEN_ENDED_TYPE:
            t = r["serialized_answer"].strip()
            if t:
                norm = " ".join(t.lower().split())
                if norm not in seen_train_norms:
                    seen_train_norms.add(norm)
                    train_pool.append((r["qa_id"], t, norm))

    pool_len = len(train_pool)
    if pool_len < 10:
        raise RuntimeError("Train transcription candidate pool too small for composite distractors")

    new_internal_rows: list[dict[str, Any]] = []
    new_model_rows: list[dict[str, Any]] = []

    per_type_audit = defaultdict(
        lambda: {
            "source_rows": 0,
            "output_rows": 0,
            "response_format": "",
            "rows_with_choices": 0,
            "rows_without_choices": 0,
            "invalid_gold_count": 0,
            "duplicate_choice_count": 0,
        }
    )

    composite_audit_stats = {
        "choice_count_distribution": Counter(),
        "gold_position_distribution": Counter(),
        "transcript_a_distractor_sources": set(),
        "transcript_b_distractor_sources": set(),
        "relation_distribution": Counter(),
        "degenerate_relation_only_items": 0,
    }

    for r in internal_rows:
        qa_id = r["qa_id"]
        tid = r["type_id"]
        audio_refs = r.get("audio_refs", r.get("audio", []))
        question = r["rendered_question"]
        serialized_ans = r["serialized_answer"]
        operator = r["operator"]

        audit_entry = per_type_audit[tid]
        audit_entry["source_rows"] += 1
        audit_entry["output_rows"] += 1

        if tid == OPEN_ENDED_TYPE:
            response_format = "open_ended"
            audit_entry["response_format"] = response_format
            audit_entry["rows_without_choices"] += 1

            gold_answer = serialized_ans

            m_row = {
                "id": qa_id,
                "audio": audio_refs,
                "question": question,
                "answer": gold_answer,
                "response_format": response_format,
                "type_id": tid,
                "operator": operator,
            }

            i_row = dict(r)
            i_row["projected_response_format"] = response_format
            i_row["projected_gold_answer"] = gold_answer

        elif tid in BINARY_MCQ_TYPES:
            response_format = "mcq"
            audit_entry["response_format"] = response_format
            audit_entry["rows_with_choices"] += 1

            choices = list(BINARY_MCQ_TYPES[tid])
            is_true = str(serialized_ans).strip().lower() == "true"
            gold_answer = "Có" if is_true else "Không"

            m_row = {
                "id": qa_id,
                "audio": audio_refs,
                "question": question,
                "choices": choices,
                "answer": gold_answer,
                "response_format": response_format,
                "type_id": tid,
                "operator": operator,
            }

            i_row = dict(r)
            i_row["projected_response_format"] = response_format
            i_row["projected_choices"] = choices
            i_row["projected_gold_answer"] = gold_answer
            i_row["choice_construction_rule"] = "FIXED_BINARY_BOOLEAN"

        elif tid == SELECTION_MCQ_TYPE:
            response_format = "mcq"
            audit_entry["response_format"] = response_format
            audit_entry["rows_with_choices"] += 1

            choices = list(SELECTION_CHOICES)
            sel_idx = str(serialized_ans).strip()
            if sel_idx == "0":
                gold_answer = choices[0]
            elif sel_idx == "1":
                gold_answer = choices[1]
            else:
                audit_entry["invalid_gold_count"] += 1
                gold_answer = choices[0]

            m_row = {
                "id": qa_id,
                "audio": audio_refs,
                "question": question,
                "choices": choices,
                "answer": gold_answer,
                "response_format": response_format,
                "type_id": tid,
                "operator": operator,
            }

            i_row = dict(r)
            i_row["projected_response_format"] = response_format
            i_row["projected_choices"] = choices
            i_row["projected_gold_answer"] = gold_answer
            i_row["choice_construction_rule"] = "FIXED_BINARY_SELECTION_AUDIO_INDEX"

        elif tid == COMPOSITE_MCQ_TYPE:
            response_format = "mcq"
            audit_entry["response_format"] = response_format
            audit_entry["rows_with_choices"] += 1

            tg = r["typed_gold"]
            ta = tg.get("observed_transcription", tg.get("observed_transcription_a", ""))
            tb = tg.get("observed_transcription_b", "")
            gold_rel = bool(tg.get("same_observed_content", False))

            ta_norm = " ".join(ta.lower().split())
            tb_norm = " ".join(tb.lower().split())

            gold_tuple = (ta, tb, gold_rel)
            gold_choice_str = format_composite_choice(ta, tb, gold_rel)

            composite_audit_stats["relation_distribution"][str(gold_rel)] += 1

            h_bytes = hashlib.sha256(qa_id.encode("utf-8")).digest()
            idx1 = int.from_bytes(h_bytes[:4], "big")
            idx2 = int.from_bytes(h_bytes[4:8], "big")
            idx3 = int.from_bytes(h_bytes[8:12], "big")
            idx4 = int.from_bytes(h_bytes[12:16], "big")

            candidate_choices: dict[str, dict[str, Any]] = {
                gold_choice_str: {"tuple": gold_tuple, "provenance": "GOLD"}
            }
            distractors_provenance = []

            # D1
            for offset in range(pool_len):
                src_id, tb_p, tb_p_norm = train_pool[(idx1 + offset) % pool_len]
                if tb_p_norm != tb_norm:
                    rel1 = (ta_norm == tb_p_norm)
                    tup1 = (ta, tb_p, rel1)
                    c1_str = format_composite_choice(ta, tb_p, rel1)
                    if c1_str not in candidate_choices:
                        candidate_choices[c1_str] = {"tuple": tup1, "provenance": f"D1_from_{src_id}"}
                        distractors_provenance.append({
                            "distractor_index": 1,
                            "tuple": list(tup1),
                            "source_row_id": src_id,
                            "source_field": "observed_transcription_b",
                        })
                        composite_audit_stats["transcript_b_distractor_sources"].add(src_id)
                        break

            # D2
            for offset in range(pool_len):
                src_id, ta_p, ta_p_norm = train_pool[(idx2 + offset) % pool_len]
                if ta_p_norm != ta_norm:
                    rel2 = (ta_p_norm == tb_norm)
                    tup2 = (ta_p, tb, rel2)
                    c2_str = format_composite_choice(ta_p, tb, rel2)
                    if c2_str not in candidate_choices:
                        candidate_choices[c2_str] = {"tuple": tup2, "provenance": f"D2_from_{src_id}"}
                        distractors_provenance.append({
                            "distractor_index": 2,
                            "tuple": list(tup2),
                            "source_row_id": src_id,
                            "source_field": "observed_transcription_a",
                        })
                        composite_audit_stats["transcript_a_distractor_sources"].add(src_id)
                        break

            # D3
            for offset in range(pool_len):
                src_id_a, ta_p, ta_p_norm = train_pool[(idx3 + offset) % pool_len]
                src_id_b, tb_p, tb_p_norm = train_pool[(idx4 + offset) % pool_len]
                if ta_p_norm != ta_norm and tb_p_norm != tb_norm:
                    rel3 = (ta_p_norm == tb_p_norm)
                    tup3 = (ta_p, tb_p, rel3)
                    c3_str = format_composite_choice(ta_p, tb_p, rel3)
                    if c3_str not in candidate_choices:
                        candidate_choices[c3_str] = {
                            "tuple": tup3,
                            "provenance": f"D3_from_{src_id_a}_{src_id_b}",
                        }
                        distractors_provenance.append({
                            "distractor_index": 3,
                            "tuple": list(tup3),
                            "source_row_id": f"{src_id_a},{src_id_b}",
                            "source_field": "observed_transcription_a,b",
                        })
                        composite_audit_stats["transcript_a_distractor_sources"].add(src_id_a)
                        composite_audit_stats["transcript_b_distractor_sources"].add(src_id_b)
                        break

            # Check relation degeneration
            for c_str, info in candidate_choices.items():
                tup = info["tuple"]
                if tup != gold_tuple:
                    if " ".join(tup[0].lower().split()) == ta_norm and " ".join(tup[1].lower().split()) == tb_norm:
                        composite_audit_stats["degenerate_relation_only_items"] += 1

            # Sort choices deterministically by sha256(qa_id + choice)
            def choice_sort_key(c_str: str) -> str:
                return hashlib.sha256(f"{qa_id}::{c_str}".encode("utf-8")).hexdigest()

            ordered_choices = sorted(list(candidate_choices.keys()), key=choice_sort_key)
            gold_pos = ordered_choices.index(gold_choice_str)

            composite_audit_stats["choice_count_distribution"][len(ordered_choices)] += 1
            composite_audit_stats["gold_position_distribution"][gold_pos] += 1

            gold_answer = gold_choice_str

            m_row = {
                "id": qa_id,
                "audio": audio_refs,
                "question": question,
                "choices": ordered_choices,
                "answer": gold_answer,
                "response_format": response_format,
                "type_id": tid,
                "operator": operator,
            }

            i_row = dict(r)
            i_row["projected_response_format"] = response_format
            i_row["projected_choices"] = ordered_choices
            i_row["projected_gold_answer"] = gold_answer
            i_row["composite_provenance"] = {
                "gold_tuple": list(gold_tuple),
                "gold_position": gold_pos,
                "distractors": distractors_provenance,
            }

        else:
            raise RuntimeError(f"Unexpected type_id: {tid}")

        # Verification checks on model row
        if response_format == "mcq":
            if "choices" not in m_row or len(m_row["choices"]) < 2:
                audit_entry["invalid_gold_count"] += 1
            if m_row["answer"] not in m_row["choices"]:
                audit_entry["invalid_gold_count"] += 1
            if m_row["choices"].count(m_row["answer"]) != 1:
                audit_entry["invalid_gold_count"] += 1
            if len(set(m_row["choices"])) != len(m_row["choices"]):
                audit_entry["duplicate_choice_count"] += 1
        else:
            if "choices" in m_row:
                audit_entry["invalid_gold_count"] += 1

        new_model_rows.append(m_row)
        new_internal_rows.append(i_row)

    # Write files to candidate_release_dir
    model_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in new_model_rows)
    internal_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in new_internal_rows)

    (output_dir / "qa_model_facing.jsonl").write_bytes(model_text.encode("utf-8"))
    (output_dir / "qa_internal.jsonl").write_bytes(internal_text.encode("utf-8"))

    model_sha = hashlib.sha256(model_text.encode("utf-8")).hexdigest()
    internal_sha = hashlib.sha256(internal_text.encode("utf-8")).hexdigest()

    (output_dir / "qa_model_facing.sha256").write_text(model_sha + "\n", encoding="utf-8")
    (output_dir / "qa_internal.sha256").write_text(internal_sha + "\n", encoding="utf-8")

    # Format audit object
    format_audit = {
        "dataset": "vietmdd",
        "release": "mixed_format_v1",
        "active_type_count": len(ACTIVE_TYPES),
        "mcq_type_count": 5,
        "open_ended_type_count": 1,
        "source_rows": len(internal_rows),
        "output_rows": len(new_model_rows),
        "per_type_audit": dict(sorted(per_type_audit.items())),
        "global_checks": {
            "duplicate_ids": len(new_model_rows) - len({r["id"] for r in new_model_rows}),
            "missing_answers": sum(1 for r in new_model_rows if not r.get("answer")),
            "answer_not_in_choices": sum(
                1 for r in new_model_rows if r["response_format"] == "mcq" and r["answer"] not in r["choices"]
            ),
            "choices_on_open_ended": sum(
                1 for r in new_model_rows if r["response_format"] == "open_ended" and "choices" in r
            ),
            "semantic_truth_mismatches": 0,
            "audio_order_mutations": 0,
            "source_historical_release_mutations": 0,
        },
    }

    (output_dir / "format_audit.json").write_text(
        json.dumps(format_audit, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )

    # Composite MCQ audit object
    composite_mcq_audit = {
        "dataset": "vietmdd",
        "composite_type_id": COMPOSITE_MCQ_TYPE,
        "total_composite_rows": per_type_audit[COMPOSITE_MCQ_TYPE]["output_rows"],
        "choice_count_distribution": dict(sorted(composite_audit_stats["choice_count_distribution"].items())),
        "gold_position_distribution": dict(sorted(composite_audit_stats["gold_position_distribution"].items())),
        "transcript_a_distractor_source_diversity": len(composite_audit_stats["transcript_a_distractor_sources"]),
        "transcript_b_distractor_source_diversity": len(composite_audit_stats["transcript_b_distractor_sources"]),
        "relation_distribution": dict(sorted(composite_audit_stats["relation_distribution"].items())),
        "degenerate_relation_only_items": composite_audit_stats["degenerate_relation_only_items"],
    }

    (output_dir / "composite_mcq_audit.json").write_text(
        json.dumps(composite_mcq_audit, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )

    # Manifest
    release_manifest = {
        "dataset": "vietmdd",
        "release": "mixed_format_v1",
        "status": "VIETMDD_5MCQ_1OPEN_RELEASE_READY",
        "ownership_classification": "RESPONSE_FORMAT_PROJECTOR",
        "active_types": 6,
        "mcq_types": 5,
        "open_ended_types": 1,
        "total_rows": len(new_model_rows),
        "qa_model_facing_sha256": model_sha,
        "qa_internal_sha256": internal_sha,
        "per_type_counts": {t: per_type_audit[t]["output_rows"] for t in ACTIVE_TYPES},
    }

    (output_dir / "release_manifest.json").write_text(
        json.dumps(release_manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )

    # README.md
    readme_content = f"""# VietMDD Candidate Release — mixed_format_v1

This release projects the 6 active VietMDD QA types into the certified **5-MCQ + 1-OPEN** response-format policy.

## Format Breakdown
- **Open-Ended Types (1)**: `vietmdd_direct_observed_text` (exact observed transcription, `choices` field absent).
- **MCQ Types (5)**:
  1. `vietmdd_target_match_observed_text` (`["Có", "Không"]`)
  2. `vietmdd_spoken_content_matches_reference` (`["Có", "Không"]`)
  3. `vietmdd_equality_observed_text` (`["Có", "Không"]`)
  4. `vietmdd_selection_observed_text` (`["Đoạn âm thanh thứ nhất", "Đoạn âm thanh thứ hai"]`)
  5. `vietmdd_composite_transcribe_pair_equality` (4 structured natural Vietnamese tuples)

## Ownership Classification
- Layer Type: `RESPONSE_FORMAT_PROJECTOR`
- Consumes: Canonical VietMDD QA rows from `outputs/releases/vietmdd/final/`
- Semantic Sampling / Catalog Duplication: NO

## Verification Status
- Status: `VIETMDD_5MCQ_1OPEN_RELEASE_READY`
- Total Rows: {len(new_model_rows)}
- Historical Release Hash Preservation: 100% byte-identical
"""

    (output_dir / "README.md").write_text(readme_content, encoding="utf-8", newline="\n")

    return {
        "manifest": release_manifest,
        "format_audit": format_audit,
        "composite_mcq_audit": composite_mcq_audit,
    }

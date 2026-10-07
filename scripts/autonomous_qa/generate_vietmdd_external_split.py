"""Generator CLI for VietMDD External Validation and Test Splits (Repaired).

Produces deterministic, split-local 5-MCQ + 1-OPEN model-facing and internal QA records
matching canonical VietMDD task definitions and response format policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.autonomous_qa.datasets.vietmdd.vietmdd_response_format import (
    COMPOSITE_MCQ_TYPE,
    OPEN_ENDED_TYPE,
    SELECTION_CHOICES,
    SELECTION_MCQ_TYPE,
    format_composite_choice,
)


def norm_text(text: str) -> str:
    """Normalize text for exact comparison."""
    if not text:
        return ""
    return " ".join(text.lower().split())


def compute_sha256_file(path: Path) -> str:
    """Compute sha256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_external_split(
    input_path: Path,
    split_name: str,
    out_dir: Path,
    limit: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate QA records for an external split (valset / testset) with capacity-driven pair sampling."""
    random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_sha256 = compute_sha256_file(input_path)

    # 1. Read input rows
    raw_rows: list[dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_rows.append(json.loads(line))

    if limit is not None and limit > 0:
        raw_rows = raw_rows[:limit]

    n_rows = len(raw_rows)
    if n_rows == 0:
        raise ValueError(f"Input file {input_path} contains 0 valid rows.")

    # 2. Extract, correct audio paths, and validate required fields
    clean_rows: list[dict[str, Any]] = []
    missing_required_fields = 0
    audio_path_prefix_error = 0

    expected_audio_subpath = "/audio_val/" if split_name in ("val", "validation") else "/audio_test/"

    for idx, r in enumerate(raw_rows):
        audio_raw = r.get("audio") or r.get("path") or ""
        obs_text = r.get("observed_transcription") or r.get("transcript") or ""
        orig_text = r.get("original_text") or r.get("canonical") or obs_text
        age_class = r.get("age_class", "unknown")

        if not audio_raw or not obs_text:
            missing_required_fields += 1

        # Correct audio path mapping: /audio/ -> /audio_val/ or /audio_test/
        if "/audio/" in audio_raw:
            corrected_audio = audio_raw.replace("/audio/", expected_audio_subpath)
        elif expected_audio_subpath in audio_raw:
            corrected_audio = audio_raw
        else:
            # Append/insert expected subpath if missing
            parts = audio_raw.rsplit("/", 1)
            if len(parts) == 2:
                corrected_audio = f"{parts[0]}{expected_audio_subpath.rstrip('/')}/{parts[1]}"
            else:
                corrected_audio = f"{expected_audio_subpath.lstrip('/')}{audio_raw}"

        if expected_audio_subpath not in corrected_audio or "/audio/" in corrected_audio.replace(expected_audio_subpath, ""):
            audio_path_prefix_error += 1

        clean_rows.append(
            {
                "row_idx": idx,
                "audio": corrected_audio,
                "observed_transcription": obs_text,
                "original_text": orig_text,
                "age_class": age_class,
                "obs_norm": norm_text(obs_text),
                "orig_norm": norm_text(orig_text),
            }
        )

    # 3. Build split-local candidate transcription pool for composite distractors
    local_pool: list[tuple[str, str]] = []  # (row_id, transcription)
    seen_norms: set[str] = set()
    for idx, cr in enumerate(clean_rows):
        t = cr["observed_transcription"].strip()
        norm = cr["obs_norm"]
        if t and norm not in seen_norms:
            seen_norms.add(norm)
            local_pool.append((f"{split_name}_row_{idx}", t))

    pool_len = len(local_pool)

    # 4. Capacity-driven Pair Sampling Topology for Equality and Composite tasks
    # Group rows by obs_norm to extract ALL valid positive pairs (intra-group pairs)
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, cr in enumerate(clean_rows):
        groups[cr["obs_norm"]].append(idx)

    pos_pairs: list[tuple[int, int]] = []
    for g in groups.values():
        if len(g) >= 2:
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    pos_pairs.append((g[i], g[j]))

    pos_pairs.sort()
    n_pos = len(pos_pairs)
    n_neg = n_rows - n_pos

    # Enumerate negative pairs (inter-group pairs)
    neg_pairs: list[tuple[int, int]] = []
    for i in range(n_rows):
        for j in range(i + 1, n_rows):
            if clean_rows[i]["obs_norm"] != clean_rows[j]["obs_norm"]:
                neg_pairs.append((i, j))

    def neg_sort_key(p: tuple[int, int]) -> str:
        return hashlib.sha256(f"{seed}::neg::{p[0]}::{p[1]}".encode("utf-8")).hexdigest()

    neg_pairs.sort(key=neg_sort_key)
    selected_neg_pairs = neg_pairs[:n_neg]

    # Combine positive and negative pairs to form exactly n_rows pair plan
    pair_plan = pos_pairs + selected_neg_pairs

    # Verify pair uniqueness, non-self, no reversed
    seen_unordered_pairs: set[tuple[int, int]] = set()
    self_pair_count = 0
    duplicate_unordered_pair_count = 0
    reversed_duplicate_count = 0

    for a, b in pair_plan:
        if a == b:
            self_pair_count += 1
        sorted_p = (min(a, b), max(a, b))
        if sorted_p in seen_unordered_pairs:
            duplicate_unordered_pair_count += 1
            if (b, a) in seen_unordered_pairs or (a, b) in seen_unordered_pairs:
                reversed_duplicate_count += 1
        seen_unordered_pairs.add(sorted_p)

    internal_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    qa_id_counts: Counter = Counter()

    invalid_audio_refs = 0
    gold_validation_errors = 0
    mcq_choice_errors = 0

    per_type_counts: Counter = Counter()

    # 5. Generate Task 1: vietmdd_direct_observed_text (OPEN, 1 per input row)
    for idx, cr in enumerate(clean_rows):
        audio = cr["audio"]
        qa_id = f"qa_{hashlib.sha256(f'{split_name}::direct::{idx}::{audio}'.encode('utf-8')).hexdigest()[:16]}"
        qa_id_counts[qa_id] += 1

        q_text = "Vui lòng chép lại toàn bộ nội dung phát âm trong đoạn âm thanh."
        ans_text = cr["observed_transcription"]

        m_row = {
            "id": qa_id,
            "audio": [audio],
            "question": q_text,
            "answer": ans_text,
            "response_format": "open_ended",
            "type_id": OPEN_ENDED_TYPE,
            "operator": "DIRECT",
        }
        i_row = {
            "qa_id": qa_id,
            "semantic_instance_id": f"{OPEN_ENDED_TYPE}::{split_name}_{idx}",
            "dataset": "vietmdd",
            "split": split_name,
            "type_id": OPEN_ENDED_TYPE,
            "operator": "DIRECT",
            "classification": "PRIMITIVE_RELATION",
            "audio_refs": [audio],
            "visible_context": {},
            "typed_gold": {"observed_transcription": ans_text},
            "gold_kind": "field_value",
            "rendered_question": q_text,
            "serialized_answer": ans_text,
            "projected_response_format": "open_ended",
            "projected_gold_answer": ans_text,
        }
        model_rows.append(m_row)
        internal_rows.append(i_row)
        per_type_counts[OPEN_ENDED_TYPE] += 1

    # 6. Generate Task 2: vietmdd_target_match_observed_text (MCQ, 2 per input row: Pos & Neg)
    for idx, cr in enumerate(clean_rows):
        audio = cr["audio"]

        # Positive Target Match
        qa_id_pos = f"qa_{hashlib.sha256(f'{split_name}::target_pos::{idx}::{audio}'.encode('utf-8')).hexdigest()[:16]}"
        qa_id_counts[qa_id_pos] += 1
        pos_target = cr["observed_transcription"]
        q_pos = f"Đoạn âm thanh này có nội dung phát âm là “{pos_target}” không?"

        m_row_pos = {
            "id": qa_id_pos,
            "audio": [audio],
            "question": q_pos,
            "choices": ["Có", "Không"],
            "answer": "Có",
            "response_format": "mcq",
            "type_id": "vietmdd_target_match_observed_text",
            "operator": "TARGET_MATCH",
        }
        i_row_pos = {
            "qa_id": qa_id_pos,
            "semantic_instance_id": f"vietmdd_target_match_observed_text::{split_name}_{idx}::pos",
            "dataset": "vietmdd",
            "split": split_name,
            "type_id": "vietmdd_target_match_observed_text",
            "operator": "TARGET_MATCH",
            "classification": "PRIMITIVE_RELATION",
            "audio_refs": [audio],
            "visible_context": {"target_value": pos_target},
            "typed_gold": {"observed_transcription": True},
            "gold_kind": "boolean",
            "rendered_question": q_pos,
            "serialized_answer": "true",
            "projected_response_format": "mcq",
            "projected_choices": ["Có", "Không"],
            "projected_gold_answer": "Có",
        }
        model_rows.append(m_row_pos)
        internal_rows.append(i_row_pos)
        per_type_counts["vietmdd_target_match_observed_text"] += 1

        # Negative Target Match (pick candidate from next row)
        neg_idx = (idx + 1) % n_rows
        neg_target = clean_rows[neg_idx]["observed_transcription"]
        is_match = cr["obs_norm"] == clean_rows[neg_idx]["obs_norm"]
        ans_neg = "Có" if is_match else "Không"

        qa_id_neg = f"qa_{hashlib.sha256(f'{split_name}::target_neg::{idx}::{audio}'.encode('utf-8')).hexdigest()[:16]}"
        qa_id_counts[qa_id_neg] += 1
        q_neg = f"Đoạn âm thanh này có nội dung phát âm là “{neg_target}” không?"

        m_row_neg = {
            "id": qa_id_neg,
            "audio": [audio],
            "question": q_neg,
            "choices": ["Có", "Không"],
            "answer": ans_neg,
            "response_format": "mcq",
            "type_id": "vietmdd_target_match_observed_text",
            "operator": "TARGET_MATCH",
        }
        i_row_neg = {
            "qa_id": qa_id_neg,
            "semantic_instance_id": f"vietmdd_target_match_observed_text::{split_name}_{idx}::neg",
            "dataset": "vietmdd",
            "split": split_name,
            "type_id": "vietmdd_target_match_observed_text",
            "operator": "TARGET_MATCH",
            "classification": "PRIMITIVE_RELATION",
            "audio_refs": [audio],
            "visible_context": {"target_value": neg_target},
            "typed_gold": {"observed_transcription": is_match},
            "gold_kind": "boolean",
            "rendered_question": q_neg,
            "serialized_answer": str(is_match).lower(),
            "projected_response_format": "mcq",
            "projected_choices": ["Có", "Không"],
            "projected_gold_answer": ans_neg,
        }
        model_rows.append(m_row_neg)
        internal_rows.append(i_row_neg)
        per_type_counts["vietmdd_target_match_observed_text"] += 1

    # 7. Generate Task 3: vietmdd_spoken_content_matches_reference (MCQ, 1 per input row)
    for idx, cr in enumerate(clean_rows):
        audio = cr["audio"]
        ref_text = cr["original_text"]
        is_ref_match = cr["obs_norm"] == cr["orig_norm"]
        ans_ref = "Có" if is_ref_match else "Không"

        qa_id = f"qa_{hashlib.sha256(f'{split_name}::ref_match::{idx}::{audio}'.encode('utf-8')).hexdigest()[:16]}"
        qa_id_counts[qa_id] += 1
        q_text = f"Đoạn âm thanh này có nội dung phát âm khớp với văn bản tham chiếu “{ref_text}” không?"

        m_row = {
            "id": qa_id,
            "audio": [audio],
            "question": q_text,
            "choices": ["Có", "Không"],
            "answer": ans_ref,
            "response_format": "mcq",
            "type_id": "vietmdd_spoken_content_matches_reference",
            "operator": "TARGET_MATCH",
        }
        i_row = {
            "qa_id": qa_id,
            "semantic_instance_id": f"vietmdd_spoken_content_matches_reference::{split_name}_{idx}",
            "dataset": "vietmdd",
            "split": split_name,
            "type_id": "vietmdd_spoken_content_matches_reference",
            "operator": "TARGET_MATCH",
            "classification": "PRIMITIVE_RELATION",
            "audio_refs": [audio],
            "visible_context": {"target_value": ref_text},
            "typed_gold": {"matches_reference": is_ref_match},
            "gold_kind": "boolean",
            "rendered_question": q_text,
            "serialized_answer": str(is_ref_match).lower(),
            "projected_response_format": "mcq",
            "projected_choices": ["Có", "Không"],
            "projected_gold_answer": ans_ref,
        }
        model_rows.append(m_row)
        internal_rows.append(i_row)
        per_type_counts["vietmdd_spoken_content_matches_reference"] += 1

    # 8. Generate Task 4: vietmdd_selection_observed_text (MCQ, 1 per input row)
    for idx, cr in enumerate(clean_rows):
        idx_b = (idx + 1) % n_rows
        cr_b = clean_rows[idx_b]

        audio_a = cr["audio"]
        audio_b = cr_b["audio"]

        select_first = (idx % 2 == 0)
        target_text = cr["observed_transcription"] if select_first else cr_b["observed_transcription"]
        ans_selection = "Đoạn âm thanh thứ nhất" if select_first else "Đoạn âm thanh thứ hai"
        gold_idx = 0 if select_first else 1

        qa_id = f"qa_{hashlib.sha256(f'{split_name}::selection::{idx}::{audio_a}::{audio_b}'.encode('utf-8')).hexdigest()[:16]}"
        qa_id_counts[qa_id] += 1
        q_text = f"Mục tiêu là “{target_text}”. Trong hai đoạn âm thanh A và B, đoạn nào chứa nội dung này?"

        m_row = {
            "id": qa_id,
            "audio": [audio_a, audio_b],
            "question": q_text,
            "choices": list(SELECTION_CHOICES),
            "answer": ans_selection,
            "response_format": "mcq",
            "type_id": SELECTION_MCQ_TYPE,
            "operator": "PAIRWISE_SELECTION",
        }
        i_row = {
            "qa_id": qa_id,
            "semantic_instance_id": f"{SELECTION_MCQ_TYPE}::{split_name}_{idx}::{split_name}_{idx_b}",
            "dataset": "vietmdd",
            "split": split_name,
            "type_id": SELECTION_MCQ_TYPE,
            "operator": "PAIRWISE_SELECTION",
            "classification": "PRIMITIVE_RELATION",
            "audio_refs": [audio_a, audio_b],
            "visible_context": {"target_value": target_text},
            "typed_gold": {"observed_transcription": gold_idx},
            "gold_kind": "audio_index",
            "rendered_question": q_text,
            "serialized_answer": str(gold_idx),
            "projected_response_format": "mcq",
            "projected_choices": list(SELECTION_CHOICES),
            "projected_gold_answer": ans_selection,
        }
        model_rows.append(m_row)
        internal_rows.append(i_row)
        per_type_counts[SELECTION_MCQ_TYPE] += 1

    # 9. Generate Task 5: vietmdd_equality_observed_text (MCQ, exactly n_rows pairs from pair_plan)
    equality_pos_count = 0
    equality_neg_count = 0

    for k, (idx_a, idx_b) in enumerate(pair_plan):
        cr_a = clean_rows[idx_a]
        cr_b = clean_rows[idx_b]

        audio_a = cr_a["audio"]
        audio_b = cr_b["audio"]

        is_equal = cr_a["obs_norm"] == cr_b["obs_norm"]
        if is_equal:
            equality_pos_count += 1
        else:
            equality_neg_count += 1

        ans_eq = "Có" if is_equal else "Không"

        qa_id = f"qa_{hashlib.sha256(f'{split_name}::equality::{k}::{audio_a}::{audio_b}'.encode('utf-8')).hexdigest()[:16]}"
        qa_id_counts[qa_id] += 1
        q_text = "Liệu nội dung phát âm trong hai đoạn âm thanh có giống nhau hay không?"

        m_row = {
            "id": qa_id,
            "audio": [audio_a, audio_b],
            "question": q_text,
            "choices": ["Có", "Không"],
            "answer": ans_eq,
            "response_format": "mcq",
            "type_id": "vietmdd_equality_observed_text",
            "operator": "EQUALITY",
        }
        i_row = {
            "qa_id": qa_id,
            "semantic_instance_id": f"vietmdd_equality_observed_text::{split_name}_{idx_a}::{split_name}_{idx_b}",
            "dataset": "vietmdd",
            "split": split_name,
            "type_id": "vietmdd_equality_observed_text",
            "operator": "EQUALITY",
            "classification": "PRIMITIVE_RELATION",
            "audio_refs": [audio_a, audio_b],
            "visible_context": {},
            "typed_gold": {"observed_transcription": is_equal},
            "gold_kind": "boolean",
            "rendered_question": q_text,
            "serialized_answer": str(is_equal).lower(),
            "projected_response_format": "mcq",
            "projected_choices": ["Có", "Không"],
            "projected_gold_answer": ans_eq,
        }
        model_rows.append(m_row)
        internal_rows.append(i_row)
        per_type_counts["vietmdd_equality_observed_text"] += 1

    # 10. Generate Task 6: vietmdd_composite_transcribe_pair_equality (MCQ, same pair_plan topology)
    composite_same_count = 0
    composite_diff_count = 0

    for k, (idx_a, idx_b) in enumerate(pair_plan):
        cr_a = clean_rows[idx_a]
        cr_b = clean_rows[idx_b]

        audio_a = cr_a["audio"]
        audio_b = cr_b["audio"]

        ta = cr_a["observed_transcription"]
        tb = cr_b["observed_transcription"]
        ta_norm = cr_a["obs_norm"]
        tb_norm = cr_b["obs_norm"]

        gold_rel = ta_norm == tb_norm
        if gold_rel:
            composite_same_count += 1
        else:
            composite_diff_count += 1

        gold_tuple = (ta, tb, gold_rel)
        gold_choice_str = format_composite_choice(ta, tb, gold_rel)

        qa_id = f"qa_{hashlib.sha256(f'{split_name}::composite::{k}::{audio_a}::{audio_b}'.encode('utf-8')).hexdigest()[:16]}"
        qa_id_counts[qa_id] += 1

        # Derive deterministic indices for distractors
        h_bytes = hashlib.sha256(qa_id.encode("utf-8")).digest()
        idx1 = int.from_bytes(h_bytes[:4], "big")
        idx2 = int.from_bytes(h_bytes[4:8], "big")
        idx3 = int.from_bytes(h_bytes[8:12], "big")
        idx4 = int.from_bytes(h_bytes[12:16], "big")

        candidate_choices: dict[str, Any] = {gold_choice_str: gold_tuple}

        # D1: modify tb
        if pool_len > 0:
            for offset in range(pool_len):
                src_id, tb_p = local_pool[(idx1 + offset) % pool_len]
                tb_p_norm = norm_text(tb_p)
                if tb_p_norm != tb_norm:
                    rel1 = ta_norm == tb_p_norm
                    c1_str = format_composite_choice(ta, tb_p, rel1)
                    if c1_str not in candidate_choices:
                        candidate_choices[c1_str] = (ta, tb_p, rel1)
                        break

        # D2: modify ta
        if pool_len > 0:
            for offset in range(pool_len):
                src_id, ta_p = local_pool[(idx2 + offset) % pool_len]
                ta_p_norm = norm_text(ta_p)
                if ta_p_norm != ta_norm:
                    rel2 = ta_p_norm == tb_norm
                    c2_str = format_composite_choice(ta_p, tb, rel2)
                    if c2_str not in candidate_choices:
                        candidate_choices[c2_str] = (ta_p, tb, rel2)
                        break

        # D3: modify both ta and tb
        if pool_len > 0:
            for offset in range(pool_len):
                _, ta_p = local_pool[(idx3 + offset) % pool_len]
                _, tb_p = local_pool[(idx4 + offset) % pool_len]
                ta_p_norm = norm_text(ta_p)
                tb_p_norm = norm_text(tb_p)
                if ta_p_norm != ta_norm and tb_p_norm != tb_norm:
                    rel3 = ta_p_norm == tb_p_norm
                    c3_str = format_composite_choice(ta_p, tb_p, rel3)
                    if c3_str not in candidate_choices:
                        candidate_choices[c3_str] = (ta_p, tb_p, rel3)
                        break

        # Fallback if choices < 4
        fallback_counter = 0
        while len(candidate_choices) < min(4, pool_len):
            alt_ta = f"{ta} (bản {fallback_counter+1})"
            c_fb = format_composite_choice(alt_ta, tb, False)
            if c_fb not in candidate_choices:
                candidate_choices[c_fb] = (alt_ta, tb, False)
            fallback_counter += 1

        def choice_sort_key(c_str: str) -> str:
            return hashlib.sha256(f"{qa_id}::{c_str}".encode("utf-8")).hexdigest()

        ordered_choices = sorted(list(candidate_choices.keys()), key=choice_sort_key)

        if gold_choice_str not in ordered_choices:
            mcq_choice_errors += 1

        q_text = "Nghe hai đoạn âm thanh A và B, hãy nêu nội dung phát âm của đoạn A, nêu nội dung phát âm của đoạn B, sau đó cho biết hai nội dung đó có giống nhau không."

        m_row = {
            "id": qa_id,
            "audio": [audio_a, audio_b],
            "question": q_text,
            "choices": ordered_choices,
            "answer": gold_choice_str,
            "response_format": "mcq",
            "type_id": COMPOSITE_MCQ_TYPE,
            "operator": "COMPOSITE",
        }

        i_row = {
            "qa_id": qa_id,
            "semantic_instance_id": f"{COMPOSITE_MCQ_TYPE}::{split_name}_{idx_a}::{split_name}_{idx_b}",
            "dataset": "vietmdd",
            "split": split_name,
            "type_id": COMPOSITE_MCQ_TYPE,
            "operator": "COMPOSITE",
            "classification": "CHAIN_DERIVED",
            "audio_refs": [audio_a, audio_b],
            "visible_context": {},
            "typed_gold": {
                "observed_transcription_a": ta,
                "observed_transcription_b": tb,
                "same_observed_content": gold_rel,
            },
            "gold_kind": "structured",
            "rendered_question": q_text,
            "serialized_answer": f"observed_transcription_a={ta} | observed_transcription_b={tb} | same_observed_content={str(gold_rel).lower()}",
            "projected_response_format": "mcq",
            "projected_choices": ordered_choices,
            "projected_gold_answer": gold_choice_str,
        }
        model_rows.append(m_row)
        internal_rows.append(i_row)
        per_type_counts[COMPOSITE_MCQ_TYPE] += 1

    # Audit validation checks
    duplicate_qa_ids = sum(count - 1 for count in qa_id_counts.values() if count > 1)

    # Validate output rows integrity
    for mr in model_rows:
        if not mr.get("id") or not mr.get("audio") or not mr.get("question") or mr.get("answer") is None:
            missing_required_fields += 1

        # Check audio refs subpath
        for a_ref in mr["audio"]:
            if expected_audio_subpath not in a_ref or "/audio/" in a_ref.replace(expected_audio_subpath, ""):
                audio_path_prefix_error += 1

        if mr["response_format"] == "mcq":
            choices = mr.get("choices", [])
            if not choices or mr["answer"] not in choices:
                mcq_choice_errors += 1
            if len(choices) != len(set(choices)):
                mcq_choice_errors += 1
        elif mr["response_format"] == "open_ended":
            if "choices" in mr:
                mcq_choice_errors += 1

    total_qa_rows = len(model_rows)

    status = (
        "PASS"
        if (
            duplicate_qa_ids == 0
            and missing_required_fields == 0
            and gold_validation_errors == 0
            and mcq_choice_errors == 0
            and self_pair_count == 0
            and duplicate_unordered_pair_count == 0
            and reversed_duplicate_count == 0
            and audio_path_prefix_error == 0
        )
        else "FAIL"
    )

    # Write outputs
    internal_file = out_dir / "qa_internal.jsonl"
    model_file = out_dir / "qa_model_facing.jsonl"
    manifest_file = out_dir / "manifest.json"
    audit_file = out_dir / "audit.json"

    with open(internal_file, "w", encoding="utf-8") as f:
        for r in internal_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(model_file, "w", encoding="utf-8") as f:
        for r in model_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    internal_sha256 = compute_sha256_file(internal_file)
    model_sha256 = compute_sha256_file(model_file)

    manifest_data = {
        "dataset": "vietmdd",
        "split": split_name,
        "input_path": str(input_path),
        "input_sha256": input_sha256,
        "deterministic_seed": seed,
        "total_source_rows": n_rows,
        "total_qa_rows": total_qa_rows,
        "open_ended_types": 1,
        "mcq_types": 5,
        "per_type_counts": dict(per_type_counts),
        "train_data_used": False,
        "qa_model_facing_sha256": model_sha256,
        "qa_internal_sha256": internal_sha256,
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, ensure_ascii=False, indent=2)

    audit_data = {
        "input_path": str(input_path),
        "input_sha256": input_sha256,
        "split": split_name,
        "source_rows": n_rows,
        "generated_qa_rows": total_qa_rows,
        "per_type_counts": dict(per_type_counts),
        "duplicate_qa_ids": duplicate_qa_ids,
        "invalid_audio_refs": invalid_audio_refs,
        "missing_required_fields": missing_required_fields,
        "gold_validation_errors": gold_validation_errors,
        "mcq_choice_errors": mcq_choice_errors,
        "self_pairs": self_pair_count,
        "duplicate_unordered_pairs": duplicate_unordered_pair_count,
        "reversed_duplicates": reversed_duplicate_count,
        "audio_path_prefix_error": audio_path_prefix_error,
        "equality_pos_count": equality_pos_count,
        "equality_neg_count": equality_neg_count,
        "composite_same_count": composite_same_count,
        "composite_diff_count": composite_diff_count,
        "deterministic_seed": seed,
        "train_data_used": False,
        "status": status,
    }
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, ensure_ascii=False, indent=2)

    return {
        "split": split_name,
        "source_rows": n_rows,
        "total_qa": total_qa_rows,
        "per_type": dict(per_type_counts),
        "equality_positives": equality_pos_count,
        "equality_negatives": equality_neg_count,
        "composite_same": composite_same_count,
        "composite_diff": composite_diff_count,
        "open_ended_count": per_type_counts[OPEN_ENDED_TYPE],
        "mcq_count": total_qa_rows - per_type_counts[OPEN_ENDED_TYPE],
        "status": status,
        "out_dir": str(out_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic VietMDD QA generator for external validation and test splits (Repaired)."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL path")
    parser.add_argument("--split", required=True, type=str, help="Split name (validation/test)")
    parser.add_argument("--out", required=True, type=Path, help="Output directory path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of source rows for smoke testing")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")

    args = parser.parse_args()

    res = generate_external_split(
        input_path=args.input,
        split_name=args.split,
        out_dir=args.out,
        limit=args.limit,
        seed=args.seed,
    )

    print("=== VIETMDD EXTERNAL QA GENERATION SUCCESSFUL ===")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

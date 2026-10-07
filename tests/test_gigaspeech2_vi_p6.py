"""Focused contracts for GigaSpeech2-VI P6 (ASR) training QA.

Covers the TSV parser, comparator, audit duplicate/empty handling, the all-row
distractor index (not a fixed pool), deterministic modulo candidate lookup with
self/comparator/duplicate rejection, nested target selection, exact-source gold,
comparator-distinct four-choice invariants, logical audio IDs, TRAIN-only
isolation, zero-LLM generation and byte-identical double runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.config import ROOT
from src.sauvi_perception.tasks.p6_asr.gigaspeech2_vi_p6 import (
    AUDIO_NAMESPACE,
    BUCKETS,
    CHOICE_COUNT,
    P6SourceError,
    assert_train_source,
    audit_source,
    build_index_and_anchors,
    build_record,
    generate_stream,
    independent_audit,
    load_policy,
    parse_tsv_line,
    spoken_content_key,
    to_model_facing,
    word_bucket,
    write_index,
)
from src.common.zero_llm import scan_modules_for_forbidden_calls

CURRENT = ROOT / "outputs" / "training_qa" / "gigaspeech2_vi" / "p6" / "current"
POLICY_PATH = ROOT / "resources" / "semantics" / "p6_gigaspeech2_vi_policy.json"


@pytest.fixture()
def policy() -> dict:
    return load_policy(POLICY_PATH)


def _write_tsv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(f"{segment_id}\t{text}\n" for segment_id, text in rows)


@pytest.fixture()
def small_tsv(tmp_path: Path) -> Path:
    rows: list[tuple[str, str]] = []
    patterns = [
        "MỘT HAI BA",
        "BỐN NĂM SÁU BẢY",
        "TÁM CHÍN MƯỜI MỘT HAI BA BỐN",
        "CÂU NÀY DÀI HƠN MỘT CHÚT ĐỂ KHÁC ĐỘ DÀI RÕ RÀNG",
    ]
    for i in range(200):
        rows.append((f"seg-{i:04d}", f"{patterns[i % len(patterns)]} {i}"))
    # many rows sharing one transcript (exercises comparator rejection)
    for i in range(12):
        rows.append((f"dup-{i:03d}", "XIN CHÀO CÁC BẠN"))
    path = tmp_path / "train_refined.tsv"
    _write_tsv(path, rows)
    return path


# --- Parser -----------------------------------------------------------------


def test_parse_splits_on_first_tab_only():
    assert parse_tsv_line("id\ttext\n") == ("id", "text")
    assert parse_tsv_line("id\ta\tb\n") == ("id", "a\tb")
    assert parse_tsv_line("id\t") == ("id", "")
    assert parse_tsv_line("no tab here\n") is None


def test_assert_train_source_rejects_dev_test_sauvi(tmp_path):
    for name in ("test.tsv", "dev.tsv", "sauvi.tsv", "train_raw.tsv"):
        with pytest.raises(P6SourceError):
            assert_train_source(tmp_path / name)
    assert_train_source(tmp_path / "train_refined.tsv")


# --- Comparator -------------------------------------------------------------


def test_comparator_nfkc_casefold_punct_whitespace():
    assert spoken_content_key("ＡＢＣ") == "abc"
    assert spoken_content_key("CAO BẰNG") == "cao bằng"
    assert spoken_content_key("a, b!") == "a b"
    assert spoken_content_key("  a\t b \n") == "a b"


def test_comparator_keeps_vietnamese_diacritics():
    assert spoken_content_key("CAO BẰNG") != "cao bang"
    assert spoken_content_key("ĐÀ NẴNG") == "đà nẵng"


def test_word_bucket_boundaries():
    assert word_bucket(1) == "1-5"
    assert word_bucket(6) == "6-10"
    assert word_bucket(11) == "11-20"
    assert word_bucket(21) == "21-40"
    assert word_bucket(41) == "41-80"
    assert word_bucket(81) == ">80"


# --- Audit ------------------------------------------------------------------


def test_audit_counts_duplicate_segment_ids_and_empty(tmp_path):
    path = tmp_path / "train_refined.tsv"
    _write_tsv(
        path,
        [("a", "MỘT HAI BA"), ("a", "KHÁC NỘI DUNG"), ("b", ""), ("c", "XIN CHÀO")],
    )
    audit = audit_source(path, verify=False)
    assert audit["parsing"]["parsed_rows"] == 3
    assert audit["parsing"]["empty_transcript"] == 1
    assert audit["segment_ids"]["duplicate_rows"] == 1
    assert audit["segment_ids"]["unique"] == 2


# --- All-row distractor index ----------------------------------------------


def test_distractor_universe_is_all_rows_not_fixed_pool(small_tsv):
    eligible, _anchors, index = build_index_and_anchors(small_tsv, target_count=20)
    assert eligible == 212
    assert index.total_indexed() == eligible
    assert sum(index.bucket_size(b) for b in BUCKETS) == eligible
    import src.sauvi_perception.tasks.p6_asr.gigaspeech2_vi_p6 as module

    assert not hasattr(module, "DISTRACTOR_POOL_SIZE")


def test_bucket_index_construction(small_tsv):
    _eligible, _anchors, index = build_index_and_anchors(small_tsv, target_count=5)
    # every indexed offset resolves to a parseable row
    with index:
        for bucket in BUCKETS:
            for position in range(min(index.bucket_size(bucket), 3)):
                parsed = index.read_row(index.bucket_arrays[bucket][position])
                assert parsed is not None and parsed[0] != ""


def test_index_round_trips_to_disk(small_tsv, tmp_path):
    _eligible, _anchors, index = build_index_and_anchors(small_tsv, target_count=5)
    meta = write_index(index, tmp_path / "index")
    assert meta["total_indexed"] == 212
    assert meta["binary_bytes"] > 0


# --- Selection + distractors ------------------------------------------------


def test_target_selection_is_nested_and_order_independent(small_tsv, tmp_path):
    _e, anchors_5, _i = build_index_and_anchors(small_tsv, target_count=5)
    _e, anchors_10, _i = build_index_and_anchors(small_tsv, target_count=10)
    _e, anchors_20, _i = build_index_and_anchors(small_tsv, target_count=20)
    assert [a["segment_id"] for a in anchors_5] == [
        a["segment_id"] for a in anchors_10[:5]
    ]
    assert [a["segment_id"] for a in anchors_10] == [
        a["segment_id"] for a in anchors_20[:10]
    ]

    lines = small_tsv.read_text(encoding="utf-8").splitlines()
    reversed_path = tmp_path / "train_refined.tsv"
    reversed_path.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
    _e, anchors_rev, _i = build_index_and_anchors(reversed_path, target_count=20)
    assert [a["segment_id"] for a in anchors_20] == [
        a["segment_id"] for a in anchors_rev
    ]


def test_distractors_reject_self_comparator_and_duplicates(small_tsv, policy):
    _eligible, anchors, index = build_index_and_anchors(small_tsv, target_count=50)
    with index:
        for anchor in anchors:
            distractors = index.select_distractors(anchor)
            assert len(distractors) == 3
            keys = [spoken_content_key(d["text"]) for d in distractors]
            assert len(set(keys)) == 3
            assert anchor["key"] not in keys
            assert all(d["segment_id"] != anchor["segment_id"] for d in distractors)


def test_distractor_selection_deterministic(small_tsv):
    _eligible, anchors, index = build_index_and_anchors(small_tsv, target_count=20)
    with index:
        first = [index.select_distractors(a) for a in anchors]
        second = [index.select_distractors(a) for a in anchors]
    assert first == second


def test_qa_four_choices_and_exact_gold(small_tsv, policy):
    _eligible, anchors, index = build_index_and_anchors(small_tsv, target_count=30)
    with index:
        for anchor in anchors:
            record = build_record(anchor, index.select_distractors(anchor), policy)
            assert len(record["choices"]) == CHOICE_COUNT
            assert record["answer"] == anchor["text"]
            assert record["answer"] in record["choices"]
            keys = [spoken_content_key(c) for c in record["choices"]]
            assert len(set(keys)) == CHOICE_COUNT


def test_logical_audio_id_only(small_tsv, policy):
    _eligible, anchors, index = build_index_and_anchors(small_tsv, target_count=10)
    with index:
        for anchor in anchors:
            record = build_record(anchor, index.select_distractors(anchor), policy)
            assert record["audio_id"] == f"{AUDIO_NAMESPACE}/{record['segment_id']}"
            assert ":" not in record["audio_id"]
            assert "\\" not in record["audio_id"]
            model = to_model_facing(record)
            assert set(model) == {
                "id",
                "task",
                "audio_id",
                "question",
                "choices",
                "answer",
            }


# --- Streaming generation + reproducibility --------------------------------


def test_byte_identical_double_run_and_independent_audit(tmp_path, small_tsv, policy):
    _eligible, anchors, index = build_index_and_anchors(small_tsv, target_count=20)
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    generate_stream(dir_a, anchors, index, policy)
    generate_stream(dir_b, anchors, index, policy)
    for name in ("qa_internal.jsonl", "qa_model_facing.jsonl", "audit.json"):
        assert (dir_a / name).read_bytes() == (dir_b / name).read_bytes()
    audit = independent_audit(
        dir_a / "qa_internal.jsonl", dir_a / "qa_model_facing.jsonl"
    )
    assert audit["qa_count"] == 20
    assert audit["unique_qa_ids"] == 20
    assert audit["invalid_gold"] == 0
    assert audit["answer_not_in_choices"] == 0
    assert audit["choice_count_not_4"] == 0
    assert audit["comparator_duplicate_choices"] == 0
    assert audit["self_distractor"] == 0
    assert audit["absolute_paths"] == 0
    assert audit["model_facing_schema_ok"] is True


# --- Isolation / zero LLM ---------------------------------------------------


def test_generator_source_has_no_test_dev_sauvi_or_llm():
    text = (ROOT / "src" / "sauvi_perception" / "tasks" / "p6_asr" / "gigaspeech2_vi_p6.py").read_text(encoding="utf-8")
    assert "manifest_v2" not in text
    assert "openai" not in text
    assert "llm_client" not in text
    assert "urllib" not in text
    assert "requests" not in text


def test_zero_llm_module_scan():
    result = scan_modules_for_forbidden_calls(modules=("src/sauvi_perception/tasks/p6_asr/gigaspeech2_vi_p6.py",))
    assert result["status"] == "PASS"
    assert result["llm_calls"] == 0


# --- Full generated artifacts ----------------------------------------------


def test_source_audit_reports_zero_duplicate_segment_ids():
    audit_path = CURRENT / "source_audit.json"
    if not audit_path.is_file():
        pytest.skip("P6 source audit not generated")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["parsing"]["malformed_rows"] == 0
    assert audit["segment_ids"]["duplicate_rows"] == 0
    assert audit["eligibility"]["eligible_rows"] == 4_996_366


def test_smoke_artifacts_pass():
    if not (CURRENT / "smoke" / "qa_internal.jsonl").is_file():
        pytest.skip("P6 smoke artifacts not generated")
    audit = json.loads((CURRENT / "smoke" / "audit.json").read_text(encoding="utf-8"))
    assert audit["counts"]["qa"] == 100
    assert audit["validity"]["comparator_collision_choices"] == 0


def test_production_100k_artifacts_pass():
    dest = CURRENT / "100k"
    if not (dest / "qa_internal.jsonl").is_file():
        pytest.skip("P6 100k artifacts not generated")
    independent = json.loads(
        (dest / "independent_audit.json").read_text(encoding="utf-8")
    )
    assert independent["qa_count"] == 100_000
    assert independent["unique_anchor_segment_ids"] == 100_000
    assert independent["unique_logical_audio_ids"] == 100_000
    assert independent["invalid_gold"] == 0
    assert independent["comparator_duplicate_choices"] == 0
    assert independent["absolute_paths"] == 0
    assert independent["model_facing_schema_ok"] is True

"""Focused contracts for SAUVI P5 (speech sentiment) training QA.

Covers pinned-revision enforcement, TRAIN-only loading, label mapping and
unsupported-label exclusion, three-choice QA invariants, model-facing leakage
exclusion, TRAIN/TEST contamination + duplicate-audio decisions, zero-LLM
generation and byte-identical double runs.
"""

from __future__ import annotations

import json

import pytest

from src.common.config import ROOT
from src.common.zero_llm import scan_modules_for_forbidden_calls
from src.sauvi_perception.tasks.p5_sentiment.speech_sentiment_p5 import (
    ANSWER_SPACE,
    LABEL_MAP,
    SOURCE_REVISION,
    P5SourceError,
    audit_ok,
    build_qa,
    class_information,
    contamination_audit,
    duplicate_audio_decisions,
    eligible_rows,
    load_policy,
    load_provenance,
    load_test_rows,
    load_train_rows,
    spoken_content_key,
    to_model_facing,
    verify_source,
    write_release,
)

OUT = ROOT / "outputs" / "training_qa" / "sentiment_reasoning" / "p5" / "current"
POLICY_PATH = ROOT / "resources" / "semantics" / "p5_sentiment_policy.json"


@pytest.fixture()
def policy() -> dict:
    return load_policy(POLICY_PATH)


def _anchor(index: int, label: str) -> dict:
    return {
        "row_index": index,
        "source_row_id": f"train-00000-of-00001/{index}",
        "audio_id": f"sentiment_reasoning/train/train-00000-of-00001/{index}",
        "source_label": label,
        "source_text": "nội dung",
        "gold": LABEL_MAP[label],
    }


# --- Source -----------------------------------------------------------------


def test_pinned_revision_and_provenance():
    prov = load_provenance()
    assert prov["revision"] == SOURCE_REVISION
    assert prov["train_files"][0]["path"].endswith("train-00000-of-00001.parquet")
    assert prov["reserved_test_files"][0]["path"].endswith(
        "test-00000-of-00001.parquet"
    )


def test_verify_source_rejects_wrong_revision(tmp_path):
    prov = load_provenance()
    bad = dict(prov)
    bad["revision"] = "deadbeef"
    with pytest.raises(P5SourceError):
        verify_source(tmp_path / "train.parquet", tmp_path / "test.parquet", bad)


def test_verify_source_rejects_hash_mismatch(tmp_path):
    train = tmp_path / "train-00000-of-00001.parquet"
    test = tmp_path / "test-00000-of-00001.parquet"
    train.write_bytes(b"not the real train")
    test.write_bytes(b"not the real test")
    with pytest.raises(P5SourceError):
        verify_source(train, test)


def test_train_loader_rejects_test_path(tmp_path):
    with pytest.raises(P5SourceError):
        load_train_rows(tmp_path / "test-00000-of-00001.parquet")


def test_test_loader_rejects_train_path(tmp_path):
    with pytest.raises(P5SourceError):
        load_test_rows(tmp_path / "train-00000-of-00001.parquet")


# --- Label mapping ----------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "gold"),
    [("positive", "Tích cực"), ("neutral", "Trung tính"), ("negative", "Tiêu cực")],
)
def test_label_mapping(label, gold):
    anchors = eligible_rows([{"label": label, "text": "x"}])
    assert anchors[0]["gold"] == gold
    assert anchors[0]["source_label"] == label


def test_unsupported_label_excluded():
    rows = [
        {"label": "positive", "text": "a"},
        {"label": "mixed", "text": "b"},
        {"label": "", "text": "c"},
    ]
    anchors = eligible_rows(rows)
    assert [a["source_label"] for a in anchors] == ["positive"]


# --- QA ---------------------------------------------------------------------


def test_three_choices_and_source_gold(policy):
    records = build_qa([_anchor(1, "negative")], policy)
    record = records[0]
    assert len(record["choices"]) == 3
    assert set(record["choices"]) == set(ANSWER_SPACE)
    assert record["answer"] == "Tiêu cực"
    assert record["answer"] in record["choices"]
    assert record["gold_origin"] == "SOURCE"
    assert record["compiler_tier"] == "T1_PERCEPTION"


def test_deterministic_template_and_choice_order(policy):
    anchors = [_anchor(1, "positive"), _anchor(2, "neutral")]
    assert build_qa(anchors, policy) == build_qa(anchors, policy)


def test_no_transcript_label_or_metadata_leakage(policy):
    record = build_qa([_anchor(7, "positive")], policy)[0]
    model = to_model_facing(record)
    assert set(model) == {"id", "task", "audio_id", "question", "choices", "answer"}
    serialized = json.dumps(model, ensure_ascii=False)
    assert "nội dung" not in serialized
    assert "positive" not in serialized
    assert "source_text" not in serialized
    assert "human_justification" not in serialized
    assert ":" not in model["audio_id"]
    assert "\\" not in model["audio_id"]


# --- Contamination ----------------------------------------------------------


def test_identical_audio_between_train_and_test_detected():
    report = contamination_audit(["h1", "h2", "h3"], ["h9", "h2", "h8"])
    assert report["encoded_byte_overlap_hashes"] == 1
    assert report["contaminated_train_rows"] == [1]


def test_contaminated_row_excluded_from_eligibility():
    rows = [
        {"label": "positive", "text": "a"},
        {"label": "negative", "text": "b"},
    ]
    report = contamination_audit(["h1", "h2"], ["h2"])
    excluded = set(report["contaminated_train_rows"])
    anchors = eligible_rows(rows, excluded_indices=excluded)
    assert [a["row_index"] for a in anchors] == [0]


def test_transcript_only_overlap_is_report_only():
    report = contamination_audit(
        ["h1", "h2"],
        ["h3", "h4"],
        train_texts=["xin chào", "cảm ơn"],
        test_texts=["xin chào", "khác"],
    )
    assert report["contaminated_train_row_count"] == 0
    assert report["transcript_exact_overlap"] == 1
    assert report["transcript_overlap_policy"] == "REPORT_ONLY"


def test_pcm_overlap_detected_separately():
    report = contamination_audit(
        ["h1", "h2"], ["h3"], train_pcm_hashes=["p1", "p2"], test_pcm_hashes=["p2"]
    )
    assert report["encoded_byte_overlap_hashes"] == 0
    assert report["pcm_overlap_hashes"] == 1
    assert report["pcm_contaminated_train_rows"] == [1]


# --- Duplicates -------------------------------------------------------------


def test_same_audio_same_label_canonical_dedup():
    decisions = duplicate_audio_decisions(
        ["h1", "h1", "h2"], ["positive", "positive", "neutral"]
    )
    assert decisions["same_label_duplicate_groups"] == 1
    assert decisions["canonical_keep_indices"] == [0]
    assert decisions["same_label_exclude_indices"] == [1]
    assert decisions["conflicting_exclude_indices"] == []


def test_same_audio_conflicting_labels_exclude_all():
    decisions = duplicate_audio_decisions(
        ["h1", "h1", "h2"], ["positive", "negative", "neutral"]
    )
    assert decisions["conflicting_label_duplicate_groups"] == 1
    assert decisions["conflicting_exclude_indices"] == [0, 1]
    assert decisions["canonical_keep_indices"] == []


def test_transcript_normalization_keeps_diacritics():
    assert spoken_content_key("XIN CHÀO") == "xin chào"
    assert spoken_content_key("XIN CHÀO") != "xin chao"


# --- Metrics / global -------------------------------------------------------


def test_class_information_three_classes():
    info = class_information({"Tích cực": 1148, "Trung tính": 2800, "Tiêu cực": 1664})
    assert info["classes_present"] == 3
    assert 0 < info["normalized_entropy"] <= 1


def test_byte_identical_double_run(tmp_path, policy):
    records = build_qa([_anchor(1, "positive"), _anchor(2, "negative")], policy)
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    write_release(dir_a, records, policy)
    write_release(dir_b, records, policy)
    for name in (
        "qa_internal.jsonl",
        "qa_model_facing.jsonl",
        "audit.json",
        "template_registry.json",
        "manifest.json",
    ):
        assert (dir_a / name).read_bytes() == (dir_b / name).read_bytes()


def test_zero_llm_module_scan():
    result = scan_modules_for_forbidden_calls(modules=("src/sauvi_perception/tasks/p5_sentiment/speech_sentiment_p5.py",))
    assert result["status"] == "PASS"
    assert result["llm_calls"] == 0


def test_generator_source_has_no_llm_or_sauvi_manifest():
    text = (ROOT / "src" / "sauvi_perception" / "tasks" / "p5_sentiment" / "speech_sentiment_p5.py").read_text(encoding="utf-8")
    assert "manifest_v2" not in text
    assert "openai" not in text
    assert "llm_client" not in text


# --- Full artifact ----------------------------------------------------------


def test_full_p5_release_artifacts_pass():
    if not (OUT / "qa_internal.jsonl").is_file():
        pytest.skip("P5 release not generated")
    audit = json.loads((OUT / "audit.json").read_text(encoding="utf-8"))
    assert audit_ok(audit)
    assert audit["counts"]["qa"] == 5612
    distribution = json.loads(
        (OUT / "class_distribution.json").read_text(encoding="utf-8")
    )
    assert distribution["counts"] == {
        "Tích cực": 1148,
        "Trung tính": 2800,
        "Tiêu cực": 1664,
    }
    independent = json.loads(
        (OUT / "independent_audit.json").read_text(encoding="utf-8")
    )
    assert independent["wrong_source_row"] == 0
    assert independent["wrong_gold"] == 0
    overlap = json.loads((OUT / "split_overlap_audit.json").read_text(encoding="utf-8"))
    assert overlap["contaminated_train_row_count"] == 0
    assert overlap["pcm_contaminated_train_row_count"] == 0

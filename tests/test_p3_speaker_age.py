"""Focused contracts for SAUVI P3 (speaker age recognition) training QA.

Covers the strict age parser, deterministic age-bin mapping, speaker-level
raw/bin conflict policy, Unidentified/malformed/<18 exclusion, three-choice
invariants, model-facing exact-age/metadata exclusion, source isolation,
zero-LLM generation and byte-identical double runs.
"""

from __future__ import annotations

import json

import pytest

from src.common.config import ROOT
from src.common.zero_llm import scan_modules_for_forbidden_calls
from src.sauvi_perception.tasks.p3_speaker_age.speaker_age_p3 import (
    ANSWER_SPACE,
    GROUP_MIDDLE,
    GROUP_SENIOR,
    GROUP_YOUNG,
    P3SourceError,
    age_to_group,
    audit_ok,
    build_qa,
    class_distribution,
    class_information,
    eligible_rows,
    load_policy,
    parse_age,
    speaker_age_consistency,
    to_model_facing,
    write_release,
)

P3_OUT = ROOT / "outputs" / "training_qa" / "speech_massive_vi" / "p3" / "current"
POLICY_PATH = ROOT / "resources" / "semantics" / "p3_age_policy.json"


@pytest.fixture()
def policy() -> dict:
    return load_policy(POLICY_PATH)


def _row(
    row_id: str,
    speaker: str,
    age,
    *,
    validated: bool = True,
    locale: str = "vi-VN",
    partition: str = "train",
) -> dict:
    return {
        "id": row_id,
        "locale": locale,
        "partition": partition,
        "is_validated": validated,
        "speaker_id": speaker,
        "speaker_age": age,
    }


def _anchor(row_id: str, speaker: str, age: int) -> dict:
    return {
        "source_row_id": row_id,
        "audio_id": f"speech_massive_vi/train_115/{row_id}",
        "speaker_id": speaker,
        "source_speaker_age": str(age),
        "parsed_age": age,
        "derived_age_group": age_to_group(age),
    }


# --- Parser + binning -------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("18", 18), ("29", 29), ("30", 30), ("59", 59), ("60", 60), ("100", 100)],
)
def test_parse_valid_integers(raw, expected):
    assert parse_age(raw) == ("age", expected)


@pytest.mark.parametrize("raw", ["", "  ", "25 years", "20-29", "unknown", "abc", None])
def test_parse_malformed(raw):
    assert parse_age(raw) == ("malformed", None)


def test_parse_unidentified():
    assert parse_age("Unidentified") == ("unidentified", None)
    assert parse_age(" unidentified ") == ("unidentified", None)


@pytest.mark.parametrize(
    ("age", "group"),
    [
        (18, GROUP_YOUNG),
        (29, GROUP_YOUNG),
        (30, GROUP_MIDDLE),
        (59, GROUP_MIDDLE),
        (60, GROUP_SENIOR),
        (100, GROUP_SENIOR),
    ],
)
def test_age_bin_mapping(age, group):
    assert age_to_group(age) == group


def test_age_below_18_has_no_group():
    assert age_to_group(17) is None
    assert age_to_group(0) is None


# --- Eligibility / exclusions ----------------------------------------------


def test_below_18_unidentified_malformed_excluded():
    rows = [
        _row("1", "s1", "15"),
        _row("2", "s2", "Unidentified"),
        _row("3", "s3", "25 years"),
        _row("4", "s4", "24"),
    ]
    anchors = eligible_rows(rows, set())
    assert [a["source_row_id"] for a in anchors] == ["4"]


def test_vi_vn_only_and_validated_only():
    rows = [
        _row("1", "s1", "24", locale="fr-FR"),
        _row("2", "s2", "24", validated=False),
        _row("3", "s3", "24"),
    ]
    anchors = eligible_rows(rows, set())
    assert [a["source_row_id"] for a in anchors] == ["3"]


def test_raw_age_conflict_same_bin_keeps_rows():
    rows = [_row("1", "s1", "27"), _row("2", "s1", "28")]
    consistency = speaker_age_consistency(rows)
    assert consistency["raw_age_conflict_speakers"] == 1
    assert consistency["p3_bin_conflict_speakers"] == 0
    anchors = eligible_rows(rows, set(consistency["p3_bin_conflicting_speaker_ids"]))
    assert [a["source_row_id"] for a in anchors] == ["1", "2"]
    assert {a["derived_age_group"] for a in anchors} == {GROUP_YOUNG}


def test_raw_age_conflict_cross_bin_drops_whole_speaker():
    rows = [_row("1", "s1", "29"), _row("2", "s1", "30"), _row("3", "s2", "30")]
    consistency = speaker_age_consistency(rows)
    assert consistency["p3_bin_conflict_speakers"] == 1
    assert consistency["p3_bin_conflicting_speaker_ids"] == ["s1"]
    anchors = eligible_rows(rows, set(consistency["p3_bin_conflicting_speaker_ids"]))
    assert [a["source_row_id"] for a in anchors] == ["3"]


def test_unidentified_row_does_not_inherit_age():
    rows = [_row("1", "s1", "24"), _row("2", "s1", "Unidentified")]
    anchors = eligible_rows(rows, set())
    assert [a["source_row_id"] for a in anchors] == ["1"]
    assert anchors[0]["parsed_age"] == 24


# --- QA rendering -----------------------------------------------------------


def test_three_choices_and_valid_answer(policy):
    records = build_qa([_anchor("1", "s1", 24)], policy)
    record = records[0]
    assert len(record["choices"]) == 3
    assert set(record["choices"]) == set(ANSWER_SPACE)
    assert record["answer"] in record["choices"]
    assert record["answer"] == GROUP_YOUNG
    assert record["gold_origin"] == "DERIVED_SOURCE"
    assert record["compiler_tier"] == "T2_DERIVED"


def test_deterministic_template_and_choice_order(policy):
    anchors = [_anchor("1", "s1", 24), _anchor("2", "s2", 34)]
    assert build_qa(anchors, policy) == build_qa(anchors, policy)


def test_no_exact_age_or_metadata_leakage(policy):
    records = build_qa([_anchor("1", "s1", 24)], policy)
    model = to_model_facing(records[0])
    assert set(model) == {"id", "task", "audio_id", "question", "choices", "answer"}
    assert "24" not in model["choices"]
    assert model["answer"] in ANSWER_SPACE
    serialized = json.dumps(model, ensure_ascii=False)
    assert "speaker" not in serialized
    assert "age" not in serialized.lower()
    assert "revision" not in serialized
    assert "path" not in serialized


# --- Metrics ----------------------------------------------------------------


def test_class_information_metrics():
    info = class_information({GROUP_YOUNG: 40, GROUP_MIDDLE: 74, GROUP_SENIOR: 0})
    assert info["classes_present"] == 2
    assert info["majority_proportion"] == round(74 / 114, 6)
    assert 0 < info["normalized_entropy"] < 1


def test_class_distribution_reports_gap(policy):
    records = build_qa([_anchor("1", "s1", 24), _anchor("2", "s2", 34)], policy)
    distribution = class_distribution(records)
    assert distribution["counts"] == {GROUP_YOUNG: 1, GROUP_MIDDLE: 1, GROUP_SENIOR: 0}


# --- Reproducibility / isolation -------------------------------------------


def test_write_release_byte_identical_double_run(tmp_path, policy):
    records = build_qa([_anchor("1", "s1", 24), _anchor("2", "s2", 34)], policy)
    raw_audit = {"rows": 2}
    consistency = {"unique_speakers": 2}
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    write_release(dir_a, records, raw_audit, consistency, policy)
    write_release(dir_b, records, raw_audit, consistency, policy)
    for name in (
        "qa_internal.jsonl",
        "qa_model_facing.jsonl",
        "audit.json",
        "template_registry.json",
        "manifest.json",
    ):
        assert (dir_a / name).read_bytes() == (dir_b / name).read_bytes()


def test_loader_rejects_non_train_115(tmp_path):
    from src.sauvi_perception.tasks.p3_speaker_age.speaker_age_p3 import load_rows

    with pytest.raises(P3SourceError):
        load_rows(tmp_path / "validation-00000.parquet")
    with pytest.raises(P3SourceError):
        load_rows(tmp_path / "test-00000.parquet")


def test_zero_llm_module_scan():
    result = scan_modules_for_forbidden_calls(modules=("src/sauvi_perception/tasks/p3_speaker_age/speaker_age_p3.py",))
    assert result["status"] == "PASS"
    assert result["llm_calls"] == 0


def test_generator_source_has_no_llm_or_sauvi_manifest():
    text = (ROOT / "src" / "sauvi_perception" / "tasks" / "p3_speaker_age" / "speaker_age_p3.py").read_text(encoding="utf-8")
    assert "manifest_v2" not in text
    assert "openai" not in text
    assert "llm_client" not in text


# --- Full generated artifact ------------------------------------------------


def test_full_p3_release_artifacts_pass():
    if not (P3_OUT / "qa_internal.jsonl").is_file():
        pytest.skip("P3 release not generated")
    audit = json.loads((P3_OUT / "audit.json").read_text(encoding="utf-8"))
    assert audit_ok(audit)
    assert audit["counts"]["qa"] == 114
    distribution = json.loads(
        (P3_OUT / "class_distribution.json").read_text(encoding="utf-8")
    )
    assert distribution["counts"] == {
        GROUP_YOUNG: 40,
        GROUP_MIDDLE: 74,
        GROUP_SENIOR: 0,
    }
    assert distribution["coverage_incomplete"] is True
    independent = json.loads(
        (P3_OUT / "independent_gold_audit.json").read_text(encoding="utf-8")
    )
    assert independent["wrong_age_parse"] == 0
    assert independent["wrong_age_bin_gold"] == 0

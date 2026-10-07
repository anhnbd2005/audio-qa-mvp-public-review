"""Focused contracts for SAUVI P1 TRAINING QA generated from ViMD TRAIN.

Covers gold from speakerID, positive/negative pair uniqueness, 50/50 balance,
deterministic construction, the two-path beep-separated audio identity, the
absence of speaker/gender/region/province leakage, TRAIN-only isolation,
zero-LLM generation, byte-identical double runs and the composite renderer.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.common.config import ROOT
from src.autonomous_qa.production.production_zero_llm import scan_production_zero_llm
from src.sauvi_perception.tasks.p1_speaker_verification.vimd_p1_audio_recipe import (
    compose,
    generate_separator,
    load_recipe,
    verify_composite,
)
from src.sauvi_perception.tasks.p1_speaker_verification.vimd_p1_training_qa import (
    P1SourceError,
    audio_id,
    audit_ok,
    build_pair_plan,
    build_positive_pairs,
    build_qa,
    generate_artifacts,
    load_policy,
    load_train_rows,
    pair_id,
    profile_speakers,
    rendered_order,
    select_smoke_positives,
    speaker_maps,
    to_model_facing,
)

CURRENT = ROOT / "outputs" / "training_qa" / "vimd" / "p1" / "current"
POLICY_PATH = ROOT / "resources" / "semantics" / "p1_pair_policy.json"
RECIPE_PATH = ROOT / "resources" / "semantics" / "p1_audio_recipe.json"
SAME = "Cùng người nói"
DIFFERENT = "Khác người nói"


def _row(filename: str, speaker: str, gender: int, region: str, province: str) -> dict:
    return {
        "filename": filename,
        "speakerID": speaker,
        "gender": gender,
        "region": region,
        "province_name": province,
    }


@pytest.fixture()
def policy() -> dict:
    return load_policy(POLICY_PATH)


@pytest.fixture()
def recipe() -> dict:
    return load_recipe(RECIPE_PATH)


@pytest.fixture()
def rows() -> list[dict]:
    return [
        _row("s1_a.wav", "S1", 1, "North", "HaNoi"),
        _row("s1_b.wav", "S1", 1, "North", "HaNoi"),
        _row("s1_c.wav", "S1", 1, "North", "HaNoi"),
        _row("s2_a.wav", "S2", 1, "North", "CaoBang"),
        _row("s2_b.wav", "S2", 1, "North", "CaoBang"),
        _row("s3_a.wav", "S3", 0, "Central", "DaNang"),
        _row("s3_b.wav", "S3", 0, "Central", "DaNang"),
        _row("s4_a.wav", "S4", 1, "South", "CaMau"),
        _row("s5_a.wav", "S5", 0, "South", "CanTho"),
        _row("s5_b.wav", "S5", 1, "South", "CanTho"),
    ]


# --- Speaker capacity -------------------------------------------------------


def test_positive_capacity_and_histogram(rows):
    profile = profile_speakers(rows)
    assert profile["unique_speakers"] == 5
    assert profile["positive_pair_capacity"] == 6
    assert profile["speakers_conflicting_gender"] == 1
    assert profile["utterance_histogram"] == {1: 1, 2: 3, 3: 1}


def test_positive_pairs_unique_and_same_speaker(rows):
    speaker_of, _, _ = speaker_maps(rows)
    positives = build_positive_pairs(rows)
    assert len(positives) == 6
    seen = set()
    for positive in positives:
        a, b = positive["components"]
        assert a != b
        assert speaker_of[a] == speaker_of[b]
        key = tuple(sorted((a, b)))
        assert key not in seen
        seen.add(key)


# --- Gold from speakerID only ----------------------------------------------


def test_same_speaker_gets_same_gold(rows, policy):
    plan = build_pair_plan(rows, answers=policy["answers"])
    same = [e for e in plan if e["pair_class"] == "same"]
    assert all(e["gold"] == SAME for e in same)
    assert len(same) == 6


def test_different_speaker_gets_different_gold(rows, policy):
    plan = build_pair_plan(rows, answers=policy["answers"])
    speaker_of, _, _ = speaker_maps(rows)
    different = [e for e in plan if e["pair_class"] == "different"]
    assert all(e["gold"] == DIFFERENT for e in different)
    assert len(different) == 6
    for entry in different:
        a, b = entry["component_source_ids"]
        assert speaker_of[a] != speaker_of[b]


def test_exact_class_balance(rows, policy):
    plan = build_pair_plan(rows, answers=policy["answers"])
    same = sum(1 for e in plan if e["pair_class"] == "same")
    different = sum(1 for e in plan if e["pair_class"] == "different")
    assert same == different == 6


def test_negative_pairs_unique_no_reversal_no_self(rows, policy):
    plan = build_pair_plan(rows, answers=policy["answers"])
    keys = [tuple(e["component_source_ids"]) for e in plan]
    assert len(set(keys)) == len(keys)
    for entry in plan:
        a, b = entry["component_source_ids"]
        assert a != b
        assert a < b
    same_keys = {
        tuple(e["component_source_ids"]) for e in plan if e["pair_class"] == "same"
    }
    diff_keys = {
        tuple(e["component_source_ids"]) for e in plan if e["pair_class"] == "different"
    }
    assert not (same_keys & diff_keys)


def test_negative_prefers_same_region_same_gender(rows, policy):
    plan = build_pair_plan(rows, answers=policy["answers"])
    tiers = {e["negative_tier"] for e in plan if e["pair_class"] == "different"}
    assert "same_region_same_gender" in tiers


# --- Determinism ------------------------------------------------------------


def test_pair_construction_deterministic(rows, policy):
    assert build_pair_plan(rows, answers=policy["answers"]) == build_pair_plan(
        rows, answers=policy["answers"]
    )


def test_component_order_deterministic_and_gold_neutral(rows):
    assert rendered_order("a.wav", "b.wav") == rendered_order("a.wav", "b.wav")
    assert sorted(rendered_order("a.wav", "b.wav")) == ["a.wav", "b.wav"]
    assert pair_id("a.wav", "b.wav") == pair_id("b.wav", "a.wav")
    assert audio_id("a.wav", "b.wav") == audio_id("b.wav", "a.wav")


def test_template_and_choices_deterministic(rows, policy):
    speaker_of, _, _ = speaker_maps(rows)
    plan = build_pair_plan(rows, answers=policy["answers"])
    first = build_qa(plan, policy, speaker_of)
    second = build_qa(plan, policy, speaker_of)
    assert first == second
    for record in first:
        assert len(record["choices"]) == 2
        assert record["answer"] in record["choices"]
        assert set(record["choices"]) == {SAME, DIFFERENT}


# --- Audio identity ---------------------------------------------------------


def test_audio_identity_is_two_logical_paths(rows, policy):
    speaker_of, _, _ = speaker_maps(rows)
    plan = build_pair_plan(rows, answers=policy["answers"])
    for record in build_qa(plan, policy, speaker_of):
        assert isinstance(record["audio_id"], list)
        assert len(record["audio_id"]) == 2
        assert record["composite_audio_id"].startswith("vimd_p1_")
        for component in record["audio_id"]:
            assert "/" not in component and "\\" not in component
        model = to_model_facing(record)
        assert model["audio_id"] == record["audio_id"]


def test_model_facing_has_no_metadata_leak(rows, policy):
    speaker_of, _, _ = speaker_maps(rows)
    plan = build_pair_plan(rows, answers=policy["answers"])
    allowed = {"id", "task", "audio_id", "question", "choices", "answer"}
    for record in build_qa(plan, policy, speaker_of):
        model = to_model_facing(record)
        assert set(model) == allowed
        serialized = json.dumps(model, ensure_ascii=False)
        assert "speaker" not in serialized.lower()
        assert "gender" not in serialized.lower()
        assert "pair_class" not in serialized
        for region in ("North", "Central", "South"):
            assert region not in serialized
        for province in ("HaNoi", "CaoBang", "DaNang", "CaMau", "CanTho"):
            assert province not in serialized


# --- Isolation / zero LLM ---------------------------------------------------


def test_loader_refuses_non_train_split_names(tmp_path):
    with pytest.raises(P1SourceError):
        load_train_rows(tmp_path / "test.jsonl")
    with pytest.raises(P1SourceError):
        load_train_rows(tmp_path / "valid.jsonl")
    with pytest.raises(P1SourceError):
        load_train_rows(tmp_path / "sauvi_test.jsonl")


def test_generated_records_are_train_only(rows, policy):
    speaker_of, _, _ = speaker_maps(rows)
    plan = build_pair_plan(rows, answers=policy["answers"])
    for record in build_qa(plan, policy, speaker_of):
        assert record["source_split"] == "train"
        assert record["source_dataset"] == "ViMD"


def test_generator_source_never_references_sauvi_manifest():
    text = (ROOT / "src" / "sauvi_perception" / "tasks" / "p1_speaker_verification" / "vimd_p1_training_qa.py").read_text(encoding="utf-8")
    assert "manifest_v2" not in text
    assert "normalized.jsonl" not in text
    assert "openai" not in text
    assert "llm_client" not in text
    assert "urllib" not in text
    assert "requests" not in text


def test_zero_llm_module_scan():
    result = scan_production_zero_llm(
        modules=("src/sauvi_perception/tasks/p1_speaker_verification/vimd_p1_training_qa.py", "src/sauvi_perception/tasks/p1_speaker_verification/vimd_p1_audio_recipe.py")
    )
    assert result["status"] == "PASS"
    assert result["llm_calls"] == 0


# --- Reproducibility --------------------------------------------------------


def test_byte_identical_double_run(tmp_path, rows, policy, recipe):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    plan = build_pair_plan(rows, answers=policy["answers"])
    generate_artifacts(rows, policy, recipe, dir_a, plan=plan)
    generate_artifacts(rows, policy, recipe, dir_b, plan=plan)
    for name in (
        "p1_pair_plan.jsonl",
        "qa_internal.jsonl",
        "qa_model_facing.jsonl",
        "composite_audio_manifest.jsonl",
        "template_registry.json",
        "p1_audio_recipe.json",
    ):
        assert (dir_a / name).read_bytes() == (dir_b / name).read_bytes()


def test_smoke_selection_deterministic(rows):
    positives = build_positive_pairs(rows)
    first = select_smoke_positives(positives, 3)
    second = select_smoke_positives(list(reversed(positives)), 3)
    assert [p["components"] for p in first] == [p["components"] for p in second]


# --- Composite renderer -----------------------------------------------------


def _tone(freq: float, seconds: float, sr: int) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_renderer_sample_rate_mono_beep_and_duration(recipe):
    sr = 16000
    a = _tone(220.0, 1.0, sr)
    b = _tone(330.0, 1.5, sr)
    composite, out_sr = compose(a, sr, b, sr, recipe)
    report = verify_composite(composite, out_sr, recipe, len(a), len(b))
    assert out_sr == sr
    assert report["mono"] is True
    assert report["duration_ok"] is True
    assert report["clipped"] is False
    assert report["beep_frequency_ok"] is True
    assert report["beep_nonzero"] is True
    expected = len(a) + len(b) + round(sr * (300 + 500 + 300) / 1000)
    assert report["frames"] == expected


def test_renderer_deterministic_waveform(recipe):
    sr = 16000
    a = _tone(220.0, 0.5, sr)
    b = _tone(330.0, 0.5, sr)
    first, _ = compose(a, sr, b, sr, recipe)
    second, _ = compose(a, sr, b, sr, recipe)
    assert np.array_equal(first, second)


def test_separator_has_expected_length_and_amplitude(recipe):
    sr = 48000
    separator = generate_separator(recipe, sr)
    assert separator.dtype == np.float32
    expected = round(sr * (300 + 500 + 300) / 1000)
    assert len(separator) == expected
    assert (
        float(np.max(np.abs(separator))) <= recipe["separator"]["peak_amplitude"] + 1e-6
    )


# --- Full generated artifact ------------------------------------------------


def test_full_artifact_counts_and_audit():
    if not (CURRENT / "qa_internal.jsonl").is_file():
        pytest.skip("P1 full training artifacts not generated")
    manifest = json.loads((CURRENT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["same"] == manifest["counts"]["different"]
    assert manifest["counts"]["total"] == 2 * manifest["counts"]["same"]
    audit = json.loads((CURRENT / "audit.json").read_text(encoding="utf-8"))
    assert audit_ok(audit)
    assert audit["pairs"]["capacity_met"] is True
    assert audit["validity"]["duplicate_unordered_pairs"] == 0
    assert audit["validity"]["different_with_same_speaker"] == 0
    assert audit["validity"]["same_with_different_speaker"] == 0
    assert audit["leakage"]["test_row_used"] == 0
    assert audit["leakage"]["sauvi_manifest_accessed"] is False
    assert audit["llm"]["row_level_llm_calls"] == 0


def test_full_model_facing_has_two_paths_and_no_metadata():
    if not (CURRENT / "qa_model_facing.jsonl").is_file():
        pytest.skip("P1 full training artifacts not generated")
    import re

    absolute = re.compile(r"[A-Za-z]:[\\/]")
    text = (CURRENT / "qa_model_facing.jsonl").read_text(encoding="utf-8")
    assert "kaggle" not in text.lower()
    for line in text.splitlines():
        record = json.loads(line)
        assert set(record) == {
            "id",
            "task",
            "audio_id",
            "question",
            "choices",
            "answer",
        }
        assert isinstance(record["audio_id"], list)
        assert len(record["audio_id"]) == 2
        serialized = json.dumps(record, ensure_ascii=False)
        assert not absolute.search(serialized)
        assert "speaker" not in serialized.lower()

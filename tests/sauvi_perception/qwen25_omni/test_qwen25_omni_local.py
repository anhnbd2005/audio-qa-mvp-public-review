"""CPU-safe unit tests for sauvi_perception qwen25_omni local QLoRA training package."""

import json
import random
import tempfile
from pathlib import Path

import pytest
import yaml

from src.sauvi_perception.qwen25_omni.data import (
    ManifestError,
    PathEscapeError,
    ForbiddenTestAccessError,
    choice_letter,
    forbid_test_access,
    load_manifest,
    permuted_choices,
    render_mcq_user_text,
    resolve_audio_path,
    target_letter_for,
    validate_choice_row,
)
from src.sauvi_perception.qwen25_omni.evaluate import (
    evaluate_predictions,
    is_correct,
    normalise_answer_text,
    parse_answer,
)
from src.sauvi_perception.qwen25_omni.run import load_config


def test_yaml_config_parsing_and_required_fields(tmp_path):
    config_dict = {
        "model": {"path": "E:/models/3B", "local_files_only": True},
        "data": {"audio_root": "E:/data", "train_jsonl": "E:/data/train.jsonl", "validation_jsonl": "E:/data/val.jsonl"},
        "output": {"dir": "E:/output"},
        "runtime": {"device": "cuda:0"},
        "quantization": {"enabled": True},
        "lora": {"r": 8},
        "training": {"epochs": 1},
        "resume": {"mode": "fresh"},
    }
    cfg_file = tmp_path / "test_config.yaml"
    cfg_file.write_text(yaml.dump(config_dict), encoding="utf-8")

    loaded = load_config(cfg_file)
    assert loaded["model"]["path"] == "E:/models/3B"
    assert loaded["runtime"]["device"] == "cuda:0"


def test_invalid_config_fails_closed(tmp_path):
    invalid_cfg = tmp_path / "invalid.yaml"
    invalid_cfg.write_text("model: {path: 'foo'}", encoding="utf-8")
    with pytest.raises(ValueError, match="Config missing required top-level section"):
        load_config(invalid_cfg)


def test_path_resolution_and_traversal_rejection(tmp_path):
    audio_root = tmp_path / "audio_root"
    audio_root.mkdir()
    valid_file = audio_root / "sub" / "audio.wav"
    valid_file.parent.mkdir(parents=True)
    valid_file.write_bytes(b"dummy")

    resolved = resolve_audio_path(audio_root, "sub/audio.wav")
    assert resolved == valid_file.resolve()

    with pytest.raises(PathEscapeError):
        resolve_audio_path(audio_root, "../outside.wav")


def test_test_path_rejection():
    with pytest.raises(ForbiddenTestAccessError):
        forbid_test_access("data/vimd_test.jsonl")

    with pytest.raises(ForbiddenTestAccessError):
        forbid_test_access("E:/datasets/test.jsonl")

    forbid_test_access("data/train.jsonl")
    forbid_test_access("data/val.jsonl")


def test_mcq_formatting_and_validation():
    user_text = render_mcq_user_text("What is this sound?", ["Option A", "Option B", "Option C"])
    assert "What is this sound?" in user_text
    assert "A. Option A" in user_text
    assert "B. Option B" in user_text
    assert "C. Option C" in user_text

    row = {
        "id": "s1",
        "audio": "a.wav",
        "question": "Q?",
        "choices": ["Option A", "Option B"],
        "answer": "Option A",
    }
    val = validate_choice_row(row)
    assert val["id"] == "s1"
    assert val["answer"] == "Option A"

    bad_row = {
        "id": "s2",
        "audio": "a.wav",
        "question": "Q?",
        "choices": ["Option A", "Option B"],
        "answer": "Option C",
    }
    with pytest.raises(ManifestError):
        validate_choice_row(bad_row)


def test_deterministic_permutation_and_target_derivation():
    choices = ["North", "Central", "South"]
    gold_answer = "North"

    p_choices = permuted_choices(choices, gold_answer, base_seed=42, epoch=0, sample_id="sample_100")
    assert set(p_choices) == set(choices)
    assert len(p_choices) == len(choices)

    target_letter = target_letter_for(p_choices, gold_answer)
    assert target_letter in "ABC"
    assert p_choices[ord(target_letter) - ord("A")] == gold_answer

    p_choices_again = permuted_choices(choices, gold_answer, base_seed=42, epoch=0, sample_id="sample_100")
    assert p_choices_again == p_choices


def test_evaluation_parsing_and_metrics():
    records = [
        {"gold_answer": "North", "raw_prediction": "A", "choices": ["North", "Central", "South"]},
        {"gold_answer": "South", "raw_prediction": "C", "choices": ["North", "Central", "South"]},
        {"gold_answer": "Central", "raw_prediction": "wrong text", "choices": ["North", "Central", "South"]},
    ]
    res = evaluate_predictions(records)
    assert res["rows"] == 3
    assert res["correct"] == 2
    assert pytest.approx(res["accuracy"], 0.01) == 0.666
    assert res["parser_failures"] == 1


def test_static_scans_no_kaggle_no_old_repo_no_hardcoded_paths():
    import src.sauvi_perception.qwen25_omni.data as data_mod
    import src.sauvi_perception.qwen25_omni.engine as engine_mod
    import src.sauvi_perception.qwen25_omni.evaluate as eval_mod
    import src.sauvi_perception.qwen25_omni.run as run_mod

    for mod in (data_mod, engine_mod, eval_mod, run_mod):
        import inspect
        src_text = inspect.getsource(mod)

        assert "/kaggle/" not in src_text
        assert "A_qwen" not in src_text
        assert "snapshot_download" not in src_text
        assert "pip install" not in src_text

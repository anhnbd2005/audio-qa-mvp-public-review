"""Regression test: ViMD raw gender encoding is 0=female, 1=male.

Loads normalize_vimd_gender from scripts/sauvi_perception/prepare_vimd.py (the single
normalization point used when regenerating data/vimd/sample.jsonl).
"""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sauvi_perception" / "prepare_vimd.py"


def _load():
    spec = importlib.util.spec_from_file_location("prepare_vimd", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


normalize_vimd_gender = _load().normalize_vimd_gender


def test_vimd_gender_mapping():
    assert normalize_vimd_gender(0) == "female"
    assert normalize_vimd_gender(1) == "male"


def test_vimd_gender_mapping_rejects_unknown_value():
    with pytest.raises(ValueError):
        normalize_vimd_gender(2)


def test_vimd_gender_mapping_rejects_invalid_value():
    with pytest.raises(ValueError):
        normalize_vimd_gender(None)

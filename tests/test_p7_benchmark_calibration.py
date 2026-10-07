"""Focused contracts for benchmark-calibrated P7 lyric-density rule inference.

Covers monotonicity, same-WPM conflicts, feasible boundaries, Decimal midpoints,
exact boundary behavior, exhaustive best monotonic search, determinism,
no per-ID overrides, policy provenance and zero-LLM/no-audio guarantees.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from src.common.config import ROOT
from src.sauvi_perception.tasks.p7_tempo.p7_benchmark_calibration import (
    CALIBRATION_DIR,
    POLICY_ORIGIN,
    POLICY_PATH,
    best_monotonic_search,
    class_statistics,
    derive_thresholds,
    feasible_boundaries,
    monotonicity,
    predict,
    reproduce,
)
from src.common.zero_llm import scan_modules_for_forbidden_calls


def _item(wpm: str, label: str, qa_id: str = "q") -> dict:
    return {
        "qa_id": qa_id,
        "zing_id": "Z0000000",
        "wpm": Decimal(wpm),
        "gold_label": label,
    }


def _clean_usable() -> list[dict]:
    return [
        _item("40", "Thưa", "a"),
        _item("50", "Thưa", "b"),
        _item("60", "Vừa", "c"),
        _item("70", "Vừa", "d"),
        _item("80", "Dày", "e"),
        _item("90", "Dày", "f"),
        _item("100", "Rất dày", "g"),
        _item("110", "Rất dày", "h"),
    ]


# --- Monotonicity -----------------------------------------------------------


def test_monotonicity_clean():
    result = monotonicity(_clean_usable())
    assert result["monotonic"] is True
    assert result["adjacent_inversions"] == 0


def test_monotonicity_detects_inversion():
    items = [_item("60", "Dày"), _item("70", "Thưa")]
    result = monotonicity(items)
    assert result["monotonic"] is False
    assert result["adjacent_inversions"] == 1


def test_same_wpm_conflicting_labels_detected():
    items = [_item("80", "Vừa"), _item("80", "Dày")]
    result = monotonicity(items)
    assert result["monotonic"] is False
    assert result["same_wpm_conflicting_labels"] == {"80": ["Dày", "Vừa"]}


# --- Boundaries + thresholds ------------------------------------------------


def test_feasible_boundaries_clean():
    boundaries = feasible_boundaries(_clean_usable())
    assert [b["clean"] for b in boundaries] == [True, True, True]
    assert boundaries[0]["lower_max"] == "50"
    assert boundaries[0]["upper_min"] == "60"


def test_feasible_boundaries_overlap():
    items = [
        _item("40", "Thưa"),
        _item("70", "Thưa"),
        _item("60", "Vừa"),
        _item("90", "Dày"),
        _item("100", "Rất dày"),
    ]
    boundaries = feasible_boundaries(items)
    assert boundaries[0]["clean"] is False  # max Thưa 70 >= min Vừa 60


def test_decimal_midpoint_exact():
    boundaries = [
        {
            "clean": True,
            "lower_max": "56.07237659206584",
            "upper_min": "56.147218799133135",
        },
        {
            "clean": True,
            "lower_max": "66.52089730425166",
            "upper_min": "67.3167718283625",
        },
        {
            "clean": True,
            "lower_max": "80.53470662356872",
            "upper_min": "80.79714221422142",
        },
    ]
    thresholds = derive_thresholds(boundaries)
    assert thresholds["t1"] == Decimal("56.1097976955994875")
    assert thresholds["t2"] == Decimal("66.91883456630708")
    assert thresholds["t3"] == Decimal("80.66592441889507")


def test_derive_thresholds_requires_clean():
    from src.sauvi_perception.tasks.p7_tempo.p7_benchmark_calibration import CalibrationError

    boundaries = [{"clean": False, "lower_max": "70", "upper_min": "60"}]
    with pytest.raises(CalibrationError):
        derive_thresholds(boundaries)


# --- Prediction boundary behavior ------------------------------------------


def test_exact_boundary_behavior():
    thresholds = {
        "t1": Decimal("56.1097976955994875"),
        "t2": Decimal("66.91883456630708"),
        "t3": Decimal("80.66592441889507"),
    }
    assert predict(Decimal("56.0"), thresholds) == "Thưa"
    assert predict(thresholds["t1"], thresholds) == "Vừa"
    assert predict(thresholds["t2"], thresholds) == "Dày"
    assert predict(thresholds["t3"], thresholds) == "Rất dày"
    assert predict(Decimal(1000), thresholds) == "Rất dày"


def test_no_per_id_overrides():
    thresholds = {
        "t1": Decimal("56.1"),
        "t2": Decimal("66.9"),
        "t3": Decimal("80.7"),
    }
    # same WPM -> same label regardless of qa_id
    assert predict(Decimal(80), thresholds) == predict(Decimal(80), thresholds)


# --- Exhaustive search ------------------------------------------------------


def test_best_monotonic_search_exact():
    best = best_monotonic_search(_clean_usable())
    assert best["correct"] == 8
    assert best["accuracy"] == 1.0


def test_reproduce_exact_and_mismatch():
    usable = _clean_usable()
    thresholds = derive_thresholds(feasible_boundaries(usable))
    result = reproduce(usable, thresholds)
    assert result["correct"] == result["total"]
    assert result["accuracy"] == 1.0
    assert result["mismatches"] == []
    # a deliberately shifted threshold set produces mismatches
    shifted = {"t1": Decimal(45), "t2": Decimal(75), "t3": Decimal(95)}
    assert reproduce(usable, shifted)["correct"] < len(usable)


def test_class_statistics():
    stats = class_statistics(_clean_usable())
    assert stats["Thưa"]["n"] == 2
    assert stats["Rất dày"]["max"] == 110.0


def test_deterministic_results():
    assert feasible_boundaries(_clean_usable()) == feasible_boundaries(_clean_usable())
    assert best_monotonic_search(_clean_usable()) == best_monotonic_search(
        _clean_usable()
    )


# --- Zero LLM / no audio ----------------------------------------------------


def test_zero_llm_module_scan():
    result = scan_modules_for_forbidden_calls(modules=("src/sauvi_perception/tasks/p7_tempo/p7_benchmark_calibration.py",))
    assert result["status"] == "PASS"
    assert result["llm_calls"] == 0


def test_no_llm_or_audio_in_source():
    text = (ROOT / "src" / "sauvi_perception" / "tasks" / "p7_tempo" / "p7_benchmark_calibration.py").read_text(encoding="utf-8")
    assert "openai" not in text
    assert "llm_client" not in text
    assert "soundfile" not in text
    assert "import yt_dlp" not in text


# --- Generated artifacts / policy ------------------------------------------


def test_calibration_summary_present():
    path = CALIBRATION_DIR / "calibration_summary.json"
    if not path.is_file():
        pytest.skip("P7 calibration not generated")
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["p7_rows"] == 89
    assert summary["usable_rows"] == 81
    assert summary["ambiguous_rows"] == 8
    assert summary["monotonicity"]["monotonic"] is True
    assert summary["reproduction"]["correct"] == 81
    assert summary["reproduction"]["accuracy"] == 1.0
    assert summary["status"] == "P7_BENCHMARK_DERIVED_RULE_EXACT"


def test_policy_provenance_marked_benchmark_derived():
    if not POLICY_PATH.is_file():
        pytest.skip("P7 policy not created")
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    assert (
        policy["policy_origin"] == POLICY_ORIGIN == "BENCHMARK_DERIVED_SEMANTIC_POLICY"
    )
    assert policy["thresholds"]["t1"] == "56.1097976955994875"
    assert policy["benchmark_reproduction"]["correct"] == 81
    assert "BENCHMARK_DERIVED" in policy["provenance_warning"]

"""Deterministic composite-audio renderer for SAUVI P1 training pairs.

Implements the project's already-frozen separator recipe (see
``resources/semantics/p1_audio_recipe.json``):

    speech_A  +  [pre_silence | 1000 Hz sine beep | post_silence]  +  speech_B

The beep has a short linear fade in/out. Native sample rate and channel count
are preserved when the two components agree; otherwise both are resampled to
the recipe sample rate. The result is clipped to [-1, 1].

This module performs no I/O of its own beyond reading the frozen recipe and
loading source waveforms supplied by the caller. It is deterministic: the same
inputs and recipe always yield the same samples.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from src.common.config import ROOT

RECIPE_PATH = ROOT / "resources" / "semantics" / "p1_audio_recipe.json"


def load_recipe(path: Path | None = None) -> dict:
    target = Path(path) if path is not None else RECIPE_PATH
    with open(target, "r", encoding="utf-8") as fh:
        return json.load(fh)


def to_mono(wave: np.ndarray) -> np.ndarray:
    """Collapse a (frames, channels) array to 1-D mono; leave 1-D unchanged."""
    array = np.asarray(wave)
    if array.ndim == 1:
        return array.astype(np.float32)
    return array.mean(axis=1).astype(np.float32)


def resample(wave: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
    if sample_rate == target_rate:
        return wave.astype(np.float32)
    return resample_poly(wave, target_rate, sample_rate).astype(np.float32)


def generate_separator(recipe: dict, sample_rate: int) -> np.ndarray:
    """pre_silence + faded sine beep + post_silence at ``sample_rate``."""
    spec = recipe["separator"]
    n_pre = round(sample_rate * spec["pre_silence_ms"] / 1000.0)
    n_beep = round(sample_rate * spec["beep_duration_ms"] / 1000.0)
    n_post = round(sample_rate * spec["post_silence_ms"] / 1000.0)

    t = np.arange(n_beep) / float(sample_rate)
    beep = (
        spec["peak_amplitude"] * np.sin(2.0 * np.pi * spec["beep_frequency_hz"] * t)
    ).astype(np.float32)

    n_fade = round(sample_rate * spec["fade_ms"] / 1000.0)
    if n_fade > 0 and 2 * n_fade <= n_beep:
        fade_in = np.linspace(0.0, 1.0, n_fade, endpoint=False, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, n_fade, endpoint=False, dtype=np.float32)
        beep[:n_fade] = beep[:n_fade] * fade_in
        beep[-n_fade:] = beep[-n_fade:] * fade_out

    silence_pre = np.zeros(n_pre, dtype=np.float32)
    silence_post = np.zeros(n_post, dtype=np.float32)
    return np.concatenate([silence_pre, beep, silence_post]).astype(np.float32)


def _prepare(wave: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
    return resample(to_mono(wave), sample_rate, target_rate)


def compose(
    wave_a: np.ndarray,
    sample_rate_a: int,
    wave_b: np.ndarray,
    sample_rate_b: int,
    recipe: dict | None = None,
) -> tuple[np.ndarray, int]:
    """Render ``A + separator + B``; returns (float32 mono waveform, sample rate)."""
    recipe = recipe or load_recipe()
    target = int(recipe["sample_rate"])
    if sample_rate_a == sample_rate_b:
        target = sample_rate_a
    signal_a = _prepare(wave_a, sample_rate_a, target)
    signal_b = _prepare(wave_b, sample_rate_b, target)
    separator = generate_separator(recipe, target)
    composite = np.concatenate([signal_a, separator, signal_b]).astype(np.float32)
    clip = recipe["clip"]
    return np.clip(composite, clip["min"], clip["max"]).astype(np.float32), target


def dominant_frequency(wave: np.ndarray, sample_rate: int) -> float:
    if wave.size == 0:
        return 0.0
    spectrum = np.abs(np.fft.rfft(wave))
    if spectrum.size == 0:
        return 0.0
    return float(np.argmax(spectrum) * sample_rate / wave.size)


def verify_composite(
    composite: np.ndarray,
    sample_rate: int,
    recipe: dict,
    frames_a: int,
    frames_b: int,
) -> dict:
    """Structural checks for a rendered composite (used by the audio smoke)."""
    spec = recipe["separator"]
    n_pre = round(sample_rate * spec["pre_silence_ms"] / 1000.0)
    n_beep = round(sample_rate * spec["beep_duration_ms"] / 1000.0)
    n_post = round(sample_rate * spec["post_silence_ms"] / 1000.0)
    n_sep = n_pre + n_beep + n_post
    beep = composite[frames_a + n_pre : frames_a + n_pre + n_beep]
    expected = frames_a + n_sep + frames_b
    peak = float(np.max(np.abs(composite))) if composite.size else 0.0
    return {
        "sample_rate": sample_rate,
        "mono": bool(np.asarray(composite).ndim == 1),
        "frames": int(composite.size),
        "expected_frames": int(expected),
        "duration_ok": int(composite.size) == expected,
        "peak_amplitude": peak,
        "clipped": peak > 1.0,
        "beep_frequency_hz": round(dominant_frequency(beep, sample_rate), 2),
        "beep_frequency_ok": abs(
            dominant_frequency(beep, sample_rate) - spec["beep_frequency_hz"]
        )
        < 5.0,
        "beep_nonzero": bool(np.any(beep != 0.0)),
    }

"""ViMedCSS pairwise audio composition helper.

Renders the delivery form of the two-audio pairwise task as ONE physical WAV:

    speech_A  +  [pre_silence | 1000 Hz sine beep | post_silence]  +  speech_B

The separator recipe is the project's frozen recipe at
``resources/semantics/p1_audio_recipe.json`` (``vimd_p1_pair_beep_v1``). The
numerical behaviour here intentionally mirrors that frozen recipe exactly
(pre/post silence, 500 ms 1000 Hz beep, 10 ms linear fades, peak amplitude 0.2,
clip to [-1, 1]) without making the generic production generator depend on any
SAUVI task pipeline.

Deterministic: same inputs + same recipe always yield the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from src.common.config import ROOT

RECIPE_PATH = ROOT / "resources" / "semantics" / "p1_audio_recipe.json"


class VimedcssAudioError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}:{detail}" if detail else code)


def load_recipe(path: Path | str | None = None) -> dict:
    target = Path(path) if path is not None else RECIPE_PATH
    return json.loads(target.read_text(encoding="utf-8"))


def recipe_sha256(path: Path | str | None = None) -> str:
    target = Path(path) if path is not None else RECIPE_PATH
    return hashlib.sha256(target.read_bytes()).hexdigest()


def to_mono(wave: np.ndarray) -> np.ndarray:
    array = np.asarray(wave)
    if array.ndim == 1:
        return array.astype(np.float32)
    return array.mean(axis=1).astype(np.float32)


def resample(wave: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
    if sample_rate == target_rate:
        return wave.astype(np.float32)
    return resample_poly(wave, target_rate, sample_rate).astype(np.float32)


def generate_separator(recipe: dict, sample_rate: int) -> np.ndarray:
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
    return np.concatenate(
        [np.zeros(n_pre, np.float32), beep, np.zeros(n_post, np.float32)]
    ).astype(np.float32)


def separator_frame_counts(recipe: dict, sample_rate: int) -> dict[str, int]:
    spec = recipe["separator"]
    n_pre = round(sample_rate * spec["pre_silence_ms"] / 1000.0)
    n_beep = round(sample_rate * spec["beep_duration_ms"] / 1000.0)
    n_post = round(sample_rate * spec["post_silence_ms"] / 1000.0)
    return {"pre": n_pre, "beep": n_beep, "post": n_post, "total": n_pre + n_beep + n_post}


def compose_pair_with_beep(
    wave_a: np.ndarray,
    sample_rate_a: int,
    wave_b: np.ndarray,
    sample_rate_b: int,
    recipe: dict | None = None,
) -> tuple[np.ndarray, int]:
    recipe = recipe or load_recipe()
    target = int(recipe["sample_rate"])
    if sample_rate_a == sample_rate_b:
        target = sample_rate_a
    signal_a = resample(to_mono(wave_a), sample_rate_a, target)
    signal_b = resample(to_mono(wave_b), sample_rate_b, target)
    separator = generate_separator(recipe, target)
    composite = np.concatenate([signal_a, separator, signal_b]).astype(np.float32)
    clip = recipe["clip"]
    return np.clip(composite, clip["min"], clip["max"]).astype(np.float32), target


def _pcm_to_float(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (data - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise VimedcssAudioError("UNSUPPORTED_SAMPLE_WIDTH", str(sample_width))


def load_wav(path: Path | str) -> tuple[np.ndarray, int]:
    path = Path(path)
    if not path.exists():
        raise VimedcssAudioError("SOURCE_AUDIO_MISSING", str(path))
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.getnframes()
        raw = handle.readframes(frames)
    data = _pcm_to_float(raw, sample_width)
    if channels > 1:
        data = data.reshape(-1, channels)
    return to_mono(data), sample_rate


def _float_to_pcm16(wave: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(wave, dtype=np.float32), -1.0, 1.0)
    return np.round(clipped * 32767.0).astype("<i2")


def write_wav(path: Path | str, samples: np.ndarray, sample_rate: int) -> str:
    """Write a deterministic mono 16-bit PCM WAV; return the output SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = _float_to_pcm16(samples)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    spec = recipe["separator"]
    counts = separator_frame_counts(recipe, sample_rate)
    n_pre = counts["pre"]
    n_beep = counts["beep"]
    n_sep = counts["total"]
    beep = composite[frames_a + n_pre : frames_a + n_pre + n_beep]
    expected = frames_a + n_sep + frames_b
    peak = float(np.max(np.abs(composite))) if composite.size else 0.0
    return {
        "sample_rate": int(sample_rate),
        "mono": bool(np.asarray(composite).ndim == 1),
        "frames": int(composite.size),
        "frames_a": int(frames_a),
        "frames_separator": int(n_sep),
        "frames_b": int(frames_b),
        "expected_frames": int(expected),
        "duration_ok": int(composite.size) == expected,
        "separator_frames_nonzero": bool(n_sep > 0),
        "peak_amplitude": peak,
        "clipped": peak > 1.0,
        "beep_frequency_hz": round(dominant_frequency(beep, sample_rate), 2),
        "beep_frequency_ok": abs(
            dominant_frequency(beep, sample_rate) - spec["beep_frequency_hz"]
        )
        < 5.0,
        "beep_nonzero": bool(np.any(beep != 0.0)),
    }

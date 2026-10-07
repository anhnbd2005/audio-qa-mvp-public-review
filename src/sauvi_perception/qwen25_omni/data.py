"""Data loading, manifest validation, and MCQ rendering for Qwen2.5-Omni."""

from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from pathlib import Path
from typing import Any, Sequence

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
FORBIDDEN_TEST_TOKENS = ("test.jsonl", "/test/", "\\test\\", "test_manifest", "vimd_test")


class ManifestError(ValueError):
    """Raised when manifest validation fails."""


class PathEscapeError(ValueError):
    """Raised when audio path attempts directory traversal."""


class EmptyAudioError(ValueError):
    """Raised when audio decoding yields an empty audio signal."""


class ForbiddenTestAccessError(PermissionError):
    """Raised when a test manifest path is illegally accessed."""


def choice_letter(index: int) -> str:
    if index < 0 or index >= len(LETTERS):
        raise ValueError(f"choice index out of range: {index}")
    return LETTERS[index]


def forbid_test_access(path: str | Path) -> None:
    path_str = str(path).lower()
    for token in FORBIDDEN_TEST_TOKENS:
        if token in path_str:
            raise ForbiddenTestAccessError(f"Access to forbidden TEST path rejected: {path!r} contains {token!r}")


def resolve_audio_path(audio_root: str | Path, rel_path: str | Path) -> Path:
    root = Path(audio_root).resolve()
    rel = Path(rel_path)
    if rel.is_absolute():
        resolved = rel.resolve()
    else:
        resolved = (root / rel).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise PathEscapeError(f"Audio path {rel_path!r} escapes audio root {audio_root!r}")
    return resolved


def render_mcq_user_text(question: str, choices: Sequence[str]) -> str:
    """Render canonical MCQ semantic user text: <question>\n\nA. c0\nB. c1..."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    choice_list = list(choices)
    if len(choice_list) < 2:
        raise ValueError(f"need at least 2 choices, got {len(choice_list)}")
    for choice in choice_list:
        if not isinstance(choice, str) or not choice.strip():
            raise ValueError(f"choices must be non-empty strings, got {choice!r}")
    choice_lines = "\n".join(
        f"{choice_letter(index)}. {choice}" for index, choice in enumerate(choice_list)
    )
    return f"{question}\n\n{choice_lines}"


def permuted_choices(
    choices: Sequence[str],
    answer: str,
    base_seed: int,
    epoch: int,
    sample_id: str,
) -> list[str]:
    """Deterministically permute choice order for TRAIN epoch."""
    choice_list = list(choices)
    seed_str = f"{base_seed}_{epoch}_{sample_id}"
    rng = random.Random(seed_str)
    shuffled = list(choice_list)
    rng.shuffle(shuffled)
    return shuffled


def target_letter_for(permuted_choices_seq: Sequence[str], answer: str) -> str:
    """Derive letter target ('A'/'B'/...) for the given choice permutation and gold answer."""
    norm_gold = unicodedata.normalize("NFKC", answer).strip().casefold()
    matches = [
        index for index, choice in enumerate(permuted_choices_seq)
        if unicodedata.normalize("NFKC", choice).strip().casefold() == norm_gold
    ]
    if len(matches) != 1:
        raise ValueError(f"Answer {answer!r} must match exactly once in choices {permuted_choices_seq!r}, found {len(matches)} matches")
    return choice_letter(matches[0])


def validate_choice_row(row: dict[str, Any]) -> dict[str, Any]:
    sample_id = row.get("id") or row.get("sample_id")
    if not sample_id:
        raise ManifestError("Sample missing 'id'")

    audio_item = row.get("audio") or row.get("audio_path") or row.get("audio_file") or row.get("filename")
    if not audio_item:
        raise ManifestError(f"Sample {sample_id} missing audio path")
    if isinstance(audio_item, str):
        audio_list = [audio_item]
    elif isinstance(audio_item, (list, tuple)):
        audio_list = [str(a) for a in audio_item]
    else:
        raise ManifestError(f"Sample {sample_id} invalid audio field: {audio_item!r}")

    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ManifestError(f"Sample {sample_id} invalid question")

    choices = row.get("choices")
    if not isinstance(choices, (list, tuple)) or len(choices) < 2:
        raise ManifestError(f"Sample {sample_id} choices must be list of >= 2 items")

    answer = row.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        target_letter = row.get("target_letter")
        if target_letter and isinstance(target_letter, str):
            idx = ord(target_letter.upper()) - ord("A")
            if 0 <= idx < len(choices):
                answer = choices[idx]
    if not isinstance(answer, str) or not answer.strip():
        raise ManifestError(f"Sample {sample_id} missing valid answer text")

    norm_ans = unicodedata.normalize("NFKC", answer).strip().casefold()
    matches = [c for c in choices if unicodedata.normalize("NFKC", str(c)).strip().casefold() == norm_ans]
    if len(matches) != 1:
        raise ManifestError(f"Sample {sample_id}: answer {answer!r} must match choices {choices!r} exactly once, got {len(matches)}")

    return {
        "id": str(sample_id),
        "audio": audio_list,
        "question": str(question),
        "choices": [str(c) for c in choices],
        "answer": str(answer),
        "raw_row": row,
    }


def load_manifest(
    jsonl_path: str | Path,
    audio_root: str | Path | None = None,
    expected_rows: int | None = None,
    expected_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    path = Path(jsonl_path)
    forbid_test_access(path)

    if not path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    content_bytes = path.read_bytes()
    sha256_hash = hashlib.sha256(content_bytes).hexdigest()
    if expected_sha256 and sha256_hash.lower() != expected_sha256.lower():
        raise ManifestError(f"Manifest SHA256 mismatch for {path}: expected {expected_sha256}, got {sha256_hash}")

    rows: list[dict[str, Any]] = []
    seen_ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ManifestError(f"Invalid JSON at line {line_num} in {path}: {e}")
            validated = validate_choice_row(raw_row)
            if validated["id"] in seen_ids:
                raise ManifestError(f"Duplicate sample ID {validated['id']} at line {line_num} in {path}")
            seen_ids.add(validated["id"])

            if audio_root:
                for rel in validated["audio"]:
                    resolved = resolve_audio_path(audio_root, rel)
                    if not resolved.is_file():
                        raise FileNotFoundError(f"Audio file for sample {validated['id']} not found: {resolved}")

            rows.append(validated)

    if expected_rows is not None and len(rows) != expected_rows:
        raise ManifestError(f"Manifest row count mismatch for {path}: expected {expected_rows}, got {len(rows)}")

    return rows, sha256_hash


def load_audio_16k_mono(audio_path: str | Path, expected_sample_rate: int = 16000):
    import soundfile as sf
    import numpy as np

    path = Path(audio_path)
    data, sr = sf.read(path, dtype="float32")
    if sr != expected_sample_rate:
        raise ValueError(f"Audio sample rate mismatch for {path}: expected {expected_sample_rate} Hz, got {sr} Hz")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if len(data) == 0:
        raise EmptyAudioError(f"Audio file is empty: {path}")
    return data

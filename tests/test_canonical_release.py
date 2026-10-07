"""Permanent integrity contract for the canonical ViMD release."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "outputs" / "releases" / "vimd" / "final"
EXPECTED_MODEL_SHA = "6f2d4103d9d1211bba07409777a6f011b0c7fbd5dd01c6135c8c3235e5fe8346"
EXPECTED_INTERNAL_SHA = "e9bd8b5b29db15c967be983bdd45798de013b8b999281f6631d72dde4fa8f310"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (RELEASE / name).read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_release_jsonl_matches_validated_content_hashes():
    assert _sha(RELEASE / "qa_model_facing.jsonl") == EXPECTED_MODEL_SHA
    assert _sha(RELEASE / "qa_internal.jsonl") == EXPECTED_INTERNAL_SHA


def test_release_shape_and_audio_contract():
    model = _rows("qa_model_facing.jsonl")
    internal = _rows("qa_internal.jsonl")
    distribution = Counter(row["type_id"] for row in model)
    assert len(model) == len(internal) == 22000
    assert len(distribution) == 11
    assert set(distribution.values()) == {2000}
    assert sum(len(row["audio"]) for row in model) == 32000
    assert len({audio for row in model for audio in row["audio"]}) == 13344
    assert len({row["semantic_instance_id"] for row in internal}) == 22000


def test_release_content_audit_is_clean():
    audit = json.loads((RELEASE / "content_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    for key in (
        "semantic_mismatches",
        "gold_mismatches",
        "operator_errors",
        "a_b_errors",
        "leakage",
        "critical",
        "major",
        "minor",
    ):
        assert audit[key] == 0


def test_release_hash_files_match_exact_bytes():
    assert (RELEASE / "qa_model_facing.sha256").read_text().strip() == _sha(
        RELEASE / "qa_model_facing.jsonl"
    )
    assert (RELEASE / "qa_internal.sha256").read_text().strip() == _sha(
        RELEASE / "qa_internal.jsonl"
    )

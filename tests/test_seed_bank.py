"""Seed-bank correction regression: no fake-grounded fixed seeds.

The fine_label -> coarse_label metadata mapping was removed (appending
"audio" to its uses would have been fake grounding). The bank keeps only
genuinely audio-grounded examples and is NOT required to cover every
answer kind.
"""

import json

from src.autonomous_qa.language.question_style import build_question_style_prompt

SEED_PATH = "fewshots/question_style_seed.jsonl"


def _seeds(isolated_root):
    return [
        json.loads(line) for line in
        (isolated_root / SEED_PATH).read_text(
            encoding="utf-8").splitlines() if line.strip()
    ]


def test_no_fine_coarse_metadata_mapping_in_seeds(isolated_root):
    text = (isolated_root / SEED_PATH).read_text(encoding="utf-8")
    assert "fine_label" not in text
    assert "coarse_label" not in text
    assert "Fine-to-Coarse" not in text


def test_every_seed_lists_audio_in_uses(isolated_root):
    seeds = _seeds(isolated_root)
    assert len(seeds) >= 1
    for seed in seeds:
        assert "audio" in seed["output"]["uses"], seed


def test_every_seed_genuinely_needs_audio(isolated_root):
    # Each seed must describe audio evidence (a recording/speech the
    # model must listen to), not merely name-drop the audio field.
    for seed in _seeds(isolated_root):
        fields = seed["input"]["fields"]
        assert "audio" in fields, seed
        desc = fields["audio"].lower()
        assert any(w in desc for w in (
            "record", "speech", "listen", "utterance", "sound")), seed


def test_seed_bank_need_not_cover_every_answer_kind(isolated_root):
    kinds = {s["output"]["answer"]["kind"] for s in _seeds(isolated_root)}
    # No derived_field (or any other kind) is required to be present;
    # seeds are examples, not an operation taxonomy.
    for seed in _seeds(isolated_root):
        assert seed["output"]["answer"].get("kind")
        assert seed["output"]["uses"]


def test_style_prompt_has_anti_fake_grounding_rule(isolated_root):
    prompt = build_question_style_prompt("README", [], 1)
    assert "Do not add" in prompt
    assert "merely to satisfy" in prompt
    assert "without listening" in prompt.replace("\n", " ")
    assert "should NOT be returned" in prompt
    assert "AUDIO NECESSITY TEST" in prompt
    assert "DO NOT RETURN THIS PROPOSAL" in prompt

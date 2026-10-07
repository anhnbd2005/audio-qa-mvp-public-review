"""Reconciled Tests for VietMDD Generic Card-Driven Pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.autonomous_qa.core.dataset_profile import parse_source_card
from src.autonomous_qa.datasets.context_adapter import load_dataset_rows, load_schema
from src.autonomous_qa.production.render_preview import render_preview

ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = ROOT / "data_sources" / "vietmdd" / "dataset_card.md"
SCHEMA_PATH = ROOT / "data" / "vietmdd" / "schema.json"


def test_vietmdd_card_parsing():
    """Verify VietMDD canonical card parses into SourceCard."""
    card = parse_source_card(str(CARD_PATH))
    assert card.pretty_name == "VietMMD (Mispronunciation Detection and Diagnosis)"
    assert "audio" in card.fields
    assert "observed_transcription" in card.fields
    assert "original_text" in card.fields


def test_vietmdd_manifest_row_loading():
    """Verify generic context_adapter loads 612 normalized VietMDD rows."""
    rows = load_dataset_rows("vietmdd")
    assert len(rows) == 612
    assert "audio" in rows[0]
    assert "canonical" in rows[0]
    assert "transcript" in rows[0]


def test_vietmdd_schema_loading():
    """Verify generic context_adapter loads valid structural schema for VietMDD."""
    schema = load_schema("vietmdd")
    assert schema.get("dataset") == "vietmdd"
    assert "fields" in schema
    assert "canonical" in schema["fields"]


def test_vietmdd_generic_render_preview():
    """Verify render_preview generates preview items for VietMDD without LLM."""
    rows = load_dataset_rows("vietmdd")
    approved_types = [
        {
            "type_id": "vietmdd_canonical_direct",
            "proposition_description": "Direct phonetic transcription",
            "kept_templates": [{"text": "Hãy cho biết [KEY] của đoạn nói.", "template_id": "tpl_1"}],
            "answer": {"kind": "field_value", "key": "canonical"},
            "evaluation": "eligible"
        }
    ]
    key_realizations = {"canonical": "phiên âm chuẩn"}
    preview = render_preview(
        rows=rows,
        approved_types=approved_types,
        key_realizations=key_realizations,
        n=5,
        random_seed=42,
        dataset="vietmdd"
    )
    assert len(preview) == 5
    assert preview[0]["question_type_id"] == "vietmdd_canonical_direct"
    assert "question" in preview[0]

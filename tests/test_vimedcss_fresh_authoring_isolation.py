"""Characterization and isolation regression tests for fresh ViMedCSS authoring.

Verifies:
- ViMedCSS dataset config resolves canonical card and source
- Old ViMedCSS semantic catalog is isolated from pre-gate discovery prompts
- Fresh discovery prompts contain zero old canonical task IDs
- Composite discovery and language generation receive only current-run accepted semantics
- Authoring pipeline never mutates canonical resources
- CLI options (--offline, real_llm) behave correctly
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.common.config import ROOT
from src.autonomous_qa.authoring import pipeline as pl


def test_vimedcss_authoring_config_resolves_canonical_inputs():
    assert "vimedcss" in pl.DATASETS
    cfg = pl.DATASETS["vimedcss"]
    assert cfg.dataset_id == "vimedcss"
    assert cfg.card_path.exists()
    assert cfg.card_path == ROOT / "data_sources" / "vimedcss" / "dataset_card.md"
    assert cfg.materialized_path.exists()
    assert cfg.materialized_path == ROOT / "data_sources" / "vimedcss" / "source" / "train.jsonl"
    assert cfg.allowed_splits == ("train",)


def test_fresh_authoring_prompts_isolate_old_canonical_task_ids():
    card_text = (ROOT / "data_sources" / "vimedcss" / "dataset_card.md").read_text(encoding="utf-8")
    
    doc_prompt = pl.render_prompt(
        pl.load_prompt("documentation_interpretation"),
        DATASET_ID="vimedcss",
        SOURCE_REVISION="canonical",
        SOURCE_DOCUMENTATION=card_text,
        RESPONSE_SCHEMA=pl._schema_hint(),
    )
    
    old_task_ids = [
        "vimedcss_spoken_content_transcription",
        "vimedcss_cs_term_extraction",
        "vimedcss_topic_classification",
        "vimedcss_cs_term_presence",
        "vimedcss_pairwise_topic_same",
    ]
    
    for task_id in old_task_ids:
        assert task_id not in doc_prompt, f"Old task ID {task_id} found in documentation_interpretation prompt!"
        
    prim_prompt = pl.render_prompt(
        pl.load_prompt("primitive_semantic_discovery"),
        DATASET_ID="vimedcss",
        DOCUMENTATION_INTERPRETATION={"field_semantics": {"segment_text": "text"}},
        DATASET_PROFILE={"dataset_id": "vimedcss", "row_count": 11832, "fields": {}},
        RECONCILIATION={"claims": []},
        RESPONSE_SCHEMA=pl._schema_hint(),
    )
    
    for task_id in old_task_ids:
        assert task_id not in prim_prompt, f"Old task ID {task_id} found in primitive_semantic_discovery prompt!"


def test_composite_discovery_uses_only_current_run_accepted_primitives():
    fresh_primitives = [
        {"candidate_id": "fresh_prim_001", "operator": "DIRECT", "hidden_source_annotations": ["cs_terms_list"]}
    ]
    comp_prompt = pl.render_prompt(
        pl.load_prompt("composite_discovery"),
        DATASET_ID="vimedcss",
        ACCEPTED_PRIMITIVES=fresh_primitives,
        LEAKAGE_FINDINGS={},
        RESPONSE_SCHEMA=pl._schema_hint(),
    )
    assert "fresh_prim_001" in comp_prompt
    assert "vimedcss_spoken_content_transcription" not in comp_prompt


def test_language_generation_uses_only_current_run_accepted_semantics():
    fresh_primitives = [
        {"candidate_id": "fresh_prim_001", "operator": "DIRECT", "hidden_source_annotations": ["topic"]}
    ]
    lang_prompt = pl.render_prompt(
        pl.load_prompt("language_generation"),
        DATASET_ID="vimedcss",
        ACCEPTED_CONTRACTS=fresh_primitives,
        RESPONSE_SCHEMA=pl._schema_hint(),
    )
    assert "fresh_prim_001" in lang_prompt
    assert "vimedcss_topic_classification" not in lang_prompt


def test_authoring_does_not_mutate_canonical_resources(tmp_path):
    cfg = pl.DATASETS["vimedcss"]
    
    # Snapshot canonical catalog content
    catalog_before = cfg.canonical_catalog_path.read_text(encoding="utf-8")
    
    # Mock LLM response for all 6 stages
    def mock_llm_call(stage, prompt, temp, schema):
        if stage == "documentation_interpretation":
            parsed = {"field_semantics": {"segment_text": "text"}}
        elif stage == "primitive_semantic_discovery":
            parsed = {
                "candidates": [
                    {
                        "candidate_id": "cand_001",
                        "operator": "DIRECT",
                        "source_evidence": ["segment_text"],
                        "hidden_source_annotations": ["segment_text"],
                        "visible_inputs": [],
                        "gold_derivation_rule": "identity",
                        "answer_schema_proposal": {"kind": "field_value"},
                        "allowed_split": "train",
                    }
                ]
            }
        elif stage == "semantic_contract_critic":
            parsed = {"critique": "ok"}
        elif stage == "composite_discovery":
            parsed = {"composites": []}
        elif stage == "language_generation":
            parsed = {"entries": []}
        elif stage == "language_semantic_review":
            parsed = {"reviews": []}
        else:
            parsed = {}
        return {"parsed": parsed, "raw": json.dumps(parsed), "cache_hit": False, "provenance": {"stage": stage}}
    
    manifest = pl.run_authoring(
        "vimedcss",
        run_id="test_dry_run",
        real_llm=False,
        llm_call=mock_llm_call,
        out_root=tmp_path / "out",
    )
    
    catalog_after = cfg.canonical_catalog_path.read_text(encoding="utf-8")
    assert catalog_before == catalog_after, "Canonical catalog was mutated during authoring run!"
    assert manifest["dataset_id"] == "vimedcss"
    assert manifest["primitives_discovered"] == 1


def test_cli_offline_flag_validation(monkeypatch):
    import scripts.autonomous_qa.run_authoring as ra
    
    # Test --offline without --fixture-dir fails
    test_args = ["run_authoring.py", "vimedcss", "--offline"]
    monkeypatch.setattr("sys.argv", test_args)
    assert ra.main() == 2

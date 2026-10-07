"""Autonomous QA authoring pipeline: offline pipeline, gates, provenance, cache.

All LLM calls are faked here; pytest must never touch the live endpoint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.autonomous_qa.authoring import pipeline as ra


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


FIXTURES: dict[str, dict] = {
    "documentation_interpretation": {
        "dataset_id": "vietmdd",
        "dataset_purpose": "mispronunciation detection",
        "field_semantics": {
            "observed_transcription": {"documented_meaning": "perceived"},
            "original_text": {"documented_meaning": "canonical"},
        },
        "annotation_semantics": [],
        "split_intent": {},
        "evaluation_intent": [],
        "provenance_notes": [],
        "license_notes": [],
        "explicit_field_relations": [],
        "empirical_claims_to_verify": [],
        "ambiguities": [],
    },
    "primitive_semantic_discovery": {
        "candidates": [
            {
                "candidate_id": "cand_direct",
                "proposition": "transcribe the audio",
                "operator": "DIRECT",
                "audio_arity": 1,
                "visible_inputs": [],
                "hidden_source_annotations": ["observed_transcription_norm"],
                "answer_schema_proposal": {"kind": "field_value"},
                "gold_derivation_rule": "read observed_transcription_norm",
                "source_evidence": ["observed_transcription_norm documented"],
            },
            {
                "candidate_id": "cand_match_reference",
                "proposition": "does spoken content match reference",
                "operator": "TARGET_MATCH",
                "audio_arity": 1,
                "visible_inputs": ["original_text_norm"],
                "hidden_source_annotations": ["observed_transcription_norm"],
                "answer_schema_proposal": {"kind": "boolean"},
                "gold_derivation_rule": "compare observed_transcription_norm vs original_text_norm",
                "source_evidence": [
                    "original_text_norm and observed_transcription_norm"
                ],
            },
        ]
    },
    "semantic_contract_critic": {"critiques": []},
    "composite_discovery": {
        "composites": [
            {
                "composite_id": "comp_transcribe_verify",
                "components": ["cand_direct", "cand_match_reference"],
                "dependency_edges": [["observed_transcription", "matches_reference"]],
                # The LLM wrongly claims no leak; code must override.
                "output_component_visible_in_input": False,
            }
        ]
    },
    "language_generation": {
        "entries": [
            {
                "type_id": "cand_match_reference",
                "question_templates": [
                    "N\u1ed9i dung c\u00f3 kh\u1edbp {{reference}} kh\u00f4ng?"
                ],
                "answer_format": "true/false",
            }
        ]
    },
    "language_semantic_review": {"reviews": []},
}


def _fake_llm(stage, prompt, temperature, schema):
    payload = FIXTURES[stage]
    return json.dumps(payload), payload


def _write_tiny_plan(path: Path) -> None:
    # A retained composite (C3) whose visible context synthetically exposes a
    # requested field_value output, so the detector can be exercised offline.
    rows = [
        {
            "type_id": "vietmdd_composite_transcribe_pair_equality",
            "visible_context": ["m\u1eb9 \u1ea1 tr\u00ean s\u00e2n"],
            "output_gold": {
                "observed_transcription_a": "m\u1eb9 \u1ea1 tr\u00ean s\u00e2n",
                "observed_transcription_b": "m\u1ed9t hai b\u1ed1n",
                "same_observed_content": False,
            },
        },
        {
            "type_id": "vietmdd_composite_transcribe_pair_equality",
            "visible_context": ["m\u1ed9t hai ba"],
            "output_gold": {
                "observed_transcription_a": "s\u00e1u b\u1ea3y",
                "observed_transcription_b": "m\u1ed9t hai ba",
                "same_observed_content": False,
            },
        },
    ]
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
        encoding="utf-8",
    )


# --- structured parsing -----------------------------------------------------


def test_parse_after_fence_strips_json_fence():
    payload = '```json\n{"ok": true}\n```'
    assert ra._parse_after_fence(payload) == {"ok": True}
    assert ra._parse_after_fence("not json") is None


# --- cache + provenance -----------------------------------------------------


def test_call_rnd_llm_caches_and_records_safe_provenance(tmp_path):
    cfg = ra.DATASETS["vietmdd"]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    prompt = "hello {{X}}"
    schema = {"note": "x"}

    first = ra.call_rnd_llm(
        cfg=cfg,
        run_dir=run_dir,
        stage="documentation_interpretation",
        prompt=prompt,
        response_schema=schema,
        input_hashes={"documentation": _hash("doc")},
        real_llm=True,
        llm_call=_fake_llm,
        out_root=tmp_path,
    )
    assert first["cache_hit"] is False
    assert first["parsed"]["dataset_id"] == "vietmdd"
    assert first["provenance"]["prompt_id"] == "documentation_interpretation_v1"
    assert first["provenance"]["parser_status"] == "parsed"

    second = ra.call_rnd_llm(
        cfg=cfg,
        run_dir=run_dir,
        stage="documentation_interpretation",
        prompt=prompt,
        response_schema=schema,
        input_hashes={"documentation": _hash("doc")},
        real_llm=True,
        llm_call=_fake_llm,
        out_root=tmp_path,
    )
    assert second["cache_hit"] is True
    assert second["cache_key"] == first["cache_key"]

    # cache key contract: any changed input invalidates the cache
    changed = ra.call_rnd_llm(
        cfg=cfg,
        run_dir=run_dir,
        stage="documentation_interpretation",
        prompt=prompt,
        response_schema=schema,
        input_hashes={"documentation": _hash("doc2")},
        real_llm=True,
        llm_call=_fake_llm,
        out_root=tmp_path,
    )
    assert changed["cache_key"] != first["cache_key"]

    blob = json.dumps(first["provenance"]).lower()
    assert "authorization" not in blob and "secret" not in blob
    assert first["provenance"]["base_url_host"] == "localhost:20128"


# --- gates ------------------------------------------------------------------


def test_primitive_gate_allows_intended_verification_context():
    cfg = ra.DATASETS["vietmdd"]
    candidate = {
        "candidate_id": "c",
        "operator": "PAIRWISE_SELECTION",
        "hidden_source_annotations": ["observed_transcription_norm"],
        "visible_inputs": ["original_text_norm"],
        "gold_derivation_rule": "compare",
        "answer_schema_proposal": {"kind": "audio_index"},
    }
    profile = {
        "fields": {
            "observed_transcription_norm": {"distinct": 100},
            "original_text_norm": {"distinct": 100},
        }
    }
    result = ra.gate_primitive_candidate(candidate, profile, cfg, set())
    assert result["gates"]["NO_DIRECT_GOLD_VISIBLE"] == ra.GATE_PASS


def test_deterministic_composite_visibility_overrides_llm_claim():
    primitive = {
        "cand_match_reference": {
            "hidden_source_annotations": ["observed_transcription_norm"],
            "visible_inputs": ["original_text_norm"],
        }
    }
    rows = [
        {
            "observed_transcription_norm": "m\u1eb9 \u1ea1",
            "original_text_norm": "m\u1eb9 \u1ea1",
        },
        {"observed_transcription_norm": "a b", "original_text_norm": "c d"},
    ]
    composite = {
        "components": ["cand_match_reference"],
        "requested_outputs": ["observed_transcription"],
        "dependency_edges": [["a", "b"]],
        "output_component_visible_in_input": False,
    }
    findings = ra.deterministic_composite_visibility(composite, primitive, rows)
    assert findings and findings[0]["visible_field"] == "original_text_norm"
    gate = ra.gate_composite_candidate(composite, {"cand_match_reference"}, findings)
    assert gate["gates"]["OUTPUT_NOT_VISIBLE_IN_INPUT"] == ra.GATE_FAIL
    assert gate["verdict"] == ra.GATE_FAIL


# --- blind prompts ----------------------------------------------------------


def test_blind_prompt_excludes_canonical_catalog(tmp_path):
    tiny = tmp_path / "plan.jsonl"
    _write_tiny_plan(tiny)
    manifest = ra.run_authoring(
        "vietmdd",
        run_id="t_blind",
        real_llm=True,
        llm_call=_fake_llm,
        out_root=tmp_path,
        plan_path=tiny,
    )
    run_dir = Path(manifest["run_dir"])
    prompt = (
        run_dir / "llm" / "primitive_semantic_discovery" / "prompt.txt"
    ).read_text(encoding="utf-8")
    for canonical_id in (
        "vietmdd_direct_observed_text",
        "vietmdd_spoken_content_matches_reference",
        "vietmdd_composite_transcribe_match_reference",
    ):
        assert canonical_id not in prompt
    assert "semantic_catalog" not in prompt


def test_language_gate_flags_bad_entries():
    bad_empty = ra.gate_language_entry(
        {"type_id": "t", "question_templates": [], "answer_format": "x"}
    )
    assert bad_empty["verdict"] == ra.GATE_FAIL
    bad_dupe = ra.gate_language_entry(
        {"type_id": "t", "question_templates": ["a", "a"], "answer_format": "x"}
    )
    assert bad_dupe["verdict"] == ra.GATE_FAIL
    good = ra.gate_language_entry(
        {"type_id": "t", "question_templates": ["a", "b"], "answer_format": "x"}
    )
    assert good["verdict"] == ra.GATE_PASS


def test_production_does_not_depend_on_authoring():
    repo = Path(__file__).resolve().parents[1]
    for rel in (
        "src/autonomous_qa/compiler/canonical_resources.py",
        "src/autonomous_qa/certification/promotion_gate.py",
        "src/autonomous_qa/production/production_qa.py",
        "src/autonomous_qa/compiler/semantic_task.py",
    ):
        source = (repo / rel).read_text(encoding="utf-8")
        assert "autonomous_qa.authoring" not in source, rel
        assert "outputs/runs" not in source, rel


def test_promoted_catalog_retains_only_clean_composite():
    from src.autonomous_qa.authoring.pipeline import DATASETS
    from src.autonomous_qa.compiler.semantic_task import load_semantic_catalog

    catalog = load_semantic_catalog(DATASETS["vietmdd"].canonical_catalog_path)
    leaky = {
        "vietmdd_composite_transcribe_match_candidate",
        "vietmdd_composite_transcribe_match_reference",
        "vietmdd_composite_transcribe_pair_selection",
    }
    types = {t.type_id for t in catalog.tasks}
    assert len(types) == 6
    assert not (leaky & types)
    assert "vietmdd_composite_transcribe_pair_equality" in types


def test_full_offline_run_flags_vietmdd_composite_leak(tmp_path):
    tiny = tmp_path / "plan.jsonl"
    _write_tiny_plan(tiny)
    manifest = ra.run_authoring(
        "vietmdd",
        run_id="t_run",
        real_llm=True,
        llm_call=_fake_llm,
        out_root=tmp_path,
        plan_path=tiny,
    )
    assert manifest["total_real_llm_calls"] == 6
    assert (
        "vietmdd_composite_transcribe_pair_equality"
        in manifest["component_leakage_flagged"]
    )
    assert manifest["readiness"]["status"] == "REVIEW_REQUIRED"
    # composite gate must FAIL on the deterministic visibility finding
    gates = json.loads(
        (Path(manifest["run_dir"]) / "gates" / "composite_gates.json").read_text(
            encoding="utf-8"
        )
    )
    assert gates["composites"][0]["verdict"] == ra.GATE_FAIL
    assert gates["composites"][0]["deterministic_visibility"]

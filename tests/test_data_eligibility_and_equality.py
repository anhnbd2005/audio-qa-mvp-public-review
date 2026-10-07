"""Unit and regression tests for generic data eligibility, missing-value normalization,
evaluation filtering, entity-scoped equality families, and subject-attribute compatibility.
Requirements: §AF through §AO.
"""

import copy
import json
import pytest
from pathlib import Path

from src.autonomous_qa.core.validity import (
    ENTITY_SCOPE_SPEAKER,
    ENTITY_SCOPE_UTTERANCE,
    ENTITY_SCOPES,
    EVALUATION_ELIGIBLE,
    EVALUATION_DISCOVERY_ONLY,
    EVALUATION_POLICIES,
    extract_entity_scopes,
    extract_source_missing_values,
    extract_evaluation_policies,
    normalize_row_missing_values,
    required_answer_fields,
    is_evaluation_eligible_type,
    template_family_signature,
)
from src.autonomous_qa.language.family_templates import (
    FAMILY_EQUALITY_SEMANTIC_SPEAKER,
    FAMILY_EQUALITY_SEMANTIC_UTTERANCE,
)
from src.autonomous_qa.production.render_preview import _gold_answer, _find_eligible_row, _find_pair_for_stream
from src.autonomous_qa.datasets.context_adapter import filter_evaluation_eligible_types
from src.autonomous_qa.compiler.pipeline_runner import (
    extract_representative_values,
    load_schema,
)


# ---------------------------------------------------------------------------
# §AF: Generic Missing-Value Normalization
# ---------------------------------------------------------------------------

def test_generic_missing_value_normalization_synthetic():
    """Synthetic schema with arbitrary missing markers normalizes values to None."""
    synth_schema = {
        "fields": {
            "attribute_x": {
                "role": "semantic",
                "entity_scope": "utterance",
                "source_missing_values": ["UNKNOWN_MARKER", "N/A", "missing"],
            },
            "valid_col": {
                "role": "semantic",
                "entity_scope": "speaker",
            },
        }
    }
    missing_map = extract_source_missing_values(synth_schema)
    assert missing_map == {"attribute_x": ["UNKNOWN_MARKER", "N/A", "missing"]}

    row1 = {"attribute_x": "UNKNOWN_MARKER", "valid_col": "normal_val"}
    row2 = {"attribute_x": "  n/a  ", "valid_col": "normal_val"}
    row3 = {"attribute_x": "real_val", "valid_col": "normal_val"}

    norm1 = normalize_row_missing_values(row1, missing_map)
    norm2 = normalize_row_missing_values(row2, missing_map)
    norm3 = normalize_row_missing_values(row3, missing_map)

    assert norm1["attribute_x"] is None
    assert norm1["valid_col"] == "normal_val"
    assert norm2["attribute_x"] is None
    assert norm3["attribute_x"] == "real_val"



# ---------------------------------------------------------------------------
# §AG: Direct QA Row Ineligibility on Null Fields
# ---------------------------------------------------------------------------

def test_direct_null_skip():
    """field_value(field) requires field to be non-null; eligible row finder skips nulls."""
    qtype = {
        "type_id": "QT_attr",
        "answer": {"kind": "field_value", "key": "attribute_x"},
    }
    req = required_answer_fields(qtype)
    assert req == ["attribute_x"]

    rows = [
        {"audio": "a1.wav", "attribute_x": None},
        {"audio": "a2.wav", "attribute_x": ""},
        {"audio": "a3.wav", "attribute_x": "ValidValue"},
    ]
    # Row with None cannot compute gold answer
    with pytest.raises(ValueError, match="render_missing_answer_field"):
        _gold_answer(qtype, rows[0])

    # Finder skips null / empty
    eligible = _find_eligible_row(rows, qtype)
    assert eligible is not None
    assert eligible["audio"] == "a3.wav"
    assert eligible["attribute_x"] == "ValidValue"


# ---------------------------------------------------------------------------
# §AH: Multi-Audio Equality Ineligibility on Missing Values
# ---------------------------------------------------------------------------

def test_equality_null_ineligible():
    """Two rows where both have None for equality key MUST NEVER form a valid pair."""
    qtype = {
        "type_id": "QT_eq_age",
        "answer": {"kind": "equality", "keys": ["speaker_age_1", "speaker_age_2"]},
    }
    req = required_answer_fields(qtype)
    assert req == ["speaker_age"]

    r_none1 = {"audio": "a1.wav", "speaker_age": None, "speaker_id": "spk_1"}
    r_none2 = {"audio": "a2.wav", "speaker_age": None, "speaker_id": "spk_2"}
    r_valid1 = {"audio": "a3.wav", "speaker_age": "20s", "speaker_id": "spk_3"}
    r_valid2 = {"audio": "a4.wav", "speaker_age": "20s", "speaker_id": "spk_4"}
    r_valid3 = {"audio": "a5.wav", "speaker_age": "30s", "speaker_id": "spk_5"}

    # Two Nones must raise ValueError and never produce 'Có'
    with pytest.raises(ValueError, match="render_missing_answer_field"):
        _gold_answer(qtype, r_none1, r_none2)

    rows = [r_none1, r_none2, r_valid1, r_valid2, r_valid3]
    pair_idx = _find_pair_for_stream(rows, qtype, want_match=True)
    assert pair_idx is not None
    i, j = pair_idx
    ans, _ = _gold_answer(qtype, rows[i], rows[j])
    assert ans == "Có"
    # Ensure neither row has None
    assert rows[i]["speaker_age"] is not None
    assert rows[j]["speaker_age"] is not None
    assert rows[i]["speaker_age"] == rows[j]["speaker_age"]


# ---------------------------------------------------------------------------
# §AI: Evaluation-Eligibility Schema Filtering
# ---------------------------------------------------------------------------

def test_evaluation_policy_filtering():
    """Types targeting fields with evaluation: 'discovery_only' are excluded from evaluation pool."""
    synth_schema = {
        "fields": {
            "eligible_field": {
                "role": "semantic",
                "entity_scope": "speaker",
                "evaluation": "eligible",
            },
            "discovery_field": {
                "role": "semantic",
                "entity_scope": "speaker",
                "evaluation": "discovery_only",
            },
        }
    }
    eval_map = extract_evaluation_policies(synth_schema)
    assert eval_map["eligible_field"] == EVALUATION_ELIGIBLE
    assert eval_map["discovery_field"] == EVALUATION_DISCOVERY_ONLY

    t_eligible = {
        "type_id": "QT_el",
        "answer": {"kind": "field_value", "key": "eligible_field"},
    }
    t_discovery = {
        "type_id": "QT_disc",
        "answer": {"kind": "field_value", "key": "discovery_field"},
    }

    assert is_evaluation_eligible_type(t_eligible, eval_map) is True
    assert is_evaluation_eligible_type(t_discovery, eval_map) is False

    pool = [t_eligible, t_discovery]
    eval_pool = filter_evaluation_eligible_types(pool, eval_map)
    assert len(eval_pool) == 1
    assert eval_pool[0]["type_id"] == "QT_el"



# ---------------------------------------------------------------------------
# §AJ: Entity-Scoped Equality Families
# ---------------------------------------------------------------------------

def test_entity_scoped_families():
    """Semantic equality types split into speaker vs utterance operation families."""
    synth_roles = {"spk_col": "semantic", "utt_col": "semantic"}
    synth_fields = frozenset(synth_roles.keys())
    synth_scopes = {"spk_col": ENTITY_SCOPE_SPEAKER, "utt_col": ENTITY_SCOPE_UTTERANCE}

    t_spk = {"id": "T_spk", "uses": ["audio", "spk_col"], "answer": {"kind": "equality", "keys": ["spk_col_1", "spk_col_2"]}}
    t_utt = {"id": "T_utt", "uses": ["audio", "utt_col"], "answer": {"kind": "equality", "keys": ["utt_col_1", "utt_col_2"]}}

    assert template_family_signature(t_spk, synth_roles, synth_fields, synth_scopes) == FAMILY_EQUALITY_SEMANTIC_SPEAKER
    assert template_family_signature(t_utt, synth_roles, synth_fields, synth_scopes) == FAMILY_EQUALITY_SEMANTIC_UTTERANCE


# ---------------------------------------------------------------------------
# §AK: ViMD Equality Regression
# ---------------------------------------------------------------------------

def test_vimd_equality_family_regression():
    """ViMD fields gender, region, province map to speaker equality; speakerID maps to None."""
    schema = load_schema("vimd")
    roles = {f: meta.get("role", "semantic") for f, meta in schema.get("fields", {}).items()}
    scopes = extract_entity_scopes(schema)
    fields = frozenset(roles.keys())

    for f in ("gender", "region", "province"):
        t = {"id": f"V_{f}", "uses": ["audio", f], "answer": {"kind": "equality", "keys": [f"{f}_1", f"{f}_2"]}}
        assert template_family_signature(t, roles, fields, scopes) == FAMILY_EQUALITY_SEMANTIC_SPEAKER

    t_spk = {"id": "V_spk", "uses": ["audio", "speakerID"], "answer": {"kind": "equality", "keys": ["speakerID_1", "speakerID_2"]}}
    assert template_family_signature(t_spk, roles, fields, scopes) is None



# ---------------------------------------------------------------------------
# §AM: Fail-Safe Scope Handling
# ---------------------------------------------------------------------------

def test_fail_safe_family_behavior():
    """Missing or unrecognized entity_scope safely produces None (no crash, no base reuse)."""
    roles = {"f1": "semantic"}
    fields = frozenset(["f1"])

    t = {"id": "T1", "uses": ["audio", "f1"], "answer": {"kind": "equality", "keys": ["f1_1", "f1_2"]}}

    # Missing scopes dict
    assert template_family_signature(t, roles, fields, {}) is None
    assert template_family_signature(t, roles, fields, None) is None

    # Unrecognized scope
    assert template_family_signature(t, roles, fields, {"f1": "unrecognized_scope"}) is None


# ---------------------------------------------------------------------------
# §AN & §AO: Representative Values & Context Grounding
# ---------------------------------------------------------------------------

def test_representative_values_extraction():
    """Extract small deterministic non-null categorical values for prompt context (§W, §X)."""
    synth_schema = {
        "fields": {
            "cat_semantic": {"role": "semantic", "is_categorical": True},
            "num_semantic": {"role": "semantic", "is_categorical": False},
            "hidden_id": {"role": "hidden_identifier", "is_categorical": True},
        }
    }
    rows = [
        {"cat_semantic": "val_a", "num_semantic": "100", "hidden_id": "id1"},
        {"cat_semantic": "val_b", "num_semantic": "200", "hidden_id": "id2"},
        {"cat_semantic": "val_a", "num_semantic": "300", "hidden_id": "id3"},
        {"cat_semantic": None, "num_semantic": "400", "hidden_id": "id4"},
        {"cat_semantic": "val_c", "num_semantic": "500", "hidden_id": "id5"},
    ]
    rep = extract_representative_values(rows, synth_schema, max_values_per_field=2)
    assert "cat_semantic" in rep
    assert len(rep["cat_semantic"]) == 2
    assert rep["cat_semantic"] == ["val_a", "val_b"]
    assert "num_semantic" not in rep
    assert "hidden_id" not in rep


def test_subject_attribute_compatibility_and_exact_equality_in_prompts():
    """Verify prompts enforce subject-attribute compatibility and exact equality (§AN, §O)."""
    from src.common.io import load_prompt
    tpl_prompt = load_prompt("question_template")
    quality_prompt = load_prompt("quality")

    # Template prompt enforces entity scope distinction
    assert "Speaker-scoped family" in tpl_prompt
    assert "Utterance-scoped family" in tpl_prompt
    assert "hai người nói" in tpl_prompt
    assert "hai phát ngôn" in tpl_prompt

    # Quality prompt enforces subject-attribute compatibility
    assert "SUBJECT" in quality_prompt and "ATTRIBUTE COMPATIBILITY" in quality_prompt
    assert "STRICT EQUALITY" in quality_prompt
    assert "KEY NATURALNESS" in quality_prompt


def test_contextual_representative_values_in_prompt():
    """Verify representative sample values appear in prompt as context only (§AO, §W, §X)."""
    from src.autonomous_qa.language.question_template import build_question_template_prompt
    rep = {"scenario": ["calendar", "weather", "music"]}
    prompt = build_question_template_prompt(
        readme="Test readme",
        round_types=[{"id": "QT_test", "name": "Test", "uses": ["audio"]}],
        template_bank=[],
        round_idx=1,
        representative_values=rep,
    )
    assert "REPRESENTATIVE SAMPLE VALUES (CONTEXT ONLY)" in prompt
    assert '"scenario"' in prompt
    assert '"calendar"' in prompt
    assert "[VALUE] is FORBIDDEN in question text" in prompt


# ---------------------------------------------------------------------------
# Safety Invariants: Null sampling, Evaluation Fail-Closed, Strict Parsing
# ---------------------------------------------------------------------------

def test_null_null_equality_cannot_be_sampled():
    """When rows have null for equality key, they cannot be sampled as pairs."""
    qtype = {
        "type_id": "QT_eq_null",
        "answer": {"kind": "equality", "keys": ["attr_x_1", "attr_x_2"]},
    }
    all_null_rows = [
        {"audio": "a1.wav", "attr_x": None},
        {"audio": "a2.wav", "attr_x": None},
        {"audio": "a3.wav", "attr_x": ""},
    ]
    # No pair can be found (neither positive nor negative)
    pair_pos = _find_pair_for_stream(all_null_rows, qtype, want_match=True)
    pair_neg = _find_pair_for_stream(all_null_rows, qtype, want_match=False)
    pair_any = _find_pair_for_stream(all_null_rows, qtype, want_match=None)
    assert pair_pos is None
    assert pair_neg is None
    assert pair_any is None

    # Calling _gold_answer directly raises controlled ValueError
    with pytest.raises(ValueError, match="render_missing_answer_field:attr_x"):
        _gold_answer(qtype, all_null_rows[0], all_null_rows[1])


def test_missing_evaluation_policy_fail_closed():
    """Missing or omitted evaluation policy MUST fail closed (never silently eligible)."""
    t_valid = {
        "type_id": "QT_valid",
        "answer": {"kind": "field_value", "key": "col_a"},
    }

    # Case 1: Empty or None evaluation policies dictionary
    assert is_evaluation_eligible_type(t_valid, {}) is False
    assert is_evaluation_eligible_type(t_valid, None) is False

    # Case 2: Field missing from evaluation policies dictionary
    assert is_evaluation_eligible_type(t_valid, {"other_col": "eligible"}) is False

    # Case 3: Field explicitly eligible
    assert is_evaluation_eligible_type(t_valid, {"col_a": "eligible"}) is True

    # Case 4: Field explicitly discovery_only
    assert is_evaluation_eligible_type(t_valid, {"col_a": "discovery_only"}) is False

    # Case 5: Equality type where base field lacks evaluation policy
    t_eq = {
        "type_id": "QT_eq",
        "answer": {"kind": "equality", "keys": ["col_a_1", "col_a_2"]},
    }
    assert is_evaluation_eligible_type(t_eq, {}) is False
    assert is_evaluation_eligible_type(t_eq, {"col_b": "eligible"}) is False
    assert is_evaluation_eligible_type(t_eq, {"col_a": "eligible"}) is True


def test_malformed_equality_cannot_infer_required_field():
    """Malformed equality answer keys MUST fail closed without string splitting."""
    # 1. Suffixes are not _1 and _2
    t_bad_suffix = {"answer": {"kind": "equality", "keys": ["attr_x_foo", "attr_x_bar"]}}
    assert required_answer_fields(t_bad_suffix) == []

    # 2. Suffixes are swapped (_2, _1)
    t_swapped = {"answer": {"kind": "equality", "keys": ["attr_x_2", "attr_x_1"]}}
    assert required_answer_fields(t_swapped) == []

    # 3. Base fields do not match
    t_mismatch = {"answer": {"kind": "equality", "keys": ["attr_x_1", "attr_y_2"]}}
    assert required_answer_fields(t_mismatch) == []

    # 4. Only one key
    t_single = {"answer": {"kind": "equality", "keys": ["attr_x_1"]}}
    assert required_answer_fields(t_single) == []

    # 5. Raw key without instance suffixes
    t_no_suffix = {"answer": {"kind": "equality", "keys": ["attr_x", "attr_x"]}}
    assert required_answer_fields(t_no_suffix) == []

    # 6. Single key field instead of keys list
    t_legacy_key = {"answer": {"kind": "equality", "key": "attr_x"}}
    assert required_answer_fields(t_legacy_key) == []

    # Evaluation eligibility on malformed equality is always False
    policies = {"attr_x": "eligible", "attr_y": "eligible"}
    for t in (t_bad_suffix, t_swapped, t_mismatch, t_single, t_no_suffix, t_legacy_key):
        assert is_evaluation_eligible_type(t, policies) is False

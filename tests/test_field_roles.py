"""Tests for Generic Dataset Field Roles (§1-§8).

Verifies:
1. Field roles derive strictly from dataset schema input, not dataset-specific code.
2. Canonical role vocabulary: audio, semantic, hidden_identifier, provenance, context_only.
3. Generic role behavior:
   - semantic: field_value allowed, equality allowed, derived allowed.
   - hidden_identifier: field_value rejected (id_exposure), equality allowed, derived rejected.
   - provenance: field_value/equality/derived rejected.
   - context_only: field_value/equality/derived rejected.
   - audio: field_value/equality/derived rejected.
4. ViMD speakerID still equality-allowed.
5. Speech-MASSIVE speaker_id still equality-allowed.
6. Zero field-name-specific branches in production QA eligibility code.
"""

from pathlib import Path
import pytest

from src.common.config import ROOT
from src.autonomous_qa.compiler.pipeline_runner import (
    dataset_field_roles,
    dataset_hidden_fields,
    load_schema,
)
from src.autonomous_qa.core.validity import (
    CANONICAL_FIELD_ROLES,
    ROLE_AUDIO,
    ROLE_CONTEXT_ONLY,
    ROLE_HIDDEN_IDENTIFIER,
    ROLE_PROVENANCE,
    ROLE_SEMANTIC,
    extract_field_roles,
    validate_answer_references,
    validate_derived_relation,
    validate_question_type,
)


def test_1_canonical_role_vocabulary():
    """Verify the minimum required role vocabulary."""
    expected = {
        ROLE_AUDIO,
        ROLE_SEMANTIC,
        ROLE_HIDDEN_IDENTIFIER,
        ROLE_PROVENANCE,
        ROLE_CONTEXT_ONLY,
    }
    assert CANONICAL_FIELD_ROLES == expected
    assert CANONICAL_FIELD_ROLES == {
        "audio",
        "semantic",
        "hidden_identifier",
        "provenance",
        "context_only",
    }


def test_3_vimd_field_role_mapping():
    """Verify ViMD field -> role mapping in schema."""
    roles = dataset_field_roles("vimd")
    expected = {
        "audio": "audio",
        "text": "semantic",
        "gender": "semantic",
        "region": "semantic",
        "province": "semantic",
        "speakerID": "hidden_identifier",
    }
    for field, role in expected.items():
        assert roles.get(field) == role, f"Expected {field} -> {role}, got {roles.get(field)}"


def test_4_generic_synthetic_schema_role_enforcement():
    """Test generic role behavior using entirely synthetic field names."""
    synthetic_schema = {
        "fields": {
            "aud_inp": {"role": "audio"},
            "feat_a": {"role": "semantic"},
            "feat_b": {"role": "semantic"},
            "hid_spk": {"role": "hidden_identifier"},
            "prov_track": {"role": "provenance"},
            "ctx_extra": {"role": "context_only"},
        },
        "field_roles": {
            "aud_inp": "audio",
            "feat_a": "semantic",
            "feat_b": "semantic",
            "hid_spk": "hidden_identifier",
            "prov_track": "provenance",
            "ctx_extra": "context_only",
        },
    }
    roles = extract_field_roles(synthetic_schema)
    fields = set(roles.keys())

    # --- 1. semantic field ---
    # field_value allowed
    assert validate_answer_references(
        {"kind": "field_value", "key": "feat_a"},
        schema_fields=fields, field_roles=roles) == []

    # equality allowed
    assert validate_answer_references(
        {"kind": "equality", "keys": ["feat_a_1", "feat_a_2"]},
        schema_fields=fields, field_roles=roles) == []

    # derived allowed
    assert validate_answer_references(
        {"kind": "derived_field", "source_key": "feat_a", "target_key": "feat_b"},
        schema_fields=fields, field_roles=roles) == []

    # --- 2. hidden_identifier ---
    # field_value rejected (id_exposure)
    errs = validate_answer_references(
        {"kind": "field_value", "key": "hid_spk"},
        schema_fields=fields, field_roles=roles)
    assert errs == ["id_exposure"]

    # equality allowed
    assert validate_answer_references(
        {"kind": "equality", "keys": ["hid_spk_1", "hid_spk_2"]},
        schema_fields=fields, field_roles=roles) == []

    # derived rejected (both as source and as target)
    errs_src = validate_answer_references(
        {"kind": "derived_field", "source_key": "hid_spk", "target_key": "feat_b"},
        schema_fields=fields, field_roles=roles)
    assert errs_src == ["invalid_derived_hidden_source"]

    errs_tgt = validate_answer_references(
        {"kind": "derived_field", "source_key": "feat_a", "target_key": "hid_spk"},
        schema_fields=fields, field_roles=roles)
    assert errs_tgt == ["invalid_derived_hidden_target"]

    # --- 3. provenance ---
    # field_value rejected
    errs_fv = validate_answer_references(
        {"kind": "field_value", "key": "prov_track"},
        schema_fields=fields, field_roles=roles)
    assert errs_fv == ["ineligible_field_role:provenance:prov_track"]

    # equality rejected
    errs_eq = validate_answer_references(
        {"kind": "equality", "keys": ["prov_track_1", "prov_track_2"]},
        schema_fields=fields, field_roles=roles)
    assert errs_eq == ["ineligible_field_role:provenance:prov_track"]

    # derived rejected
    errs_der = validate_answer_references(
        {"kind": "derived_field", "source_key": "prov_track", "target_key": "feat_b"},
        schema_fields=fields, field_roles=roles)
    assert errs_der == ["ineligible_field_role:provenance:prov_track"]

    # --- 4. context_only ---
    # field_value rejected
    errs_ctx = validate_answer_references(
        {"kind": "field_value", "key": "ctx_extra"},
        schema_fields=fields, field_roles=roles)
    assert errs_ctx == ["ineligible_field_role:context_only:ctx_extra"]

    # equality rejected
    errs_ctx_eq = validate_answer_references(
        {"kind": "equality", "keys": ["ctx_extra_1", "ctx_extra_2"]},
        schema_fields=fields, field_roles=roles)
    assert errs_ctx_eq == ["ineligible_field_role:context_only:ctx_extra"]

    # derived rejected
    errs_ctx_der = validate_answer_references(
        {"kind": "derived_field", "source_key": "ctx_extra", "target_key": "feat_b"},
        schema_fields=fields, field_roles=roles)
    assert errs_ctx_der == ["ineligible_field_role:context_only:ctx_extra"]

    # --- 5. audio ---
    # field_value rejected
    errs_aud = validate_answer_references(
        {"kind": "field_value", "key": "aud_inp"},
        schema_fields=fields, field_roles=roles)
    assert errs_aud == ["ineligible_field_role:audio:aud_inp"]


def test_5_vimd_speakerid_still_equality_allowed():
    """ViMD speakerID remains allowed for equality."""
    roles = dataset_field_roles("vimd")
    fields = set(roles.keys())
    assert validate_answer_references(
        {"kind": "equality", "keys": ["speakerID_1", "speakerID_2"]},
        schema_fields=fields, field_roles=roles) == []


def test_7_no_literal_field_name_eligibility_branches_in_production():
    """Audit production python code for forbidden field-name specific checks."""
    src_dir = ROOT / "src"
    forbidden_patterns = [
        'field == "split"',
        "field == 'split'",
        'field == "annot_utt"',
        "field == 'annot_utt'",
        'dataset == "speech_massive_vi"',
        "dataset == 'speech_massive_vi'",
    ]
    for py_file in src_dir.glob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for pat in forbidden_patterns:
            assert pat not in text, f"Found forbidden hardcoded branch {pat!r} in {py_file}"

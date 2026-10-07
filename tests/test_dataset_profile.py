"""Tests for the dataset profiling stage (source card -> canonical profile).

The profile is the single canonical source: schema.json and
metadata_context.json are exported from it deterministically. These tests
pin the contract that keeps the generated profile tied to the source card:
exact raw field names, source dtypes, canonical vocabularies, internal
consistency, no invented values, no unsupported label mappings.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.common.config import ROOT
from src.autonomous_qa.core.dataset_profile import (
    PROFILE_SCHEMA_MARKER,
    build_prompt,
    canonical_json,
    export_metadata_context,
    export_schema,
    load_mock_profile,
    parse_source_card,
    review_flags,
    run_profile,
    validate_profile,
)

CARD_PATH = ROOT / "data_sources" / "vimd" / "dataset_card.md"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "dataset_profile_vimd.json"

EXPECTED_RAW_FIELDS = {
    "region",
    "province_code",
    "province_name",
    "filename",
    "text",
    "speakerID",
    "gender",
    "audio",
}

EXPECTED_DTYPES = {
    "region": "string",
    "province_code": "int64",
    "province_name": "string",
    "filename": "string",
    "text": "string",
    "speakerID": "string",
    "gender": "int64",
    "audio": "audio",
}


@pytest.fixture(scope="module")
def card():
    return parse_source_card(CARD_PATH)


@pytest.fixture()
def profile() -> dict:
    return load_mock_profile("vimd", FIXTURE_PATH)


def _sync_mirrors(profile: dict) -> dict:
    """Recompute every mirror map from `fields` (keeps profiles coherent)."""
    fields = profile["fields"]
    profile["field_roles"] = {n: m["role"] for n, m in fields.items()}
    profile["entity_scopes"] = {
        n: m.get("entity_scope") for n, m in fields.items()}
    profile["field_evaluations"] = {
        n: m["evaluation"] for n, m in fields.items()}
    profile["context_visibilities"] = {
        n: m["context_visibility"] for n, m in fields.items()}
    profile["semantic_fields"] = sorted(
        n for n, m in fields.items() if m["role"] == "semantic")
    profile["hidden_fields"] = sorted(
        n for n, m in fields.items() if m["role"] == "hidden_identifier")
    return profile


def test_1_source_card_yields_valid_mock_profile(card, profile):
    """The shipped mock profile validates against the real source card."""
    report = validate_profile(profile, card)
    assert report["ok"] is True, report["errors"]
    assert report["errors"] == []
    assert set(card.fields) == EXPECTED_RAW_FIELDS
    assert card.fields == EXPECTED_DTYPES
    assert review_flags(profile, card) == []


def test_2_exactly_eight_raw_fields_preserved(card, profile):
    assert set(profile["fields"]) == EXPECTED_RAW_FIELDS
    for name, dtype in EXPECTED_DTYPES.items():
        assert profile["fields"][name]["type"] == dtype


def test_3_no_raw_field_renamed(card, profile):
    """Cosmetic renames (speakerID -> speaker_id) are always rejected."""
    profile = _sync_mirrors(profile)
    profile["fields"]["speaker_id"] = profile["fields"].pop("speakerID")
    profile = _sync_mirrors(profile)

    report = validate_profile(profile, card)
    assert report["ok"] is False
    assert report["checks"]["field_set"]["renamed"] == {
        "speaker_id": "speakerID"}
    assert "field_set:renamed:speaker_id->speakerID" in report["errors"]


def test_4_unknown_or_invented_field_rejected(card, profile):
    profile = copy.deepcopy(profile)
    template = profile["fields"]["region"]
    profile["fields"]["dialect_cluster"] = copy.deepcopy(template)

    report = validate_profile(profile, card)
    assert report["ok"] is False
    assert report["checks"]["field_set"]["invented"] == ["dialect_cluster"]
    assert "field_set:invented:dialect_cluster" in report["errors"]


def test_5_missing_field_rejected(card, profile):
    profile = copy.deepcopy(profile)
    profile["fields"].pop("text")

    report = validate_profile(profile, card)
    assert report["ok"] is False
    assert report["checks"]["field_set"]["missing"] == ["text"]


def test_6_dtype_mismatch_rejected(card, profile):
    profile = copy.deepcopy(profile)
    profile["fields"]["gender"]["type"] = "string"

    report = validate_profile(profile, card)
    assert report["checks"]["dtype_agreement"]["ok"] is False
    assert report["checks"]["dtype_agreement"]["mismatches"] == {
        "gender": {"expected": "int64", "actual": "string"}}
    assert report["ok"] is False


def test_7_invalid_role_rejected(card, profile):
    profile = copy.deepcopy(profile)
    profile["fields"]["region"]["role"] = "identifier"

    report = validate_profile(profile, card)
    assert report["checks"]["vocabulary"]["ok"] is False
    assert any("region:role=" in e for e in report["errors"])
    assert report["ok"] is False


def test_8_invalid_entity_scope_rejected(card, profile):
    profile = copy.deepcopy(profile)
    profile["fields"]["region"]["entity_scope"] = "province"

    report = validate_profile(profile, card)
    assert report["checks"]["vocabulary"]["ok"] is False
    assert any("region:entity_scope=" in e for e in report["errors"])
    assert report["ok"] is False


def test_9_invalid_evaluation_rejected(card, profile):
    profile = copy.deepcopy(profile)
    profile["fields"]["text"]["evaluation"] = "required"

    report = validate_profile(profile, card)
    assert report["checks"]["vocabulary"]["ok"] is False
    assert any("text:evaluation=" in e for e in report["errors"])
    assert report["ok"] is False


def test_10_invalid_context_visibility_rejected(card, profile):
    profile = copy.deepcopy(profile)
    profile["fields"]["text"]["context_visibility"] = "public"

    report = validate_profile(profile, card)
    assert report["checks"]["vocabulary"]["ok"] is False
    assert report["ok"] is False


def test_11_inconsistent_semantic_fields_rejected(card, profile):
    profile = copy.deepcopy(profile)
    profile["semantic_fields"].remove("text")

    report = validate_profile(profile, card)
    assert report["checks"]["consistency"]["ok"] is False
    assert any("semantic_fields:expected=" in e for e in report["errors"])
    assert report["ok"] is False


def test_12_inconsistent_hidden_fields_rejected(card, profile):
    profile = copy.deepcopy(profile)
    profile["hidden_fields"].append("filename")

    report = validate_profile(profile, card)
    assert report["checks"]["consistency"]["ok"] is False
    assert any("hidden_fields:expected=" in e for e in report["errors"])
    assert report["ok"] is False


def test_13_mirror_map_disagreement_rejected(card, profile):
    profile = copy.deepcopy(profile)
    profile["field_roles"]["province_code"] = "semantic"

    report = validate_profile(profile, card)
    assert report["checks"]["consistency"]["ok"] is False
    assert report["ok"] is False


def test_14_unsupported_gender_numeric_mapping_rejected(card, profile):
    """The card never defines 0/1, so any such mapping must fail."""
    profile = copy.deepcopy(profile)
    profile["fields"]["gender"]["description"] = (
        "Speaker gender; 0 = female and 1 = male.")

    report = validate_profile(profile, card)
    check = report["checks"]["unsupported_label_mappings"]
    assert check["ok"] is False
    labels = {item["label"] for item in check["unsupported"]}
    assert {"female", "male"} <= labels
    assert report["ok"] is False


def test_15_gender_numeric_mapping_in_isolated_string_rejected(card, profile):
    profile = copy.deepcopy(profile)
    profile["qa_constraints"].append("Answer gender with 1 = male.")

    report = validate_profile(profile, card)
    assert report["checks"]["unsupported_label_mappings"]["ok"] is False
    assert report["ok"] is False


def test_16_relation_to_nonexistent_field_rejected(card, profile):
    profile = copy.deepcopy(profile)
    profile["relations"][0]["target_field"] = "dialect_region"

    report = validate_profile(profile, card)
    check = report["checks"]["relation_references"]
    assert check["ok"] is False
    assert any("unknown_field" in e for e in check["messages"])
    assert report["ok"] is False


def test_17_invented_values_rejected(card, profile):
    profile = copy.deepcopy(profile)
    profile["fields"]["province_name"]["example_values"] = ["Ha Noi"]

    report = validate_profile(profile, card)
    assert report["checks"]["invented_values"]["ok"] is False
    assert any("forbidden_key" in e
               for e in report["checks"]["invented_values"]["messages"])
    assert report["ok"] is False


def test_18_exporter_deterministic(card, profile):
    first_schema = export_schema(profile, card)
    second_schema = export_schema(copy.deepcopy(profile), card)
    assert first_schema == second_schema
    assert canonical_json(first_schema) == canonical_json(second_schema)

    first_meta = export_metadata_context(profile, card)
    second_meta = export_metadata_context(copy.deepcopy(profile), card)
    assert first_meta == second_meta
    assert canonical_json(first_meta) == canonical_json(second_meta)


def test_19_same_profile_gives_byte_identical_exports(card, profile, tmp_path):
    runs = []
    for idx in range(2):
        out_root = tmp_path / f"root{idx}"
        result = run_profile(
            dataset="vimd",
            source_card_path=CARD_PATH,
            real_llm=False,
            run_id="fixed-run",
            out_root=out_root,
        )
        assert result["ok"] is True, result.get("report")
        runs.append(result["run_dir"])

    for name in ("schema.json", "metadata_context.json",
                 "dataset_profile.json"):
        left = (runs[0] / name).read_bytes()
        right = (runs[1] / name).read_bytes()
        assert left == right, f"{name} is not byte-identical across runs"


def test_20_mock_run_writes_full_artifact_set(tmp_path, capsys):
    capsys.readouterr()
    result = run_profile(
        dataset="vimd",
        source_card_path=CARD_PATH,
        real_llm=False,
        run_id="artifact-set",
        out_root=tmp_path,
    )
    capsys.readouterr()
    assert result["ok"] is True
    run_dir = Path(result["run_dir"])
    expected = {
        "source_card.md",
        "raw_response.txt",
        "dataset_profile.json",
        "schema.json",
        "metadata_context.json",
        "validation_report.json",
        "run_meta.json",
        "audit.txt",
    }
    assert expected <= {p.name for p in run_dir.iterdir()}

    schema = json.loads((run_dir / "schema.json").read_text(encoding="utf-8"))
    assert set(schema["fields"]) == EXPECTED_RAW_FIELDS
    assert schema["hidden_fields"] == ["speakerID"]
    assert schema["semantic_fields"] == sorted(
        ["text", "gender", "region", "province_name"])

    metadata = json.loads(
        (run_dir / "metadata_context.json").read_text(encoding="utf-8"))
    assert metadata["sample_rows_count"] == 0
    assert metadata["total_rows_source"] == "dataset_card"
    assert metadata["total_rows"] == 18949


def test_21_invalid_profile_exports_nothing(tmp_path, card, profile, capsys):
    profile = copy.deepcopy(profile)
    profile["fields"].pop("audio")
    profile["fields"]["dialect_cluster"] = copy.deepcopy(
        profile["fields"]["region"])
    bad_fixture = tmp_path / "bad_fixture.json"
    bad_fixture.write_text(
        json.dumps(profile, ensure_ascii=False), encoding="utf-8")

    capsys.readouterr()
    result = run_profile(
        dataset="vimd",
        source_card_path=CARD_PATH,
        real_llm=False,
        run_id="invalid-run",
        out_root=tmp_path / "out",
        fixture_path=bad_fixture,
    )
    capsys.readouterr()

    assert result["ok"] is False
    assert result["exit_code"] == 1
    run_dir = Path(result["run_dir"])
    assert (run_dir / "dataset_profile.json").is_file()
    assert (run_dir / "validation_report.json").is_file()
    assert not (run_dir / "schema.json").exists()
    assert not (run_dir / "metadata_context.json").exists()


def test_22_run_directory_is_never_reused(tmp_path, capsys):
    capsys.readouterr()
    first = run_profile(dataset="vimd", source_card_path=CARD_PATH,
                        real_llm=False, run_id="same", out_root=tmp_path)
    second = run_profile(dataset="vimd", source_card_path=CARD_PATH,
                         real_llm=False, run_id="same", out_root=tmp_path)
    capsys.readouterr()
    assert first["run_meta"]["run_id"] == "same"
    assert second["run_meta"]["run_id"] == "same-1"
    assert Path(first["run_dir"]).parent.name == "same"
    assert Path(second["run_dir"]).parent.name == "same-1"
    assert Path(first["run_dir"]).name == "profile"
    assert Path(second["run_dir"]).name == "profile"
    assert first["run_dir"] != second["run_dir"]


def test_23_legacy_vimd_artifacts_untouched(tmp_path, capsys):
    """The stage must never write into data/vimd."""
    legacy = ["schema.json", "metadata_context.json", "README.md",
              "manifest.jsonl", "sample.jsonl"]
    before = {
        name: (ROOT / "data" / "vimd" / name).read_bytes()
        for name in legacy
    }
    capsys.readouterr()
    run_profile(dataset="vimd", source_card_path=CARD_PATH, real_llm=False,
                run_id="safety", out_root=tmp_path)
    capsys.readouterr()
    for name in legacy:
        path = ROOT / "data" / "vimd" / name
        assert path.read_bytes() == before[name], f"legacy {name} changed"


def test_24_prompt_contains_required_instructions(card):
    instructions = (ROOT / "prompts" / "dataset_profile.txt").read_text(
        encoding="utf-8")
    prompt = build_prompt(card, instructions)
    assert "sole semantic authority" in instructions
    assert "0 = female" in instructions
    assert "hidden_identifier" in instructions
    assert "discovery_only" in instructions
    # The card itself is appended verbatim as the final payload.
    assert prompt.endswith(card.text)


def test_25_prompt_embeds_response_schema_before_card(card):
    """The contract travels inside the prompt: the endpoint ignores
    `response_format`, so exact key names and the JSON schema must be
    readable by the model, and the card must stay last."""
    instructions = (ROOT / "prompts" / "dataset_profile.txt").read_text(
        encoding="utf-8")
    prompt = build_prompt(card, instructions, dataset="vimd")

    contract_at = prompt.index("--- RESPONSE SCHEMA (EXACT KEY NAMES")
    last_marker_at = prompt.rindex(PROFILE_SCHEMA_MARKER)
    assert contract_at < last_marker_at < prompt.index(card.text)

    assert '"$defs"' in prompt
    assert '"ProfileField"' in prompt
    assert "`dataset_name`" in prompt          # alias called out as wrong
    assert 'must be exactly "vimd"' in prompt
    assert "exactly the field names whose `role` is" in prompt
    assert "No provenance, context_only, semantic or" in prompt

    # The intro sentence quoting the marker stays intact.
    assert "after the marker `--- SOURCE CARD (SOLE SEMANTIC AUTHORITY) ---`." \
        in prompt
    assert prompt.endswith(card.text)


def test_26_prompt_without_dataset_key_has_no_dataset_line(card):
    prompt = build_prompt(card, None)
    assert "--- RESPONSE SCHEMA (EXACT KEY NAMES" in prompt
    assert 'must be exactly "' not in prompt
    assert prompt.endswith(card.text)


def test_27_structured_dtype_parsing_regression(tmp_path):
    """Generic parser regression test for HuggingFace-style structured dtypes in card frontmatter."""
    card_content = """---
dataset_info:
  features:
  - name: audio_col
    dtype:
      audio:
        decode: false
  - name: text_col
    dtype: string
  - name: count_col
    dtype: int64
---
# Synthetic Card
    """
    card_file = tmp_path / "dataset_card.md"
    card_file.write_text(card_content, encoding="utf-8")

    parsed_card = parse_source_card(card_file)
    assert parsed_card.fields == {
        "audio_col": "audio",
        "text_col": "string",
        "count_col": "int64",
    }


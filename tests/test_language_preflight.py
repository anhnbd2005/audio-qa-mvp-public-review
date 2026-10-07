"""Global language preflight and new-production gate tests."""

from __future__ import annotations

import socket
import urllib.request
import wave

import pytest

from src.autonomous_qa.language import language_quality as glq
from src.autonomous_qa.language import language_preflight as preflight
from src.autonomous_qa.language.language_quality import ProductionGenerationConfig
from src.autonomous_qa.production.production_qa import bind_target_value, enforce_language_preflight_gate
from src.autonomous_qa.certification.qa_sampling import SemanticSampler


def _accepted(
    *,
    type_id="synthetic-type",
    operator="DIRECT",
    semantic_class="categorical_attribute",
    answer_kind="field_value",
    audio_input_count=1,
    logical_context_inputs=0,
    phrase_bindings=None,
    source_visibility="model_semantic",
):
    return preflight.AcceptedLanguageType(
        dataset_type_id=type_id,
        operator=operator,
        semantic_class=semantic_class,
        semantic_field="synthetic_field",
        answer_kind=answer_kind,
        audio_input_count=audio_input_count,
        logical_context_inputs=logical_context_inputs,
        phrase_bindings=phrase_bindings
        or {
            "entity_scope": "speaker",
            "entity_phrase": "ngÆ°á»i nÃ³i",
            "attribute_phrase": "thuá»™c tÃ­nh",
            "content_phrase": None,
            "value_phrase": "giÃ¡ trá»‹",
            "unit": None,
            "target_quote_style": "plain",
        },
        source_visibility=source_visibility,
    )


@pytest.fixture(scope="module")
def registry_result():
    return preflight.run_preflight(
        mode="registry",
        write_outputs=False,
    )


@pytest.fixture(scope="module")
def dataset_result():
    return preflight.run_preflight(
        mode="dataset",
        dataset="vimd",
        accepted_types=preflight.vimd_accepted_types(),
        write_outputs=True,
    )


def test_registry_mode_covers_every_active_entry(registry_result):
    audit = registry_result["audit"]
    assert audit["result"] == "PREFLIGHT_PASS"
    assert audit["entries_checked"] == 62
    assert len(audit["required_capability_set"]) == 13
    assert audit["fixture_count"] == 52
    assert audit["synthetic_render_count"] == 910
    assert audit["blocking_issue_count"] == 0
    assert audit["review_issue_count"] == 0
    assert (
        len({row["language_entry_id"] for row in registry_result["render_matrix"]})
        == 62
    )


def test_vimd_dataset_mode_covers_eleven_types(dataset_result):
    audit = dataset_result["audit"]
    assert audit["result"] == "PREFLIGHT_PASS"
    assert audit["accepted_type_count"] == 11
    assert audit["coverage_status"] == "COMPLETE"
    assert audit["entries_checked"] == 32
    assert audit["synthetic_render_count"] == 688
    assert audit["blocking_issue_count"] == 0
    assert audit["review_issue_count"] == 0


def test_missing_capability_does_not_reject_semantics():
    item = _accepted(semantic_class="boolean_attribute")
    result = preflight.run_preflight(
        mode="dataset",
        dataset="synthetic",
        accepted_types=[item],
        write_outputs=False,
    )
    assert result["audit"]["result"] == "LANGUAGE_CAPABILITY_MISSING"
    assert result["coverage"][0]["coverage"] == "MISSING"


def test_unknown_operator_has_explicit_result():
    result = preflight.run_preflight(
        mode="dataset",
        dataset="synthetic",
        accepted_types=[_accepted(operator="DERIVED_ATTRIBUTE")],
        write_outputs=False,
    )
    assert result["audit"]["result"] == "OPERATOR_CONTRACT_MISSING"


def test_unknown_semantic_class_has_explicit_fixture_result():
    result = preflight.run_preflight(
        mode="dataset",
        dataset="synthetic",
        accepted_types=[_accepted(semantic_class="novel_semantic_class")],
        write_outputs=False,
    )
    assert result["audit"]["result"] == "PREFLIGHT_FIXTURE_STRATEGY_MISSING"


def test_missing_phrase_binding_never_falls_back_to_raw_field_name():
    item = _accepted(
        semantic_class="text_content",
        phrase_bindings={
            "entity_scope": "utterance",
            "entity_phrase": "Ä‘oáº¡n Ã¢m thanh",
            "content_phrase": None,
            "attribute_phrase": None,
            "value_phrase": "ná»™i dung",
            "unit": None,
            "target_quote_style": "vietnamese_quotes",
        },
    )
    result = preflight.run_preflight(
        mode="dataset",
        dataset="synthetic",
        accepted_types=[item],
        write_outputs=False,
    )
    assert result["audit"]["result"] == "LANGUAGE_PHRASE_BINDING_MISSING"


def test_hidden_identifier_is_blocked_as_profile_contract_failure():
    result = preflight.run_preflight(
        mode="dataset",
        dataset="synthetic",
        accepted_types=[_accepted(source_visibility="hidden_identifier")],
        write_outputs=False,
    )
    assert result["audit"]["result"] == "PROFILE_SEMANTIC_CONTRACT_FAIL"


def test_detector_and_false_positive_regressions():
    result = preflight.detector_regression()
    assert result["double_quote_would_be_caught"] is True
    assert result["direct_text_tautology_would_be_caught"] is True
    assert result["duplicate_semantic_head_detected"] is True
    assert result["corrected_entry_passes"] is True
    assert result["natural_nguoi_noi_incorrectly_rejected"] is False


def test_valid_repetition_like_construction_is_not_naively_rejected():
    entry = preflight._synthetic_entry(
        operator="DIRECT",
        semantic_class="categorical_attribute",
        pattern="Ná»™i dung A khÃ¡c ná»™i dung B.",
    )
    issues = preflight.detect_render_issues(
        question=entry.pattern,
        entry=entry,
        spec=preflight._synthetic_spec("categorical_attribute", None),
        bound_target=None,
        operator_contract=preflight.operator_contracts()["DIRECT"],
    )
    assert issues == []


def test_preflight_is_deterministic():
    first = preflight.run_preflight(
        mode="registry",
        write_outputs=False,
    )
    second = preflight.run_preflight(
        mode="registry",
        write_outputs=False,
    )
    assert first["audit"] == second["audit"]
    assert first["render_matrix"] == second["render_matrix"]
    assert first["issues"] == second["issues"]
    assert first["coverage"] == second["coverage"]


def test_stale_and_failed_passes_are_rejected(dataset_result):
    artifact = dataset_result["audit"]
    fingerprint = artifact["contract_fingerprint"]
    assert preflight.authorize_new_production(
        require_language_preflight=True,
        current_fingerprint=fingerprint,
        pass_artifact=artifact,
    ) == {"allowed": True, "status": "PREFLIGHT_PASS"}
    assert (
        preflight.authorize_new_production(
            require_language_preflight=True,
            current_fingerprint="changed-renderer-or-registry-hash",
            pass_artifact=artifact,
        )["status"]
        == "PREFLIGHT_STALE"
    )
    assert (
        preflight.authorize_new_production(
            require_language_preflight=True,
            current_fingerprint=fingerprint,
            pass_artifact=None,
        )["status"]
        == "PREFLIGHT_REQUIRED"
    )
    assert (
        preflight.authorize_new_production(
            require_language_preflight=True,
            current_fingerprint=fingerprint,
            pass_artifact={**artifact, "result": "LANGUAGE_CONTRACT_FAIL"},
        )["status"]
        == "PREFLIGHT_FAILED"
    )


def test_production_gate_accepts_matching_canonical_pass(dataset_result):
    current = ProductionGenerationConfig(
        language_preflight_artifact=str(dataset_result["output_dir"]),
    )
    assert (
        enforce_language_preflight_gate(dataset="vimd", config=current)["status"]
        == "PREFLIGHT_PASS"
    )


def test_zero_sampler_audio_network_and_llm_calls(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("forbidden preflight side effect")

    monkeypatch.setattr(SemanticSampler, "sample", forbidden)
    monkeypatch.setattr(wave, "open", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(glq, "create_llm_client", forbidden)
    result = preflight.run_preflight(
        mode="registry",
        write_outputs=False,
    )
    assert result["audit"]["result"] == "PREFLIGHT_PASS"
    assert all(value == 0 for value in result["audit"]["zero_side_effects"].values())


def test_canonical_resource_files_exist():
    assert preflight.REGISTRY_RESOURCE.is_file()
    assert preflight.TEMPLATE_RESOURCE.is_file()
    assert preflight.PARAPHRASE_RESOURCE.is_file()
    assert preflight.RENDERER_CONTRACT_RESOURCE.is_file()


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("xin chào", "xin chào"),
        ('"đã trích dẫn"', "đã trích dẫn"),
        ("“đã trích dẫn”", "đã trích dẫn"),
        ("  nhiều   khoảng trắng  ", "nhiều khoảng trắng"),
    ],
)
def test_canonical_target_binding_owns_no_presentation_quotes(target, expected):
    assert bind_target_value(target, quote_style="vietnamese_quotes") == expected


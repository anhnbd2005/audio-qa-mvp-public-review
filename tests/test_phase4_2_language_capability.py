"""Phase 4.2 — declarative SemanticFieldSpec + entity-scope-aware language composition.

Verifies:
- DatasetProfile -> SemanticFieldSpec compilation (profile-native + migration sidecar)
- no raw-field-name semantic inference; fail-closed on missing required semantics
- deterministic field-spec logical hash
- registry-declared / conventional migration sidecar resolution (no dataset branch)
- entity-scope + entity_reference_owner capability model
- structural ownership validation retained (SEMANTIC_HEAD_OWNERSHIP_COLLISION)
- unit-policy compatibility
- candidate registry builder determinism + dataset neutrality
- explicit-registry preflight
- cross-dataset non-regression (ViMD / VietMDD)
- fresh ViMedCSS 3-task candidate preflight PASS
- canonical registry / contracts untouched
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.autonomous_qa.certification.authoring_promotion import prepare_promotion
from src.autonomous_qa.compiler.semantic_field_specs import (
    SemanticFieldSpec,
    SemanticFieldSpecBundle,
    compile_semantic_field_specs,
    load_migration_sidecar,
)
from src.autonomous_qa.core.schemas import (
    DatasetProfile,
    ProfileField,
    ProfileFieldVerbalization,
)
from src.autonomous_qa.language import language_preflight as preflight
from src.autonomous_qa.language.candidate_registry import (
    CANDIDATE_CAPABILITIES_RESOURCE,
    build_candidate_language_registry,
    build_candidate_registry_from_paths,
    load_candidate_capabilities,
)
from src.autonomous_qa.language.language_preflight import (
    AcceptedLanguageType,
    accepted_types_from_semantic_catalog,
    run_preflight,
)
from src.autonomous_qa.language.language_quality import (
    LanguageRegistryEntry,
    entry_capability_compatible,
    load_language_registry,
    pattern_ownership_conflicts,
    resolve_entry_capability,
    slot_value_map,
)
from src.common.config import ROOT

ACCEPTANCE_RUN_DIR = ROOT / "outputs" / "runs" / "vimedcss" / "20261007_vimedcss_v2_authoring"
CANONICAL_REGISTRY = ROOT / "resources" / "language" / "production_registry.json"


# ---------------------------------------------------------------------------
# field-spec compiler
# ---------------------------------------------------------------------------


def _profile(fields: dict[str, ProfileField], dataset: str = "alpha_dataset") -> DatasetProfile:
    return DatasetProfile(
        dataset=dataset,
        pretty_name="Alpha",
        description="synthetic",
        locale="vi",
        fields=fields,
        field_roles={n: f.role for n, f in fields.items()},
        entity_scopes={n: f.entity_scope for n, f in fields.items()},
        field_evaluations={n: f.evaluation for n, f in fields.items()},
        context_visibilities={n: f.context_visibility for n, f in fields.items()},
        hidden_fields=[],
        semantic_fields=sorted(n for n, f in fields.items() if f.role == "semantic"),
    )


def _field(**over) -> ProfileField:
    base = dict(
        type="string",
        role="semantic",
        entity_scope="utterance",
        description="",
        evaluation="eligible",
        context_visibility="visible",
    )
    base.update(over)
    return ProfileField(**base)


def test_compile_from_profile_native_semantics():
    profile = _profile({
        "f_a": _field(
            semantic_class="categorical_attribute",
            verbalization=ProfileFieldVerbalization(
                entity_phrase="đoạn âm thanh",
                attribute_phrase="loại sự kiện",
                value_phrase="loại",
            ),
        ),
        "f_b": _field(
            type="int64",
            entity_scope="recording",
            semantic_class="numeric_attribute",
            verbalization=ProfileFieldVerbalization(
                entity_phrase="bản ghi",
                attribute_phrase="số lần xuất hiện",
                value_phrase="số lần",
            ),
        ),
    })
    bundle = compile_semantic_field_specs(profile, required_fields={"f_a", "f_b"})
    assert bundle.source == "profile_native"
    assert bundle.field_specs["f_a"].semantic_class == "categorical_attribute"
    assert bundle.field_specs["f_a"].attribute_phrase == "loại sự kiện"
    assert bundle.field_specs["f_b"].semantic_class == "numeric_attribute"
    assert bundle.field_specs["f_b"].entity_scope == "recording"
    assert bundle.dataset_id == "alpha_dataset"
    # no dataset-ID branch: same code path for any dataset id
    profile2 = _profile({
        "g": _field(semantic_class="text_content",
                    verbalization=ProfileFieldVerbalization(
                        entity_phrase="đoạn âm thanh", content_phrase="nội dung")),
    }, dataset="beta_dataset")
    assert compile_semantic_field_specs(profile2, required_fields={"g"}).field_specs["g"].semantic_class == "text_content"


def test_missing_semantic_class_without_sidecar_fails_closed():
    profile = _profile({"topic": _field(semantic_class=None, verbalization=None)})
    with pytest.raises(ValueError) as exc:
        compile_semantic_field_specs(profile, required_fields={"topic"})
    assert "FIELD_SPEC_MISSING:topic" in str(exc.value)


def test_missing_required_verbalization_fails_closed():
    profile = _profile({"topic": _field(semantic_class="categorical_attribute", verbalization=None)})
    with pytest.raises(ValueError) as exc:
        compile_semantic_field_specs(profile, required_fields={"topic"})
    assert "FIELD_SPEC_MISSING:topic" in str(exc.value)


def test_migration_sidecar_supplies_missing_semantics():
    profile = _profile({"topic": _field(semantic_class=None, verbalization=None)})
    migration = {
        "topic": SemanticFieldSpec(
            field_name="topic",
            semantic_class="categorical_attribute",
            entity_scope="utterance",
            entity_phrase="đoạn âm thanh",
            attribute_phrase="chủ đề",
            value_phrase="chủ đề",
            source="migration_sidecar",
        )
    }
    bundle = compile_semantic_field_specs(profile, required_fields={"topic"}, migration_specs=migration)
    assert bundle.source == "migration_sidecar"
    assert bundle.field_specs["topic"].attribute_phrase == "chủ đề"


def test_raw_field_name_never_dominates_explicit_semantics():
    profile = _profile({
        "count_label": _field(
            semantic_class="categorical_attribute",
            verbalization=ProfileFieldVerbalization(
                entity_phrase="đoạn âm thanh", attribute_phrase="nhãn", value_phrase="nhãn"),
        ),
        "f_x": _field(
            type="int64",
            semantic_class="numeric_attribute",
            verbalization=ProfileFieldVerbalization(
                entity_phrase="đoạn âm thanh", attribute_phrase="số lượng", value_phrase="số"),
        ),
    })
    bundle = compile_semantic_field_specs(profile, required_fields={"count_label", "f_x"})
    # "count_label" stays categorical; "f_x" (no "count"/"num") is numeric.
    assert bundle.field_specs["count_label"].semantic_class == "categorical_attribute"
    assert bundle.field_specs["f_x"].semantic_class == "numeric_attribute"


def test_field_spec_bundle_logical_hash_is_deterministic():
    profile = _profile({
        "topic": _field(semantic_class="categorical_attribute",
                        verbalization=ProfileFieldVerbalization(
                            entity_phrase="đoạn âm thanh", attribute_phrase="chủ đề")),
    })
    b1 = compile_semantic_field_specs(profile, required_fields={"topic"})
    b2 = compile_semantic_field_specs(profile, required_fields={"topic"})
    assert b1.logical_hash() == b2.logical_hash()
    assert b1.logical_hash() == SemanticFieldSpecBundle.model_validate(
        b1.model_dump(mode="json")
    ).logical_hash()


def test_vimedcss_migration_sidecar_declares_required_fields():
    sidecar = load_migration_sidecar(ROOT / "resources" / "field_specs" / "vimedcss.json")
    assert sidecar["topic"].semantic_class == "categorical_attribute"
    assert sidecar["topic"].entity_scope == "utterance"
    assert sidecar["cs_terms_count"].semantic_class == "numeric_attribute"
    assert sidecar["cs_terms_count"].unit is None
    assert sidecar["segment_text"].semantic_class == "text_content"


# ---------------------------------------------------------------------------
# capability model
# ---------------------------------------------------------------------------


def _entry(**over) -> LanguageRegistryEntry:
    base = dict(
        language_entry_id="e",
        source_kind="CANDIDATE",
        source_id="e",
        canonical_blueprint_id="e",
        operator="DIRECT",
        semantic_class="categorical_attribute",
        pattern="[ATTRIBUTE_PHRASE] của đoạn âm thanh là gì?",
        answer_kind="field_value",
        required_slots=["[ATTRIBUTE_PHRASE]"],
        unit_policy="any",
        match_policy=["exact"],
        semantic_contract_hash="x",
        quality_status="PRODUCTION_PASS",
        enabled=True,
        template_library_hash="h",
        registry_version="v",
        entity_scopes=["utterance"],
        entity_reference_owner="literal",
    )
    base.update(over)
    return LanguageRegistryEntry(**base)


def test_resolve_capability_classifies_legacy_slot_owner():
    legacy = _entry(
        language_entry_id="legacy",
        source_kind="CANONICAL",
        entity_scopes=None,
        entity_reference_owner=None,
        pattern="[ATTRIBUTE_PHRASE] của [ENTITY_PHRASE] trong đoạn âm thanh là gì?",
    )
    caps = resolve_entry_capability(legacy)
    assert caps.entity_reference_owner == "slot"
    assert "đoạn âm thanh" in caps.literal_entity_heads


def test_inconsistent_capability_declaration_fails_closed():
    bad = _entry(entity_reference_owner="literal",
                 pattern="[ATTRIBUTE_PHRASE] của [ENTITY_PHRASE] là gì?")
    with pytest.raises(Exception) as exc:
        resolve_entry_capability(bad)
    assert "LANGUAGE_ENTRY_CAPABILITY_UNRESOLVED" in str(exc.value)


def test_entity_scope_filtering_speaker_vs_utterance():
    utterance_entry = _entry(entity_scopes=["utterance"])
    speaker_slot_entry = _entry(
        language_entry_id="spk",
        entity_scopes=None,
        entity_reference_owner=None,
        pattern="[ATTRIBUTE_PHRASE] của [ENTITY_PHRASE] trong đoạn âm thanh là gì?",
    )
    # speaker field: literal "đoạn âm thanh" is a container, slot owns entity.
    ok, _ = entry_capability_compatible(
        speaker_slot_entry, entity_scope="speaker", unit=None,
        slot_values={"[ENTITY_PHRASE]": "người nói", "[ATTRIBUTE_PHRASE]": "vùng"},
    )
    assert ok
    # utterance field: literal and slot would both own "đoạn âm thanh".
    bad, reason = entry_capability_compatible(
        speaker_slot_entry, entity_scope="utterance", unit=None,
        slot_values={"[ENTITY_PHRASE]": "đoạn âm thanh", "[ATTRIBUTE_PHRASE]": "chủ đề"},
    )
    assert not bad and reason == "SEMANTIC_OWNERSHIP_CONFLICT"
    # declared utterance entry rejects a speaker field.
    ok2, reason2 = entry_capability_compatible(
        utterance_entry, entity_scope="speaker", unit=None,
        slot_values={"[ATTRIBUTE_PHRASE]": "vùng"},
    )
    assert not ok2 and reason2 == "ENTITY_SCOPE_INCOMPATIBLE"


def test_literal_owned_entry_does_not_require_entity_slot():
    literal_entry = _entry(pattern="[ATTRIBUTE_PHRASE] của đoạn âm thanh là gì?",
                           required_slots=["[ATTRIBUTE_PHRASE]"])
    ok, _ = entry_capability_compatible(
        literal_entry, entity_scope="utterance", unit=None,
        slot_values={"[ENTITY_PHRASE]": "đoạn âm thanh", "[ATTRIBUTE_PHRASE]": "chủ đề"},
    )
    assert ok


def test_unit_policy_compatibility():
    required_entry = _entry(language_entry_id="u", unit_policy="required",
                            pattern="[ATTRIBUTE_PHRASE] trong đoạn âm thanh là bao nhiêu [UNIT]?",
                            required_slots=["[ATTRIBUTE_PHRASE]", "[UNIT]"],
                            semantic_class="numeric_attribute")
    ok_none, reason_none = entry_capability_compatible(
        required_entry, entity_scope="utterance", unit=None,
        slot_values={"[ATTRIBUTE_PHRASE]": "số từ"})
    assert not ok_none and reason_none == "UNIT_REQUIRED_BUT_ABSENT"
    ok_unit, _ = entry_capability_compatible(
        required_entry, entity_scope="utterance", unit="giây",
        slot_values={"[ATTRIBUTE_PHRASE]": "thời lượng"})
    assert ok_unit


def test_equality_utterance_pair_entry_selected():
    registry = build_candidate_registry_from_paths(CANONICAL_REGISTRY)
    item = AcceptedLanguageType(
        dataset_type_id="t",
        operator="EQUALITY",
        semantic_class="categorical_attribute",
        semantic_field="topic",
        answer_kind="boolean",
        audio_input_count=2,
        logical_context_inputs=0,
        phrase_bindings={
            "entity_scope": "utterance",
            "entity_phrase": "đoạn âm thanh",
            "attribute_phrase": "chủ đề y khoa",
            "content_phrase": None,
            "value_phrase": "chủ đề",
            "unit": None,
            "target_quote_style": "plain",
        },
    )
    entries = preflight._compatible_entries(registry, item)
    ids = {e.language_entry_id for e in entries}
    assert "cap_vi1_eq_cat_utt_01" in ids
    assert all(e.entity_reference_owner == "literal" or "[ENTITY_PHRASE]" not in e.pattern for e in entries)


def test_structural_ownership_validation_blocks_invalid_composition():
    invalid = _entry(
        language_entry_id="invalid",
        entity_scopes=["utterance"],
        entity_reference_owner="slot",
        pattern="[ATTRIBUTE_PHRASE] của [ENTITY_PHRASE] trong đoạn âm thanh là gì?",
        required_slots=["[ATTRIBUTE_PHRASE]", "[ENTITY_PHRASE]"],
    )
    spec = SemanticFieldSpec(
        field_name="topic",
        semantic_class="categorical_attribute",
        entity_scope="utterance",
        entity_phrase="đoạn âm thanh",
        attribute_phrase="chủ đề y khoa",
        value_phrase="chủ đề",
    )
    issues = preflight._composition_ownership_issues(invalid.pattern, spec)
    assert any(code == "SEMANTIC_HEAD_OWNERSHIP_COLLISION" for code, _, _ in issues)
    # and the structural precheck agrees
    assert pattern_ownership_conflicts(
        invalid.pattern, slot_value_map({"entity_phrase": "đoạn âm thanh", "attribute_phrase": "chủ đề y khoa"})
    )


# ---------------------------------------------------------------------------
# candidate registry
# ---------------------------------------------------------------------------


def test_candidate_registry_is_deterministic():
    r1 = build_candidate_registry_from_paths(CANONICAL_REGISTRY)
    r2 = build_candidate_registry_from_paths(CANONICAL_REGISTRY)
    assert r1.registry_hash == r2.registry_hash
    assert [e.language_entry_id for e in r1.entries] == sorted(e.language_entry_id for e in r1.entries)
    assert json.dumps(r1.model_dump(mode="json"), sort_keys=True) == json.dumps(r2.model_dump(mode="json"), sort_keys=True)


def test_candidate_capabilities_are_dataset_neutral():
    caps = load_candidate_capabilities(CANDIDATE_CAPABILITIES_RESOURCE)
    forbidden = ("vimedcss", "vimd", "vietmdd", "cs_terms_count", "topic_classification",
                 "pairwise_topic_same", "medical")
    blob = json.dumps(caps, ensure_ascii=False).casefold()
    for token in forbidden:
        assert token not in blob, token


def test_canonical_registry_still_loads_and_hash_stable():
    canonical = load_language_registry(CANONICAL_REGISTRY)
    candidate = build_candidate_language_registry(canonical, load_candidate_capabilities())
    capability_ids = {c["capability_id"] for c in load_candidate_capabilities()}
    installed = capability_ids <= {e.language_entry_id for e in canonical.entries}
    if installed:
        # Post-promotion: certified capabilities already canonical -> idempotent.
        assert candidate.registry_hash == canonical.registry_hash
        assert len(candidate.entries) == len(canonical.entries)
    else:
        assert candidate.registry_hash != canonical.registry_hash
        assert len(candidate.entries) == len(canonical.entries) + 7


def test_explicit_registry_preflight_uses_candidate_identity():
    candidate = build_candidate_registry_from_paths(CANONICAL_REGISTRY)
    scratch = ROOT / "outputs" / "_scratch" / "language_capability" / "phase4_2" / "test_candidate.json"
    from src.autonomous_qa.language.candidate_registry import write_registry_deterministically

    write_registry_deterministically(candidate, scratch)
    item = AcceptedLanguageType(
        dataset_type_id="t", operator="DIRECT", semantic_class="categorical_attribute",
        semantic_field="topic", answer_kind="field_value", audio_input_count=1,
        logical_context_inputs=0,
        phrase_bindings={"entity_scope": "utterance", "entity_phrase": "đoạn âm thanh",
                         "attribute_phrase": "chủ đề", "unit": None, "target_quote_style": "plain"},
    )
    res = run_preflight(mode="dataset", accepted_types=[item], dataset="t",
                        write_outputs=False, registry=candidate, registry_path=scratch)
    assert res["audit"]["language_registry_hash"] == candidate.registry_hash


# ---------------------------------------------------------------------------
# cross-dataset non-regression + ViMedCSS acceptance
# ---------------------------------------------------------------------------


def test_cross_dataset_no_new_blocking_errors():
    candidate = build_candidate_registry_from_paths(CANONICAL_REGISTRY)
    scratch = ROOT / "outputs" / "_scratch" / "language_capability" / "phase4_2" / "test_regress.json"
    from src.autonomous_qa.language.candidate_registry import write_registry_deterministically

    write_registry_deterministically(candidate, scratch)
    for dataset in ("vimd", "vietmdd"):
        accepted = preflight.get_dataset_accepted_types(dataset)
        base = run_preflight(mode="dataset", accepted_types=accepted, dataset=dataset, write_outputs=False)
        cand = run_preflight(mode="dataset", accepted_types=accepted, dataset=dataset,
                             write_outputs=False, registry=candidate, registry_path=scratch)
        assert cand["audit"]["blocking_issue_count"] <= base["audit"]["blocking_issue_count"]
        assert base["audit"]["result"] == "PREFLIGHT_PASS"


def test_fresh_vimedcss_three_task_candidate_preflight_passes(tmp_path: Path):
    bundle = prepare_promotion(
        dataset_id="vimedcss",
        run_dir=ACCEPTANCE_RUN_DIR,
        promotion_mode="replace",
        output_root=tmp_path / "prepare",
        language_capability_root=tmp_path / "cap",
    )
    assert bundle.semantic_field_specs_source == "migration_sidecar"
    assert bundle.semantic_field_specs_hash
    assert bundle.candidate_language_preflight["status"] == "PREFLIGHT_PASS"
    assert bundle.candidate_language_preflight["blocking_issues"] == 0
    assert bundle.language_infra_ready is True
    if bundle.language_registry_change_required:
        # Pre language-registry promotion: canonical staged preflight still fails.
        assert bundle.staged_language_preflight["status"] != "PREFLIGHT_PASS"
        assert bundle.apply_ready is False
        assert bundle.canonical_language_registry_hash != bundle.candidate_language_registry_hash
    else:
        # Post language-registry promotion: staged preflight passes.
        assert bundle.staged_language_preflight["status"] == "PREFLIGHT_PASS"
        assert bundle.staged_language_preflight["blocking_issues"] == 0
        assert bundle.canonical_language_registry_hash == bundle.candidate_language_registry_hash
    assert set(bundle.selected_promotion_types) == {
        "vimedcss_topic_classification",
        "vimedcss_cs_terms_count",
        "vimedcss_pairwise_topic_same",
    }


def test_normal_staged_prepare_never_uses_legacy_field_specs(monkeypatch, tmp_path: Path):
    """The normal staged path resolves specs declaratively, never via a dataset branch."""
    import src.autonomous_qa.language.template_engine as template_engine

    def _boom(dataset_id):  # pragma: no cover - must never run
        raise AssertionError("LEGACY_FIELD_SPEC_PATH_USED")

    monkeypatch.setattr(template_engine, "resolve_legacy_field_specs", _boom)
    bundle = prepare_promotion(
        dataset_id="vimedcss",
        run_dir=ACCEPTANCE_RUN_DIR,
        promotion_mode="replace",
        output_root=tmp_path / "prepare",
        language_capability_root=tmp_path / "cap",
    )
    assert bundle.candidate_language_preflight["status"] == "PREFLIGHT_PASS"


def test_missing_entity_scope_is_explicit_unsupported_state():
    profile = _profile({
        "topic": _field(
            entity_scope=None,
            semantic_class="categorical_attribute",
            verbalization=ProfileFieldVerbalization(
                entity_phrase="đoạn âm thanh", attribute_phrase="chủ đề"),
        ),
    })
    bundle = compile_semantic_field_specs(profile, required_fields={"topic"})
    assert bundle.field_specs["topic"].entity_scope == "unknown"
    # A declared-scope candidate entry must fail closed for an unknown scope.
    from src.autonomous_qa.language.candidate_registry import (
        build_candidate_registry_from_paths,
    )

    registry = build_candidate_registry_from_paths(CANONICAL_REGISTRY)
    item = AcceptedLanguageType(
        dataset_type_id="t", operator="DIRECT", semantic_class="categorical_attribute",
        semantic_field="topic", answer_kind="field_value", audio_input_count=1,
        logical_context_inputs=0,
        phrase_bindings={"entity_scope": "unknown", "entity_phrase": "đoạn âm thanh",
                         "attribute_phrase": "chủ đề", "unit": None, "target_quote_style": "plain"},
    )
    entries = preflight._compatible_entries(registry, item)
    assert "cap_vi1_dir_cat_utt_01" not in {e.language_entry_id for e in entries}


def test_vimedcss_rendered_questions_are_natural(tmp_path: Path):
    bundle = prepare_promotion(
        dataset_id="vimedcss",
        run_dir=ACCEPTANCE_RUN_DIR,
        promotion_mode="replace",
        output_root=tmp_path / "prepare",
        language_capability_root=tmp_path / "cap",
    )
    matrix = bundle.candidate_language_preflight["render_matrix"]
    questions = {row["rendered_question"] for row in matrix}
    assert "Chủ đề y khoa của đoạn âm thanh là gì?" in questions
    assert "Số thuật ngữ code-switching trong đoạn âm thanh là bao nhiêu?" in questions
    assert any("Hai đoạn âm thanh" in q and "chủ đề y khoa" in q for q in questions)
    for q in questions:
        assert q.count("đoạn âm thanh") <= 1, q
        assert "bao nhiêu đơn vị" not in q
        assert "_" not in q

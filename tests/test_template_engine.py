from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.autonomous_qa.language.template_engine import vimd_field_specs
from src.autonomous_qa.core.schemas import DatasetProfile
from src.autonomous_qa.compiler.semantic_field_specs import (
    SemanticFieldSpec,
    load_migration_sidecar,
    resolve_field_spec,
)
from src.autonomous_qa.language.template_contracts import TypeContract, instantiate_type
from src.autonomous_qa.language.template_renderer import (
    RenderedTemplate,
    TemplateBlueprint,
    TemplateLibrary,
    load_library,
    render_blueprint,
    render_type_contract,
)

LIBRARY_PATH = Path("resources/language/template_library.json")
PROFILE_PATH = Path(__file__).resolve().parent / "fixtures" / "dataset_profile_vimd.json"


def library():
    return load_library(LIBRARY_PATH)


def field_spec(semantic_class="categorical_attribute", field="region", unit=None):
    payload = {
        "field_name": field,
        "semantic_class": semantic_class,
        "entity_scope": "speaker",
        "entity_phrase": "người nói",
        "value_phrase": "giá trị",
        "unit": unit,
    }
    if semantic_class == "text_content":
        payload["content_phrase"] = "nội dung được nói"
        payload["entity_scope"] = "utterance"
    else:
        payload["attribute_phrase"] = {
            "numeric_attribute": "độ tuổi",
            "categorical_attribute": "vùng phương ngữ",
        }.get(semantic_class, "thuộc tính")
    return SemanticFieldSpec(**payload)


def contract(operator="DIRECT", field="region", status="SUPPORTED"):
    return TypeContract(
        type_id=f"type-{operator}-{field}",
        operator=operator,
        semantic_field=field,
        semantic_description="test",
        semantic_phrase_vi="test",
        entity_scope="speaker",
        audio_input_count=1 if operator in {"DIRECT", "TARGET_MATCH"} else 2,
        condition_fields=[field]
        if operator in {"TARGET_MATCH", "PAIRWISE_SELECTION"}
        else [],
        gold_source_fields=[field],
        answer_kind={
            "DIRECT": "field_value",
            "EQUALITY": "boolean",
            "TARGET_MATCH": "boolean",
            "PAIRWISE_SELECTION": "audio_index",
        }[operator],
        answer_mode={
            "DIRECT": "FIELD_VALUE",
            "EQUALITY": "BOOLEAN",
            "TARGET_MATCH": "BOOLEAN",
            "PAIRWISE_SELECTION": "A_B_SELECTION",
        }[operator],
        template_status="ELIGIBLE",
        template_policy={},
        instantiation_policy={"avoid_same_speaker": False},
        source_status=status,
    )


def test_categorical_field_spec_valid():
    assert field_spec().semantic_class == "categorical_attribute"


def test_numeric_field_spec_valid():
    assert field_spec("numeric_attribute", "age", "tuổi").unit == "tuổi"


def test_text_content_field_spec_valid():
    assert field_spec("text_content", "text").content_phrase == "nội dung được nói"


def test_unknown_semantic_class_rejected():
    with pytest.raises(ValidationError):
        SemanticFieldSpec(
            field_name="x",
            semantic_class="mystery",
            entity_scope="speaker",
            entity_phrase="người nói",
        )


def test_missing_required_verbalization_detected():
    with pytest.raises(ValidationError, match="ATTRIBUTE_PHRASE_MISSING"):
        SemanticFieldSpec(
            field_name="x",
            semantic_class="categorical_attribute",
            entity_scope="speaker",
            entity_phrase="người nói",
        )


def test_old_dataset_profile_remains_loadable():
    profile = DatasetProfile.model_validate_json(
        PROFILE_PATH.read_text(encoding="utf-8")
    )
    assert profile.dataset == "vimd"
    assert profile.fields["region"].semantic_class is None


def test_migration_sidecar_works(tmp_path):
    path = tmp_path / "sidecar.json"
    spec = field_spec()
    path.write_text(json.dumps({"field_specs": [spec.model_dump()]}), encoding="utf-8")
    loaded = load_migration_sidecar(path)
    assert resolve_field_spec("region", None, loaded) == spec


def test_no_raw_field_name_humanization_fallback():
    with pytest.raises(ValueError, match="FIELD_SPEC_MISSING:province_name"):
        resolve_field_spec("province_name", None, {})


def test_library_contains_no_dataset_specific_field_names():
    patterns = " ".join(item.pattern for item in library().blueprints).casefold()
    assert all(
        name.casefold() not in patterns
        for name in ("region", "province_name", "gender", "speakerid")
    )


def test_blueprint_ids_unique():
    ids = [item.blueprint_id for item in library().blueprints]
    assert len(ids) == len(set(ids))


def test_required_slots_valid():
    assert all(item.required_slots for item in library().blueprints)


def test_unknown_slot_rejected():
    with pytest.raises(ValidationError, match="UNKNOWN_BLUEPRINT_SLOT"):
        TemplateBlueprint(
            blueprint_id="bad",
            operator="DIRECT",
            semantic_class="categorical_attribute",
            pattern="[REGION] là gì?",
            required_slots=["[REGION]"],
            answer_kind="field_value",
        )


def test_lookup_is_deterministic():
    first = [
        x.blueprint_id for x in library().lookup("DIRECT", "categorical_attribute")
    ]
    second = [
        x.blueprint_id for x in library().lookup("DIRECT", "categorical_attribute")
    ]
    assert first == second == sorted(first)


def test_library_hash_stable():
    lib = library()
    assert lib.library_hash == lib.computed_hash()


def test_unsupported_combination_returns_blueprint_missing():
    spec = SemanticFieldSpec(
        field_name="rank",
        semantic_class="ordinal_attribute",
        entity_scope="speaker",
        entity_phrase="người nói",
        attribute_phrase="thứ hạng",
    )
    with pytest.raises(ValueError, match="BLUEPRINT_MISSING"):
        render_type_contract(library(), spec, contract(field="rank"))


@pytest.mark.parametrize(
    ("attribute", "expected"),
    [
        ("vùng phương ngữ", "Vùng phương ngữ"),
        ("giới tính", "Giới tính"),
        ("cảm xúc", "Cảm xúc"),
    ],
)
def test_same_direct_categorical_blueprint_renders_multiple_fields(attribute, expected):
    spec = field_spec()
    spec.attribute_phrase = attribute
    blueprint = library().lookup("DIRECT", "categorical_attribute")[0]
    rendered = render_blueprint(blueprint, spec, contract())
    assert rendered.question_pattern.startswith(expected)


def test_direct_numeric_renders_age_with_unit():
    spec = field_spec("numeric_attribute", "age", "tuổi")
    rendered = render_type_contract(library(), spec, contract(field="age"))
    assert len(rendered) == 2
    assert all("tuổi" in item.question_pattern for item in rendered)


def test_direct_numeric_unitless_variant():
    spec = field_spec("numeric_attribute", "score")
    rendered = render_type_contract(library(), spec, contract(field="score"))
    assert len(rendered) == 1 and "[UNIT]" not in rendered[0].question_pattern


def test_direct_text_renders_transcript_like_spec():
    spec = field_spec("text_content", "text")
    rendered = render_type_contract(library(), spec, contract(field="text"))
    assert len(rendered) == 2
    assert all(
        "nội dung được nói" in item.question_pattern.casefold() for item in rendered
    )


@pytest.mark.parametrize(
    "semantic_class,field",
    [("categorical_attribute", "region"), ("text_content", "text")],
)
def test_target_match_preserves_target_slot(semantic_class, field):
    rendered = render_type_contract(
        library(), field_spec(semantic_class, field), contract("TARGET_MATCH", field)
    )
    assert all("[TARGET_VALUE]" in item.question_pattern for item in rendered)


def test_pairwise_selection_categorical_has_ab_semantics():
    rendered = render_type_contract(
        library(), field_spec(), contract("PAIRWISE_SELECTION")
    )
    assert all(
        "A" in item.question_pattern and "B" in item.question_pattern
        for item in rendered
    )


def test_equality_categorical_preserves_two_audio_semantics():
    rendered = render_type_contract(library(), field_spec(), contract("EQUALITY"))
    assert all(
        "hai đoạn âm thanh" in item.question_pattern.casefold() for item in rendered
    )


def test_unresolved_slots_fail():
    blueprint = library().lookup("DIRECT", "categorical_attribute")[0]
    spec = field_spec()
    spec.attribute_phrase = None
    with pytest.raises(ValueError, match="REQUIRED_SLOT_MISSING"):
        render_blueprint(blueprint, spec, contract())


def test_rendering_imports_no_llm_client():
    source = Path("src/autonomous_qa/language/template_renderer.py").read_text(encoding="utf-8")
    assert "llm_client" not in source and "call_llm" not in source


def test_rendered_wording_preserves_semantic_identity():
    rendered = render_type_contract(library(), field_spec(), contract("TARGET_MATCH"))
    assert all(item.type_id == "type-TARGET_MATCH-region" for item in rendered)
    assert all(item.operator == "TARGET_MATCH" for item in rendered)
    assert all(item.semantic_class == "categorical_attribute" for item in rendered)


def test_target_match_does_not_drift_to_direct():
    rendered = render_type_contract(library(), field_spec(), contract("TARGET_MATCH"))
    assert all("không?" in item.question_pattern for item in rendered)


def test_selection_does_not_drift_to_yes_no():
    rendered = render_type_contract(
        library(), field_spec(), contract("PAIRWISE_SELECTION")
    )
    assert all(
        "đoạn nào" in item.question_pattern or "chọn đoạn" in item.question_pattern
        for item in rendered
    )


def test_equality_does_not_drift_to_direct():
    rendered = render_type_contract(library(), field_spec(), contract("EQUALITY"))
    assert all("hai đoạn" in item.question_pattern.casefold() for item in rendered)


def test_status_axes_are_separate():
    item = render_type_contract(library(), field_spec(), contract("EQUALITY"))[0]
    item.preview_capacity_status = "INSUFFICIENT_POSITIVE_CAPACITY"
    item.full_train_capacity_status = "NOT_AUDITED"
    item.generation_status = "WAITING_FULL_TRAIN_CAPACITY_AUDIT"
    assert item.template_status == "PASS"
    assert item.preview_capacity_status == "INSUFFICIENT_POSITIVE_CAPACITY"
    assert item.full_train_capacity_status == "NOT_AUDITED"
    assert item.generation_status == "WAITING_FULL_TRAIN_CAPACITY_AUDIT"


def test_unsupported_semantic_field_cannot_be_production_eligible():
    with pytest.raises(ValueError, match="SEMANTIC_FIELD_NOT_SUPPORTED"):
        render_type_contract(
            library(), field_spec(), contract(status="REVIEW_REQUIRED")
        )


def test_template_support_does_not_override_unresolved_encoding():
    spec = field_spec()
    spec.semantic_status = "UNSUPPORTED_ENCODING"
    with pytest.raises(ValueError, match="SEMANTIC_FIELD_NOT_SUPPORTED"):
        render_type_contract(library(), spec, contract())


def test_render_100_fake_fields_zero_llm_calls():
    blueprint = library().lookup("DIRECT", "categorical_attribute")[0]
    rendered = []
    for index in range(100):
        spec = field_spec(field=f"field_{index}")
        rendered.append(
            render_blueprint(blueprint, spec, contract(field=f"field_{index}"))
        )
    assert len(rendered) == 100


def test_render_10000_templates_zero_llm_calls():
    blueprint = library().lookup("DIRECT", "categorical_attribute")[0]
    spec = field_spec()
    item = contract()
    rendered = [render_blueprint(blueprint, spec, item) for _ in range(10_000)]
    assert len(rendered) == 10_000


def test_preview_instantiation_zero_llm_calls(tmp_path):
    rows = []
    for index, region in enumerate(("North", "North", "South", "South")):
        audio = tmp_path / f"{index}.wav"
        audio.write_bytes(b"RIFF")
        rows.append(
            {
                "split": "train",
                "sample_id": str(index),
                "audio_path": str(audio),
                "metadata": {"region": region, "speakerID": f"s{index}"},
            }
        )
    rendered = render_type_contract(library(), field_spec(), contract("EQUALITY"))[0]
    records = instantiate_type(
        contract("EQUALITY"),
        rendered.as_legacy_template(),
        rows,
        {"instances": 8, "seed": 42},
    )
    assert len(records) == 8


def test_vimd_migration_has_only_three_explicit_specs():
    assert set(vimd_field_specs()) == {"region", "province_name", "text"}


def test_library_schema_rejects_duplicate_ids():
    blueprint = library().blueprints[0].model_dump()
    with pytest.raises(ValidationError, match="DUPLICATE_BLUEPRINT_ID"):
        TemplateLibrary(
            language="vi",
            version="x",
            schema_version=1,
            created_from="test",
            library_hash="x",
            blueprints=[blueprint, blueprint],
        )


def test_rendered_template_model_keeps_three_status_axes():
    fields = RenderedTemplate.model_fields
    assert {
        "template_status",
        "preview_capacity_status",
        "full_train_capacity_status",
        "generation_status",
    } <= set(fields)

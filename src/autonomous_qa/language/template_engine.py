"""Canonical template engine and semantic field specifications."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.common.config import ROOT
from src.autonomous_qa.compiler.semantic_field_specs import SemanticFieldSpec
from src.autonomous_qa.language.template_contracts import (
    TypeContract,
    build_type_contracts,
    instantiate_type,
    load_train_rows,
    model_facing,
    validate_instance,
    validate_operator_output,
    write_json,
    write_jsonl,
)
from src.autonomous_qa.language.template_renderer import (
    RenderedTemplate,
    TemplateLibrary,
    load_library,
    render_type_contract,
)



def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_tree(path: Path) -> dict[str, str]:
    return {
        item.relative_to(path).as_posix(): sha256_file(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }




def vimd_field_specs() -> dict[str, SemanticFieldSpec]:
    """Canonical ViMD semantic field specifications."""
    specs = [
        SemanticFieldSpec(
            field_name="region",
            semantic_class="categorical_attribute",
            entity_scope="speaker",
            entity_phrase="người nói",
            attribute_phrase="vùng phương ngữ",
            value_phrase="vùng phương ngữ",
            value_policy={"normalization": "identity", "match_policy": "exact"},
            source="legacy_profile_migration",
        ),
        SemanticFieldSpec(
            field_name="province_name",
            semantic_class="categorical_attribute",
            entity_scope="speaker",
            entity_phrase="người nói",
            attribute_phrase="phương ngữ tỉnh/thành",
            value_phrase="phương ngữ tỉnh/thành",
            value_policy={"normalization": "identity", "match_policy": "exact"},
            source="legacy_profile_migration",
        ),
        SemanticFieldSpec(
            field_name="text",
            semantic_class="text_content",
            entity_scope="utterance",
            entity_phrase="đoạn âm thanh",
            content_phrase="nội dung được nói",
            value_phrase="nội dung",
            value_policy={
                "normalization": "normalized_text_exact",
                "match_policy": "exact",
            },
            rendering={"target_quote_style": "vietnamese_quotes"},
            source="legacy_profile_migration",
        ),
    ]
    return {item.field_name: item for item in specs}


def vimedcss_field_specs() -> dict[str, SemanticFieldSpec]:
    """Canonical ViMedCSS semantic field specifications."""
    specs = [
        SemanticFieldSpec(
            field_name="segment_text",
            semantic_class="text_content",
            entity_scope="utterance",
            entity_phrase="đoạn âm thanh",
            content_phrase="nội dung được nói",
            value_phrase="nội dung",
            value_policy={
                "normalization": "normalized_text_exact",
                "match_policy": "exact",
            },
            source="canonical_spec",
        ),
        SemanticFieldSpec(
            field_name="cs_terms_list",
            semantic_class="categorical_attribute",
            entity_scope="utterance",
            entity_phrase="đoạn âm thanh",
            attribute_phrase="thuật ngữ y khoa",
            value_phrase="thuật ngữ",
            value_policy={"normalization": "identity", "match_policy": "exact"},
            source="canonical_spec",
        ),
        SemanticFieldSpec(
            field_name="topic",
            semantic_class="categorical_attribute",
            entity_scope="utterance",
            entity_phrase="đoạn âm thanh",
            attribute_phrase="chủ đề y khoa",
            value_phrase="chủ đề",
            value_policy={"normalization": "identity", "match_policy": "exact"},
            source="canonical_spec",
        ),
        SemanticFieldSpec(
            field_name="cs_terms_count",
            semantic_class="numeric_attribute",
            entity_scope="utterance",
            entity_phrase="đoạn âm thanh",
            attribute_phrase="số từ chuyển ngữ",
            value_phrase="số từ",
            value_policy={"normalization": "identity", "match_policy": "exact"},
            source="canonical_spec",
        ),
    ]
    return {item.field_name: item for item in specs}



def _demo_contract(
    type_id: str, operator: str, field: str, answer_kind: str
) -> TypeContract:
    return TypeContract(
        type_id=type_id,
        operator=operator,
        semantic_field=field,
        semantic_description="rendering demonstration only",
        semantic_phrase_vi="rendering demonstration",
        entity_scope="speaker",
        audio_input_count=1,
        condition_fields=[],
        gold_source_fields=[field],
        answer_kind=answer_kind,
        answer_mode="FIELD_VALUE" if operator == "DIRECT" else "BOOLEAN",
        template_status="ELIGIBLE",
        template_policy={},
        instantiation_policy={},
        source_status="SUPPORTED",
    )


def future_rendering_demos(library: TemplateLibrary) -> list[dict[str, Any]]:
    examples = [
        SemanticFieldSpec(
            field_name="demo_region",
            semantic_class="categorical_attribute",
            entity_scope="speaker",
            entity_phrase="người nói",
            attribute_phrase="vùng phương ngữ",
            value_phrase="vùng phương ngữ",
        ),
        SemanticFieldSpec(
            field_name="demo_gender",
            semantic_class="categorical_attribute",
            entity_scope="speaker",
            entity_phrase="người nói",
            attribute_phrase="giới tính",
            value_phrase="giới tính",
        ),
        SemanticFieldSpec(
            field_name="demo_emotion",
            semantic_class="categorical_attribute",
            entity_scope="speaker",
            entity_phrase="người nói",
            attribute_phrase="cảm xúc",
            value_phrase="cảm xúc",
        ),
        SemanticFieldSpec(
            field_name="demo_age",
            semantic_class="numeric_attribute",
            entity_scope="speaker",
            entity_phrase="người nói",
            attribute_phrase="độ tuổi",
            value_phrase="tuổi",
            unit="tuổi",
        ),
    ]
    rows = []
    for spec in examples:
        contract = _demo_contract(
            f"demo-{spec.field_name}", "DIRECT", spec.field_name, "field_value"
        )
        rendered = render_type_contract(library, spec, contract)
        rows.append(
            {
                "field": spec.field_name,
                "semantic_class": spec.semantic_class,
                "example_question_patterns": [
                    item.question_pattern for item in rendered
                ],
                "qa_eligibility_claimed": False,
                "note": "Rendering capability demo only; a real TypeRegistry remains authoritative.",
            }
        )
    return rows







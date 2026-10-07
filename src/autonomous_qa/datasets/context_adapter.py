"""Pipeline entrypoint (MVP v3: full discovery loop).

Default (`--dataset vimd`) is MOCK: 0 real calls, fixtures drive the flow.
Only `--real-llm` creates the Vertex client and spends batched calls.

Output roots (never mixed):
- mock: outputs/runs/<dataset>/mock/
- real: outputs/runs/<dataset>/<run_id>/   (unique per run, never overwritten)

Each round runs Style -> Template -> Paraphrase -> Quality (4 calls).
A Python validity gate sits between Template and Paraphrase: deterministic
errors never spend Paraphrase/Quality calls. Saturation is measured on
question-TYPE discovery (new_type_rate), never on wording.

Checkpoints: run_state.json is saved after every successfully persisted
stage. A crashed real run resumes with:
    python -m src.run_pipeline --dataset vimd --real-llm --resume-run <run_id>
Resume reuses every completed artifact byte-for-byte and only issues the
next unfinished LLM request.

Budget counts PAID attempts (failed parses included): design needs
4R successes, plus exactly one manual-retry attempt per stage.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from src.common.config import CONFIG, ROOT
from src.autonomous_qa.core.loop import (
    LoopController,
    apply_accepted_types,
    build_round_stats,
    build_type_fewshot,
    filter_paraphrases_for_judging,
    flatten_type_bindings,
    validate_quality_partition,
)
from src.autonomous_qa.language.family_templates import (
    FAMILY_EQUALITY_SEMANTIC,
    FAMILY_EQUALITY_SEMANTIC_SPEAKER,
    FAMILY_EQUALITY_SEMANTIC_UTTERANCE,
    bind_round_family_templates,
    build_family_bank_entry,
    deserialize_family_signature,
    serialize_family_signature,
)
from src.autonomous_qa.language.paraphrase import run_paraphrase_round
from src.autonomous_qa.certification.quality import run_quality_round
from src.autonomous_qa.language.question_style import run_question_style_round
from src.autonomous_qa.language.question_template import load_template_bank, run_question_template_round
from src.autonomous_qa.production.render_preview import render_preview
from src.common.io import read_json, write_json
from src.autonomous_qa.core.validity import (
    DEFAULT_FIELD_ROLES,
    DEFAULT_ENTITY_SCOPES,
    ROLE_HIDDEN_IDENTIFIER,
    SCHEMA_FIELDS,
    extract_field_roles,
    extract_entity_scopes,
    extract_source_missing_values,
    extract_evaluation_policies,
    normalize_row_missing_values,
    required_answer_fields,
    is_evaluation_eligible_type,
    gate_round,
    get_answer_concept_field,
    template_family_signature,
)


RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

STATE_FILE = "run_state.json"

# Question Template LLM contract fingerprint (§39, §41, §AE). Pre-patch runs stored
# raw-field-keyed key_realizations or un-scoped families and MUST refuse resume
# clearly as incompatible (no migration, no reinterpretation).
TEMPLATE_CONTRACT_VERSION = "operation-family-scope-context-v3"

# Schema policy contract fingerprint: entity scope and evaluation policy metadata
SCHEMA_POLICY_CONTRACT_VERSION = "schema-policy-v1"

# Answer-reference contract fingerprint (§31). Runs produced under the OLD
# broad suffix-normalization gate semantics MUST refuse resume/replay as
# incompatible (no migration, no reinterpretation).
ANSWER_REFERENCE_CONTRACT_VERSION = "kind-scoped-v1"

# Canonical answer-shape contract fingerprint: each answer kind owns its
# fields; irrelevant non-null fields deterministically reject the type.
ANSWER_SHAPE_CONTRACT_VERSION = "canonical-v1"

# Executable-signature dedup contract fingerprint: identical executable
# signatures are deterministic duplicates before Quality.
EXECUTABLE_SIGNATURE_CONTRACT_VERSION = "v1"

# Derived-field contract fingerprint: derived relations are validated
# from dataset evidence (data-backed-v1), never from field membership
# alone and never from a hard-coded relation list.
DERIVED_FIELD_CONTRACT_VERSION = "data-backed-v1"

# Vietnamese linguistic knowledge layer contract fingerprint (§51, §61).
KNOWLEDGE_CONTRACT_VERSION = "vi-knowledge-v1"


def load_readme(dataset: str) -> str:
    path = ROOT / "data" / dataset / "README.md"
    return path.read_text(encoding="utf-8")


def load_sample(dataset: str) -> list[dict]:
    rows: list[dict] = []
    path = ROOT / "data" / dataset / "sample.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_metadata_context(dataset: str) -> dict | None:
    path = ROOT / "data" / dataset / "metadata_context.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def load_knowledge_context(dataset: str) -> dict | None:
    path = ROOT / "data" / dataset / "knowledge_context.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def dataset_manifest_path(dataset: str) -> Path:
    """Resolve the canonical metadata artifact path for a dataset."""
    ds_cfg = CONFIG.get("datasets", {}).get(dataset, {}) if isinstance(CONFIG, dict) else {}
    manifest_name = ds_cfg.get("manifest", "manifest.jsonl")
    manifest_dir = ds_cfg.get("dir")
    if manifest_dir:
        return Path(manifest_dir) / manifest_name
    return ROOT / "data" / dataset / manifest_name


def load_dataset_rows(dataset: str) -> list[dict]:
    """Load canonical normalized structured metadata rows for a dataset.

    Returns structured rows (audio, text, gender, region, province, speakerID)
    without decoding WAV files, loading audio bytes, or making external calls.
    If the canonical production metadata artifact does not exist:
    - Never falls back to sample.jsonl
    - Emits: structured_dataset_rows_unavailable:<dataset>
    - Returns [] (empty list) so callers fail closed on relation validation.
    """
    path = dataset_manifest_path(dataset)
    if not path.is_file():
        print(f"structured_dataset_rows_unavailable:{dataset}")
        return []

    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dataset_schema_path(dataset: str) -> Path:
    """Resolve the canonical schema artifact path for a dataset."""
    ds_cfg = CONFIG.get("datasets", {}).get(dataset, {}) if isinstance(CONFIG, dict) else {}
    schema_name = ds_cfg.get("schema", "schema.json")
    schema_dir = ds_cfg.get("dir")
    if schema_dir:
        return Path(schema_dir) / schema_name
    return ROOT / "data" / dataset / schema_name


def load_schema(dataset: str) -> dict:
    path = dataset_schema_path(dataset)
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dataset_schema_fields(dataset: str) -> frozenset[str]:
    schema = load_schema(dataset)
    if schema and "fields" in schema:
        return frozenset(schema["fields"].keys())
    sample = load_sample(dataset)
    if sample:
        return frozenset(sample[0].keys())
    rows = load_dataset_rows(dataset)
    if rows:
        return frozenset(rows[0].keys())
    return SCHEMA_FIELDS


def dataset_field_roles(dataset: str) -> dict[str, str]:
    schema = load_schema(dataset)
    return extract_field_roles(schema)


def dataset_hidden_fields(dataset: str) -> frozenset[str]:
    roles = dataset_field_roles(dataset)
    if roles:
        return frozenset(f for f, r in roles.items() if r == ROLE_HIDDEN_IDENTIFIER)
    schema = load_schema(dataset)
    if schema and "hidden_fields" in schema:
        return frozenset(schema["hidden_fields"])
    return frozenset(f for f, r in DEFAULT_FIELD_ROLES.items() if r == ROLE_HIDDEN_IDENTIFIER)


def dataset_entity_scopes(dataset: str) -> dict[str, str]:
    schema = load_schema(dataset)
    return extract_entity_scopes(schema)


def dataset_source_missing_values(dataset: str) -> dict[str, list[str]]:
    schema = load_schema(dataset)
    return extract_source_missing_values(schema)


def dataset_evaluation_policies(dataset: str) -> dict[str, str]:
    schema = load_schema(dataset)
    return extract_evaluation_policies(schema)


def filter_evaluation_eligible_types(
    types: list[dict],
    evaluation_policies: dict[str, str],
    schema_fields: frozenset[str] | None = None,
) -> list[dict]:
    """Filter QA types to only those eligible for evaluation/benchmark pool (§E)."""
    return [
        t for t in types
        if is_evaluation_eligible_type(t, evaluation_policies, schema_fields)
    ]


def extract_representative_values(
    rows: list[dict],
    schema: dict,
    max_values_per_field: int = 5,
) -> dict[str, list[str]]:
    """Small set of distinct non-null categorical values for semantic fields (§W, §X).

    Context only for Question Template prompt. Never leaks into question text or [VALUE].
    """
    fields = schema.get("fields", {}) if isinstance(schema, dict) else {}
    rep: dict[str, list[str]] = {}
    for f, meta in fields.items():
        if isinstance(meta, dict) and meta.get("role") == "semantic" and meta.get("is_categorical"):
            seen = set()
            val_list = []
            for r in rows:
                v = r.get(f)
                if v is not None:
                    v_str = str(v).strip()
                    if v_str and v_str not in seen:
                        seen.add(v_str)
                        val_list.append(v_str)
                        if len(val_list) >= max_values_per_field:
                            break
            if val_list:
                rep[f] = val_list
    return rep


"""Dataset semantic profiling: source card -> ONE structured LLM call -> profile.

Architecture (this is the new input stage for the QA pipeline):

    dataset_card.md  (sole semantic authority)
            |
            v
    one structured LLM call
            |
            v
    dataset_profile.json        <-- canonical source, the ONLY LLM product
            |
            +--> Python deterministic validator (obeys the card or fails)
            |
            +--> schema.json            (compatibility export)
            +--> metadata_context.json  (compatibility export)

Because both compatibility files are exported from the SAME validated
profile, they cannot drift apart the way two independent LLM calls could.

Generic on purpose: this module contains no dataset-specific field names,
roles or rules. Everything dataset-shaped comes from the source card
(raw feature names + dtypes) and from the canonical vocabularies already
defined in src.autonomous_qa.core.validity. Dataset-specific review hints are derived from
the card's own text, never hard-coded.

Outputs land in a fresh, never-reused run directory:

    outputs/runs/<dataset>/<run_id>/profile/
        source_card.md
        raw_response.txt
        dataset_profile.json
        schema.json
        metadata_context.json
        validation_report.json
        run_meta.json

Mock mode (default) makes 0 network calls. Real mode makes exactly one
call with the project's existing OpenAI-compatible client: same model,
same no-retry policy, no alternate SDK. Any failure stops the run and is
reported verbatim; nothing is silently changed and retried.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from src.common.config import ROOT, resolve_llm_base_url, resolve_llm_model
from src.autonomous_qa.core.schemas import DatasetProfile
from src.common.io import load_prompt, write_json
from src.autonomous_qa.core.validity import (
    CANONICAL_CONTEXT_VISIBILITIES,
    CANONICAL_ENTITY_SCOPES,
    CANONICAL_EVALUATION_POLICIES,
    CANONICAL_FIELD_ROLES,
)

# Single profiling request settings. Fixed here on purpose: a failed call
# must never trigger a silent change of model / temperature / schema /
# prompt / token budget. The model always comes from the shared resolver in
# src.config (OPENAI_COMPAT_MODEL > OPENAI_MODEL > config.yaml).
PROFILE_TEMPERATURE = 0.2
PROFILE_MAX_OUTPUT_TOKENS = 8192
PROFILE_THINKING_BUDGET = 2048

PROMPT_NAME = "dataset_profile"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "runs"

EXIT_OK = 0
EXIT_VALIDATION_FAILED = 1
EXIT_CALL_FAILED = 2


# ---------------------------------------------------------------------------
# source card
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceCard:
    """Parsed dataset card: raw feature names, dtypes, splits, body text."""

    path: Path
    text: str
    front_matter: dict
    fields: dict[str, str]  # raw field name -> declared dtype (card order)
    splits: list[dict] = dataclass_field(default_factory=list)
    field_descriptions: dict[str, str] = dataclass_field(default_factory=dict)
    pretty_name: str | None = None
    task_categories: list[str] = dataclass_field(default_factory=list)
    language: str | None = None

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def total_rows(self) -> int | None:
        total = 0
        for split in self.splits:
            n = split.get("num_examples")
            if not isinstance(n, int):
                return None
            total += n
        return total if self.splits else None


def _split_front_matter(text: str) -> tuple[dict, str]:
    """Return (yaml front matter, markdown body) of a card file."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            yaml_text = "\n".join(lines[1:idx])
            body = "\n".join(lines[idx + 1:])
            data = yaml.safe_load(yaml_text) or {}
            if not isinstance(data, dict):
                raise ValueError("Source card front matter is not a mapping.")
            return data, body
    raise ValueError("Source card has an unterminated front matter block.")


def _parse_field_descriptions(body: str) -> dict[str, str]:
    """Parse `- <field>:` bullets (+ indented continuation) from the body."""
    out: dict[str, str] = {}
    current: str | None = None
    bullet = re.compile(r"^- ([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$")
    for line in body.splitlines():
        m = bullet.match(line)
        if m:
            current = m.group(1)
            out[current] = m.group(2).strip()
            continue
        if current is None:
            continue
        if not line.strip():
            continue
        if line[:1] in (" ", "\t"):
            out[current] = (out[current] + " " + line.strip()).strip()
        else:
            current = None
    return out


def parse_source_card(path: Path | str) -> SourceCard:
    """Read a dataset card and extract its declared features and splits."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    front_matter, body = _split_front_matter(text)

    info = front_matter.get("dataset_info") or {}
    features = info.get("features") or []
    fields: dict[str, str] = {}
    for feat in features:
        if not isinstance(feat, dict) or "name" not in feat:
            raise ValueError(f"Malformed feature entry in card: {feat!r}")
        raw_dtype = feat.get("dtype", "")
        if isinstance(raw_dtype, dict):
            if "audio" in raw_dtype:
                dtype_str = "audio"
            else:
                dtype_str = str(next(iter(raw_dtype.keys()), "")).strip()
        else:
            dtype_str = str(raw_dtype).strip()
        fields[str(feat["name"])] = dtype_str

    splits = [s for s in (info.get("splits") or []) if isinstance(s, dict)]

    language = front_matter.get("language")
    if isinstance(language, list):
        language = language[0] if language else None

    task_categories = front_matter.get("task_categories") or []
    if not isinstance(task_categories, list):
        task_categories = [task_categories]

    if not fields:
        raise ValueError(f"Source card declares no features: {path}")

    return SourceCard(
        path=path,
        text=text,
        front_matter=front_matter,
        fields=fields,
        splits=splits,
        field_descriptions=_parse_field_descriptions(body),
        pretty_name=front_matter.get("pretty_name"),
        task_categories=[str(t) for t in task_categories],
        language=str(language) if language else None,
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------

# The instructions file ends with this marker; the card follows it verbatim,
# so the response-contract block is injected immediately BEFORE it.
PROFILE_SCHEMA_MARKER = "--- SOURCE CARD (SOLE SEMANTIC AUTHORITY) ---"


def profile_contract_block(dataset: str | None = None) -> str:
    """Machine-generated contract: exact key names + JSON schema.

    Generic on purpose: it is built from src.schemas.DatasetProfile, never
    from a dataset. The gateway does not reliably enforce
    `response_format`, so the schema travels inside the prompt where the
    model can always read it.
    """
    schema = json.dumps(DatasetProfile.model_json_schema(),
                        ensure_ascii=False, indent=2)
    lines = [
        "--- RESPONSE SCHEMA (EXACT KEY NAMES; ENFORCED BY PYTHON) ---",
        "",
        "Your object is validated against the JSON schema below. The",
        "endpoint may not deliver structured output, so the schema is",
        "repeated here. No aliases are accepted.",
        "",
        "Top-level key names (exactly these — `dataset`, never",
        "`dataset_name`; `pretty_name`, never `title`):",
        "  dataset, pretty_name, description, locale, task_categories,",
        "  fields, field_roles, entity_scopes, field_evaluations,",
        "  context_visibilities, hidden_fields, semantic_fields,",
        "  relations, qa_constraints",
        "",
        "Per-field key names (exactly these, inside every entry of",
        "`fields`):",
        "  type, role, entity_scope, description, is_categorical,",
        "  evaluation, context_visibility",
        "",
        "Deterministic derivations Python recomputes and rejects on any",
        "mismatch:",
        "- `semantic_fields` = exactly the field names whose `role` is",
        "  `semantic`.",
        "- `hidden_fields` = exactly the field names whose `role` is",
        "  `hidden_identifier`. No provenance, context_only, semantic or",
        "  audio field may appear there.",
        "- `field_roles`, `entity_scopes`, `field_evaluations` and",
        "  `context_visibilities` repeat the per-field value of every",
        "  field, one entry per field.",
        "- `pretty_name`, `locale`, `task_categories` and `description`",
        "  come from the source card, never invented.",
        "",
    ]
    if dataset:
        lines.append(f'- `dataset` must be exactly "{dataset}".')
        lines.append("")
    lines.append("JSON schema (authoritative):")
    lines.append(schema)
    return "\n".join(lines)


def build_prompt(card: SourceCard, instructions: str | None = None,
                 dataset: str | None = None) -> str:
    """Instructions + response contract + the verbatim source card."""
    if instructions is None:
        instructions = load_prompt(PROMPT_NAME)

    block = profile_contract_block(dataset)
    # rpartition: the intro sentence quotes the marker too, so only the
    # LAST occurrence (the standalone line at the end) introduces the card.
    head, marker, tail = instructions.rpartition(PROFILE_SCHEMA_MARKER)
    if marker:
        instructions = f"{head}{block}\n{marker}{tail}"
    else:
        instructions = f"{instructions.rstrip()}\n\n{block}"

    parts = [instructions.rstrip("\n"), "", card.text]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# validation — structural only, never a second semantic layer
# ---------------------------------------------------------------------------

_FORBIDDEN_PROFILE_KEYS = frozenset({
    "values", "allowed_values", "example_values", "sample_values",
    "examples", "sample_rows", "sample_row", "sample_transcripts",
    "sample_filenames", "transcripts", "filenames", "speaker_ids",
    "province_list", "provinces", "lookup_table", "lookup", "mapping",
    "mappings", "gender_map", "label_map", "label_mapping", "class_map",
    "observed_cardinality", "observed_values", "distinct_values",
    "row_values", "gold_values",
})

# Words that make a "0 -> X" pattern harmless prose rather than a label
# claim (e.g. "what 0 or 1 means").
_MAPPING_LABEL_STOPWORDS = frozenset({
    "the", "a", "an", "or", "and", "of", "to", "in", "is", "are", "was",
    "were", "not", "no", "any", "this", "that", "it", "as", "by", "for",
    "on", "be", "if", "then", "value", "values", "code", "codes",
    "number", "numbers", "field", "fields", "one", "two", "zero", "type",
})

_NUM_TO_LABEL = re.compile(
    r"(?<![\w.\-])(-?\d+(?:\.\d+)?)\s*"
    r"(?:=|→|->|:|means|denotes|represents)"
    r"\s*[\"']?([A-Za-z][\w\-/]{0,40}?)[\"']?"
    r"(?=$|[,;.!?()\]}\"']|\s)",
    re.IGNORECASE,
)

_LABEL_TO_NUM = re.compile(
    r"(?<![\w\-])([A-Za-z][\w \-/]{0,40}?)\s*"
    r"(?:=|→|->|means|denotes|represents)"
    r"\s*(-?\d+(?:\.\d+)?)(?![\d.])",
    re.IGNORECASE,
)

_ENUM_SEGMENT = re.compile(r"^[A-ZÀ-Ỹ][\wÀ-ỹ'’\- ]{0,40}$")


def _normalize_mapping(num: str, label: str) -> tuple[str, str]:
    try:
        num_norm = str(float(num))
        num_norm = num_norm.removesuffix(".0")
    except ValueError:
        num_norm = num.strip()
    return num_norm, " ".join(label.split()).strip(" \"'").casefold()


def extract_label_mappings(text: str) -> set[tuple[str, str]]:
    """Numeric-literal -> text-label assertions found in free text."""
    found: set[tuple[str, str]] = set()
    for m in _NUM_TO_LABEL.finditer(text):
        num, label = m.group(1), m.group(2)
        if label.casefold() in _MAPPING_LABEL_STOPWORDS:
            continue
        if len(label.split()) > 4:
            continue
        found.add(_normalize_mapping(num, label))
    for m in _LABEL_TO_NUM.finditer(text):
        label, num = m.group(1), m.group(2)
        if label.casefold() in _MAPPING_LABEL_STOPWORDS:
            continue
        if len(label.split()) > 4:
            continue
        found.add(_normalize_mapping(num, label))
    return found


def _walk_strings(node: Any, path: str = "$"):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk_strings(value, f"{path}.{key}")
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            yield from _walk_strings(value, f"{path}[{idx}]")
    elif isinstance(node, str):
        yield path, node


def _walk_keys(node: Any, path: str = "$"):
    if isinstance(node, dict):
        for key, value in node.items():
            yield path, str(key)
            yield from _walk_keys(value, f"{path}.{key}")
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            yield from _walk_keys(value, f"{path}[{idx}]")


def _find_enumerated_lists(profile: dict) -> list[str]:
    """A long, all-capitalized comma list is an invented value list."""
    hits: list[str] = []
    for path, text in _walk_strings(profile):
        if any(ch in text for ch in ".!?"):
            continue
        segments = [s.strip() for s in re.split(r"[,;]", text)]
        if len(segments) < 5:
            continue
        if all(_ENUM_SEGMENT.match(s) for s in segments):
            hits.append(path)
    return hits


def _is_raw_field_rename(name: str, expected: set[str]) -> str | None:
    """Return the expected raw name when `name` is a cosmetic rename of it."""

    def norm(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.lower())

    target = norm(name)
    for exp in expected:
        if norm(exp) == target:
            return exp
    return None


def _card_supports_number(card: SourceCard, number: int) -> bool:
    """True when the card itself states this number (card-declared only)."""
    return re.search(rf"(?<!\d){re.escape(str(number))}(?!\d)",
                     card.text) is not None


def _bad_cardinalities(node: Any, card: SourceCard,
                       path: str = "$") -> list[str]:
    """A stored cardinality must be card-declared, never row-observed."""
    hits: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{path}.{key}"
            if str(key).casefold() == "cardinality":
                if not (isinstance(value, int)
                        and _card_supports_number(card, value)):
                    hits.append(f"{child}:cardinality_not_card_declared:"
                                f"{value!r}")
            else:
                hits.extend(_bad_cardinalities(value, card, child))
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            hits.extend(_bad_cardinalities(value, card, f"{path}[{idx}]"))
    return hits


def validate_profile(profile: dict, card: SourceCard) -> dict:
    """Deterministic contract check of a generated profile against the card.

    Returns a JSON-serializable report. `ok` is True only when every check
    passes. This layer never repairs or reinterprets semantics.
    """
    checks: dict[str, dict] = {}
    errors: list[str] = []

    def record(name: str, ok: bool, **detail: Any) -> None:
        checks[name] = {"ok": bool(ok), **detail}
        if not ok:
            for message in detail.get("messages", []):
                errors.append(f"{name}:{message}")

    if not isinstance(profile, dict):
        return {"ok": False, "checks": {}, "errors": ["profile_not_an_object"]}

    fields = profile.get("fields")
    fields = fields if isinstance(fields, dict) else {}
    actual = set(fields)
    expected = set(card.fields)

    # --- A. exact field set (no missing, no invented, no renamed) -----------
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    renamed: dict[str, str] = {}
    invented: list[str] = []
    for name in extra:
        source = _is_raw_field_rename(name, expected)
        if source is None:
            invented.append(name)
        else:
            renamed[name] = source
    record(
        "field_set",
        not missing and not extra,
        expected=sorted(expected),
        actual=sorted(actual),
        missing=missing,
        renamed=renamed,
        invented=invented,
        messages=[f"missing:{f}" for f in missing]
        + [f"renamed:{k}->{v}" for k, v in sorted(renamed.items())]
        + [f"invented:{f}" for f in invented],
    )

    # --- B. source dtype agreement -----------------------------------------
    dtype_mismatches: dict[str, dict] = {}
    for name in sorted(actual & expected):
        meta = fields.get(name) if isinstance(fields.get(name), dict) else {}
        got = meta.get("type")
        want = card.fields[name]
        if not isinstance(got, str) or got.strip() != want:
            dtype_mismatches[name] = {"expected": want, "actual": got}
    record(
        "dtype_agreement",
        not dtype_mismatches,
        mismatches=dtype_mismatches,
        messages=[f"{k}:expected={v['expected']!r},actual={v['actual']!r}"
                  for k, v in dtype_mismatches.items()],
    )

    # --- C. vocabulary ------------------------------------------------------
    vocab_problems: list[str] = []
    for name, meta in sorted(fields.items()):
        if not isinstance(meta, dict):
            vocab_problems.append(f"{name}:not_an_object")
            continue
        role = meta.get("role")
        if role not in CANONICAL_FIELD_ROLES:
            vocab_problems.append(f"{name}:role={role!r}")
        scope = meta.get("entity_scope")
        if scope is not None and scope not in CANONICAL_ENTITY_SCOPES:
            vocab_problems.append(f"{name}:entity_scope={scope!r}")
        evaluation = meta.get("evaluation")
        if evaluation not in CANONICAL_EVALUATION_POLICIES:
            vocab_problems.append(f"{name}:evaluation={evaluation!r}")
        visibility = meta.get("context_visibility")
        if visibility not in CANONICAL_CONTEXT_VISIBILITIES:
            vocab_problems.append(f"{name}:context_visibility={visibility!r}")
    record("vocabulary", not vocab_problems, messages=vocab_problems)

    # --- D. consistency between fields[*] and the mirror maps ---------------
    consistency_problems: list[str] = []

    def _check_map(key: str, attr: str, allow_none: bool = False) -> None:
        mirror = profile.get(key)
        if not isinstance(mirror, dict):
            consistency_problems.append(f"{key}:not_a_map")
            return
        if set(mirror) != actual:
            consistency_problems.append(
                f"{key}:key_mismatch:{sorted(set(mirror) ^ actual)}")
        for name in sorted(actual & set(mirror)):
            meta = fields.get(name)
            meta = meta if isinstance(meta, dict) else {}
            if not allow_none and mirror.get(name) != meta.get(attr):
                consistency_problems.append(
                    f"{key}:{name}:expected={meta.get(attr)!r},"
                    f"actual={mirror.get(name)!r}")
            if allow_none and mirror.get(name) != meta.get(attr):
                consistency_problems.append(
                    f"{key}:{name}:expected={meta.get(attr)!r},"
                    f"actual={mirror.get(name)!r}")

    _check_map("field_roles", "role")
    _check_map("entity_scopes", "entity_scope", allow_none=True)
    _check_map("field_evaluations", "evaluation")
    _check_map("context_visibilities", "context_visibility")

    def _check_list(key: str, predicate) -> None:
        value = profile.get(key)
        if not isinstance(value, list):
            consistency_problems.append(f"{key}:not_a_list")
            return
        if len(value) != len(set(value)):
            consistency_problems.append(f"{key}:duplicate_entries")
        derived = sorted(n for n, meta in fields.items()
                         if isinstance(meta, dict) and predicate(meta))
        if sorted(value) != derived:
            consistency_problems.append(
                f"{key}:expected={derived},actual={sorted(value)}")

    _check_list("semantic_fields", lambda m: m.get("role") == "semantic")
    _check_list(
        "hidden_fields",
        lambda m: m.get("role") == "hidden_identifier")

    record("consistency", not consistency_problems,
           messages=consistency_problems)

    # --- E. label mappings not supported by the source card ----------------
    serialized = json.dumps(profile, ensure_ascii=False)
    profile_mappings = extract_label_mappings(serialized)
    card_mappings = extract_label_mappings(card.text)
    unsupported = sorted(profile_mappings - card_mappings)
    record(
        "unsupported_label_mappings",
        not unsupported,
        unsupported=[{"number": n, "label": l} for n, l in unsupported],
        messages=[f"number={n}->label={l!r}" for n, l in unsupported],
    )

    # --- F. invented values -------------------------------------------------
    invented_problems: list[str] = []
    for path, key in _walk_keys(profile):
        if key.casefold() in _FORBIDDEN_PROFILE_KEYS:
            invented_problems.append(f"{path}:forbidden_key:{key}")
    invented_problems.extend(_bad_cardinalities(profile, card))
    for path in _find_enumerated_lists(profile):
        invented_problems.append(f"{path}:enumerated_value_list")
    record("invented_values", not invented_problems,
           messages=invented_problems)

    # --- G. relation references --------------------------------------------
    relation_problems: list[str] = []
    relations = profile.get("relations")
    if relations is None:
        relations = []
    if not isinstance(relations, list):
        relation_problems.append("relations:not_a_list")
        relations = []
    for idx, rel in enumerate(relations):
        if not isinstance(rel, dict):
            relation_problems.append(f"[{idx}]:not_an_object")
            continue
        for slot in ("source_field", "target_field"):
            value = rel.get(slot)
            if not isinstance(value, str) or value not in expected:
                relation_problems.append(
                    f"[{idx}].{slot}:unknown_field:{value!r}")
        if not str(rel.get("description") or "").strip():
            relation_problems.append(f"[{idx}]:empty_description")
    record("relation_references", not relation_problems,
           messages=relation_problems)

    return {
        "ok": not errors,
        "checks": checks,
        "errors": errors,
        "field_count": len(expected),
        "source_card_sha256": card.sha256,
    }


def review_flags(profile: dict, card: SourceCard) -> list[dict]:
    """Heuristic contradiction flags (PROFILE_REVIEW_REQUIRED).

    Never repairs output. Derived from the card's own wording, so the
    checks stay generic: 'identifier', 'encoded', 'numeric code',
    'file name', 'waveform' in the card text constrain what the profile
    may claim about that field.
    """
    flags: list[dict] = []

    def _flag(field_name: str, rule: str, reason: str) -> None:
        flags.append({"field": field_name, "rule": rule, "reason": reason})

    if not isinstance(profile, dict):
        return flags
    fields = profile.get("fields") or {}
    for name, meta in fields.items():
        card_desc = card.field_descriptions.get(name, "").casefold()
        if not isinstance(meta, dict):
            continue
        role = meta.get("role")
        evaluation = meta.get("evaluation")

        if "waveform" in card_desc and role != "audio":
            _flag(name, "card_says_waveform", "card describes waveform evidence")
        if ("identifier" in card_desc or "anonymized" in card_desc) \
                and role == "semantic":
            _flag(name, "card_says_identifier", "card describes an identifier")
        if "numeric code" in card_desc and role == "semantic":
            _flag(name, "card_says_numeric_code", "card describes a numeric code")
        if ("file name" in card_desc or "filename" in card_desc) \
                and role == "semantic":
            _flag(name, "card_says_file_name", "card describes a file name")
        if "encoded" in card_desc and evaluation == "eligible":
            _flag(name, "card_says_encoded_unresolved",
                  "card marks the encoding as unresolved")
        if "encoded" in card_desc and extract_label_mappings(
                json.dumps(meta, ensure_ascii=False)) - \
                extract_label_mappings(card.text):
            _flag(name, "card_says_encoded_mapping_invented",
                  "profile asserts a label mapping the card does not define")
    return flags


# ---------------------------------------------------------------------------
# deterministic compatibility exports
# ---------------------------------------------------------------------------


def export_schema(profile: dict, card: SourceCard) -> dict:
    """schema.json, derived ONLY from a validated profile.

    Keeps the shape src.autonomous_qa.core.validity / src.run_pipeline already read
    (fields + field_roles + entity_scopes + field_evaluations +
    hidden_fields + semantic_fields + context_visibilities). No
    card-undeclared cardinality and no dataset-specific interpretation.
    """
    fields: dict[str, dict] = {}
    for name in sorted(profile["fields"]):
        meta = profile["fields"][name]
        entry: dict[str, Any] = {
            "type": meta["type"],
            "role": meta["role"],
        }
        if meta.get("entity_scope") is not None:
            entry["entity_scope"] = meta["entity_scope"]
        entry["description"] = meta.get("description", "")
        entry["is_categorical"] = bool(meta.get("is_categorical", False))
        entry["evaluation"] = meta["evaluation"]
        entry["context_visibility"] = meta["context_visibility"]
        fields[name] = entry

    return {
        "dataset": profile["dataset"],
        "description": profile.get("description", ""),
        "locale": profile.get("locale", ""),
        "fields": fields,
        "field_roles": {n: profile["fields"][n]["role"] for n in fields},
        "entity_scopes": {
            n: profile["fields"][n]["entity_scope"] for n in fields
            if profile["fields"][n].get("entity_scope") is not None
        },
        "field_evaluations": {
            n: profile["fields"][n]["evaluation"] for n in fields
        },
        "hidden_fields": sorted(profile.get("hidden_fields", [])),
        "semantic_fields": sorted(profile.get("semantic_fields", [])),
        "context_visibilities": {
            n: profile["fields"][n]["context_visibility"] for n in fields
        },
    }


def export_metadata_context(profile: dict, card: SourceCard) -> dict:
    """metadata_context.json, derived from the SAME validated profile.

    No test examples, no legacy rows, no old QA output. `total_rows` is
    labelled as dataset-card metadata (corpus declared by the card), and
    `sample_rows_count` stays 0 because no row has been materialized.
    """
    fields: dict[str, dict] = {}
    for name in sorted(profile["fields"]):
        meta = profile["fields"][name]
        entry: dict[str, Any] = {
            "type": meta["type"],
            "role": meta["role"],
        }
        if meta.get("entity_scope") is not None:
            entry["entity_scope"] = meta["entity_scope"]
        entry["description"] = meta.get("description", "")
        entry["is_categorical"] = bool(meta.get("is_categorical", False))
        entry["evaluation"] = meta["evaluation"]
        fields[name] = entry

    payload: dict[str, Any] = {
        "dataset": profile["dataset"],
        "description": profile.get("description", ""),
        "locale": profile.get("locale", ""),
        "total_rows": card.total_rows,
        "total_rows_source": "dataset_card",
        "sample_rows_count": 0,
        "fields": fields,
        "field_roles": {n: profile["fields"][n]["role"] for n in fields},
        "entity_scopes": {
            n: profile["fields"][n]["entity_scope"] for n in fields
            if profile["fields"][n].get("entity_scope") is not None
        },
        "semantic_fields": sorted(profile.get("semantic_fields", [])),
        "hidden_fields": sorted(profile.get("hidden_fields", [])),
    }
    return payload


def canonical_json(payload: Any) -> str:
    """Deterministic serialization used for every compatibility export."""
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


# ---------------------------------------------------------------------------
# audit report (human readable)
# ---------------------------------------------------------------------------


def print_audit(profile: dict, card: SourceCard, report: dict) -> str:
    """Render the required sanity audit as one printable string."""
    lines: list[str] = []
    fields = profile.get("fields") or {}

    lines.append("DATASET")
    lines.append(str(profile.get("dataset")))
    lines.append("")
    lines.append("SOURCE FIELDS")
    for name in card.fields:
        lines.append(str(name))
    lines.append("")

    for name in card.fields:
        meta = fields.get(name) or {}
        lines.append(f"FIELD: {name}")
        lines.append(f"TYPE: {meta.get('type')}")
        lines.append(f"ROLE: {meta.get('role')}")
        lines.append(f"ENTITY SCOPE: {meta.get('entity_scope')}")
        lines.append(f"EVALUATION: {meta.get('evaluation')}")
        lines.append(f"CONTEXT VISIBILITY: {meta.get('context_visibility')}")
        lines.append(f"DESCRIPTION: {meta.get('description')}")
        lines.append("")

    lines.append(
        "SEMANTIC_FIELDS: "
        + ", ".join(sorted(profile.get("semantic_fields") or [])))
    lines.append(
        "HIDDEN_FIELDS: "
        + ", ".join(sorted(profile.get("hidden_fields") or [])))
    lines.append("RELATIONS:")
    for rel in profile.get("relations") or []:
        lines.append(
            f"  - {rel.get('source_field')} -> {rel.get('target_field')}: "
            f"{rel.get('description')}")
    lines.append("QA_CONSTRAINTS:")
    for constraint in profile.get("qa_constraints") or []:
        lines.append(f"  - {constraint}")
    lines.append("")

    mapping_check = report.get("checks", {}).get(
        "unsupported_label_mappings", {})
    lines.append(
        "GENDER_NUMERIC_MAPPING_INVENTED = "
        + ("NO" if mapping_check.get("ok", True) else "YES"))
    field_set = report.get("checks", {}).get("field_set", {})
    lines.append(f"FIELD_RENAMES = {len(field_set.get('renamed', {}) or {})}")
    lines.append(f"INVENTED_FIELDS = {len(field_set.get('invented', []) or [])}")
    lines.append("SOURCE_CARD_ONLY = YES")
    lines.append("TEST_ROWS_ACCESSED = 0")
    lines.append("AUDIO_DOWNLOADED = 0")

    flags = review_flags(profile, card)
    lines.append(
        "PROFILE_REVIEW_REQUIRED = " + ("YES" if flags else "NO"))
    for flag in flags:
        lines.append(
            f"  - {flag['field']}: {flag['rule']} ({flag['reason']})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call (real mode: exactly one, no retry)
# ---------------------------------------------------------------------------


class LocalProxyUnavailable(RuntimeError):
    """The local OpenAI-compatible endpoint is not accepting connections.

    Raised BEFORE any LLM request is sent, so a dead proxy costs no call.
    The message always starts with the required LOCAL_PROXY_UNAVAILABLE
    marker and the run stops.
    """


LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _tcp_probe(host: str, port: int, timeout: float = 5.0) -> None:
    import socket

    with socket.create_connection((host, port), timeout=timeout):
        pass


def require_local_proxy(base_url: str) -> None:
    """Preflight a loopback endpoint only; never sends data, no LLM call.

    Non-loopback endpoints are left to the HTTP call itself, so this
    preflight can never become an extra request to a remote service.
    """
    parsed = urlparse(base_url or "")
    host = (parsed.hostname or "").lower()
    if host not in LOOPBACK_HOSTS:
        return
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        _tcp_probe(host, port)
    except OSError as exc:
        raise LocalProxyUnavailable(
            f"LOCAL_PROXY_UNAVAILABLE: {host}:{port} is not accepting "
            f"connections ({type(exc).__name__}: {exc}). The profiling "
            "request was never sent; no retry was attempted."
        ) from exc


def _is_connection_failure(exc: BaseException) -> bool:
    """True for transport-level failures (proxy down / refused / timeout)."""
    names = {cls.__name__ for cls in type(exc).__mro__}
    return bool(names & {
        "APIConnectionError", "APITimeoutError", "APIError",
        "ConnectError", "ConnectTimeout", "ReadTimeout",
        "ConnectionError", "NewConnectionError", "gaierror",
        "TimeoutError", "SSLError",
    })


def run_profile_call(prompt: str):
    """One structured profiling request via the project's existing client."""
    from src.autonomous_qa.authoring.llm_client import (
        GenerateContentConfig,
        ThinkingConfig,
        create_llm_client,
        print_response_debug,
    )

    base_url = resolve_llm_base_url()
    model = resolve_llm_model()
    print("LLM_PROVIDER=openai_compatible")
    print(f"LLM_BASE_URL={base_url}")
    print(f"LLM_MODEL={model}")
    print("REMOTE_OPENAI_API=NO")
    require_local_proxy(base_url)

    client = create_llm_client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=GenerateContentConfig(
            temperature=PROFILE_TEMPERATURE,
            max_output_tokens=PROFILE_MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
            response_schema=DatasetProfile,
            thinking_config=ThinkingConfig(
                thinking_budget=PROFILE_THINKING_BUDGET),
        ),
    )
    print_response_debug(response, "dataset_profile")
    return response


_CODE_FENCE = re.compile(
    r"\A\s*```[ \t]*[A-Za-z0-9_+-]*[ \t]*\r?\n(.*?)\r?\n?```[ \t]*\s*\Z",
    re.DOTALL,
)


def parse_profile_payload(response) -> tuple[dict, str]:
    """Parse the single profiling response. Returns (payload, status).

    Parsing tolerance only: an enclosing markdown code fence is stripped
    after the normal parser rejects the text, and the fact is reported in
    the status (`parsed_after_fence_strip`). Model, temperature, schema,
    prompt, token budget and call count are untouched — this never causes
    a second request. Everything else stops the run.
    """
    from src.autonomous_qa.authoring.llm_client import (
        StageOutputValidationError,
        parse_structured_response,
    )

    try:
        parsed = parse_structured_response(response, "dataset_profile")
    except StageOutputValidationError:
        text = getattr(response, "text", "") or ""
        match = _CODE_FENCE.match(text)
        if match is None:
            raise
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise StageOutputValidationError(
                "Stage 'dataset_profile' returned non-JSON text inside a "
                "code fence. Raw response has been saved. No automatic "
                "retry was attempted."
            ) from exc
        if not isinstance(parsed, dict):
            raise StageOutputValidationError(
                "Stage 'dataset_profile' returned a fenced non-object JSON "
                "value. Raw response has been saved. No automatic retry "
                "was attempted.")
        print("NOTE: response arrived wrapped in a markdown code fence; "
              "parsed after fence removal (same request, no retry).")
        return parsed, "parsed_after_fence_strip"

    if not isinstance(parsed, dict):
        raise StageOutputValidationError(
            "Stage 'dataset_profile' returned a non-object JSON value. "
            "Raw response has been saved. No automatic retry was attempted.")
    return parsed, "parsed"


def load_mock_profile(dataset: str, fixture_path: Path | None = None) -> dict:
    if fixture_path is None:
        fixture_path = ROOT / "tests" / "fixtures" / \
            f"dataset_profile_{dataset}.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# run orchestration
# ---------------------------------------------------------------------------


def _write_text(path: Path, text: str) -> None:
    """Atomic write inside an already-private run directory."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        try:
            os.fsync(f.fileno())
        except (OSError, ValueError):
            pass
    os.replace(tmp, path)


def _new_run_dir(out_root: Path, dataset: str, run_id: str | None) -> tuple[Path, str]:
    """Create a brand-new run directory. Never reuses an existing one."""
    base = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = base
    counter = 1
    while True:
        run_dir = out_root / dataset / candidate / "profile"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir, candidate
        except FileExistsError:
            candidate = f"{base}-{counter}"
            counter += 1


def run_profile(
    dataset: str,
    source_card_path: Path | str,
    real_llm: bool = False,
    run_id: str | None = None,
    out_root: Path | str | None = None,
    prompt_path: Path | str | None = None,
    fixture_path: Path | str | None = None,
) -> dict:
    """Full stage: card -> profile -> validate -> compatibility exports."""
    source_card_path = Path(source_card_path)
    out_root = Path(out_root) if out_root else DEFAULT_OUT_ROOT

    card = parse_source_card(source_card_path)
    instructions = Path(prompt_path).read_text(encoding="utf-8") \
        if prompt_path else None
    prompt = build_prompt(card, instructions, dataset=dataset)

    run_dir, resolved_run_id = _new_run_dir(out_root, dataset, run_id)
    shutil.copyfile(source_card_path, run_dir / "source_card.md")

    started = datetime.now(timezone.utc).isoformat()
    model = resolve_llm_model()
    base_url = resolve_llm_base_url()
    call_count = 0
    parsing_status = "not_attempted"
    failure: str | None = None

    if real_llm:
        try:
            response = run_profile_call(prompt)
            call_count = 1
            finish = None
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                finish = getattr(candidates[0], "finish_reason", None)
            raw_text = getattr(response, "text", "") or ""
            _write_text(run_dir / "raw_response.txt", raw_text)
            if str(finish) == "MAX_TOKENS":
                failure = (
                    "PROFILE_CALL_FAILED: truncated response "
                    "(finish_reason=MAX_TOKENS). Raw response saved at "
                    f"{run_dir / 'raw_response.txt'}. "
                    "No retry was attempted; model, temperature, schema, "
                    "prompt and token budget are unchanged."
                )
                parsing_status = "truncated"
            else:
                parsed, status = parse_profile_payload(response)
                profile = DatasetProfile.model_validate(parsed).model_dump()
                parsing_status = status
        # The proxy being down is reported verbatim and stops the run
        # before any request is sent.
        except LocalProxyUnavailable as exc:
            failure = str(exc)
            parsing_status = "proxy_unavailable"
            profile = None
        # Deliberately broad: transport, HTTP, truncation and schema
        # failures all STOP the run with the exact error. Never retry,
        # never fall back to another model/prompt/schema.
        except Exception as exc:  # noqa: BLE001
            if _is_connection_failure(exc):
                failure = (
                    "LOCAL_PROXY_UNAVAILABLE: "
                    f"{type(exc).__name__}: {exc}. The profiling request "
                    f"did not reach {base_url}. No retry was attempted; "
                    "model, temperature, schema, prompt and token budget "
                    "are unchanged."
                )
                parsing_status = "proxy_unavailable"
            else:
                failure = (
                    f"PROFILE_CALL_FAILED: {type(exc).__name__}: {exc}. "
                    "No retry was attempted; model, temperature, schema, "
                    "prompt and token budget are unchanged."
                )
                parsing_status = "failed"
            # A provider exception means no response returned, so no raw
            # response file is fabricated here.
            profile = None
    else:
        profile_payload = load_mock_profile(
            dataset, Path(fixture_path) if fixture_path else None)
        raw_text = canonical_json(profile_payload)
        _write_text(run_dir / "raw_response.txt", raw_text)
        call_count = 0
        try:
            profile = DatasetProfile.model_validate(
                profile_payload).model_dump()
            parsing_status = "parsed"
        except Exception as exc:  # noqa: BLE001 - stop, never repair silently
            failure = f"MOCK_PROFILE_INVALID: {type(exc).__name__}: {exc}"
            parsing_status = "failed"
            profile = None

    run_meta: dict[str, Any] = {
        "dataset": dataset,
        "run_id": resolved_run_id,
        "mode": "real" if real_llm else "mock",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "source_card": str(source_card_path),
        "source_card_sha256": card.sha256,
        "source_card_field_count": len(card.fields),
        "source_card_fields": dict(card.fields),
        "source_card_splits": card.splits,
        "source_card_total_rows": card.total_rows,
        "prompt_sha256": sha256_text(prompt),
        "provider": "openai_compatible" if real_llm else None,
        "base_url": base_url if real_llm else None,
        "model": model,
        "temperature": PROFILE_TEMPERATURE if real_llm else None,
        "max_output_tokens": PROFILE_MAX_OUTPUT_TOKENS if real_llm else None,
        "thinking_budget": PROFILE_THINKING_BUDGET if real_llm else None,
        "llm_calls": call_count,
        "parsing_status": parsing_status,
        "failure": failure,
        "contract": "dataset-profile-v1",
    }

    if failure is not None:
        write_json(run_dir / "run_meta.json", run_meta)
        print(failure)
        print(f"RUN_DIR = {run_dir}")
        return {"ok": False, "exit_code": EXIT_CALL_FAILED,
                "run_dir": run_dir, "failure": failure,
                "run_meta": run_meta}

    assert profile is not None
    _write_text(run_dir / "dataset_profile.json", canonical_json(profile))

    report = validate_profile(profile, card)
    write_json(run_dir / "validation_report.json", report)

    if not report["ok"]:
        run_meta["validation_ok"] = False
        run_meta["validation_errors"] = report["errors"]
        write_json(run_dir / "run_meta.json", run_meta)
        print("VALIDATION FAILED — no compatibility files were exported.")
        for message in report["errors"]:
            print(f"  - {message}")
        print(f"RUN_DIR = {run_dir}")
        return {"ok": False, "exit_code": EXIT_VALIDATION_FAILED,
                "run_dir": run_dir, "report": report, "profile": profile,
                "run_meta": run_meta}

    schema = export_schema(profile, card)
    metadata_context = export_metadata_context(profile, card)
    _write_text(run_dir / "schema.json", canonical_json(schema))
    _write_text(run_dir / "metadata_context.json",
                canonical_json(metadata_context))

    run_meta["validation_ok"] = True
    run_meta["profile_sha256"] = sha256_file(run_dir / "dataset_profile.json")
    run_meta["schema_sha256"] = sha256_file(run_dir / "schema.json")
    run_meta["metadata_context_sha256"] = sha256_file(
        run_dir / "metadata_context.json")
    run_meta["review_flags"] = review_flags(profile, card)
    write_json(run_dir / "run_meta.json", run_meta)

    audit = print_audit(profile, card, report)
    _write_text(run_dir / "audit.txt", audit + "\n")
    print(audit)
    print()
    print(f"RUN_DIR = {run_dir}")
    return {"ok": True, "exit_code": EXIT_OK, "run_dir": run_dir,
            "report": report, "profile": profile, "schema": schema,
            "metadata_context": metadata_context, "audit": audit,
            "run_meta": run_meta}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.autonomous_qa.core.dataset_profile",
        description="Profile a dataset source card into one canonical "
                    "dataset_profile.json, then export schema.json and "
                    "metadata_context.json deterministically.",
    )
    parser.add_argument("--dataset", required=True,
                        help="Dataset key, e.g. vimd")
    parser.add_argument("--source-card", required=True,
                        help="Path to the dataset card markdown file")
    parser.add_argument("--real-llm", action="store_true",
                        help="Make one real profiling call (default: mock)")
    parser.add_argument("--run-id", default=None,
                        help="Run directory id (default: UTC timestamp)")
    parser.add_argument("--out-root", default=None,
                        help="Output root (default: outputs/runs)")
    parser.add_argument("--prompt", default=None,
                        help="Override the profiler prompt file")
    parser.add_argument("--fixture", default=None,
                        help="Override the mock profile fixture")
    args = parser.parse_args(argv)

    result = run_profile(
        dataset=args.dataset,
        source_card_path=args.source_card,
        real_llm=args.real_llm,
        run_id=args.run_id,
        out_root=args.out_root,
        prompt_path=args.prompt,
        fixture_path=args.fixture,
    )
    return int(result["exit_code"])


if __name__ == "__main__":
    sys.exit(main())

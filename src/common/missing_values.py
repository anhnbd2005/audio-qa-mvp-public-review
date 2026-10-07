"""Generic missing value extraction, normalization, and data-backed derived-relation validation utilities."""

from __future__ import annotations

from typing import Any, Container

ROLE_HIDDEN_IDENTIFIER = "hidden_identifier"
ROLE_SEMANTIC = "semantic"
ROLE_PROVENANCE = "provenance"
ROLE_AUDIO = "audio"
ROLE_CONTEXT_ONLY = "context_only"


def extract_source_missing_values(schema: dict[str, Any] | None) -> dict[str, list[str]]:
    """Extract source-specific missing value markers from schema dictionary."""
    if not isinstance(schema, dict):
        return {}
    if "source_missing_values" in schema and isinstance(schema["source_missing_values"], dict):
        return {
            str(k): list(v) if isinstance(v, list) else [str(v)]
            for k, v in schema["source_missing_values"].items()
        }
    missing_by_field: dict[str, list[str]] = {}
    fields = schema.get("fields", {})
    if isinstance(fields, dict):
        for f, meta in fields.items():
            if isinstance(meta, dict) and "source_missing_values" in meta:
                val = meta["source_missing_values"]
                if isinstance(val, list):
                    missing_by_field[f] = [str(x) for x in val]
                elif isinstance(val, (str, int, float)):
                    missing_by_field[f] = [str(val)]
    return missing_by_field


def normalize_row_missing_values(
    row: dict[str, Any],
    missing_values_by_field: dict[str, list[str]],
) -> dict[str, Any]:
    """Generic missing-value normalization.

    Normalizes any value matching declared source_missing_values for that field
    to None. Deterministic, schema-driven, no global hard-coding.
    """
    if not isinstance(row, dict) or not missing_values_by_field:
        return row
    normalized = dict(row)
    for field, val in row.items():
        if val is None:
            continue
        markers = missing_values_by_field.get(field)
        if not markers:
            continue
        val_str = str(val).strip()
        lower_markers = {str(m).strip().lower() for m in markers}
        if val_str.lower() in lower_markers:
            normalized[field] = None
    return normalized


def get_field_role(
    field_name: str,
    field_roles: dict[str, str] | None = None,
    hidden_fields: Container[str] | None = None,
) -> str:
    """Determine generic field role."""
    if field_roles and field_name in field_roles:
        return field_roles[field_name]
    if hidden_fields and field_name in hidden_fields:
        return ROLE_HIDDEN_IDENTIFIER
    if field_name.lower().endswith("id") or field_name.lower().startswith("id_"):
        return ROLE_HIDDEN_IDENTIFIER
    return ROLE_SEMANTIC


def validate_derived_relation(
    source_key: str,
    target_key: str,
    dataset_rows: list[dict[str, Any]],
    schema_fields: Container[str] | None = None,
    hidden_fields: Container[str] | None = None,
    relation_cache: dict[tuple[str, str], dict[str, Any]] | None = None,
    field_roles: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Generic data-backed derived-relation validator.

    Knows NO dataset name and NO allowed pair list: proves proposed source->target
    relation from normalized dataset rows. Returns dict with valid/reason/mapping
    plus audit metadata.
    """
    schema = set(schema_fields) if schema_fields is not None else None
    hidden = set(hidden_fields) if hidden_fields is not None else set()
    cache = relation_cache if isinstance(relation_cache, dict) else {}
    cache_key = (source_key, target_key)
    if cache_key in cache:
        return cache[cache_key]

    def _fail(reason: str) -> dict[str, Any]:
        result = {
            "valid": False,
            "reason": reason,
            "source_key": source_key,
            "target_key": target_key,
            "mapping": None,
            "distinct_source_count": 0,
            "distinct_target_count": 0,
            "mapping_conflicts": 0,
            "reusable_source_values": 0,
        }
        cache[cache_key] = result
        return result

    if source_key == target_key:
        return _fail("invalid_derived_same_field")
    if schema is not None and source_key not in schema:
        return _fail(f"unknown_answer_field:{source_key}")
    if schema is not None and target_key not in schema:
        return _fail(f"unknown_answer_field:{target_key}")

    src_role = get_field_role(source_key, field_roles, hidden_fields=hidden)
    tgt_role = get_field_role(target_key, field_roles, hidden_fields=hidden)
    if src_role == ROLE_HIDDEN_IDENTIFIER:
        return _fail("invalid_derived_hidden_source")
    if tgt_role == ROLE_HIDDEN_IDENTIFIER:
        return _fail("invalid_derived_hidden_target")
    if src_role != ROLE_SEMANTIC:
        return _fail(f"ineligible_field_role:{src_role}:{source_key}")
    if tgt_role != ROLE_SEMANTIC:
        return _fail(f"ineligible_field_role:{tgt_role}:{target_key}")
    if not isinstance(dataset_rows, list) or not dataset_rows:
        return _fail("derived_relation_data_unavailable")
    pairs = [
        (r.get(source_key), r.get(target_key))
        for r in dataset_rows
        if isinstance(r, dict)
    ]
    pairs = [(s, t) for s, t in pairs if s is not None and t is not None]
    if not pairs:
        return _fail("derived_relation_data_unavailable")
    sources = [s for s, _ in pairs]
    targets = [t for _, t in pairs]
    try:
        distinct_sources = set(sources)
        distinct_targets = set(targets)
        if len(distinct_sources) < 2:
            result = _fail("invalid_derived_insufficient_source_values")
            result["distinct_source_count"] = len(distinct_sources)
            result["distinct_target_count"] = len(distinct_targets)
            return result
        if len(distinct_targets) < 2:
            result = _fail("invalid_derived_insufficient_target_values")
            result["distinct_source_count"] = len(distinct_sources)
            result["distinct_target_count"] = len(distinct_targets)
            return result
        observed: dict[Any, set[Any]] = {}
        for s, t in pairs:
            observed.setdefault(s, set()).add(t)
    except TypeError:
        return _fail("derived_relation_data_unavailable")
    conflicts = sum(1 for v in observed.values() if len(v) != 1)
    if conflicts:
        result = _fail("invalid_derived_nonfunctional_relation")
        result["mapping_conflicts"] = conflicts
        result["distinct_source_count"] = len(distinct_sources)
        result["distinct_target_count"] = len(distinct_targets)
        return result
    counts: dict[Any, int] = {}
    for s in sources:
        counts[s] = counts.get(s, 0) + 1
    reusable = sum(1 for c in counts.values() if c >= 2)
    if reusable < 2:
        result = _fail("invalid_derived_nonreusable_source")
        result["reusable_source_values"] = reusable
        result["distinct_source_count"] = len(distinct_sources)
        result["distinct_target_count"] = len(distinct_targets)
        return result
    if not len(distinct_targets) < len(distinct_sources):
        result = _fail("invalid_derived_not_coarsening")
        result["distinct_source_count"] = len(distinct_sources)
        result["distinct_target_count"] = len(distinct_targets)
        result["reusable_source_values"] = reusable
        return result

    mapping = {str(k): str(list(v)[0]) for k, v in observed.items()}
    result = {
        "valid": True,
        "reason": None,
        "source_key": source_key,
        "target_key": target_key,
        "mapping": mapping,
        "distinct_source_count": len(distinct_sources),
        "distinct_target_count": len(distinct_targets),
        "mapping_conflicts": 0,
        "reusable_source_values": reusable,
    }
    cache[cache_key] = result
    return result

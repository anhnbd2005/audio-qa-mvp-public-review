"""Source row identity and dataset indexing for production QA generation.

Extracts source row identity construction and dataset indexing from production_qa.py.
"""

from __future__ import annotations

from typing import Any

from src.autonomous_qa.certification.qa_sampling import TrainIndex, meta_of
from src.autonomous_qa.compiler.semantic_field_specs import SemanticFieldSpec
from src.autonomous_qa.language.template_contracts import TypeContract


def build_row_identities(
    rows: list[dict[str, Any]], source: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    """Internal stable row identity; never emitted model-facing."""
    key_field = source.get("row_key_field")
    row_ids: list[str] = []
    scheme = "row_index"
    if key_field:
        keys = [str(meta_of(row).get(key_field) or "") for row in rows]
        if all(keys) and len(set(keys)) == len(keys):
            scheme = f"unique_field:{key_field}"
            row_ids = keys
    if not row_ids:
        row_ids = [f"row_{position:06d}" for position in range(len(rows))]
    return row_ids, {
        "scheme": scheme,
        "row_key_field": key_field if scheme.startswith("unique_field") else None,
        "unique": len(set(row_ids)) == len(row_ids),
        "exposed_model_facing": False,
    }


def index_dataset_rows(
    rows: list[dict[str, Any]],
    row_ids: list[str],
    contracts: list[TypeContract],
    specs: dict[str, SemanticFieldSpec],
    hidden_identifier_field: str | None = None,
) -> TrainIndex:
    """Build O(N) TrainIndex over source rows for all supported type contracts."""
    index_fields = {
        contract.semantic_field: specs[contract.semantic_field]
        for contract in contracts
    }
    return TrainIndex(
        rows,
        row_ids,
        index_fields,
        hidden_identifier_field=hidden_identifier_field,
    )

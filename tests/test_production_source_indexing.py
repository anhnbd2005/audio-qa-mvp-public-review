"""Characterization tests for production source row identity and dataset indexing.

Validates behavior preservation for build_row_identities and index_dataset_rows.
"""

import json
from pathlib import Path
import pytest

from src.common.config import ROOT
from src.autonomous_qa.language.template_engine import vimd_field_specs
from src.autonomous_qa.language.template_contracts import build_type_contracts
from src.autonomous_qa.production.production_qa import (
    build_row_identities,
    index_dataset_rows,
)
from src.autonomous_qa.certification.qa_sampling import TrainIndex


@pytest.fixture(scope="module")
def real_contracts():
    path = ROOT / "resources" / "semantics" / "vimd_type_registry.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [c for c in build_type_contracts(raw) if c.source_status == "SUPPORTED"]


def test_build_row_identities_default_positional():
    rows = [
        {"province_name": "P1"},
        {"province_name": "P2"},
    ]
    source = {"row_key_field": None}
    row_ids, audit = build_row_identities(rows, source)
    assert row_ids == ["row_000000", "row_000001"]
    assert audit["scheme"] == "row_index"
    assert audit["unique"] is True
    assert audit["exposed_model_facing"] is False


def test_build_row_identities_unique_field():
    rows = [
        {"filename": "clip_a.wav"},
        {"filename": "clip_b.wav"},
    ]
    source = {"row_key_field": "filename"}
    row_ids, audit = build_row_identities(rows, source)
    assert row_ids == ["clip_a.wav", "clip_b.wav"]
    assert audit["scheme"] == "unique_field:filename"
    assert audit["row_key_field"] == "filename"
    assert audit["unique"] is True


def test_build_row_identities_fallback_on_duplicate_key():
    rows = [
        {"filename": "clip_a.wav"},
        {"filename": "clip_a.wav"},
    ]
    source = {"row_key_field": "filename"}
    row_ids, audit = build_row_identities(rows, source)
    assert row_ids == ["row_000000", "row_000001"]
    assert audit["scheme"] == "row_index"
    assert audit["row_key_field"] is None


def test_index_dataset_rows_construction(real_contracts):
    rows = [
        {"province_name": "P1", "speakerID": "s1"},
        {"province_name": "P2", "speakerID": "s2"},
    ]
    row_ids = ["row_000000", "row_000001"]
    specs = vimd_field_specs()

    index = index_dataset_rows(
        rows=rows,
        row_ids=row_ids,
        contracts=real_contracts,
        specs=specs,
        hidden_identifier_field="speakerID",
    )

    assert isinstance(index, TrainIndex)
    assert index.row_ids == row_ids
    assert len(index.rows) == 2
    assert "province_name" in index.rows_by_value
    assert index.hidden_value_by_row["row_000000"] == "s1"
    assert index.hidden_value_by_row["row_000001"] == "s2"


def test_index_dataset_rows_alignment_mismatch(real_contracts):
    rows = [{"province_name": "P1"}]
    row_ids = ["row_000000", "row_000001"]
    specs = vimd_field_specs()

    with pytest.raises(ValueError, match="ROW_ID_ALIGNMENT_MISMATCH"):
        index_dataset_rows(rows, row_ids, real_contracts, specs)


def test_index_dataset_rows_duplicate_id_raises(real_contracts):
    rows = [{"province_name": "P1"}, {"province_name": "P2"}]
    row_ids = ["row_000000", "row_000000"]
    specs = vimd_field_specs()

    with pytest.raises(ValueError, match="DUPLICATE_SOURCE_ROW_ID"):
        index_dataset_rows(rows, row_ids, real_contracts, specs)

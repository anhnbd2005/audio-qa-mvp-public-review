"""Characterization tests for production budget resolution logic."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.autonomous_qa.production import production_qa as pq
from src.autonomous_qa.language.language_quality import ProductionGenerationConfig
from src.autonomous_qa.language.template_contracts import build_type_contracts

ROOT = Path(__file__).resolve().parents[1]
TYPE_REGISTRY_PATH = ROOT / "resources" / "semantics" / "vimd_type_registry.json"


@pytest.fixture
def sample_contracts():
    raw = json.loads(TYPE_REGISTRY_PATH.read_text(encoding="utf-8"))
    contracts = build_type_contracts(raw)
    return [c for c in contracts if c.source_status == "SUPPORTED"][:3]


@pytest.fixture
def mock_capacities(sample_contracts):
    caps = {}
    for c in sample_contracts:
        caps[c.type_id] = {
            "operator": c.operator,
            "theoretical_semantic_capacity": {"positive": 100, "negative": 100},
            "feasible_capacity": 100,
            "feasible_positive_capacity": 50,
            "feasible_negative_capacity": 50,
        }
    return caps


def test_resolve_budget_no_budget_returns_awaiting():
    cfg = ProductionGenerationConfig()
    contracts = []
    caps = {}
    res = pq.resolve_budget(cfg, contracts, caps)
    assert res["status"] == "READY_AWAITING_BUDGET"
    assert res["explicit_budget_present"] is False
    assert res["resolved_total"] is None
    assert res["per_type"] == {}


def test_resolve_budget_debug_cap(sample_contracts, mock_capacities):
    cfg = ProductionGenerationConfig()
    res = pq.resolve_budget(cfg, sample_contracts, mock_capacities, debug_cap=10)
    assert res["status"] == "DEBUG_SAMPLE_BUDGET"
    assert res["explicit_budget_present"] is True
    assert res["resolved_total"] == 10
    assert res["debug_sample"] is True
    assert sum(res["per_type"].values()) == 10


def test_resolve_budget_per_type_budget(sample_contracts, mock_capacities):
    type_ids = [c.type_id for c in sample_contracts]
    per_type = {type_ids[0]: 20, type_ids[1]: 30, type_ids[2]: 40}
    cfg = ProductionGenerationConfig(per_type_budget=per_type)
    res = pq.resolve_budget(cfg, sample_contracts, mock_capacities)
    assert res["status"] == "PER_TYPE_BUDGET"
    assert res["explicit_budget_present"] is True
    assert res["resolved_total"] == 90
    assert res["per_type"] == per_type


def test_resolve_budget_per_type_unknown_type_raises():
    cfg = ProductionGenerationConfig(per_type_budget={"UNKNOWN_TYPE": 10})
    contracts = []
    caps = {}
    with pytest.raises(pq.ProductionQAError) as exc_info:
        pq.resolve_budget(cfg, contracts, caps)
    assert "BUDGET_TYPE_UNKNOWN" in str(exc_info.value)


def test_resolve_budget_total_target_qa_equal_split(sample_contracts, mock_capacities):
    cfg = ProductionGenerationConfig(total_target_qa=60)
    res = pq.resolve_budget(cfg, sample_contracts, mock_capacities)
    assert res["status"] == "TOTAL_BUDGET_EQUAL_BY_TYPE"
    assert res["explicit_budget_present"] is True
    assert res["resolved_total"] == 60
    assert sum(res["per_type"].values()) == 60


def test_resolve_budget_capacity_capping(sample_contracts):
    type_ids = [c.type_id for c in sample_contracts]
    caps = {
        type_ids[0]: {
            "operator": sample_contracts[0].operator,
            "theoretical_semantic_capacity": {"positive": 10, "negative": 10},
            "feasible_capacity": 10,
        },
        type_ids[1]: {
            "operator": sample_contracts[1].operator,
            "theoretical_semantic_capacity": {"positive": 100, "negative": 100},
            "feasible_capacity": 100,
        },
        type_ids[2]: {
            "operator": sample_contracts[2].operator,
            "theoretical_semantic_capacity": {"positive": 100, "negative": 100},
            "feasible_capacity": 100,
        },
    }
    per_type = {type_ids[0]: 50, type_ids[1]: 50, type_ids[2]: 50}
    cfg = ProductionGenerationConfig(per_type_budget=per_type, strict_budget=False)
    res = pq.resolve_budget(cfg, sample_contracts, caps)
    assert res["status"] == "PER_TYPE_BUDGET"
    assert res["resolved_total"] == 110  # 10 + 50 + 50
    assert res["shortfall_total"] == 40
    assert len(res["capacity_capping"]) == 1
    assert res["capacity_capping"][0]["type_id"] == type_ids[0]


def test_resolve_budget_strict_budget_infeasible(sample_contracts):
    type_ids = [c.type_id for c in sample_contracts]
    caps = {
        type_ids[0]: {
            "operator": sample_contracts[0].operator,
            "theoretical_semantic_capacity": {"positive": 10, "negative": 10},
            "feasible_capacity": 10,
        },
        type_ids[1]: {
            "operator": sample_contracts[1].operator,
            "theoretical_semantic_capacity": {"positive": 100, "negative": 100},
            "feasible_capacity": 100,
        },
        type_ids[2]: {
            "operator": sample_contracts[2].operator,
            "theoretical_semantic_capacity": {"positive": 100, "negative": 100},
            "feasible_capacity": 100,
        },
    }
    per_type = {type_ids[0]: 50, type_ids[1]: 50, type_ids[2]: 50}
    cfg = ProductionGenerationConfig(per_type_budget=per_type, strict_budget=True)
    res = pq.resolve_budget(cfg, sample_contracts, caps)
    assert res["status"] == "BUDGET_INFEASIBLE"
    assert res["resolved_total"] == 110
    assert res["shortfall_total"] == 40

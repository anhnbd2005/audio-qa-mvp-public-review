"""R3 — architecture ownership characterization tests.

Pins the single authoritative execution path after R1/R2 and asserts that the
obsolete alternative planning/budget modules were removed.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from src.autonomous_qa.production import budget, production_qa, source_preparation


def test_active_planner_is_generation_plan():
    assert callable(production_qa.build_generation_plan)
    from src.autonomous_qa.certification.qa_sampling import SemanticSampler

    # The active sampler is the generic anchor-neighborhood SemanticSampler.
    assert callable(SemanticSampler)


def test_single_budget_authority():
    assert callable(budget.resolve_budget)
    assert callable(budget.derive_full_split_budget)
    assert callable(source_preparation.finalize_allocation)
    assert callable(source_preparation.build_allocation_contract)
    assert callable(source_preparation.verify_allocation_contract)


def test_production_plan_retained_for_certification():
    # ProductionPlan (schema v1) is a distinct CERTIFICATION path, not the active
    # production planner; it is intentionally retained.
    from src.autonomous_qa.certification import core_invariants
    from src.autonomous_qa.production.production_plan import ProductionPlan

    assert core_invariants.ProductionPlan is ProductionPlan


def test_obsolete_planning_modules_removed():
    for name in ("selection_policies", "production_contract"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"src.autonomous_qa.production.{name}")


def test_retained_frozen_dependencies_resolver_present():
    # `frozen_dependencies.py` resolves the committed `resources/frozen/` evidence
    # store; retained (not removed) to avoid orphaning certified artifacts.
    from src.autonomous_qa.production import frozen_dependencies

    assert callable(frozen_dependencies.resolve_frozen)
    assert (Path(__file__).resolve().parents[1] / "resources" / "frozen").is_dir()

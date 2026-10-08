"""R1 — Source Preparation + Early Budget Authorization tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.autonomous_qa.authoring import pipeline as ra
from src.autonomous_qa.language.language_quality import (
    ProductionGenerationConfig,
    SamplingPolicyConfig,
)
from src.autonomous_qa.language.template_contracts import TypeContract
from src.autonomous_qa.production import source_preparation as sp
from src.autonomous_qa.production.budget import derive_full_split_budget
from tests.test_authoring_pipeline import _fake_llm, _write_tiny_plan

ROWS = [
    {"segment_id": "s1", "topic": "A", "cs_terms_count": 1},
    {"segment_id": "s2", "topic": "B", "cs_terms_count": 2},
    {"segment_id": "s3", "topic": "A", "cs_terms_count": None},
]


def _write_source(tmp_path: Path, rows=ROWS) -> Path:
    path = tmp_path / "source.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _contract(type_id: str, operator: str) -> TypeContract:
    return TypeContract(
        type_id=type_id,
        operator=operator,
        semantic_field="topic",
        semantic_description="x",
        semantic_phrase_vi="x",
        entity_scope="utterance",
        audio_input_count=2 if operator == "EQUALITY" else 1,
        condition_fields=[],
        gold_source_fields=["topic"],
        answer_kind="boolean" if operator == "EQUALITY" else "field_value",
        answer_mode="BOOLEAN" if operator == "EQUALITY" else "FIELD_VALUE",
        template_status="ELIGIBLE",
        template_policy={},
        instantiation_policy={},
        source_status="SUPPORTED",
    )


def _capacities(contracts, anchors: int, capacity: int = 100000) -> dict:
    return {
        c.type_id: {
            "operator": c.operator,
            "theoretical_semantic_capacity": {
                "positive": capacity,
                "negative": capacity,
            },
            "evidence_summary": {"valid_rows": anchors},
        }
        for c in contracts
    }


def _full_split_fixture():
    contracts = [
        _contract("d1", "DIRECT"),
        _contract("d2", "DIRECT"),
        _contract("eq", "EQUALITY"),
    ]
    return contracts, _capacities(contracts, anchors=100)


def _full_split_cfg() -> ProductionGenerationConfig:
    return ProductionGenerationConfig(
        coverage_mode="full_split",
        target_qa_per_audio=10,
        sampling=SamplingPolicyConfig(
            equality_strategy="anchor_neighborhood",
            equality_positive_per_anchor=4,
            equality_negative_per_anchor=4,
        ),
    )


def _early(target, policy="report"):
    return sp.EarlyBudget(
        dataset="d",
        split="train",
        coverage_mode="full_split",
        target_total_qa=target,
        target_qa_per_audio=10,
        allowed_operator_families=(),
        shortfall_policy=policy,
        source_fingerprint="src",
        status="AUTHORIZED",
        fingerprint="eb",
    )


# ---------------------------------------------------------------------------
# SOURCE PREPARATION
# ---------------------------------------------------------------------------


def test_prepare_source_inventory_basic(tmp_path):
    inv = sp.prepare_source_inventory(
        dataset="d",
        split="train",
        path=_write_source(tmp_path),
        identity_field="segment_id",
        eligibility_fields=["topic", "cs_terms_count"],
        group_fields=["topic"],
    )
    assert inv.row_count == 3
    assert inv.eligible_rows == {"topic": 3, "cs_terms_count": 2}
    assert inv.label_groups == {"topic": 2}
    assert len(inv.sha256) == 64
    assert inv.fingerprint


def test_prepare_reuses_preloaded_rows_without_reload(tmp_path):
    path = _write_source(tmp_path)
    # Passing rows must not require the path to be re-read for rows.
    inv = sp.prepare_source_inventory(
        dataset="d", split="train", path=path, rows=list(ROWS),
        eligibility_fields=["topic"],
    )
    assert inv.row_count == 3


def test_source_sha_mismatch(tmp_path):
    with pytest.raises(sp.BudgetAuthorityError) as exc:
        sp.prepare_source_inventory(
            dataset="d", split="train", path=_write_source(tmp_path),
            expected_sha256="deadbeef",
        )
    assert exc.value.code == "SOURCE_SHA_MISMATCH"


def test_source_row_count_mismatch(tmp_path):
    with pytest.raises(sp.BudgetAuthorityError) as exc:
        sp.prepare_source_inventory(
            dataset="d", split="train", path=_write_source(tmp_path), expected_rows=99
        )
    assert exc.value.code == "SOURCE_ROW_COUNT_MISMATCH"


def test_repeated_preparation_deterministic(tmp_path):
    path = _write_source(tmp_path)
    a = sp.prepare_source_inventory(
        dataset="d", split="train", path=path, eligibility_fields=["topic"]
    )
    b = sp.prepare_source_inventory(
        dataset="d", split="train", path=path, eligibility_fields=["topic"]
    )
    assert a.fingerprint == b.fingerprint


# ---------------------------------------------------------------------------
# EARLY BUDGET
# ---------------------------------------------------------------------------


def test_resolve_early_budget_full_split(tmp_path):
    inv = sp.prepare_source_inventory(
        dataset="d", split="train", path=_write_source(tmp_path),
        eligibility_fields=["topic"],
    )
    cfg = _full_split_cfg()
    eb = sp.resolve_early_budget(inventory=inv, config=cfg)
    assert eb.coverage_mode == "full_split"
    assert eb.target_total_qa == 10 * 3
    assert eb.status == "AUTHORIZED"


def test_resolve_early_budget_exploratory(tmp_path):
    inv = sp.prepare_source_inventory(
        dataset="d", split="train", path=_write_source(tmp_path)
    )
    eb = sp.resolve_early_budget(inventory=inv, config=None, exploratory=True)
    assert eb.status == "EXPLORATORY"
    assert eb.target_total_qa is None
    assert eb.coverage_mode == "exploratory"


def test_resolve_early_budget_missing_policy_is_awaiting(tmp_path):
    inv = sp.prepare_source_inventory(
        dataset="d", split="train", path=_write_source(tmp_path),
        eligibility_fields=["topic"],
    )
    cfg = ProductionGenerationConfig(coverage_mode="explicit")  # no total, no per-type
    eb = sp.resolve_early_budget(inventory=inv, config=cfg)
    assert eb.status == "READY_AWAITING_BUDGET"
    assert eb.target_total_qa is None


# ---------------------------------------------------------------------------
# APPROVED ALLOCATION <= EARLY BUDGET
# ---------------------------------------------------------------------------


def test_finalize_allocation_matches_derived_full_split():
    contracts, caps = _full_split_fixture()
    cfg = _full_split_cfg()
    out = sp.finalize_allocation(
        early=_early(1000), contracts=contracts, capacities=caps, config=cfg
    )
    assert out["allocation"] == {"d1": 100, "d2": 100, "eq": 800}
    assert out["allocation_total"] == 1000
    assert out["shortfall"] == 0
    assert out["budget"]["per_type"] == out["allocation"]


def test_allocation_cannot_exceed_early_budget():
    contracts, caps = _full_split_fixture()
    cfg = _full_split_cfg()
    with pytest.raises(sp.BudgetAuthorityError) as exc:
        sp.finalize_allocation(
            early=_early(5), contracts=contracts, capacities=caps, config=cfg
        )
    assert exc.value.code == "ALLOCATION_EXCEEDS_EARLY_BUDGET"


def test_shortfall_policy_fail_raises():
    contracts, caps = _full_split_fixture()
    cfg = _full_split_cfg()
    with pytest.raises(sp.BudgetAuthorityError) as exc:
        sp.finalize_allocation(
            early=_early(2000, policy="fail"),
            contracts=contracts, capacities=caps, config=cfg,
        )
    assert exc.value.code == "EARLY_BUDGET_SHORTFALL"


def test_shortfall_reported_when_policy_report():
    contracts, caps = _full_split_fixture()
    cfg = _full_split_cfg()
    out = sp.finalize_allocation(
        early=_early(2000, policy="report"),
        contracts=contracts, capacities=caps, config=cfg,
    )
    assert out["shortfall"] == 1000
    assert out["allocation_total"] == 1000


def test_full_split_budget_reused_from_budget_module():
    # The canonical algorithm lives once, in budget.py, and is reused (not
    # duplicated) by the preparation/freeze stage.
    assert sp.derive_full_split_budget is derive_full_split_budget


# ---------------------------------------------------------------------------
# AUTHORING INTEGRATION
# ---------------------------------------------------------------------------


def test_budget_available_before_first_llm_call(tmp_path):
    tiny = tmp_path / "plan.jsonl"
    _write_tiny_plan(tiny)
    run_dir = tmp_path / "vietmdd" / "t_early"
    seen: list[bool] = []

    def spy(stage, prompt, temperature, schema):
        seen.append((run_dir / "early_budget.json").exists())
        return _fake_llm(stage, prompt, temperature, schema)

    manifest = ra.run_authoring(
        "vietmdd", run_id="t_early", real_llm=True, llm_call=spy,
        out_root=tmp_path, plan_path=tiny,
    )
    assert seen and all(seen), "early budget must exist before every LLM call"
    assert manifest["early_budget"]["status"] == "EXPLORATORY"
    assert manifest["source_inventory"]["row_count"] > 0
    assert (run_dir / "preparation.json").exists()


def test_production_config_early_budget_before_llm(tmp_path):
    tiny = tmp_path / "plan.jsonl"
    _write_tiny_plan(tiny)
    cfg = _full_split_cfg()
    run_dir = tmp_path / "vietmdd" / "t_prod"
    seen: list[bool] = []

    def spy(stage, prompt, temperature, schema):
        seen.append((run_dir / "early_budget.json").exists())
        return _fake_llm(stage, prompt, temperature, schema)

    manifest = ra.run_authoring(
        "vietmdd", run_id="t_prod", real_llm=True, llm_call=spy,
        out_root=tmp_path, plan_path=tiny, production_config=cfg,
    )
    assert seen and all(seen)
    assert manifest["early_budget"]["coverage_mode"] == "full_split"
    assert manifest["early_budget"]["target_total_qa"] is not None

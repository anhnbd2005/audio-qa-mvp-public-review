"""R2 — operator policy, approved allocation authority, planner/release enforcement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.autonomous_qa.language.language_quality import (
    ProductionGenerationConfig,
    SamplingPolicyConfig,
)
from src.autonomous_qa.language.template_contracts import TypeContract
from src.autonomous_qa.production import source_preparation as sp
from src.autonomous_qa.production.production_qa import (
    ProductionQAError,
    build_generation_plan,
)


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
            "theoretical_semantic_capacity": {"positive": capacity, "negative": capacity},
            "evidence_summary": {"valid_rows": anchors},
        }
        for c in contracts
    }


def _cfg(**over) -> ProductionGenerationConfig:
    base = {
        "seed": 42,
        "coverage_mode": "full_split",
        "target_qa_per_audio": 10,
        "sampling": SamplingPolicyConfig(
            equality_strategy="anchor_neighborhood",
            equality_positive_per_anchor=4,
            equality_negative_per_anchor=4,
        ),
    }
    base.update(over)
    return ProductionGenerationConfig(**base)


def _inventory(tmp_path: Path, rows):
    path = tmp_path / "s.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return sp.prepare_source_inventory(
        dataset="d", split="train", path=path, rows=rows, eligibility_fields=["topic"]
    )


ROWS = [
    {"segment_id": f"s{i}", "topic": "A" if i % 2 else "B"} for i in range(100)
]


# ---------------------------------------------------------------------------
# 1. OPERATOR POLICY
# ---------------------------------------------------------------------------


def test_unknown_operator_rejected():
    with pytest.raises(sp.BudgetAuthorityError) as exc:
        sp.validate_operator_policy(("DIRECT", "NOT_AN_OPERATOR"))
    assert exc.value.code == "OPERATOR_POLICY_UNKNOWN_OPERATOR"


def test_operator_allowed_and_empty_means_all():
    assert sp.operator_allowed("EQUALITY", ()) is True
    assert sp.operator_allowed("EQUALITY", ("DIRECT",)) is False
    assert sp.operator_allowed("DIRECT", ("DIRECT", "EQUALITY")) is True


def test_candidate_rejected_without_repair_and_reasons_recorded():
    items = [
        {"candidate_id": "a", "operator": "DIRECT"},
        {"candidate_id": "b", "operator": "EQUALITY"},
        {"candidate_id": "c", "operator": "COMPOSITE"},
    ]
    accepted, rejected = sp.filter_by_operator_policy(
        items, ("DIRECT",), id_key="candidate_id"
    )
    assert [c["candidate_id"] for c in accepted] == ["a"]
    assert {r["candidate_id"] for r in rejected} == {"b", "c"}
    assert all(r["reason"] == "OPERATOR_NOT_ALLOWED" for r in rejected)


def test_operator_policy_does_not_change_early_budget_target(tmp_path):
    inv = _inventory(tmp_path, ROWS)
    cfg = _cfg(allowed_operator_families=("DIRECT",))
    eb = sp.resolve_early_budget(inventory=inv, config=cfg)
    assert eb.target_total_qa == 10 * 100
    assert eb.allowed_operator_families == ("DIRECT",)


# ---------------------------------------------------------------------------
# 2. PROPOSED vs AUTHORIZED ALLOCATION
# ---------------------------------------------------------------------------


def test_proposed_allocation_is_not_authorized(tmp_path):
    inv = _inventory(tmp_path, ROWS)
    eb = sp.resolve_early_budget(inventory=inv, config=_cfg())
    proposal = sp.propose_allocation(
        early=eb, accepted_type_ids=["t2", "t1"], config=_cfg()
    )
    assert proposal["authorization"] == "PROPOSED"
    assert set(proposal["accepted_types"]) == {"t1", "t2"}
    assert proposal["total_proposed"] == 10 * 100


def test_authorized_allocation_contract_roundtrip(tmp_path):
    inv = _inventory(tmp_path, ROWS)
    cfg = _cfg()
    eb = sp.resolve_early_budget(inventory=inv, config=cfg)
    contracts = [_contract("d1", "DIRECT"), _contract("d2", "DIRECT"), _contract("eq", "EQUALITY")]
    caps = _capacities(contracts, anchors=100)
    final = sp.finalize_allocation(early=eb, contracts=contracts, capacities=caps, config=cfg)
    contract = sp.build_allocation_contract(
        early=eb, contracts=contracts, allocation=final["allocation"], config=cfg,
        authorization="AUTHORIZED",
    )
    assert contract.authorization == "AUTHORIZED"
    assert contract.total_approved == 1000
    sp.verify_allocation_contract(
        contract,
        dataset="d",
        split="train",
        source_fingerprint=eb.source_fingerprint,
        early_budget_fingerprint=eb.fingerprint,
        semantic_contract_fp=sp.semantic_contract_fingerprint(contracts),
        sampling_fp=sp.sampling_fingerprint(cfg),
    )


# ---------------------------------------------------------------------------
# 3. TAMPER / STALENESS
# ---------------------------------------------------------------------------


def _authorized_contract(tmp_path):
    inv = _inventory(tmp_path, ROWS)
    cfg = _cfg()
    eb = sp.resolve_early_budget(inventory=inv, config=cfg)
    contracts = [_contract("d1", "DIRECT"), _contract("d2", "DIRECT"), _contract("eq", "EQUALITY")]
    caps = _capacities(contracts, anchors=100)
    final = sp.finalize_allocation(early=eb, contracts=contracts, capacities=caps, config=cfg)
    contract = sp.build_allocation_contract(
        early=eb, contracts=contracts, allocation=final["allocation"], config=cfg,
        authorization="AUTHORIZED",
    )
    return contract, cfg, contracts


def test_tampered_contract_rejected(tmp_path):
    contract, _, _ = _authorized_contract(tmp_path)
    tampered = contract.__class__(**{**contract.to_dict(), "total_approved": 999})
    with pytest.raises(sp.BudgetAuthorityError) as exc:
        sp.verify_allocation_contract(tampered)
    assert exc.value.code == "ALLOCATION_CONTRACT_TAMPERED"


def test_changed_source_revision_rejected(tmp_path):
    contract, _, _ = _authorized_contract(tmp_path)
    with pytest.raises(sp.BudgetAuthorityError) as exc:
        sp.verify_allocation_contract(contract, source_fingerprint="different")
    assert exc.value.code == "ALLOCATION_CONTRACT_MISMATCH"


def test_changed_semantic_catalog_rejected(tmp_path):
    contract, _, _ = _authorized_contract(tmp_path)
    with pytest.raises(sp.BudgetAuthorityError) as exc:
        sp.verify_allocation_contract(contract, semantic_contract_fp="different")
    assert exc.value.code == "ALLOCATION_CONTRACT_MISMATCH"


def test_changed_sampling_quota_rejected(tmp_path):
    contract, _, _ = _authorized_contract(tmp_path)
    other = _cfg(sampling=SamplingPolicyConfig(equality_strategy="anchor_neighborhood",
                                               equality_positive_per_anchor=8,
                                               equality_negative_per_anchor=8))
    with pytest.raises(sp.BudgetAuthorityError) as exc:
        sp.verify_allocation_contract(contract, sampling_fp=sp.sampling_fingerprint(other))
    assert exc.value.code == "ALLOCATION_CONTRACT_MISMATCH"


# ---------------------------------------------------------------------------
# 4. INSUFFICIENT CAPACITY / SHORTFALL
# ---------------------------------------------------------------------------


def test_insufficient_capacity_yields_shortfall(tmp_path):
    inv = _inventory(tmp_path, ROWS)
    cfg = _cfg()
    eb = sp.resolve_early_budget(inventory=inv, config=cfg)
    contracts = [_contract("d1", "DIRECT"), _contract("d2", "DIRECT"), _contract("eq", "EQUALITY")]
    # capacity caps EQUALITY far below its derived 800
    caps = _capacities(contracts, anchors=100, capacity=120)
    final = sp.finalize_allocation(early=eb, contracts=contracts, capacities=caps, config=cfg)
    assert final["allocation_total"] < 1000
    assert final["shortfall"] and final["shortfall"] > 0


def test_strict_shortfall_policy_fails(tmp_path):
    inv = _inventory(tmp_path, ROWS)
    cfg = _cfg(shortfall_policy="fail")
    eb = sp.resolve_early_budget(inventory=inv, config=cfg)
    contracts = [_contract("d1", "DIRECT"), _contract("d2", "DIRECT"), _contract("eq", "EQUALITY")]
    caps = _capacities(contracts, anchors=100, capacity=120)
    with pytest.raises(sp.BudgetAuthorityError) as exc:
        sp.finalize_allocation(early=eb, contracts=contracts, capacities=caps, config=cfg)
    assert exc.value.code == "EARLY_BUDGET_SHORTFALL"


def test_report_policy_reports_shortfall(tmp_path):
    inv = _inventory(tmp_path, ROWS)
    cfg = _cfg(shortfall_policy="report")
    eb = sp.resolve_early_budget(inventory=inv, config=cfg)
    contracts = [_contract("d1", "DIRECT"), _contract("d2", "DIRECT"), _contract("eq", "EQUALITY")]
    caps = _capacities(contracts, anchors=100, capacity=120)
    final = sp.finalize_allocation(early=eb, contracts=contracts, capacities=caps, config=cfg)
    assert final["shortfall_policy"] == "report"
    assert final["shortfall"] > 0


# ---------------------------------------------------------------------------
# 5. PLANNER ENFORCEMENT
# ---------------------------------------------------------------------------


def test_planner_refuses_mismatched_allocation(tmp_path):
    contract, cfg, _ = _authorized_contract(tmp_path)
    with pytest.raises(ProductionQAError) as exc:
        build_generation_plan(
            contracts=[],
            specs={},
            registry=None,
            index=None,
            config=cfg,
            seed=0,
            budget_by_type={"d1": 1},  # != authorized
            dataset="d",
            dataset_revision="r",
            split="train",
            authorized_allocation=contract,
        )
    assert exc.value.code == "PLANNER_ALLOCATION_MISMATCH"


def test_planner_rejects_tampered_authorized_allocation(tmp_path):
    contract, cfg, _ = _authorized_contract(tmp_path)
    tampered = contract.__class__(**{**contract.to_dict(), "total_approved": 1})
    with pytest.raises(ProductionQAError) as exc:
        build_generation_plan(
            contracts=[], specs={}, registry=None, index=None, config=cfg, seed=0,
            budget_by_type=dict(contract.per_type_budget), dataset="d",
            dataset_revision="r", split="train", authorized_allocation=tampered,
        )
    assert exc.value.code == "ALLOCATION_CONTRACT_TAMPERED"


# ---------------------------------------------------------------------------
# 6. RELEASE AUTHORITY
# ---------------------------------------------------------------------------


def test_release_loads_authorized_allocation(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_vimedcss_release",
        Path(__file__).resolve().parents[1] / "scripts" / "autonomous_qa" / "build_vimedcss_release.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    run = tmp_path / "run"
    run.mkdir()
    path = tmp_path / "vc.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in ROWS) + "\n", encoding="utf-8"
    )
    inv = sp.prepare_source_inventory(
        dataset="vimedcss", split="train", path=path, rows=ROWS,
        eligibility_fields=["topic"],
    )
    cfg = _cfg()
    eb = sp.resolve_early_budget(inventory=inv, config=cfg)
    contracts = [_contract("d1", "DIRECT"), _contract("d2", "DIRECT"), _contract("eq", "EQUALITY")]
    caps = _capacities(contracts, anchors=100)
    final = sp.finalize_allocation(early=eb, contracts=contracts, capacities=caps, config=cfg)
    contract = sp.build_allocation_contract(
        early=eb, contracts=contracts, allocation=final["allocation"], config=cfg,
        authorization="AUTHORIZED",
    )
    (run / "allocation_contract.json").write_text(
        json.dumps(contract.to_dict()), encoding="utf-8"
    )
    counts, total, authority = mod._load_authorized_counts(run, split="train")
    assert authority == "allocation_contract"
    assert counts == contract.per_type_budget
    assert total == contract.total_approved


def test_release_falls_back_to_budget_resolution(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_vimedcss_release2",
        Path(__file__).resolve().parents[1] / "scripts" / "autonomous_qa" / "build_vimedcss_release.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run = tmp_path / "run2"
    run.mkdir()
    (run / "budget_resolution.json").write_text(
        json.dumps({"status": "PER_TYPE_BUDGET", "per_type": {"a": 3, "b": 7}}),
        encoding="utf-8",
    )
    counts, total, authority = mod._load_authorized_counts(run, split="train")
    assert authority == "budget_resolution"
    assert counts == {"a": 3, "b": 7}
    assert total == 10


def test_release_missing_budget_returns_none(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_vimedcss_release3",
        Path(__file__).resolve().parents[1] / "scripts" / "autonomous_qa" / "build_vimedcss_release.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run = tmp_path / "empty"
    run.mkdir()
    counts, total, authority = mod._load_authorized_counts(run, split="train")
    assert counts is None and total is None and authority is None


# ---------------------------------------------------------------------------
# 7. AUDIT IMPLEMENTATION MOVED TO SRC
# ---------------------------------------------------------------------------


def test_production_audit_lives_in_src():
    import src.autonomous_qa.production.qa_audit as qa

    assert callable(qa.audit_qa_records)
    assert callable(qa.audit_record)
    assert not (Path(__file__).resolve().parents[1] / "tests" / "regression" / "qa_audit.py").exists()

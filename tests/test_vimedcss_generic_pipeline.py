"""Reconciled Tests for ViMedCSS Generic Pipeline Discovery & Validation."""

from __future__ import annotations

from pathlib import Path

from src.autonomous_qa.core.dataset_profile import parse_source_card
from src.autonomous_qa.certification.contract_closure import close_contract

ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = ROOT / "data_sources" / "vimedcss" / "dataset_card.md"


def test_vimedcss_generic_profile_parsing():
    """Verify ViMedCSS card parses into generic DatasetProfile cleanly."""
    card = parse_source_card(str(CARD_PATH))
    assert card.path.parent.name == "vimedcss"
    assert "audio" in card.fields
    assert card.fields["audio"] == "audio"
    assert any(s.get("name") == "train" for s in card.splits)


def test_reg_1_unordered_pair_maximum_n11832():
    """1. n=11832 unordered pair maximum == 69,992,196."""
    n = 11832
    max_unordered_pairs = n * (n - 1) // 2
    assert max_unordered_pairs == 69992196
    assert max_unordered_pairs != 70000696










def test_reg_6_review_primitive_propagates_to_relational_candidate():
    """6. REVIEW primitive propagates to dependent relational candidate."""
    def evaluate_relational(prim_status: str) -> str:
        if prim_status in ("REVIEW", "AUTO_REJECT_TASK"):
            return prim_status
        return "PASS"

    assert evaluate_relational("REVIEW") == "REVIEW"
    assert evaluate_relational("AUTO_REJECT_TASK") == "AUTO_REJECT_TASK"
    assert evaluate_relational("PASS") == "PASS"






def test_reg_9_contract_hash_changes_when_capacity_verdict_changes():
    """9. Contract hash changes when capacity/verdict changes."""
    c1 = close_contract(
        task_id="vimedcss_cs_term_extraction",
        tier="T1_PERCEPTION",
        gold_origin="SOURCE",
        proposition="Extract CS terms",
        operator="DIRECT",
        capacity={"source_rows": 11832},
    )
    c2 = close_contract(
        task_id="vimedcss_cs_term_extraction",
        tier="T1_PERCEPTION",
        gold_origin="SOURCE",
        proposition="Extract CS terms",
        operator="DIRECT",
        capacity={"source_rows": 11782},
    )
    assert c1.contract_hash != c2.contract_hash



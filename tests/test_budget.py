"""Budget invariants: 4 calls/round design + 1 manual retry per stage.

Covers the design maximum (4R successes) plus retry headroom.
No automatic retry anywhere; a second attempt of the same stage is only
ever issued by an explicit manual resume.
"""

import pytest

from src.autonomous_qa.authoring.llm_client import LLMCallBudget, build_budget_from_config


def _budget(max_rounds=3):
    return LLMCallBudget(
        {
            "question_style": max_rounds + 1,
            "question_template": max_rounds + 1,
            "paraphrase": max_rounds + 1,
            "quality": max_rounds + 1,
        }
    )


def test_full_design_allowance_ok():
    budget = _budget(3)
    assert budget.max_calls == 16
    for _ in range(3):
        budget.consume("question_style")
        budget.consume("question_template")
        budget.consume("paraphrase")
        budget.consume("quality")
    assert budget.count == 12


def test_one_manual_retry_per_stage_ok():
    budget = _budget(3)
    for _ in range(4):
        budget.consume("question_style")  # 3 rounds + 1 retry
    assert budget.count == 4


def test_second_retry_raises():
    budget = _budget(3)
    for _ in range(4):
        budget.consume("question_style")
    with pytest.raises(RuntimeError, match="cap"):
        budget.consume("question_style")


def test_seventeenth_call_raises():
    budget = _budget(3)
    for _ in range(4):
        budget.consume("question_style")
        budget.consume("question_template")
        budget.consume("paraphrase")
        budget.consume("quality")
    assert budget.count == 16
    with pytest.raises(RuntimeError, match="budget exceeded|cap"):
        budget.consume("quality")


def test_unknown_stage_raises():
    budget = _budget(3)
    with pytest.raises(RuntimeError, match="Unknown stage"):
        budget.consume("style_round_extra")


def test_build_budget_from_config():
    budget = build_budget_from_config({"loop": {"max_rounds": 3}})
    assert budget.max_calls == 16
    assert budget.allowed_calls["question_style"] == 4
    assert budget.allowed_calls["paraphrase"] == 4
    assert budget.allowed_calls["quality"] == 4

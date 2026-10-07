"""Production zero-LLM guard for Autonomous QA.

Production (planner / knowledge lookup / rule executor / renderer / final
audit) must never import or call the LLM client. This module configures
the AutoQA production inventory scanned by common zero-LLM mechanics.
"""

from __future__ import annotations

from typing import Sequence

from src.common.zero_llm import FORBIDDEN_PATTERNS, scan_modules_for_forbidden_calls

# Production path modules (planning, rendering, audit, canonical resolution).
PRODUCTION_MODULES: tuple[str, ...] = (
    "src/autonomous_qa/production/production_qa.py",
    "tests/regression/production_plan_audit.py",
    "src/autonomous_qa/language/template_renderer.py",
    "src/autonomous_qa/language/composite_language.py",
    # NOTE: src/autonomous_qa/language/language_quality.py is a MIXED module (it also hosts an
    # R&D-only language-quality function guarded by `real_llm`). It is excluded
    # from the source scan and instead covered by the runtime zero-LLM test.
    "src/autonomous_qa/compiler/semantic_task.py",
    "src/autonomous_qa/compiler/semantic_comparators.py",
    "src/common/answer_schema.py",
    "src/autonomous_qa/compiler/canonical_resources.py",
    "src/autonomous_qa/certification/promotion_gate.py",
    "src/autonomous_qa/certification/contract_closure.py",
)


def scan_production_zero_llm(
    modules: Sequence[str] | None = None,
) -> dict:
    """Scan AutoQA production modules for forbidden LLM/network patterns."""
    target_modules = modules if modules is not None else PRODUCTION_MODULES
    return scan_modules_for_forbidden_calls(target_modules)


def assert_production_zero_llm() -> None:
    result = scan_production_zero_llm()
    if result["status"] != "PASS":
        raise RuntimeError(f"PRODUCTION_LLM_DEPENDENCY_DETECTED:{result['offenders']}")

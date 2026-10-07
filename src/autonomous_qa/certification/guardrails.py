"""Generic Guardrail Registry.

Chooses required checks from contract properties (tier, operator, answer kind,
visible roles and gold origin) — never from dataset identity.
Guardrail functions are dataset-agnostic and consume a context dict supplied
by the caller (profile, label counts, comparator, capacity, anomaly signatures, ...).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.autonomous_qa.certification.comparator_collision import comparator_collision_audit
from src.autonomous_qa.core.information_value import (
    information_metrics,
    information_value_decision,
)
from src.common.severity import max_severity

GUARDRAIL_IDS = (
    "SOURCE_SUPPORT",
    "GOLD_DERIVABILITY",
    "MODALITY_OBSERVABILITY",
    "LABEL_INFORMATION",
    "CAPACITY",
    "COMPARATOR_COLLISION",
    "INVARIANCE",
    "SPLIT_ISOLATION",
    "NEGATIVE_VALIDITY",
    "PAIR_VALIDITY",
    "COMPONENT_VISIBILITY",
    "TEXT_ONLY_SHORTCUT",
    "COMPOSITE_REDUNDANCY",
    "SOURCE_CONSISTENCY",
    "LANGUAGE_ALIGNMENT",
)



class GuardrailResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    guardrail_id: str
    severity: str
    detail: dict[str, Any] = Field(default_factory=dict)


class GuardrailReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    task_id: str
    tier: str | None
    gold_origin: str | None
    results: tuple[GuardrailResult, ...]
    severity: str
    status: str

    def by_id(self, guardrail_id: str) -> GuardrailResult:
        for result in self.results:
            if result.guardrail_id == guardrail_id:
                return result
        raise KeyError(guardrail_id)


def _spec_attr(spec: Any, name: str, default: Any = None) -> Any:
    if isinstance(spec, dict):
        return spec.get(name, default)
    return getattr(spec, name, default)


def _hidden_fields(spec: Any) -> list[str]:
    mapping = _spec_attr(spec, "source_role_mapping", {}) or {}
    hidden = []
    for key in ("source_field", "audio_field", "visible_field"):
        value = mapping.get(key)
        if isinstance(value, list):
            hidden.extend(str(v) for v in value)
        elif value:
            hidden.append(str(value))
    return hidden


# ---------------------------------------------------------------------------
# guardrail implementations
# ---------------------------------------------------------------------------


def guard_source_support(spec, context) -> GuardrailResult:
    fields = set(context.get("source_fields", []))
    hidden = _hidden_fields(spec)
    if fields and hidden and not set(hidden) <= fields:
        return GuardrailResult(
            guardrail_id="SOURCE_SUPPORT",
            severity="AUTO_REJECT_TASK",
            detail={"unknown_fields": sorted(set(hidden) - fields)},
        )
    return GuardrailResult(guardrail_id="SOURCE_SUPPORT", severity="PASS")


def guard_gold_derivability(spec, context) -> GuardrailResult:
    outputs = _spec_attr(spec, "outputs", ()) or ()
    if not outputs:
        return GuardrailResult(
            guardrail_id="GOLD_DERIVABILITY",
            severity="AUTO_REJECT_TASK",
            detail={"reason": "NO_OUTPUTS"},
        )
    return GuardrailResult(guardrail_id="GOLD_DERIVABILITY", severity="PASS")


def guard_modality_observability(spec, context) -> GuardrailResult:
    non_observable = set(context.get("non_observable_fields", []))
    hidden = _hidden_fields(spec)
    if non_observable and hidden and set(hidden) & non_observable:
        return GuardrailResult(
            guardrail_id="MODALITY_OBSERVABILITY",
            severity="AUTO_REJECT_TASK",
            detail={"non_observable": sorted(set(hidden) & non_observable)},
        )
    return GuardrailResult(guardrail_id="MODALITY_OBSERVABILITY", severity="PASS")


def guard_label_information(spec, context) -> GuardrailResult:
    label_counts = context.get("label_counts")
    if not label_counts:
        return GuardrailResult(
            guardrail_id="LABEL_INFORMATION",
            severity="PASS",
            detail={"note": "no_label_counts_supplied"},
        )
    metrics = information_metrics(
        [label for label, count in label_counts.items() for _ in range(int(count))]
    )
    decision = information_value_decision(
        metrics, target_size=context.get("target_size")
    )
    return GuardrailResult(
        guardrail_id="LABEL_INFORMATION",
        severity=decision["severity"],
        detail={"reasons": decision["reasons"], "metrics": metrics},
    )


def guard_capacity(spec, context) -> GuardrailResult:
    capacity = context.get("capacity")
    if capacity is None:
        return GuardrailResult(guardrail_id="CAPACITY", severity="PASS")
    if int(capacity) < int(context.get("min_capacity", 2)):
        return GuardrailResult(
            guardrail_id="CAPACITY",
            severity="AUTO_REJECT_TASK",
            detail={"capacity": capacity},
        )
    return GuardrailResult(guardrail_id="CAPACITY", severity="PASS")


def guard_comparator_collision(spec, context) -> GuardrailResult:
    comparator = context.get("comparator")
    raw_values = context.get("authoritative_values")
    if comparator is None or not raw_values:
        return GuardrailResult(guardrail_id="COMPARATOR_COLLISION", severity="PASS")
    audit = comparator_collision_audit(
        raw_values,
        comparator,
        allowed_collapse_groups=context.get("allowed_collapses", ()),
    )
    return GuardrailResult(
        guardrail_id="COMPARATOR_COLLISION",
        severity=audit["severity"],
        detail=audit,
    )


def guard_invariance(spec, context) -> GuardrailResult:
    comparator_id = _spec_attr(spec, "comparator_id")
    if not comparator_id:
        return GuardrailResult(guardrail_id="INVARIANCE", severity="PASS")
    invariances = _spec_attr(spec, "invariances", ()) or ()
    if not invariances:
        return GuardrailResult(
            guardrail_id="INVARIANCE",
            severity="REVIEW",
            detail={"reason": "NO_DECLARED_INVARIANCES"},
        )
    return GuardrailResult(guardrail_id="INVARIANCE", severity="PASS")


def guard_split_isolation(spec, context) -> GuardrailResult:
    reserved_refs = int(context.get("reserved_split_refs", 0))
    if reserved_refs > 0:
        return GuardrailResult(
            guardrail_id="SPLIT_ISOLATION",
            severity="BLOCKING",
            detail={"reserved_split_refs": reserved_refs},
        )
    return GuardrailResult(guardrail_id="SPLIT_ISOLATION", severity="PASS")


def guard_negative_validity(spec, context) -> GuardrailResult:
    minimum = context.get("negative_capacity_min")
    if minimum is None:
        return GuardrailResult(guardrail_id="NEGATIVE_VALIDITY", severity="PASS")
    if int(minimum) < 1:
        return GuardrailResult(
            guardrail_id="NEGATIVE_VALIDITY",
            severity="AUTO_REJECT_TASK",
            detail={"negative_capacity_min": minimum},
        )
    return GuardrailResult(guardrail_id="NEGATIVE_VALIDITY", severity="PASS")


def guard_pair_validity(spec, context) -> GuardrailResult:
    arity = int(_spec_attr(spec, "audio_arity", 1) or 1)
    if arity < 2:
        return GuardrailResult(guardrail_id="PAIR_VALIDITY", severity="PASS")
    capacity = context.get("pair_capacity")
    if capacity is not None and int(capacity) < 1:
        return GuardrailResult(
            guardrail_id="PAIR_VALIDITY",
            severity="AUTO_REJECT_TASK",
            detail={"pair_capacity": capacity},
        )
    return GuardrailResult(guardrail_id="PAIR_VALIDITY", severity="PASS")


def guard_component_visibility(spec, context) -> GuardrailResult:
    visible = set(context.get("component_visible_types", []))
    task_id = _spec_attr(spec, "type_id")
    if task_id in visible:
        return GuardrailResult(
            guardrail_id="COMPONENT_VISIBILITY",
            severity="AUTO_REJECT_TASK",
            detail={"reason": "OUTPUT_COMPONENT_VISIBLE_IN_INPUT"},
        )
    return GuardrailResult(guardrail_id="COMPONENT_VISIBILITY", severity="PASS")


def guard_text_only_shortcut(spec, context) -> GuardrailResult:
    risk = context.get("text_only_shortcut_risk")
    if risk:
        return GuardrailResult(
            guardrail_id="TEXT_ONLY_SHORTCUT",
            severity="REVIEW",
            detail=context.get("text_only_shortcut_detail", {}),
        )
    return GuardrailResult(guardrail_id="TEXT_ONLY_SHORTCUT", severity="PASS")


def guard_composite_redundancy(spec, context) -> GuardrailResult:
    classification = context.get("composite_classification")
    if classification == "REDUNDANT_COMPOSITE":
        return GuardrailResult(
            guardrail_id="COMPOSITE_REDUNDANCY",
            severity="AUTO_REJECT_TASK",
            detail={"classification": classification},
        )
    return GuardrailResult(guardrail_id="COMPOSITE_REDUNDANCY", severity="PASS")


def guard_source_consistency(spec, context) -> GuardrailResult:
    signatures = context.get("anomaly_signatures")
    if signatures is None:
        return GuardrailResult(guardrail_id="SOURCE_CONSISTENCY", severity="PASS")
    from src.autonomous_qa.certification.task_sanitation import SanitationPolicy, classify_anomalies

    classification = classify_anomalies(
        signatures,
        int(context.get("total_rows", len(signatures) or 1)),
        context.get("sanitation_policy") or SanitationPolicy(),
    )
    return GuardrailResult(
        guardrail_id="SOURCE_CONSISTENCY",
        severity=classification["severity"],
        detail=classification,
    )










def guard_language_alignment(spec, context) -> GuardrailResult:
    if context.get("language_preflight_result") == "PREFLIGHT_REVIEW":
        return GuardrailResult(
            guardrail_id="LANGUAGE_ALIGNMENT",
            severity="REVIEW",
            detail=context.get("language_preflight_detail", {}),
        )
    return GuardrailResult(guardrail_id="LANGUAGE_ALIGNMENT", severity="PASS")


_GUARDRAIL_FUNCTIONS: dict[str, Callable] = {
    "SOURCE_SUPPORT": guard_source_support,
    "GOLD_DERIVABILITY": guard_gold_derivability,
    "MODALITY_OBSERVABILITY": guard_modality_observability,
    "LABEL_INFORMATION": guard_label_information,
    "CAPACITY": guard_capacity,
    "COMPARATOR_COLLISION": guard_comparator_collision,
    "INVARIANCE": guard_invariance,
    "SPLIT_ISOLATION": guard_split_isolation,
    "NEGATIVE_VALIDITY": guard_negative_validity,
    "PAIR_VALIDITY": guard_pair_validity,
    "COMPONENT_VISIBILITY": guard_component_visibility,
    "TEXT_ONLY_SHORTCUT": guard_text_only_shortcut,
    "COMPOSITE_REDUNDANCY": guard_composite_redundancy,
    "SOURCE_CONSISTENCY": guard_source_consistency,
    "LANGUAGE_ALIGNMENT": guard_language_alignment,
}


class GuardrailRegistry:
    """Selects guardrails from contract properties; runs them generically."""

    def applicable(self, spec, context: dict | None = None) -> tuple[str, ...]:
        context = context or {}
        tier = _spec_attr(spec, "tier")
        operator = str(_spec_attr(spec, "operator") or "")
        arity = int(_spec_attr(spec, "audio_arity", 1) or 1)
        kind = str(_spec_attr(spec, "kind") or "")
        comparator_id = _spec_attr(spec, "comparator_id")

        ids = [
            "SOURCE_SUPPORT",
            "GOLD_DERIVABILITY",
            "MODALITY_OBSERVABILITY",
            "LABEL_INFORMATION",
            "CAPACITY",
            "SPLIT_ISOLATION",
            "COMPONENT_VISIBILITY",
            "TEXT_ONLY_SHORTCUT",
            "SOURCE_CONSISTENCY",
        ]
        if comparator_id:
            ids += ["COMPARATOR_COLLISION", "INVARIANCE"]
        if operator in ("TARGET_MATCH", "PAIRWISE_SELECTION"):
            ids.append("NEGATIVE_VALIDITY")
        if arity >= 2:
            ids.append("PAIR_VALIDITY")
        if kind == "STRUCTURED" or operator == "COMPOSITE":
            ids.append("COMPOSITE_REDUNDANCY")
        if context.get("language_preflight_result") is not None:
            ids.append("LANGUAGE_ALIGNMENT")
        # stable, de-duplicated order
        seen: list[str] = []
        for guardrail_id in ids:
            if guardrail_id not in seen:
                seen.append(guardrail_id)
        return tuple(seen)

    def run(self, spec, context: dict | None = None) -> GuardrailReport:
        context = context or {}
        results = [
            _GUARDRAIL_FUNCTIONS[guardrail_id](spec, context)
            for guardrail_id in self.applicable(spec, context)
        ]
        severity = max_severity([r.severity for r in results])
        if severity == "PASS" or severity in ("REPORT_ONLY",):
            status = "PASS"
        elif severity == "REVIEW":
            status = "REVIEW"
        else:
            status = "FAIL"
        return GuardrailReport(
            task_id=str(_spec_attr(spec, "type_id") or ""),
            tier=_spec_attr(spec, "tier"),
            gold_origin=_spec_attr(spec, "gold_origin"),
            results=tuple(results),
            severity=severity,
            status=status,
        )


DEFAULT_REGISTRY = GuardrailRegistry()


def run_guardrails(spec, context: dict | None = None, registry=None) -> GuardrailReport:
    return (registry or DEFAULT_REGISTRY).run(spec, context)

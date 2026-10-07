"""Generic output-component visibility / leakage audit.

This is the stronger, component-level invariant introduced for the R&D
re-authoring milestone:

    For every requested output component (role, kind), compare the
    model-visible textual fields against the deterministic gold value under
    RAW equality AND the output's relevant semantic comparator.

When a visible field already contains a requested gold *field_value* output,
the component is recoverable from the input alone.  For composite tasks that
is ``OUTPUT_COMPONENT_VISIBLE_IN_INPUT``.

The audit never makes a semantic decision by itself; it classifies each
observation deterministically so a human/promotion gate can act on it.

Vocabulary (kept deliberately distinct, see the R&D taxonomy):

- INTENDED_VERIFICATION_CONTEXT: the visible candidate is the object being
  verified; a boolean/index output still needs the audio.
- TEXT_ONLY_SHORTCUT_RISK: a text-only heuristic predicts the label above the
  configured risk threshold.
- OUTPUT_COMPONENT_VISIBLE_IN_INPUT: a requested field_value component is
  literally present in the model-visible text.
- DIRECT_LABEL_LEAKAGE: the requested atomic label itself is visible.
- SOURCE_GROUP_SHORTCUT: labels concentrate by source group (e.g. speaker).
- SPLIT_LEAKAGE: reserved-split dependency (not evaluated here).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.autonomous_qa.compiler.semantic_comparators import (
    ComparatorRegistry,
    apply_comparator,
    load_comparator_registry,
)
from src.autonomous_qa.compiler.semantic_task import SemanticCatalog, SemanticTaskSpec, load_semantic_catalog

FIELD_VALUE = "field_value"
BOOLEAN = "boolean"
AUDIO_INDEX = "audio_index"

CLASS_INTENDED = "INTENDED_VERIFICATION_CONTEXT"
CLASS_SHORTCUT = "TEXT_ONLY_SHORTCUT_RISK"
CLASS_COMPONENT_VISIBLE = "OUTPUT_COMPONENT_VISIBLE_IN_INPUT"
CLASS_DIRECT_LABEL = "DIRECT_LABEL_LEAKAGE"

TEXT_ONLY_SHORTCUT_THRESHOLD = 0.60


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_outputs(task: SemanticTaskSpec) -> list[tuple[str, str]]:
    return [(c.role, c.kind) for c in task.outputs]


def _visible_and_gold(row: dict) -> tuple[list[str], dict[str, Any]]:
    """Normalize the two canonical plan row shapes into (visible, gold)."""
    if "output_gold" in row:
        visible = [str(v) for v in (row.get("visible_context") or [])]
        return visible, dict(row.get("output_gold") or {})
    # ViMD generation-plan shape: gold={"kind","value"}, target/target_display.
    gold_obj = row.get("gold") or {}
    visible: list[str] = []
    if row.get("target") is not None:
        visible.append(str(row["target"]))
    if row.get("target_display") is not None:
        visible.append(str(row["target_display"]))
    return visible, {
        "__single__": gold_obj.get("value"),
        "__kind__": gold_obj.get("kind"),
    }


def _role_gold(gold: dict[str, Any], role: str) -> Any:
    if "__single__" in gold:
        return gold.get("__single__")
    return gold.get(role)


def audit_task_rows(
    task: SemanticTaskSpec,
    rows: list[dict],
    registry: ComparatorRegistry,
) -> dict[str, Any]:
    """Compute component-visibility + shortcut metrics for one task."""
    comparator = registry.by_id(task.comparator_id) if task.comparator_id else None
    role_stats: dict[str, dict[str, Any]] = {}

    bool_roles = [c.role for c in task.outputs if c.kind == BOOLEAN]
    index_roles = [c.role for c in task.outputs if c.kind == AUDIO_INDEX]

    for component in task.outputs:
        if component.kind != FIELD_VALUE:
            continue
        stat = {
            "n": 0,
            "raw_exposed": 0,
            "comparator_exposed": 0,
            "comparator_id": task.comparator_id,
            "branches": {},
            "visible_values": Counter(),
        }
        branches: dict[str, list[int]] = {}
        for row in rows:
            visible, gold = _visible_and_gold(row)
            value = _role_gold(gold, component.role)
            if value is None and "__single__" not in gold:
                continue
            stat["n"] += 1
            gtext = str(value)
            raw_hit = any(v == gtext for v in visible)
            comp_hit = raw_hit
            if not comp_hit and comparator is not None:
                comp_hit = any(apply_comparator(comparator, v, gtext) for v in visible)
            if raw_hit:
                stat["raw_exposed"] += 1
            if comp_hit:
                stat["comparator_exposed"] += 1
            for v in visible:
                stat["visible_values"][v] += 1
            branch = _branch_label(gold, bool_roles, index_roles)
            entry = branches.setdefault(branch, [0, 0])
            entry[0] += 1
            if comp_hit:
                entry[1] += 1
        total = stat["n"]
        stat["pct_exposed"] = (
            round(100.0 * stat["comparator_exposed"] / total, 2) if total else 0.0
        )
        stat["branches"] = {
            k: {"n": v[0], "exposed": v[1]} for k, v in sorted(branches.items())
        }
        stat["visible_values"] = dict(stat["visible_values"].most_common(5))
        role_stats[component.role] = stat

    # Text-only shortcut on the primary atomic label (boolean/index).
    shortcut = _text_only_shortcut(task, rows, bool_roles, index_roles, registry)

    classification = _classify(task, role_stats, shortcut)
    return {
        "type_id": task.type_id,
        "kind": task.kind,
        "operator": task.operator,
        "comparator_id": task.comparator_id,
        "row_count": len(rows),
        "role_stats": role_stats,
        "text_only_shortcut": shortcut,
        "classification": classification,
    }


def _branch_label(
    gold: dict[str, Any], bool_roles: list[str], index_roles: list[str]
) -> str:
    for role in bool_roles:
        value = _role_gold(gold, role)
        if value is True:
            return "true"
        if value is False:
            return "false"
    for role in index_roles:
        value = _role_gold(gold, role)
        if value is not None:
            return f"index_{value}"
    return "all"


def _text_only_shortcut(
    task: SemanticTaskSpec,
    rows: list[dict],
    bool_roles: list[str],
    index_roles: list[str],
    registry: ComparatorRegistry,
) -> dict[str, Any]:
    """How well can a pure text-only heuristic predict the label?

    Heuristic (deterministic, no LLM): if the visible context contains any
    requested field_value gold, answer the "positive" label; otherwise the
    "negative" label.  This is an upper bound on a copy/constant shortcut.
    """
    label_roles = bool_roles or index_roles
    if not label_roles:
        # DIRECT transcription tasks have no label to shortcut.
        return {"applicable": False}
    role = label_roles[0]
    positive_label = 1 if role in index_roles else True
    negative_label = 0 if role in index_roles else False

    has_field_output = any(c.kind == FIELD_VALUE for c in task.outputs)
    has_visible_text = any(
        (row.get("visible_context") or row.get("target") is not None) for row in rows
    )

    correct = 0
    total = 0
    label_counts: Counter = Counter()
    for row in rows:
        visible, gold = _visible_and_gold(row)
        label = _role_gold(gold, role)
        if label is None:
            continue
        total += 1
        label_counts[str(label)] += 1
        field_visible = False
        for component in task.outputs:
            if component.kind != FIELD_VALUE:
                continue
            value = _role_gold(gold, component.role)
            if value is None:
                continue
            gtext = str(value)
            comp = registry.by_id(task.comparator_id) if task.comparator_id else None
            if any(str(v) == gtext for v in visible):
                field_visible = True
                break
            if comp is not None and any(
                apply_comparator(comp, v, gtext) for v in visible
            ):
                field_visible = True
                break
        predicted = positive_label if field_visible else negative_label
        if predicted == label:
            correct += 1
    copy_accuracy = correct / total if total else 0.0
    base_rate = max(label_counts.values()) / total if total else 0.0

    text_driven = has_field_output and has_visible_text
    risk = False
    reason = None
    if text_driven and copy_accuracy >= TEXT_ONLY_SHORTCUT_THRESHOLD:
        risk = True
        reason = "COPY_VISIBLE_FIELD"
    elif base_rate >= TEXT_ONLY_SHORTCUT_THRESHOLD:
        risk = True
        reason = "BASE_RATE_PRIOR"
    return {
        "applicable": True,
        "label_role": role,
        "n": total,
        "base_rate": round(base_rate, 4),
        "copy_accuracy": round(copy_accuracy, 4),
        "text_driven": text_driven,
        "risk": risk,
        "reason": reason,
        "label_distribution": dict(label_counts),
    }


def _classify(
    task: SemanticTaskSpec,
    role_stats: dict[str, dict[str, Any]],
    shortcut: dict[str, Any],
) -> str:
    exposed = {
        role: stat["comparator_exposed"]
        for role, stat in role_stats.items()
        if stat["comparator_exposed"] > 0
    }
    if task.is_composite and exposed:
        return CLASS_COMPONENT_VISIBLE
    if task.kind == "ATOMIC" and task.outputs[0].kind != FIELD_VALUE:
        if shortcut.get("risk"):
            return CLASS_SHORTCUT
        return CLASS_INTENDED
    if task.kind == "ATOMIC" and task.outputs[0].kind == FIELD_VALUE:
        # DIRECT: the requested text is the answer; visible context should be
        # empty. Any exposure would be direct label leakage.
        if exposed:
            return CLASS_DIRECT_LABEL
        return CLASS_INTENDED
    if shortcut.get("risk"):
        return CLASS_SHORTCUT
    return CLASS_INTENDED


def audit_plan(
    dataset_id: str,
    plan_path: Path,
    catalog_path: Path,
    registry: ComparatorRegistry | None = None,
) -> dict[str, Any]:
    """Full deterministic audit of a frozen plan against its catalog."""
    registry = registry or load_comparator_registry()
    catalog: SemanticCatalog = load_semantic_catalog(catalog_path)
    tasks = {t.type_id: t for t in catalog.tasks}
    rows = _load_jsonl(Path(plan_path))
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["type_id"], []).append(row)

    results = {}
    for type_id, task in tasks.items():
        results[type_id] = audit_task_rows(task, grouped.get(type_id, []), registry)

    flagged = {
        t: r["classification"]
        for t, r in results.items()
        if r["classification"]
        in (CLASS_COMPONENT_VISIBLE, CLASS_DIRECT_LABEL, CLASS_SHORTCUT)
    }
    return {
        "dataset_id": dataset_id,
        "plan_path": str(plan_path),
        "row_count": len(rows),
        "type_count": len(tasks),
        "types": results,
        "flagged": flagged,
        "component_visible_types": sorted(
            t
            for t, r in results.items()
            if r["classification"] == CLASS_COMPONENT_VISIBLE
        ),
    }


def render_leakage_table(audit: dict[str, Any]) -> str:
    lines = [
        f"# Component leakage audit: {audit['dataset_id']}",
        f"plan={audit['plan_path']} rows={audit['row_count']}",
        "",
        "| type | kind | role | n | raw | comparator | % | classification |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for type_id, result in audit["types"].items():
        if not result["role_stats"]:
            continue
        for role, stat in result["role_stats"].items():
            lines.append(
                f"| {type_id} | {result['kind']} | {role} | {stat['n']} | "
                f"{stat['raw_exposed']} | {stat['comparator_exposed']} | "
                f"{stat['pct_exposed']} | {result['classification']} |"
            )
    return "\n".join(lines) + "\n"

"""Audit a ViMedCSS V3 generation plan (generic 4+4, split-aware).

Derives all expected counts from the config (per-anchor quotas) and the actual
eligible-row count for the split — nothing is hardcoded. Validates:
  * split isolation (every record in the requested split);
  * DIRECT strict 1:1 coverage;
  * pairwise anchor-neighborhood N×(pos+neg), unordered uniqueness, no self
    pairs, deterministic gold.

Exits non-zero on any failure. No LLM, no network, no canonical mutation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.autonomous_qa.certification.anchor_neighborhood import (
    anchor_neighborhood_feasibility,
    audit_anchor_neighborhood_plan,
    audit_direct_coverage,
    audit_split_isolation,
)
from src.autonomous_qa.certification.qa_sampling import meta_of
from src.autonomous_qa.language.language_quality import (
    ProductionGenerationConfig,
)
from src.autonomous_qa.language.template_contracts import (
    build_type_contracts,
)
from src.autonomous_qa.production.production_qa import (
    DATASET_SOURCES,
    load_flat_metadata,
    split_metadata_path,
)
from src.autonomous_qa.production.source_indexing import (
    build_row_identities,
    index_dataset_rows,
)

DIRECT_TASKS = ("vimedcss_topic_classification", "vimedcss_cs_terms_count")
PAIRWISE_TASK = "vimedcss_pairwise_topic_same"
FACTOR_FIELD = "topic"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", default="vimedcss")
    parser.add_argument("--split", default=None)
    args = parser.parse_args(argv)

    config = ProductionGenerationConfig.model_validate(
        yaml.safe_load(args.config.read_text(encoding="utf-8"))
    )
    split = args.split or config.split or "train"
    pos = config.sampling.equality_positive_per_anchor
    neg = config.sampling.equality_negative_per_anchor

    plan_path = args.run_dir / "generation_plan.jsonl"
    if not plan_path.exists():
        print(json.dumps({"error": "PLAN_MISSING", "detail": str(plan_path)}))
        return 2
    records = [
        json.loads(line)
        for line in plan_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    source = dict(DATASET_SOURCES[args.dataset])
    source["dataset"] = args.dataset
    metadata_path = split_metadata_path(source, split)
    rows = load_flat_metadata(metadata_path)
    row_ids, _ = build_row_identities(rows, source)
    contracts_all = build_type_contracts(
        json.loads(source["type_registry"].read_text(encoding="utf-8"))
    )
    contracts = [c for c in contracts_all if c.source_status == "SUPPORTED"]

    # Generic field spec resolution (declarative sidecar when available).
    from src.autonomous_qa.language.template_engine import vimedcss_field_specs

    specs = vimedcss_field_specs()
    index = index_dataset_rows(
        rows, row_ids, contracts, specs, hidden_identifier_field="speakerID"
    )
    anchors = len(index.valid_row_ids[FACTOR_FIELD])

    expected = {
        **{task: anchors for task in DIRECT_TASKS},
        PAIRWISE_TASK: anchors * (pos + neg),
    }
    expected_total = sum(expected.values())

    failures: list[str] = []
    isolation = audit_split_isolation(records, split=split)
    if not isolation["passed"]:
        failures.extend(isolation["failures"])

    feasibility = anchor_neighborhood_feasibility(
        index, FACTOR_FIELD, positive_per_anchor=pos, negative_per_anchor=neg
    )
    if not feasibility["feasible"]:
        failures.append("INFEASIBLE_POLICY")

    report: dict = {
        "split": split,
        "anchors": anchors,
        "expected_counts": expected,
        "expected_total": expected_total,
        "isolation": isolation,
        "feasibility": feasibility,
        "direct": {},
        "pairwise": {},
    }
    for task in DIRECT_TASKS:
        result = audit_direct_coverage(records, type_id=task, expected_rows=anchors)
        report["direct"][task] = result
        if not result["passed"]:
            failures.append(f"DIRECT_FAIL:{task}")

    def lookup(row_id: str):
        return meta_of(index.row_by_id[row_id]).get(FACTOR_FIELD)

    pairwise = audit_anchor_neighborhood_plan(
        records,
        type_id=PAIRWISE_TASK,
        row_value_lookup=lookup,
        positive_per_anchor=pos,
        negative_per_anchor=neg,
        expected_anchors=anchors,
    )
    report["pairwise"] = pairwise
    if not pairwise["passed"]:
        failures.append("PAIRWISE_FAIL")

    produced_total = len(records)
    if produced_total != expected_total:
        failures.append(f"TOTAL:{produced_total}!={expected_total}")

    report["produced_total"] = produced_total
    report["failures"] = failures
    report["passed"] = not failures
    (args.run_dir / "final_plan_audit_v3.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "passed": report["passed"],
        "split": split,
        "anchors": anchors,
        "expected_total": expected_total,
        "produced_total": produced_total,
        "pairwise_same": pairwise.get("same_topic"),
        "pairwise_different": pairwise.get("different_topic"),
        "failures": failures,
    }, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

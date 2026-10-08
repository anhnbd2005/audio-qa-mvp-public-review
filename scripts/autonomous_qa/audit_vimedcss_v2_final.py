"""Audit the ViMedCSS V2 final generation plan.

Validates the eventually-generated plan (by the human operator) before release
freeze:
  * DIRECT tasks: strict 1:1 TRAIN coverage (11832 each);
  * pairwise: anchor-neighborhood 2 SAME + 2 DIFFERENT per anchor,
    exact totals, no unordered duplicates, no self pairs, deterministic gold.

Exits non-zero on any failure. No LLM, no network, no canonical mutation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.autonomous_qa.certification.anchor_neighborhood import (
    anchor_neighborhood_feasibility,
    audit_anchor_neighborhood_plan,
    audit_direct_coverage,
)
from src.autonomous_qa.certification.qa_sampling import meta_of
from src.autonomous_qa.language.template_contracts import (
    build_type_contracts,
)
from src.autonomous_qa.language.template_engine import (
    vimedcss_field_specs,
)
from src.autonomous_qa.production.production_qa import (
    DATASET_SOURCES,
    load_flat_metadata,
)
from src.autonomous_qa.production.source_indexing import (
    build_row_identities,
    index_dataset_rows,
)

EXPECTED_DIRECT = 11832
EXPECTED_PAIRWISE = 47328
EXPECTED_ANCHORS = 11832
POS_PER_ANCHOR = 2
NEG_PER_ANCHOR = 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    plan_path = args.run_dir / "generation_plan.jsonl"
    if not plan_path.exists():
        print(json.dumps({"error": "PLAN_MISSING", "detail": str(plan_path)}))
        return 2
    records = [
        json.loads(line)
        for line in plan_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    source = dict(DATASET_SOURCES["vimedcss"])
    rows = load_flat_metadata(source["metadata_path"])
    row_ids, _ = build_row_identities(rows, source)
    contracts_all = build_type_contracts(
        json.loads(source["type_registry"].read_text(encoding="utf-8"))
    )
    contracts = [c for c in contracts_all if c.source_status == "SUPPORTED"]
    specs = vimedcss_field_specs()
    index = index_dataset_rows(
        rows, row_ids, contracts, specs, hidden_identifier_field="speakerID"
    )

    field = "topic"
    feasibility = anchor_neighborhood_feasibility(
        index, field,
        positive_per_anchor=POS_PER_ANCHOR,
        negative_per_anchor=NEG_PER_ANCHOR,
    )

    report: dict = {"feasibility": feasibility, "direct": {}, "pairwise": {}}
    failures: list[str] = []
    if not feasibility["feasible"]:
        failures.append("INFEASIBLE_POLICY")

    for type_id in ("vimedcss_topic_classification", "vimedcss_cs_terms_count"):
        result = audit_direct_coverage(records, type_id=type_id, expected_rows=EXPECTED_DIRECT)
        report["direct"][type_id] = result
        if not result["passed"]:
            failures.append(f"DIRECT_FAIL:{type_id}")

    def lookup(row_id: str):
        return meta_of(index.row_by_id[row_id]).get(field)

    pairwise = audit_anchor_neighborhood_plan(
        records,
        type_id="vimedcss_pairwise_topic_same",
        row_value_lookup=lookup,
        positive_per_anchor=POS_PER_ANCHOR,
        negative_per_anchor=NEG_PER_ANCHOR,
        expected_anchors=EXPECTED_ANCHORS,
    )
    report["pairwise"] = pairwise
    if not pairwise["passed"]:
        failures.append("PAIRWISE_FAIL")

    report["failures"] = failures
    report["passed"] = not failures
    (args.run_dir / "final_plan_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "passed": report["passed"],
        "failures": failures,
        "feasible": feasibility["feasible"],
        "pairwise_same": pairwise.get("same_topic"),
        "pairwise_different": pairwise.get("different_topic"),
        "pairwise_anchors": pairwise.get("anchors"),
    }, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

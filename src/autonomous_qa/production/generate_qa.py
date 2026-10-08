"""Production QA generator CLI (zero-LLM runtime).

Usage:
    python -m src.autonomous_qa.production.generate_qa --dataset vimd \
        --config configs/qa_generation_production.yaml \
        [--readiness-only | --plan-only] [--debug-sample N] \
        [--output-root PATH] [--run-id ID]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.common.config import ROOT
from src.autonomous_qa.production.production_qa import ProductionQAError, run_production_qa

_SUCCESS_CONCLUSIONS = {
    "PASS",
    "PASS_WITH_REVIEW",
    "READY_AWAITING_BUDGET",
    "READY_UNTESTED",
    "BLOCKED",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic production QA generator (no LLM calls)."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "qa_generation_production.yaml",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--readiness-only",
        action="store_true",
        help="validate frozen inputs and compute readiness only",
    )
    mode_group.add_argument(
        "--plan-only",
        action="store_true",
        help="resolve budget and write the generation plan, no QA output",
    )
    parser.add_argument(
        "--debug-sample",
        type=int,
        default=None,
        metavar="N",
        help="produce N debug QA records with a debug budget",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--split",
        default=None,
        help="Authorized dataset split (default: config split or 'train')",
    )
    args = parser.parse_args(argv)

    if args.debug_sample is not None:
        mode = "debug_sample"
    elif args.readiness_only:
        mode = "readiness"
    elif args.plan_only:
        mode = "plan"
    else:
        mode = "full"

    try:
        summary = run_production_qa(
            dataset=args.dataset,
            config_path=args.config,
            mode=mode,  # type: ignore[arg-type]
            output_root=args.output_root,
            run_id=args.run_id,
            debug_sample=args.debug_sample,
            split=args.split,
        )
    except ProductionQAError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    if summary.get("blocked_reasons"):
        return 3
    return 0 if summary.get("conclusion") in _SUCCESS_CONCLUSIONS else 3


if __name__ == "__main__":
    raise SystemExit(main())

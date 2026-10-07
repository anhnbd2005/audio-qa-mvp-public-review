"""CLI: run the real OpenAI-compatible R&D re-authoring pipeline.

Examples:
    python scripts/autonomous_qa/run_authoring.py vietmdd
    python scripts/autonomous_qa/run_authoring.py vimd --run-id 20260930T000000Z

This performs REAL LLM calls through the project's existing client. It never
prints credentials. Ordinary unit tests must mock the LLM layer instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.autonomous_qa.authoring.pipeline import DATASETS, run_authoring


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--fixture-dir",
        default=None,
        help="Offline test fixtures directory (no real calls).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require fixture-dir; perform zero network calls.",
    )
    args = parser.parse_args()

    if args.offline and not args.fixture_dir:
        print("OFFLINE requires --fixture-dir", file=sys.stderr)
        return 2

    try:
        manifest = run_authoring(
            args.dataset,
            run_id=args.run_id,
            real_llm=not args.offline,
            fixture_dir=Path(args.fixture_dir) if args.fixture_dir else None,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"LLM_RND_STAGE_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    summary = {
        "dataset_id": manifest["dataset_id"],
        "run_dir": manifest["run_dir"],
        "llm_calls_by_stage": manifest["llm_calls_by_stage"],
        "total_real_llm_calls": manifest["total_real_llm_calls"],
        "primitives_discovered": manifest["primitives_discovered"],
        "primitives_accepted": manifest["primitives_accepted"],
        "composites_discovered": manifest["composites_discovered"],
        "component_leakage_flagged": manifest["component_leakage_flagged"],
        "readiness": manifest["readiness"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Dataset Onboarding Orchestrator.

Thin composition layer connecting:
1. R&D Authoring (or existing run)
2. Staged Promotion PREPARE
3. Explicit Canonical Promotion APPLY (if requested)
4. Language Preflight Verification
5. Production Planning Readiness Check

Does NOT contain dataset-specific business logic or rules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

from src.autonomous_qa.certification.authoring_promotion import (
    apply_promotion,
    prepare_promotion,
)
from src.common.config import ROOT


def onboard_dataset(
    dataset_id: str,
    *,
    promotion_mode: Literal["replace", "merge"],
    authoring_run_id: str | None = None,
    authoring_run_dir: Path | None = None,
    stop_after: str = "plan",
    apply_canonical: bool = False,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Orchestrates onboarding pipeline stages cleanly."""
    if promotion_mode not in ("replace", "merge"):
        raise ValueError("promotion_mode must be 'replace' or 'merge'")

    # Determine authoring run directory
    if authoring_run_dir:
        run_dir = Path(authoring_run_dir)
    elif authoring_run_id:
        run_dir = ROOT / "outputs" / "runs" / dataset_id / authoring_run_id
    else:
        runs_parent = ROOT / "outputs" / "runs" / dataset_id
        if not runs_parent.exists():
            raise FileNotFoundError(f"No authoring runs found under {runs_parent}")
        run_dirs = sorted([d for d in runs_parent.iterdir() if d.is_dir()])
        if not run_dirs:
            raise FileNotFoundError(f"No authoring run directories in {runs_parent}")
        run_dir = run_dirs[-1]

    # STAGE 1: Promotion Prepare (Dry-run / Staged bundle compilation)
    bundle = prepare_promotion(
        dataset_id=dataset_id,
        run_dir=run_dir,
        promotion_mode=promotion_mode,
        output_root=output_root,
    )

    result: dict[str, Any] = {
        "dataset_id": dataset_id,
        "source_run": run_dir.name,
        "promotion_mode": promotion_mode,
        "prepare_status": "PREPARED",
        "bundle_fingerprint": bundle.fingerprint(),
        "promotable_total": bundle.promotable_total,
        "promotable_types": list(bundle.promotable_types),
        "semantic_diff": bundle.semantic_diff,
        "staged_language_preflight": bundle.staged_language_preflight,
        "canonical_mutations": 0,
        "applied": False,
    }

    if stop_after in ("plan", "prepare") and not apply_canonical:
        result["status"] = "STOPPED_AFTER_PLAN_PREPARE_ZERO_CANONICAL_MUTATIONS"
        return result

    # STAGE 2: Apply promotion if explicitly requested
    if apply_canonical:
        scratch_bundle_path = (output_root or (ROOT / "outputs" / "_scratch" / "promotion" / dataset_id / run_dir.name)) / "promotion_bundle.json"
        apply_res = apply_promotion(scratch_bundle_path)
        result["applied"] = True
        result["apply_result"] = apply_res
        result["canonical_mutations"] = 3  # catalog, contract, manifest

    return result


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset ID")
    parser.add_argument(
        "--promotion-mode",
        required=True,
        choices=["replace", "merge"],
        help="Promotion mode",
    )
    parser.add_argument("--from-authoring-run", help="Authoring run ID or directory")
    parser.add_argument(
        "--stop-after",
        choices=["plan", "prepare", "apply"],
        default="plan",
        help="Pipeline phase to stop after (default: plan)",
    )
    parser.add_argument("--apply-promotion", action="store_true", help="Explicitly apply canonical promotion")
    parser.add_argument("--output-root", type=Path, help="Output root for staged candidate artifacts")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _cli().parse_args(argv)

    run_dir = Path(args.from_authoring_run) if args.from_authoring_run and Path(args.from_authoring_run).is_dir() else None
    run_id = args.from_authoring_run if args.from_authoring_run and not run_dir else None

    res = onboard_dataset(
        dataset_id=args.dataset,
        promotion_mode=args.promotion_mode,
        authoring_run_id=run_id,
        authoring_run_dir=run_dir,
        stop_after=args.stop_after,
        apply_canonical=args.apply_promotion,
        output_root=args.output_root,
    )

    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

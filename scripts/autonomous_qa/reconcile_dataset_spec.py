"""Governed DatasetSpec-hash reconciliation CLI (HUMAN-OPERATED).

Rebinds a dataset's ProductionContract + PromotionManifest to the CURRENT
canonical DatasetSpec hash after a policy-only spec change (e.g. split
authorization). Semantic catalog, language registry, active semantic types and
promotion evidence are preserved; only spec-derived fields change. Dry-run by
default (`--dry-run`); pass `--apply` to write atomically.

    python scripts/autonomous_qa/reconcile_dataset_spec.py --dataset vimedcss --dry-run
    python scripts/autonomous_qa/reconcile_dataset_spec.py --dataset vimedcss --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.autonomous_qa.certification.promotion_gate import (
    PromotionError,
    reconcile_dataset_spec_hash,
)


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="plan only (default)")
    mode.add_argument("--apply", action="store_true", help="write canonical resources")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _cli().parse_args(argv)
    try:
        result = reconcile_dataset_spec_hash(
            args.dataset, write=bool(args.apply)
        )
    except PromotionError as exc:
        print(json.dumps({"result": "BLOCKED", "code": exc.code, "detail": exc.detail}))
        return 2
    print(json.dumps({"result": result["status"], **result}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

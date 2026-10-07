"""Stage 4 (per round): Quality judges question-TYPE novelty.

Input per candidate type: the type + its templates + paraphrases, plus the
approved type few-shots from earlier rounds. Output is ids + reasons only.

Duplicate == same capability / answer-derivation as an approved type.
Equivalent paraphrase wording is NEVER a duplicate type (regression rule).
"""

from __future__ import annotations

import json

from src.autonomous_qa.certification.machine_contract import (
    QUALITY_CONTRACT_PRECEDENCE,
    format_machine_contract,
)
from src.autonomous_qa.core.schemas import QualityRoundOutput
from src.common.io import load_prompt


def build_quality_round_prompt(
    bundles: list[dict],
    approved_fewshots: list[dict],
    round_idx: int,
) -> str:
    base = load_prompt("quality")
    contracts = []
    for b in bundles or []:
        btype = dict(b.get("type", {}))
        if btype.get("id") is None and b.get("type_id"):
            btype["id"] = b.get("type_id")
        contracts.append(format_machine_contract(btype))
    return "\n".join(
        [
            base,
            f"\nROUND: {round_idx}\n",
            "\n--- MACHINE CONTRACTS (authoritative) ---\n"
            + QUALITY_CONTRACT_PRECEDENCE + "\n"
            + "\n".join(contracts),
            "\n--- APPROVED TYPES FROM EARLIER ROUNDS ---\n"
            + json.dumps(approved_fewshots, ensure_ascii=False),
            "\n--- CANDIDATE BUNDLES TO JUDGE (this round only) ---\n"
            + json.dumps(bundles, ensure_ascii=False),
        ]
    )


def run_quality_round(
    client,
    bundles: list[dict],
    approved_fewshots: list[dict],
    round_idx: int,
    real_llm: bool,
    raw_path=None,
    on_attempt=None,
    dataset: str = "vimd",
) -> dict:
    from src.common.io import load_fixture

    if not real_llm:
        return load_fixture(f"loop/round_{round_idx:02d}/quality", dataset=dataset)

    from src.autonomous_qa.authoring.llm_client import complete_structured_stage

    if raw_path is None:
        raise ValueError("raw_path is required for real LLM quality runs")

    prompt = build_quality_round_prompt(bundles, approved_fewshots, round_idx)
    result = complete_structured_stage(
        client,
        stage="quality",
        prompt=prompt,
        response_schema=QualityRoundOutput,
        raw_path=raw_path,
        on_attempt=on_attempt,
    )
    print("Quality round parsed successfully.")
    return result

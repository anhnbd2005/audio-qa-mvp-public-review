"""Stage 3 (per round): Paraphrase current-round templates.

Wording variants only. Same semantics is the GOAL here — Quality must
never mistake equivalent phrasings for duplicate question types.
"""

from __future__ import annotations

import json

from src.common.config import CONFIG
from src.autonomous_qa.core.schemas import ParaphraseRoundOutput
from src.common.io import load_prompt


def build_paraphrase_round_prompt(
    templates: list[dict],
    round_idx: int,
) -> str:
    base = load_prompt("paraphrase")
    per_round = CONFIG["stages"]["paraphrase"]["paraphrases_per_round"]
    return "\n".join(
        [
            base,
            f"\nROUND: {round_idx}\nPARAPHRASES THIS ROUND: {per_round}\n",
            "\n--- CURRENT ROUND TEMPLATES (paraphrase exactly these) ---\n"
            + json.dumps(templates, ensure_ascii=False),
        ]
    )


def run_paraphrase_round(
    client,
    templates: list[dict],
    round_idx: int,
    real_llm: bool,
    raw_path=None,
    on_attempt=None,
    dataset: str = "vimd",
) -> dict:
    from src.common.io import load_fixture

    if not real_llm:
        return load_fixture(f"loop/round_{round_idx:02d}/paraphrase", dataset=dataset)

    from src.autonomous_qa.authoring.llm_client import complete_structured_stage

    if raw_path is None:
        raise ValueError("raw_path is required for real LLM paraphrase runs")

    prompt = build_paraphrase_round_prompt(templates, round_idx)
    return complete_structured_stage(
        client,
        stage="paraphrase",
        prompt=prompt,
        response_schema=ParaphraseRoundOutput,
        raw_path=raw_path,
        on_attempt=on_attempt,
    )

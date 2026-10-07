"""Stage 1 (per round): Question Style — batched type discovery.

Input: dataset README/schema + fixed seed few-shots + dynamic approved
few-shots from previous rounds. Creativity lives here: seeds demonstrate
what a valid proposal looks like, never a taxonomy to stay inside.
"""

from __future__ import annotations

import json

from src.common.config import CONFIG, ROOT
from src.autonomous_qa.core.schemas import StyleRoundOutput
from src.common.io import load_jsonl, load_prompt


def style_output_contract_block() -> str:
    """Machine-generated output contract for Question Style calls.

    The JSON schema is produced by the Pydantic response model itself
    (`StyleRoundOutput.model_json_schema()`), so the model stays the single
    contract source: no hand-copied keys, no duplicated schema text.
    """
    schema = StyleRoundOutput.model_json_schema()
    top_keys = list(schema.get("properties") or {})
    return (
        "\n--- OUTPUT CONTRACT (authoritative, machine-generated) ---\n"
        "Required top-level keys (derived from the response model): "
        + json.dumps(top_keys)
        + "\nJSON Schema generated from the response model:\n"
        + json.dumps(schema, indent=2, ensure_ascii=False)
        + "\nReturn exactly ONE JSON object matching the schema.\n"
        "Do not use Markdown fences.\n"
        "Do not add commentary.\n"
        "Do not add extra top-level keys.\n"
        "Do not rename keys.\n"
    )


def build_question_style_prompt(
    readme: str,
    dynamic_fewshots: list[dict],
    round_idx: int,
    metadata_context: dict | None = None,
    knowledge_context: dict | None = None,
    sample_rows: list[dict] | None = None,
    max_sample_rows: int | None = 5,
    include_contract: bool = True,
) -> str:
    base = load_prompt("question_style")
    seeds = load_jsonl(ROOT / "fewshots" / "question_style_seed.jsonl")
    approved_bank = load_jsonl(ROOT / "fewshots" / "approved_types.jsonl")
    per_round = CONFIG["stages"]["question_style"]["candidates_per_round"]

    parts = [
        "\n--- DATASET README / SCHEMA (only source of truth) ---\n" + readme,
    ]
    if metadata_context:
        parts.append(
            "\n--- METADATA CONTEXT ---\n"
            + json.dumps(metadata_context, indent=2, ensure_ascii=False)
        )
    if knowledge_context:
        parts.append(
            "\n--- VIETNAMESE LINGUISTIC KNOWLEDGE CONTEXT (available evidence & limitations) ---\n"
            + json.dumps(knowledge_context, indent=2, ensure_ascii=False)
        )
    if sample_rows:
        shown = (sample_rows if max_sample_rows is None
                 else sample_rows[:max_sample_rows])
        parts.append(
            "\n--- DATASET SAMPLE ROWS ---\n"
            + "\n".join(json.dumps(r, ensure_ascii=False) for r in shown)
        )
    parts.extend([
        f"\nROUND: {round_idx}\nCANDIDATES THIS ROUND: {per_round}\n",
        base,
        "\n--- FIXED SEED FEW-SHOTS (proposal shape only, NOT a taxonomy) ---\n"
        + "\n".join(json.dumps(s, ensure_ascii=False) for s in seeds),
    ])
    if approved_bank:
        parts.append(
            "\n--- HUMAN-APPROVED TYPES FROM EARLIER DATASETS ---\n"
            + "\n".join(json.dumps(a, ensure_ascii=False) for a in approved_bank)
        )
    if dynamic_fewshots:
        parts.append(
            "\n--- ALREADY COVERED BY THIS RUN (do not propose again) ---\n"
            + "\n".join(
                json.dumps(s, ensure_ascii=False) for s in dynamic_fewshots)
        )
    if include_contract:
        # Contract block is ALWAYS the final instruction; saturation runs
        # (include_contract=False) re-append it after known/novelty blocks.
        parts.append(style_output_contract_block())
    return "\n".join(parts)


def run_question_style_round(
    client,
    readme: str,
    dynamic_fewshots: list[dict],
    round_idx: int,
    real_llm: bool,
    raw_path=None,
    on_attempt=None,
    dataset: str = "vimd",
    metadata_context: dict | None = None,
    knowledge_context: dict | None = None,
    sample_rows: list[dict] | None = None,
) -> dict:
    from src.common.io import load_fixture

    if not real_llm:
        return load_fixture(f"loop/round_{round_idx:02d}/style", dataset=dataset)

    from src.autonomous_qa.authoring.llm_client import complete_structured_stage

    if raw_path is None:
        raise ValueError("raw_path is required for real LLM question_style runs")

    prompt = build_question_style_prompt(
        readme,
        dynamic_fewshots,
        round_idx,
        metadata_context=metadata_context,
        knowledge_context=knowledge_context,
        sample_rows=sample_rows,
    )
    return complete_structured_stage(
        client,
        stage="question_style",
        prompt=prompt,
        response_schema=StyleRoundOutput,
        raw_path=raw_path,
        on_attempt=on_attempt,
    )

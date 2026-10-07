"""Structured-response tests: parsed preferred, truncation fails loudly."""

import json
from types import SimpleNamespace

import src.autonomous_qa.authoring.llm_client as llm_client
from src.autonomous_qa.authoring.llm_client import (
    StageOutputValidationError,
    complete_structured_stage,
    parse_structured_response,
)
from src.autonomous_qa.core.schemas import QualityRoundOutput


def _response(text, parsed=None):
    return SimpleNamespace(
        text=text,
        parsed=parsed,
        candidates=[SimpleNamespace(finish_reason="STOP")],
        usage_metadata=None,
    )


def test_parsed_response_used_without_text_parse():
    payload = {
        "round": 1,
        "accepted_new": [{"type_id": "QS_R1_01",
                          "keep_template_ids": ["T_R1_01_01"]}],
        "duplicates": [],
        "rejected": [],
    }
    response = _response(
        text="not json at all {{{",
        parsed=QualityRoundOutput(**payload),
    )
    assert parse_structured_response(response, "quality") == payload


def test_truncated_output_raises_without_retry(tmp_path):
    truncated = '{"accepted_new":[{"type_id":"QS_R1'  # cut mid-string
    response = _response(text=truncated, parsed=None)

    calls = []

    def fake_generate_content(**kwargs):
        calls.append(1)
        return response

    fake_client = SimpleNamespace(
        models=SimpleNamespace(generate_content=fake_generate_content))

    saved = (llm_client.BUDGET.count, dict(llm_client.BUDGET.stage_counts),
             llm_client.BUDGET.display_total)
    try:
        try:
            complete_structured_stage(
                fake_client, stage="quality", prompt="p",
                response_schema=QualityRoundOutput,
                raw_path=tmp_path / "raw" / "quality.txt",
            )
        except StageOutputValidationError as exc:
            assert "No automatic retry" in str(exc)
        else:
            raise AssertionError("expected StageOutputValidationError")
    finally:
        llm_client.BUDGET.count = saved[0]
        llm_client.BUDGET.stage_counts.clear()
        llm_client.BUDGET.stage_counts.update(saved[1])
        llm_client.BUDGET.display_total = saved[2]

    assert len(calls) == 1  # exactly one request, no retry
    raw = (tmp_path / "raw" / "quality_attempt_01.txt").read_text(
        encoding="utf-8")
    assert raw == truncated  # raw saved before parsing
    assert json.loads(json.dumps(raw)) == truncated

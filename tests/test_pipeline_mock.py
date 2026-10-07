"""Mock pipeline: full discovery loop, 0 real API calls (isolated root)."""

import json
import re

from src.autonomous_qa.authoring.llm_client import BUDGET
from src.autonomous_qa.compiler.pipeline_runner import run_pipeline


def test_mock_pipeline_converges_and_previews(isolated_root):
    before = (BUDGET.count, dict(BUDGET.stage_counts))
    preview = run_pipeline(dataset="vimd", real_llm=False)

    assert BUDGET.count == before[0]  # mock mode charges nothing
    assert BUDGET.stage_counts == before[1]
    assert len(preview) == 10

    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "mock"
    summary = json.loads((out_dir / "loop_summary.json").read_text(encoding="utf-8"))
    assert summary["rounds_run"] == 3
    assert summary["stop_reason"] == "no_novel_types_after_signature_dedup"
    rounds = summary["rounds"]
    assert [r["generated_types"] for r in rounds] == [8, 8, 8]
    assert [r["accepted_new_types"] for r in rounds] == [5, 2, 0]
    assert [r["new_type_rate"] for r in rounds] == [0.625, 0.25, 0.0]
    assert [r["total_approved_types"] for r in rounds] == [5, 7, 7]
    assert [r["deterministic_duplicates"] for r in rounds] == [0, 6, 8]

    final_types = json.loads(
        (out_dir / "final_approved_types.json").read_text(encoding="utf-8"))
    assert final_types["total_approved_types"] == 7

    for r in (1, 2):
        round_dir = out_dir / "rounds" / f"round_{r:02d}"
        for name in ("01_question_styles.json", "02_templates.json",
                     "03_paraphrases.json", "04_quality.json",
                     "approved_types_after.json"):
            assert (round_dir / name).is_file(), name
    # Terminal dedup round: Gate artifacts + audit only, no P/Q stages.
    round_dir = out_dir / "rounds" / "round_03"
    for name in ("01_question_styles.json", "02_templates.json",
                 "signature_dedup.json"):
        assert (round_dir / name).is_file(), name
    assert not (round_dir / "03_paraphrases.json").exists()
    assert not (round_dir / "04_quality.json").exists()

    assert (out_dir / "final_template_pool.json").is_file()
    assert (out_dir / "preview.jsonl").is_file()

    # Every approved type is covered at least once (quota allows).
    covered = {item["question_type_id"] for item in preview}
    approved = {t["type_id"] for t in final_types["approved_types"]}
    assert covered == approved

    # No unresolved placeholders anywhere in the preview.
    for item in preview:
        assert not re.search(r"\[[A-Za-z_][A-Za-z0-9_]*\]", item["question"])
        assert item["answer_source"]["kind"] in (
            "field_value", "equality", "derived_field")

    # Gold answers come straight from sample rows (no second mapping).
    rows = {}
    for line in (isolated_root / "data" / "vimd" / "sample.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["audio"]] = row
    for item in preview:
        src = item.get("answer_source", {})
        if src.get("kind") == "field_value" and src.get("field") == "gender":
            assert item["answer"] == rows[item["audio"]]["gender"]

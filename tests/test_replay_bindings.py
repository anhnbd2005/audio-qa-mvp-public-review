"""Replay reconstructs the accepted-only binding map (§54).

Hand-built single-round run dir:
  candidate mappings = gender + province
  Quality: gender type accepted with [KEY] kept; province type rejected.
Expected approved mapping after replay: gender only (no province leakage).
Resume then continues with exactly the same mapping.
"""

import json

from src.autonomous_qa.compiler.pipeline_runner import TEMPLATE_CONTRACT_VERSION, _ResumedRun
from tests.test_safety import clean_budget  # noqa: F401


def _write(out_dir, isolated_root):
    fx = isolated_root / "tests" / "fixtures" / "loop"
    raw_dir = out_dir / "raw"
    (out_dir / "rounds" / "round_01").mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    for kind, name in (("style", "01_question_styles.json"),
                       ("template", "02_templates.json"),
                       ("paraphrase", "03_paraphrases.json")):
        (out_dir / "rounds" / "round_01" / name).write_bytes(
            (fx / "round_01" / f"{kind}.json").read_bytes())
    # Durable paid-attempt evidence (1 per stage) so resume inventory
    # matches the checkpoint (D == C).
    (raw_dir / "round_01_style_attempt_01.txt").write_text("{}", encoding="utf-8")
    (raw_dir / "round_01_template_attempt_01.txt").write_text("{}", encoding="utf-8")
    (raw_dir / "paraphrase_round_1_attempt_01.txt").write_text("{}", encoding="utf-8")
    (raw_dir / "quality_round_1_attempt_01.txt").write_text("{}", encoding="utf-8")
    # Gender accepted with [KEY] kept; province rejected outright.
    (out_dir / "rounds" / "round_01" / "04_quality.json").write_text(
        json.dumps({
            "round": 1,
            "accepted_new": [{"type_id": "QS_R1_01",
                              "keep_template_ids": ["T_R1_01_01"]}],
            "duplicates": [],
            "rejected": [{"type_id": "QS_R1_02",
                          "reason": "hand-built replay fixture"},
                         {"type_id": "QS_R1_03",
                          "reason": "hand-built replay fixture"},
                         {"type_id": "QS_R1_04",
                          "reason": "hand-built replay fixture"},
                         {"type_id": "QS_R1_05",
                          "reason": "hand-built replay fixture"}],
            "auto_rejected": [
                {"type_id": "QS_R1_06",
                 "reason": "unknown_uses_field:duration"},
                {"type_id": "QS_R1_07",
                 "reason": "unknown_uses_field:age"},
                {"type_id": "QS_R1_08",
                 "reason": "missing_audio_input"}],
            "dropped_templates": [],
        }), encoding="utf-8")
    (out_dir / "run_state.json").write_text(json.dumps({
        "run_id": "replay-kr", "dataset": "vimd", "mode": "real",
        "completed": ["style_round_1", "template_round_1",
                      "paraphrase_round_1", "quality_round_1"],
        "current_round": 1, "next_action": "style_round_2",
        "low_gain_streak": 1, "stopped": False, "stop_reason": None,
        "spent_calls": {"question_style": 1, "question_template": 1,
                        "paraphrase": 1, "quality": 1},
        "loop_config": {
            "max_rounds": 3, "min_new_type_rate": 0.15,
            "saturation_patience": 2, "immediate_stop_if_zero_new": True},
        "template_contract_version": TEMPLATE_CONTRACT_VERSION,
        "answer_reference_contract_version": "kind-scoped-v1",
        "answer_shape_contract_version": "canonical-v1",
        "executable_signature_contract_version": "v1",
        "derived_field_contract_version": "data-backed-v1",
    }), encoding="utf-8")


def test_replay_reconstructs_accepted_only_mapping(
        isolated_root, clean_budget):
    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "replay-kr"
    _write(out_dir, isolated_root)
    from src.autonomous_qa.compiler.pipeline_runner import TEMPLATE_CONTRACT_VERSION, load_sample
    run = _ResumedRun(dataset="vimd", out_dir=out_dir,
                      raw_dir=out_dir / "raw",
                      rounds_dir=out_dir / "rounds", readme="",
                      rows=load_sample("vimd"), client=None)
    run.load_and_validate()
    # Gender committed (accepted + kept [KEY]); province rejected upstream.
    assert run.approved_key_realizations == {"gender": "giới tính"}
    assert run._next_action == "style_round_2"
    # Round-2 candidate for province must NOT resurrect anything.
    assert "province" not in run.approved_key_realizations

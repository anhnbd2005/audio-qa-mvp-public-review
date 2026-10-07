"""Comprehensive test suite for data-backed derived_field semantics (§69 - §94).

Verifies:
- Data-backed derived relation validation (functional, coarsening, anti-vacuity, hidden ID)
- Machine contract output mode and prompt inclusion with precedence
- Key binding parity (target_key across gate, candidates, commit, replay, render)
- Gold derivation, mapping lookup, and parity error handling
- Quality verdict semantics for derived types (reasoning kept, binary/direct omitted)
- Signature dedup happening strictly after relation validation
- Four-stage pipeline unchanged, no candidate DSL or new answer kinds
"""

import json
from pathlib import Path
import pytest

from src.autonomous_qa.certification.machine_contract import (
    MACHINE_CONTRACT_PRECEDENCE,
    QUALITY_CONTRACT_PRECEDENCE,
    describe_machine_contract,
    format_machine_contract,
)
from src.autonomous_qa.language.question_template import build_question_template_prompt
from src.autonomous_qa.certification.quality import build_quality_round_prompt
from src.autonomous_qa.production.render_preview import _gold_answer, render_question
from src.autonomous_qa.core.validity import (
    ANSWER_KINDS,
    ALLOWED_PLACEHOLDERS,
    SCHEMA_FIELDS,
    binding_key_for,
    build_approved_signature_index,
    build_round_field_candidates,
    deduplicate_by_signature,
    executable_signature,
    gate_round,
    get_answer_concept_field,
    validate_answer_references,
    validate_answer_shape,
    validate_derived_relation,
    validate_question_type,
    validate_template_text,
)


# ============================================================
# §69: VALID DATA-BACKED RELATION
# ============================================================
def test_69_valid_data_backed_relation():
    # Synthetic generic rows with repeated source values and fine-to-coarse mapping
    rows = [
        {"source_col": "A", "target_col": "North"},
        {"source_col": "A", "target_col": "North"},
        {"source_col": "B", "target_col": "North"},
        {"source_col": "B", "target_col": "North"},
        {"source_col": "C", "target_col": "South"},
        {"source_col": "C", "target_col": "South"},
    ]
    schema = {"audio", "source_col", "target_col"}
    res = validate_derived_relation("source_col", "target_col", rows, schema_fields=schema)
    assert res["valid"] is True
    assert res["reason"] is None
    assert res["mapping"] == {"A": "North", "B": "North", "C": "South"}
    assert res["distinct_source_count"] == 3
    assert res["distinct_target_count"] == 2
    assert res["mapping_conflicts"] == 0
    assert res["reusable_source_values"] == 3


# ============================================================
# §70: NONFUNCTIONAL RELATION
# ============================================================
def test_70_nonfunctional_relation():
    rows = [
        {"province": "A", "region": "North"},
        {"province": "A", "region": "South"},
        {"province": "B", "region": "North"},
        {"province": "B", "region": "North"},
    ]
    res = validate_derived_relation("province", "region", rows)
    assert res["valid"] is False
    assert res["reason"] == "invalid_derived_nonfunctional_relation"
    assert res["mapping_conflicts"] > 0


# ============================================================
# §71: ROW-UNIQUE / VACUOUS SOURCE
# ============================================================
def test_71_row_unique_vacuous_source():
    rows = [
        {"text": "t1", "gender": "female"},
        {"text": "t2", "gender": "male"},
        {"text": "t3", "gender": "female"},
        {"text": "t4", "gender": "male"},
    ]
    res = validate_derived_relation("text", "gender", rows)
    assert res["valid"] is False
    assert res["reason"] == "invalid_derived_nonreusable_source"


# ============================================================
# §72: NOT COARSENING
# ============================================================
def test_72_not_coarsening():
    # 2 sources map 1:1 to 2 targets: |target| == |source|
    rows = [
        {"source": "A", "target": "X"},
        {"source": "A", "target": "X"},
        {"source": "B", "target": "Y"},
        {"source": "B", "target": "Y"},
    ]
    schema = {"audio", "source", "target"}
    res = validate_derived_relation("source", "target", rows, schema_fields=schema)
    assert res["valid"] is False
    assert res["reason"] == "invalid_derived_not_coarsening"


# ============================================================
# §73: HIDDEN IDENTIFIER SOURCE
# ============================================================
def test_73_hidden_identifier_source():
    rows = [
        {"speakerID": "spk1", "gender": "female"},
        {"speakerID": "spk1", "gender": "female"},
        {"speakerID": "spk2", "gender": "male"},
        {"speakerID": "spk2", "gender": "male"},
    ]
    res = validate_derived_relation("speakerID", "gender", rows)
    assert res["valid"] is False
    assert res["reason"] == "invalid_derived_hidden_source"


# ============================================================
# §74: SAME FIELD
# ============================================================
def test_74_same_field():
    res = validate_derived_relation("region", "region", [{"region": "North"}])
    assert res["valid"] is False
    assert res["reason"] == "invalid_derived_same_field"


# ============================================================
# §75: DATA UNAVAILABLE FAIL-CLOSED
# ============================================================
def test_75_data_unavailable():
    q = {
        "id": "QS_D",
        "name": "Derived Test",
        "uses": ["audio", "province", "region"],
        "goal": "Test",
        "answer_rule": "derive",
        "answer": {"kind": "derived_field", "source_key": "province", "target_key": "region"},
    }
    errs = validate_question_type(q, dataset_rows=None)
    assert "derived_relation_data_unavailable" in errs


# ============================================================
# §76: TARGET BINDING PARITY
# ============================================================
def test_76_target_binding():
    q = {
        "id": "QS_D1",
        "name": "Province to Region",
        "uses": ["audio", "province", "region"],
        "goal": "Infer region from province",
        "answer_rule": "derived",
        "answer": {"kind": "derived_field", "source_key": "province", "target_key": "region"},
    }
    # Invariant: get_answer_concept_field == binding_key_for == target_key
    assert get_answer_concept_field(q) == "region"
    assert binding_key_for(q) == "region"

    # round_field_candidates uses target_key
    cands = build_round_field_candidates(
        types_in_style_order=[q],
        valid_templates=[{"template_id": "T1", "question_type_id": "QS_D1", "text": "Hỏi [KEY]"}],
        valid_type_ids={"QS_D1"},
        type_key_realizations={"QS_D1": "vùng phương ngữ"},
        approved_key_realizations={},
    )
    assert cands == {"region": "vùng phương ngữ"}
    assert "province" not in cands


# ============================================================
# §77: REPLAY TARGET BINDING PARITY
# ============================================================
def test_77_replay_target_binding():
    # Approved binding commit preserves target_key
    q = {
        "id": "QS_D1",
        "name": "Province to Region",
        "uses": ["audio", "province", "region"],
        "goal": "Infer region from province",
        "answer_rule": "derived",
        "answer": {"kind": "derived_field", "source_key": "province", "target_key": "region"},
    }
    cands = build_round_field_candidates(
        types_in_style_order=[q],
        valid_templates=[{"template_id": "T1", "question_type_id": "QS_D1", "text": "Hỏi [KEY]"}],
        valid_type_ids={"QS_D1"},
        type_key_realizations={"QS_D1": "vùng phương ngữ"},
        approved_key_realizations={},
    )
    assert "region" in cands
    # Replay with approved binding
    approved = dict(cands)
    cands_round2 = build_round_field_candidates(
        types_in_style_order=[q],
        valid_templates=[{"template_id": "T1", "question_type_id": "QS_D1", "text": "Hỏi [KEY]"}],
        valid_type_ids={"QS_D1"},
        type_key_realizations={"QS_D1": "vùng mới"},
        approved_key_realizations=approved,
    )
    assert cands_round2 == {}  # Already approved, not overwritten
    assert approved["region"] == "vùng phương ngữ"


# ============================================================
# §78: MACHINE CONTRACT OUTPUT MODE
# ============================================================
def test_78_machine_contract_output_mode():
    fv = describe_machine_contract({"answer": {"kind": "field_value", "key": "province"}})
    eq = describe_machine_contract({"answer": {"kind": "equality", "keys": ["speakerID_1", "speakerID_2"]}})
    df = describe_machine_contract({"answer": {"kind": "derived_field", "source_key": "province", "target_key": "region"}})

    assert fv["answer_mode"] == "value"
    assert eq["answer_mode"] == "binary"
    assert df["answer_mode"] == "value"


# ============================================================
# §79: DERIVED GOLD CALCULATION
# ============================================================
def test_79_derived_gold():
    q = {
        "id": "QS_D",
        "name": "Derived Test",
        "uses": ["audio", "province", "region"],
        "answer": {"kind": "derived_field", "source_key": "province", "target_key": "region"},
    }
    row = {"audio": "a.wav", "province": "CaoBang", "region": "North"}
    mapping = {("province", "region"): {"CaoBang": "North"}}
    gold, provenance = _gold_answer(q, row, None, [row], derived_mappings=mapping)
    assert gold == "North"
    assert gold != "Có"
    assert gold != "Không"
    assert provenance["kind"] == "derived_field"
    assert provenance["target_field"] == "region"


# ============================================================
# §80: DERIVED MAPPING PARITY
# ============================================================
def test_80_derived_mapping_parity():
    q = {
        "id": "QS_D",
        "name": "Derived Test",
        "uses": ["audio", "province", "region"],
        "answer": {"kind": "derived_field", "source_key": "province", "target_key": "region"},
    }
    # Inconsistent row: province says CaoBang, but region says South while map says North
    row = {"audio": "a.wav", "province": "CaoBang", "region": "South"}
    mapping = {("province", "region"): {"CaoBang": "North"}}
    with pytest.raises(ValueError) as excinfo:
        _gold_answer(q, row, None, [row], derived_mappings=mapping)
    assert "render_derived_target_mismatch" in str(excinfo.value)


# ============================================================
# §81: TEMPLATE MACHINE CONTRACT PRESENT
# ============================================================
def test_81_template_machine_contract_present():
    q = {
        "id": "QS_R2_07",
        "name": "Province to Region",
        "goal": "Noisy goal prose that might say check candidate",
        "uses": ["audio", "province", "region"],
        "input_count": 1,
        "answer_rule": "noisy prose",
        "answer": {"kind": "derived_field", "source_key": "province", "target_key": "region"},
    }
    prompt = build_question_template_prompt(
        readme="readme content",
        round_types=[q],
        template_bank=[],
        round_idx=2,
    )
    assert "--- MACHINE CONTRACTS (authoritative) ---" in prompt
    assert MACHINE_CONTRACT_PRECEDENCE in prompt
    assert "source_field: province" in prompt
    assert "target_field: region" in prompt
    assert "answer_mode: value" in prompt


# ============================================================
# §82: QUALITY MACHINE CONTRACT PRESENT
# ============================================================
def test_82_quality_machine_contract_present():
    bundle = {
        "type_id": "QS_R2_07",
        "type": {
            "id": "QS_R2_07",
            "name": "Province to Region",
            "goal": "Infer region from province",
            "uses": ["audio", "province", "region"],
            "answer": {"kind": "derived_field", "source_key": "province", "target_key": "region"},
        },
        "templates": [],
        "paraphrases": [],
    }
    prompt = build_quality_round_prompt([bundle], [], 2)
    assert "--- MACHINE CONTRACTS (authoritative) ---" in prompt
    assert QUALITY_CONTRACT_PRECEDENCE in prompt
    assert "source_field: province" in prompt
    assert "target_field: region" in prompt
    assert "answer_mode: value" in prompt


# ============================================================
# §83, §84, §85, §86: QUALITY VERDICT SEMANTICS FOR DERIVED
# ============================================================
def test_83_84_85_86_quality_template_rules():
    from src.autonomous_qa.core.loop import apply_accepted_types

    q = {
        "id": "QS_R2_07",
        "name": "Province to Region",
        "goal": "Infer region from province",
        "uses": ["audio", "province", "region"],
        "input_count": 1,
        "answer_rule": "derive region",
        "answer": {"kind": "derived_field", "source_key": "province", "target_key": "region"},
    }
    by_id = {"QS_R2_07": q}

    # §85: valid reasoning template
    t_valid = "Dựa trên phương ngữ tỉnh/thành nhận biết được từ đoạn âm thanh, hãy xác định [KEY] của người nói."
    # §83: candidate yes/no template
    t_binary = "Phương ngữ tỉnh/thành có thuộc [KEY] đã cho không?"
    # §84: direct recognition template
    t_direct = "Người nói thuộc [KEY] nào?"

    texts = {
        "T_GOOD": t_valid,
        "T_BIN": t_binary,
        "T_DIR": t_direct,
    }
    owners = {"T_GOOD": "QS_R2_07", "T_BIN": "QS_R2_07", "T_DIR": "QS_R2_07"}

    # Case 1: Quality correctly keeps only T_GOOD (§85) and omits T_BIN (§83) and T_DIR (§84)
    verdict_good = {
        "accepted_new": [{"type_id": "QS_R2_07", "keep_template_ids": ["T_GOOD"]}],
        "duplicates": [],
        "rejected": [],
    }
    pool = []
    accepted = apply_accepted_types(pool, by_id, texts, verdict_good, 2, owners)
    assert len(accepted) == 1
    assert [k["template_id"] for k in accepted[0]["kept_templates"]] == ["T_GOOD"]

    # Case 2 (§86): All templates bad -> Quality rejects type, not accepted_new with empty list
    verdict_all_bad = {
        "accepted_new": [],
        "duplicates": [],
        "rejected": [{"type_id": "QS_R2_07", "reason": "No valid reasoning templates survived"}],
    }
    pool2 = []
    accepted2 = apply_accepted_types(pool2, by_id, texts, verdict_all_bad, 2, owners)
    assert len(accepted2) == 0
    assert len(pool2) == 0


# ============================================================
# §87 & §88: SIGNATURE DEDUP AFTER RELATION VALIDATION
# ============================================================
def test_87_88_signature_dedup_after_relation_validation():
    rows = [
        {"audio": "a.wav", "province": "A", "region": "North"},
        {"audio": "b.wav", "province": "A", "region": "North"},
        {"audio": "c.wav", "province": "B", "region": "North"},
        {"audio": "d.wav", "province": "B", "region": "North"},
        {"audio": "e.wav", "province": "C", "region": "South"},
        {"audio": "f.wav", "province": "C", "region": "South"},
    ]
    # R2 accepted type
    r2_type = {
        "id": "QS_R2_07",
        "name": "Province Dialect to Region",
        "uses": ["audio", "province", "region"],
        "answer_rule": "derive region",
        "answer": {"kind": "derived_field", "source_key": "province", "target_key": "region"},
    }
    index = build_approved_signature_index([{"type_id": "QS_R2_07", "answer": r2_type["answer"]}])

    # R3 proposes same valid relation under a different name
    r3_type = {
        "id": "QS_R3_07",
        "name": "Regional Accent Inference",
        "uses": ["audio", "province", "region"],
        "answer_rule": "derive region",
        "answer": {"kind": "derived_field", "source_key": "province", "target_key": "region"},
    }

    # §87: Passes gate, then signature dedup catches it as deterministic duplicate
    gate = gate_round(
        [r3_type],
        [{"template_id": "T1", "question_type_id": "QS_R3_07", "text": "Hỏi [KEY]"}],
        dataset_rows=rows,
    )
    assert len(gate["valid_types"]) == 1
    novel, det_dups = deduplicate_by_signature(gate["valid_types"], index)
    assert len(novel) == 0
    assert len(det_dups) == 1
    assert det_dups[0]["type_id"] == "QS_R3_07"
    assert det_dups[0]["duplicate_of"] == "QS_R2_07"

    # §88: Invalid relation (text -> gender) is auto-rejected at gate, NEVER reaches signature dedup
    invalid_type = {
        "id": "QS_R3_08",
        "name": "Vacuous Text to Gender",
        "uses": ["audio", "text", "gender"],
        "answer_rule": "derive gender",
        "answer": {"kind": "derived_field", "source_key": "text", "target_key": "gender"},
    }
    gate_invalid = gate_round(
        [invalid_type],
        [{"template_id": "T2", "question_type_id": "QS_R3_08", "text": "Hỏi [KEY]"}],
        dataset_rows=rows,
    )
    assert len(gate_invalid["valid_types"]) == 0
    assert any(rej["type_id"] == "QS_R3_08" for rej in gate_invalid["auto_rejected"])
    # Not passed to deduplicate_by_signature
    novel_inv, dups_inv = deduplicate_by_signature(gate_invalid["valid_types"], index)
    assert len(novel_inv) == 0
    assert len(dups_inv) == 0


# ============================================================
# §89: FOUR-STAGE PIPELINE UNCHANGED
# ============================================================
def test_89_four_stage_pipeline_unchanged():
    from src.common.config import CONFIG
    # Exactly four top-level LLM stages
    expected_stages = {"question_style", "question_template", "paraphrase", "quality"}
    assert set(CONFIG["stages"].keys()) == expected_stages


# ============================================================
# §90: STYLE DISCOVERY STILL DATA-DRIVEN (NO MANUAL RELATION LIST)
# ============================================================
def test_90_style_discovery_data_driven():
    # An arbitrary field pair not in ViMD
    schema = {"audio", "acoustic_cluster", "language_family"}
    rows = [
        {"acoustic_cluster": "C1", "language_family": "FamA"},
        {"acoustic_cluster": "C1", "language_family": "FamA"},
        {"acoustic_cluster": "C2", "language_family": "FamA"},
        {"acoustic_cluster": "C2", "language_family": "FamA"},
        {"acoustic_cluster": "C3", "language_family": "FamB"},
        {"acoustic_cluster": "C3", "language_family": "FamB"},
    ]
    # No hard-coded list knows acoustic_cluster -> language_family, yet it passes
    res = validate_derived_relation(
        "acoustic_cluster", "language_family", rows, schema_fields=schema
    )
    assert res["valid"] is True
    assert res["mapping"]["C1"] == "FamA"


# ============================================================
# §91: DYNAMIC FEW-SHOT LOOP ACCEPTS ONLY APPROVED
# ============================================================
def test_91_dynamic_fewshot_loop():
    from src.autonomous_qa.core.loop import build_type_fewshot
    pool = [
        {
            "type_id": "QS_R1_05",
            "name": "Province to Region",
            "uses": ["audio", "province", "region"],
            "goal": "Infer region",
            "answer_rule": "derive",
            "answer": {"kind": "derived_field", "source_key": "province", "target_key": "region"},
        }
    ]
    fewshots = build_type_fewshot(pool)
    assert len(fewshots) == 1
    assert fewshots[0]["name"] == "Province to Region"
    assert "province" in fewshots[0]["uses"]


# ============================================================
# §92: NO [VALUE] IN QUESTION TEMPLATES
# ============================================================
def test_92_no_value_in_question():
    assert "VALUE" not in ALLOWED_PLACEHOLDERS
    errs = validate_template_text("Hỏi [KEY] là [VALUE]?")
    assert "answer_leakage" in errs


# ============================================================
# §93: NO CANDIDATE DSL ADDED
# ============================================================
def test_93_no_candidate_dsl():
    # Only 3 canonical answer kinds remain
    assert ANSWER_KINDS == {"field_value", "equality", "derived_field"}
    forbidden_kinds = {
        "candidate_match", "boolean_check", "multi_field", "and", "or",
        "inequality", "pairwise_derived", "tuple", "chain", "list"
    }
    assert forbidden_kinds.isdisjoint(ANSWER_KINDS)
    forbidden_fields = {"candidate_region", "candidate_gender", "candidate_text"}
    assert forbidden_fields.isdisjoint(SCHEMA_FIELDS)


# ============================================================
# §94: MOCK EXPECTED BEHAVIOR
# ============================================================
def test_94_mock_expected_behavior():
    from src.autonomous_qa.core.loop import apply_accepted_types

    dataset_rows = [
        {"audio": "1.wav", "province": "A", "region": "North", "text": "t1", "gender": "female"},
        {"audio": "2.wav", "province": "A", "region": "North", "text": "t2", "gender": "male"},
        {"audio": "3.wav", "province": "B", "region": "North", "text": "t3", "gender": "female"},
        {"audio": "4.wav", "province": "B", "region": "North", "text": "t4", "gender": "male"},
        {"audio": "5.wav", "province": "C", "region": "South", "text": "t5", "gender": "female"},
        {"audio": "6.wav", "province": "C", "region": "South", "text": "t6", "gender": "male"},
    ]

    # A: valid derived relation proposal: province -> region backed by repeated many-to-one data
    type_valid = {
        "id": "QS_V1",
        "name": "Province to Region Reasoning",
        "uses": ["audio", "province", "region"],
        "answer_rule": "derive region",
        "answer": {"kind": "derived_field", "source_key": "province", "target_key": "region"},
    }
    # B: vacuous relation: text -> gender
    type_vacuous = {
        "id": "QS_V2",
        "name": "Vacuous Text to Gender",
        "uses": ["audio", "text", "gender"],
        "answer_rule": "derive gender",
        "answer": {"kind": "derived_field", "source_key": "text", "target_key": "gender"},
    }

    tpl_reasoning = {"template_id": "T_REAS", "question_type_id": "QS_V1", "text": "Dựa trên phương ngữ tỉnh/thành, hãy xác định [KEY]."}
    tpl_binary = {"template_id": "T_BIN", "question_type_id": "QS_V1", "text": "Phương ngữ tỉnh/thành có thuộc [KEY] đã cho không?"}
    tpl_vac = {"template_id": "T_VAC", "question_type_id": "QS_V2", "text": "Hỏi [KEY]"}

    # Gate test: valid passes (A), vacuous fails (B)
    gate = gate_round(
        [type_valid, type_vacuous],
        [tpl_reasoning, tpl_binary, tpl_vac],
        type_key_realizations={"QS_V1": "vùng phương ngữ", "QS_V2": "giới tính"},
        dataset_rows=dataset_rows,
    )
    valid_ids = {t["id"] for t in gate["valid_types"]}
    assert "QS_V1" in valid_ids  # A: passes gate
    assert "QS_V2" not in valid_ids  # B: auto-rejected
    assert any(rej["type_id"] == "QS_V2" for rej in gate["auto_rejected"])

    # Quality test: D (binary template not kept), E (proper reasoning template kept)
    verdict = {
        "accepted_new": [{"type_id": "QS_V1", "keep_template_ids": ["T_REAS"]}],  # D omitted, E kept
        "duplicates": [],
        "rejected": [],
    }
    texts = {"T_REAS": tpl_reasoning["text"], "T_BIN": tpl_binary["text"]}
    owners = {"T_REAS": "QS_V1", "T_BIN": "QS_V1"}
    pool = []
    accepted = apply_accepted_types(pool, {"QS_V1": type_valid}, texts, verdict, 1, owners)
    assert len(accepted) == 1
    assert [k["template_id"] for k in accepted[0]["kept_templates"]] == ["T_REAS"]

    # C: same valid derived relation next round under new name -> signature duplicate
    type_r2 = {
        "id": "QS_V3",
        "name": "Regional Accent Inference Round 2",
        "uses": ["audio", "province", "region"],
        "answer_rule": "derive region",
        "answer": {"kind": "derived_field", "source_key": "province", "target_key": "region"},
    }
    index = build_approved_signature_index([{"type_id": "QS_V1", "answer": type_valid["answer"]}])
    gate_r2 = gate_round(
        [type_r2],
        [{"template_id": "T_R2", "question_type_id": "QS_V3", "text": "Hỏi [KEY]"}],
        dataset_rows=dataset_rows,
    )
    assert len(gate_r2["valid_types"]) == 1
    novel_r2, dups_r2 = deduplicate_by_signature(gate_r2["valid_types"], index)
    assert len(novel_r2) == 0
    assert len(dups_r2) == 1
    assert dups_r2[0]["duplicate_of"] == "QS_V1"


# ============================================================
# CONTRACT CLEANUP: TRANSCRIPTION REMOVAL & SAMPLE FALLBACK
# ============================================================

def test_contract_a_answer_kinds_exact():
    """A: ANSWER_KINDS exactly equals {'field_value', 'equality', 'derived_field'}."""
    assert ANSWER_KINDS == {"field_value", "equality", "derived_field"}
    assert "transcription" not in ANSWER_KINDS


def test_contract_b_transcription_rejected_as_unsupported():
    """B: kind='transcription' deterministically rejects as unsupported answer kind."""
    ans = {"kind": "transcription"}
    # 1. validate_answer_shape rejects
    shape_errs = validate_answer_shape(ans)
    assert any("bad_answer_kind:transcription" in e for e in shape_errs)

    # 2. validate_answer_references rejects
    ref_errs = validate_answer_references(ans)
    assert any("bad_answer_kind:transcription" in e for e in ref_errs)

    # 3. validate_question_type rejects
    qtype = {
        "id": "QS_T", "name": "Speech Transcription", "goal": "transcribe speech",
        "uses": ["audio", "text"], "input_count": 1,
        "answer_rule": "ground truth text", "answer": ans,
    }
    type_errs = validate_question_type(qtype)
    assert any("bad_answer_kind:transcription" in e for e in type_errs)

    # 4. _gold_answer raises ValueError
    with pytest.raises(ValueError, match="Unsupported answer kind"):
        _gold_answer(qtype, {"text": "hello"}, None, rows=[])


def test_contract_c_field_value_text_passes():
    """C: field_value(text) passes normally."""
    ans = {"kind": "field_value", "key": "text"}
    assert validate_answer_shape(ans) == []
    assert validate_answer_references(ans) == []

    qtype = {
        "id": "QS_T", "name": "Speech Transcription", "goal": "transcribe speech",
        "uses": ["audio", "text"], "input_count": 1,
        "answer_rule": "ground truth text", "answer": ans,
    }
    assert validate_question_type(qtype) == []


def test_contract_d_executable_signature_field_value_text():
    """D: executable signature: ('audio1', 'field_value', 'text')."""
    qtype = {
        "id": "QS_T", "name": "Speech Transcription",
        "answer": {"kind": "field_value", "key": "text"},
    }
    assert executable_signature(qtype) == ("audio1", "field_value", "text")


def test_contract_e_no_transcription_branches_in_production():
    """E: no production execution branch relies on answer.kind == 'transcription'."""
    import inspect
    import src.autonomous_qa.core.validity as val_mod
    import src.autonomous_qa.production.render_preview as rend_mod
    import src.autonomous_qa.compiler.pipeline_runner as run_mod

    for mod in (val_mod, rend_mod, run_mod):
        src = inspect.getsource(mod)
        assert 'kind == "transcription"' not in src
        assert "kind == 'transcription'" not in src
        assert 'answer.kind == "transcription"' not in src
        assert "answer.kind == 'transcription'" not in src
        assert 'ans.get("kind") == "transcription"' not in src
        assert "ans.get('kind') == 'transcription'" not in src


def test_contract_7_missing_real_rows_rejects_derived():
    """7: dataset_rows=[] -> derived_field auto-rejected as derived_relation_data_unavailable. No sample fallback."""
    qtype = {
        "id": "QS_D", "name": "Derived Relation", "goal": "infer",
        "uses": ["audio", "province", "region"], "input_count": 1,
        "answer_rule": "derive",
        "answer": {"kind": "derived_field", "source_key": "province", "target_key": "region"},
    }
    # No sample fallback: explicit empty rows fails closed
    errs = validate_question_type(qtype, dataset_rows=[])
    assert "derived_relation_data_unavailable" in errs

    # In gate_round with empty rows:
    gate = gate_round([qtype], [{"template_id": "T1", "question_type_id": "QS_D", "text": "Hỏi [KEY]"}], dataset_rows=[])
    assert len(gate["valid_types"]) == 0
    assert any(rej["reason"] == "derived_relation_data_unavailable" for rej in gate["auto_rejected"])


def test_contract_8_direct_and_equality_without_rows():
    """8: dataset_rows=[] -> direct (field_value) and equality types pass normally."""
    # Direct
    direct = {
        "id": "QS_G", "name": "Gender Recognition", "goal": "identify gender",
        "uses": ["audio", "gender"], "input_count": 1,
        "answer_rule": "gender label",
        "answer": {"kind": "field_value", "key": "gender"},
    }
    assert validate_question_type(direct, dataset_rows=[]) == []

    # Equality
    eq = {
        "id": "QS_E", "name": "Speaker Verification", "goal": "verify",
        "uses": ["audio", "speakerID"], "input_count": 2,
        "answer_rule": "same speakerID",
        "answer": {"kind": "equality", "keys": ["speakerID_1", "speakerID_2"]},
    }
    assert validate_question_type(eq, dataset_rows=[]) == []

    # Both pass gate_round with empty rows
    gate = gate_round(
        [direct, eq],
        [
            {"template_id": "T_G", "question_type_id": "QS_G", "text": "Hỏi [KEY]"},
            {"template_id": "T_E", "question_type_id": "QS_E", "text": "Hai đoạn có cùng người nói không?"},
        ],
        type_key_realizations={"QS_G": "giới tính"},
        dataset_rows=[],
    )
    assert len(gate["valid_types"]) == 2
    assert {t["id"] for t in gate["valid_types"]} == {"QS_G", "QS_E"}


def test_contract_9_explicit_sample_rows_allowed_and_no_runbase_fallback():
    """9: Tests/mock may explicitly supply sample rows; _RunBase has no implicit fallback."""
    from src.autonomous_qa.compiler.pipeline_runner import _RunBase
    from src.autonomous_qa.datasets.context_adapter import load_sample

    sample_rows = load_sample("vimd")
    assert len(sample_rows) >= 20

    # Explicitly supplying sample rows is allowed
    qtype = {
        "id": "QS_D", "name": "Derived", "goal": "infer",
        "uses": ["audio", "province", "region"], "input_count": 1,
        "answer_rule": "derive",
        "answer": {"kind": "derived_field", "source_key": "province", "target_key": "region"},
    }
    assert validate_question_type(qtype, dataset_rows=sample_rows) == []

    # _RunBase does NOT fall back to sample.jsonl when rows is empty or None
    run_empty = _RunBase(dataset="vimd", run_id="r1", out_dir=None, raw_dir=None,
                         rounds_dir=None, readme="", rows=[], client=None)
    assert run_empty.rows == []

    run_none = _RunBase(dataset="vimd", run_id="r2", out_dir=None, raw_dir=None,
                        rounds_dir=None, readme="", rows=None, client=None)
    assert run_none.rows == []

    # Explicit rows are preserved
    run_explicit = _RunBase(dataset="vimd", run_id="r3", out_dir=None, raw_dir=None,
                            rounds_dir=None, readme="", rows=sample_rows, client=None)
    assert run_explicit.rows == sample_rows



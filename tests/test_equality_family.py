"""Tests for shared operation-family base template refactor (§43-§55).

Verifies that semantic equality types share delexicalized base skeletons with [KEY],
bind type-specific realizations before Gate and Paraphrase, persist approved bases
into family_template_bank across rounds, keep QA types and executable signatures
completely distinct, and preserve separation of speaker verification.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from src.autonomous_qa.language.family_templates import (
    FAMILY_EQUALITY_SEMANTIC,
    FAMILY_EQUALITY_SEMANTIC_SPEAKER,
    FAMILY_EQUALITY_SEMANTIC_UTTERANCE,
    bind_family_template,
    bind_round_family_templates,
    build_family_bank_entry,
    deserialize_family_signature,
    serialize_family_signature,
)
from src.autonomous_qa.core.loop import commit_approved_bindings
from src.autonomous_qa.production.render_preview import render_question
from src.autonomous_qa.compiler.pipeline_runner import (
    TEMPLATE_CONTRACT_VERSION,
    _ResumedRun,
    _RunBase,
)
from src.autonomous_qa.core.validity import (
    DEFAULT_FIELD_ROLES,
    ROLE_HIDDEN_IDENTIFIER,
    ROLE_SEMANTIC,
    SCHEMA_FIELDS,
    build_approved_signature_index,
    build_round_field_candidates,
    deduplicate_by_signature,
    executable_signature,
    gate_round,
    template_family_signature,
    validate_family_base_template,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eq_type(tid: str, base_field: str, roles: dict[str, str] | None = None) -> dict:
    return {
        "id": tid,
        "name": f"Same {base_field.title()} Verification",
        "goal": f"Decide whether two recordings have the same {base_field}.",
        "uses": ["audio", base_field],
        "input_count": 2,
        "answer_rule": f"same {base_field} => yes, else no",
        "answer": {
            "kind": "equality",
            "keys": [f"{base_field}_1", f"{base_field}_2"],
        },
    }


def _fv_type(tid: str, field: str) -> dict:
    return {
        "id": tid,
        "name": f"{field.title()} Recognition",
        "goal": f"Identify the speaker's {field}.",
        "uses": ["audio", field],
        "input_count": 1,
        "answer_rule": f"the {field} label",
        "answer": {"kind": "field_value", "key": field},
    }


def _derived_type(tid: str, src: str, tgt: str) -> dict:
    return {
        "id": tid,
        "name": f"{src.title()} to {tgt.title()}",
        "goal": f"Infer {tgt} from {src}.",
        "uses": ["audio", src, tgt],
        "input_count": 1,
        "answer_rule": f"the {tgt} value",
        "answer": {"kind": "derived_field", "source_key": src, "target_key": tgt},
    }


# ---------------------------------------------------------------------------
# §43: Family signature check
# ---------------------------------------------------------------------------

def test_43_family_signature_detection():
    vimd_roles = {
        "audio": "audio_input",
        "text": "semantic",
        "gender": "semantic",
        "region": "semantic",
        "province": "semantic",
        "speakerID": "hidden_identifier",
    }
    # ViMD semantic equality -> ("equality", "semantic", "speaker")
    for f in ("gender", "region", "province"):
        t = _eq_type(f"QS_{f}", f)
        assert template_family_signature(t, vimd_roles) == FAMILY_EQUALITY_SEMANTIC_SPEAKER

    # Speech-MASSIVE semantic equality -> partitioned by entity_scope
    massive_roles = {
        "audio": "audio_input",
        "text": "semantic",
        "intent": "semantic",
        "scenario": "semantic",
        "speaker_sex": "semantic",
        "speaker_age": "semantic",
        "speaker_id": "hidden_identifier",
        "id": "provenance",
    }
    massive_fields = frozenset(massive_roles.keys())
    massive_scopes = {
        "intent": "utterance",
        "scenario": "utterance",
        "speaker_sex": "speaker",
        "speaker_age": "speaker",
    }
    for f in ("speaker_sex", "speaker_age"):
        t = _eq_type(f"QS_{f}", f)
        assert template_family_signature(t, massive_roles, massive_fields, massive_scopes) == FAMILY_EQUALITY_SEMANTIC_SPEAKER
    for f in ("intent", "scenario"):
        t = _eq_type(f"QS_{f}", f)
        assert template_family_signature(t, massive_roles, massive_fields, massive_scopes) == FAMILY_EQUALITY_SEMANTIC_UTTERANCE

    # Speaker verification -> None (hidden_identifier)
    t_spk_vimd = _eq_type("QS_spk", "speakerID")
    assert template_family_signature(t_spk_vimd, vimd_roles) is None

    t_spk_massive = _eq_type("QS_spk2", "speaker_id")
    assert template_family_signature(t_spk_massive, massive_roles, massive_fields) is None

    # field_value and derived_field -> None
    assert template_family_signature(_fv_type("QS_fv", "gender"), vimd_roles) is None
    assert template_family_signature(_derived_type("QS_der", "province", "region"), vimd_roles) is None


# ---------------------------------------------------------------------------
# §44: Same base skeleton, different bindings
# ---------------------------------------------------------------------------

def test_44_same_base_different_bindings():
    base = {
        "base_template_id": "BF_EQSEM_001",
        "text": "Giọng người nói trong hai đoạn âm thanh có cùng [KEY] không?",
    }
    assert validate_family_base_template(base["text"]) == []

    bound_gender = bind_family_template(base, "QS_gender", "giới tính")
    assert bound_gender["template_id"] == "BT_QS_gender__BF_EQSEM_001"
    assert bound_gender["text"] == "Giọng người nói trong hai đoạn âm thanh có cùng giới tính không?"
    assert bound_gender["key_realization"] == "giới tính"
    assert bound_gender["base_template_id"] == "BF_EQSEM_001"
    assert "[KEY]" not in bound_gender["text"]

    bound_region = bind_family_template(base, "QS_region", "vùng miền")
    assert bound_region["template_id"] == "BT_QS_region__BF_EQSEM_001"
    assert bound_region["text"] == "Giọng người nói trong hai đoạn âm thanh có cùng vùng miền không?"
    assert bound_region["key_realization"] == "vùng miền"

    bound_province = bind_family_template(base, "QS_province", "tỉnh thành")
    assert bound_province["template_id"] == "BT_QS_province__BF_EQSEM_001"
    assert bound_province["text"] == "Giọng người nói trong hai đoạn âm thanh có cùng tỉnh thành không?"
    assert bound_province["key_realization"] == "tỉnh thành"


# ---------------------------------------------------------------------------
# §45: Paraphrase receives already-lexicalized templates
# ---------------------------------------------------------------------------

def test_45_paraphrase_input_is_lexicalized():
    base = {
        "base_template_id": "BF_EQSEM_001",
        "text": "Hai bản ghi có cùng [KEY] không?",
    }
    types = [_eq_type("QS_G", "gender")]
    krs = {"QS_G": "giới tính"}
    bound = bind_round_family_templates([base], types, krs)
    assert len(bound) == 1
    # Check that the text delivered to downstream stages contains no [KEY]
    assert "[KEY]" not in bound[0]["text"]
    assert bound[0]["text"] == "Hai bản ghi có cùng giới tính không?"


# ---------------------------------------------------------------------------
# §46: Type-specific paraphrase ownership
# ---------------------------------------------------------------------------

def test_46_type_specific_paraphrase_scoping():
    t_gender = _eq_type("QS_G", "gender")
    t_region = _eq_type("QS_R", "region")
    base = {"base_template_id": "BF_EQSEM_001", "text": "Có cùng [KEY] không?"}
    bound = bind_round_family_templates([base], [t_gender, t_region], {
        "QS_G": "giới tính", "QS_R": "vùng miền",
    })
    assert len(bound) == 2
    b_g = [b for b in bound if b["question_type_id"] == "QS_G"][0]
    b_r = [b for b in bound if b["question_type_id"] == "QS_R"][0]

    # Paraphrase for gender
    para_g = {
        "template_id": "P_QS_G_01",
        "source_template_id": b_g["template_id"],
        "text": "Hai đoạn có chung giới tính không?",
    }
    # Paraphrase for region
    para_r = {
        "template_id": "P_QS_R_01",
        "source_template_id": b_r["template_id"],
        "text": "Hai đoạn có chung vùng miền không?",
    }

    from src.autonomous_qa.core.loop import filter_paraphrases_for_judging
    filt = filter_paraphrases_for_judging(bound, [para_g, para_r])
    assert filt["owners"]["P_QS_G_01"] == "QS_G"
    assert filt["owners"]["P_QS_R_01"] == "QS_R"


# ---------------------------------------------------------------------------
# §47: Round 2 reuse from family_template_bank
# ---------------------------------------------------------------------------

def test_47_round_2_bank_reuse():
    # Setup RunBase with empty bank
    run = _RunBase(
        dataset="vimd", run_id="test-bank-reuse", out_dir=Path("."),
        raw_dir=Path("."), rounds_dir=Path("."), readme="", rows=[], client=None,
    )
    assert run.family_template_bank == {}

    # Round 1 introduces base BF_EQSEM_001
    r1_types = [_eq_type("QS_R1_G", "gender")]
    r1_tpl_out = {
        "round": 1,
        "family_templates": [
            {"base_template_id": "BF_EQSEM_001",
             "text": "Hai file audio có cùng [KEY] không?"}
        ],
        "key_realizations": {"QS_R1_G": "giới tính"},
        "templates": [],
    }
    r1_phrases, r1_combined = run._process_round_family_templates(1, r1_types, r1_tpl_out)
    assert len(r1_combined) == 1
    assert r1_combined[0]["template_id"] == "BT_QS_R1_G__BF_EQSEM_001"

    # Quality accepts QS_R1_G and keeps BT_QS_R1_G__BF_EQSEM_001
    verdict_r1 = {
        "accepted_new": [{"type_id": "QS_R1_G", "keep_template_ids": ["BT_QS_R1_G__BF_EQSEM_001"]}],
        "duplicates": [],
        "rejected": [],
    }
    run._commit_family_bases(1, verdict_r1, [{"type_id": "QS_R1_G"}], [])
    assert FAMILY_EQUALITY_SEMANTIC_SPEAKER in run.family_template_bank
    assert len(run.family_template_bank[FAMILY_EQUALITY_SEMANTIC_SPEAKER]) == 1
    assert run.family_template_bank[FAMILY_EQUALITY_SEMANTIC_SPEAKER][0]["base_template_id"] == "BF_EQSEM_001"

    # Round 2 introduces new type QS_R2_R (region) WITHOUT generating new skeletons
    r2_types = [_eq_type("QS_R2_R", "region")]
    r2_tpl_out = {
        "round": 2,
        # LLM does NOT provide family_templates because bank already has it
        "family_templates": [],
        "key_realizations": {"QS_R2_R": "vùng miền"},
        "templates": [],
    }
    r2_phrases, r2_combined = run._process_round_family_templates(2, r2_types, r2_tpl_out)
    assert len(r2_combined) == 1
    assert r2_combined[0]["template_id"] == "BT_QS_R2_R__BF_EQSEM_001"
    assert r2_combined[0]["text"] == "Hai file audio có cùng vùng miền không?"


# ---------------------------------------------------------------------------
# §48: Cross-dataset Speech-MASSIVE
# ---------------------------------------------------------------------------

def test_48_cross_dataset_speech_massive_family():
    massive_roles = {
        "audio": "audio_input",
        "text": "semantic",
        "intent": "semantic",
        "scenario": "semantic",
        "speaker_sex": "semantic",
        "speaker_id": "hidden_identifier",
    }
    massive_fields = frozenset(massive_roles.keys())

    massive_scopes = {
        "intent": "utterance",
        "scenario": "utterance",
        "speaker_sex": "speaker",
    }

    t_intent = _eq_type("SM_intent", "intent")
    t_scenario = _eq_type("SM_scenario", "scenario")
    t_sex = _eq_type("SM_sex", "speaker_sex")

    for t in (t_intent, t_scenario):
        assert template_family_signature(t, massive_roles, massive_fields, massive_scopes) == FAMILY_EQUALITY_SEMANTIC_UTTERANCE
    assert template_family_signature(t_sex, massive_roles, massive_fields, massive_scopes) == FAMILY_EQUALITY_SEMANTIC_SPEAKER

    base = {"base_template_id": "BF_EQSEM_001", "text": "Hai câu nói có cùng [KEY] không?"}
    bound = bind_round_family_templates([base], [t_intent, t_scenario, t_sex], {
        "SM_intent": "ý định",
        "SM_scenario": "tình huống",
        "SM_sex": "giới tính người nói",
    })
    assert len(bound) == 3
    texts = {b["question_type_id"]: b["text"] for b in bound}
    assert texts["SM_intent"] == "Hai câu nói có cùng ý định không?"
    assert texts["SM_scenario"] == "Hai câu nói có cùng tình huống không?"
    assert texts["SM_sex"] == "Hai câu nói có cùng giới tính người nói không?"


# ---------------------------------------------------------------------------
# §49: Speaker verification remains type-specific
# ---------------------------------------------------------------------------

def test_49_speaker_verification_separate():
    roles = {
        "audio": "audio_input",
        "speakerID": "hidden_identifier",
        "gender": "semantic",
    }
    t_spk = _eq_type("QS_spk", "speakerID")
    assert template_family_signature(t_spk, roles) is None

    # Family binding skips speaker verification
    base = {"base_template_id": "BF_EQSEM_001", "text": "Hai file có cùng [KEY] không?"}
    bound = bind_round_family_templates([base], [t_spk], {"QS_spk": "người nói"})
    # bind_round_family_templates binds only what is given, but in pipeline:
    run = _RunBase(
        dataset="vimd", run_id="test-spk-sep", out_dir=Path("."),
        raw_dir=Path("."), rounds_dir=Path("."), readme="", rows=[], client=None,
    )
    run.field_roles = roles
    out = {
        "round": 1,
        "family_templates": [base],
        "templates": [{"template_id": "T_spk_01", "question_type_id": "QS_spk",
                       "text": "Hai đoạn âm thanh có phải do cùng một người nói không?"}],
        "key_realizations": {},
    }
    phrases, combined = run._process_round_family_templates(1, [t_spk], out)
    assert len(combined) == 1
    assert combined[0]["template_id"] == "T_spk_01"
    assert "người nói" in combined[0]["text"]


# ---------------------------------------------------------------------------
# §50: Final templates free of [KEY]
# ---------------------------------------------------------------------------

def test_50_final_templates_free_of_key():
    base = {"base_template_id": "BF_EQSEM_001", "text": "Hai file có cùng [KEY] không?"}
    t_gender = _eq_type("QS_G", "gender")
    bound = bind_family_template(base, "QS_G", "giới tính")

    # In final preview rendering, slot-free question renders without [KEY]
    rendered, kr = render_question(bound["text"], bound["template_id"], None, {}, "Có")
    assert "[KEY]" not in rendered
    assert "[" not in rendered
    assert rendered == "Hai file có cùng giới tính không?"


# ---------------------------------------------------------------------------
# §51: QA types and executable signatures remain distinct
# ---------------------------------------------------------------------------

def test_51_qa_types_and_signatures_remain_distinct():
    t_gender = _eq_type("QS_G", "gender")
    t_region = _eq_type("QS_R", "region")
    t_prov = _eq_type("QS_P", "province")

    sig_g = executable_signature(t_gender)
    sig_r = executable_signature(t_region)
    sig_p = executable_signature(t_prov)

    assert sig_g == ("audio2", "equality", "gender")
    assert sig_r == ("audio2", "equality", "region")
    assert sig_p == ("audio2", "equality", "province")

    assert len({sig_g, sig_r, sig_p}) == 3

    # Deduplication does not conflate them
    idx = build_approved_signature_index([{"type_id": "QS_G", "answer": t_gender["answer"]}])
    novel, det_dups = deduplicate_by_signature([t_region, t_prov], idx)
    assert len(novel) == 2
    assert len(det_dups) == 0


# ---------------------------------------------------------------------------
# §52: Quality judges each concrete type independently
# ---------------------------------------------------------------------------

def test_52_quality_difference_per_type():
    t_gender = _eq_type("QS_G", "gender")
    t_region = _eq_type("QS_R", "region")

    verdict = {
        "round": 1,
        "accepted_new": [{"type_id": "QS_G", "keep_template_ids": ["BT_QS_G__BF_EQSEM_001"]}],
        "duplicates": [],
        "rejected": [{"type_id": "QS_R", "reason": "dialect boundaries too vague"}],
    }

    from src.autonomous_qa.core.loop import apply_accepted_types
    pool: list[dict] = []
    types_by_id = {"QS_G": t_gender, "QS_R": t_region}
    texts = {
        "BT_QS_G__BF_EQSEM_001": "Hai file có cùng giới tính không?",
        "BT_QS_R__BF_EQSEM_001": "Hai file có cùng vùng miền không?",
    }
    owners = {
        "BT_QS_G__BF_EQSEM_001": "QS_G",
        "BT_QS_R__BF_EQSEM_001": "QS_R",
    }
    accepted = apply_accepted_types(pool, types_by_id, texts, verdict, 1, owners)
    assert len(accepted) == 1
    assert accepted[0]["type_id"] == "QS_G"
    assert len(pool) == 1
    assert pool[0]["type_id"] == "QS_G"


# ---------------------------------------------------------------------------
# §53: Family bank conditional persistence
# ---------------------------------------------------------------------------

def test_53_family_bank_conditional_persistence():
    run = _RunBase(
        dataset="vimd", run_id="test-bank-cond", out_dir=Path("."),
        raw_dir=Path("."), rounds_dir=Path("."), readme="", rows=[], client=None,
    )

    base = {"base_template_id": "BF_EQSEM_001", "text": "Cùng [KEY] không?"}
    run._round_provisional_family_bases[1] = [base]

    # Case A: Quality rejects all types -> base is NOT persisted
    verdict_reject = {
        "accepted_new": [],
        "rejected": [{"type_id": "QS_G", "reason": "bad"}],
    }
    run._commit_family_bases(1, verdict_reject, [], [])
    assert FAMILY_EQUALITY_SEMANTIC_SPEAKER not in run.family_template_bank or len(run.family_template_bank[FAMILY_EQUALITY_SEMANTIC_SPEAKER]) == 0

    # Case B: Quality accepts QS_G and keeps BT_QS_G__BF_EQSEM_001 -> base IS persisted
    verdict_accept = {
        "accepted_new": [{"type_id": "QS_G", "keep_template_ids": ["BT_QS_G__BF_EQSEM_001"]}],
        "rejected": [],
    }
    run._commit_family_bases(1, verdict_accept, [{"type_id": "QS_G"}], [])
    assert len(run.family_template_bank[FAMILY_EQUALITY_SEMANTIC_SPEAKER]) == 1
    assert run.family_template_bank[FAMILY_EQUALITY_SEMANTIC_SPEAKER][0]["base_template_id"] == "BF_EQSEM_001"


# ---------------------------------------------------------------------------
# §54: Dynamic style feedback preserves approved equality types
# ---------------------------------------------------------------------------

def test_54_dynamic_style_feedback():
    from src.autonomous_qa.core.loop import build_type_fewshot
    pool = [{
        "type_id": "QS_G",
        "name": "Speaker Gender Verification",
        "goal": "Verify same gender.",
        "uses": ["audio", "gender"],
        "input_count": 2,
        "answer_rule": "same gender",
        "answer": {"kind": "equality", "keys": ["gender_1", "gender_2"]},
    }]
    fewshot = build_type_fewshot(pool)
    assert len(fewshot) == 1
    assert fewshot[0]["name"] == "Speaker Gender Verification"
    assert fewshot[0]["uses"] == ["audio", "gender"]
    assert fewshot[0]["goal"] == "Verify same gender."
    assert fewshot[0]["answer_rule"] == "same gender"


# ---------------------------------------------------------------------------
# §55: Historical runs incompatible
# ---------------------------------------------------------------------------

def test_55_historical_runs_incompatible(tmp_path):
    run_dir = tmp_path / "old_run"
    run_dir.mkdir()
    (run_dir / "run_state.json").write_text(json.dumps({
        "run_id": "old_run",
        "dataset": "vimd",
        "mode": "real",
        "completed": ["style_round_1"],
        "next_action": "template_round_1",
        "loop_config": {
            "max_rounds": 3, "min_new_type_rate": 0.15,
            "saturation_patience": 2, "immediate_stop_if_zero_new": True,
        },
        "template_contract_version": "type-keyed-v1",
    }))
    run = _ResumedRun(
        dataset="vimd", out_dir=run_dir, raw_dir=run_dir,
        rounds_dir=run_dir, readme="", rows=[], client=None,
    )
    with pytest.raises(ValueError, match="Question Template contract changed"):
        run.load_and_validate()

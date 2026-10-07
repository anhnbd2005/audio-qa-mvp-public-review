"""Canonical answer shape + executable-signature dedup (§88-§122).

Pure unit tests plus pipeline control-flow tests. All LLM traffic faked
(0 real calls).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.autonomous_qa.authoring.llm_client as llm_client
from src.autonomous_qa.core.loop import commit_approved_bindings
from src.autonomous_qa.compiler.pipeline_runner import (
    ANSWER_SHAPE_CONTRACT_VERSION,
    EXECUTABLE_SIGNATURE_CONTRACT_VERSION,
    TEMPLATE_CONTRACT_VERSION,
    run_pipeline,
)
from src.autonomous_qa.core.validity import (
    build_approved_signature_index,
    deduplicate_by_signature,
    executable_signature,
    validate_answer_shape,
    validate_question_type,
)
from tests.test_safety import clean_budget  # noqa: F401
from tests.test_zero_valid_parity import _r1_quality_no_derived_keep


def _t(tid="QX", answer=None, **over):
    base = {"id": tid, "name": "N", "goal": "g",
            "uses": ["audio", "gender"], "input_count": 1,
            "answer_rule": "r",
            "answer": {"kind": "field_value", "key": "gender"}}
    if answer is not None:
        base["answer"] = answer
    base.update(over)
    return base


# §88 — field_value canonical passes (bare + explicit nulls).
def test_88_field_value_canonical():
    assert validate_answer_shape(
        {"kind": "field_value", "key": "province"}) == []
    assert validate_answer_shape(
        {"kind": "field_value", "key": "province", "keys": None,
         "source_key": None, "target_key": None}) == []


# §89/§90 — real false-novel shape + other extras fail.
def test_89_90_field_value_extras():
    assert validate_answer_shape(
        {"kind": "field_value", "key": "province",
         "source_key": "region"}) == [
        "answer_shape_unexpected_field:field_value:source_key"]
    assert validate_answer_shape(
        {"kind": "field_value", "key": "province",
         "keys": ["province"]}) == [
        "answer_shape_unexpected_field:field_value:keys"]
    assert validate_answer_shape(
        {"kind": "field_value", "key": "province",
         "target_key": "region"}) == [
        "answer_shape_unexpected_field:field_value:target_key"]
    # Gate-level: the QS_R2_04-style type auto-rejects.
    errs = validate_question_type(_t(
        "QS_R2_04", {"kind": "field_value", "key": "province",
                     "source_key": "region"},
        uses=["audio", "province"]))
    assert "answer_shape_unexpected_field:field_value:source_key" in errs
    errs2 = validate_question_type(_t(
        "QS_R2_05", {"kind": "field_value", "key": "region",
                     "source_key": "province"},
        uses=["audio", "region"]))
    assert "answer_shape_unexpected_field:field_value:source_key" in errs2


# §7 — []/"" are noncanonical, not null.
def test_07_empty_is_not_null():
    assert validate_answer_shape(
        {"kind": "field_value", "key": "province", "keys": []}) == [
        "answer_shape_unexpected_field:field_value:keys"]
    assert validate_answer_shape(
        {"kind": "derived_field", "source_key": "province",
         "target_key": "region", "key": ""}) == [
        "answer_shape_unexpected_field:derived_field:key"]


# §91/§92 — equality canonical + extras.
def test_91_92_equality_shape():
    assert validate_answer_shape(
        {"kind": "equality",
         "keys": ["province_1", "province_2"]}) == []
    assert validate_answer_shape(
        {"kind": "equality", "key": "province",
         "keys": ["province_1", "province_2"]}) == [
        "answer_shape_unexpected_field:equality:key"]
    assert validate_answer_shape(
        {"kind": "equality", "keys": ["province_1", "province_2"],
         "source_key": "province"}) == [
        "answer_shape_unexpected_field:equality:source_key"]
    assert validate_answer_shape(
        {"kind": "equality", "keys": ["province_1", "province_2"],
         "target_key": "province"}) == [
        "answer_shape_unexpected_field:equality:target_key"]


# §93/§94 — derived canonical + extras.
def test_93_94_derived_shape():
    assert validate_answer_shape(
        {"kind": "derived_field", "source_key": "province",
         "target_key": "region"}) == []
    assert validate_answer_shape(
        {"kind": "derived_field", "key": "region",
         "source_key": "province", "target_key": "region"}) == [
        "answer_shape_unexpected_field:derived_field:key"]
    assert validate_answer_shape(
        {"kind": "derived_field", "keys": ["province", "region"],
         "source_key": "province", "target_key": "region"}) == [
        "answer_shape_unexpected_field:derived_field:keys"]


# §95 — required fields missing.
def test_95_missing_required():
    assert validate_answer_shape({"kind": "field_value"}) == [
        "answer_shape_missing_field:field_value:key"]
    assert validate_answer_shape({"kind": "equality"}) == [
        "answer_shape_missing_field:equality:keys"]
    assert validate_answer_shape(
        {"kind": "derived_field", "target_key": "region"}) == [
        "answer_shape_missing_field:derived_field:source_key"]
    assert validate_answer_shape(
        {"kind": "derived_field", "source_key": "province"}) == [
        "answer_shape_missing_field:derived_field:target_key"]


# §96/§97 — the 8 MVP signatures + derived direction.
def test_96_97_signatures():
    s = executable_signature
    assert s(_t(answer={"kind": "field_value", "key": "text"},
                 uses=["audio", "text"])) == (
                     "audio1", "field_value", "text")
    assert s(_t(answer={"kind": "field_value",
                       "key": "gender"})) == (
                           "audio1", "field_value", "gender")
    assert s(_t(answer={"kind": "field_value", "key": "region"},
                 uses=["audio", "region"])) == (
                     "audio1", "field_value", "region")
    assert s(_t(answer={"kind": "field_value", "key": "province"},
                 uses=["audio", "province"])) == (
                     "audio1", "field_value", "province")
    assert s(_t(answer={"kind": "equality",
                       "keys": ["speakerID_1", "speakerID_2"]},
                 uses=["audio", "speakerID"])) == (
                     "audio2", "equality", "speakerID")
    assert s(_t(answer={"kind": "equality",
                       "keys": ["gender_1", "gender_2"]})) == (
                           "audio2", "equality", "gender")
    assert s(_t(answer={"kind": "equality",
                       "keys": ["region_1", "region_2"]},
                 uses=["audio", "region"])) == (
                     "audio2", "equality", "region")
    assert s(_t(answer={"kind": "equality",
                       "keys": ["province_1", "province_2"]},
                 uses=["audio", "province"])) == (
                     "audio2", "equality", "province")
    assert s(_t(answer={"kind": "derived_field", "source_key": "province",
                       "target_key": "region"},
                 uses=["audio", "province", "region"])) == (
                     "audio1", "derived_field", "province", "region")
    fwd = s(_t("A", {"kind": "derived_field", "source_key": "province",
                     "target_key": "region"},
                uses=["audio", "province", "region"]))
    bwd = s(_t("B", {"kind": "derived_field", "source_key": "region",
                     "target_key": "province"},
                uses=["audio", "province", "region"]))
    assert fwd != bwd
    with pytest.raises(ValueError):
        s(_t(answer={"kind": "field_value", "key": "province",
                     "source_key": "region"}))


# §49/§50/§55/§57/§98 — name/goal/prose/id do not matter.
def test_98_99_names_goals_ids_ignored():
    a = _t("QS_A", {"kind": "field_value", "key": "province"},
           name="Province Dialect Recognition",
           goal="identify province-level dialect",
           answer_rule="return province field",
           uses=["audio", "province"])
    b = _t("QS_B", {"kind": "field_value", "key": "province"},
           name="Fine-Grained Local Accent Classification",
           goal="infer speaker's local dialect category",
           answer_rule="infer fine-grained province dialect",
           uses=["audio", "province", "region"])
    assert executable_signature(a) == executable_signature(b)


# §52/§99 — extra uses do not matter (mandatory).
def test_99_uses_ignored():
    a = _t("QS_A", {"kind": "field_value", "key": "province"},
           uses=["audio", "province"])
    b = _t("QS_B", {"kind": "field_value", "key": "province"},
           uses=["audio", "province", "region"])
    assert executable_signature(a) == executable_signature(b)


# §101/§102 — prior approved dup (incl. canonical-form real case).
def test_101_102_prior_approved_dup():
    index = build_approved_signature_index(
        [_t("QS_R1_04", {"kind": "field_value", "key": "province"},
            uses=["audio", "province"])])
    novel, dups = deduplicate_by_signature(
        [_t("QS_R2_04", {"kind": "field_value", "key": "province"},
            name="Speaker Province Recognition (from Region context)",
            uses=["audio", "province", "region"])], index)
    assert novel == []
    assert dups == [{"type_id": "QS_R2_04", "duplicate_of": "QS_R1_04",
                     "signature": ["audio1", "field_value", "province"],
                     "duplicate_source": "prior_approved"}]


# §103 — current-round first wins.
def test_103_current_round_first_wins():
    novel, dups = deduplicate_by_signature(
        [_t("QS_A", {"kind": "field_value", "key": "region"},
            uses=["audio", "region"]),
         _t("QS_B", {"kind": "field_value", "key": "region"},
            uses=["audio", "region"])], {})
    assert [t["id"] for t in novel] == ["QS_A"]
    assert dups[0]["duplicate_of"] == "QS_A"
    assert dups[0]["duplicate_source"] == "current_round"


# §104 — prior wins over both current types.
def test_104_prior_wins_over_current():
    index = build_approved_signature_index(
        [_t("QS_OLD", {"kind": "field_value", "key": "region"},
            uses=["audio", "region"])])
    novel, dups = deduplicate_by_signature(
        [_t("QS_A", {"kind": "field_value", "key": "region"},
            uses=["audio", "region"]),
         _t("QS_B", {"kind": "field_value", "key": "region"},
            uses=["audio", "region"])], index)
    assert novel == []
    assert [d["duplicate_of"] for d in dups] == ["QS_OLD", "QS_OLD"]


# §105/§106/§107/§48 — distinct operations are not dups.
def test_105_107_distinct_ops():
    index = {}
    reg = _t("Q1", {"kind": "field_value", "key": "region"},
             uses=["audio", "region"])
    prov = _t("Q2", {"kind": "field_value", "key": "province"},
              uses=["audio", "province"])
    novel, dups = deduplicate_by_signature([reg, prov], index)
    assert len(novel) == 2 and dups == []
    peq = _t("Q3", {"kind": "equality",
                    "keys": ["province_1", "province_2"]},
             uses=["audio", "province"])
    novel, dups = deduplicate_by_signature([prov, peq], index)
    assert len(novel) == 2 and dups == []
    der = _t("Q4", {"kind": "derived_field", "source_key": "province",
                    "target_key": "region"},
             uses=["audio", "province", "region"])
    novel, dups = deduplicate_by_signature([reg, der], index)
    assert len(novel) == 2 and dups == []


# §24 — duplicate approved pool raises.
def test_24_duplicate_approved_pool_raises():
    with pytest.raises(ValueError,
                       match="duplicate_approved_executable_signature"):
        build_approved_signature_index([
            _t("QS_A", {"kind": "field_value", "key": "province"},
               uses=["audio", "province"]),
            _t("QS_B", {"kind": "field_value", "key": "province"},
               uses=["audio", "province"])])


# §108/§109 — dup contributes no candidate and never commits.
def test_108_109_dup_no_candidate_no_commit():
    from src.autonomous_qa.core.validity import build_round_field_candidates
    approved = {"province": "tỉnh thành"}
    cands = build_round_field_candidates(
        [], [], [], {}, approved)
    assert cands == {}
    # A duplicate's private phrase is not the canonical candidate.
    round_cands = {"province": "tỉnh thành"}
    pool_approved: dict = dict(approved)
    out = commit_approved_bindings(
        pool_approved, round_cands,
        [{"type_id": "QS_DUP",
          "kept_templates": []}],  # dup never reaches Quality/apply
        {}, type_order=[])
    assert out["committed"] == []
    assert pool_approved == approved


def _payload(root: Path, *parts: str) -> dict:
    return json.loads(
        (root / "tests" / "fixtures" / Path(*parts)).with_suffix(
            ".json").read_text(encoding="utf-8"))


def _ns(item) -> SimpleNamespace:
    return SimpleNamespace(text=json.dumps(item, ensure_ascii=False),
                           parsed=None, candidates=[], usage_metadata=None)


# §112/§113/§114 — all-dups terminal + mixed + accounting.
def test_112_all_dups_terminal(isolated_root, clean_budget, monkeypatch):
    style = _payload(isolated_root, "loop", "round_01", "style")
    tpl = _payload(isolated_root, "loop", "round_01", "template")
    para = _payload(isolated_root, "loop", "round_01", "paraphrase")
    # R1 accepts gender + region (+ slot-free) so R2 re-proposals are
    # prior-approved duplicates; derived 05 rejected to dodge the
    # unrelated derived-render quirk.
    qual = {"round": 1,
            "accepted_new": [
                {"type_id": "QS_R1_01",
                 "keep_template_ids": ["T_R1_01_01", "T_R1_01_01_P01"]},
                {"type_id": "QS_R1_02",
                 "keep_template_ids": ["T_R1_02_01", "T_R1_02_01_P01"]},
                {"type_id": "QS_R1_03",
                 "keep_template_ids": ["T_R1_03_01", "T_R1_03_01_P01"]},
                {"type_id": "QS_R1_04",
                 "keep_template_ids": ["T_R1_04_01", "T_R1_04_01_P01"]}],
            "duplicates": [],
            "rejected": [{"type_id": "QS_R1_05", "reason": "judge reject"}]}
    # R2 re-proposes only already-approved capabilities (canonical).
    r2style = {"round": 2, "question_types": [
        {"id": "QS_R2_01", "name": "Gender Again", "goal": "g",
         "uses": ["audio", "gender"], "input_count": 1,
         "answer_rule": "the gender label",
         "answer": {"kind": "field_value", "key": "gender"}},
        {"id": "QS_R2_02", "name": "Region Again", "goal": "g",
         "uses": ["audio", "region"], "input_count": 1,
         "answer_rule": "the region label",
         "answer": {"kind": "field_value", "key": "region"}},
    ]}
    r2tpl = {"round": 2,
             "key_realizations": {"QS_R2_01": "giới tính",
                                  "QS_R2_02": "vùng giọng"},
             "templates": [
                 {"template_id": "T_A", "question_type_id": "QS_R2_01",
                  "text": "Hãy xác định [KEY]?"},
                 {"template_id": "T_B", "question_type_id": "QS_R2_02",
                  "text": "Cho biết [KEY]?"}]}
    fake_calls: list[str] = []

    from tests.test_safety import ScriptedLLM
    script = ScriptedLLM([style, tpl, para, qual, r2style, r2tpl])
    orig = script.__call__

    def counting(client, stage, prompt, response_schema):
        fake_calls.append(stage)
        return orig(client, stage, prompt, response_schema)

    monkeypatch.setattr(llm_client, "call_llm", counting)
    monkeypatch.setattr(llm_client, "create_llm_client", lambda: object())
    preview = run_pipeline(dataset="vimd", real_llm=True, run_id="alldup")
    # R2 used Style+Template only; Paraphrase/Quality never called for R2.
    assert fake_calls == ["question_style", "question_template",
                          "paraphrase", "quality",
                          "question_style", "question_template"]
    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "alldup"
    state = json.loads((out_dir / "run_state.json").read_text(
        encoding="utf-8"))
    assert state["stop_reason"] == "no_novel_types_after_signature_dedup"
    assert list((out_dir / "raw").glob(
        "paraphrase_round_2_attempt_*.txt")) == []
    assert list((out_dir / "raw").glob(
        "quality_round_2_attempt_*.txt")) == []
    assert not (out_dir / "rounds" / "round_02"
                / "03_paraphrases.json").exists()
    assert not (out_dir / "rounds" / "round_02"
                / "04_quality.json").exists()
    audit = json.loads((out_dir / "rounds" / "round_02"
                        / "signature_dedup.json").read_text(
                            encoding="utf-8"))
    assert audit["novel_type_ids"] == []
    assert {d["type_id"] for d in audit["deterministic_duplicates"]} == {
        "QS_R2_01", "QS_R2_02"}
    assert all(d["duplicate_source"] == "prior_approved"
               for d in audit["deterministic_duplicates"])
    summary = json.loads((out_dir / "loop_summary.json").read_text(
        encoding="utf-8"))
    r2 = summary["rounds"][-1]
    assert (r2["generated_types"]
            == r2["rejected"] + r2["deterministic_duplicates"]
            + r2["accepted_new_types"] + r2["duplicates"])
    assert len(preview) > 0


def test_113_mixed_dup_novel(isolated_root, clean_budget, monkeypatch):
    from tests.test_zero_valid_parity import (
        _r1_quality_no_derived_keep as _q1,
        _zero_valid_style as _zv_s,
        _zero_valid_template as _zv_t,
    )
    style = _payload(isolated_root, "loop", "round_01", "style")
    tpl = _payload(isolated_root, "loop", "round_01", "template")
    para = _payload(isolated_root, "loop", "round_01", "paraphrase")
    qual = _q1()
    r2style = {"round": 2, "question_types": [
        {"id": "QS_R2_01", "name": "Gender Again", "goal": "g",
         "uses": ["audio", "gender"], "input_count": 1,
         "answer_rule": "the gender label",
         "answer": {"kind": "field_value", "key": "gender"}},
        {"id": "QS_R2_02", "name": "Province New", "goal": "g",
         "uses": ["audio", "province"], "input_count": 1,
         "answer_rule": "the province label",
         "answer": {"kind": "field_value", "key": "province"}},
    ]}
    r2tpl = {"round": 2,
             "key_realizations": {"QS_R2_01": "giới tính",
                                  "QS_R2_02": "tỉnh thành"},
             "templates": [
                 {"template_id": "T_A", "question_type_id": "QS_R2_01",
                  "text": "Hãy xác định [KEY]?"},
                 {"template_id": "T_B", "question_type_id": "QS_R2_02",
                  "text": "Tỉnh nào?"}]}
    r2para = {"round": 2, "paraphrases": [
        {"template_id": "T_B_P1", "source_template_id": "T_B",
         "text": "Thuộc tỉnh nào?"}]}
    # V after dedup = {QS_R2_02} only.
    r2qual = {"round": 2,
              "accepted_new": [{"type_id": "QS_R2_02",
                                "keep_template_ids": ["T_B"]}],
              "duplicates": [], "rejected": []}
    from tests.test_safety import ScriptedLLM
    fake = ScriptedLLM([style, tpl, para, qual,
                        r2style, r2tpl, r2para, r2qual,
                        _zv_s(), _zv_t()])
    monkeypatch.setattr(llm_client, "call_llm", fake)
    monkeypatch.setattr(llm_client, "create_llm_client", lambda: object())
    run_pipeline(dataset="vimd", real_llm=True, run_id="mixed-dedup")
    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "mixed-dedup"
    audit = json.loads((out_dir / "rounds" / "round_02"
                        / "signature_dedup.json").read_text(
                            encoding="utf-8"))
    assert audit["novel_type_ids"] == ["QS_R2_02"]
    assert [d["type_id"] for d in audit["deterministic_duplicates"]] == [
        "QS_R2_01"]
    # Paraphrase ran once per non-terminal round (R1+R2); R2's stored
    # paraphrases all descend from the novel type's template.
    assert fake.calls.count("paraphrase") == 2
    r2para_doc = json.loads((out_dir / "rounds" / "round_02"
                             / "03_paraphrases.json").read_text(
                                 encoding="utf-8"))
    assert {p["source_template_id"]
            for p in r2para_doc["paraphrases"]} == {"T_B"}
    record = json.loads((out_dir / "rounds" / "round_02"
                         / "04_quality.json").read_text(encoding="utf-8"))
    got = ([a["type_id"] for a in record["accepted_new"]]
           + [d["type_id"] for d in record["duplicates"]]
           + [r["type_id"] for r in record["rejected"]])
    assert got == ["QS_R2_02"]
    summary = json.loads((out_dir / "loop_summary.json").read_text(
        encoding="utf-8"))
    r2 = [r for r in summary["rounds"] if r["round"] == 2][0]
    assert r2["deterministic_duplicates"] == 1
    assert (r2["generated_types"] == r2["rejected"]
            + r2["deterministic_duplicates"] + r2["accepted_new_types"]
            + r2["duplicates"])


# §115 — Quality semantic duplicates still work on distinct signatures.
def test_115_quality_dup_distinct_signatures(isolated_root, clean_budget,
                                             monkeypatch):
    style = _payload(isolated_root, "loop", "round_01", "style")
    tpl = _payload(isolated_root, "loop", "round_01", "template")
    para = _payload(isolated_root, "loop", "round_01", "paraphrase")
    qual = _r1_quality_no_derived_keep()
    # Structurally novel (province not approved) but Quality judges it a
    # semantic duplicate of the approved gender type.
    r2style = {"round": 2, "question_types": [
        {"id": "QS_R2_01", "name": "Province Via Gender", "goal": "g",
         "uses": ["audio", "province"], "input_count": 1,
         "answer_rule": "the province label",
         "answer": {"kind": "field_value", "key": "province"}},
    ]}
    r2tpl = {"round": 2,
             "key_realizations": {"QS_R2_01": "tỉnh thành"},
             "templates": [
                 {"template_id": "T_A", "question_type_id": "QS_R2_01",
                  "text": "Tỉnh nào?"}]}
    r2para = {"round": 2, "paraphrases": []}
    r2qual = {"round": 2, "accepted_new": [],
              "duplicates": [{"type_id": "QS_R2_01",
                              "duplicate_of": "QS_R1_01"}],
              "rejected": []}
    from tests.test_safety import ScriptedLLM
    fake = ScriptedLLM([style, tpl, para, qual,
                        r2style, r2tpl, r2para, r2qual])
    monkeypatch.setattr(llm_client, "call_llm", fake)
    monkeypatch.setattr(llm_client, "create_llm_client", lambda: object())
    run_pipeline(dataset="vimd", real_llm=True, run_id="qdup")
    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "qdup"
    summary = json.loads((out_dir / "loop_summary.json").read_text(
        encoding="utf-8"))
    r2 = [r for r in summary["rounds"] if r["round"] == 2][0]
    assert r2["duplicates"] == 1
    assert r2["deterministic_duplicates"] == 0


# §116 — terminal dedup resume targets finalize with zero LLM calls.
def test_116_terminal_dedup_resume(isolated_root, clean_budget, monkeypatch):
    from src.autonomous_qa.compiler.pipeline_runner import _ResumedRun
    fx = isolated_root / "tests" / "fixtures" / "loop"
    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "term-dedup"
    raw_dir = out_dir / "raw"
    (out_dir / "rounds" / "round_01").mkdir(parents=True)
    (out_dir / "rounds" / "round_02").mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    for kind, name in (("style", "01_question_styles.json"),
                       ("template", "02_templates.json"),
                       ("paraphrase", "03_paraphrases.json")):
        (out_dir / "rounds" / "round_01" / name).write_bytes(
            (fx / "round_01" / f"{kind}.json").read_bytes())
    # R1 verdict without kept derived-[KEY] templates (control-flow only).
    (out_dir / "rounds" / "round_01" / "04_quality.json").write_text(
        json.dumps(_r1_quality_no_derived_keep()), encoding="utf-8")
    # R2 gate-valid gender type; R1 already approved gender.
    (out_dir / "rounds" / "round_02" / "01_question_styles.json"
     ).write_text(json.dumps({
         "round": 2, "question_types": [{
             "id": "QS_R2_01", "name": "Gender Again", "goal": "g",
             "uses": ["audio", "gender"], "input_count": 1,
             "answer_rule": "the gender label",
             "answer": {"kind": "field_value", "key": "gender"}}]}),
        encoding="utf-8")
    (out_dir / "rounds" / "round_02" / "02_templates.json"
     ).write_text(json.dumps({
         "round": 2, "key_realizations": {"QS_R2_01": "giới tính"},
         "templates": [{"template_id": "T_A",
                        "question_type_id": "QS_R2_01",
                        "text": "Hãy xác định [KEY]?"}]}),
        encoding="utf-8")
    (out_dir / "rounds" / "round_02" / "signature_dedup.json"
     ).write_text(json.dumps({
         "round": 2, "novel_type_ids": [],
         "deterministic_duplicates": [{
             "type_id": "QS_R2_01", "duplicate_of": "QS_R1_01",
             "signature": ["audio1", "field_value", "gender"],
             "duplicate_source": "prior_approved"}]}),
        encoding="utf-8")
    for name in ("round_01_style_attempt_01.txt",
                 "round_01_template_attempt_01.txt",
                 "paraphrase_round_1_attempt_01.txt",
                 "quality_round_1_attempt_01.txt",
                 "round_02_style_attempt_02.txt",
                 "round_02_template_attempt_02.txt"):
        (raw_dir / name).write_text("{}", encoding="utf-8")
    (out_dir / "run_state.json").write_text(json.dumps({
        "run_id": "term-dedup", "dataset": "vimd", "mode": "real",
        "completed": ["style_round_1", "template_round_1",
                      "paraphrase_round_1", "quality_round_1",
                      "style_round_2", "template_round_2"],
        "current_round": 2, "next_action": "finalize",
        "low_gain_streak": 0, "stopped": True,
        "stop_reason": "no_novel_types_after_signature_dedup",
        "spent_calls": {"question_style": 2, "question_template": 2,
                        "paraphrase": 1, "quality": 1},
        "stage_attempts": {"question_style": 2, "question_template": 2,
                           "paraphrase": 1, "quality": 1},
        "attempted_calls_total": 6, "successful_calls_total": 5,
        "last_finish_reason": "STOP", "last_failed_stage": None,
        "loop_config": {"max_rounds": 3, "min_new_type_rate": 0.15,
                        "saturation_patience": 2,
                        "immediate_stop_if_zero_new": True},
        "template_contract_version": TEMPLATE_CONTRACT_VERSION,
        "answer_reference_contract_version": "kind-scoped-v1",
        "answer_shape_contract_version": "canonical-v1",
        "executable_signature_contract_version": "v1",
        "derived_field_contract_version": "data-backed-v1",
    }), encoding="utf-8")
    from src.autonomous_qa.compiler.pipeline_runner import load_sample
    run = _ResumedRun(dataset="vimd", out_dir=out_dir, raw_dir=raw_dir,
                      rounds_dir=out_dir / "rounds", readme="",
                      rows=load_sample("vimd"), client=None)
    run.load_and_validate()
    assert run._next_action == "finalize"

    def _boom(*a, **k):
        raise AssertionError("no LLM request may be issued on resume")

    monkeypatch.setattr(llm_client, "call_llm", _boom)
    monkeypatch.setattr(llm_client, "create_llm_client", lambda: object())
    preview = run_pipeline(dataset="vimd", real_llm=True,
                           resume_run="term-dedup")
    assert len(preview) > 0


# §117 — final uniqueness preflight.
def test_117_final_uniqueness_preflight():
    import tempfile
    from src.autonomous_qa.compiler.pipeline_runner import _FreshRun
    tmp = Path(tempfile.mkdtemp())
    run = _FreshRun(dataset="vimd", run_id="t", out_dir=tmp,
                    raw_dir=tmp / "raw", rounds_dir=tmp / "rounds",
                    readme="", rows=[], client=None)
    run.approved_pool = [
        {"type_id": "QS_A", "answer": {"kind": "field_value",
                                      "key": "province"}},
        {"type_id": "QS_B", "answer": {"kind": "field_value",
                                      "key": "province"}}]
    with pytest.raises(ValueError,
                       match="final_duplicate_executable_signature"):
        run._preflight_final_signature_uniqueness()


# §118/§60 — det dups never feed dynamic few-shots.
def test_118_fewshots_exclude_det_dups():
    from src.autonomous_qa.core.loop import build_type_fewshot
    pool = [{"type_id": "QS_R1_01", "name": "Gender", "goal": "g",
             "uses": ["audio", "gender"], "answer_rule": "r"}]
    fewshots = build_type_fewshot(pool)
    assert [f["name"] for f in fewshots] == ["Gender"]


# §120 — old contract versions refuse.
def test_120_old_contract_refusal(isolated_root, clean_budget):
    from src.autonomous_qa.compiler.pipeline_runner import _ResumedRun
    out_dir = isolated_root / "outputs" / "runs" / "vimd" / "old-shape"
    raw_dir = out_dir / "raw"
    (out_dir / "rounds" / "round_01").mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    fx = isolated_root / "tests" / "fixtures" / "loop"
    (out_dir / "rounds" / "round_01" / "01_question_styles.json"
     ).write_bytes(
        (fx / "round_01" / "style.json").read_bytes())
    (raw_dir / "round_01_style_attempt_01.txt").write_text(
        "{}", encoding="utf-8")
    (out_dir / "run_state.json").write_text(json.dumps({
        "run_id": "old-shape", "dataset": "vimd", "mode": "real",
        "completed": ["style_round_1"], "current_round": 1,
        "next_action": "template_round_1", "low_gain_streak": 0,
        "stopped": False, "stop_reason": None,
        "spent_calls": {"question_style": 1, "question_template": 0,
                        "paraphrase": 0, "quality": 0},
        "loop_config": {"max_rounds": 3, "min_new_type_rate": 0.15,
                        "saturation_patience": 2,
                        "immediate_stop_if_zero_new": True},
        "template_contract_version": TEMPLATE_CONTRACT_VERSION,
        "answer_reference_contract_version": "kind-scoped-v1",
    }), encoding="utf-8")
    run = _ResumedRun(dataset="vimd", out_dir=out_dir, raw_dir=raw_dir,
                      rounds_dir=out_dir / "rounds", readme="",
                      rows=[], client=None)
    with pytest.raises(ValueError, match="contract changed"):
        run.load_and_validate()


# Versions pinned.
def test_versions_pinned():
    assert ANSWER_SHAPE_CONTRACT_VERSION == "canonical-v1"
    assert EXECUTABLE_SIGNATURE_CONTRACT_VERSION == "v1"


# FOLLOW-UP: no transcription special-case in executable signature.
def test_no_transcription_signature_branch_in_production():
    import inspect
    import src.autonomous_qa.core.validity as validity_mod
    src = inspect.getsource(validity_mod.executable_signature)
    assert "transcription" not in src
    assert 'key == "text"' not in src and "key=='text'" not in src
    assert '"text"' not in src
    assert "type.name" not in src and '["name"]' not in src
    assert '["goal"]' not in src


def test_field_value_text_signature_is_plain_field_value():
    a = _t("QS_A", {"kind": "field_value", "key": "text"},
           name="Speech Transcription",
           goal="Transcribe the spoken words",
           uses=["audio", "text"])
    b = _t("QS_B", {"kind": "field_value", "key": "text"},
           name="Literal Spoken Content Retrieval",
           goal="Return exactly what was spoken",
           uses=["audio", "text"])
    assert executable_signature(a) == ("audio1", "field_value", "text")
    assert executable_signature(b) == ("audio1", "field_value", "text")
    index = build_approved_signature_index([a])
    novel, dups = deduplicate_by_signature([b], index)
    assert novel == []
    assert dups[0]["duplicate_of"] == "QS_A"
    assert dups[0]["duplicate_source"] == "prior_approved"

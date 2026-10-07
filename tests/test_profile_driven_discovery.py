"""Profile-driven Question Style discovery: materialization, capability
policy, single-call parsing/salvage, and CLI wiring (sections B-S)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from src.autonomous_qa.production.materialize import (
    UnsupportedSplitError,
    assert_supported_split,
    canonical_row,
    materialize_selection,
    sample_stats,
    select_discovery_sample,
    selection_key,
)
from src.autonomous_qa.language.question_discovery import (
    PARSE_SALVAGED,
    aggregate_coercion_notes,
    derive_answer_rule,
    extract_json_object,
    load_style_output,
    prompt_rows,
    strip_code_fences,
)
from src.autonomous_qa.core.type_policy import (
    GROUP_METADATA_SOLVABLE,
    GROUP_UNSUPPORTED_ENCODING,
    PolicyContract,
    audio_necessity_verdict,
    audit_candidate_type,
    type_contract,
)

RAW_FIELDS = ["region", "province_code", "province_name", "filename",
              "text", "speakerID", "gender"]


def _row(province: str, filename: str, speaker: str, *, region: str = "North",
         gender: int = 1, shard: str = "train-00000-of-00001.parquet",
         row_group: int = 0, row_index: int = 0) -> dict:
    return {
        "_shard": shard,
        "_row_group": row_group,
        "_row_index": row_index,
        "region": region,
        "province_code": 84,
        "province_name": province,
        "filename": filename,
        "text": "mot cau van ban day du",
        "speakerID": speaker,
        "gender": gender,
    }


def _profile() -> dict:
    return {
        "dataset": "vimd",
        "fields": {
            "audio": {"role": "audio", "type": "audio"},
            "region": {"role": "semantic", "type": "string",
                       "is_categorical": True},
            "province_name": {"role": "semantic", "type": "string",
                              "is_categorical": True},
            "text": {"role": "semantic", "type": "string"},
            "speakerID": {"role": "hidden_identifier", "type": "string"},
            "gender": {"role": "context_only", "type": "int",
                       "is_categorical": True},
            "province_code": {"role": "provenance", "type": "int"},
            "filename": {"role": "provenance", "type": "string"},
        },
        "field_roles": {
            "region": "semantic", "province_name": "semantic",
            "text": "semantic", "speakerID": "hidden_identifier",
            "gender": "context_only", "province_code": "provenance",
            "filename": "provenance", "audio": "audio",
        },
        "context_visibilities": {
            "audio": "visible", "region": "visible",
            "province_name": "visible", "text": "visible",
            "speakerID": "hidden", "gender": "visible",
            "province_code": "visible", "filename": "hidden",
        },
        "hidden_fields": ["speakerID"],
        "semantic_fields": ["region", "province_name", "text"],
        "relations": [],
        "qa_constraints": [],
    }


def _qtype(uses, answer, *, input_context_fields=None, input_count=1,
           answer_rule="gold is the declared field", type_id="QS_T1"):
    return {
        "id": type_id,
        "name": "A type",
        "goal": "A goal",
        "uses": uses,
        "input_count": input_count,
        "answer_rule": answer_rule,
        "answer": answer,
        "input_context_fields": input_context_fields or [],
    }


# -- section E/F: selection + overlap ------------------------------------
def test_split_gate_refuses_test_and_unknown():
    assert_supported_split("train")
    assert_supported_split("valid")
    with pytest.raises(UnsupportedSplitError):
        assert_supported_split("test")
    with pytest.raises(UnsupportedSplitError):
        assert_supported_split("dev")


def test_selection_is_one_row_per_province_and_deterministic():
    train = [
        _row("HaNoi", "h1.wav", "s1"),
        _row("HaNoi", "h2.wav", "s2"),
        _row("CaMau", "c1.wav", "s3", region="South"),
        _row("DaNang", "d1.wav", "s4", region="Central"),
    ]
    valid = [_row("HaNoi", "v1.wav", "s5", shard="valid-00000-of-00001.parquet")]
    first = materialize_selection(train, valid, is_usable=lambda r: True)
    second = materialize_selection(train, valid, is_usable=lambda r: True)
    assert len(first["train"]) == 3
    assert len(first["valid"]) == 1
    assert ([r["filename"] for r in first["train"]]
            == [r["filename"] for r in second["train"]])
    assert {r["province_name"] for r in first["train"]} == {
        "HaNoi", "CaMau", "DaNang"}
    assert first["train_advances"] == second["train_advances"]


def _ranked_names(split: str, names: list[str]) -> list[str]:
    return sorted(names, key=lambda n: (selection_key(split, n), n))


def test_valid_advances_past_train_speaker():
    first, second = _ranked_names("valid", ["vA.wav", "vB.wav"])
    train = [_row("HaNoi", "train_only.wav", "shared_speaker")]
    valid = [
        _row("HaNoi", first, "shared_speaker",
             shard="valid-00000-of-00001.parquet"),
        _row("HaNoi", second, "fresh_speaker",
             shard="valid-00000-of-00001.parquet"),
    ]
    sel = materialize_selection(train, valid, is_usable=lambda r: True)
    assert sel["valid"][0]["filename"] == second
    assert sel["valid_advances"]["HaNoi"] == 1
    assert any(r["reason"] == "speaker_overlap"
               for r in sel["rejections"])


def test_valid_advances_past_shared_filename():
    first, second = _ranked_names("valid", ["vC.wav", "vD.wav"])
    train = [_row("HaNoi", first, "train_speaker")]
    valid = [
        _row("HaNoi", first, "other_speaker",
             shard="valid-00000-of-00001.parquet"),
        _row("HaNoi", second, "fresh_speaker",
             shard="valid-00000-of-00001.parquet"),
    ]
    sel = materialize_selection(train, valid, is_usable=lambda r: True)
    assert sel["valid"][0]["filename"] == second
    assert sel["valid_advances"]["HaNoi"] == 1
    assert any(r["reason"] == "filename_overlap"
               for r in sel["rejections"])


def test_unusable_audio_advances_to_next_candidate():
    train = [
        _row("HaNoi", "bad.wav", "s1"),
        _row("HaNoi", "good.wav", "s2"),
    ]
    valid = [_row("HaNoi", "v1.wav", "s3",
                  shard="valid-00000-of-00001.parquet")]
    sel = materialize_selection(
        train, valid, is_usable=lambda r: r["filename"] != "bad.wav")
    assert sel["train"][0]["filename"] == "good.wav"
    assert sel["train_advances"]["HaNoi"] == 1
    assert any(r["reason"] == "audio_unusable" for r in sel["rejections"])


# -- section K: discovery sample -----------------------------------------
def _train_pool() -> list[dict]:
    rows = []
    regions = [("North", ["HaNoi", "HaiDuong", "NamDinh"]),
               ("Central", ["DaNang", "QuangNam", "Hue"]),
               ("South", ["CaMau", "TraVinh", "CanTho"])]
    for region, provinces in regions:
        for p_idx, province in enumerate(provinces):
            for n in range(4):
                rows.append(_row(
                    province, f"{province}_{n}.wav", f"spk_{province}_{n}",
                    region=region, gender=n % 2,
                    row_index=len(rows),
                ))
    return rows


def test_discovery_sample_is_train_only_balanced_and_deterministic():
    rows = _train_pool()
    sample = select_discovery_sample(rows, target=12, split="train")
    again = select_discovery_sample(rows, target=12, split="train")
    assert len(sample) == 12
    assert [r["filename"] for r in sample] == [r["filename"] for r in again]
    assert {r["region"] for r in sample} == {"North", "Central", "South"}
    assert len({r["province_name"] for r in sample}) == 9
    assert len({r["speakerID"] for r in sample}) == 12
    with pytest.raises(UnsupportedSplitError):
        select_discovery_sample(rows, target=4, split="valid")
    with pytest.raises(UnsupportedSplitError):
        select_discovery_sample(rows, target=4, split="test")


def test_canonical_row_preserves_raw_source_field_names():
    row = _row("TraVinh", "84_0104.wav", "spk_1", gender=0)
    envelope = canonical_row("vimd", "train", row, "/tmp/x.wav")
    assert envelope["split"] == "train"
    assert envelope["sample_id"] == "vimd:train:84_0104.wav"
    assert sorted(envelope["metadata"]) == sorted(RAW_FIELDS)
    assert envelope["metadata"]["gender"] == 0
    assert envelope["metadata"]["speakerID"] == "spk_1"
    with pytest.raises(UnsupportedSplitError):
        canonical_row("vimd", "test", row, "/tmp/x.wav")


def test_sample_stats_reports_raw_gender_codes_and_regions():
    rows = [
        _row("HaNoi", "a.wav", "s1", gender=0),
        _row("HaNoi", "b.wav", "s2", gender=1),
        _row("CaMau", "c.wav", "s3", region="South", gender=1),
    ]
    stats = sample_stats(rows)
    assert stats["rows"] == 3
    assert stats["unique_provinces"] == 2
    assert stats["raw_gender_code_distribution"] == {"0": 1, "1": 2}
    assert stats["region_distribution"] == {"North": 2, "South": 1}
    assert stats["duplicate_speakers"] == []


def test_selection_key_is_stable_sha256():
    key = selection_key("train", "84_0104.wav")
    assert key == selection_key("train", "84_0104.wav")
    assert len(key) == 64
    assert key != selection_key("valid", "84_0104.wav")


# -- section B-D: capability policy --------------------------------------
def test_contract_derives_roles_visibilities_and_unresolved_encoding():
    contract = PolicyContract.from_profile(_profile())
    assert contract.field_roles()["speakerID"] == "hidden_identifier"
    assert contract.context_visibilities()["speakerID"] == "hidden"
    assert sorted(contract.hidden_fields) == ["speakerID"]
    assert "gender" in contract.unresolved_encoded_fields
    assert "region" not in contract.unresolved_encoded_fields
    assert contract.dataset == "vimd"


def test_type_contract_answers_the_four_questions():
    contract = PolicyContract.from_profile(_profile())
    qtype = _qtype(["audio", "region"], {"kind": "field_value",
                                         "key": "region"})
    view = type_contract(qtype, contract)
    assert view["audio_required"] is True
    assert view["visible_context_fields"] == []
    assert view["gold_source_fields"] == ["region"]
    assert view["uses_fields"] == ["audio", "region"]
    assert view["answer_kind"] == "field_value"

    no_audio = _qtype(["region"], {"kind": "field_value", "key": "region"})
    assert type_contract(no_audio, contract)["audio_required"] is False


def test_metadata_solvable_gold_in_visible_context_is_rejected():
    contract = PolicyContract.from_profile(_profile())
    qtype = _qtype(
        ["audio", "province_name"],
        {"kind": "field_value", "key": "province_name"},
        input_context_fields=["province_name"],
    )
    rec = audit_candidate_type(qtype, contract, dataset_rows=[])
    assert rec["metadata_solvable"] is True
    assert rec["status"] == "REJECTED"
    assert any(r.startswith("metadata_solvable") for r in rec["reasons"])
    assert rec["primary_reason_group"] == GROUP_METADATA_SOLVABLE
    assert audio_necessity_verdict(rec)["answer_without_audio"] == "YES"


def test_metadata_solvable_via_declared_relation_is_rejected():
    profile = _profile()
    profile["relations"] = [{"source_field": "province_name",
                             "target_field": "region"}]
    contract = PolicyContract.from_profile(profile)
    qtype = _qtype(
        ["audio", "province_name"],
        {"kind": "field_value", "key": "region"},
        input_context_fields=["province_name"],
    )
    rec = audit_candidate_type(qtype, contract, dataset_rows=[])
    assert rec["metadata_solvable"] is True
    assert rec["status"] == "REJECTED"
    assert rec["primary_reason_group"] == GROUP_METADATA_SOLVABLE


def test_visible_text_context_is_flagged_as_audio_bypass():
    contract = PolicyContract.from_profile(_profile())
    qtype = _qtype(
        ["audio", "text"],
        {"kind": "field_value", "key": "region"},
        input_context_fields=["text"],
    )
    rec = audit_candidate_type(qtype, contract, dataset_rows=[])
    assert rec["status"] in ("REVIEW_REQUIRED", "REJECTED")
    assert any(f.startswith("context_channel_bypasses_audio:")
               for f in rec["review_flags"] + rec["reasons"])


def test_unsupported_encoding_gold_is_rejected():
    contract = PolicyContract.from_profile(_profile())
    qtype = _qtype(["audio", "gender"],
                   {"kind": "field_value", "key": "gender"})
    rec = audit_candidate_type(qtype, contract, dataset_rows=[])
    assert rec["status"] == "REJECTED"
    assert rec["primary_reason_group"] == GROUP_UNSUPPORTED_ENCODING
    verdict = audio_necessity_verdict(rec)
    assert verdict["answer_without_audio"] == "NO"


def test_audio_necessity_verdict_for_metadata_only_type():
    contract = PolicyContract.from_profile(_profile())
    qtype = _qtype(["province_name", "region"],
                   {"kind": "derived_field", "source_key": "province_name",
                    "target_key": "region"})
    rec = audit_candidate_type(qtype, contract, dataset_rows=[])
    verdict = audio_necessity_verdict(rec)
    assert verdict["answer_without_audio"] in ("YES", "PARTIAL")
    assert verdict["classification"] == rec["status"]


# -- sections L-O: single response parsing / salvage ----------------------
def test_strip_code_fences_removes_wrapper():
    text = '```json\n{"a": 1}\n```'
    assert json.loads(strip_code_fences(text)) == {"a": 1}
    assert strip_code_fences('{"a": 1}') == '{"a": 1}'


def test_extract_json_object_finds_payload_inside_prose():
    text = 'noise before {"round": 1, "q": [1, 2]} after'
    assert extract_json_object(text) == {"round": 1, "q": [1, 2]}
    with pytest.raises(ValueError):
        extract_json_object("no json at all")


@pytest.mark.parametrize("answer,fragment", [
    ({"kind": "field_value", "key": "region"}, "region"),
    ({"kind": "equality", "keys": ["speakerID_1", "speakerID_2"]},
     "speakerID_1 equals speakerID_2"),
    ({"kind": "derived_field", "source_key": "province_name",
      "target_key": "region"}, "province_name"),
])
def test_derive_answer_rule_covers_kinds(answer, fragment):
    rule = derive_answer_rule(answer)
    assert rule is not None
    assert fragment in rule


def test_load_style_output_salvages_fenced_candidates_payload(tmp_path):
    payload = {
        "candidates": [
            {"type_id": "QS_R1_01", "type": "ASR",
             "uses": ["audio", "text"], "input_count": 1,
             "answer": {"kind": "field_value", "key": "text"},
             "insight": "transcribe"},
            {"type_id": "QS_R1_02", "type": "Macro Dialect",
             "uses": ["audio", "region"], "input_count": 1,
             "answer": {"kind": "field_value", "key": "region"},
             "insight": "region"},
        ],
    }
    (tmp_path / "question_style_raw_response_attempt_01.txt").write_text(
        "```json\n" + json.dumps(payload) + "\n```", encoding="utf-8")

    out, status, notes, raw = load_style_output(tmp_path, round_idx=1)
    assert status == PARSE_SALVAGED
    assert out["round"] == 1
    assert [t["id"] for t in out["question_types"]] == [
        "QS_R1_01", "QS_R1_02"]
    assert out["question_types"][0]["name"] == "ASR"
    assert out["question_types"][0]["goal"] == "transcribe"
    assert out["question_types"][0]["answer_rule"]
    rules = {(n["rule"], n.get("from")) for n in notes}
    assert ("envelope:question_types", "candidates") in rules
    assert ("alias:goal", "insight") in rules
    assert ("derive:answer_rule", "answer") in rules
    assert raw.strip().startswith("```json")


def test_coercion_notes_are_aggregated_with_type_ids():
    notes = [
        {"rule": "alias:goal", "from": "insight", "type_id": "A"},
        {"rule": "alias:goal", "from": "insight", "type_id": "B"},
        {"rule": "strip_code_fences"},
    ]
    agg = aggregate_coercion_notes(notes)
    by_rule = {n["rule"]: n for n in agg}
    assert by_rule["alias:goal"]["count"] == 2
    assert by_rule["alias:goal"]["type_ids"] == ["A", "B"]
    assert by_rule["strip_code_fences"]["count"] == 1
    assert "type_ids" not in by_rule["strip_code_fences"]


def test_prompt_rows_expose_only_raw_metadata(tmp_path):
    row = {
        "dataset": "vimd", "split": "train",
        "sample_id": "vimd:train:a.wav",
        "audio_path": str(tmp_path / "a.wav"),
        "metadata": {"region": "North", "gender": 1},
    }
    shown = prompt_rows([row])
    assert shown == [{"split": "train",
                      "metadata": {"region": "North", "gender": 1}}]


def test_no_persisted_response_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_style_output(tmp_path, round_idx=1)


# -- sections L/O: TEST refusal + CLI wiring -----------------------------
def test_question_discovery_refuses_test_split_sample(tmp_path):
    from src.autonomous_qa.language.question_discovery import run_question_discovery

    profile_path = tmp_path / "dataset_profile.json"
    profile_path.write_text(json.dumps(_profile()), encoding="utf-8")
    sample_path = tmp_path / "sample_for_discovery.jsonl"
    sample_path.write_text(
        json.dumps({"dataset": "vimd", "split": "test",
                    "sample_id": "vimd:test:a.wav",
                    "audio_path": str(tmp_path / "a.wav"),
                    "metadata": {"region": "North"}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="discovery_sample_split_not_allowed"):
        run_question_discovery(
            dataset="vimd",
            dataset_profile=profile_path,
            sample_manifest=sample_path,
            out_dir=tmp_path / "out",
        )


def test_stop_after_requires_profile_and_sample_manifest(monkeypatch):
    from src.autonomous_qa.compiler import run_pipeline_cli as run_pipeline

    monkeypatch.setattr(sys, "argv", [
        "run_pipeline", "--stop-after", "question_style"])
    with pytest.raises(SystemExit) as exc:
        run_pipeline.main()
    assert exc.value.code == 2


def test_stop_after_requires_real_llm(monkeypatch, tmp_path):
    from src.autonomous_qa.compiler import run_pipeline_cli as run_pipeline

    monkeypatch.setattr(sys, "argv", [
        "run_pipeline", "--stop-after", "question_style",
        "--dataset-profile", str(tmp_path / "p.json"),
        "--sample-manifest", str(tmp_path / "s.jsonl"),
    ])
    with pytest.raises(SystemExit) as exc:
        run_pipeline.main()
    assert exc.value.code == 2


def test_dataset_profile_flags_require_stop_after(monkeypatch, tmp_path):
    from src.autonomous_qa.compiler import run_pipeline_cli as run_pipeline

    monkeypatch.setattr(sys, "argv", [
        "run_pipeline", "--dataset-profile", str(tmp_path / "p.json")])
    with pytest.raises(SystemExit) as exc:
        run_pipeline.main()
    assert exc.value.code == 2


# -- section E: materializer script helpers ------------------------------
def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / \
        "autonomous_qa" / "materialize_vimd_discovery_sample.py"
    spec = importlib.util.spec_from_file_location("materialize_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audio_out_name_never_doubles_extension():
    module = _load_script_module()
    assert module.audio_out_name("84_0104.wav", ".wav") == "84_0104.wav"
    assert module.audio_out_name("84_0104.flac", ".wav") == "84_0104.flac"
    assert module.audio_out_name("84_0104", ".wav") == "84_0104.wav"


def test_script_never_lists_test_shards():
    module = _load_script_module()
    source = (Path(__file__).resolve().parents[1] / "scripts" /
              "autonomous_qa" / "materialize_vimd_discovery_sample.py").read_text(encoding="utf-8")
    assert '"train", "valid"' in source or "'train', 'valid'" in source
    assert module.ALLOWED_SPLITS == ("train", "valid")
    assert "test" not in module.ALLOWED_SPLITS

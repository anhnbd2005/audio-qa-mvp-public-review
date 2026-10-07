"""Focused contracts for SAUVI P2 (speaker gender recognition) training QA.

Covers the common two-choice contract, deterministic template/choice order,
model-facing metadata exclusion, ViMD and Speech-MASSIVE label mapping and
conflict/missing policies, source isolation, the deterministic combined union,
zero-LLM generation and byte-identical double runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common.config import ROOT
from src.common.zero_llm import scan_modules_for_forbidden_calls
from src.sauvi_perception.tasks.p2_speaker_gender.speaker_gender_p2 import (
    FEMALE_LABEL,
    MALE_LABEL,
    SM_CONFIG,
    SM_SPLIT,
    P2SourceError,
    audit_ok,
    build_sm_qa,
    build_vimd_qa,
    combined_union,
    load_policy,
    load_sm_provenance,
    sm_consistency,
    sm_eligible_rows,
    to_model_facing,
    vimd_consistency,
    vimd_eligible_rows,
    write_combined,
    write_release,
)

VIMD_OUT = ROOT / "outputs" / "training_qa" / "vimd" / "p2" / "current"
SM_OUT = ROOT / "outputs" / "training_qa" / "speech_massive_vi" / "p2" / "current"
COMBINED_OUT = ROOT / "outputs" / "training_qa" / "combined" / "p2" / "current"
POLICY_PATH = ROOT / "resources" / "semantics" / "p2_gender_policy.json"


@pytest.fixture()
def policy() -> dict:
    return load_policy(POLICY_PATH)


def _vimd_row(filename: str, speaker: str, gender) -> dict:
    return {
        "filename": filename,
        "speakerID": speaker,
        "gender": gender,
        "region": "North",
        "province_name": "HaNoi",
    }


def _sm_row(row_id: str, speaker: str, sex: str, validated: bool = True) -> dict:
    return {
        "id": row_id,
        "locale": "vi-VN",
        "partition": "train",
        "path": f"train-115/{row_id}.wav",
        "is_validated": validated,
        "speaker_id": speaker,
        "speaker_sex": sex,
    }


# --- Common -----------------------------------------------------------------


def test_two_choices_and_valid_answer(policy):
    records = build_vimd_qa(
        [
            {
                "source_row_id": "a.wav",
                "audio_id": "vimd/train/a.wav",
                "speaker_id": "s1",
                "source_label": "male",
                "gold": MALE_LABEL,
            }
        ],
        policy,
    )
    record = records[0]
    assert record["choices"] in ([MALE_LABEL, FEMALE_LABEL], [FEMALE_LABEL, MALE_LABEL])
    assert len(record["choices"]) == 2
    assert record["answer"] in record["choices"]
    assert record["answer"] == MALE_LABEL


def test_template_and_choice_order_deterministic(policy):
    anchors = [
        {
            "source_row_id": "a.wav",
            "audio_id": "vimd/train/a.wav",
            "speaker_id": "s1",
            "source_label": "male",
            "gold": MALE_LABEL,
        }
    ]
    assert build_vimd_qa(anchors, policy) == build_vimd_qa(anchors, policy)


def test_model_facing_metadata_exclusion(policy):
    records = build_vimd_qa(
        [
            {
                "source_row_id": "a.wav",
                "audio_id": "vimd/train/a.wav",
                "speaker_id": "s1",
                "source_label": "female",
                "gold": FEMALE_LABEL,
            }
        ],
        policy,
    )
    model = to_model_facing(records[0])
    assert set(model) == {"id", "task", "audio_id", "question", "choices", "answer"}
    serialized = json.dumps(model, ensure_ascii=False)
    assert "speaker" not in serialized
    assert "gender" not in serialized.lower()
    assert "sex" not in serialized.lower()
    assert "path" not in serialized
    assert "revision" not in serialized


# --- ViMD -------------------------------------------------------------------


def test_vimd_gender_mapping(policy):
    rows = [
        _vimd_row("a.wav", "s1", 1),
        _vimd_row("b.wav", "s2", 0),
    ]
    consistency = vimd_consistency(rows)
    anchors = vimd_eligible_rows(rows, set(consistency["conflicting_speaker_ids"]))
    records = build_vimd_qa(anchors, policy)
    answers = {r["source_row_id"]: r["answer"] for r in records}
    assert answers["a.wav"] == MALE_LABEL
    assert answers["b.wav"] == FEMALE_LABEL


def test_vimd_conflict_drops_entire_speaker(policy):
    rows = [
        _vimd_row("m1.wav", "s1", 1),
        _vimd_row("f1.wav", "s1", 0),
        _vimd_row("m2.wav", "s1", 1),
        _vimd_row("ok.wav", "s2", 1),
    ]
    consistency = vimd_consistency(rows)
    assert consistency["conflicting_speakers"] == 1
    assert consistency["conflicting_speaker_ids"] == ["s1"]
    assert consistency["rows_removed_by_conflict"] == 3
    anchors = vimd_eligible_rows(rows, set(consistency["conflicting_speaker_ids"]))
    assert [a["source_row_id"] for a in anchors] == ["ok.wav"]


def test_vimd_missing_row_excluded_without_propagation(policy):
    rows = [
        _vimd_row("m1.wav", "s1", 1),
        _vimd_row("m2.wav", "s1", None),
        _vimd_row("m3.wav", "s1", ""),
    ]
    consistency = vimd_consistency(rows)
    assert consistency["conflicting_speakers"] == 0
    anchors = vimd_eligible_rows(rows, set())
    assert [a["source_row_id"] for a in anchors] == ["m1.wav"]
    assert anchors[0]["gold"] == MALE_LABEL


# --- Speech-MASSIVE ---------------------------------------------------------


def test_sm_sex_mapping(policy):
    rows = [
        _sm_row("1", "s1", "Male"),
        _sm_row("2", "s2", "Female"),
    ]
    consistency = sm_consistency(rows)
    anchors = sm_eligible_rows(rows, set(consistency["conflicting_speaker_ids"]))
    records = build_sm_qa(anchors, policy)
    answers = {r["source_row_id"]: r["answer"] for r in records}
    assert answers["1"] == MALE_LABEL
    assert answers["2"] == FEMALE_LABEL


def test_sm_unidentified_excluded(policy):
    rows = [
        _sm_row("1", "s1", "Male"),
        _sm_row("2", "s1", "Unidentified"),
    ]
    consistency = sm_consistency(rows)
    anchors = sm_eligible_rows(rows, set())
    assert [a["source_row_id"] for a in anchors] == ["1"]
    assert anchors[0]["gold"] == MALE_LABEL
    assert consistency["unidentified_rows"] == 1


def test_sm_conflict_drops_entire_speaker(policy):
    rows = [
        _sm_row("1", "s1", "Male"),
        _sm_row("2", "s1", "Female"),
        _sm_row("3", "s2", "Female"),
    ]
    consistency = sm_consistency(rows)
    assert consistency["conflicting_speaker_ids"] == ["s1"]
    anchors = sm_eligible_rows(rows, set(consistency["conflicting_speaker_ids"]))
    assert [a["source_row_id"] for a in anchors] == ["3"]


def test_sm_unvalidated_excluded(policy):
    rows = [
        _sm_row("1", "s1", "Male", validated=True),
        _sm_row("2", "s1", "Male", validated=False),
    ]
    anchors = sm_eligible_rows(rows, set())
    assert [a["source_row_id"] for a in anchors] == ["1"]


def test_sm_provenance_is_vi_vn_train_115():
    prov = load_sm_provenance()
    assert prov["config"] == SM_CONFIG == "vi-VN"
    assert prov["split"] == SM_SPLIT == "train_115"
    assert "validation" in prov["forbidden_splits"]
    assert "test" in prov["forbidden_splits"]


def test_sm_loader_rejects_non_train_115(tmp_path):
    from src.sauvi_perception.tasks.p2_speaker_gender.speaker_gender_p2 import load_sm_rows

    with pytest.raises(P2SourceError):
        load_sm_rows(tmp_path / "validation-00000.parquet")
    with pytest.raises(P2SourceError):
        load_sm_rows(tmp_path / "test-00000.parquet")


# --- Combined ---------------------------------------------------------------


def _both_sources(policy) -> tuple[list[dict], list[dict]]:
    vrows = [_vimd_row("a.wav", "v1", 1), _vimd_row("b.wav", "v2", 0)]
    vcons = vimd_consistency(vrows)
    vqa = build_vimd_qa(
        vimd_eligible_rows(vrows, set(vcons["conflicting_speaker_ids"])), policy
    )
    srows = [_sm_row("1", "s1", "Male"), _sm_row("2", "s2", "Female")]
    scons = sm_consistency(srows)
    sqa = build_sm_qa(
        sm_eligible_rows(srows, set(scons["conflicting_speaker_ids"])), policy
    )
    return vqa, sqa


def test_combined_unique_ids_and_count(policy):
    vqa, sqa = _both_sources(policy)
    combined = combined_union(vqa, sqa)
    assert len(combined) == len(vqa) + len(sqa)
    ids = [r["id"] for r in combined]
    audio_ids = [r["audio_id"] for r in combined]
    assert len(set(ids)) == len(ids)
    assert len(set(audio_ids)) == len(audio_ids)
    assert all(
        r["audio_id"].startswith(("vimd/", "speech_massive_vi/")) for r in combined
    )


def test_combined_never_regenerates_gold(policy):
    vqa, sqa = _both_sources(policy)
    combined = combined_union(vqa, sqa)
    source_by_id = {r["id"]: r for r in (*vqa, *sqa)}
    for record in combined:
        original = source_by_id[record["id"]]
        assert record["answer"] == original["answer"]
        assert (
            record[
                "source_gender" if "source_gender" in original else "source_speaker_sex"
            ]
            == original[
                "source_gender" if "source_gender" in original else "source_speaker_sex"
            ]
        )


def test_combined_order_deterministic(policy):
    vqa, sqa = _both_sources(policy)
    assert combined_union(vqa, sqa) == combined_union(vqa, sqa)


def test_combined_source_contribution(policy, tmp_path):
    vqa, sqa = _both_sources(policy)
    combined = combined_union(vqa, sqa)
    result = write_combined(
        tmp_path / "combined", combined, vimd_count=len(vqa), sm_count=len(sqa)
    )
    assert result["contribution"]["combined_total"] == len(vqa) + len(sqa)
    assert result["manifest"]["audit_status"] == "PASS"


# --- Global -----------------------------------------------------------------


def test_write_release_byte_identical_double_run(tmp_path, policy):
    vqa, _sqa = _both_sources(policy)
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    write_release(dir_a, vqa, source_label="ViMD")
    write_release(dir_b, vqa, source_label="ViMD")
    for name in ("qa_internal.jsonl", "qa_model_facing.jsonl", "audit.json"):
        assert (dir_a / name).read_bytes() == (dir_b / name).read_bytes()


def test_zero_llm_module_scan():
    result = scan_modules_for_forbidden_calls(modules=("src/sauvi_perception/tasks/p2_speaker_gender/speaker_gender_p2.py",))
    assert result["status"] == "PASS"
    assert result["llm_calls"] == 0


def test_generator_source_has_no_llm_or_sauvi_manifest():
    text = (ROOT / "src" / "sauvi_perception" / "tasks" / "p2_speaker_gender" / "speaker_gender_p2.py").read_text(encoding="utf-8")
    assert "manifest_v2" not in text
    assert "openai" not in text
    assert "llm_client" not in text


# --- Full generated artifacts ----------------------------------------------


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_vimd_release_artifacts_pass():
    if not (VIMD_OUT / "qa_internal.jsonl").is_file():
        pytest.skip("ViMD P2 release not generated")
    audit = _load(VIMD_OUT / "audit.json")
    assert audit_ok(audit)
    assert audit["counts"]["qa"] == 14_995
    assert audit["counts"]["male_rows"] == 10_800
    assert audit["counts"]["female_rows"] == 4_195
    consistency = _load(VIMD_OUT / "gender_consistency_audit.json")
    assert consistency["conflicting_speakers"] == 12
    assert consistency["rows_removed_by_conflict"] == 28


def test_speech_massive_release_artifacts_pass():
    if not (SM_OUT / "qa_internal.jsonl").is_file():
        pytest.skip("Speech-MASSIVE P2 release not generated")
    audit = _load(SM_OUT / "audit.json")
    assert audit_ok(audit)
    assert audit["counts"]["qa"] == 114
    assert audit["counts"]["male_rows"] == 40
    assert audit["counts"]["female_rows"] == 74
    consistency = _load(SM_OUT / "speaker_sex_consistency_audit.json")
    assert consistency["rows"] == 115
    assert consistency["unidentified_rows"] == 1


def test_combined_release_artifacts_pass():
    if not (COMBINED_OUT / "qa_internal.jsonl").is_file():
        pytest.skip("Combined P2 release not generated")
    independent = _load(COMBINED_OUT / "independent_audit.json")
    assert independent["qa_count"] == 15_109
    assert independent["unique_qa_ids"] == 15_109
    assert independent["unique_audio_ids"] == 15_109
    assert independent["wrong_gold"] == 0
    assert independent["model_facing_schema_ok"] is True

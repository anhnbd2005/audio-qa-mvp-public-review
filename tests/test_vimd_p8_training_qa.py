"""Focused contracts for SAUVI P8 TRAINING QA generated from ViMD TRAIN.

Covers the frozen ontology, the three granularities, deterministic choice /
distractor / template behavior, TRAIN-only isolation, the absence of any
absolute path or speaker metadata in model-facing QA, zero-LLM generation,
and byte-identical double runs.
"""

from __future__ import annotations

import json

import pytest

from src.common.config import ROOT
from src.autonomous_qa.production.production_zero_llm import scan_production_zero_llm
from src.sauvi_perception.tasks.p8_dialect.vimd_p8_training_qa import (
    P8SourceError,
    audit_ok,
    build_qa,
    generate_artifacts,
    load_ontology,
    load_train_rows,
    select_smoke_rows,
    to_model_facing,
)

CURRENT = ROOT / "outputs" / "training_qa" / "vimd" / "p8" / "current"
ONTOLOGY_PATH = ROOT / "resources" / "semantics" / "p8_ontology.json"

# Expected ontology, authored independently of the frozen resource.
EXPECTED_REGION8 = {
    "đông bắc": [
        "BacGiang",
        "BacKan",
        "CaoBang",
        "HaGiang",
        "LangSon",
        "QuangNinh",
        "ThaiNguyen",
        "TuyenQuang",
    ],
    "tây bắc": [
        "HoaBinh",
        "LaiChau",
        "LaoCai",
        "PhuTho",
        "SonLa",
        "YenBai",
        "DienBien",
    ],
    "đồng bằng sông hồng": [
        "BacNinh",
        "HaNam",
        "HaNoi",
        "HungYen",
        "HaiDuong",
        "HaiPhong",
        "NamDinh",
        "NinhBinh",
        "ThaiBinh",
        "VinhPhuc",
    ],
    "bắc trung bộ": [
        "ThanhHoa",
        "NgheAn",
        "HaTinh",
        "QuangBinh",
        "QuangTri",
        "ThuaThienHue",
    ],
    "duyên hải nam trung bộ": [
        "DaNang",
        "QuangNam",
        "QuangNgai",
        "BinhDinh",
        "PhuYen",
        "KhanhHoa",
        "NinhThuan",
        "BinhThuan",
    ],
    "tây nguyên": ["KonTum", "GiaLai", "DakLak", "DakNong", "LamDong"],
    "đông nam bộ": [
        "BinhPhuoc",
        "TayNinh",
        "BinhDuong",
        "DongNai",
        "BaRiaVungTau",
        "HoChiMinh",
    ],
    "đồng bằng sông cửu long": [
        "LongAn",
        "TienGiang",
        "BenTre",
        "TraVinh",
        "VinhLong",
        "DongThap",
        "AnGiang",
        "KienGiang",
        "CanTho",
        "HauGiang",
        "SocTrang",
        "BacLieu",
        "CaMau",
    ],
}

EXPECTED_MACRO = {"North": "bắc", "Central": "trung", "South": "nam"}


@pytest.fixture()
def ontology() -> dict:
    return load_ontology(ONTOLOGY_PATH)


def _row(filename: str, region: str, province: str) -> dict:
    return {"filename": filename, "region": region, "province_name": province}


@pytest.fixture()
def sample_rows() -> list[dict]:
    return [
        _row("a.wav", "North", "HaNoi"),
        _row("b.wav", "Central", "DaNang"),
        _row("c.wav", "South", "CaMau"),
        _row("d.wav", "North", "CaoBang"),
        _row("e.wav", "Central", "DakLak"),
        _row("f.wav", "South", "HoChiMinh"),
    ]


# --- Ontology ---------------------------------------------------------------


def test_macro_region_mapping(ontology):
    assert ontology["macro_region_label_map"] == EXPECTED_MACRO


def test_all_63_provinces_covered_and_mapped_exactly_once(ontology):
    p2r = ontology["province_to_region8"]
    assert len(p2r) == 63
    expected_flat = {
        province: region
        for region, provinces in EXPECTED_REGION8.items()
        for province in provinces
    }
    assert len(expected_flat) == 63
    assert p2r == expected_flat
    assert set(ontology["region8_labels"]) == set(EXPECTED_REGION8)


def test_province_display_is_complete_and_lowercase(ontology):
    display = ontology["province_display"]
    assert set(display) == set(ontology["province_to_region8"])
    for label in display.values():
        assert label == label.lower()
        assert label.strip() == label


# --- Choice construction ----------------------------------------------------


def test_l1_three_choices_l2_l3_four(sample_rows, ontology):
    records = build_qa(sample_rows, ontology)
    by_level = {level: [] for level in ("macro_region", "region8", "province")}
    for record in records:
        by_level[record["level"]].append(record)
    assert len(by_level["macro_region"]) == len(sample_rows)
    assert len(by_level["region8"]) == len(sample_rows)
    assert len(by_level["province"]) == len(sample_rows)
    for record in by_level["macro_region"]:
        assert len(record["choices"]) == 3
    for level in ("region8", "province"):
        for record in by_level[level]:
            assert len(record["choices"]) == 4
    for record in records:
        assert record["answer"] in record["choices"]
        assert len(set(record["choices"])) == len(record["choices"])


def test_gold_matches_source_labels(sample_rows, ontology):
    records = build_qa(sample_rows, ontology)
    for record in records:
        province = record["source_province"]
        if record["level"] == "macro_region":
            assert record["answer"] == EXPECTED_MACRO[record["source_region"]]
        elif record["level"] == "region8":
            assert record["answer"] == ontology["province_to_region8"][province]
        else:
            assert record["answer"] == ontology["province_display"][province]


def test_gold_origin_and_tier_per_level(sample_rows, ontology):
    records = build_qa(sample_rows, ontology)
    expected = {
        "macro_region": ("SOURCE", "T1_PERCEPTION"),
        "region8": ("DERIVED_SOURCE", "T2_DERIVED"),
        "province": ("SOURCE", "T1_PERCEPTION"),
    }
    for record in records:
        gold_origin, tier = expected[record["level"]]
        assert record["gold_origin"] == gold_origin
        assert record["compiler_tier"] == tier
        assert record["benchmark_category"] == "perception"


# --- Determinism ------------------------------------------------------------


def test_generation_is_deterministic(sample_rows, ontology):
    first = build_qa(sample_rows, ontology)
    second = build_qa(sample_rows, ontology)
    assert first == second


def test_distractor_and_order_ranking_is_input_order_independent():
    from src.sauvi_perception.tasks.p8_dialect.vimd_p8_training_qa import _ranked

    assert _ranked(["a", "b", "c"], "salt") == _ranked(["c", "a", "b"], "salt")


def test_different_seed_changes_some_derived_choice(sample_rows, ontology):
    import copy

    other = copy.deepcopy(ontology)
    other["seed"] = 7
    base = build_qa(sample_rows, ontology)
    changed = build_qa(sample_rows, other)
    assert base != changed


def test_byte_identical_double_run(tmp_path, sample_rows, ontology):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    generate_artifacts(sample_rows, ontology, dir_a)
    generate_artifacts(sample_rows, ontology, dir_b)
    for name in ("qa_internal.jsonl", "qa_model_facing.jsonl"):
        assert (dir_a / name).read_bytes() == (dir_b / name).read_bytes()


# --- Isolation / leakage ----------------------------------------------------


def test_loader_refuses_non_train_split_names(tmp_path):
    with pytest.raises(P8SourceError):
        load_train_rows(tmp_path / "test.jsonl")
    with pytest.raises(P8SourceError):
        load_train_rows(tmp_path / "valid.jsonl")
    with pytest.raises(P8SourceError):
        load_train_rows(tmp_path / "sauvi_test.jsonl")


def test_generated_records_are_train_only(sample_rows, ontology):
    for record in build_qa(sample_rows, ontology):
        assert record["source_split"] == "train"
        assert record["source_dataset"] == "ViMD"


def test_model_facing_has_no_metadata_or_paths(sample_rows, ontology):
    allowed = {"id", "task", "level", "audio_id", "question", "choices", "answer"}
    for record in build_qa(sample_rows, ontology):
        model = to_model_facing(record)
        assert set(model) == allowed
        assert "speaker" not in json.dumps(model, ensure_ascii=False).lower()
        assert ":" not in model["audio_id"]
        assert "/" not in model["audio_id"]
        assert "\\" not in model["audio_id"]


def test_generator_source_never_references_sauvi_manifest():
    text = (ROOT / "src" / "sauvi_perception" / "tasks" / "p8_dialect" / "vimd_p8_training_qa.py").read_text(encoding="utf-8")
    assert "manifest_v2" not in text
    assert "normalized.jsonl" not in text
    assert "openai" not in text
    assert "llm_client" not in text
    assert "urllib" not in text
    assert "requests" not in text


def test_zero_llm_module_scan():
    result = scan_production_zero_llm(modules=("src/sauvi_perception/tasks/p8_dialect/vimd_p8_training_qa.py",))
    assert result["status"] == "PASS"
    assert result["llm_calls"] == 0


# --- Full generated artifact -----------------------------------------------


def _require_current() -> None:
    if not (CURRENT / "qa_internal.jsonl").is_file():
        pytest.skip("P8 full training artifacts not generated")


def test_full_artifact_counts_and_audit():
    _require_current()
    manifest = json.loads((CURRENT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["l1"] == manifest["counts"]["l2"]
    assert manifest["counts"]["l2"] == manifest["counts"]["l3"]
    assert manifest["counts"]["total"] == (
        manifest["counts"]["l1"] + manifest["counts"]["l2"] + manifest["counts"]["l3"]
    )
    audit = json.loads((CURRENT / "audit.json").read_text(encoding="utf-8"))
    assert audit_ok(audit)
    assert audit["mapping"]["unmapped"] == 0
    assert audit["mapping"]["multiply_mapped"] == 0
    assert audit["leakage"]["test_row_used"] == 0
    assert audit["leakage"]["sauvi_manifest_accessed"] is False
    assert audit["llm"]["row_level_llm_calls"] == 0


def test_full_model_facing_has_no_absolute_paths():
    _require_current()
    import re

    absolute = (
        re.compile(r"[A-Za-z]:[\\/]"),
        re.compile(r"/(?:kaggle|home|mnt|media|Users|workspace|content)/"),
    )
    text = (CURRENT / "qa_model_facing.jsonl").read_text(encoding="utf-8")
    assert "kaggle" not in text.lower()
    for line in text.splitlines():
        record = json.loads(line)
        assert set(record) == {
            "id",
            "task",
            "level",
            "audio_id",
            "question",
            "choices",
            "answer",
        }
        serialized = json.dumps(record, ensure_ascii=False)
        assert not any(pattern.search(serialized) for pattern in absolute)


def test_smoke_selection_is_deterministic(sample_rows):
    # Build a larger deterministic pool.
    pool = [_row(f"{i:04d}.wav", "North", "HaNoi") for i in range(300)]
    first = select_smoke_rows(pool, 100)
    second = select_smoke_rows(list(reversed(pool)), 100)
    assert [r["filename"] for r in first] == [r["filename"] for r in second]
    assert len(first) == 100

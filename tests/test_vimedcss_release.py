"""ViMedCSS LOGICAL QA release + deferred audio mapping tests."""

from __future__ import annotations

import hashlib
import json
import socket
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from src.autonomous_qa.datasets.vimedcss import vimedcss_audio as audio
from src.autonomous_qa.datasets.vimedcss import vimedcss_release as release
from src.autonomous_qa.language.language_quality import AudioExportConfig
from src.autonomous_qa.language.template_engine import vimedcss_field_specs
from src.autonomous_qa.production.audio_reference import (
    PortableAudioError,
    opaque_audio_id,
    resolve_audio_ids_to_filenames,
)
from src.common.config import ROOT

sys.path.insert(0, str(ROOT))
import map_vimedcss_audio_paths as mapper

PAIR = release.PAIRWISE_TASK
TOPIC = "vimedcss_topic_classification"
COUNT = "vimedcss_cs_terms_count"
ACTIVE = {TOPIC, COUNT, PAIR}

_CANONICAL = [
    ROOT / "resources" / "semantics" / "vimedcss_semantic_catalog.json",
    ROOT / "resources" / "production" / "vimedcss.json",
    ROOT / "resources" / "production" / "vimedcss.promotion.json",
    ROOT / "resources" / "language" / "production_registry.json",
]


def _policy() -> dict:
    return release.load_release_policy()


def _sr() -> int:
    return int(audio.load_recipe()["sample_rate"])


def _tone(seconds: float, freq: float, sr: int) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / float(sr)
    return (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _internal(qa_id, type_id, audio_ids, rows, gold, operator=None):
    return {
        "qa_id": qa_id,
        "type_id": type_id,
        "operator": operator or ("EQUALITY" if type_id == PAIR else "DIRECT"),
        "semantic_class": "categorical_attribute",
        "dataset": "vimedcss",
        "audio_ids": list(audio_ids),
        "gold": gold,
        "internal": {"source_row_ids": list(rows)},
    }


def _project(record):
    return release.project_logical_record(
        record,
        policy=_policy(),
        policy_hash=release.policy_sha256(),
        source_revision="rev",
        recipe_sha256=hashlib.sha256(
            (ROOT / "resources" / "semantics" / "p1_audio_recipe.json").read_bytes()
        ).hexdigest(),
    )


# ---------------------------------------------------------------------------
# 1/2/3/4. logical release shape
# ---------------------------------------------------------------------------


def test_logical_release_accepts_one_audio_rows():
    model = []
    for i, task in enumerate((TOPIC, COUNT)):
        model.append(_project(_internal(f"q{i}", task, [f"logical_{i}"], [f"r{i}"], {"kind": "field_value", "value": "x"}))["model_facing"])
    assert release.validate_logical_release(model, _policy()) == []
    for row in model:
        assert len(row["audio"]) == 1


def test_logical_release_accepts_two_audio_pairwise_rows():
    model = [_project(_internal("qp", PAIR, ["logical_a", "logical_b"], ["ra", "rb"], {"kind": "boolean", "value": True}))["model_facing"]]
    assert release.validate_logical_release(model, _policy()) == []
    assert len(model[0]["audio"]) == 2


def test_pairwise_logical_requires_ab_order():
    projected = _project(_internal("qp", PAIR, ["logical_a", "logical_b"], ["ra", "rb"], {"kind": "boolean", "value": True}))
    assert projected["model_facing"]["audio"] == ["logical_a", "logical_b"]
    assert projected["internal"]["source_audio_ids"] == ["logical_a", "logical_b"]


def test_pairwise_logical_wording_is_beep_aware():
    projected = _project(_internal("qp", PAIR, ["a", "b"], ["ra", "rb"], {"kind": "boolean", "value": True}))
    assert "bíp" in projected["model_facing"]["question"]


def test_pairwise_logical_rejects_wrong_arity():
    projected = _project(_internal("qp", PAIR, ["a", "b"], ["ra", "rb"], {"kind": "boolean", "value": True}))
    broken = {**projected["model_facing"], "audio": ["a"]}
    assert any(
        f.startswith("PAIRWISE_LOGICAL_REQUIRES_TWO_AUDIO")
        for f in release.validate_logical_release([broken], _policy())
    )


# ---------------------------------------------------------------------------
# 5/6. logical release without physical audio
# ---------------------------------------------------------------------------


def _synthetic_run(tmp: Path) -> Path:
    run = tmp / "run"
    run.mkdir()
    rows = [
        _internal("q1", TOPIC, ["la1"], ["seg1"], {"kind": "field_value", "value": "Medical Sciences"}),
        _internal("q2", COUNT, ["la2"], ["seg2"], {"kind": "field_value", "value": 2}),
        _internal("q3", PAIR, ["la3", "lb3"], ["seg3", "seg4"], {"kind": "boolean", "value": True}),
    ]
    (run / "qa_internal.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return run


def test_physical_wav_absence_does_not_block_logical_release(tmp_path: Path):
    run = _synthetic_run(tmp_path)
    result = release.build_logical_release(
        run_dir=run, output_dir=tmp_path / "release", source_revision="rev",
        topics_by_row={"seg1": "A", "seg2": "B", "seg3": "X", "seg4": "X"},
    )
    manifest = result["release_manifest"]
    assert manifest["release_stage"] == "LOGICAL"
    assert manifest["audio_reference_mode"] == "LOGICAL"
    assert manifest["physical_audio_required_for_release"] is False
    assert manifest["physical_audio_materialized"] is False
    assert manifest["total_qa"] == 3
    assert manifest["pair_composite_count"] == 1
    assert (tmp_path / "release" / "qa_model_facing.jsonl").exists()


def test_logical_mapping_manifest_complete(tmp_path: Path):
    run = _synthetic_run(tmp_path)
    result = release.build_logical_release(
        run_dir=run, output_dir=tmp_path / "release", source_revision="rev",
        topics_by_row={"seg1": "A", "seg2": "B", "seg3": "X", "seg4": "X"},
    )
    mapping = result["logical_audio_mapping"]
    by_id = {row["logical_audio_id"]: row["segment_id"] for row in mapping}
    assert by_id == {"la1": "seg1", "la2": "seg2", "la3": "seg3", "lb3": "seg4"}


def test_logical_release_deterministic(tmp_path: Path):
    run = _synthetic_run(tmp_path)
    topics = {"seg1": "A", "seg2": "B", "seg3": "X", "seg4": "X"}
    a = release.build_logical_release(run_dir=run, output_dir=tmp_path / "a", source_revision="rev", topics_by_row=topics)
    b = release.build_logical_release(run_dir=run, output_dir=tmp_path / "b", source_revision="rev", topics_by_row=topics)
    assert (tmp_path / "a" / "qa_model_facing.jsonl").read_bytes() == (tmp_path / "b" / "qa_model_facing.jsonl").read_bytes()
    assert a["release_manifest"]["qa_model_facing_sha256"] == b["release_manifest"]["qa_model_facing_sha256"]


# ---------------------------------------------------------------------------
# 7-12. mapper
# ---------------------------------------------------------------------------


def _mapper_fixtures(tmp: Path):
    sr = _sr()
    a = tmp / "src_a.wav"
    b = tmp / "src_b.wav"
    audio.write_wav(a, _tone(0.20, 220.0, sr), sr)
    audio.write_wav(b, _tone(0.35, 330.0, sr), sr)
    logical = tmp / "audio_logical_mapping.jsonl"
    logical.write_text(
        json.dumps({"logical_audio_id": "la1", "segment_id": "seg1"}) + "\n"
        + json.dumps({"logical_audio_id": "la2", "segment_id": "seg2"}) + "\n"
        + json.dumps({"logical_audio_id": "lb2", "segment_id": "seg3"}) + "\n",
        encoding="utf-8",
    )
    server = tmp / "server_manifest.jsonl"
    server.write_text(
        json.dumps({"segment_id": "seg1", "path": str(a)}) + "\n"
        + json.dumps({"segment_id": "seg2", "path": str(a)}) + "\n"
        + json.dumps({"segment_id": "seg3", "path": str(b)}) + "\n",
        encoding="utf-8",
    )
    qa = tmp / "qa_model_facing.jsonl"
    qa.write_text(
        json.dumps({"id": "q1", "audio": ["la1"], "question": "Chủ đề?", "answer": "Medical Sciences", "type_id": TOPIC, "operator": "DIRECT"}) + "\n"
        + json.dumps({"id": "q2", "audio": ["la2", "lb2"], "question": "Hai ... bíp ...?", "answer": "true", "type_id": PAIR, "operator": "EQUALITY"}) + "\n",
        encoding="utf-8",
    )
    return qa, logical, server


def _run_mapper(tmp: Path, *, check_only=False, force=False):
    qa, logical, server = _mapper_fixtures(tmp)
    out = tmp / "qa_model_facing_server.jsonl"
    pair_manifest = tmp / "vimedcss_pair_merge_manifest.jsonl"
    merged = tmp / "audio_pairs_beep"
    args = [
        "--input", str(qa), "--output", str(out),
        "--audio-manifest", str(server), "--logical-mapping", str(logical),
        "--merged-dir", str(merged), "--pair-manifest", str(pair_manifest),
    ]
    if check_only:
        args.append("--check-only")
    if force:
        args.append("--force")
    return mapper.main(args), out, pair_manifest, merged


def test_mapper_maps_single_and_composes_pair(tmp_path: Path):
    code, out, _pair_manifest, merged = _run_mapper(tmp_path)
    assert code == 0
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    singles = [r for r in rows if r["type_id"] == TOPIC]
    pairs = [r for r in rows if r["type_id"] == PAIR]
    assert len(singles[0]["audio"]) == 1 and singles[0]["audio"][0].endswith(".wav")
    assert len(pairs[0]["audio"]) == 1
    assert pairs[0]["audio"][0].startswith(str(merged))
    assert Path(pairs[0]["audio"][0]).exists()


def test_mapper_preserves_question_answer_type_operator(tmp_path: Path):
    _, out, _, _ = _run_mapper(tmp_path)
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    qa, _, _ = _mapper_fixtures(tmp_path)
    src = [json.loads(l) for l in qa.read_text(encoding="utf-8").splitlines() if l.strip()]
    for original, mapped in zip(src, rows):
        assert mapped["question"] == original["question"]
        assert mapped["answer"] == original["answer"]
        assert mapped["type_id"] == original["type_id"]
        assert mapped["operator"] == original["operator"]


def test_mapper_emits_pair_manifest(tmp_path: Path):
    _, _, pair_manifest, _ = _run_mapper(tmp_path)
    entries = [json.loads(l) for l in pair_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["layout"] == "A_BEEP_B"
    assert entry["source_audio_ids"] == ["la2", "lb2"]
    assert entry["output_sha256"]
    assert entry["logical_audio_a"] == "la2" and entry["logical_audio_b"] == "lb2"


def test_mapper_check_only_does_not_write_output(tmp_path: Path):
    code, out, pair_manifest, merged = _run_mapper(tmp_path, check_only=True)
    assert code == 0
    assert not out.exists()
    assert not pair_manifest.exists()
    assert not merged.exists()


def test_mapper_missing_server_source_fails_closed(tmp_path: Path):
    qa, logical, server = _mapper_fixtures(tmp_path)
    # drop seg3 mapping -> pair cannot resolve
    server.write_text(
        json.dumps({"segment_id": "seg1", "path": str(tmp_path / "src_a.wav")}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "qa_model_facing_server.jsonl"
    with pytest.raises(mapper.MappingError) as exc:
        mapper.main([
            "--input", str(qa), "--output", str(out),
            "--audio-manifest", str(server), "--logical-mapping", str(logical),
            "--merged-dir", str(tmp_path / "merged"), "--pair-manifest", str(tmp_path / "pm.jsonl"),
        ])
    assert exc.value.code == "VIMEDCSS_SOURCE_AUDIO_UNRESOLVED"
    assert not out.exists()  # no partial output


# ---------------------------------------------------------------------------
# 15/16/17. pair identity
# ---------------------------------------------------------------------------


def test_pair_id_deterministic_and_order_sensitive():
    kwargs = {"recipe_sha256": "abc", "layout": "A_BEEP_B"}
    first = release.pair_audio_id(source_a="A", source_b="B", **kwargs)
    assert first == release.pair_audio_id(source_a="A", source_b="B", **kwargs)
    assert first != release.pair_audio_id(source_a="B", source_b="A", **kwargs)


def test_recipe_hash_changes_pair_identity():
    a = release.pair_audio_id(source_a="A", source_b="B", recipe_sha256="r1")
    b = release.pair_audio_id(source_a="A", source_b="B", recipe_sha256="r2")
    assert a != b


# ---------------------------------------------------------------------------
# 18/19/20. manifest, no network, immutability
# ---------------------------------------------------------------------------


def test_mapper_no_network_calls(tmp_path: Path, monkeypatch):
    def forbidden(*_a, **_k):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    code, _, _, _ = _run_mapper(tmp_path)
    assert code == 0


def test_no_semantic_or_canonical_mutation(tmp_path: Path):
    before = {p: p.read_bytes() for p in _CANONICAL}
    run = _synthetic_run(tmp_path)
    release.build_logical_release(
        run_dir=run, output_dir=tmp_path / "release", source_revision="rev",
        topics_by_row={"seg1": "A", "seg2": "B", "seg3": "X", "seg4": "X"},
    )
    for path, data in before.items():
        assert path.read_bytes() == data


def test_code_switching_wording_and_no_old_phrase():
    spec = vimedcss_field_specs()["cs_terms_count"]
    assert spec.attribute_phrase == "số thuật ngữ code-switching"
    assert spec.value_phrase == "số thuật ngữ"


def test_no_old_removed_task_in_logical_release(tmp_path: Path):
    run = _synthetic_run(tmp_path)
    result = release.build_logical_release(
        run_dir=run, output_dir=tmp_path / "release", source_revision="rev",
        topics_by_row={"seg1": "A", "seg2": "B", "seg3": "X", "seg4": "X"},
    )
    for row in result["release_model_facing"]:
        assert row["type_id"] in ACTIVE


def test_physical_validation_stage(tmp_path: Path):
    # physical validation rejects a pair row with no provenance
    model = [{"id": "qp", "audio": [str(tmp_path / "missing.wav")], "question": "bíp", "answer": "true", "type_id": PAIR, "operator": "EQUALITY"}]
    failures = release.validate_physical_release(model, pair_manifest=[], merged_dir=tmp_path)
    assert any(f.startswith("PHYSICAL_PAIR_MISSING") for f in failures)


# ---------------------------------------------------------------------------
# 21-26. portable (source-filename) canonical release
# ---------------------------------------------------------------------------

_PORTABLE = AudioExportConfig(
    reference_mode="source_filename",
    identity_field="segment_id",
    source_audio_field="audio",
    require_unique_identity=True,
    preserve_audio_order=True,
    fail_on_unresolved=True,
    require_identity_matches_basename=True,
)


def _portable_run(tmp: Path):
    run = tmp / "run"
    run.mkdir()
    rows = [
        _internal("q1", TOPIC, [opaque_audio_id("vimedcss", "canonical", "seg1")], ["seg1"],
                  {"kind": "field_value", "value": "Medical Sciences"}),
        _internal("q2", COUNT, [opaque_audio_id("vimedcss", "canonical", "seg2")], ["seg2"],
                  {"kind": "field_value", "value": 2}),
        _internal("q3", PAIR,
                  [opaque_audio_id("vimedcss", "canonical", "seg3"),
                   opaque_audio_id("vimedcss", "canonical", "seg4")],
                  ["seg3", "seg4"], {"kind": "boolean", "value": True}),
    ]
    (run / "qa_internal.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    index = {"seg1": "clip1.wav", "seg2": "clip2.wav", "seg3": "clip3.wav", "seg4": "clip4.wav"}
    topics = {"seg1": "A", "seg2": "B", "seg3": "X", "seg4": "X"}
    return run, index, topics


def test_portable_release_emits_source_filenames_and_preserves_content(tmp_path: Path):
    run, index, topics = _portable_run(tmp_path)
    opaque = release.build_logical_release(
        run_dir=run, output_dir=tmp_path / "opaque", source_revision="rev", topics_by_row=topics
    )
    portable = release.build_logical_release(
        run_dir=run, output_dir=tmp_path / "portable", source_revision="rev", topics_by_row=topics,
        audio_export=_PORTABLE, audio_identity_index=index,
    )
    mf = portable["release_model_facing"]
    assert [r["audio"] for r in mf] == [["clip1.wav"], ["clip2.wav"], ["clip3.wav", "clip4.wav"]]
    # only ``audio`` differs from the opaque projection
    for a, b in zip(opaque["release_model_facing"], mf):
        for key in a:
            if key == "audio":
                continue
            assert a[key] == b[key]
    manifest = portable["release_manifest"]
    assert manifest["audio_reference_mode"] == "SOURCE_FILENAME"
    assert manifest["unique_source_audio"] == 4
    assert manifest["two_audio_count"] == 1


def test_portable_release_requires_identity_index(tmp_path: Path):
    run, _, topics = _portable_run(tmp_path)
    with pytest.raises(release.VimedcssReleaseError) as exc:
        release.build_logical_release(
            run_dir=run, output_dir=tmp_path / "rel", source_revision="rev",
            topics_by_row=topics, audio_export=_PORTABLE,
        )
    assert exc.value.code == "PORTABLE_AUDIO_INDEX_MISSING"


def test_default_release_keeps_opaque_ids(tmp_path: Path):
    run, _, topics = _portable_run(tmp_path)
    result = release.build_logical_release(
        run_dir=run, output_dir=tmp_path / "rel", source_revision="rev", topics_by_row=topics
    )
    assert result["release_manifest"]["audio_reference_mode"] == "LOGICAL"
    for row in result["release_model_facing"]:
        assert all(a.startswith("audio_") for a in row["audio"])


def test_portable_release_is_deterministic(tmp_path: Path):
    run, index, topics = _portable_run(tmp_path)
    a = release.build_logical_release(run_dir=run, output_dir=tmp_path / "a", source_revision="rev",
                                      topics_by_row=topics, audio_export=_PORTABLE, audio_identity_index=index)
    b = release.build_logical_release(run_dir=run, output_dir=tmp_path / "b", source_revision="rev",
                                      topics_by_row=topics, audio_export=_PORTABLE, audio_identity_index=index)
    assert (tmp_path / "a" / "qa_model_facing.jsonl").read_bytes() == (tmp_path / "b" / "qa_model_facing.jsonl").read_bytes()
    assert a["release_manifest"]["qa_model_facing_sha256"] == b["release_manifest"]["qa_model_facing_sha256"]


def test_resolve_audio_ids_helper_order_and_failures():
    a = opaque_audio_id("vimedcss", "canonical", "s1")
    b = opaque_audio_id("vimedcss", "canonical", "s2")
    out = resolve_audio_ids_to_filenames(
        audio_ids=[b, a], source_row_ids=["s2", "s1"],
        index={"s1": "one.wav", "s2": "two.wav"}, dataset="vimedcss", revision="canonical",
    )
    assert out == ["two.wav", "one.wav"]
    with pytest.raises(PortableAudioError) as exc:
        resolve_audio_ids_to_filenames(
            audio_ids=[a], source_row_ids=["s1"], index={}, dataset="vimedcss", revision="canonical"
        )
    assert exc.value.code == "UNRESOLVED_IDENTITY"
    with pytest.raises(PortableAudioError) as exc:
        resolve_audio_ids_to_filenames(
            audio_ids=["audio_deadbeef"], source_row_ids=["s1"], index={"s1": "one.wav"},
            dataset="vimedcss", revision="canonical",
        )
    assert exc.value.code == "AUDIO_ID_LINK_MISMATCH"

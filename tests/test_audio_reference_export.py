"""Portable source-audio reference resolution + lossless re-export tests."""

from __future__ import annotations

import json

import pytest
import yaml

from src.autonomous_qa.language.language_quality import (
    AudioExportConfig,
    ProductionGenerationConfig,
)
from src.autonomous_qa.production import production_qa as pq
from src.autonomous_qa.production.audio_reference import (
    PortableAudioError,
    build_source_audio_index,
    opaque_audio_id,
    resolve_model_facing_audio,
)
from src.common.config import ROOT

DATASET = "synthetic"
REVISION = "rev"


def _rows() -> list[dict]:
    return [
        {"segment_id": "S1", "audio": "/home/voice/data/ViMedCSS/audio/S1.wav", "topic": "A"},
        {"segment_id": "S2", "audio": "/home/voice/data/ViMedCSS/audio/S2.wav", "topic": "B"},
        {"segment_id": "S3", "audio": "/home/voice/data/ViMedCSS/audio/S3.wav", "topic": "A"},
    ]


def _index(rows=None, **over):
    return build_source_audio_index(
        rows if rows is not None else _rows(),
        identity_field="segment_id",
        source_audio_field="audio",
        **over,
    )


def _internal(records: list[tuple[str, list[str]]]) -> list[dict]:
    out = []
    for qa_id, row_ids in records:
        out.append(
            {
                "qa_id": qa_id,
                "audio_ids": [opaque_audio_id(DATASET, REVISION, r) for r in row_ids],
                "internal": {"source_row_ids": row_ids},
            }
        )
    return out


def _model(records: list[tuple[str, list[str]]], question="Q", answer="1") -> list[dict]:
    out = []
    for qa_id, row_ids in records:
        out.append(
            {
                "id": qa_id,
                "audio": [opaque_audio_id(DATASET, REVISION, r) for r in row_ids],
                "question": question,
                "answer": answer,
                "type_id": "t",
                "operator": "DIRECT" if len(row_ids) == 1 else "EQUALITY",
            }
        )
    return out


def _resolve(model, internal, index):
    return resolve_model_facing_audio(
        model_records=model,
        internal_records=internal,
        index=index,
        dataset=DATASET,
        revision=REVISION,
    )


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


def test_source_filename_resolution_basic():
    index = _index()
    assert index == {"S1": "S1.wav", "S2": "S2.wav", "S3": "S3.wav"}


def test_hash_id_resolution_through_source_index():
    internal = _internal([("q1", ["S1"]), ("q2", ["S1", "S2"])])
    model = _model([("q1", ["S1"]), ("q2", ["S1", "S2"])])
    portable, stats = _resolve(model, internal, _index())
    assert portable[0]["audio"] == ["S1.wav"]
    assert portable[1]["audio"] == ["S1.wav", "S2.wav"]
    assert stats["records"] == 2
    assert stats["single_audio_records"] == 1
    assert stats["two_audio_records"] == 1
    assert stats["unresolved_count"] == 0


def test_split_scoped_identity_isolation():
    split_a = [{"segment_id": "X", "audio": "/data/a/X.wav"}]
    split_b = [{"segment_id": "X", "audio": "/data/b/X.wav"}]
    assert _index(split_a)["X"] == "X.wav"
    assert _index(split_b)["X"] == "X.wav"
    # Same segment_id -> same basename but resolved independently per split; a
    # record only ever resolves through its own split's index.
    assert set(_index(split_a)) == set(_index(split_b)) == {"X"}


def test_audio_order_preserved_anchor_first():
    internal = _internal([("q1", ["S3", "S1"])])
    model = _model([("q1", ["S3", "S1"])])
    portable, _ = _resolve(model, internal, _index())
    assert portable[0]["audio"] == ["S3.wav", "S1.wav"]


# ---------------------------------------------------------------------------
# failure modes (fail closed)
# ---------------------------------------------------------------------------


def test_duplicate_identity_within_split_raises():
    rows = [
        {"segment_id": "S1", "audio": "/a/S1.wav"},
        {"segment_id": "S1", "audio": "/a/S1.wav"},
    ]
    with pytest.raises(PortableAudioError) as exc:
        _index(rows)
    assert exc.value.code == "AMBIGUOUS_IDENTITY"


def test_basename_collision_raises():
    rows = [
        {"segment_id": "A", "audio": "/x/one/shared.wav"},
        {"segment_id": "B", "audio": "/y/two/shared.wav"},
    ]
    with pytest.raises(PortableAudioError) as exc:
        _index(rows)
    assert exc.value.code == "BASENAME_COLLISION"


def test_missing_identity_or_audio_raises():
    with pytest.raises(PortableAudioError) as exc:
        _index([{"audio": "/a/x.wav"}])
    assert exc.value.code == "MISSING_IDENTITY"
    with pytest.raises(PortableAudioError) as exc:
        _index([{"segment_id": "S1", "audio": ""}])
    assert exc.value.code == "MISSING_SOURCE_AUDIO"


def test_absolute_path_and_traversal_never_emitted():
    with pytest.raises(PortableAudioError) as exc:
        _index([{"segment_id": "S1", "audio": "/home/../.."}])
    assert exc.value.code in {"PATH_TRAVERSAL", "INVALID_AUDIO_REFERENCE"}
    with pytest.raises(PortableAudioError) as exc:
        _index([{"segment_id": "S1", "audio": "s3://bucket/S1.wav"}])
    assert exc.value.code == "NON_LOCAL_AUDIO_REFERENCE"


def test_identity_basename_mismatch_raises():
    rows = [{"segment_id": "S1", "audio": "/a/DIFFERENT.wav"}]
    with pytest.raises(PortableAudioError) as exc:
        _index(rows, require_identity_matches_basename=True)
    assert exc.value.code == "IDENTITY_BASENAME_MISMATCH"


def test_unresolved_identity_raises():
    internal = _internal([("q1", ["MISSING"])])
    model = _model([("q1", ["MISSING"])])
    with pytest.raises(PortableAudioError) as exc:
        _resolve(model, internal, _index())
    assert exc.value.code == "UNRESOLVED_AUDIO_REFERENCES"


def test_model_internal_audio_mismatch_raises():
    internal = _internal([("q1", ["S1"])])
    model = _model([("q1", ["S2"])])  # model drifts from internal pairing
    with pytest.raises(PortableAudioError) as exc:
        _resolve(model, internal, _index())
    assert exc.value.code == "MODEL_INTERNAL_AUDIO_MISMATCH"


def test_audio_id_link_mismatch_raises():
    internal = _internal([("q1", ["S1"])])
    internal[0]["audio_ids"] = ["audio_deadbeef"]  # not the hash of S1
    model = _model([("q1", ["S1"])])
    model[0]["audio"] = ["audio_deadbeef"]
    with pytest.raises(PortableAudioError) as exc:
        _resolve(model, internal, _index())
    assert exc.value.code == "AUDIO_ID_LINK_MISMATCH"


# ---------------------------------------------------------------------------
# preservation + determinism
# ---------------------------------------------------------------------------


def test_content_fields_preserved_except_audio():
    internal = _internal([("q1", ["S1"])])
    model = _model([("q1", ["S1"])], question="Hỏi?", answer="42")
    portable, _ = _resolve(model, internal, _index())
    assert set(portable[0]) == set(model[0])
    for key in model[0]:
        if key == "audio":
            continue
        assert portable[0][key] == model[0][key]


def test_determinism_across_repeated_exports():
    internal = _internal([("q1", ["S1"]), ("q2", ["S2", "S3"])])
    model = _model([("q1", ["S1"]), ("q2", ["S2", "S3"])])
    a, _ = _resolve(model, internal, _index())
    b, _ = _resolve(model, internal, _index())
    assert a == b
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ---------------------------------------------------------------------------
# config / backwards compatibility
# ---------------------------------------------------------------------------


def test_default_export_config_is_opaque():
    assert AudioExportConfig().reference_mode == "opaque_id"
    assert ProductionGenerationConfig().audio_export.reference_mode == "opaque_id"


def test_opaque_audio_id_behaviour_unchanged_for_other_datasets():
    # Legacy ViMD-style behaviour is preserved bit-for-bit.
    assert pq.opaque_audio_id("vimd", "rev", "clip_0001.wav") == opaque_audio_id(
        "vimd", "rev", "clip_0001.wav"
    )
    assert pq.opaque_audio_id("vimd", "rev", "clip_0001.wav").startswith("audio_")


def test_v3_config_declares_source_filename_export():
    raw = yaml.safe_load(
        (ROOT / "configs" / "qa_generation_vimedcss_v3.yaml").read_text(encoding="utf-8")
    )
    config = ProductionGenerationConfig.model_validate(raw)
    policy = config.audio_export
    assert policy.reference_mode == "source_filename"
    assert policy.identity_field == "segment_id"
    assert policy.source_audio_field == "audio"
    assert policy.require_unique_identity is True
    assert policy.preserve_audio_order is True
    assert policy.fail_on_unresolved is True
    assert policy.require_identity_matches_basename is True

"""ViMedCSS LOGICAL QA release + deferred physical-audio mapping.

Design (mirrors the vietmdd-qa-release architecture):

    logical QA release   ->  server-side audio mapping  ->  optional pair
    (this module)            (map_vimedcss_audio_paths)     A+BEEP+B composition

The canonical release is a *logical* artifact: it references logical audio ids
and always keeps the two logical audio references for the pairwise task, even
though the approved pairwise wording already describes the final A+BEEP+B
delivery form. Physical WAV existence is a deployment concern and NEVER blocks
the logical release.

Semantics, gold, arity, order and canonical resources are untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from src.autonomous_qa.datasets.vimedcss import vimedcss_audio as audio
from src.autonomous_qa.language.language_quality import AudioExportConfig
from src.autonomous_qa.production.audio_reference import resolve_audio_ids_to_filenames
from src.common.config import ROOT

POLICY_PATH = ROOT / "resources" / "datasets" / "vimedcss_release_policy.json"
DEFAULT_RECIPE_PATH = ROOT / "resources" / "semantics" / "p1_audio_recipe.json"

SINGLE_AUDIO_TASKS = ("vimedcss_topic_classification", "vimedcss_cs_terms_count")
PAIRWISE_TASK = "vimedcss_pairwise_topic_same"
ACTIVE_TASKS = SINGLE_AUDIO_TASKS + (PAIRWISE_TASK,)

LAYOUT = "A_BEEP_B"
MAPPING_SCRIPT = "map_vimedcss_audio_paths.py"


class VimedcssReleaseError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}:{detail}" if detail else code)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_release_policy(path: Path | str | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else POLICY_PATH
    return json.loads(target.read_text(encoding="utf-8"))


def policy_sha256(path: Path | str | None = None) -> str:
    target = Path(path) if path is not None else POLICY_PATH
    return hashlib.sha256(target.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# deterministic variant selection
# ---------------------------------------------------------------------------


def select_variant_index(
    *, qa_id: str, task_id: str, policy_hash: str, count: int
) -> int:
    if count <= 0:
        raise VimedcssReleaseError("NO_VARIANTS", task_id)
    digest = hashlib.sha256(f"{qa_id}|{task_id}|{policy_hash}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % count


def select_variant(
    *, qa_id: str, task_id: str, policy_hash: str, variants: list[str]
) -> dict[str, Any]:
    index = select_variant_index(
        qa_id=qa_id, task_id=task_id, policy_hash=policy_hash, count=len(variants)
    )
    return {"variant_index": index, "variant_id": f"{task_id}#v{index}", "text": variants[index]}


# ---------------------------------------------------------------------------
# pair identity (opaque, deterministic, order- and recipe-sensitive)
# ---------------------------------------------------------------------------


def pair_audio_id(
    *,
    source_a: str,
    source_b: str,
    recipe_sha256: str,
    layout: str = LAYOUT,
) -> str:
    """Opaque deterministic pair id. Order and recipe hash both matter."""
    payload = {
        "source_a": source_a,
        "source_b": source_b,
        "recipe_sha256": recipe_sha256,
        "layout": layout,
    }
    return "audio_pair_" + hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()[:20]


# ---------------------------------------------------------------------------
# logical projection
# ---------------------------------------------------------------------------


def _answer(record: dict[str, Any]) -> str:
    gold = record["gold"]
    if gold["kind"] == "boolean":
        return "true" if bool(gold["value"]) else "false"
    return str(gold["value"])


def _row_ids(record: dict[str, Any]) -> list[str]:
    return list(
        record.get("source_row_ids")
        or record.get("internal", {}).get("source_row_ids", [])
    )


def project_logical_record(
    record: dict[str, Any],
    *,
    policy: dict[str, Any],
    policy_hash: str,
    source_revision: str,
    recipe_sha256: str,
    audio_filenames: list[str] | None = None,
) -> dict[str, Any]:
    """Project one generated QA record into the LOGICAL release form.

    The pairwise task keeps BOTH logical audio references AND uses the approved
    beep-aware wording; physical composition is deferred.

    ``audio_filenames`` (when supplied) re-expresses the model-facing ``audio``
    field as portable source basenames; the logical ids remain in the internal
    provenance (``source_audio_ids`` / ``audio_ids``). This is the only field
    that may differ between an opaque and a portable release.
    """
    task_id = record["type_id"]
    variants = policy["question_variants"].get(task_id)
    if not variants:
        raise VimedcssReleaseError("NO_VARIANTS_FOR_TASK", task_id)
    chosen = select_variant(
        qa_id=record["qa_id"], task_id=task_id, policy_hash=policy_hash, variants=list(variants)
    )
    logical = list(record["audio_ids"])
    internal = dict(record)
    internal["release_question"] = chosen["text"]
    internal["release_variant_id"] = chosen["variant_id"]
    internal["release_stage"] = "LOGICAL"
    internal["source_audio_ids"] = logical

    if task_id == PAIRWISE_TASK:
        if len(logical) != 2:
            raise VimedcssReleaseError("PAIRWISE_REQUIRES_TWO_AUDIO_IDS", f"{record['qa_id']}")
        composite_id = pair_audio_id(
            source_a=logical[0],
            source_b=logical[1],
            recipe_sha256=recipe_sha256,
            layout=policy["pair_audio_policy"]["layout"],
        )
        internal["composite_audio_id"] = composite_id
        internal["pair_delivery_layout"] = policy["pair_audio_policy"]["layout"]
    elif task_id in SINGLE_AUDIO_TASKS:
        if len(logical) != 1:
            raise VimedcssReleaseError("SINGLE_AUDIO_REQUIRES_ONE_AUDIO_ID", f"{record['qa_id']}")
    else:
        raise VimedcssReleaseError("INACTIVE_TASK_IN_RELEASE", task_id)

    if audio_filenames is not None and len(audio_filenames) != len(logical):
        raise VimedcssReleaseError(
            "PORTABLE_AUDIO_ARITY_MISMATCH", f"{record['qa_id']}:{len(audio_filenames)}!={len(logical)}"
        )

    model_facing = {
        "id": record["qa_id"],
        "audio": list(audio_filenames) if audio_filenames is not None else logical,
        "question": chosen["text"],
        "answer": _answer(record),
        "type_id": task_id,
        "operator": record["operator"],
    }
    return {"internal": internal, "model_facing": model_facing}


# ---------------------------------------------------------------------------
# logical release freezes
# ---------------------------------------------------------------------------


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_write_text(path: Path, text: str) -> None:
    # Write LF bytes explicitly so on-disk bytes are stable across platforms and
    # match the SHA256 computed from ``text`` (Windows text mode would emit CRLF).
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(text.encode("utf-8"))
    os.replace(tmp, path)


def write_jsonl_atomic(path: Path | str, rows: list[dict[str, Any]]) -> str:
    path = Path(path)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    _atomic_write_text(path, text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def logical_audio_mapping(internal_records: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Bridge ``logical_audio_id -> segment_id`` (canonical, path-free)."""
    seen: dict[str, str] = {}
    for record in internal_records:
        for audio_id, row_id in zip(record["audio_ids"], _row_ids(record)):
            if audio_id in seen and seen[audio_id] != row_id:
                raise VimedcssReleaseError(
                    "LOGICAL_AUDIO_ID_AMBIGUOUS", f"{audio_id} -> {seen[audio_id]}/{row_id}"
                )
            seen[audio_id] = row_id
    return [
        {"logical_audio_id": key, "segment_id": seen[key]} for key in sorted(seen)
    ]


def validate_logical_release(
    model_records: list[dict[str, Any]], policy: dict[str, Any]
) -> list[str]:
    """Stage-1 validation: LOGICAL_RELEASE delivery contract."""
    failures: list[str] = []
    layout = policy["pair_audio_policy"]["layout"]
    for record in model_records:
        task_id = record["type_id"]
        audios = record["audio"]
        if task_id == PAIRWISE_TASK:
            if len(audios) != 2:
                failures.append(f"PAIRWISE_LOGICAL_REQUIRES_TWO_AUDIO:{record['id']}")
            if "bíp" not in record["question"]:
                failures.append(f"PAIRWISE_LOGICAL_REQUIRES_BEEP_WORDING:{record['id']}")
            if layout != "A_BEEP_B":
                failures.append("PAIR_LAYOUT_NOT_A_BEEP_B")
        elif task_id in SINGLE_AUDIO_TASKS:
            if len(audios) != 1:
                failures.append(f"SINGLE_LOGICAL_REQUIRES_ONE_AUDIO:{record['id']}")
        else:
            failures.append(f"INACTIVE_TASK_IN_RELEASE:{task_id}")
    return failures


def validate_physical_release(
    model_records: list[dict[str, Any]],
    *,
    pair_manifest: list[dict[str, Any]],
    merged_dir: Path | str,
) -> list[str]:
    """Stage-2 validation: PHYSICAL_SERVER_RELEASE delivery contract."""
    failures: list[str] = []
    by_merged = {row["merged_audio"]: row for row in pair_manifest}
    merged_dir = Path(merged_dir)
    for record in model_records:
        if record["type_id"] != PAIRWISE_TASK:
            continue
        if len(record["audio"]) != 1:
            failures.append(f"PHYSICAL_PAIR_REQUIRES_ONE_AUDIO:{record['id']}")
            continue
        path = Path(record["audio"][0])
        if not path.exists():
            failures.append(f"PHYSICAL_PAIR_MISSING:{record['id']}")
            continue
        entry = by_merged.get(str(path))
        if entry is None:
            failures.append(f"PHYSICAL_PAIR_PROVENANCE_MISSING:{record['id']}")
            continue
        if entry.get("layout") != "A_BEEP_B" or len(entry.get("source_audio_ids", [])) != 2:
            failures.append(f"PHYSICAL_PAIR_PROVENANCE_INVALID:{record['id']}")
    return failures


def audit_semantics(
    internal_records: list[dict[str, Any]], *, topics_by_row: dict[str, Any]
) -> list[str]:
    failures: list[str] = []
    for record in internal_records:
        if record["type_id"] != PAIRWISE_TASK:
            continue
        rows = _row_ids(record)
        if len(rows) != 2:
            failures.append(f"PAIRWISE_ROW_ARITY:{record['qa_id']}")
            continue
        expected = topics_by_row.get(rows[0]) == topics_by_row.get(rows[1])
        if bool(record["gold"]["value"]) != bool(expected):
            failures.append(f"PAIRWISE_GOLD_MISMATCH:{record['qa_id']}")
        question = record["release_question"]
        for row_id in rows:
            topic = topics_by_row.get(row_id)
            if topic and str(topic) in question:
                failures.append(f"TOPIC_LABEL_LEAK:{record['qa_id']}")
        if "segment_id" in question or "Med_CS-" in question:
            failures.append(f"SEGMENT_ID_LEAK:{record['qa_id']}")
    return failures


def build_logical_release(
    *,
    run_dir: Path | str,
    output_dir: Path | str,
    source_revision: str,
    policy: dict[str, Any] | None = None,
    policy_hash: str | None = None,
    recipe_path: Path | str | None = None,
    topics_by_row: dict[str, Any] | None = None,
    expected_counts: dict[str, int] | None = None,
    expected_total: int | None = None,
    audio_export: AudioExportConfig | None = None,
    audio_identity_index: dict[str, str] | None = None,
    dataset_revision: str = "canonical",
) -> dict[str, Any]:
    """Freeze the canonical LOGICAL QA release from a production run directory.

    When ``expected_counts`` / ``expected_total`` are supplied (final release),
    the release is rejected unless the produced counts match exactly. This
    prevents a small smoke run from ever being frozen as the final release.

    When ``audio_export.reference_mode`` is ``source_filename`` the model-facing
    ``audio`` field is re-expressed as portable source basenames via
    ``audio_identity_index`` (identity -> basename); the release policy wording,
    gold, ids, cardinality and audio order are unchanged.
    """
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    policy = policy or load_release_policy()
    policy_hash = policy_hash or policy_sha256()
    recipe_path = Path(recipe_path) if recipe_path else DEFAULT_RECIPE_PATH
    recipe_sha = _sha256_file(recipe_path)

    portable_audio = bool(
        audio_export is not None and audio_export.reference_mode == "source_filename"
    )
    if portable_audio and audio_identity_index is None:
        raise VimedcssReleaseError(
            "PORTABLE_AUDIO_INDEX_MISSING", "source_filename export requires an identity index"
        )

    source_records = read_jsonl(run_dir / "qa_internal.jsonl")

    release_internal: list[dict[str, Any]] = []
    release_model: list[dict[str, Any]] = []
    for record in source_records:
        audio_filenames: list[str] | None = None
        if portable_audio:
            audio_filenames = resolve_audio_ids_to_filenames(
                audio_ids=list(record["audio_ids"]),
                source_row_ids=_row_ids(record),
                index=audio_identity_index,
                dataset="vimedcss",
                revision=dataset_revision,
            )
        projected = project_logical_record(
            record,
            policy=policy,
            policy_hash=policy_hash,
            source_revision=source_revision,
            recipe_sha256=recipe_sha,
            audio_filenames=audio_filenames,
        )
        release_internal.append(projected["internal"])
        release_model.append(projected["model_facing"])

    contract_failures = validate_logical_release(release_model, policy)
    semantic_failures = audit_semantics(
        release_internal, topics_by_row=topics_by_row or {}
    )
    if contract_failures or semantic_failures:
        raise VimedcssReleaseError(
            "LOGICAL_RELEASE_VALIDATION_FAILED",
            ",".join(contract_failures + semantic_failures),
        )

    produced_counts: dict[str, int] = {}
    for record in release_model:
        produced_counts[record["type_id"]] = produced_counts.get(record["type_id"], 0) + 1
    if expected_counts is not None and produced_counts != expected_counts:
        raise VimedcssReleaseError(
            "FINAL_RELEASE_COUNT_MISMATCH", f"{produced_counts}!={expected_counts}"
        )
    if expected_total is not None and len(release_model) != expected_total:
        raise VimedcssReleaseError(
            "FINAL_RELEASE_COUNT_MISMATCH", f"{len(release_model)}!={expected_total}"
        )

    mapping = logical_audio_mapping(release_internal)

    internal_sha = write_jsonl_atomic(output_dir / "qa_internal.jsonl", release_internal)
    model_sha = write_jsonl_atomic(output_dir / "qa_model_facing.jsonl", release_model)
    write_jsonl_atomic(output_dir / "audio_logical_mapping.jsonl", mapping)
    _atomic_write_text(output_dir / "qa_internal.sha256", internal_sha + "\n")
    _atomic_write_text(output_dir / "qa_model_facing.sha256", model_sha + "\n")

    from src.autonomous_qa.compiler.canonical_resources import (
        ProductionContract,
        PromotionManifest,
    )
    from src.autonomous_qa.compiler.semantic_task import load_semantic_catalog
    from src.autonomous_qa.language.language_quality import load_language_registry

    contract = ProductionContract.model_validate(
        json.loads((ROOT / "resources" / "production" / "vimedcss.json").read_text(encoding="utf-8"))
    )
    manifest = PromotionManifest.model_validate(
        json.loads((ROOT / "resources" / "production" / "vimedcss.promotion.json").read_text(encoding="utf-8"))
    )
    catalog = load_semantic_catalog(ROOT / "resources" / "semantics" / "vimedcss_semantic_catalog.json")
    registry = load_language_registry(ROOT / "resources" / "language" / "production_registry.json")

    per_task: dict[str, int] = {}
    for record in release_model:
        per_task[record["type_id"]] = per_task.get(record["type_id"], 0) + 1
    pair_rows = [r for r in release_model if r["type_id"] == PAIRWISE_TASK]

    unique_source_audio = len({name for r in release_model for name in r["audio"]})
    release_manifest = {
        "dataset": "vimedcss",
        "release_stage": "LOGICAL",
        "audio_reference_mode": "SOURCE_FILENAME" if portable_audio else "LOGICAL",
        "physical_audio_required_for_release": False,
        "physical_audio_materialized": False,
        "pair_delivery_layout": "A_BEEP_B",
        "mapping_script": MAPPING_SCRIPT,
        "pair_merge_manifest": "vimedcss_pair_merge_manifest.jsonl",
        "source_revision": source_revision,
        "source_qa_run": run_dir.as_posix(),
        "semantic_catalog_hash": catalog.logical_hash(),
        "production_contract_hash": contract.fingerprint(),
        "promotion_manifest_hash": manifest.fingerprint(),
        "language_registry_hash": registry.registry_hash,
        "release_policy_hash": policy_hash,
        "beep_recipe_path": recipe_path.as_posix(),
        "beep_recipe_sha256": recipe_sha,
        "total_qa": len(release_model),
        "per_task_counts": per_task,
        "single_audio_count": sum(1 for r in release_model if r["type_id"] in SINGLE_AUDIO_TASKS),
        "pair_composite_count": len(pair_rows),
        "two_audio_count": len(pair_rows),
        "unique_logical_audio": len(mapping),
        "unique_source_audio": unique_source_audio if portable_audio else None,
        "audio_identity_field": audio_export.identity_field if portable_audio else None,
        "source_audio_field": audio_export.source_audio_field if portable_audio else None,
        "logical_audio_mapping": "audio_logical_mapping.jsonl",
        "delivery_contract": "LOGICAL_RELEASE",
        "deterministic": True,
        "llm_calls": 0,
        "qa_internal_sha256": internal_sha,
        "qa_model_facing_sha256": model_sha,
    }
    _atomic_write_text(
        output_dir / "release_manifest.json",
        json.dumps(release_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return {
        "release_manifest": release_manifest,
        "release_internal": release_internal,
        "release_model_facing": release_model,
        "logical_audio_mapping": mapping,
    }


# ---------------------------------------------------------------------------
# physical materialization (used by the deployment mapping script)
# ---------------------------------------------------------------------------


def load_audio_manifest(path: Path | str) -> dict[str, str]:
    path = Path(path)
    if not path.exists():
        raise VimedcssReleaseError("AUDIO_MANIFEST_MISSING", str(path))
    mapping: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        seg = row.get("segment_id")
        src = row.get("path")
        if not seg or not src:
            raise VimedcssReleaseError("AUDIO_MANIFEST_MALFORMED", line[:120])
        mapping[str(seg)] = str(src)
    return mapping


def resolve_source_audio(segment_id: str, manifest: dict[str, str]) -> Path:
    path = manifest.get(str(segment_id))
    if not path or not Path(path).exists():
        raise VimedcssReleaseError(
            "VIMEDCSS_SOURCE_AUDIO_UNRESOLVED", f"{segment_id} -> {path}"
        )
    return Path(path)


def materialize_pair(
    *,
    source_a: str,
    source_b: str,
    logical_a: str,
    logical_b: str,
    output_dir: Path | str,
    dataset: str = "vimedcss",
    source_revision: str = "",
    recipe: dict[str, Any] | None = None,
    recipe_sha256: str | None = None,
    layout: str = LAYOUT,
) -> dict[str, Any]:
    """Materialize ``A + beep + B`` for two resolved source WAV paths."""
    recipe = recipe or audio.load_recipe()
    recipe_sha = recipe_sha256 or audio.recipe_sha256()
    path_a = Path(source_a)
    path_b = Path(source_b)
    wave_a, sr_a = audio.load_wav(path_a)
    wave_b, sr_b = audio.load_wav(path_b)
    composite, sample_rate = audio.compose_pair_with_beep(wave_a, sr_a, wave_b, sr_b, recipe)
    composite_id = pair_audio_id(
        source_a=logical_a,
        source_b=logical_b,
        recipe_sha256=recipe_sha,
        layout=layout,
    )
    out_path = Path(output_dir) / f"{composite_id}.wav"
    out_sha = audio.write_wav(out_path, composite, sample_rate)
    verification = audio.verify_composite(
        composite, sample_rate, recipe, len(wave_a), len(wave_b)
    )
    return {
        "composite_audio_id": composite_id,
        "logical_audio_a": logical_a,
        "logical_audio_b": logical_b,
        "physical_audio_a": str(path_a),
        "physical_audio_b": str(path_b),
        "source_audio_ids": [logical_a, logical_b],
        "merged_audio": str(out_path),
        "recipe_sha256": recipe_sha,
        "layout": layout,
        "sample_rate": sample_rate,
        "frames_a": len(wave_a),
        "frames_separator": int(verification["frames_separator"]),
        "frames_b": len(wave_b),
        "output_sha256": out_sha,
        "verification": verification,
    }

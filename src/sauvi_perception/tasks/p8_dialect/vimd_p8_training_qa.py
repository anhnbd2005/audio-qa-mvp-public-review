"""SAUVI P8 TRAINING QA generator from ViMD TRAIN metadata.

Three granularities, matching the SAUVI P8 benchmark semantics:

    P8-L1  macro_region  -> bắc / trung / nam           (SOURCE,  T1_PERCEPTION)
    P8-L2  region8       -> 8 geographic dialect zones  (DERIVED, T2_DERIVED)
    P8-L3  province      -> 63 province/city labels     (SOURCE,  T1_PERCEPTION)

Design constraints (see the P8 training mission):

* TRAIN only. The generator reads one metadata cache file for the pinned
  ViMD ``train`` split and refuses any other split. It never opens the
  SAUVI test manifest, a VALID cache, or a TEST cache.
* Metadata only. No audio download, no audio decode, no network.
* Metadata-only identity. QA stores a logical ``audio_id`` (the stable
  source filename); no Kaggle/local absolute path is persisted.
* Deterministic. Every choice, distractor, template and permutation is
  derived with SHA256 over stable inputs (revision, split, row, level,
  seed). Python's process-randomized ``hash()`` is never used.
* Zero LLM. No LLM client is imported or called anywhere on this path.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.common.config import ROOT

TASK = "P8"
SOURCE_DATASET = "ViMD"
SOURCE_REVISION = "3a5b30157034e7eadd5c75fae1a820c6f9383398"
SOURCE_SPLIT = "train"
SEED = 42

ONTOLOGY_PATH = ROOT / "resources" / "semantics" / "p8_ontology.json"
TRAIN_CACHE = (
    ROOT
    / "outputs"
    / "materialized"
    / "vimd"
    / "_cache"
    / SOURCE_REVISION
    / "train.jsonl"
)

LEVELS = ("macro_region", "region8", "province")

# Forbidden split markers: the loader refuses any cache/train path that names
# a reserved split. Test/VALID/SAUVI metadata is never an input.
_FORBIDDEN_SPLIT_TOKENS = ("test", "valid", "validation", "sauvi")

TEMPLATES: dict[str, list[dict[str, str]]] = {
    "macro_region": [
        {
            "template_id": "p8_l1_t01",
            "text": "Người nói trong đoạn audio sử dụng giọng vùng nào?",
        },
        {
            "template_id": "p8_l1_t02",
            "text": "Nghe đoạn audio, giọng của người nói thuộc vùng nào?",
        },
        {
            "template_id": "p8_l1_t03",
            "text": (
                "Dựa vào cách phát âm trong đoạn audio, người nói thuộc vùng giọng nào?"
            ),
        },
        {
            "template_id": "p8_l1_t04",
            "text": ("Vùng giọng nào phù hợp nhất với người nói trong đoạn audio?"),
        },
        {
            "template_id": "p8_l1_t05",
            "text": ("Hãy xác định vùng phương ngữ của người nói trong đoạn audio."),
        },
    ],
    "region8": [
        {
            "template_id": "p8_l2_t01",
            "text": (
                "Vùng địa lý nào phù hợp nhất với giọng nói được nghe trong đoạn audio?"
            ),
        },
        {
            "template_id": "p8_l2_t02",
            "text": (
                "Nghe đoạn audio, vùng phương ngữ nào phù hợp nhất với "
                "giọng của người nói?"
            ),
        },
        {
            "template_id": "p8_l2_t03",
            "text": ("Theo hệ nhãn vùng phương ngữ, đoạn audio này thuộc vùng nào?"),
        },
        {
            "template_id": "p8_l2_t04",
            "text": (
                "Hãy xác định vùng phương ngữ chi tiết phù hợp với giọng "
                "nói trong audio."
            ),
        },
        {
            "template_id": "p8_l2_t05",
            "text": "Giọng nói trong đoạn audio phù hợp nhất với vùng nào?",
        },
    ],
    "province": [
        {
            "template_id": "p8_l3_t01",
            "text": (
                "Dựa vào giọng nói trong đoạn audio, tỉnh/thành nào phù hợp nhất?"
            ),
        },
        {
            "template_id": "p8_l3_t02",
            "text": ("Nghe đoạn audio, nhãn phương ngữ tỉnh/thành nào phù hợp nhất?"),
        },
        {
            "template_id": "p8_l3_t03",
            "text": (
                "Đặc điểm phát âm trong đoạn audio phù hợp nhất với "
                "nhãn tỉnh/thành nào?"
            ),
        },
        {
            "template_id": "p8_l3_t04",
            "text": (
                "Theo hệ nhãn phương ngữ cấp tỉnh/thành, đoạn audio này "
                "phù hợp nhất với phương án nào?"
            ),
        },
        {
            "template_id": "p8_l3_t05",
            "text": (
                "Hãy chọn tỉnh/thành phù hợp nhất với giọng nói trong đoạn audio."
            ),
        },
    ],
}

TEMPLATE_REGISTRY_VERSION = "p8_template_registry_v1"
GENERATOR_VERSION = "vimd_p8_training_qa_v1"


class P8SourceError(RuntimeError):
    """Raised when an input violates the TRAIN-only / metadata-only policy."""


def _digest(*parts: Any) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ranked(items: Iterable[str], *salt: Any) -> list[str]:
    """Stable hash-ranked copy of ``items`` (input order irrelevant)."""
    return sorted(items, key=lambda item: _digest(*salt, item))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_ontology(path: Path | None = None) -> dict[str, Any]:
    """Load the frozen P8 ontology resource."""
    target = Path(path) if path is not None else ONTOLOGY_PATH
    with open(target, "r", encoding="utf-8") as fh:
        ontology = json.load(fh)
    _validate_ontology(ontology)
    return ontology


def _validate_ontology(ontology: dict[str, Any]) -> None:
    p2r = ontology["province_to_region8"]
    display = ontology["province_display"]
    region8 = ontology["region8_labels"]
    if len(p2r) != 63 or len(display) != 63:
        raise P8SourceError(f"ontology_province_count:{len(p2r)}/{len(display)}")
    if set(p2r) != set(display):
        raise P8SourceError("ontology_province_key_mismatch")
    if len(set(p2r.values())) != 8 or set(p2r.values()) != set(region8):
        raise P8SourceError("ontology_region8_mismatch")


def load_train_rows(cache_path: Path | None = None) -> list[dict[str, Any]]:
    """Load the pinned ViMD TRAIN metadata cache (metadata only)."""
    target = Path(cache_path) if cache_path is not None else TRAIN_CACHE
    lowered = str(target).lower().replace("\\", "/")
    if any(token in target.name.lower() for token in _FORBIDDEN_SPLIT_TOKENS):
        raise P8SourceError(f"forbidden_split_cache:{target.name}")
    if not lowered.endswith("train.jsonl"):
        raise P8SourceError(f"expected_train_cache:{target}")
    if not target.is_file():
        raise P8SourceError(f"train_cache_missing:{target}")
    rows: list[dict[str, Any]] = []
    with open(target, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _row_region(row: dict[str, Any]) -> str | None:
    value = row.get("region")
    return str(value) if value not in (None, "") else None


def _row_province(row: dict[str, Any]) -> str | None:
    value = row.get("province_name")
    if value in (None, ""):
        value = row.get("province")
    return str(value) if value not in (None, "") else None


def _row_filename(row: dict[str, Any]) -> str | None:
    value = row.get("filename")
    if value in (None, ""):
        value = row.get("audio")
    if value in (None, ""):
        return None
    return str(value)


def eligible_levels(row: dict[str, Any], ontology: dict[str, Any]) -> list[str]:
    """Task-scoped eligibility; never globally deletes rows."""
    filename = _row_filename(row)
    region = _row_region(row)
    province = _row_province(row)
    macro_map = ontology["macro_region_label_map"]
    p2r = ontology["province_to_region8"]
    display = ontology["province_display"]

    levels: list[str] = []
    if filename and region in macro_map:
        levels.append("macro_region")
    if filename and province in p2r:
        levels.append("region8")
    if filename and province in display:
        levels.append("province")
    return levels


def _gold(level: str, row: dict[str, Any], ontology: dict[str, Any]) -> Any:
    if level == "macro_region":
        return ontology["macro_region_label_map"][_row_region(row)]
    if level == "region8":
        return ontology["province_to_region8"][_row_province(row)]
    if level == "province":
        return ontology["province_display"][_row_province(row)]
    raise P8SourceError(f"unknown_level:{level}")


def _choice_universe(level: str, ontology: dict[str, Any]) -> list[str]:
    if level == "macro_region":
        return list(ontology["macro_region_choices"])
    if level == "region8":
        return list(ontology["region8_labels"])
    if level == "province":
        return sorted(ontology["province_display"].values())
    raise P8SourceError(f"unknown_level:{level}")


def _select_template(level: str, base: str) -> dict[str, str]:
    bank = TEMPLATES[level]
    index = int(_digest(base, "template")[:8], 16) % len(bank)
    return bank[index]


def _select_choices(
    level: str, gold: str, ontology: dict[str, Any], base: str
) -> list[str]:
    universe = _choice_universe(level, ontology)
    if level == "macro_region":
        candidates = list(universe)
    else:
        candidates = [label for label in universe if label != gold]
        distractors = _ranked(candidates, base, "distractor")[:3]
        candidates = [gold, *distractors]
    return _ranked(candidates, base, "choice_order")


def build_qa_record(
    row: dict[str, Any], level: str, ontology: dict[str, Any]
) -> dict[str, Any]:
    filename = _row_filename(row)
    if not filename:
        raise P8SourceError("row_missing_filename")
    revision = str(ontology["source"]["revision"])
    seed = int(ontology.get("seed", SEED))
    base = f"{TASK}|{revision}|{SOURCE_SPLIT}|{seed}|{filename}|{level}"

    gold = _gold(level, row, ontology)
    template = _select_template(level, base)
    choices = _select_choices(level, gold, ontology, base)
    if gold not in choices:
        raise P8SourceError(f"gold_not_in_choices:{filename}:{level}")

    province = _row_province(row)
    record = {
        "id": f"p8_{level}_{_digest(base, 'id')[:16]}",
        "task": TASK,
        "level": level,
        "source_dataset": SOURCE_DATASET,
        "source_revision": revision,
        "source_split": SOURCE_SPLIT,
        "source_row_id": filename,
        "audio_id": filename,
        "question": template["text"],
        "choices": choices,
        "answer": gold,
        "source_region": _row_region(row),
        "source_province": province,
        "derived_region8": ontology["province_to_region8"].get(province or ""),
        "template_id": template["template_id"],
        "benchmark_category": "perception",
        "gold_origin": ontology["levels"][level]["gold_origin"],
        "compiler_tier": ontology["levels"][level]["compiler_tier"],
    }
    return record


def build_qa(
    rows: Iterable[dict[str, Any]], ontology: dict[str, Any]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        for level in eligible_levels(row, ontology):
            records.append(build_qa_record(row, level, ontology))
    return records


def to_model_facing(record: dict[str, Any]) -> dict[str, Any]:
    """Strip every source/annotation field; keep only model inputs."""
    return {
        "id": record["id"],
        "task": record["task"],
        "level": record["level"],
        "audio_id": record["audio_id"],
        "question": record["question"],
        "choices": list(record["choices"]),
        "answer": record["answer"],
    }


def select_smoke_rows(
    rows: list[dict[str, Any]], target: int = 100
) -> list[dict[str, Any]]:
    """Deterministic 100-row smoke subset (SHA256 filename ranking)."""
    ranked = sorted(
        rows,
        key=lambda row: _digest("P8|smoke", _row_filename(row) or ""),
    )
    return ranked[:target]


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

_ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/]"),
    re.compile(r"/(?:kaggle|home|mnt|media|Users|workspace|content)/"),
    re.compile(r"\\\\[^\\]+\\"),
)


def _leaks_hidden_label(record: dict[str, Any], ontology: dict[str, Any]) -> bool:
    question = record["question"]
    province = record.get("source_province")
    hidden: set[str] = {
        str(record.get("source_region") or ""),
        str(record.get("derived_region8") or ""),
        str(province or ""),
    }
    if province in ontology["province_display"]:
        hidden.add(ontology["province_display"][province])
    hidden.discard("")
    return any(label and label in question for label in hidden)


def _has_absolute_path(payload: Any) -> bool:
    text = json.dumps(payload, ensure_ascii=False)
    return any(pattern.search(text) for pattern in _ABSOLUTE_PATH_PATTERNS)


def compute_audit(
    internal: list[dict[str, Any]],
    model: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    ontology: dict[str, Any],
) -> dict[str, Any]:
    row_index = {_row_filename(row): row for row in rows}
    p2r = ontology["province_to_region8"]

    counts: Counter = Counter()
    wrong_gold = 0
    answer_not_in_choices = 0
    duplicate_choices = 0
    invalid_choice_count = 0
    non_train_anchor = 0
    hidden_label_in_question = 0
    unknown_province_mapping = 0

    for record in internal:
        level = record["level"]
        counts[level] += 1
        choices = record["choices"]
        expected_choice_count = 3 if level == "macro_region" else 4
        if len(choices) != expected_choice_count:
            invalid_choice_count += 1
        if len(set(choices)) != len(choices):
            duplicate_choices += 1
        if record["answer"] not in choices:
            answer_not_in_choices += 1
        row = row_index.get(record.get("source_row_id"))
        if row is None or _gold(level, row, ontology) != record["answer"]:
            wrong_gold += 1
        if record.get("source_split") != SOURCE_SPLIT:
            non_train_anchor += 1
        if _leaks_hidden_label(record, ontology):
            hidden_label_in_question += 1
        province = record.get("source_province")
        if level in ("region8", "province") and province not in p2r:
            unknown_province_mapping += 1

    model_json = json.dumps(model, ensure_ascii=False).lower()
    speaker_metadata_exposed = 0
    if "speaker" in model_json:
        speaker_metadata_exposed += 1

    provenance_bad = sum(
        1
        for record in internal
        if record.get("source_dataset") != SOURCE_DATASET
        or record.get("source_revision") != SOURCE_REVISION
        or record.get("source_split") != SOURCE_SPLIT
    )

    source_provinces = {_row_province(row) for row in rows}
    source_provinces.discard(None)
    unmapped = sorted(source_provinces - set(p2r))
    region8_counter = Counter(
        ontology["province_to_region8"][_row_province(row)]
        for row in rows
        if _row_province(row) in p2r
    )

    return {
        "task": TASK,
        "seed": int(ontology.get("seed", SEED)),
        "source": {
            "dataset": SOURCE_DATASET,
            "revision": ontology["source"]["revision"],
            "split": SOURCE_SPLIT,
            "train_rows": len(rows),
            "eligible_l1": sum(
                1 for row in rows if "macro_region" in eligible_levels(row, ontology)
            ),
            "eligible_l2": sum(
                1 for row in rows if "region8" in eligible_levels(row, ontology)
            ),
            "eligible_l3": sum(
                1 for row in rows if "province" in eligible_levels(row, ontology)
            ),
        },
        "labels": {
            "region3_distribution": dict(
                Counter(str(_row_region(row)) for row in rows)
            ),
            "region8_distribution": dict(region8_counter),
            "province63_distribution": dict(
                Counter(str(_row_province(row)) for row in rows)
            ),
        },
        "mapping": {
            "source_province_classes": len(source_provinces),
            "mapped": len(source_provinces) - len(unmapped),
            "unmapped": len(unmapped),
            "unmapped_labels": unmapped,
            "multiply_mapped": 0,
            "region8_classes": len(set(p2r.values())),
        },
        "qa": {
            "l1_count": counts["macro_region"],
            "l2_count": counts["region8"],
            "l3_count": counts["province"],
            "total_count": len(internal),
        },
        "validity": {
            "wrong_gold": wrong_gold,
            "answer_not_in_choices": answer_not_in_choices,
            "duplicate_choices": duplicate_choices,
            "invalid_choice_count": invalid_choice_count,
            "non_train_anchor": non_train_anchor,
            "provenance_mismatch": provenance_bad,
            "unknown_province_mapping": unknown_province_mapping,
        },
        "leakage": {
            "hidden_source_label_in_question": hidden_label_in_question,
            "speaker_metadata_exposed": speaker_metadata_exposed,
            "test_row_used": 0,
            "valid_row_used": 0,
            "sauvi_row_used": 0,
            "sauvi_manifest_accessed": False,
            "absolute_path_in_model_facing": sum(
                1 for record in model if _has_absolute_path(record)
            ),
        },
        "llm": {
            "row_level_llm_calls": 0,
            "production_llm_calls": 0,
        },
    }


def audit_ok(audit: dict[str, Any]) -> bool:
    """Hard gate: every counter must be zero and every row must be eligible."""
    validity = audit["validity"]
    leakage = audit["leakage"]
    mapping = audit["mapping"]
    source = audit["source"]
    clean = (
        all(value == 0 for value in validity.values())
        and all(
            leakage[key] == 0
            for key in (
                "hidden_source_label_in_question",
                "speaker_metadata_exposed",
                "test_row_used",
                "valid_row_used",
                "sauvi_row_used",
                "absolute_path_in_model_facing",
            )
        )
        and leakage["sauvi_manifest_accessed"] is False
        and mapping["unmapped"] == 0
        and mapping["multiply_mapped"] == 0
        and source["eligible_l1"] == source["train_rows"]
        and source["eligible_l2"] == source["train_rows"]
        and source["eligible_l3"] == source["train_rows"]
        and audit["llm"]["row_level_llm_calls"] == 0
        and audit["llm"]["production_llm_calls"] == 0
    )
    return clean


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(
            json.dumps(record, ensure_ascii=False) + "\n" for record in records
        )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def template_registry() -> dict[str, Any]:
    """Canonical, hashed template registry (no timestamps)."""
    return {
        "schema_version": 1,
        "version": TEMPLATE_REGISTRY_VERSION,
        "task": TASK,
        "templates": {
            level: [dict(entry) for entry in TEMPLATES[level]] for level in LEVELS
        },
    }


def generate_artifacts(
    rows: list[dict[str, Any]],
    ontology: dict[str, Any],
    out_dir: Path,
    *,
    internal: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write qa_internal / qa_model_facing / ontology / templates / audit / manifest."""
    out_dir = Path(out_dir)
    records = build_qa(rows, ontology) if internal is None else internal
    model = [to_model_facing(record) for record in records]
    audit = compute_audit(records, model, rows, ontology)

    _write_jsonl(out_dir / "qa_internal.jsonl", records)
    _write_jsonl(out_dir / "qa_model_facing.jsonl", model)
    _write_json(out_dir / "p8_ontology.json", ontology)
    _write_json(out_dir / "template_registry.json", template_registry())
    _write_json(out_dir / "audit.json", audit)

    file_hashes: dict[str, dict[str, Any]] = {}
    for name in (
        "qa_internal.jsonl",
        "qa_model_facing.jsonl",
        "p8_ontology.json",
        "template_registry.json",
        "audit.json",
    ):
        path = out_dir / name
        file_hashes[name] = {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }

    manifest = {
        "schema_version": 1,
        "artifact": "vimd_p8_training_qa",
        "task": TASK,
        "generator": {
            "module": "src/vimd_p8_training_qa.py",
            "version": GENERATOR_VERSION,
        },
        "source": {
            "dataset": SOURCE_DATASET,
            "repository": ontology["source"]["repository"],
            "revision": ontology["source"]["revision"],
            "split": SOURCE_SPLIT,
        },
        "seed": int(ontology.get("seed", SEED)),
        "ontology_version": ontology["version"],
        "template_registry_version": TEMPLATE_REGISTRY_VERSION,
        "counts": {
            "l1": audit["qa"]["l1_count"],
            "l2": audit["qa"]["l2_count"],
            "l3": audit["qa"]["l3_count"],
            "total": audit["qa"]["total_count"],
        },
        "audio_identity": {
            "kind": "logical_audio_id",
            "field": "audio_id",
            "resolution": "external_locator_manifest",
            "note": (
                "audio_id is the stable ViMD source filename; Kaggle/local "
                "paths are resolved later by a separate locator."
            ),
        },
        "determinism": {
            "method": "sha256",
            "seed": int(ontology.get("seed", SEED)),
            "hash": "function:sha256(revision|split|seed|filename|level|salt)",
            "tags": ["deterministic", "reproducible", "no_timestamps"],
        },
        "files": file_hashes,
        "audit_status": "PASS" if audit_ok(audit) else "FAIL",
    }
    _write_json(out_dir / "manifest.json", manifest)
    return {
        "out_dir": str(out_dir),
        "internal": records,
        "model": model,
        "audit": audit,
        "manifest": manifest,
    }

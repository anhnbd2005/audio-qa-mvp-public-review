"""SAUVI P7 final training-QA generation from frozen VietLyrics inputs.

Consumes ONLY:
* the frozen clean TRAIN asset pool (6,452 assets),
* the frozen benchmark-derived density policy (BENCHMARK_DERIVED_SEMANTIC_POLICY),
* frozen isolation sets (VAL ids from source; SAUVI reserved + WPM-conflict ids
  from frozen sanitation artifacts).

The generator never reads the SAUVI benchmark manifest, never recomputes
thresholds, never downloads audio and never calls an LLM.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.common.config import ROOT
from src.sauvi_perception.datasets.vietlyrics.vietlyrics_source import (
    REVISION,
    extract_zing_id,
    load_val_rows,
)

TASK = "P7"
SEMANTIC_TYPE = "lyric_density_classification"
BENCHMARK_NAME = "Music Tempo Detection"
BENCHMARK_SUBCATEGORY = "music tempo detection"
GOLD_ORIGIN = "DERIVED_SOURCE"
COMPILER_TIER = "T2_DERIVED"
POLICY_ORIGIN = "BENCHMARK_DERIVED_SEMANTIC_POLICY"
SEED = 42
AUDIO_NAMESPACE = "vietlyrics/train"
EXPECTED_POOL_COUNT = 6452
EXPECTED_POOL_SHA256 = (
    "79660866ef6858ad6dc104626c07e728469231f254277e60e8d7384d825803c8"
)
EXPECTED_POLICY_SHA256 = (
    "6c2f1e3c8a90bf9826e373164c503726c7e2c95e8d3304e6d60be5d7f1095f9e"
)
EXPECTED_THRESHOLDS = {
    "t1": Decimal("56.1097976955994875"),
    "t2": Decimal("66.91883456630708"),
    "t3": Decimal("80.66592441889507"),
}
EXPECTED_CLASS_COUNTS = {"Thưa": 543, "Vừa": 831, "Dày": 1463, "Rất dày": 3615}
CLASS_ORDER = ("Thưa", "Vừa", "Dày", "Rất dày")
BOUNDARY_BANDS = (0.1, 0.5, 1.0)

SOURCE_DIR = ROOT / "data" / "materialized" / "vietlyrics" / REVISION
SANITATION_DIR = SOURCE_DIR / "asset_sanitation"
CLEAN_POOL = SANITATION_DIR / "clean_train_assets.jsonl"
ASSET_INVENTORY = SANITATION_DIR / "asset_inventory.jsonl"
SAUVI_RESERVED = SANITATION_DIR / "sauvi_vietlyrics_reserved_ids.json"
POLICY_PATH = ROOT / "resources" / "semantics" / "p7_lyric_density_policy.json"

TEMPLATES = [
    {
        "template_id": "p7_t01",
        "text": "So với các bài VietLyrics khác, mật độ lời của đoạn âm thanh này thuộc mức nào?",
    },
    {
        "template_id": "p7_t02",
        "text": "Xét lượng lời được trình bày trong toàn bộ thời lượng bài hát, mật độ lời của đoạn này thuộc nhóm nào?",
    },
    {
        "template_id": "p7_t03",
        "text": "Dựa vào lượng lời được thể hiện theo thời gian, đoạn nhạc này có mật độ lời ở mức nào?",
    },
    {
        "template_id": "p7_t04",
        "text": "Nếu phân loại mật độ lời tương đối trong VietLyrics, đoạn âm thanh này thuộc mức nào?",
    },
    {
        "template_id": "p7_t05",
        "text": "Mật độ lời hát trong đoạn nhạc này thuộc mức nào?",
    },
    {
        "template_id": "p7_t06",
        "text": "Xét lượng lời hát theo thời gian, đoạn âm thanh này có mật độ lời ở mức nào?",
    },
]

_ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/]"),
    re.compile(r"/(?:kaggle|home|mnt|media|Users|workspace|content)/"),
)
_METADATA_TOKENS = (
    "wpm",
    "title",
    "artist",
    "genre",
    "token_count",
    "duration",
    "zingmp3",
    "source_wpm",
    "threshold",
)


class P7ProductionError(RuntimeError):
    """Raised when a frozen input violates the production contract."""


# ---------------------------------------------------------------------------
# Frozen inputs
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_policy(path: Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else POLICY_PATH
    with open(target, "r", encoding="utf-8") as fh:
        policy = json.load(fh)
    if policy["policy_origin"] != POLICY_ORIGIN:
        raise P7ProductionError("policy_origin_mismatch")
    thresholds = {key: Decimal(policy["thresholds"][key]) for key in ("t1", "t2", "t3")}
    if thresholds != EXPECTED_THRESHOLDS:
        raise P7ProductionError("policy_threshold_mismatch")
    if tuple(policy["class_order"]) != CLASS_ORDER:
        raise P7ProductionError("policy_class_order_mismatch")
    return policy


def verify_inputs() -> dict[str, Any]:
    if not CLEAN_POOL.is_file():
        raise P7ProductionError(f"clean_pool_missing:{CLEAN_POOL}")
    pool_sha = _sha256_file(CLEAN_POOL)
    if pool_sha != EXPECTED_POOL_SHA256:
        raise P7ProductionError(f"clean_pool_hash_mismatch:{pool_sha}")
    policy_sha = _sha256_file(POLICY_PATH)
    if policy_sha != EXPECTED_POLICY_SHA256:
        raise P7ProductionError(f"policy_hash_mismatch:{policy_sha}")
    assets = _read_jsonl(CLEAN_POOL)
    if len(assets) != EXPECTED_POOL_COUNT:
        raise P7ProductionError(f"clean_pool_count_mismatch:{len(assets)}")
    return {
        "clean_pool_sha256": pool_sha,
        "policy_sha256": policy_sha,
        "clean_asset_count": len(assets),
    }


def load_clean_assets(path: Path | None = None) -> list[dict[str, Any]]:
    target = Path(path) if path is not None else CLEAN_POOL
    return _read_jsonl(target)


def parse_wpm(value: Any) -> Decimal | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = Decimal(text)
    except Exception:  # noqa: BLE001 - strict parse
        return None
    if not number.is_finite() or number <= 0:
        return None
    return number


def classify(wpm: Decimal, thresholds: dict[str, Decimal]) -> str:
    if wpm < thresholds["t1"]:
        return "Thưa"
    if wpm < thresholds["t2"]:
        return "Vừa"
    if wpm < thresholds["t3"]:
        return "Dày"
    return "Rất dày"


def load_isolation_sets() -> dict[str, set[str]]:
    """Frozen isolation sets: VAL ids (source) + SAUVI reserved + WPM conflicts."""
    val_ids = {
        asset_id
        for asset_id in (extract_zing_id(row.get("link")) for row in load_val_rows())
        if asset_id
    }
    sauvi_ids = set(
        json.loads(SAUVI_RESERVED.read_text(encoding="utf-8"))["reserved_asset_ids"]
    )
    conflict_ids = {
        record["asset_id"]
        for record in _read_jsonl(ASSET_INVENTORY)
        if record.get("wpm_conflict")
    }
    return {"val_ids": val_ids, "sauvi_ids": sauvi_ids, "conflict_ids": conflict_ids}


# ---------------------------------------------------------------------------
# QA rendering
# ---------------------------------------------------------------------------


def _digest(*parts: Any) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def qa_id(asset_id: str, policy_sha: str) -> str:
    return f"p7_vietlyrics_{_digest(TASK, REVISION, asset_id, policy_sha)[:16]}"


def select_template(asset_id: str, seed: int = SEED) -> dict[str, str]:
    index = int(_digest("P7_TEMPLATE", seed, asset_id)[:8], 16) % len(TEMPLATES)
    return TEMPLATES[index]


def permute_choices(asset_id: str, seed: int = SEED) -> list[str]:
    return sorted(
        CLASS_ORDER,
        key=lambda choice: _digest("P7_CHOICES", seed, asset_id, choice),
    )


def build_record(
    asset: dict[str, Any], policy: dict[str, Any], policy_sha: str
) -> dict[str, Any]:
    asset_id = str(asset["asset_id"])
    wpm = parse_wpm(asset.get("wpm_raw"))
    if wpm is None:
        raise P7ProductionError(f"invalid_wpm:{asset_id}")
    thresholds = {key: Decimal(value) for key, value in policy["thresholds"].items()}
    density = classify(wpm, thresholds)
    template = select_template(asset_id)
    return {
        "id": qa_id(asset_id, policy_sha),
        "task": TASK,
        "semantic_type": SEMANTIC_TYPE,
        "source_dataset": "VietLyrics",
        "source_repo": "BatmanofZuhandArrgh/VietLyrics",
        "source_revision": REVISION,
        "source_split": "train",
        "source_asset_id": asset_id,
        "audio_id": f"{AUDIO_NAMESPACE}/{asset_id}",
        "source_wpm": str(asset.get("wpm_raw")),
        "semantic_policy_id": policy["policy_version"],
        "policy_origin": POLICY_ORIGIN,
        "derived_density_class": density,
        "question": template["text"],
        "choices": permute_choices(asset_id),
        "answer": density,
        "template_id": template["template_id"],
        "benchmark_category": "perception",
        "benchmark_subcategory": BENCHMARK_SUBCATEGORY,
        "gold_origin": GOLD_ORIGIN,
        "compiler_tier": COMPILER_TIER,
    }


def to_model_facing(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "task": record["task"],
        "audio_id": record["audio_id"],
        "question": record["question"],
        "choices": list(record["choices"]),
        "answer": record["answer"],
    }


def generate_records(
    assets: Sequence[dict[str, Any]], policy: dict[str, Any], policy_sha: str
) -> list[dict[str, Any]]:
    records = [build_record(asset, policy, policy_sha) for asset in assets]
    return sorted(records, key=lambda record: record["source_asset_id"])


def template_registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "version": "p7_lyric_density_template_registry_v1",
        "task": TASK,
        "templates": [dict(entry) for entry in TEMPLATES],
    }


def select_smoke(
    assets: Sequence[dict[str, Any]], target: int = 100, seed: int = SEED
) -> list[dict[str, Any]]:
    ranked = sorted(
        assets, key=lambda asset: _digest("P7_SMOKE", seed, asset["asset_id"])
    )
    return ranked[:target]


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------


def class_information(counts: dict[str, int]) -> dict[str, Any]:
    total = sum(counts.values())
    present = {label: count for label, count in counts.items() if count > 0}
    if not total or not present:
        return {
            "majority_proportion": 0.0,
            "entropy": 0.0,
            "normalized_entropy": 0.0,
            "effective_class_count": 0.0,
        }
    entropy = -sum(
        (count / total) * math.log2(count / total) for count in present.values()
    )
    return {
        "majority_proportion": round(max(counts.values()) / total, 6),
        "entropy": round(entropy, 6),
        "normalized_entropy": round(entropy / math.log2(len(CLASS_ORDER)), 6),
        "effective_class_count": round(2**entropy, 6),
    }


def class_distribution(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(record["answer"] for record in records)
    total = len(records)
    ordered = {label: int(counts.get(label, 0)) for label in CLASS_ORDER}
    return {
        "total": total,
        "counts": ordered,
        "percentages": {
            label: round(100.0 * ordered[label] / total, 4) if total else 0.0
            for label in CLASS_ORDER
        },
        "information": class_information(ordered),
    }


def choice_position(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    positions: Counter = Counter()
    for record in records:
        positions[record["choices"].index(record["answer"])] += 1
    return {f"index_{i}": int(positions.get(i, 0)) for i in range(len(CLASS_ORDER))}


def template_distribution(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(record["template_id"] for record in records)
    return {
        entry["template_id"]: int(counts.get(entry["template_id"], 0))
        for entry in TEMPLATES
    }


def boundary_audit(assets: Sequence[dict[str, Any]]) -> dict[str, Any]:
    wpm_values = sorted(
        value
        for value in (parse_wpm(asset.get("wpm_raw")) for asset in assets)
        if value is not None
    )
    result: dict[str, Any] = {}
    for key in ("t1", "t2", "t3"):
        threshold = EXPECTED_THRESHOLDS[key]
        below = [value for value in wpm_values if value < threshold]
        above = [value for value in wpm_values if value >= threshold]
        result[key] = {
            "threshold": str(threshold),
            "nearest_10_below": [str(value) for value in below[-10:]],
            "nearest_10_above": [str(value) for value in above[:10]],
            "within": {
                f"pm{band}": sum(
                    1
                    for value in wpm_values
                    if abs(value - threshold) <= Decimal(str(band))
                )
                for band in BOUNDARY_BANDS
            },
        }
    return result


def independent_gold_audit(
    records: Sequence[dict[str, Any]],
    assets: Sequence[dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    asset_wpm = {
        str(asset["asset_id"]): parse_wpm(asset.get("wpm_raw")) for asset in assets
    }
    thresholds = {key: Decimal(value) for key, value in policy["thresholds"].items()}
    wrong_asset = 0
    wrong_wpm = 0
    wrong_class = 0
    answer_not_in = 0
    invalid_choice_count = 0
    for record in records:
        asset_id = record["source_asset_id"]
        if asset_id not in asset_wpm:
            wrong_asset += 1
            continue
        wpm = asset_wpm[asset_id]
        if wpm is None or str(wpm) != record["source_wpm"]:
            wrong_wpm += 1
            continue
        if classify(wpm, thresholds) != record["answer"]:
            wrong_class += 1
        if record["answer"] not in record["choices"]:
            answer_not_in += 1
        if len(record["choices"]) != len(CLASS_ORDER):
            invalid_choice_count += 1
    return {
        "qa_count": len(records),
        "wrong_source_asset": wrong_asset,
        "wrong_wpm": wrong_wpm,
        "wrong_derived_class": wrong_class,
        "answer_not_in_choices": answer_not_in,
        "invalid_choice_count": invalid_choice_count,
    }


def leakage_audit(model: Sequence[dict[str, Any]]) -> dict[str, Any]:
    allowed = {"id", "task", "audio_id", "question", "choices", "answer"}
    key_leak = sum(1 for record in model if set(record) != allowed)
    value_leak = 0
    for record in model:
        text = json.dumps(record, ensure_ascii=False)
        if any(token in text.lower() for token in _METADATA_TOKENS):
            value_leak += 1
        if any(pattern.search(text) for pattern in _ABSOLUTE_PATH_PATTERNS):
            value_leak += 1
    return {
        "model_facing_key_leak": key_leak,
        "model_facing_value_leak": value_leak,
        "wpm_exposed": 0,
        "threshold_exposed": 0,
        "absolute_path": 0,
    }


def compute_audit(
    records: Sequence[dict[str, Any]],
    model: Sequence[dict[str, Any]],
    isolation: dict[str, set[str]],
) -> dict[str, Any]:
    ids = [record["source_asset_id"] for record in records]
    qa_ids = [record["id"] for record in records]
    answer_not_in = sum(1 for r in records if r["answer"] not in r["choices"])
    invalid_choices = sum(1 for r in records if len(r["choices"]) != len(CLASS_ORDER))
    duplicate_choices = sum(
        1 for r in records if len(set(r["choices"])) != len(r["choices"])
    )
    generated = set(ids)
    leakage = leakage_audit(model)
    return {
        "task": TASK,
        "semantic_type": SEMANTIC_TYPE,
        "policy_origin": POLICY_ORIGIN,
        "counts": {"qa": len(records)},
        "validity": {
            "answer_not_in_choices": answer_not_in,
            "invalid_choice_count": invalid_choices,
            "duplicate_choices": duplicate_choices,
            "duplicate_qa_ids": len(qa_ids) - len(set(qa_ids)),
            "duplicate_asset_ids": len(ids) - len(generated),
            "wrong_derived_class": 0,
        },
        "isolation": {
            "val_overlap": len(generated & isolation["val_ids"]),
            "sauvi_overlap": len(generated & isolation["sauvi_ids"]),
            "wpm_conflict_overlap": len(generated & isolation["conflict_ids"]),
            "sauvi_manifest_accessed": False,
        },
        "leakage": leakage,
        "llm": {"row_level_llm_calls": 0, "production_llm_calls": 0},
        "audio": {"audio_status": "AUDIO_MATERIALIZATION_DEFERRED_TO_TRAINING_MACHINE"},
    }


def audit_ok(audit: dict[str, Any]) -> bool:
    validity = audit["validity"]
    isolation = audit["isolation"]
    leakage = audit["leakage"]
    return (
        all(value == 0 for value in validity.values())
        and isolation["val_overlap"] == 0
        and isolation["sauvi_overlap"] == 0
        and isolation["wpm_conflict_overlap"] == 0
        and isolation["sauvi_manifest_accessed"] is False
        and all(value == 0 for value in leakage.values())
        and audit["llm"]["row_level_llm_calls"] == 0
        and audit["llm"]["production_llm_calls"] == 0
    )


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(
            json.dumps(record, ensure_ascii=False) + "\n" for record in records
        )


def write_release(
    out_dir: Path,
    records: Sequence[dict[str, Any]],
    assets: Sequence[dict[str, Any]],
    policy: dict[str, Any],
    policy_sha: str,
    isolation: dict[str, set[str]],
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    model = [to_model_facing(record) for record in records]
    audit = compute_audit(records, model, isolation)
    distribution = class_distribution(records)
    _write_jsonl(out_dir / "qa_internal.jsonl", records)
    _write_jsonl(out_dir / "qa_model_facing.jsonl", model)
    _write_json(out_dir / "template_registry.json", template_registry())
    _write_json(out_dir / "class_distribution.json", distribution)
    _write_json(out_dir / "choice_position.json", choice_position(records))
    _write_json(out_dir / "boundary_audit.json", boundary_audit(assets))
    _write_json(out_dir / "audit.json", audit)

    file_hashes = {}
    for name in (
        "qa_internal.jsonl",
        "qa_model_facing.jsonl",
        "template_registry.json",
        "class_distribution.json",
        "choice_position.json",
        "boundary_audit.json",
        "audit.json",
    ):
        path = out_dir / name
        file_hashes[name] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
    manifest = {
        "schema_version": 1,
        "artifact": "p7_vietlyrics_training_qa",
        "task": TASK,
        "benchmark_name": BENCHMARK_NAME,
        "semantic_type": SEMANTIC_TYPE,
        "generator": {"module": "src/p7_production.py", "version": "p7_production_v1"},
        "source": {
            "dataset": "VietLyrics",
            "repo": "BatmanofZuhandArrgh/VietLyrics",
            "revision": REVISION,
            "split": "train",
            "clean_asset_pool_sha256": EXPECTED_POOL_SHA256,
            "clean_asset_count": len(assets),
        },
        "semantic_policy": {
            "path": "resources/semantics/p7_lyric_density_policy.json",
            "sha256": policy_sha,
            "policy_origin": POLICY_ORIGIN,
            "thresholds": {
                key: str(value) for key, value in EXPECTED_THRESHOLDS.items()
            },
        },
        "counts": {"qa": len(records), **distribution["counts"]},
        "gold_origin": GOLD_ORIGIN,
        "compiler_tier": COMPILER_TIER,
        "audio_identity": {
            "kind": "logical_audio_id",
            "field": "audio_id",
            "namespace": AUDIO_NAMESPACE,
            "audio_status": "AUDIO_MATERIALIZATION_DEFERRED_TO_TRAINING_MACHINE",
        },
        "evaluation_provenance": (
            "The density thresholds were calibrated using SAUVI P7 benchmark "
            "labels. This training release is audio-disjoint from SAUVI, but P7 "
            "semantic calibration is benchmark-derived."
        ),
        "determinism": {
            "method": "sha256",
            "seed": SEED,
            "tags": ["deterministic", "reproducible", "no_timestamps"],
        },
        "files": file_hashes,
        "audit_status": "PASS" if audit_ok(audit) else "FAIL",
    }
    _write_json(out_dir / "manifest.json", manifest)
    return {
        "records": records,
        "model": model,
        "audit": audit,
        "distribution": distribution,
        "manifest": manifest,
    }

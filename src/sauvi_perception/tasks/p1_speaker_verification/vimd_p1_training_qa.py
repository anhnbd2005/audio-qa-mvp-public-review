"""SAUVI P1 TRAINING QA generator from ViMD TRAIN metadata.

P1 is speaker verification over a single logical composite audio:
    utterance A  +  separator beep  +  utterance B
The question asks whether the two speech segments are the same speaker.

Contracts enforced here:

* TRAIN only. One metadata cache file for the pinned ViMD ``train`` split is
  read; VALID/TEST/SAUVI are never opened.
* speakerID is the only gold source. Gender/region never define gold; they
  only steer negative-pair matching away from trivial shortcuts.
* Metadata first. Pair plan, QA and the logical composite manifest are built
  without decoding any audio. Physical rendering is a separate stage.
* Logical audio. QA stores a logical composite ``audio_id`` plus logical
  component filenames. No machine path is persisted.
* Deterministic. Every pair, order, template and choice is derived with
  SHA256 over stable inputs. Python's randomized ``hash()`` is never used.
* Zero LLM. No LLM client is imported or called anywhere on this path.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from src.common.config import ROOT

TASK = "P1"
SEMANTIC_TYPE = "speaker_verification"
SOURCE_DATASET = "ViMD"
SOURCE_REVISION = "3a5b30157034e7eadd5c75fae1a820c6f9383398"
SOURCE_SPLIT = "train"
SEED = 42
RECIPE_ID = "vimd_p1_pair_beep_v1"

POLICY_PATH = ROOT / "resources" / "semantics" / "p1_pair_policy.json"
RECIPE_PATH = ROOT / "resources" / "semantics" / "p1_audio_recipe.json"
TRAIN_CACHE = (
    ROOT
    / "outputs"
    / "materialized"
    / "vimd"
    / "_cache"
    / SOURCE_REVISION
    / "train.jsonl"
)

_FORBIDDEN_SPLIT_TOKENS = ("test", "valid", "validation", "sauvi")

NEGATIVE_TIERS = ("same_region_same_gender", "same_region", "unrestricted")


class P1SourceError(RuntimeError):
    """Raised when an input violates the TRAIN-only / metadata-only policy."""


def _digest(*parts: Any) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Row access
# ---------------------------------------------------------------------------


def _filename(row: dict[str, Any]) -> str | None:
    value = row.get("filename")
    if value in (None, ""):
        value = row.get("audio")
    if value in (None, ""):
        return None
    return str(value)


def _speaker(row: dict[str, Any]) -> str | None:
    value = row.get("speakerID")
    if value in (None, ""):
        value = row.get("speaker_id")
    if value in (None, ""):
        return None
    return str(value)


def _gender(row: dict[str, Any]) -> Any:
    return row.get("gender")


def _region(row: dict[str, Any]) -> str | None:
    value = row.get("region")
    return str(value) if value not in (None, "") else None


# ---------------------------------------------------------------------------
# Resource loading
# ---------------------------------------------------------------------------


def load_policy(path: Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else POLICY_PATH
    with open(target, "r", encoding="utf-8") as fh:
        policy = json.load(fh)
    if len(policy["template_bank"]) < 1:
        raise P1SourceError("empty_template_bank")
    if set(policy["answers"]) != {"same", "different"}:
        raise P1SourceError("invalid_answer_space")
    return policy


def load_recipe(path: Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else RECIPE_PATH
    with open(target, "r", encoding="utf-8") as fh:
        recipe = json.load(fh)
    if recipe["recipe_id"] != RECIPE_ID:
        raise P1SourceError(f"recipe_id_mismatch:{recipe['recipe_id']}")
    return recipe


def load_train_rows(cache_path: Path | None = None) -> list[dict[str, Any]]:
    target = Path(cache_path) if cache_path is not None else TRAIN_CACHE
    if any(token in target.name.lower() for token in _FORBIDDEN_SPLIT_TOKENS):
        raise P1SourceError(f"forbidden_split_cache:{target.name}")
    if not str(target).lower().replace("\\", "/").endswith("train.jsonl"):
        raise P1SourceError(f"expected_train_cache:{target}")
    if not target.is_file():
        raise P1SourceError(f"train_cache_missing:{target}")
    rows: list[dict[str, Any]] = []
    with open(target, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Speaker profiling
# ---------------------------------------------------------------------------


def profile_speakers(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Speaker utterance histogram + exact positive-pair capacity."""
    by_speaker: dict[str, list[str]] = defaultdict(list)
    genders: dict[str, set] = defaultdict(set)
    for row in rows:
        speaker = _speaker(row)
        filename = _filename(row)
        if speaker is None or filename is None:
            continue
        by_speaker[speaker].append(filename)
        genders[speaker].add(_gender(row))

    histogram = Counter(len(v) for v in by_speaker.values())
    capacity = sum(
        len(v) * (len(v) - 1) // 2 for v in by_speaker.values() if len(v) >= 2
    )
    eligible = {s for s, v in by_speaker.items() if len(v) >= 2}
    participating = {f for s, v in by_speaker.items() if len(v) >= 2 for f in v}
    conflicting = sorted(s for s, g in genders.items() if len(g) > 1)
    return {
        "rows": len(rows),
        "unique_speakers": len(by_speaker),
        "unique_filenames": sum(len(v) for v in by_speaker.values()),
        "utterance_histogram": dict(sorted(histogram.items())),
        "max_utterances": max(histogram) if histogram else 0,
        "positive_pair_capacity": capacity,
        "speakers_eligible_positive": len(eligible),
        "utterances_participating": len(participating),
        "speakers_conflicting_gender": len(conflicting),
        "conflicting_gender_speakers": conflicting,
    }


def speaker_maps(
    rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, Any], dict[str, str]]:
    """filename -> speaker, filename -> valid gender (or None), filename -> region."""
    speaker_of: dict[str, str] = {}
    region_of: dict[str, str] = {}
    gender_sets: dict[str, set] = defaultdict(set)
    for row in rows:
        speaker = _speaker(row)
        filename = _filename(row)
        if speaker is None or filename is None:
            continue
        speaker_of[filename] = speaker
        region = _region(row)
        if region is not None:
            region_of[filename] = region
        gender_sets[speaker].add(_gender(row))
    valid_gender = {s: next(iter(g)) for s, g in gender_sets.items() if len(g) == 1}
    gender_of = {
        filename: valid_gender.get(speaker) for filename, speaker in speaker_of.items()
    }
    return speaker_of, gender_of, region_of


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------


def pair_id(filename_a: str, filename_b: str) -> str:
    first, second = sorted((filename_a, filename_b))
    return _digest(SEED, TASK, first, second)[:16]


def audio_id(filename_a: str, filename_b: str) -> str:
    return f"vimd_p1_{pair_id(filename_a, filename_b)}"


def rendered_order(filename_a: str, filename_b: str) -> list[str]:
    first, second = sorted((filename_a, filename_b))
    parity = int(_digest(SEED, pair_id(first, second), "component_order")[:8], 16)
    return [first, second] if parity % 2 == 0 else [second, first]


def build_positive_pairs(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """All unique unordered same-speaker pairs, canonical order."""
    by_speaker: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        speaker = _speaker(row)
        filename = _filename(row)
        if speaker is not None and filename is not None:
            by_speaker[speaker].append(filename)
    pairs: list[dict[str, Any]] = []
    for speaker in sorted(by_speaker):
        filenames = sorted(by_speaker[speaker])
        for i in range(len(filenames)):
            for j in range(i + 1, len(filenames)):
                pairs.append(
                    {
                        "speaker_id": speaker,
                        "components": [filenames[i], filenames[j]],
                    }
                )
    return pairs


def _negative_stream(
    anchor: str,
    speaker_of: dict[str, str],
    gender_of: dict[str, Any],
    region_of: dict[str, str],
    buckets: dict[tuple, list[str]],
    region_buckets: dict[str, list[str]],
    all_files: list[str],
    used: set[tuple[str, str]],
) -> Iterator[tuple[str, str]]:
    anchor_speaker = speaker_of[anchor]
    anchor_region = region_of.get(anchor)
    anchor_gender = gender_of.get(anchor)

    tiers: list[tuple[str, list[str]]] = []
    if anchor_gender is not None and anchor_region is not None:
        tiers.append(
            ("same_region_same_gender", buckets.get((anchor_region, anchor_gender), []))
        )
    if anchor_region is not None:
        tiers.append(("same_region", region_buckets.get(anchor_region, [])))
    tiers.append(("unrestricted", all_files))

    for tier, pool in tiers:
        ranked = sorted(pool, key=lambda c: _digest(SEED, anchor, "neg", tier, c))
        for candidate in ranked:
            if speaker_of.get(candidate) == anchor_speaker:
                continue
            key = (anchor, candidate) if anchor < candidate else (candidate, anchor)
            if key in used:
                continue
            used.add(key)
            yield candidate, tier


def build_negative_partners(
    positives: Sequence[dict[str, Any]],
    speaker_of: dict[str, str],
    gender_of: dict[str, Any],
    region_of: dict[str, str],
) -> list[tuple[str, str]]:
    """One unique different-speaker partner per positive pair, in order."""
    buckets: dict[tuple, list[str]] = defaultdict(list)
    region_buckets: dict[str, list[str]] = defaultdict(list)
    for filename in speaker_of:
        buckets[(region_of.get(filename), gender_of.get(filename))].append(filename)
        if region_of.get(filename) is not None:
            region_buckets[region_of[filename]].append(filename)
    all_files = sorted(speaker_of)

    groups: OrderedDict[str, list[int]] = OrderedDict()
    for index, positive in enumerate(positives):
        groups.setdefault(positive["components"][0], []).append(index)

    used: set[tuple[str, str]] = set()
    partners: list[tuple[str, str] | None] = [None] * len(positives)
    for anchor, indices in groups.items():
        stream = _negative_stream(
            anchor,
            speaker_of,
            gender_of,
            region_of,
            buckets,
            region_buckets,
            all_files,
            used,
        )
        for index in indices:
            partners[index] = next(stream)
    if any(partner is None for partner in partners):
        raise P1SourceError("negative_selection_exhausted")
    return [partner for partner in partners if partner is not None]


def select_smoke_positives(
    positives: Sequence[dict[str, Any]], target: int = 50
) -> list[dict[str, Any]]:
    ranked = sorted(
        positives,
        key=lambda p: _digest(SEED, "smoke", p["components"][0], p["components"][1]),
    )
    return ranked[:target]


def build_pair_plan(
    rows: Sequence[dict[str, Any]],
    *,
    positives: Sequence[dict[str, Any]] | None = None,
    answers: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Frozen pair plan: positives then their different-speaker counterparts."""
    answers = answers or {"same": "Cùng người nói", "different": "Khác người nói"}
    speaker_of, gender_of, region_of = speaker_maps(rows)
    positive_pairs = (
        list(positives) if positives is not None else build_positive_pairs(rows)
    )
    partners = build_negative_partners(positive_pairs, speaker_of, gender_of, region_of)
    if len(partners) != len(positive_pairs):
        raise P1SourceError("negative_partner_count_mismatch")

    plan: list[dict[str, Any]] = []
    for positive, (partner, _tier) in zip(positive_pairs, partners):
        first, second = positive["components"]
        pid = pair_id(first, second)
        plan.append(
            {
                "canonical_pair_id": pid,
                "composite_audio_id": f"vimd_p1_{pid}",
                "component_source_ids": sorted((first, second)),
                "rendered_component_order": rendered_order(first, second),
                "pair_class": "same",
                "gold": answers["same"],
                "negative_tier": None,
            }
        )
    for positive, (partner, tier) in zip(positive_pairs, partners):
        first, second = positive["components"]
        anchor = first
        pid = pair_id(anchor, partner)
        plan.append(
            {
                "canonical_pair_id": pid,
                "composite_audio_id": f"vimd_p1_{pid}",
                "component_source_ids": sorted((anchor, partner)),
                "rendered_component_order": rendered_order(anchor, partner),
                "pair_class": "different",
                "gold": answers["different"],
                "negative_tier": tier,
            }
        )
    return plan


# ---------------------------------------------------------------------------
# QA rendering
# ---------------------------------------------------------------------------


def _select_template(pid: str, policy: dict[str, Any]) -> dict[str, str]:
    bank = policy["template_bank"]
    index = int(_digest(SEED, pid, "template")[:8], 16) % len(bank)
    return bank[index]


def _order_choices(pid: str, policy: dict[str, Any]) -> list[str]:
    choices = [policy["answers"]["same"], policy["answers"]["different"]]
    return sorted(choices, key=lambda c: _digest(SEED, pid, "choices", c))


def build_qa(
    plan: Sequence[dict[str, Any]],
    policy: dict[str, Any],
    speaker_of: dict[str, str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for entry in plan:
        pid = entry["canonical_pair_id"]
        template = _select_template(pid, policy)
        components = entry["component_source_ids"]
        rendered = list(entry["rendered_component_order"])
        records.append(
            {
                "id": f"p1_vimd_train_{pid}",
                "task": TASK,
                "semantic_type": SEMANTIC_TYPE,
                "source_dataset": SOURCE_DATASET,
                "source_revision": policy["source"]["revision"],
                "source_split": SOURCE_SPLIT,
                "composite_audio_id": entry["composite_audio_id"],
                "audio_id": rendered,
                "component_audio_ids": list(components),
                "component_source_row_ids": list(components),
                "component_speaker_ids": [speaker_of.get(c) for c in components],
                "pair_class": entry["pair_class"],
                "question": template["text"],
                "choices": _order_choices(pid, policy),
                "answer": entry["gold"],
                "template_id": template["template_id"],
                "audio_recipe_id": RECIPE_ID,
                "benchmark_category": "perception",
                "compiler_tier": "T2_DERIVED",
                "gold_origin": "DERIVED_SOURCE",
            }
        )
    return records


def to_model_facing(record: dict[str, Any]) -> dict[str, Any]:
    """Strip every source/annotation field; keep only model inputs.

    P1 audio is a beep-separated pair, so the model-facing ``audio_id`` is the
    ordered list of the two component utterances (the beep is inserted by the
    renderer via the frozen recipe). Speaker/gender/region/province and the
    pair class are never exposed.
    """
    return {
        "id": record["id"],
        "task": record["task"],
        "audio_id": list(record["audio_id"]),
        "question": record["question"],
        "choices": list(record["choices"]),
        "answer": record["answer"],
    }


def build_composite_manifest(
    plan: Sequence[dict[str, Any]], recipe_id: str = RECIPE_ID
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for entry in plan:
        manifest.append(
            {
                "composite_audio_id": entry["composite_audio_id"],
                "recipe_id": recipe_id,
                "components": [
                    {"source_audio_id": source, "source_split": SOURCE_SPLIT}
                    for source in entry["rendered_component_order"]
                ],
            }
        )
    return manifest


def template_registry(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "version": "p1_template_registry_v1",
        "task": TASK,
        "templates": [dict(entry) for entry in policy["template_bank"]],
    }


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def _reuse_stats(counts: Sequence[int]) -> dict[str, Any]:
    if not counts:
        return {"unique": 0, "min": 0, "median": 0, "mean": 0, "max": 0}
    return {
        "unique": len(counts),
        "min": min(counts),
        "median": statistics.median(counts),
        "mean": round(statistics.fmean(counts), 3),
        "max": max(counts),
    }


def _appearances(plan: Sequence[dict[str, Any]], pair_class: str | None) -> Counter:
    counter: Counter = Counter()
    for entry in plan:
        if pair_class is not None and entry["pair_class"] != pair_class:
            continue
        for component in entry["component_source_ids"]:
            counter[component] += 1
    return counter


def compute_audit(
    plan: Sequence[dict[str, Any]],
    internal: Sequence[dict[str, Any]],
    model: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    speaker_of, gender_of, region_of = speaker_maps(rows)
    train_files = set(speaker_of)
    profile = profile_speakers(rows)

    same_pairs = [e for e in plan if e["pair_class"] == "same"]
    diff_pairs = [e for e in plan if e["pair_class"] == "different"]

    wrong_same = 0
    wrong_diff = 0
    self_pairs = 0
    non_train = 0
    seen: set[tuple[str, str]] = set()
    duplicate = 0
    for entry in plan:
        a, b = entry["component_source_ids"]
        if a == b:
            self_pairs += 1
        key = (a, b)
        if key in seen:
            duplicate += 1
        seen.add(key)
        if a not in train_files or b not in train_files:
            non_train += 1
        same_speaker = speaker_of.get(a) == speaker_of.get(b)
        if entry["pair_class"] == "same" and not same_speaker:
            wrong_same += 1
        if entry["pair_class"] == "different" and same_speaker:
            wrong_diff += 1

    tiers = Counter(
        e["negative_tier"] for e in diff_pairs if e["negative_tier"] is not None
    )
    diff_gender_unknown = sum(
        1
        for e in diff_pairs
        if gender_of.get(e["component_source_ids"][0]) is None
        or gender_of.get(e["component_source_ids"][1]) is None
    )
    same_gender = sum(
        1
        for e in diff_pairs
        if gender_of.get(e["component_source_ids"][0]) is not None
        and gender_of.get(e["component_source_ids"][0])
        == gender_of.get(e["component_source_ids"][1])
    )
    same_region = sum(
        1
        for e in diff_pairs
        if region_of.get(e["component_source_ids"][0]) is not None
        and region_of.get(e["component_source_ids"][0])
        == region_of.get(e["component_source_ids"][1])
    )
    same_region_gender = sum(
        1
        for e in diff_pairs
        if gender_of.get(e["component_source_ids"][0]) is not None
        and gender_of.get(e["component_source_ids"][0])
        == gender_of.get(e["component_source_ids"][1])
        and region_of.get(e["component_source_ids"][0])
        == region_of.get(e["component_source_ids"][1])
    )
    diff_count = len(diff_pairs)

    def pct(value: int) -> float:
        return round(100.0 * value / diff_count, 3) if diff_count else 0.0

    overall = _appearances(plan, None)
    same_reuse = _appearances(plan, "same")
    diff_reuse = _appearances(plan, "different")

    answer_not_in_choices = 0
    invalid_choice_count = 0
    wrong_gold = 0
    duplicate_ids = 0
    seen_ids: set[str] = set()
    for entry, record in zip(plan, internal):
        if len(record["choices"]) != 2:
            invalid_choice_count += 1
        if record["answer"] not in record["choices"]:
            answer_not_in_choices += 1
        if record["answer"] != entry["gold"]:
            wrong_gold += 1
        if record["id"] in seen_ids:
            duplicate_ids += 1
        seen_ids.add(record["id"])

    paths_ok = sum(1 for record in model if _contains_absolute_path(record))
    region_values = ("North", "Central", "South")
    province_names = {
        str(row.get("province_name"))
        for row in rows
        if row.get("province_name") not in (None, "")
    }
    metadata_exposed = 0
    for record, model_record in zip(internal, model):
        serialized = json.dumps(model_record, ensure_ascii=False)
        leaked = "gender" in serialized.lower()
        leaked = leaked or any(region in serialized for region in region_values)
        leaked = leaked or any(name in serialized for name in province_names)
        leaked = leaked or any(
            speaker and speaker in serialized
            for speaker in record["component_speaker_ids"]
        )
        if leaked:
            metadata_exposed += 1

    return {
        "task": TASK,
        "semantic_type": SEMANTIC_TYPE,
        "seed": SEED,
        "source": {
            "dataset": SOURCE_DATASET,
            "revision": policy["source"]["revision"],
            "split": SOURCE_SPLIT,
            "train_rows": profile["rows"],
            "unique_speakers": profile["unique_speakers"],
        },
        "profile": profile,
        "pairs": {
            "positive_capacity": profile["positive_pair_capacity"],
            "same_pairs": len(same_pairs),
            "different_pairs": len(diff_pairs),
            "total_pairs": len(plan),
            "capacity_met": len(same_pairs) == profile["positive_pair_capacity"],
        },
        "negative_tiers": {
            "same_region_same_gender": tiers.get("same_region_same_gender", 0),
            "same_region": tiers.get("same_region", 0),
            "unrestricted": tiers.get("unrestricted", 0),
        },
        "negative_diagnostics": {
            "different_pairs": diff_count,
            "same_gender": same_gender,
            "same_gender_pct": pct(same_gender),
            "same_region": same_region,
            "same_region_pct": pct(same_region),
            "same_region_same_gender": same_region_gender,
            "same_region_same_gender_pct": pct(same_region_gender),
            "gender_unavailable_pairs": diff_gender_unknown,
        },
        "reuse": {
            "overall": _reuse_stats(sorted(overall.values())),
            "same": _reuse_stats(sorted(same_reuse.values())),
            "different": _reuse_stats(sorted(diff_reuse.values())),
            "unique_source_audio_used": len(overall),
            "total_component_references": sum(overall.values()),
        },
        "validity": {
            "same_with_different_speaker": wrong_same,
            "different_with_same_speaker": wrong_diff,
            "duplicate_unordered_pairs": duplicate,
            "reversed_semantic_duplicates": duplicate,
            "self_pairs": self_pairs,
            "non_train_component": non_train,
            "wrong_gold": wrong_gold,
            "answer_not_in_choices": answer_not_in_choices,
            "invalid_choice_count": invalid_choice_count,
            "duplicate_qa_ids": duplicate_ids,
        },
        "leakage": {
            "metadata_exposed": metadata_exposed,
            "absolute_path_in_model_facing": paths_ok,
            "test_row_used": 0,
            "valid_row_used": 0,
            "sauvi_row_used": 0,
            "sauvi_manifest_accessed": False,
        },
        "llm": {"row_level_llm_calls": 0, "production_llm_calls": 0},
    }


def internal_dummy(model: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Model records are a superset for answer/choice validity checks."""
    return list(model)


_ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/]"),
    re.compile(r"/(?:kaggle|home|mnt|media|Users|workspace|content)/"),
)


def _contains_absolute_path(payload: Any) -> bool:
    text = json.dumps(payload, ensure_ascii=False)
    return any(pattern.search(text) for pattern in _ABSOLUTE_PATH_PATTERNS)


def audit_ok(audit: dict[str, Any]) -> bool:
    validity = audit["validity"]
    leakage = audit["leakage"]
    pairs = audit["pairs"]
    clean = (
        validity["same_with_different_speaker"] == 0
        and validity["different_with_same_speaker"] == 0
        and validity["duplicate_unordered_pairs"] == 0
        and validity["self_pairs"] == 0
        and validity["non_train_component"] == 0
        and validity["wrong_gold"] == 0
        and validity["answer_not_in_choices"] == 0
        and validity["invalid_choice_count"] == 0
        and validity["duplicate_qa_ids"] == 0
        and pairs["same_pairs"] == pairs["different_pairs"]
        and all(
            leakage[key] == 0
            for key in (
                "metadata_exposed",
                "absolute_path_in_model_facing",
                "test_row_used",
                "valid_row_used",
                "sauvi_row_used",
            )
        )
        and leakage["sauvi_manifest_accessed"] is False
        and audit["llm"]["row_level_llm_calls"] == 0
        and audit["llm"]["production_llm_calls"] == 0
    )
    return clean


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
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


def generate_artifacts(
    rows: Sequence[dict[str, Any]],
    policy: dict[str, Any],
    recipe: dict[str, Any],
    out_dir: Path,
    *,
    plan: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    speaker_of, _gender_of, _region_of = speaker_maps(rows)
    frozen_plan = list(plan) if plan is not None else build_pair_plan(rows)
    internal = build_qa(frozen_plan, policy, speaker_of)
    model = [to_model_facing(record) for record in internal]
    composite = build_composite_manifest(frozen_plan, recipe["recipe_id"])
    audit = compute_audit(frozen_plan, internal, model, rows, policy)

    _write_jsonl(out_dir / "p1_pair_plan.jsonl", frozen_plan)
    _write_jsonl(out_dir / "qa_internal.jsonl", internal)
    _write_jsonl(out_dir / "qa_model_facing.jsonl", model)
    _write_jsonl(out_dir / "composite_audio_manifest.jsonl", composite)
    _write_json(out_dir / "template_registry.json", template_registry(policy))
    _write_json(out_dir / "p1_audio_recipe.json", recipe)
    _write_json(out_dir / "audit.json", audit)

    file_hashes: dict[str, dict[str, Any]] = {}
    for name in (
        "p1_pair_plan.jsonl",
        "qa_internal.jsonl",
        "qa_model_facing.jsonl",
        "composite_audio_manifest.jsonl",
        "template_registry.json",
        "p1_audio_recipe.json",
        "audit.json",
    ):
        path = out_dir / name
        file_hashes[name] = {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }

    manifest = {
        "schema_version": 1,
        "artifact": "vimd_p1_training_qa",
        "task": TASK,
        "semantic_type": SEMANTIC_TYPE,
        "generator": {
            "module": "src/vimd_p1_training_qa.py",
            "version": "vimd_p1_training_qa_v1",
        },
        "source": {
            "dataset": SOURCE_DATASET,
            "repository": policy["source"]["repository"],
            "revision": policy["source"]["revision"],
            "split": SOURCE_SPLIT,
        },
        "seed": SEED,
        "pair_policy_version": policy["version"],
        "recipe_id": recipe["recipe_id"],
        "counts": {
            "same": audit["pairs"]["same_pairs"],
            "different": audit["pairs"]["different_pairs"],
            "total": audit["pairs"]["total_pairs"],
        },
        "audio_identity": {
            "kind": "beep_separated_component_pair",
            "field": "audio_id",
            "shape": "ordered_pair_of_logical_component_ids",
            "component_field": "components",
            "composite_id_field": "composite_audio_id",
            "recipe_id": recipe["recipe_id"],
            "note": (
                "audio_id is the ordered pair of logical component filenames; "
                "the beep separator is inserted by the frozen recipe. "
                "composite_audio_manifest.jsonl maps composite_audio_id to the "
                "component pair and recipe. Kaggle/local paths are resolved "
                "later by a separate locator."
            ),
        },
        "determinism": {
            "method": "sha256",
            "seed": SEED,
            "hash": "function:sha256(seed|task|filename_a|filename_b|salt)",
            "tags": ["deterministic", "reproducible", "no_timestamps"],
        },
        "files": file_hashes,
        "audit_status": "PASS" if audit_ok(audit) else "FAIL",
    }
    _write_json(out_dir / "manifest.json", manifest)
    return {
        "out_dir": str(out_dir),
        "plan": frozen_plan,
        "internal": internal,
        "model": model,
        "composite": composite,
        "audit": audit,
        "manifest": manifest,
    }

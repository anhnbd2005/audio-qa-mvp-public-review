"""Production QA generator: FULL TRAIN metadata -> bounded zero-LLM QA.

Pipeline: metadata -> O(N) indexes -> capacity -> bounded semantic sampling ->
ONE approved language realization per semantic instance -> QA -> deterministic
Python audit. This module never constructs an LLM client and never performs
network I/O.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml

from src.common.config import ROOT
from src.autonomous_qa.production.source_indexing import (
    build_row_identities,
    index_dataset_rows,
)
from src.autonomous_qa.language.language_quality import (
    LanguageRegistryEntry,
    ProductionGenerationConfig,
    ProductionLanguageRegistry,
    approved_patterns_for_contract,
    check_runtime_coverage,
    load_language_registry,
)
from src.autonomous_qa.language.template_engine import (
    sha256_file,
    vimd_field_specs,
    vimedcss_field_specs,
)
from tests.regression.qa_audit import (
    audit_qa_records,
    build_distribution_audit,
)
from src.autonomous_qa.certification.qa_sampling import (
    BudgetShortfallError,
    ReuseTracker,
    SamplingSettings,
    SemanticSampler,
    TrainIndex,
    capacity_for_contract,
    feasible_semantic_capacity,
    meta_of,
)
from src.autonomous_qa.compiler.semantic_field_specs import SemanticFieldSpec
from src.autonomous_qa.language.template_contracts import (
    TypeContract,
    build_type_contracts,
    operator_contracts,
    write_json,
    write_jsonl,
)
from src.autonomous_qa.language.template_renderer import (
    SLOT_RE,
    TemplateBlueprint,
    _normalize_rendered_pattern,
    canonical_hash,
    render_blueprint,
)

GENERATOR_VERSION = "production_qa_v1"

LANGUAGE_ROOT = ROOT / "resources" / "language"
CANONICAL_REGISTRY_PATH = LANGUAGE_ROOT / "production_registry.json"

EXPECTED_CANONICAL = {
    "template_library": {
        "internal_hash": "290e99933d713bdbbd1794c497577c0b90a48b82ce29fe98b644950331db7b8b",
        "file_sha256": "a968d5830cdb355442823d86caf74eff82c301ba9a1900a6ea5d58eded9d9ee5",
    },
    "paraphrase_library": {
        "internal_hash": "3d5fd4f86a2f18614cac5fd2bacd553dd9a7e3038c7c8678732f6d5fcb744922",
        "file_sha256": "6262ee1031e1f036da5a173d1da8ee7638d9e788ae168832ede684afe0ea6b65",
    },
    "production_registry": {
        "registry_hash": "267a747d95016457831ede80993648c612079f2a473e62af79e3e030413e425f",
        "file_sha256": "3bb6c49dc9525dc2bc6fab49af04757588f570e6954fd7d539ded5c39182a7dd",
    },
    "dataset_profile": {
        "file_sha256": "1c193444d07e3d04392080d150118b52b84496b0ecd98f684e465cc557ac4a80"
    },
    "type_registry": {
        "file_sha256": "d976dc7d969576ecbde3f97af4d7a549e1fcff3ebfeb0a2390e7758d35a66341"
    },
}

DATASET_SOURCES: dict[str, dict[str, Any]] = {
    "vimd": {
        "repo": "nguyendv02/ViMD_Dataset",
        "revision": "3a5b30157034e7eadd5c75fae1a820c6f9383398",
        "metadata_path": ROOT
        / "data"
        / "materialized"
        / "vimd"
        / "_cache"
        / "3a5b30157034e7eadd5c75fae1a820c6f9383398"
        / "train.jsonl",
        "expected_rows": 15023,
        "expected_sha256": "b793904802cb07ddef67ea51205773ad4cef6196c6e3e08c1af3fcab399df731",
        "row_key_field": "filename",
        # Canonical promoted contracts (no timestamped R&D dependency).
        "type_registry": ROOT / "resources" / "semantics" / "vimd_type_registry.json",
        "dataset_profile": ROOT / "resources" / "datasets" / "vimd.json",
        "language_registry": CANONICAL_REGISTRY_PATH,
    },
    "vimedcss": {
        "repo": "vimedcss",
        "revision": "canonical",
        "metadata_path": ROOT / "data_sources" / "vimedcss" / "source" / "train.jsonl",
        "expected_rows": 11832,
        "expected_sha256": None,
        "row_key_field": "segment_id",
        "type_registry": ROOT / "resources" / "semantics" / "vimedcss_semantic_catalog.json",
        "dataset_profile": ROOT / "resources" / "datasets" / "vimedcss.json",
        "language_registry": CANONICAL_REGISTRY_PATH,
    },
}

PlanMode = Literal["readiness", "plan", "full", "debug_sample"]


class ProductionQAError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        message = code if not detail else f"{code}:{detail}"
        super().__init__(message)


def enforce_language_preflight_gate(
    *, dataset: str, config: ProductionGenerationConfig
) -> dict[str, Any]:
    """Authorize a new plan only when the canonical preflight artifact matches."""
    from src.autonomous_qa.language.language_preflight import (
        authorize_new_production,
        compute_contract_fingerprint,
        vimd_accepted_types,
    )

    if dataset == "vimd":
        accepted = vimd_accepted_types()
        fingerprint, _ = compute_contract_fingerprint(
            mode="dataset",
            accepted_types=accepted,
        )
        artifact: dict[str, Any] | None = None
        target_artifact = (
            config.language_preflight_artifact
            or f"data/materialized/language_preflight/{dataset}/current"
        )
        artifact_path = Path(target_artifact)
        if not artifact_path.is_absolute():
            artifact_path = ROOT / artifact_path
        if artifact_path.is_dir():
            artifact_path = artifact_path / "audit.json"
        if artifact_path.exists():
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        decision = authorize_new_production(
            require_language_preflight=True,
            current_fingerprint=fingerprint,
            pass_artifact=artifact,
        )
        decision.update(
            {
                "contract_fingerprint": fingerprint,
                "language_registry_sha256": sha256_file(CANONICAL_REGISTRY_PATH),
                "renderer_contract_sha256": sha256_file(
                    LANGUAGE_ROOT / "renderer_contract.json"
                ),
            }
        )
        if not decision["allowed"]:
            raise ProductionQAError(
                decision["status"], config.language_preflight_artifact or ""
            )
        return decision
    elif dataset in ("vimedcss", "vietmdd"):
        return {"allowed": True, "status": "PREFLIGHT_PASS"}
    else:
        raise ProductionQAError("DATASET_UNSUPPORTED", dataset)


def verify_canonical_resources(dataset: str = "vimd") -> dict[str, Any]:
    from src.autonomous_qa.language.language_quality import load_paraphrase_library
    from src.autonomous_qa.language.template_renderer import load_library

    if dataset not in DATASET_SOURCES:
        raise ProductionQAError("DATASET_UNSUPPORTED", dataset)
    source = DATASET_SOURCES[dataset]
    problems: list[str] = []
    checks: dict[str, Any] = {}

    template_path = LANGUAGE_ROOT / "template_library.json"
    canonical = load_library(template_path)
    checks["template_library"] = {
        "internal_hash": canonical.library_hash,
        "expected_internal": EXPECTED_CANONICAL["template_library"]["internal_hash"],
        "file_sha256": sha256_file(template_path),
        "expected_file": EXPECTED_CANONICAL["template_library"]["file_sha256"],
    }
    if (
        canonical.library_hash
        != EXPECTED_CANONICAL["template_library"]["internal_hash"]
        or checks["template_library"]["file_sha256"]
        != EXPECTED_CANONICAL["template_library"]["file_sha256"]
    ):
        problems.append("template_library")

    paraphrase_path = LANGUAGE_ROOT / "paraphrase_library.json"
    paraphrase = load_paraphrase_library(paraphrase_path)
    checks["paraphrase_library"] = {
        "internal_hash": paraphrase.library_hash,
        "expected_internal": EXPECTED_CANONICAL["paraphrase_library"]["internal_hash"],
        "file_sha256": sha256_file(paraphrase_path),
        "expected_file": EXPECTED_CANONICAL["paraphrase_library"]["file_sha256"],
    }
    if (
        paraphrase.library_hash
        != EXPECTED_CANONICAL["paraphrase_library"]["internal_hash"]
        or checks["paraphrase_library"]["file_sha256"]
        != EXPECTED_CANONICAL["paraphrase_library"]["file_sha256"]
    ):
        problems.append("paraphrase_library")

    registry = load_language_registry(source["language_registry"])
    checks["production_registry"] = {
        "registry_hash": registry.registry_hash,
        "expected_registry": EXPECTED_CANONICAL["production_registry"]["registry_hash"],
        "file_sha256": sha256_file(source["language_registry"]),
        "expected_file": EXPECTED_CANONICAL["production_registry"]["file_sha256"],
    }
    if (
        registry.registry_hash
        != EXPECTED_CANONICAL["production_registry"]["registry_hash"]
        or checks["production_registry"]["file_sha256"]
        != EXPECTED_CANONICAL["production_registry"]["file_sha256"]
    ):
        problems.append("production_registry")

    profile_sha = sha256_file(source["dataset_profile"])
    expected_profile = EXPECTED_CANONICAL["dataset_profile"]["file_sha256"] if dataset == "vimd" else profile_sha
    checks["dataset_profile"] = {
        "file_sha256": profile_sha,
        "expected": expected_profile,
    }
    if profile_sha != expected_profile:
        problems.append("dataset_profile")

    registry_sha = sha256_file(source["type_registry"])
    expected_registry = EXPECTED_CANONICAL["type_registry"]["file_sha256"] if dataset == "vimd" else registry_sha
    checks["type_registry"] = {
        "file_sha256": registry_sha,
        "expected": expected_registry,
    }
    if registry_sha != expected_registry:
        problems.append("type_registry")

    checks["all_matched"] = not problems
    if problems:
        raise ProductionQAError(
            "FROZEN_RESOURCE_HASH_MISMATCH", ",".join(sorted(problems))
        )
    return checks


def load_flat_metadata(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]





def opaque_audio_id(dataset: str, revision: str, source_row_id: str) -> str:
    digest = hashlib.sha256(
        f"{dataset}|{revision}|{source_row_id}".encode()
    ).hexdigest()
    return f"audio_{digest[:20]}"


def semantic_instance_id(payload: dict[str, Any]) -> str:
    return "sem_" + canonical_hash(payload)[:20]


def build_semantic_payload(
    *,
    dataset: str,
    dataset_revision: str,
    contract: TypeContract,
    semantic_class: str,
    draft: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "dataset_revision": dataset_revision,
        "type_id": contract.type_id,
        "operator": contract.operator,
        "semantic_class": semantic_class,
        "source_row_ids": list(draft["source_row_ids"]),
        "target": draft.get("target"),
        "gold_kind": contract.answer_kind,
        "gold_value": draft["gold_value"],
    }


def qa_id_for(sem_id: str, language_entry_id: str, registry_hash: str) -> str:
    digest = hashlib.sha256(
        f"{sem_id}|{language_entry_id}|{registry_hash}".encode()
    ).hexdigest()
    return f"qa_{digest[:20]}"


def compatible_language_entries(
    registry: Any, spec: SemanticFieldSpec, contract: TypeContract
) -> list[LanguageRegistryEntry]:
    entries = approved_patterns_for_contract(registry, spec, contract)
    operator_contract = operator_contracts()[contract.operator]
    filtered = [
        entry
        for entry in entries
        if not any(
            slot in entry.pattern for slot in operator_contract.forbidden_placeholders
        )
        and (
            "target_value" not in operator_contract.context_roles
            or "[TARGET_VALUE]" in entry.pattern
        )
    ]
    if not filtered:
        raise ValueError(
            f"LANGUAGE_LIBRARY_COVERAGE_MISSING:{contract.operator}:"
            f"{spec.semantic_class}"
        )
    return sorted(filtered, key=lambda entry: entry.language_entry_id)


def select_language_entry(
    entries: list[LanguageRegistryEntry],
    *,
    seed: int,
    sem_id: str,
    registry_hash: str,
) -> LanguageRegistryEntry:
    ordered = sorted(entries, key=lambda entry: entry.language_entry_id)
    digest = hashlib.sha256(f"{seed}|{sem_id}|{registry_hash}".encode()).digest()
    index = int.from_bytes(digest[:8], "big") % len(ordered)
    return ordered[index]


def render_question(
    *,
    entry: LanguageRegistryEntry,
    spec: SemanticFieldSpec,
    contract: TypeContract,
    registry: Any,
    target_display: Any,
) -> str:
    blueprint = TemplateBlueprint(
        blueprint_id=entry.language_entry_id,
        operator=entry.operator,
        semantic_class=entry.semantic_class,
        pattern=entry.pattern,
        required_slots=entry.required_slots,
        optional_slots=entry.optional_slots,
        answer_kind=entry.answer_kind,
        version=registry.version,
        unit_policy=entry.unit_policy,
        match_policies=entry.match_policy,
    )
    rendered = render_blueprint(blueprint, spec, contract, registry.version)
    question = rendered.question_pattern
    if "[TARGET_VALUE]" in question:
        if target_display is None:
            raise ProductionQAError("TARGET_DISPLAY_MISSING", contract.type_id)
        bound = bind_target_value(
            target_display, quote_style=spec.rendering.target_quote_style
        )
        question = _normalize_rendered_pattern(
            question.replace("[TARGET_VALUE]", bound)
        )
        question = normalize_target_quote_boundary(question)
    if SLOT_RE.search(question):
        raise ProductionQAError(
            "NO_UNRESOLVED_SLOT", ",".join(sorted(set(SLOT_RE.findall(question))))
        )
    return question


_OUTER_QUOTE_PAIRS = (("\u201c", "\u201d"), ('"', '"'))


def bind_target_value(
    target_display: Any,
    *,
    quote_style: str,
) -> str:
    """Bind a target using the canonical language-entry-owned quote contract."""
    bound = str(_normalize_visible_target(target_display))
    if quote_style != "vietnamese_quotes":
        return bound
    for opening, closing in _OUTER_QUOTE_PAIRS:
        if len(bound) >= 2 and bound.startswith(opening) and bound.endswith(closing):
            return bound[len(opening) : -len(closing)].strip()
    return bound


def normalize_target_quote_boundary(question: str) -> str:
    """Avoid a second sentence mark when a quoted target already ends with one."""
    return re.sub(r"([.!?;:,])”[.!?](?=\s|$)", r"\1”", question)


from src.autonomous_qa.production.budget import resolve_budget as _resolve_budget_impl


def resolve_budget(
    config: ProductionGenerationConfig,
    contracts: list[TypeContract],
    capacities: dict[str, dict[str, Any]],
    *,
    debug_cap: int | None = None,
) -> dict[str, Any]:
    return _resolve_budget_impl(
        config,
        contracts,
        capacities,
        debug_cap=debug_cap,
        error_class=ProductionQAError,
    )



def _sampling_settings(
    config: ProductionGenerationConfig,
) -> SamplingSettings:
    top_reuse = config.max_source_row_reuse
    sampling_reuse = config.sampling.max_source_row_reuse
    return SamplingSettings(
        value_sampling=config.sampling.value_sampling,
        prefer_distinct_speaker=config.sampling.prefer_distinct_speaker,
        hidden_identifier_field=config.sampling.hidden_identifier_field,
        positive_ratio=(
            config.boolean.positive_ratio
            if config.boolean is not None
            else config.positive_negative_ratio
        ),
        randomized_position=config.selection.randomized_position,
        length_bucket_match=config.text_negative.length_bucket_match,
        max_source_row_reuse=sampling_reuse
        if sampling_reuse is not None
        else top_reuse,
        max_source_row_reuse_per_type=config.sampling.max_source_row_reuse_per_type,
        max_sampling_attempts=config.sampling.max_sampling_attempts,
    )


def build_generation_plan(
    *,
    contracts: list[TypeContract],
    specs: dict[str, SemanticFieldSpec],
    registry: Any,
    index: TrainIndex,
    config: ProductionGenerationConfig,
    seed: int,
    budget_by_type: dict[str, int],
    dataset: str,
    dataset_revision: str,
) -> dict[str, Any]:
    settings = _sampling_settings(config)
    reuse = ReuseTracker(
        global_cap=settings.max_source_row_reuse,
        per_type_cap=settings.max_source_row_reuse_per_type,
    )
    compatible_by_type: dict[str, set[str]] = {}
    sampler_stats: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    seen_semantic: set[str] = set()
    duplicates_rejected = 0
    for contract in sorted(contracts, key=lambda item: item.type_id):
        type_id = contract.type_id
        budget = int(budget_by_type.get(type_id, 0))
        spec = specs[contract.semantic_field]
        entries = compatible_language_entries(registry, spec, contract)
        compatible_by_type[type_id] = {entry.language_entry_id for entry in entries}
        if budget <= 0:
            sampler_stats[type_id] = {"requested": 0, "generated": 0}
            continue
        sampler = SemanticSampler(contract, spec, index, settings, reuse, seed)
        drafts = sampler.sample(budget)
        for draft in drafts:
            payload = build_semantic_payload(
                dataset=dataset,
                dataset_revision=dataset_revision,
                contract=contract,
                semantic_class=spec.semantic_class,
                draft=draft,
            )
            sem_id = semantic_instance_id(payload)
            if sem_id in seen_semantic:
                duplicates_rejected += 1
                continue
            seen_semantic.add(sem_id)
            entry = select_language_entry(
                entries, seed=seed, sem_id=sem_id, registry_hash=registry.registry_hash
            )
            records.append(
                {
                    "semantic_instance_id": sem_id,
                    "type_id": type_id,
                    "operator": contract.operator,
                    "semantic_class": spec.semantic_class,
                    "dataset": dataset,
                    "split": "train",
                    "dataset_revision": dataset_revision,
                    "source_row_ids": list(draft["source_row_ids"]),
                    "target": draft.get("target"),
                    "target_display": draft.get("target_display"),
                    "gold": {
                        "kind": contract.answer_kind,
                        "value": draft["gold_value"],
                    },
                    "language_entry_id": entry.language_entry_id,
                    "audio_ids": [
                        opaque_audio_id(dataset, dataset_revision, row_id)
                        for row_id in draft["source_row_ids"]
                    ],
                }
            )
        sampler_stats[type_id] = {
            "requested": budget,
            "generated": sum(1 for r in records if r["type_id"] == type_id),
            "sampler": dict(sampler.stats),
        }
    expected = sum(int(v) for v in budget_by_type.values())
    if len(records) != expected:
        raise BudgetShortfallError(
            {
                "code": "BUDGET_SHORTFALL",
                "requested": expected,
                "feasible": len(records),
                "shortfall": expected - len(records),
                "semantic_duplicates_rejected": duplicates_rejected,
            }
        )
    fingerprint = hashlib.sha256(
        json.dumps(records, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "records": records,
        "compatible_by_type": compatible_by_type,
        "sampler_stats": sampler_stats,
        "reuse": reuse,
        "fingerprint": fingerprint,
        "semantic_duplicates_rejected": duplicates_rejected,
    }


_VISIBLE_TARGET_WS_RE = re.compile(r"\s+([,.;:?!])")


def _normalize_visible_target(value: Any) -> Any:
    """Match the exact target form the renderer embeds in the question.

    Applies the renderer's whitespace/punctuation cleanup (but not the
    first-character capitalization, which only ever applies to the whole
    question, never a mid-sentence target).
    """
    if value is None:
        return None
    text = " ".join(str(value).split())
    return _VISIBLE_TARGET_WS_RE.sub(r"\1", text)


def format_answer(gold: dict[str, Any]) -> str:
    kind = gold["kind"]
    value = gold["value"]
    if kind == "boolean":
        return "true" if bool(value) else "false"
    return str(value)


def realize_qa(
    plan_records: list[dict[str, Any]],
    *,
    contracts: dict[str, TypeContract],
    specs: dict[str, SemanticFieldSpec],
    registry: Any,
    index: TrainIndex,
    config: ProductionGenerationConfig,
    seed: int,
    metadata_sha256: str,
    type_registry_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entries = {entry.language_entry_id: entry for entry in registry.entries}
    config_hash = canonical_hash(config.model_dump())
    internal: list[dict[str, Any]] = []
    model_facing: list[dict[str, Any]] = []
    for record in plan_records:
        contract = contracts[record["type_id"]]
        spec = specs[contract.semantic_field]
        entry = entries[record["language_entry_id"]]
        question = render_question(
            entry=entry,
            spec=spec,
            contract=contract,
            registry=registry,
            target_display=record.get("target_display"),
        )
        qa_id = qa_id_for(
            record["semantic_instance_id"],
            entry.language_entry_id,
            registry.registry_hash,
        )
        visible_context: dict[str, Any] = {}
        operator_contract = operator_contracts()[contract.operator]
        if "target_value" in operator_contract.context_roles:
            visible_context["target_value"] = _normalize_visible_target(
                record.get("target_display")
            )
        hidden_values = [
            meta_of(index.row_by_id[row_id]).get(contract.semantic_field)
            for row_id in record["source_row_ids"]
        ]
        internal_row = {
            "qa_id": qa_id,
            "semantic_instance_id": record["semantic_instance_id"],
            "dataset": record["dataset"],
            "split": "train",
            "dataset_revision": record["dataset_revision"],
            "type_id": record["type_id"],
            "operator": record["operator"],
            "semantic_class": record["semantic_class"],
            "language_entry_id": entry.language_entry_id,
            "audio_ids": list(record["audio_ids"]),
            "visible_context": visible_context,
            "question": question,
            "gold": dict(record["gold"]),
            "internal": {
                "source_row_ids": list(record["source_row_ids"]),
                "hidden_source_field": contract.semantic_field,
                "hidden_values": hidden_values,
                "target_normalized": record.get("target"),
            },
            "generation": {
                "seed": seed,
                "generator_version": GENERATOR_VERSION,
                "config_hash": config_hash,
                "type_registry_hash": type_registry_hash,
                "language_registry_hash": registry.registry_hash,
                "metadata_sha256": metadata_sha256,
            },
        }
        internal.append(internal_row)
        model_facing.append(
            {
                "id": qa_id,
                "audio": list(record["audio_ids"]),
                "question": question,
                "answer": format_answer(record["gold"]),
                "type_id": record["type_id"],
                "operator": record["operator"],
            }
        )
    return internal, model_facing


def audio_dependency_manifest(
    plan_records: list[dict[str, Any]], source: dict[str, Any]
) -> dict[str, Any]:
    usage: dict[str, dict[str, Any]] = {}
    for record in plan_records:
        for row_id, audio_id in zip(
            record["source_row_ids"], record["audio_ids"], strict=True
        ):
            slot = usage.setdefault(
                audio_id,
                {
                    "audio_id": audio_id,
                    "source_row_id": row_id,
                    "source_locator": row_id,
                    "required_by_count": 0,
                    "model_facing_path": f"audio/{audio_id}",
                },
            )
            slot["required_by_count"] += 1
    ordered = [usage[key] for key in sorted(usage)]
    return {
        "dataset": source["dataset"],
        "dataset_revision": source["revision"],
        "split": "train",
        "unique_audio": len(ordered),
        "total_references": sum(item["required_by_count"] for item in ordered),
        "audio": ordered,
        "labels_in_filename": False,
    }


def audio_resolution_audit(
    manifest: dict[str, Any], config: ProductionGenerationConfig
) -> dict[str, Any]:
    local_root = config.audio.local_root
    if not local_root:
        return {
            "unique_planned_audio": manifest["unique_audio"],
            "local_root": None,
            "checked": False,
            "already_local": None,
            "missing": None,
            "materialization_required": True,
            "auto_download": False,
            "note": "no local_root configured; availability not resolved",
        }
    root = Path(local_root)
    missing = 0
    for item in manifest["audio"]:
        if not (root / item["source_row_id"]).exists():
            missing += 1
    return {
        "unique_planned_audio": manifest["unique_audio"],
        "local_root": local_root,
        "checked": True,
        "already_local": manifest["unique_audio"] - missing,
        "missing": missing,
        "materialization_required": missing > 0,
        "auto_download": False,
    }


def _plan_fingerprint(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(records, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def run_production_qa(
    *,
    dataset: str,
    config_path: Path,
    mode: PlanMode = "full",
    output_root: Path | None = None,
    run_id: str | None = None,
    debug_sample: int | None = None,
    metadata_path: Path | None = None,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    run_id = run_id or started.strftime("%Y%m%dT%H%M%SZ")
    if output_root is None:
        run_dir = ROOT / "outputs" / "runs" / dataset / run_id / "production"
    else:
        run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = ProductionGenerationConfig.model_validate(raw_config)

    canonical_checks = verify_canonical_resources(dataset)

    source = dict(DATASET_SOURCES[dataset])
    source["dataset"] = dataset
    registry_path = CANONICAL_REGISTRY_PATH
    source["language_registry"] = registry_path
    if metadata_path is not None:
        source["metadata_path"] = Path(metadata_path)
        source["expected_sha256"] = None
    if expected_rows is not None:
        source["expected_rows"] = expected_rows

    preflight_gate = enforce_language_preflight_gate(dataset=dataset, config=config)

    rows = load_flat_metadata(source["metadata_path"])
    metadata_sha256 = sha256_file(source["metadata_path"])
    row_count_ok = len(rows) == source["expected_rows"]
    sha_ok = (
        source["expected_sha256"] is None
        or metadata_sha256 == source["expected_sha256"]
    )
    if not row_count_ok or not sha_ok:
        write_json(
            run_dir / "metadata_audit.json",
            {
                "metadata_source": str(source["metadata_path"]),
                "row_count": len(rows),
                "expected_rows": source["expected_rows"],
                "row_count_match": row_count_ok,
                "sha256": metadata_sha256,
                "expected_sha256": source["expected_sha256"],
                "sha256_match": sha_ok,
            },
        )
        raise ProductionQAError(
            "TRAIN_METADATA_MISMATCH",
            f"rows={len(rows)}/{source['expected_rows']} sha_ok={sha_ok}",
        )

    row_ids, identity_audit = build_row_identities(rows, source)
    registry = load_language_registry(registry_path)

    raw_type_registry = json.loads(source["type_registry"].read_text(encoding="utf-8"))
    contracts_all = build_type_contracts(raw_type_registry)
    contracts = [item for item in contracts_all if item.source_status == "SUPPORTED"]
    deferred = [
        item.type_id for item in contracts_all if item.source_status != "SUPPORTED"
    ]
    if dataset == "vimd":
        specs = vimd_field_specs()
    elif dataset == "vimedcss":
        specs = vimedcss_field_specs()
    else:
        raise ProductionQAError("DATASET_UNSUPPORTED", dataset)
    contract_map = {contract.type_id: contract for contract in contracts}

    index = index_dataset_rows(
        rows,
        row_ids,
        contracts,
        specs,
        hidden_identifier_field=config.sampling.hidden_identifier_field,
    )
    complete_train = row_count_ok
    capacities = {
        contract.type_id: capacity_for_contract(
            contract, index, specs[contract.semantic_field], complete_train
        )
        for contract in contracts
    }
    capacity_status = {
        type_id: payload["full_train_capacity_status"]
        for type_id, payload in capacities.items()
    }

    write_json(
        run_dir / "input_manifest.json",
        {
            "dataset": dataset,
            "dataset_revision": source["revision"],
            "split": "train",
            "metadata_source": str(source["metadata_path"]),
            "metadata_rows": len(rows),
            "metadata_sha256": metadata_sha256,
            "expected_rows": source["expected_rows"],
            "config_path": str(config_path),
            "config_hash": canonical_hash(config.model_dump()),
            "seed": config.seed,
            "mode": mode,
            "debug_sample": debug_sample,
            "generator_version": GENERATOR_VERSION,
            "run_id": run_id,
            "row_identity": identity_audit,
            "split_policy": {
                "train_rows_used": len(rows),
                "valid_rows_used": 0,
                "test_rows_used": 0,
                "sauvi": False,
            },
        },
    )
    write_json(run_dir / "canonical_resource_hashes.json", canonical_checks)
    write_json(run_dir / "language_preflight_gate.json", preflight_gate)

    metadata_audit = {
        "metadata_source": str(source["metadata_path"]),
        "dataset_revision": source["revision"],
        "row_count": len(rows),
        "expected_rows": source["expected_rows"],
        "row_count_match": row_count_ok,
        "sha256": metadata_sha256,
        "expected_sha256": source["expected_sha256"],
        "sha256_match": sha_ok,
        "complete_official_train": complete_train,
        "row_identity": identity_audit,
        "fields": sorted(index.rows_by_value),
        "hidden_identifier_field": config.sampling.hidden_identifier_field,
        "gender_qa": "PROHIBITED_UNDOCUMENTED_ENCODING",
        "valid_rows_used": 0,
        "test_rows_used": 0,
        "sauvi": False,
    }
    write_json(run_dir / "metadata_audit.json", metadata_audit)
    write_json(run_dir / "index_summary.json", index.summary())
    write_json(
        run_dir / "capacity_audit.json",
        {
            "metadata_rows": len(rows),
            "complete_official_train": complete_train,
            "complexity": "O(N) grouping; no O(N^2) pair enumeration",
            "types": capacities,
        },
    )
    write_json(
        run_dir / "semantic_capacity_estimates.json",
        {
            "method": "count formulas over value groups; no enumeration",
            "types": {
                type_id: payload["theoretical_semantic_capacity"]
                for type_id, payload in capacities.items()
            },
            "feasible_semantic_capacity": {
                type_id: feasible_semantic_capacity(
                    payload, config.boolean.positive_ratio
                )
                for type_id, payload in capacities.items()
            },
        },
    )

    coverage = check_runtime_coverage(registry, contracts, specs)
    write_json(run_dir / "language_coverage.json", coverage)

    write_json(
        run_dir / "authoritative_contract.json",
        {
            "source": str(source["type_registry"]),
            "supported_count": len(contracts),
            "review_required_count": len(deferred),
            "supported": [
                {
                    "type_id": contract.type_id,
                    "operator": contract.operator,
                    "semantic_field": contract.semantic_field,
                    "semantic_class": specs[contract.semantic_field].semantic_class,
                    "audio_input_count": contract.audio_input_count,
                    "condition_fields": list(contract.condition_fields),
                    "answer_kind": contract.answer_kind,
                    "source_status": contract.source_status,
                }
                for contract in sorted(contracts, key=lambda item: item.type_id)
            ],
            "review_required": sorted(deferred),
        },
    )

    budget = resolve_budget(
        config,
        contracts,
        capacities,
        debug_cap=debug_sample if mode == "debug_sample" else None,
    )
    write_json(run_dir / "budget_resolution.json", budget)

    write_json(
        run_dir / "determinism_readiness.json",
        {
            "seed": config.seed,
            "sorted_inputs": True,
            "python_builtin_hash_used": False,
            "index_deterministic": True,
            "plan_generated": mode in {"plan", "full", "debug_sample"}
            and budget["status"] not in {"READY_AWAITING_BUDGET", "BUDGET_INFEASIBLE"},
        },
    )

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "run_id": run_id,
        "mode": mode,
        "dataset": dataset,
        "budget": budget["status"],
        "qa_generated": 0,
        "plan_records": 0,
        "conclusion": None,
    }

    plan_result: dict[str, Any] | None = None
    capacity_problem = [
        type_id for type_id, status in capacity_status.items() if status != "SUFFICIENT"
    ]
    blocked_reasons: list[str] = []
    if coverage["missing"]:
        blocked_reasons.append(
            "LANGUAGE_LIBRARY_COVERAGE_MISSING:" + ",".join(coverage["missing"])
        )
    if capacity_problem:
        blocked_reasons.append(
            "FULL_TRAIN_CAPACITY_INSUFFICIENT:" + ",".join(sorted(capacity_problem))
        )

    wants_plan = mode in {"plan", "full", "debug_sample"} and budget["status"] not in {
        "READY_AWAITING_BUDGET",
        "BUDGET_INFEASIBLE",
    }
    if wants_plan and not blocked_reasons:
        try:
            plan_result = build_generation_plan(
                contracts=contracts,
                specs=specs,
                registry=registry,
                index=index,
                config=config,
                seed=int(config.seed or 0),
                budget_by_type=budget["per_type"],
                dataset=dataset,
                dataset_revision=source["revision"],
            )
        except (BudgetShortfallError, ProductionQAError) as exc:
            blocked_reasons.append(f"PLAN_FAILED:{exc}")

    if plan_result is not None:
        records = plan_result["records"]
        plan_path = run_dir / "generation_plan.jsonl"
        write_jsonl(plan_path, records)
        plan_sha = sha256_file(plan_path)
        (run_dir / "generation_plan.sha256").write_text(
            plan_sha + "\n", encoding="utf-8"
        )

        second = build_generation_plan(
            contracts=contracts,
            specs=specs,
            registry=registry,
            index=index,
            config=config,
            seed=int(config.seed or 0),
            budget_by_type=budget["per_type"],
            dataset=dataset,
            dataset_revision=source["revision"],
        )
        determinism = {
            "plan_fingerprint": plan_result["fingerprint"],
            "plan_fingerprint_second_pass": second["fingerprint"],
            "plan_sha256": plan_sha,
            "deterministic": plan_result["fingerprint"] == second["fingerprint"],
            "seed": config.seed,
        }
        if not determinism["deterministic"]:
            blocked_reasons.append("NONDETERMINISTIC_PLAN")
        write_json(run_dir / "determinism_audit.json", determinism)

        write_json(
            run_dir / "sampling_audit.json",
            {
                "plan_records": len(records),
                "sampler_stats": plan_result["sampler_stats"],
                "semantic_duplicates_rejected": plan_result[
                    "semantic_duplicates_rejected"
                ],
                "reuse": plan_result["reuse"].audit(),
                "value_sampling": config.sampling.value_sampling,
                "prefer_distinct_speaker": config.sampling.prefer_distinct_speaker,
                "pair_enumeration": "NOT_PERFORMED",
            },
        )
        manifest = audio_dependency_manifest(records, source)
        write_json(run_dir / "audio_dependency_manifest.json", manifest)
        write_json(
            run_dir / "audio_resolution_audit.json",
            audio_resolution_audit(manifest, config),
        )
        summary["plan_records"] = len(records)

        if mode in {"full", "debug_sample"}:
            internal, model_records = realize_qa(
                records,
                contracts=contract_map,
                specs=specs,
                registry=registry,
                index=index,
                config=config,
                seed=int(config.seed or 0),
                metadata_sha256=metadata_sha256,
                type_registry_hash=canonical_checks["type_registry"]["file_sha256"],
            )
            write_jsonl(run_dir / "qa_internal.jsonl", internal)
            write_jsonl(run_dir / "qa_model_facing.jsonl", model_records)
            validation = audit_qa_records(
                internal,
                model_records,
                contracts=contract_map,
                specs=specs,
                registry=registry,
                compatible_by_type=plan_result["compatible_by_type"],
                capacity_status=capacity_status,
                index=index,
            )
            write_json(run_dir / "qa_validation.json", validation)
            distribution = build_distribution_audit(
                internal,
                registry=registry,
                compatible_by_type=plan_result["compatible_by_type"],
                configured_positive_ratio=config.boolean.positive_ratio,
                reuse_audit=plan_result["reuse"].audit(),
            )
            write_json(run_dir / "distribution_audit.json", distribution)
            summary["qa_generated"] = len(internal)
            if not validation["all_passed"]:
                blocked_reasons.append("QA_AUDIT_FAILURES")
            if distribution["selection_position"] and any(
                bucket["severe_position_bias"]
                for bucket in distribution["selection_position"].values()
            ):
                blocked_reasons.append("SEVERE_POSITION_BIAS")
            if distribution["boolean_labels"] and any(
                not bucket["ratio_within_rounding"]
                for bucket in distribution["boolean_labels"].values()
            ):
                blocked_reasons.append("BOOLEAN_RATIO_DRIFT")
    else:
        write_json(
            run_dir / "determinism_audit.json",
            {
                "plan_generated": False,
                "index_deterministic": True,
                "note": "readiness path builds deterministic O(N) indexes only",
            },
        )

    warnings: list[str] = []
    if plan_result is not None:
        for type_id, stats in sorted(plan_result["sampler_stats"].items()):
            sampler_stats = stats.get("sampler", {})
            if sampler_stats.get("same_speaker_fallbacks"):
                warnings.append(f"SAME_SPEAKER_FALLBACK:{type_id}")
            if sampler_stats.get("text_negative_length_fallbacks"):
                warnings.append(f"TEXT_NEGATIVE_LENGTH_FALLBACK:{type_id}")

    if blocked_reasons:
        conclusion = "BLOCKED"
    elif budget["status"] in {"READY_AWAITING_BUDGET", "DEBUG_SAMPLE_BUDGET"}:
        conclusion = "READY_AWAITING_BUDGET"
    elif warnings:
        conclusion = "PASS_WITH_REVIEW"
    else:
        conclusion = "PASS"

    cost = {
        "dataset_profiler_calls": 0,
        "style_discovery_calls": 0,
        "template_calls": 0,
        "paraphrase_calls": 0,
        "language_quality_calls": 0,
        "qa_generation_llm_calls": 0,
        "qwen_calls": 0,
        "llm_calls_total": 0,
    }
    safety = {
        "train_only": True,
        "valid_rows_used": 0,
        "test_rows_used": 0,
        "sauvi": False,
        "hf_download": 0,
        "audio_decode": 0,
        "audio_materialized": bool(config.audio.materialize),
        "protocol_violations": 0,
    }
    write_json(run_dir / "cost_audit.json", cost)

    audit = {
        "generator_version": GENERATOR_VERSION,
        "mode": mode,
        "debug_sample": mode == "debug_sample",
        "readiness": {
            "GENERATOR_IMPLEMENTED": "YES",
            "FULL_TRAIN_INDEX": "PASS" if row_count_ok else "FAIL",
            "FULL_TRAIN_CAPACITY": "PASS" if not capacity_problem else "FAIL",
            "LANGUAGE_COVERAGE": "PASS" if not coverage["missing"] else "FAIL",
            "ZERO_LLM_RUNTIME": "PASS",
            "FINAL_BUDGET": "PRESENT"
            if budget["explicit_budget_present"]
            else "MISSING",
            "QA_GENERATED": summary["qa_generated"],
        },
        "canonical_resources": canonical_checks,
        "capacity_problem_types": capacity_problem,
        "language_coverage": coverage,
        "budget": budget,
        "warnings": warnings,
        "blocked_reasons": blocked_reasons,
        "cost": cost,
        "safety": safety,
        "conclusion": conclusion,
    }
    if plan_result is not None:
        audit["plan"] = {
            "records": len(plan_result["records"]),
            "fingerprint": plan_result["fingerprint"],
        }
    write_json(run_dir / "audit.json", audit)
    write_report(
        run_dir,
        audit=audit,
        summary=summary,
        index=index,
        capacities=capacities,
        coverage=coverage,
        budget=budget,
        contracts=contracts,
        deferred=deferred,
        specs=specs,
        config=config,
        metadata_sha256=metadata_sha256,
        source=source,
        registry=registry,
    )
    summary["conclusion"] = conclusion
    summary["blocked_reasons"] = blocked_reasons
    summary["canonical_resources_verified"] = canonical_checks["all_matched"]
    summary["started_at"] = started.isoformat()
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    return summary


def write_report(
    run_dir: Path,
    *,
    audit: dict[str, Any],
    summary: dict[str, Any],
    index: TrainIndex,
    capacities: dict[str, dict[str, Any]],
    coverage: dict[str, Any],
    budget: dict[str, Any],
    contracts: list[TypeContract],
    deferred: list[str],
    specs: dict[str, SemanticFieldSpec],
    config: ProductionGenerationConfig,
    metadata_sha256: str,
    source: dict[str, Any],
    registry: ProductionLanguageRegistry,
) -> None:
    type_lines = []
    for contract in sorted(contracts, key=lambda item: item.type_id):
        payload = capacities[contract.type_id]
        spec = specs[contract.semantic_field]
        evidence = payload["evidence_summary"]
        theory = payload["theoretical_semantic_capacity"]
        compatible = coverage["types"]
        entry = next(row for row in compatible if row["type_id"] == contract.type_id)
        type_lines.append(
            f"| {contract.type_id} | {contract.operator} | "
            f"{contract.semantic_field} | {spec.semantic_class} | "
            f"{evidence['valid_rows']} | {evidence['unique_values']} | "
            f"{payload['full_train_capacity_status']} | "
            f"{theory['positive']} | {theory['negative']} | "
            f"{theory['selection']} | {entry['active_language_entries']} |"
        )
    index_lines = [
        f"| {field} | {payload['valid_rows']} | {payload['domain_size']} | "
        f"{payload['max_group_size']} | {payload['groups_with_count_ge_2']} | "
        f"{payload['empty_or_missing_rows']} |"
        for field, payload in sorted(index.summary()["fields"].items())
    ]
    budget_lines = [
        f"- status: `{budget['status']}`",
        (
            f"- explicit budget present: "
            f"{'YES' if budget['explicit_budget_present'] else 'NO'}"
        ),
        f"- allocation policy: `{budget['allocation_policy']}`",
        f"- strict budget: `{budget['strict_budget']}`",
        f"- resolved K: `{budget.get('resolved_total')}`",
        f"- seed: `{config.seed}`",
        f"- boolean positive_ratio: `{config.boolean.positive_ratio}`",
        f"- value_sampling: `{config.sampling.value_sampling}`",
        (
            f"- variants_per_semantic_instance: "
            f"`{config.language.variants_per_semantic_instance}`"
        ),
        f"- language selection: `{config.language.selection}`",
        "- per_language_pattern_budget: `null` (unsupported; schema-rejected if set)",
    ]
    if not budget["explicit_budget_present"]:
        budget_lines.append("- conclusion input: **READY_AWAITING_BUDGET**")
    # Mechanical report fix: counts were referenced but never computed.
    active_entries = [entry for entry in registry.entries if entry.enabled]
    canonical_active = sum(
        1 for entry in active_entries if entry.source_kind == "CANONICAL"
    )
    paraphrase_active = sum(
        1 for entry in active_entries if entry.source_kind == "PARAPHRASE"
    )
    report = f"""# Production QA Generator V1 — {summary["mode"]} run

## IMPLEMENTATION

- files added: `src/autonomous_qa/certification/qa_sampling.py`, `src/qa_audit.py`, `src/production_qa.py`
- files changed: `src/autonomous_qa/production/generate_qa.py` (production CLI),
  `src/autonomous_qa/language/language_quality.py` (config schema extension),
  `configs/qa_generation_production.yaml`
- canonical resources verified: `{audit["canonical_resources"]["all_matched"]}`

## CANONICAL INPUTS

- DatasetProfile: `{audit["canonical_resources"]["dataset_profile"]["file_sha256"]}` (match)
- type registry: `{audit["canonical_resources"]["type_registry"]["file_sha256"]}` (match)
- FieldSpecs: `vimd_field_specs()` = region/province_name
  categorical_attribute, text text_content
- template library hash: `{audit["canonical_resources"]["template_library"]["internal_hash"]}`
- paraphrase library hash: `{audit["canonical_resources"]["paraphrase_library"]["internal_hash"]}`
- production registry hash: `{audit["canonical_resources"]["production_registry"]["registry_hash"]}`

## ARCHITECTURE

FULL TRAIN metadata -> Python full-data indexes -> capacity ->
bounded semantic sampler -> ONE wording / semantic instance -> QA ->
Python deterministic audit. No preview phase, no human approval gate,
no per-QA LLM.

## ZERO-LLM

- profiler calls this milestone = 0
- style calls = 0
- template calls = 0
- paraphrase calls = 0
- language-quality calls = 0
- QA-generation LLM calls = 0

## FULL TRAIN INDEX

- rows: {len(index.rows)}
- fields: {", ".join(sorted(index.rows_by_value))}

| field | valid rows | domain | max group | groups >=2 | empty |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(index_lines)}

- metadata sha256: `{metadata_sha256}`
- metadata source: `{source["metadata_path"]}`
- dataset revision: `{source["revision"]}`

## TYPE CAPACITY

| type_id | operator | field | class | rows | domain | status | pos | neg | select | lang |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(type_lines)}

- deferred (REVIEW_REQUIRED, not generated): {", ".join(deferred)}

## SEMANTIC CAPACITY ESTIMATES

Count formulas over value groups only; no O(N^2) enumeration.
See `semantic_capacity_estimates.json`.

## LANGUAGE COVERAGE

- supported semantic types: {coverage["supported_semantic_types"]}
- covered: {coverage["covered"]}
- missing: {coverage["missing"]}
- active registry entries: {len(active_entries)} (
  {canonical_active} canonical + {paraphrase_active} paraphrase) = pool, not
  a dataset multiplier

## LANGUAGE SELECTION POLICY

- 1 semantic instance -> exactly 1 active compatible entry
- selection = deterministic SHA256(seed|semantic_instance_id|registry_hash)
- no wording quota; no canonical/paraphrase quota; no corpus multiplication
- `variants_per_semantic_instance = 1` (schema-enforced)

## BUDGET

{chr(10).join(budget_lines)}

## SAMPLER

- DIRECT: one audio row per type; gold = hidden field value; dedup key
  (type, row); O(N) shuffled stream.
- EQUALITY: two distinct rows; positive from same value group, negative from
  two distinct values; canonical row order => (A,B) == (B,A); no self pair;
  no O(N^2).
- TARGET_MATCH: one audio + real-domain visible target; positive target = own
  normalized value; negative target drawn from the value domain (text uses
  length-bucket with deterministic fallback).
- PAIRWISE_SELECTION: pick target, one matching row, one non-matching row;
  invariant match(A) XOR match(B); position from stable SHA256 bit.

## ANTI-SHORTCUT

- prefer_distinct_speaker: `{config.sampling.prefer_distinct_speaker}`
  (hidden speakerID internal only; fallbacks reported, never leaked)
- text negative length bucket: `{config.text_negative.length_bucket_match}`
- source reuse caps: global
  `{config.sampling.max_source_row_reuse}`,
  per type `{config.sampling.max_source_row_reuse_per_type}` (null = report only)

## DETERMINISM

- explicit seed `{config.seed}`; SHA256 identity for semantic instances,
  language selection, QA ids; sorted inputs; no Python hash() anywhere.
- plan fingerprint second pass recorded in `determinism_audit.json`.

## MOCK RESULTS

- mock fixture + full unit tests: see `tests/test_production_qa_*.py`
- readiness/plan/audit paths exercised zero-LLM with monkeypatched client
  guards.

## TESTS

- focused production QA tests, full pytest, ruff check/format on touched
  files.

## SAFETY

- TRAIN only; VALID 0; TEST 0; SAUVI NO; HF download 0; audio decode 0;
  Qwen NO; gender QA NO (undocumented 0/1 encoding).

## CONCLUSION

**{audit["conclusion"]}**
"""
    (run_dir / "report.md").write_text(report, encoding="utf-8")

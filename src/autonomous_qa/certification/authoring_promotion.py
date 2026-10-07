"""Framework-generic R&D Authoring -> Deterministic Canonical Promotion Bridge.

Provides two decoupled phases:
1. PREPARE (--prepare): Staged compilation, evidence hashing, candidate validation,
   real staged preflight execution, and PromotionBundle compilation.
   GUARANTEE: 0 canonical resource mutations, 0 real LLM calls.
2. APPLY (--apply): Transaction-safe, hash-verified, rollback-protected promotion
   of a previously prepared PromotionBundle into canonical resources.

No dataset-specific hardcoded task IDs, field heuristics, or bypasses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.autonomous_qa.certification.promotion_gate import (
    PromotionError,
    validate_catalog,
)
from src.autonomous_qa.compiler.canonical_resources import (
    RESOURCE_ROOT,
    DatasetSpec,
    ProductionContract,
    PromotionManifest,
)
from src.autonomous_qa.compiler.semantic_comparators import (
    ComparatorRegistry,
    comparator_set_hash,
    load_comparator_registry,
)
from src.autonomous_qa.compiler.semantic_field_specs import (
    SemanticFieldSpecBundle,
    compile_semantic_field_specs,
    load_migration_sidecar,
)
from src.autonomous_qa.compiler.semantic_task import (
    SemanticCatalog,
    SemanticTaskSpec,
    load_semantic_catalog,
)
from src.autonomous_qa.language.candidate_registry import (
    CANDIDATE_CAPABILITIES_RESOURCE,
    build_candidate_language_registry,
    load_candidate_capability_resource,
    validate_candidate_capability_resource,
    write_registry_deterministically,
)
from src.autonomous_qa.language.language_quality import (
    ProductionLanguageRegistry,
    load_language_registry,
)
from src.autonomous_qa.language.language_preflight import (
    AcceptedLanguageType,
    compute_contract_fingerprint,
    operator_contracts,
    run_preflight,
)
from src.autonomous_qa.language.template_renderer import canonical_hash
from src.common.config import ROOT

LANGUAGE_REGISTRY_PATH = RESOURCE_ROOT / "language" / "production_registry.json"
DEFAULT_SCRATCH_DIR = ROOT / "outputs" / "_scratch" / "promotion"
DEFAULT_LANGUAGE_CAPABILITY_SCRATCH = (
    ROOT / "outputs" / "_scratch" / "language_capability" / "phase4_2"
)

STAGE_PRIMITIVE_DISCOVERY = "primitive_semantic_discovery"
STAGE_COMPOSITE_DISCOVERY = "composite_discovery"
STAGE_LANGUAGE_GENERATION = "language_generation"


def get_parsed_response_path(run_dir: Path, stage: str) -> Path:
    """Authoring stage artifact path. Composite proposals live under
    ``llm/composite_discovery`` (the exact authoring stage name), never under the
    historical ``composite_semantic_discovery`` mismatch."""
    return run_dir / "llm" / stage / "parsed_response.json"


# ---------------------------------------------------------------------------
# Resource-root-scoped canonical loaders.
#
# APPLY must resolve every canonical resource through the supplied
# ``resource_root`` (default `resources/`). These helpers keep reads/writes
# below the given root so isolated tests can operate on a synthetic tree.
# ---------------------------------------------------------------------------


def load_dataset_registry(resource_root: Path) -> dict[str, dict[str, str]]:
    path = resource_root / "registry" / "datasets.json"
    if not path.exists():
        raise PromotionError("CANONICAL_DATASET_REGISTRY_MISSING", str(path))
    return _read_json(path).get("datasets", {})


def dataset_registry_entry(resource_root: Path, dataset_id: str) -> dict[str, str]:
    registry = load_dataset_registry(resource_root)
    if dataset_id not in registry:
        raise PromotionError("UNKNOWN_DATASET", dataset_id)
    return registry[dataset_id]


def _resolve_entry_path(resource_root: Path, rel: str) -> Path:
    # Registry paths are expressed relative to the project root (resource_root.parent).
    return resource_root.parent / rel


def load_dataset_spec_from_root(resource_root: Path, dataset_id: str) -> DatasetSpec:
    entry = dataset_registry_entry(resource_root, dataset_id)
    rel = entry.get("dataset_spec")
    if not rel:
        raise PromotionError("CANONICAL_DATASET_SPEC_MISSING", dataset_id)
    return DatasetSpec.model_validate(_read_json(_resolve_entry_path(resource_root, rel)))


def load_semantic_catalog_from_root(
    resource_root: Path, dataset_id: str
) -> SemanticCatalog:
    entry = dataset_registry_entry(resource_root, dataset_id)
    rel = entry.get("semantic_catalog")
    if not rel:
        raise PromotionError("CANONICAL_SEMANTIC_CATALOG_MISSING", dataset_id)
    path = _resolve_entry_path(resource_root, rel)
    if not path.exists():
        raise PromotionError("CANONICAL_SEMANTIC_CATALOG_MISSING", str(path))
    return load_semantic_catalog(path)


def load_production_contract_from_root(
    resource_root: Path, dataset_id: str
) -> ProductionContract | None:
    entry = dataset_registry_entry(resource_root, dataset_id)
    rel = entry.get("production_contract")
    if not rel:
        return None
    path = _resolve_entry_path(resource_root, rel)
    if not path.exists():
        return None
    return ProductionContract.model_validate(_read_json(path))


def load_promotion_manifest_from_root(
    resource_root: Path, dataset_id: str
) -> PromotionManifest | None:
    entry = dataset_registry_entry(resource_root, dataset_id)
    rel = entry.get("promotion_manifest")
    if not rel:
        return None
    path = _resolve_entry_path(resource_root, rel)
    if not path.exists():
        return None
    return PromotionManifest.model_validate(_read_json(path))


def load_language_registry_from_root(resource_root: Path) -> ProductionLanguageRegistry:
    return load_language_registry(resource_root / "language" / "production_registry.json")


def load_comparator_registry_from_root(resource_root: Path) -> ComparatorRegistry:
    path = resource_root / "semantics" / "comparators.json"
    return load_comparator_registry(path)


class PromotionBundle(BaseModel):
    """Machine-readable, deterministic promotion preparation bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    dataset_id: str
    source_run_id: str
    source_run_path: str
    promotion_mode: Literal["replace", "merge"]

    authoring_run_hash: str
    source_documentation_hash: str
    dataset_profile_hash: str

    primitive_candidates: int
    primitive_semantic_accepted: int = 0
    primitive_semantic_pass_count: int = 0
    composite_candidates: int
    composite_semantic_accepted: int = 0
    composite_semantic_pass_count: int = 0
    primitive_language_ready: int = 0
    primitive_authoring_language_gate_pass_count: int = 0
    composite_language_ready: int = 0
    composite_authoring_language_gate_pass_count: int = 0
    authoring_language_gate_pass_count: int = 0
    primitive_executable_selected_count: int = 0
    staged_canonical_language_preflight_pass_count: int = 0
    staged_preflight_type_count: int = 0
    staged_preflight_pass_type_count: int = 0
    promotable_primitive_count: int = 0
    promotable_composite_count: int = 0
    promotable_total: int = 0

    promotable_types: tuple[str, ...] = ()
    non_promotable_types: dict[str, str] = Field(default_factory=dict)
    discovered_semantic_pass_types: tuple[str, ...] = ()
    non_selected_types: dict[str, str] = Field(default_factory=dict)
    selected_promotion_types: tuple[str, ...] = ()
    selected_type_blockers: tuple[str, ...] = ()

    canonical_id_mapping: dict[str, str]

    old_active_types: tuple[str, ...]
    proposed_active_types: tuple[str, ...]

    semantic_diff: dict[str, list[str]]

    candidate_semantic_catalog: dict[str, Any]
    candidate_language_resource: dict[str, Any]
    candidate_dataset_spec: dict[str, Any] | None = None
    candidate_production_contract: dict[str, Any] | None = None
    candidate_promotion_manifest: dict[str, Any] | None = None

    expected_current_hashes: dict[str, str] = Field(default_factory=dict)
    staged_language_preflight: dict[str, Any] = Field(default_factory=dict)

    # Phase 4.2 language-infrastructure evidence. Candidate language preflight
    # is computed against a SCRATCH candidate registry; it is deliberately
    # distinct from the canonical staged preflight that gates apply_ready.
    semantic_field_specs_hash: str = ""
    semantic_field_specs_source: str = ""
    canonical_language_registry_hash: str = ""
    candidate_language_registry_hash: str = ""
    candidate_language_registry: dict[str, Any] | None = None
    candidate_language_preflight: dict[str, Any] = Field(default_factory=dict)
    language_infra_ready: bool = False
    language_registry_change_required: bool = False

    fresh_contract_fingerprints: dict[str, str] = Field(default_factory=dict)
    stale_current_artifacts: tuple[str, ...] = ()

    evidence_summary: dict[str, Any] = Field(default_factory=dict)
    zero_llm_evidence: dict[str, Any] = Field(default_factory=dict)

    authoring_semantic_pass_types: tuple[str, ...] = ()
    executable_promotion_candidates: tuple[str, ...] = ()
    staged_preflight_types: tuple[str, ...] = ()
    staged_catalog_type_ids: tuple[str, ...] = ()
    staged_accepted_type_ids: tuple[str, ...] = ()
    staged_language_candidate_type_ids: tuple[str, ...] = ()

    prepare_status: str = "PREPARED"
    apply_ready: bool = False

    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        return canonical_hash(payload)


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise PromotionError("MISSING_FILE", str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_promotion_readiness(run_dir: Path) -> dict[str, Any]:
    """Evaluates explicit gate and language readiness from run artifacts without semantic capability inference."""
    gates_dir = run_dir / "gates"
    lang_dir = run_dir / "language"
    reconciliation_path = run_dir / "reconciliation_canonical.json"

    primitive_gates_path = gates_dir / "primitive_gates.json"
    composite_gates_path = gates_dir / "composite_gates.json"
    preflight_path = lang_dir / "preflight.json"

    if not primitive_gates_path.exists():
        raise PromotionError("MISSING_AUTHORING_EVIDENCE", "gates/primitive_gates.json missing")

    primitive_gates = _read_json(primitive_gates_path)
    composite_gates = _read_json(composite_gates_path) if composite_gates_path.exists() else {"composites": []}
    preflight_data = _read_json(preflight_path) if preflight_path.exists() else {"entries": []}
    reconciliation_data = _read_json(reconciliation_path) if reconciliation_path.exists() else {"matches": []}

    authoring_preflight_pass_types = {
        e["type_id"] for e in preflight_data.get("entries", []) if e.get("verdict") == "PASS"
    }

    primitive_candidates = len(primitive_gates.get("candidates", []))
    primitive_semantic_pass_count = 0
    primitive_authoring_language_gate_pass_count = 0
    authoring_gate_pass_primitives: list[str] = []
    non_gate_pass_types: dict[str, str] = {}

    for cand in primitive_gates.get("candidates", []):
        cid = cand["candidate_id"]
        sem_pass = cand.get("verdict") == "PASS"
        if sem_pass:
            primitive_semantic_pass_count += 1

        lang_ready = cid in authoring_preflight_pass_types
        if sem_pass and lang_ready:
            primitive_authoring_language_gate_pass_count += 1
            authoring_gate_pass_primitives.append(cid)
        elif sem_pass and not lang_ready:
            non_gate_pass_types[cid] = "SEMANTIC_GATE_PASS_BUT_LANGUAGE_NOT_READY"
        else:
            non_gate_pass_types[cid] = "SEMANTIC_GATE_FAILED"

    composite_candidates = len(composite_gates.get("composites", []))
    composite_semantic_pass_count = 0
    composite_authoring_language_gate_pass_count = 0
    authoring_gate_pass_composites: list[str] = []

    for comp in composite_gates.get("composites", []):
        cid = comp.get("composite_id", comp.get("candidate_id"))
        sem_pass = comp.get("verdict") == "PASS"
        if sem_pass:
            composite_semantic_pass_count += 1

        lang_ready = cid in authoring_preflight_pass_types
        if sem_pass and lang_ready:
            composite_authoring_language_gate_pass_count += 1
            authoring_gate_pass_composites.append(cid)
        elif sem_pass and not lang_ready:
            non_gate_pass_types[cid] = "SEMANTIC_GATE_PASS_BUT_LANGUAGE_NOT_READY"
        else:
            non_gate_pass_types[cid] = "SEMANTIC_GATE_FAILED"

    authoring_semantic_pass_types = tuple(sorted(
        [c["candidate_id"] for c in primitive_gates.get("candidates", []) if c.get("verdict") == "PASS"]
        + [comp.get("composite_id", comp.get("candidate_id")) for comp in composite_gates.get("composites", []) if comp.get("verdict") == "PASS"]
    ))
    authoring_gate_pass_candidates = authoring_gate_pass_primitives + authoring_gate_pass_composites

    return {
        "primitive_candidates": primitive_candidates,
        "primitive_semantic_pass_count": primitive_semantic_pass_count,
        "primitive_authoring_language_gate_pass_count": primitive_authoring_language_gate_pass_count,
        "composite_candidates": composite_candidates,
        "composite_semantic_pass_count": composite_semantic_pass_count,
        "composite_authoring_language_gate_pass_count": composite_authoring_language_gate_pass_count,
        "authoring_language_gate_pass_count": primitive_authoring_language_gate_pass_count + composite_authoring_language_gate_pass_count,
        "authoring_semantic_pass_types": authoring_semantic_pass_types,
        "authoring_gate_pass_candidates": authoring_gate_pass_candidates,
        "authoring_gate_pass_primitives": authoring_gate_pass_primitives,
        "authoring_gate_pass_composites": authoring_gate_pass_composites,
        "non_gate_pass_types": non_gate_pass_types,
        "reconciliation": reconciliation_data,
    }


def filter_executable_candidates(
    candidates: list[str],
    run_dir: Path,
    capability_exclusions: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Evaluates executable contract capability from structured candidate proposal evidence and run-local overrides."""
    run_exclusions: dict[str, str] = {}
    override_file = run_dir / "promotion_overrides.json"
    if override_file.exists():
        data = _read_json(override_file)
        run_exclusions = data.get("capability_exclusions", data.get("exclusions", {}))

    all_exclusions = {**run_exclusions, **(capability_exclusions or {})}

    raw_candidates: dict[str, dict[str, Any]] = {}
    prim_file = get_parsed_response_path(run_dir, STAGE_PRIMITIVE_DISCOVERY)
    if prim_file.exists():
        for c in _read_json(prim_file).get("candidates", []):
            if "candidate_id" in c:
                raw_candidates[c["candidate_id"]] = c

    comp_file = get_parsed_response_path(run_dir, STAGE_COMPOSITE_DISCOVERY)
    if comp_file.exists():
        comp_data = _read_json(comp_file)
        for c in comp_data.get("composites", comp_data.get("candidates", [])):
            cid = c.get("composite_id", c.get("candidate_id"))
            if cid:
                raw_candidates[cid] = c

    executable: list[str] = []
    excluded: dict[str, str] = {}

    for cid in candidates:
        if cid in all_exclusions:
            excluded[cid] = all_exclusions[cid]
            continue

        raw = raw_candidates.get(cid)
        if not raw:
            excluded[cid] = "MISSING_STRUCTURED_PROPOSAL"
            continue

        op = raw.get("operator")
        if op == "TARGET_MATCH":
            target_rel = raw.get("target_relation") or raw.get("relation")
            if target_rel == "membership":
                excluded[cid] = "OPERATOR_CAPABILITY_GAP:MEMBERSHIP"
                continue
            elif target_rel is None and not raw.get("scalar_equality"):
                # Without structured specification proving scalar equality or IR membership support:
                excluded[cid] = "EXECUTABLE_RELATION_UNDER_SPECIFIED"
                continue

        executable.append(cid)

    return executable, excluded


def _check_reconciliation_compatibility(
    raw_cand: dict[str, Any],
    canon_task: SemanticTaskSpec,
) -> bool:
    """Fail-closed executable compatibility proof before reusing a canonical
    type ID or comparator.

    Reuse is only permitted when the candidate carries EXPLICIT evidence for the
    full executable signature (operator, audio arity, answer kind, source field,
    visible context roles) that matches the canonical task. Missing evidence is
    never silently defaulted; an unproven equivalence must not reuse a canonical
    executable identity.
    """
    if "operator" not in raw_cand or raw_cand["operator"] != canon_task.operator:
        return False

    if "audio_arity" not in raw_cand or raw_cand["audio_arity"] != canon_task.audio_arity:
        return False

    answer = raw_cand.get("answer_schema_proposal")
    if not isinstance(answer, dict) or not answer.get("kind"):
        return False
    canon_kind = canon_task.outputs[0].kind if canon_task.outputs else None
    if not canon_kind or answer["kind"] != canon_kind:
        return False

    if "hidden_source_annotations" not in raw_cand:
        return False
    hidden_src = raw_cand["hidden_source_annotations"]
    visible_inputs = raw_cand.get("visible_context_roles", raw_cand.get("visible_inputs"))
    if visible_inputs is None:
        return False
    if hidden_src:
        cand_src = hidden_src[0]
    elif visible_inputs:
        cand_src = visible_inputs[0]
    else:
        cand_src = raw_cand.get("source_role_mapping", {}).get("source_field")
    canon_src = canon_task.source_role_mapping.get("source_field", "")
    if not cand_src or cand_src != canon_src:
        return False

    cand_visible = tuple(sorted(visible_inputs))
    canon_visible = tuple(sorted(canon_task.visible_context_roles))
    if cand_visible != canon_visible:
        return False

    # Explicit comparator reuse must be consistent with the canonical task.
    cand_comparator = raw_cand.get("comparator_id")
    if cand_comparator and cand_comparator != canon_task.comparator_id:
        return False

    # Structured relation semantics must be compatible whenever declared.
    if "target_relation" in raw_cand or "relation" in raw_cand:
        target_rel = raw_cand.get("target_relation") or raw_cand.get("relation")
        # The canonical task model cannot yet strongly represent a membership
        # relation, so reuse is refused rather than inferred from the name.
        if target_rel == "membership":
            return False
        if target_rel is None and not raw_cand.get("scalar_equality"):
            return False
    if "scalar_equality" in raw_cand:
        if not raw_cand.get("scalar_equality"):
            return False

    return True


def resolve_canonical_ids(
    dataset_id: str,
    candidate_ids: list[str],
    reconciliation_data: dict[str, Any],
    raw_candidates_map: dict[str, dict[str, Any]] | None = None,
    existing_catalog_tasks: list[SemanticTaskSpec] | None = None,
    canonical_id_overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Resolves blind discovery candidate IDs to stable canonical IDs deterministically."""
    canonical_id_overrides = canonical_id_overrides or {}
    mapping: dict[str, str] = {}
    reverse_map: dict[str, str] = {}
    raw_candidates_map = raw_candidates_map or {}
    canon_map = {t.type_id: t for t in (existing_catalog_tasks or [])}

    matches = {m["blind_candidate"]: m for m in reconciliation_data.get("matches", []) if m.get("blind_candidate")}

    for cand_id in candidate_ids:
        if cand_id in canonical_id_overrides:
            target_id = canonical_id_overrides[cand_id]
        elif cand_id in matches and matches[cand_id].get("classification") == "SAME_PROPOSITION_DIFFERENT_NAME" and matches[cand_id].get("canonical_type"):
            canon_type = matches[cand_id]["canonical_type"]
            raw_cand = raw_candidates_map.get(cand_id, {})
            canon_task = canon_map.get(canon_type)
            if not canon_task:
                raise PromotionError(
                    "RECONCILIATION_EXECUTABLE_COMPATIBILITY_UNPROVEN",
                    f"Canonical task '{canon_type}' not found in existing catalog for '{cand_id}'",
                )
            if not _check_reconciliation_compatibility(raw_cand, canon_task):
                raise PromotionError("RECONCILIATION_EXECUTABLE_COMPATIBILITY_UNPROVEN", f"{cand_id} -> {canon_type}")
            target_id = canon_type
        else:
            match = re.match(r"^([a-zA-Z0-9_]+?)_\d+_(.+)$", cand_id)
            if match:
                prefix = match.group(1)
                slug = match.group(2)
                target_id = f"{prefix}_{slug}"
            elif cand_id.startswith(f"{dataset_id}_"):
                target_id = cand_id
            else:
                raise PromotionError("CANONICAL_ID_UNRESOLVED", cand_id)

        if target_id in reverse_map and reverse_map[target_id] != cand_id:
            raise PromotionError(
                "CANONICAL_ID_COLLISION",
                f"Candidates '{cand_id}' and '{reverse_map[target_id]}' map to same canonical ID '{target_id}'",
            )

        mapping[cand_id] = target_id
        reverse_map[target_id] = cand_id

    return mapping


def resolve_candidate_comparator(
    blind_id: str,
    raw_cand: dict[str, Any],
    reconciliation_data: dict[str, Any],
    existing_catalog_tasks: list[SemanticTaskSpec] | None = None,
    comparator_registry: ComparatorRegistry | None = None,
) -> str:
    """Capability-based comparator resolution order with structural compatibility verification."""
    matches = {m["blind_candidate"]: m for m in reconciliation_data.get("matches", []) if m.get("blind_candidate")}
    if blind_id in matches and matches[blind_id].get("classification") == "SAME_PROPOSITION_DIFFERENT_NAME":
        canon_type = matches[blind_id].get("canonical_type")
        if canon_type:
            canon_map = {t.type_id: t for t in (existing_catalog_tasks or [])}
            canon_task = canon_map.get(canon_type)
            if not canon_task:
                raise PromotionError(
                    "RECONCILIATION_EXECUTABLE_COMPATIBILITY_UNPROVEN",
                    f"Canonical task '{canon_type}' not found for candidate '{blind_id}'",
                )
            if not _check_reconciliation_compatibility(raw_cand, canon_task):
                raise PromotionError(
                    "RECONCILIATION_EXECUTABLE_COMPATIBILITY_UNPROVEN",
                    f"Candidate {blind_id} incompatible with canonical {canon_type}",
                )
            if canon_task.comparator_id:
                return canon_task.comparator_id

    reg = comparator_registry or load_comparator_registry()

    if raw_cand.get("comparator_id"):
        cid = raw_cand["comparator_id"]
        try:
            reg.by_id(cid)
            return cid
        except KeyError:
            pass

    operator = raw_cand.get("operator", "DIRECT")
    answer_schema = raw_cand.get("answer_schema_proposal", {})
    kind = answer_schema.get("kind")
    val_type = answer_schema.get("type")

    comp_ids = {c.comparator_id for c in reg.comparators}

    if operator == "DIRECT":
        if val_type == "integer" or kind == "integer":
            if "integer_exact" in comp_ids:
                return "integer_exact"
        if "categorical_label_exact" in comp_ids:
            return "categorical_label_exact"
    elif operator == "EQUALITY":
        if "categorical_label_exact" in comp_ids:
            return "categorical_label_exact"
    elif operator == "PAIRWISE_SELECTION":
        if "audio_choice_exact" in comp_ids:
            return "audio_choice_exact"
        if "categorical_label_exact" in comp_ids:
            return "categorical_label_exact"
    elif operator == "TARGET_MATCH":
        if "boolean_exact" in comp_ids:
            return "boolean_exact"
        if "categorical_label_exact" in comp_ids:
            return "categorical_label_exact"

    raise PromotionError("UNSUPPORTED_COMPARATOR_CAPABILITY", f"{blind_id}:{operator}")


def compile_candidate_semantic_catalog(
    dataset_id: str,
    promotable_candidates: list[str],
    id_mapping: dict[str, str],
    run_dir: Path,
    reconciliation_data: dict[str, Any],
    existing_catalog_tasks: list[SemanticTaskSpec] | None = None,
    promotion_mode: str = "replace",
    comparator_registry: ComparatorRegistry | None = None,
) -> dict[str, Any]:
    """Compiles candidate SemanticCatalog from structured proposals with zero silent defaults."""
    parsed_primitives_path = get_parsed_response_path(run_dir, STAGE_PRIMITIVE_DISCOVERY)
    parsed_composites_path = get_parsed_response_path(run_dir, STAGE_COMPOSITE_DISCOVERY)

    if not parsed_primitives_path.exists():
        raise PromotionError("MISSING_AUTHORING_EVIDENCE", "primitive_semantic_discovery parsed_response.json missing")

    raw_primitives = _read_json(parsed_primitives_path).get("candidates", [])
    raw_composites = []
    if parsed_composites_path.exists():
        comp_data = _read_json(parsed_composites_path)
        raw_composites = comp_data.get("composites", comp_data.get("candidates", []))

    cand_map: dict[str, dict[str, Any]] = {c["candidate_id"]: c for c in raw_primitives if "candidate_id" in c}
    for comp in raw_composites:
        cid = comp.get("composite_id", comp.get("candidate_id"))
        if cid:
            cand_map[cid] = comp

    tasks: list[dict[str, Any]] = []

    retained_tasks_map: dict[str, SemanticTaskSpec] = {}
    if promotion_mode == "merge" and existing_catalog_tasks:
        for ex in existing_catalog_tasks:
            retained_tasks_map[ex.type_id] = ex

    for blind_id in promotable_candidates:
        if blind_id not in cand_map:
            raise PromotionError("MISSING_PROPOSAL_EVIDENCE", f"Missing proposal evidence for {blind_id}")

        raw = cand_map[blind_id]
        canonical_id = id_mapping[blind_id]

        if "operator" not in raw:
            raise PromotionError("MISSING_PROPOSAL_EVIDENCE", f"Candidate {blind_id} lacks operator")
        operator = raw["operator"]

        if operator not in {"DIRECT", "TARGET_MATCH", "EQUALITY", "PAIRWISE_SELECTION", "COMPOSITE"}:
            raise PromotionError("UNSUPPORTED_EXECUTABLE_SEMANTIC", f"{blind_id}:{operator}")

        if "proposition" not in raw:
            raise PromotionError("MISSING_PROPOSAL_EVIDENCE", f"Candidate {blind_id} lacks proposition")
        prop_description = raw["proposition"]

        if "audio_arity" not in raw:
            raise PromotionError("MISSING_PROPOSAL_EVIDENCE", f"Candidate {blind_id} lacks audio_arity")
        audio_arity = raw["audio_arity"]

        visible_inputs = tuple(raw.get("visible_inputs", raw.get("visible_context_roles", [])))
        hidden_annotations = raw.get("hidden_source_annotations", [])

        if not hidden_annotations and not visible_inputs and operator != "COMPOSITE":
            raise PromotionError("MISSING_PROPOSAL_EVIDENCE", f"Candidate {blind_id} lacks source field annotations")

        source_field = hidden_annotations[0] if hidden_annotations else (visible_inputs[0] if visible_inputs else "")
        if not source_field and operator != "COMPOSITE":
            raise PromotionError("MISSING_PROPOSAL_EVIDENCE", f"Candidate {blind_id} source_field cannot be determined")

        if "answer_schema_proposal" not in raw:
            raise PromotionError("MISSING_PROPOSAL_EVIDENCE", f"Candidate {blind_id} lacks answer_schema_proposal")
        answer_schema = raw["answer_schema_proposal"]

        comparator_id = resolve_candidate_comparator(
            blind_id,
            raw,
            reconciliation_data,
            existing_catalog_tasks,
            comparator_registry=comparator_registry,
        )

        outputs = [
            {
                "role": "answer" if operator != "DIRECT" else (source_field if source_field else "value"),
                "kind": answer_schema.get("kind", "field_value"),
                "dependencies": [],
            }
        ]

        is_composite = (operator == "COMPOSITE")
        classification = "CHAIN_DERIVED" if is_composite else "PRIMITIVE_RELATION"
        kind = "STRUCTURED" if is_composite else "ATOMIC"

        invariances = raw.get("expected_invariances")
        if invariances is None:
            raise PromotionError("MISSING_PROPOSAL_EVIDENCE", f"Candidate {blind_id} lacks expected_invariances")

        non_invariances = raw.get("expected_non_invariances")
        if non_invariances is None:
            raise PromotionError("MISSING_PROPOSAL_EVIDENCE", f"Candidate {blind_id} lacks expected_non_invariances")

        task_spec = {
            "type_id": canonical_id,
            "proposition_id": canonical_id,
            "proposition_description": prop_description,
            "operator": operator,
            "classification": classification,
            "kind": kind,
            "audio_arity": audio_arity,
            "visible_context_roles": visible_inputs,
            "outputs": outputs,
            "dependency_graph": [],
            "comparator_id": comparator_id,
            "invariances": invariances,
            "non_invariances": non_invariances,
            "base_type_id": None,
            "source_role_mapping": {"source_field": source_field} if source_field else {},
            "evaluation_value": "",
            "tier": "T1_PERCEPTION" if operator == "DIRECT" else ("T2_DERIVED" if operator == "TARGET_MATCH" else "T3_RELATIONAL"),
            "gold_origin": "SOURCE" if operator == "DIRECT" else "DERIVED_SOURCE",
            "guardrail_status": None,
            "closure_hash": None,
        }
        tasks.append(task_spec)

        if canonical_id in retained_tasks_map:
            del retained_tasks_map[canonical_id]

    if promotion_mode == "merge":
        for ret_task in retained_tasks_map.values():
            tasks.append(ret_task.model_dump(mode="json"))

    catalog_dict = {
        "dataset": dataset_id,
        "catalog_version": f"{dataset_id}_semantic_catalog_candidate",
        "comparators_resource": "comparators.json",
        "tasks": tasks,
    }

    catalog_obj = SemanticCatalog.model_validate(catalog_dict)
    known_comparator_ids = (
        {c.comparator_id for c in comparator_registry.comparators}
        if comparator_registry is not None
        else None
    )
    issues = validate_catalog(catalog_obj, comparator_ids=known_comparator_ids)
    if issues:
        raise PromotionError("CANONICAL_RESOURCE_VALIDATION_FAILED", ",".join(issues))

    return catalog_dict


def run_staged_preflight_for_promotion(
    dataset_id: str,
    catalog_dict: dict[str, Any],
    field_specs: dict[str, Any] | None = None,
    run_dir: Path | None = None,
    *,
    registry: ProductionLanguageRegistry | None = None,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Runs real staged language preflight validation against compiled candidate catalog.

    ``field_specs`` must be supplied by the declarative compiler in the normal
    path. The legacy dataset-ID fallback remains ONLY for historical callers and
    is never the normal onboarding mechanism.
    """
    from src.autonomous_qa.language.language_preflight import accepted_types_from_semantic_catalog

    if field_specs is None:
        from src.autonomous_qa.language.template_engine import resolve_legacy_field_specs

        # LEGACY / MIGRATION ONLY — not the normal staged path.
        field_specs = resolve_legacy_field_specs(dataset_id)

    if field_specs is None:
        raise PromotionError("FIELD_SPECS_REQUIRED", f"Explicit field_specs required for {dataset_id}")

    accepted_types = accepted_types_from_semantic_catalog(catalog_dict, field_specs=field_specs)
    res = run_preflight(
        mode="dataset",
        accepted_types=accepted_types,
        dataset=dataset_id,
        write_outputs=False,
        registry=registry,
        registry_path=registry_path,
    )
    audit = res["audit"]
    return {
        "status": audit["result"],
        "contract_fingerprint": audit["contract_fingerprint"],
        "blocking_issues": audit["blocking_issue_count"],
        "review_issues": audit["review_issue_count"],
        "accepted_type_count": audit["accepted_type_count"],
        "accepted_types": accepted_types,
        "issues": res.get("issues", []),
        "coverage": res.get("coverage", []),
        "render_matrix": res.get("render_matrix", []),
        "registry_hash": audit.get("language_registry_hash"),
    }


def _resolve_declarative_field_specs(
    dataset_id: str,
    profile: dict[str, Any],
    required_fields: set[str],
    resource_root: Path,
) -> tuple[dict[str, Any], str, str]:
    """Compile field specs from the DatasetProfile + optional migration sidecar.

    Returns (field_specs, source, logical_hash). No dataset-ID Python branch:
    the sidecar is located by a generic path convention.
    """
    sidecar_path = resource_root / "field_specs" / f"{dataset_id}.json"
    migration_specs = (
        load_migration_sidecar(sidecar_path, expected_dataset_id=dataset_id)
        if sidecar_path.exists()
        else None
    )
    bundle = compile_semantic_field_specs(
        profile,
        required_fields=required_fields,
        migration_specs=migration_specs,
        dataset_id=dataset_id,
    )
    return bundle.field_specs, bundle.source, bundle.logical_hash()


def _required_source_fields(catalog_dict: dict[str, Any]) -> set[str]:
    fields: set[str] = set()
    for task in catalog_dict.get("tasks", []):
        field = task.get("source_role_mapping", {}).get("source_field")
        if field:
            fields.add(field)
        for output in task.get("outputs", []):
            for dep in output.get("dependencies", []):
                fields.add(dep)
    return fields


def _registry_entry_dump(registry: ProductionLanguageRegistry) -> dict[str, dict[str, Any]]:
    return {
        entry.language_entry_id: entry.model_dump(mode="json")
        for entry in registry.entries
    }


def _language_registry_impact(
    resource_root: Path,
    canonical_registry: ProductionLanguageRegistry,
    candidate_registry: ProductionLanguageRegistry,
    *,
    cross_dataset_preflight_delta: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Report the impact of replacing the canonical registry with the candidate.

    Read-only: no canonical resource is mutated. Identifies entries added,
    modified or removed, and every canonical contract/manifest whose pinned
    ``language_registry_hash`` would become stale.
    """
    candidate_hash = candidate_registry.registry_hash
    affected_contracts: list[str] = []
    affected_manifests: list[str] = []
    production_dir = resource_root / "production"
    if production_dir.exists():
        for path in sorted(production_dir.glob("*.json")):
            data = _read_json(path)
            if not isinstance(data, dict):
                continue
            if "language_registry_hash" not in data:
                continue
            if data.get("language_registry_hash") != candidate_hash:
                if path.name.endswith(".promotion.json"):
                    affected_manifests.append(path.name)
                else:
                    affected_contracts.append(path.name)

    base = _registry_entry_dump(canonical_registry)
    cand = _registry_entry_dump(candidate_registry)
    added = sorted(set(cand) - set(base))
    removed = sorted(set(base) - set(cand))
    modified = sorted(
        entry_id for entry_id in (set(base) & set(cand)) if base[entry_id] != cand[entry_id]
    )
    return {
        "current_registry_hash": canonical_registry.registry_hash,
        "candidate_registry_hash": candidate_hash,
        "language_registry_change_required": candidate_hash != canonical_registry.registry_hash,
        "added_entry_ids": added,
        "modified_entry_ids": modified,
        "removed_entry_ids": removed,
        "affected_production_contracts": affected_contracts,
        "affected_promotion_manifests": affected_manifests,
        "cross_dataset_preflight_delta": cross_dataset_preflight_delta or [],
    }


def _cross_dataset_preflight_delta(
    canonical_registry: ProductionLanguageRegistry,
    candidate_registry: ProductionLanguageRegistry,
    candidate_registry_path: Path,
    datasets: tuple[str, ...] = ("vimd", "vietmdd", "vimedcss"),
) -> list[dict[str, Any]]:
    from src.autonomous_qa.language.language_preflight import get_dataset_accepted_types

    delta: list[dict[str, Any]] = []
    for dataset in datasets:
        try:
            accepted = get_dataset_accepted_types(dataset)
        except Exception:  # pragma: no cover - dataset resolver unavailable
            continue
        base = run_preflight(
            mode="dataset", accepted_types=accepted, dataset=dataset, write_outputs=False
        )
        cand = run_preflight(
            mode="dataset",
            accepted_types=accepted,
            dataset=dataset,
            write_outputs=False,
            registry=candidate_registry,
            registry_path=candidate_registry_path,
        )
        delta.append(
            {
                "dataset": dataset,
                "baseline_status": base["audit"]["result"],
                "baseline_blocking": base["audit"]["blocking_issue_count"],
                "candidate_status": cand["audit"]["result"],
                "candidate_blocking": cand["audit"]["blocking_issue_count"],
                "new_blocking": max(
                    0,
                    cand["audit"]["blocking_issue_count"]
                    - base["audit"]["blocking_issue_count"],
                ),
            }
        )
    return delta


def compile_candidate_language_resource(
    dataset_id: str,
    promotable_candidates: list[str],
    id_mapping: dict[str, str],
    run_dir: Path,
) -> dict[str, Any]:
    """Compiles candidate language entries for proposal evidence."""
    parsed_lang_path = get_parsed_response_path(run_dir, STAGE_LANGUAGE_GENERATION)
    if not parsed_lang_path.exists():
        return {"entries": []}

    raw_entries = _read_json(parsed_lang_path).get("entries", [])
    compiled_entries: list[dict[str, Any]] = []

    for raw in raw_entries:
        blind_id = raw.get("type_id")
        if blind_id in promotable_candidates:
            canonical_id = id_mapping[blind_id]
            entry_copy = dict(raw)
            entry_copy["type_id"] = canonical_id
            compiled_entries.append(entry_copy)

    return {"dataset_id": dataset_id, "entries": compiled_entries}


def prepare_promotion(
    dataset_id: str,
    run_dir: Path,
    *,
    promotion_mode: Literal["replace", "merge"],
    output_root: Path | None = None,
    canonical_id_overrides: dict[str, str] | None = None,
    capability_exclusions: dict[str, str] | None = None,
    field_specs: dict[str, Any] | None = None,
    resource_root: Path = RESOURCE_ROOT,
    language_capability_root: Path | None = None,
    candidate_capabilities_path: Path = CANDIDATE_CAPABILITIES_RESOURCE,
) -> PromotionBundle:
    """Performs the STAGED PREPARE (dry-run) promotion transaction.

    ``canonical_id_overrides`` maps candidate ID -> canonical ID ONLY.
    ``capability_exclusions`` maps candidate ID -> capability exclusion reason.
    These two override domains are deliberately distinct.
    """
    if promotion_mode not in ("replace", "merge"):
        raise PromotionError("INVALID_PROMOTION_MODE", str(promotion_mode))

    if not run_dir.exists():
        raise PromotionError("RUN_DIR_NOT_FOUND", str(run_dir))

    run_manifest_path = run_dir / "run_manifest.json"
    authoring_hash = _sha256_file(run_manifest_path)

    doc_dir = run_dir / "source_documentation"
    doc_hash = canonical_hash([_sha256_file(p) for p in sorted(doc_dir.glob("*"))]) if doc_dir.exists() else ""

    profile_path = run_dir / "deterministic" / "dataset_profile.json"
    if not profile_path.exists():
        profile_path = run_dir / "dataset_profile.json"
    if not profile_path.exists():
        raise PromotionError("MISSING_AUTHORING_EVIDENCE", "deterministic/dataset_profile.json missing")
    profile_hash = _sha256_file(profile_path)
    if not profile_hash:
        raise PromotionError("MISSING_AUTHORING_EVIDENCE", "dataset_profile_hash is empty")
    profile_payload = _read_json(profile_path)

    readiness = evaluate_promotion_readiness(run_dir)
    gate_pass_candidates = readiness["authoring_gate_pass_candidates"]

    executable_candidates, exclusions_detected = filter_executable_candidates(
        gate_pass_candidates, run_dir, capability_exclusions=capability_exclusions
    )

    non_selected_types = {**readiness["non_gate_pass_types"], **exclusions_detected}
    selected_candidates = executable_candidates

    if not selected_candidates:
        raise PromotionError("NO_PROMOTABLE_CANDIDATES", "No candidates passed semantic gates, language gates, and capability checks")

    current_spec_hash = ""
    current_cat_hash = ""
    current_contract_hash = ""
    current_manifest_hash = ""
    current_lang_hash = ""

    old_active_types: tuple[str, ...] = ()
    existing_catalog_tasks: list[SemanticTaskSpec] | None = None

    comparator_registry = load_comparator_registry_from_root(resource_root)

    registry = load_dataset_registry(resource_root)
    if dataset_id in registry:
        spec_obj = load_dataset_spec_from_root(resource_root, dataset_id)
        current_spec_hash = spec_obj.logical_hash()

        try:
            cat_obj = load_semantic_catalog_from_root(resource_root, dataset_id)
        except PromotionError:
            cat_obj = None
        if cat_obj is not None:
            current_cat_hash = cat_obj.logical_hash()
            existing_catalog_tasks = list(cat_obj.tasks)
            old_active_types = tuple(t.type_id for t in cat_obj.tasks)

        contract_obj = load_production_contract_from_root(resource_root, dataset_id)
        if contract_obj is not None:
            current_contract_hash = contract_obj.fingerprint()

        manifest_obj = load_promotion_manifest_from_root(resource_root, dataset_id)
        if manifest_obj is not None:
            current_manifest_hash = manifest_obj.fingerprint()

    lang_registry_path = resource_root / "language" / "production_registry.json"
    if lang_registry_path.exists():
        lang_reg = load_language_registry(lang_registry_path)
        current_lang_hash = lang_reg.registry_hash

    raw_candidates_map: dict[str, dict[str, Any]] = {}
    prim_file = get_parsed_response_path(run_dir, STAGE_PRIMITIVE_DISCOVERY)
    if prim_file.exists():
        for c in _read_json(prim_file).get("candidates", []):
            if "candidate_id" in c:
                raw_candidates_map[c["candidate_id"]] = c

    comp_file = get_parsed_response_path(run_dir, STAGE_COMPOSITE_DISCOVERY)
    if comp_file.exists():
        comp_data = _read_json(comp_file)
        for c in comp_data.get("composites", comp_data.get("candidates", [])):
            cid = c.get("composite_id", c.get("candidate_id"))
            if cid:
                raw_candidates_map[cid] = c

    id_mapping = resolve_canonical_ids(
        dataset_id,
        selected_candidates,
        readiness["reconciliation"],
        raw_candidates_map=raw_candidates_map,
        existing_catalog_tasks=existing_catalog_tasks,
        canonical_id_overrides=canonical_id_overrides,
    )
    proposed_promoted_types = tuple(sorted(id_mapping[cid] for cid in selected_candidates))

    if promotion_mode == "replace":
        proposed_active_types = proposed_promoted_types
    else:
        proposed_active_types = tuple(sorted(set(old_active_types) | set(proposed_promoted_types)))

    old_set = set(old_active_types)
    new_set = set(proposed_active_types)
    reconciled_set = set(id_mapping.values())
    diff = {
        "added_types": sorted(new_set - old_set),
        "removed_types": sorted(old_set - new_set),
        "retained_types": sorted(old_set & new_set),
        "renamed_reused_types": sorted(reconciled_set & old_set),
    }

    candidate_catalog = compile_candidate_semantic_catalog(
        dataset_id,
        selected_candidates,
        id_mapping,
        run_dir,
        readiness["reconciliation"],
        existing_catalog_tasks,
        promotion_mode,
        comparator_registry=comparator_registry,
    )

    compiled_type_ids = set(t["type_id"] for t in candidate_catalog.get("tasks", []))
    if compiled_type_ids != set(proposed_active_types):
        raise PromotionError(
            "INCONSISTENT_CATALOG_ACTIVE_SET",
            f"Catalog tasks ({compiled_type_ids}) do not match proposed active set ({proposed_active_types})",
        )

    candidate_language = compile_candidate_language_resource(dataset_id, selected_candidates, id_mapping, run_dir)

    # --- Phase 4.2: declarative field-spec resolution -----------------------
    required_fields = _required_source_fields(candidate_catalog)
    if field_specs is None:
        field_specs, field_specs_source, field_specs_hash = _resolve_declarative_field_specs(
            dataset_id, profile_payload, required_fields, resource_root
        )
    else:
        explicit_bundle = SemanticFieldSpecBundle(
            dataset_id=dataset_id, source="explicit", field_specs=field_specs
        )
        field_specs_source = "explicit"
        field_specs_hash = explicit_bundle.logical_hash()

    staged_preflight_res = run_staged_preflight_for_promotion(
        dataset_id, candidate_catalog, field_specs=field_specs, run_dir=run_dir
    )
    staged_pass = (staged_preflight_res["status"] == "PREFLIGHT_PASS")

    # --- Phase 4.2: candidate language registry (SCRATCH ONLY) --------------
    canonical_registry = load_language_registry_from_root(resource_root)
    capability_resource = load_candidate_capability_resource(candidate_capabilities_path)
    capability_entries = validate_candidate_capability_resource(
        capability_resource, expected_language=canonical_registry.language
    )
    candidate_registry = build_candidate_language_registry(
        canonical_registry,
        capability_entries,
        capability_language=capability_resource["language"],
    )
    cap_scratch_dir = (language_capability_root or DEFAULT_LANGUAGE_CAPABILITY_SCRATCH) / dataset_id
    candidate_registry_path = (
        cap_scratch_dir / "candidate_production_language_registry.json"
    )
    write_registry_deterministically(candidate_registry, candidate_registry_path)

    candidate_preflight_res = run_staged_preflight_for_promotion(
        dataset_id,
        candidate_catalog,
        field_specs=field_specs,
        run_dir=run_dir,
        registry=candidate_registry,
        registry_path=candidate_registry_path,
    )
    candidate_pass = (candidate_preflight_res["status"] == "PREFLIGHT_PASS")
    language_registry_change_required = (
        candidate_registry.registry_hash != canonical_registry.registry_hash
    )
    cross_delta: list[dict[str, Any]] = []
    if resource_root.resolve() == RESOURCE_ROOT.resolve():
        cross_delta = _cross_dataset_preflight_delta(
            canonical_registry, candidate_registry, candidate_registry_path
        )
    impact = _language_registry_impact(
        resource_root,
        canonical_registry,
        candidate_registry,
        cross_dataset_preflight_delta=cross_delta,
    )
    _write_json(
        cap_scratch_dir / "registry_impact.json",
        {"dataset_id": dataset_id, **impact},
    )

    staged_catalog_type_ids = tuple(sorted(t["type_id"] for t in candidate_catalog.get("tasks", [])))
    staged_accepted_type_ids = tuple(sorted(a.dataset_type_id for a in staged_preflight_res.get("accepted_types", [])))
    staged_language_candidate_type_ids = tuple(sorted(set(e["type_id"] for e in candidate_language.get("entries", []))))

    authoring_semantic_pass_types = tuple(sorted(readiness.get("authoring_semantic_pass_types", [])))
    executable_promotion_candidates = proposed_promoted_types
    staged_preflight_types = staged_accepted_type_ids

    identity_match = (
        set(staged_catalog_type_ids) == set(proposed_active_types)
        and set(staged_accepted_type_ids) == set(proposed_active_types)
    )

    if not identity_match:
        raise PromotionError(
            "STAGED_PREFLIGHT_IDENTITY_MISMATCH",
            f"Staged preflight types {staged_accepted_type_ids} do not match proposed active types {proposed_active_types}",
        )

    selected_type_blockers: list[str] = []
    for cid in selected_candidates:
        if cid not in gate_pass_candidates:
            selected_type_blockers.append(f"GATE_NOT_PASSED:{cid}")

    apply_ready = (
        bool(selected_candidates)
        and len(selected_type_blockers) == 0
        and staged_pass
        and identity_match
    )

    primitive_executable_selected_count = sum(
        1 for c in selected_candidates if c in readiness["authoring_gate_pass_primitives"]
    )
    composite_executable_selected_count = sum(
        1 for c in selected_candidates if c in readiness["authoring_gate_pass_composites"]
    )
    promotable_total = primitive_executable_selected_count + composite_executable_selected_count
    if promotable_total != (
        primitive_executable_selected_count + composite_executable_selected_count
    ) or promotable_total != len(selected_candidates):
        raise PromotionError(
            "PROMOTABLE_ACCOUNTING_INVARIANT_VIOLATED",
            f"total={promotable_total} primitive={primitive_executable_selected_count} "
            f"composite={composite_executable_selected_count} selected={len(selected_candidates)}",
        )
    staged_preflight_type_count = len(staged_accepted_type_ids)
    staged_preflight_pass_type_count = len(staged_accepted_type_ids) if staged_pass else 0

    bundle = PromotionBundle(
        schema_version=1,
        dataset_id=dataset_id,
        source_run_id=run_dir.name,
        source_run_path=str(run_dir),
        promotion_mode=promotion_mode,
        authoring_run_hash=authoring_hash,
        source_documentation_hash=doc_hash,
        dataset_profile_hash=profile_hash,
        primitive_candidates=readiness["primitive_candidates"],
        primitive_semantic_accepted=readiness["primitive_semantic_pass_count"],
        primitive_semantic_pass_count=readiness["primitive_semantic_pass_count"],
        composite_candidates=readiness["composite_candidates"],
        composite_semantic_accepted=readiness["composite_semantic_pass_count"],
        composite_semantic_pass_count=readiness["composite_semantic_pass_count"],
        primitive_language_ready=readiness["primitive_authoring_language_gate_pass_count"],
        primitive_authoring_language_gate_pass_count=readiness["primitive_authoring_language_gate_pass_count"],
        composite_language_ready=readiness["composite_authoring_language_gate_pass_count"],
        composite_authoring_language_gate_pass_count=readiness["composite_authoring_language_gate_pass_count"],
        authoring_language_gate_pass_count=readiness["authoring_language_gate_pass_count"],
        primitive_executable_selected_count=primitive_executable_selected_count,
        staged_canonical_language_preflight_pass_count=staged_preflight_pass_type_count,
        staged_preflight_type_count=staged_preflight_type_count,
        staged_preflight_pass_type_count=staged_preflight_pass_type_count,
        promotable_primitive_count=primitive_executable_selected_count,
        promotable_composite_count=composite_executable_selected_count,
        promotable_total=promotable_total,
        promotable_types=proposed_promoted_types,
        non_promotable_types=non_selected_types,
        discovered_semantic_pass_types=authoring_semantic_pass_types,
        non_selected_types=non_selected_types,
        selected_promotion_types=proposed_promoted_types,
        selected_type_blockers=tuple(selected_type_blockers),
        canonical_id_mapping=id_mapping,
        old_active_types=old_active_types,
        proposed_active_types=proposed_active_types,
        semantic_diff=diff,
        candidate_semantic_catalog=candidate_catalog,
        candidate_language_resource=candidate_language,
        expected_current_hashes={
            "dataset_spec_hash": current_spec_hash,
            "semantic_catalog_hash": current_cat_hash,
            "production_contract_hash": current_contract_hash,
            "promotion_manifest_hash": current_manifest_hash,
            "language_registry_hash": current_lang_hash,
        },
        staged_language_preflight=staged_preflight_res,
        semantic_field_specs_hash=field_specs_hash,
        semantic_field_specs_source=field_specs_source,
        canonical_language_registry_hash=canonical_registry.registry_hash,
        candidate_language_registry_hash=candidate_registry.registry_hash,
        candidate_language_registry=candidate_registry.model_dump(mode="json"),
        candidate_language_preflight=candidate_preflight_res,
        language_infra_ready=candidate_pass,
        language_registry_change_required=language_registry_change_required,
        fresh_contract_fingerprints={
            "contract_fingerprint": staged_preflight_res.get("contract_fingerprint", ""),
        },
        stale_current_artifacts=(),
        evidence_summary={
            "run_manifest_hash": authoring_hash,
            "profile_hash": profile_hash,
            "documentation_hash": doc_hash,
            "readiness": readiness,
        },
        zero_llm_evidence={
            "framework_calls": 0,
            "dataset_apply": False,
        },
        authoring_semantic_pass_types=authoring_semantic_pass_types,
        executable_promotion_candidates=executable_promotion_candidates,
        staged_preflight_types=staged_preflight_types,
        staged_catalog_type_ids=staged_catalog_type_ids,
        staged_accepted_type_ids=staged_accepted_type_ids,
        staged_language_candidate_type_ids=staged_language_candidate_type_ids,
        prepare_status="PREPARED",
        apply_ready=apply_ready,
    )

    out_dir = output_root or (DEFAULT_SCRATCH_DIR / dataset_id / run_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_json(out_dir / "promotion_bundle.json", bundle.model_dump(mode="json"))
    _write_json(out_dir / "candidate_semantic_catalog.json", candidate_catalog)
    _write_json(out_dir / "candidate_language_resource.json", candidate_language)

    return bundle


def _default_atomic_replace(src: str, dst: str) -> None:
    os.replace(src, dst)


def _assert_destination_within_root(path: Path, resource_root: Path) -> None:
    try:
        path.resolve().relative_to(resource_root.resolve())
    except ValueError:
        raise PromotionError(
            "CANONICAL_DESTINATION_OUTSIDE_RESOURCE_ROOT",
            f"Refusing to write outside resource_root: {path}",
        )


def apply_promotion(
    bundle_input: Path | PromotionBundle,
    *,
    resource_root: Path = RESOURCE_ROOT,
    replace_fn: Callable[[str, str], Any] = _default_atomic_replace,
    validation_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Applies a previously prepared PromotionBundle to canonical resources transaction-safely.

    Every canonical read/write is resolved through ``resource_root``. Isolated
    tests can therefore supply ``tmp_path / "resources"`` and guarantee that no
    path under the real ``ROOT/resources`` is touched.
    """
    if isinstance(bundle_input, Path):
        bundle = PromotionBundle.model_validate(_read_json(bundle_input))
    else:
        bundle = bundle_input

    dataset_id = bundle.dataset_id

    if not bundle.apply_ready:
        raise PromotionError(
            "PROMOTION_NOT_APPLY_READY",
            f"Cannot apply promotion bundle for dataset '{dataset_id}': apply_ready is False (staged preflight status = '{bundle.staged_language_preflight.get('status')}')",
        )

    # 1. Authoring source freshness verification (fail closed: if the bundle
    #    claims evidence existed at PREPARE time, that evidence must still exist
    #    and be byte-identical).
    run_dir = Path(bundle.source_run_path)
    if bundle.authoring_run_hash or bundle.dataset_profile_hash or bundle.source_documentation_hash:
        if not run_dir.exists():
            raise PromotionError("PROMOTION_SOURCE_EVIDENCE_STALE", f"Source run directory missing: {run_dir}")

    if bundle.authoring_run_hash:
        manifest_p = run_dir / "run_manifest.json"
        if not manifest_p.exists():
            raise PromotionError("PROMOTION_SOURCE_EVIDENCE_STALE", f"run_manifest.json missing: {manifest_p}")
        if _sha256_file(manifest_p) != bundle.authoring_run_hash:
            raise PromotionError("PROMOTION_SOURCE_EVIDENCE_STALE", "run_manifest.json hash changed")

    if bundle.dataset_profile_hash:
        profile_p = run_dir / "deterministic" / "dataset_profile.json"
        if not profile_p.exists():
            raise PromotionError("PROMOTION_SOURCE_EVIDENCE_STALE", f"dataset_profile.json missing: {profile_p}")
        if _sha256_file(profile_p) != bundle.dataset_profile_hash:
            raise PromotionError("PROMOTION_SOURCE_EVIDENCE_STALE", "dataset_profile.json hash changed")

    if bundle.source_documentation_hash:
        doc_dir = run_dir / "source_documentation"
        if not doc_dir.exists():
            raise PromotionError("PROMOTION_SOURCE_EVIDENCE_STALE", f"source_documentation directory missing: {doc_dir}")
        curr_doc_hash = canonical_hash([_sha256_file(p) for p in sorted(doc_dir.glob("*"))])
        if curr_doc_hash != bundle.source_documentation_hash:
            raise PromotionError("PROMOTION_SOURCE_EVIDENCE_STALE", "source_documentation hash changed")

    # 2. Resolve canonical destinations via resource_root.
    entry = dataset_registry_entry(resource_root, dataset_id)
    for key in ("dataset_spec", "semantic_catalog", "production_contract", "promotion_manifest"):
        if not entry.get(key):
            raise PromotionError("CANONICAL_REGISTRY_ENTRY_INCOMPLETE", f"{dataset_id}:{key}")

    catalog_path = _resolve_entry_path(resource_root, entry["semantic_catalog"])
    contract_path = _resolve_entry_path(resource_root, entry["production_contract"])
    manifest_path = _resolve_entry_path(resource_root, entry["promotion_manifest"])
    spec_path = _resolve_entry_path(resource_root, entry["dataset_spec"])
    for dest in (catalog_path, contract_path, manifest_path):
        _assert_destination_within_root(dest, resource_root)

    comparator_registry = load_comparator_registry_from_root(resource_root)

    # 3. Verify stale bundle protection (all expected current hashes match)
    spec_obj = DatasetSpec.model_validate(_read_json(spec_path))
    current_spec_hash = spec_obj.logical_hash()

    cat_obj = load_semantic_catalog(catalog_path)
    current_cat_hash = cat_obj.logical_hash()

    contract_obj = load_production_contract_from_root(resource_root, dataset_id)
    current_contract_hash = contract_obj.fingerprint() if contract_obj else ""

    manifest_obj = load_promotion_manifest_from_root(resource_root, dataset_id)
    current_manifest_hash = manifest_obj.fingerprint() if manifest_obj else ""

    lang_registry_path = resource_root / "language" / "production_registry.json"
    lang_reg = load_language_registry(lang_registry_path)
    current_lang_hash = lang_reg.registry_hash

    exp = bundle.expected_current_hashes
    if (
        (exp.get("dataset_spec_hash") and exp["dataset_spec_hash"] != current_spec_hash)
        or (exp.get("semantic_catalog_hash") and exp["semantic_catalog_hash"] != current_cat_hash)
        or (exp.get("production_contract_hash") and exp["production_contract_hash"] != current_contract_hash)
        or (exp.get("promotion_manifest_hash") and exp["promotion_manifest_hash"] != current_manifest_hash)
        or (exp.get("language_registry_hash") and exp["language_registry_hash"] != current_lang_hash)
    ):
        raise PromotionError("PROMOTION_BUNDLE_STALE", "Current canonical resource hashes differ from prepare-time hashes")

    # 4. Prepare new canonical catalog object and compute logical hash
    candidate_cat_obj = SemanticCatalog.model_validate(bundle.candidate_semantic_catalog)
    cat_issues = validate_catalog(
        candidate_cat_obj,
        comparator_ids={c.comparator_id for c in comparator_registry.comparators},
    )
    if cat_issues:
        raise PromotionError("CANONICAL_RESOURCE_VALIDATION_FAILED", ",".join(cat_issues))
    candidate_cat_logical_hash = candidate_cat_obj.logical_hash()

    # 5. Prepare ProductionContract and PromotionManifest with exact logical hashes
    comp_ids = tuple(sorted({t.comparator_id for t in candidate_cat_obj.tasks if t.comparator_id}))
    comp_hashes = {cid: comparator_registry.by_id(cid).logical_hash() for cid in comp_ids if comp_ids}
    comp_reg_hash = comparator_set_hash(comp_ids, comparator_registry) if comp_ids else ""

    source_revision = spec_obj.source.get("revision_identity")
    if not source_revision:
        raise PromotionError(
            "MISSING_DATASET_SOURCE_REVISION",
            f"DatasetSpec source.revision_identity missing for {dataset_id}",
        )

    contract = ProductionContract(
        schema_version=1,
        dataset_id=dataset_id,
        dataset_spec_hash=current_spec_hash,
        semantic_catalog_hash=candidate_cat_logical_hash,
        comparator_registry_hash=comp_reg_hash,
        comparator_referenced_hashes=comp_hashes,
        language_registry_hash=current_lang_hash,
        active_semantic_types=bundle.proposed_active_types,
        production_planner_policy={
            "allowed_splits": list(spec_obj.allowed_splits),
            "plan_location": spec_obj.production_constraints.get("plan_location", f"data/materialized/{dataset_id}/current"),
        },
        promotion_evidence={
            "authoring_run_id": bundle.source_run_id,
            "promotion_mode": bundle.promotion_mode,
            "bundle_fingerprint": bundle.fingerprint(),
        },
        status="PROMOTED",
        migration_source="authoring_promotion_bridge",
        promotion_fingerprint="",
    )
    contract_fingerprint = contract.fingerprint()
    contract = contract.model_copy(update={"promotion_fingerprint": contract_fingerprint})

    manifest = PromotionManifest(
        schema_version=1,
        dataset_id=dataset_id,
        status="PROMOTED",
        migration_source="authoring_promotion_bridge",
        dataset_spec_hash=current_spec_hash,
        semantic_catalog_hash=candidate_cat_logical_hash,
        comparator_registry_hash=comp_reg_hash,
        language_registry_hash=current_lang_hash,
        production_contract_hash=contract_fingerprint,
        promotion_evidence={
            "authoring_run_id": bundle.source_run_id,
            "promotion_mode": bundle.promotion_mode,
            "bundle_fingerprint": bundle.fingerprint(),
        },
        promoted_semantic_types=bundle.proposed_active_types,
        source_revision=source_revision,
        allowed_splits=tuple(spec_obj.allowed_splits),
        semantic_change_status="SEMANTIC_UPGRADE" if bundle.semantic_diff["added_types"] else "NO_SEMANTIC_CHANGE",
        promotion_fingerprint="",
    )
    manifest_fingerprint = manifest.fingerprint()
    manifest = manifest.model_copy(update={"promotion_fingerprint": manifest_fingerprint})

    # 6. Atomic multi-resource replacement with robust rollback protection.
    txn = uuid.uuid4().hex
    backups: dict[Path, bytes] = {}
    newly_created: list[Path] = []
    for p in (catalog_path, contract_path, manifest_path):
        if p.exists():
            backups[p] = p.read_bytes()
        else:
            newly_created.append(p)

    tmp_catalog = catalog_path.parent / f".{catalog_path.name}.{txn}.tmp"
    tmp_contract = contract_path.parent / f".{contract_path.name}.{txn}.tmp"
    tmp_manifest = manifest_path.parent / f".{manifest_path.name}.{txn}.tmp"
    temps = (tmp_catalog, tmp_contract, tmp_manifest)

    try:
        _write_json(tmp_catalog, bundle.candidate_semantic_catalog)
        _write_json(tmp_contract, contract.model_dump(mode="json"))
        _write_json(tmp_manifest, manifest.model_dump(mode="json"))

        replace_fn(str(tmp_catalog), str(catalog_path))
        replace_fn(str(tmp_contract), str(contract_path))
        replace_fn(str(tmp_manifest), str(manifest_path))

        if validation_hook is not None:
            validation_hook()

        # Cross-resource final validation
        reloaded_cat = load_semantic_catalog(catalog_path)
        reloaded_contract = ProductionContract.model_validate(_read_json(contract_path))
        reloaded_manifest = PromotionManifest.model_validate(_read_json(manifest_path))
        reloaded_spec = DatasetSpec.model_validate(_read_json(spec_path))
        reloaded_lang = load_language_registry(lang_registry_path)

        catalog_tasks = tuple(sorted(t.type_id for t in reloaded_cat.tasks))
        if catalog_tasks != tuple(sorted(reloaded_contract.active_semantic_types)):
            raise PromotionError("CANONICAL_RESOURCE_VALIDATION_FAILED", "Catalog tasks != contract active types")
        if catalog_tasks != tuple(sorted(reloaded_manifest.promoted_semantic_types)):
            raise PromotionError("CANONICAL_RESOURCE_VALIDATION_FAILED", "Catalog tasks != manifest promoted types")

        if reloaded_contract.semantic_catalog_hash != reloaded_cat.logical_hash():
            raise PromotionError("CANONICAL_RESOURCE_VALIDATION_FAILED", "contract catalog hash mismatch")
        if reloaded_manifest.semantic_catalog_hash != reloaded_cat.logical_hash():
            raise PromotionError("CANONICAL_RESOURCE_VALIDATION_FAILED", "manifest catalog hash mismatch")
        if reloaded_manifest.production_contract_hash != reloaded_contract.fingerprint():
            raise PromotionError("CANONICAL_RESOURCE_VALIDATION_FAILED", "manifest contract hash mismatch")
        if reloaded_contract.language_registry_hash != reloaded_lang.registry_hash:
            raise PromotionError("CANONICAL_RESOURCE_VALIDATION_FAILED", "contract language hash mismatch")
        if reloaded_contract.dataset_spec_hash != reloaded_spec.logical_hash():
            raise PromotionError("CANONICAL_RESOURCE_VALIDATION_FAILED", "contract dataset spec hash mismatch")

        if reloaded_contract.promotion_fingerprint != reloaded_contract.fingerprint():
            raise PromotionError("CANONICAL_RESOURCE_VALIDATION_FAILED", "contract self-fingerprint mismatch")
        if not reloaded_contract.promotion_fingerprint:
            raise PromotionError("CANONICAL_RESOURCE_VALIDATION_FAILED", "contract promotion_fingerprint is empty")
        if reloaded_manifest.promotion_fingerprint != reloaded_manifest.fingerprint():
            raise PromotionError("CANONICAL_RESOURCE_VALIDATION_FAILED", "manifest self-fingerprint mismatch")
        if not reloaded_manifest.promotion_fingerprint:
            raise PromotionError("CANONICAL_RESOURCE_VALIDATION_FAILED", "manifest promotion_fingerprint is empty")
        if reloaded_manifest.source_revision != source_revision:
            raise PromotionError("CANONICAL_RESOURCE_VALIDATION_FAILED", "manifest source_revision mismatch")

    except Exception as exc:
        try:
            for p, data in backups.items():
                p.write_bytes(data)
            for p in newly_created:
                if p.exists():
                    p.unlink()
            for tmp in temps:
                if tmp.exists():
                    tmp.unlink()
        except Exception as rb_exc:
            raise PromotionError("PROMOTION_ROLLBACK_FAILED", f"Rollback failed: {rb_exc}") from rb_exc

        affected = ", ".join(
            str(p) for p in (catalog_path, contract_path, manifest_path) if p.exists()
        )
        raise PromotionError(
            "TRANSACTION_FAILED_ROLLED_BACK",
            f"{exc}; rolled back affected canonical paths: {affected}",
        ) from exc

    return {
        "dataset_id": dataset_id,
        "status": "APPLIED",
        "promoted_types": list(bundle.proposed_active_types),
        "semantic_diff": bundle.semantic_diff,
        "contract_fingerprint": contract_fingerprint,
        "manifest_fingerprint": manifest_fingerprint,
        "semantic_change": bool(bundle.semantic_diff["added_types"] or bundle.semantic_diff["removed_types"]),
        "production_plan_status": "STALE_REQUIRES_REPLAN" if (bundle.semantic_diff["added_types"] or bundle.semantic_diff["removed_types"]) else "CURRENT",
    }


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset ID (e.g. vimedcss)")
    parser.add_argument("--run-id", help="Authoring run ID directory name under outputs/runs/<dataset>/")
    parser.add_argument("--run-dir", help="Full path to authoring run directory")
    parser.add_argument(
        "--promotion-mode",
        choices=["replace", "merge"],
        help="Promotion mode: replace or merge",
    )
    
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--prepare", action="store_true", help="Run PREPARE dry-run (0 canonical mutations)")
    action_group.add_argument("--apply", action="store_true", help="Run APPLY transaction (requires --bundle)")

    parser.add_argument("--bundle", type=Path, help="Path to prepared promotion_bundle.json for --apply")
    parser.add_argument("--output-root", type=Path, help="Custom output scratch root")
    parser.add_argument("--canonical-id-overrides", type=Path, help="JSON file with canonical ID overrides")
    parser.add_argument("--capability-exclusions", type=Path, help="JSON file with capability exclusions")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _cli().parse_args(argv)

    canonical_id_overrides = None
    if args.canonical_id_overrides:
        canonical_id_overrides = _read_json(args.canonical_id_overrides)

    capability_exclusions = None
    if args.capability_exclusions:
        capability_exclusions = _read_json(args.capability_exclusions)

    if args.prepare:
        if not args.promotion_mode:
            print(json.dumps({"error": "PROMOTION_MODE_REQUIRED", "detail": "--promotion-mode replace|merge is required for --prepare"}, indent=2))
            return 2

        if args.run_dir:
            run_dir = Path(args.run_dir)
        elif args.run_id:
            run_dir = ROOT / "outputs" / "runs" / args.dataset / args.run_id
        else:
            runs_parent = ROOT / "outputs" / "runs" / args.dataset
            if not runs_parent.exists():
                print(json.dumps({"error": "RUN_NOT_FOUND", "detail": f"No runs directory for {args.dataset}"}, indent=2))
                return 1
            run_dirs = sorted([d for d in runs_parent.iterdir() if d.is_dir() and (d / "run_manifest.json").exists()])
            if not run_dirs:
                print(json.dumps({"error": "RUN_NOT_FOUND", "detail": f"No valid run found under {runs_parent}"}, indent=2))
                return 1
            run_dir = run_dirs[-1]

        try:
            bundle = prepare_promotion(
                args.dataset,
                run_dir,
                promotion_mode=args.promotion_mode,
                output_root=args.output_root,
                canonical_id_overrides=canonical_id_overrides,
                capability_exclusions=capability_exclusions,
            )
            print(json.dumps({
                "status": "PREPARED",
                "apply_ready": bundle.apply_ready,
                "dataset_id": bundle.dataset_id,
                "bundle_fingerprint": bundle.fingerprint(),
                "promotable_total": bundle.promotable_total,
                "promotable_types": list(bundle.promotable_types),
                "staged_preflight_status": bundle.staged_language_preflight.get("status"),
                "staged_blocking_issues": bundle.staged_language_preflight.get("blocking_issues"),
                "canonical_mutations": 0,
            }, indent=2))
            return 0
        except PromotionError as exc:
            print(json.dumps({"error": exc.code, "detail": exc.detail}, indent=2))
            return 1

    elif args.apply:
        if not args.bundle:
            print(json.dumps({"error": "BUNDLE_REQUIRED", "detail": "--bundle <path> is required for --apply"}, indent=2))
            return 2
        try:
            res = apply_promotion(args.bundle)
            print(json.dumps(res, indent=2))
            return 0
        except PromotionError as exc:
            print(json.dumps({"error": exc.code, "detail": exc.detail}, indent=2))
            return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

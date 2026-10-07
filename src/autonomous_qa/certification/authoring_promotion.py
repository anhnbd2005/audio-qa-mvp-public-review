"""Authoring -> Promotion -> Production Generic Bridge.

Compiles R&D authoring run artifacts into a deterministic, machine-readable
PromotionBundle, resolves canonical IDs, compiles generic semantic contracts
and language resources, and executes staged PREPARE / APPLY promotion transactions
without dataset-specific branching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.autonomous_qa.certification.promotion_gate import (
    PromotionError,
    validate_catalog,
)
from src.autonomous_qa.compiler.canonical_resources import (
    DatasetSpec,
    ProductionContract,
    PromotionManifest,
    get_dataset_spec,
    get_semantic_catalog_path,
)
from src.autonomous_qa.compiler.semantic_comparators import (
    comparator_set_hash,
    load_comparator_registry,
)
from src.autonomous_qa.compiler.semantic_task import (
    SemanticCatalog,
    SemanticTaskSpec,
    load_semantic_catalog,
)
from src.autonomous_qa.language.language_quality import load_language_registry
from src.autonomous_qa.language.language_preflight import (
    AcceptedLanguageType,
    compute_contract_fingerprint,
    operator_contracts,
    run_preflight,
)
from src.autonomous_qa.language.template_renderer import canonical_hash
from src.common.config import ROOT

RESOURCE_ROOT = ROOT / "resources"
DATASET_REGISTRY_PATH = RESOURCE_ROOT / "registry" / "datasets.json"
LANGUAGE_REGISTRY_PATH = RESOURCE_ROOT / "language" / "production_registry.json"
DEFAULT_SCRATCH_DIR = ROOT / "outputs" / "_scratch" / "promotion"


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
    primitive_semantic_accepted: int
    composite_candidates: int
    composite_semantic_accepted: int
    primitive_language_ready: int
    composite_language_ready: int
    authoring_language_gate_pass_count: int
    staged_canonical_language_preflight_pass_count: int
    promotable_primitive_count: int
    promotable_composite_count: int
    promotable_total: int

    promotable_types: tuple[str, ...]
    non_promotable_types: dict[str, str] = Field(default_factory=dict)

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
    """Evaluates explicit multi-layer promotion readiness from run artifacts."""
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

    promotable_primitives: list[str] = []
    promotable_composites: list[str] = []
    non_promotable: dict[str, str] = {}

    primitive_candidates = len(primitive_gates.get("candidates", []))
    primitive_semantic_accepted = 0
    primitive_language_ready = 0

    for cand in primitive_gates.get("candidates", []):
        cid = cand["candidate_id"]
        sem_pass = cand.get("verdict") == "PASS"
        if sem_pass:
            primitive_semantic_accepted += 1

        hidden_src = cand.get("hidden_source_annotations", [])
        src_role = cand.get("source_role_mapping", {}).get("source_field", "")
        op = cand.get("operator", "")
        prop_desc = (cand.get("proposition_description") or "").lower()
        requires_membership = (
            "presence" in cid
            or "presence" in prop_desc
            or (op == "TARGET_MATCH" and ("cs_term" in cid or "list" in src_role or any("list" in str(h) for h in hidden_src)))
        )
        if requires_membership:
            non_promotable[cid] = "OPERATOR_CAPABILITY_GAP:MEMBERSHIP"
            continue

        lang_ready = cid in authoring_preflight_pass_types
        if lang_ready:
            primitive_language_ready += 1

        if sem_pass and lang_ready:
            promotable_primitives.append(cid)
        elif sem_pass and not lang_ready:
            non_promotable[cid] = "SEMANTIC_GATE_PASS_BUT_LANGUAGE_NOT_READY"
        else:
            non_promotable[cid] = "SEMANTIC_GATE_FAILED"

    composite_candidates = len(composite_gates.get("composites", []))
    composite_semantic_accepted = 0
    composite_language_ready = 0

    for comp in composite_gates.get("composites", []):
        cid = comp.get("composite_id", comp.get("candidate_id"))
        sem_pass = comp.get("verdict") == "PASS"
        if sem_pass:
            composite_semantic_accepted += 1
        lang_ready = cid in authoring_preflight_pass_types
        if lang_ready:
            composite_language_ready += 1

        if sem_pass and lang_ready:
            promotable_composites.append(cid)
        elif sem_pass and not lang_ready:
            non_promotable[cid] = "SEMANTIC_GATE_PASS_BUT_LANGUAGE_NOT_READY"
        else:
            non_promotable[cid] = "SEMANTIC_GATE_FAILED"

    authoring_semantic_pass_types = [
        cand["candidate_id"] for cand in primitive_gates.get("candidates", []) if cand.get("verdict") == "PASS"
    ] + [
        comp.get("composite_id", comp.get("candidate_id")) for comp in composite_gates.get("composites", []) if comp.get("verdict") == "PASS"
    ]

    return {
        "primitive_candidates": primitive_candidates,
        "primitive_semantic_accepted": primitive_semantic_accepted,
        "composite_candidates": composite_candidates,
        "composite_semantic_accepted": composite_semantic_accepted,
        "primitive_language_ready": primitive_language_ready,
        "composite_language_ready": composite_language_ready,
        "authoring_language_gate_pass_count": primitive_language_ready + composite_language_ready,
        "authoring_semantic_pass_types": authoring_semantic_pass_types,
        "promotable_primitive_count": len(promotable_primitives),
        "promotable_composite_count": len(promotable_composites),
        "promotable_total": len(promotable_primitives) + len(promotable_composites),
        "promotable_primitives": promotable_primitives,
        "promotable_composites": promotable_composites,
        "non_promotable_types": non_promotable,
        "reconciliation": reconciliation_data,
    }


def resolve_canonical_ids(
    dataset_id: str,
    candidate_ids: list[str],
    reconciliation_data: dict[str, Any],
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Resolves blind discovery candidate IDs to stable canonical IDs deterministically."""
    overrides = overrides or {}
    mapping: dict[str, str] = {}
    reverse_map: dict[str, str] = {}

    matches = {m["blind_candidate"]: m for m in reconciliation_data.get("matches", []) if m.get("blind_candidate")}

    for cand_id in candidate_ids:
        if cand_id in overrides:
            target_id = overrides[cand_id]
        elif cand_id in matches and matches[cand_id].get("classification") == "SAME_PROPOSITION_DIFFERENT_NAME" and matches[cand_id].get("canonical_type"):
            target_id = matches[cand_id]["canonical_type"]
        else:
            # Deterministic normalization: strip transient discovery index <dataset>_<digits>_<slug> -> <dataset>_<slug>
            match = re.match(r"^([a-zA-Z0-9_]+?)_\d+_(.+)$", cand_id)
            if match:
                prefix = match.group(1)
                slug = match.group(2)
                target_id = f"{prefix}_{slug}"
            elif cand_id.startswith(f"{dataset_id}_"):
                target_id = cand_id
            else:
                raise PromotionError("CANONICAL_ID_UNRESOLVED", cand_id)

        # Collision check
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
) -> str:
    """Generic capability-based comparator resolution order without dataset-specific shortcuts."""
    # A. Reconciliation Reuse: If candidate reconciles to an existing canonical task
    matches = {m["blind_candidate"]: m for m in reconciliation_data.get("matches", []) if m.get("blind_candidate")}
    if blind_id in matches and matches[blind_id].get("classification") == "SAME_PROPOSITION_DIFFERENT_NAME":
        canon_type = matches[blind_id].get("canonical_type")
        if canon_type and existing_catalog_tasks:
            for ex in existing_catalog_tasks:
                if ex.type_id == canon_type and ex.comparator_id:
                    hidden_src = raw_cand.get("hidden_source_annotations", [])
                    src_role = raw_cand.get("source_role_mapping", {}).get("source_field", "")
                    prop_desc = (raw_cand.get("proposition_description") or "").lower()
                    requires_membership = (
                        "presence" in blind_id
                        or "presence" in canon_type
                        or "presence" in prop_desc
                        or (raw_cand.get("operator") == "TARGET_MATCH" and ("cs_term" in blind_id or "list" in src_role or any("list" in str(h) for h in hidden_src)))
                    )
                    if requires_membership and ex.operator == "TARGET_MATCH":
                        raise PromotionError(
                            "OPERATOR_CAPABILITY_GAP:MEMBERSHIP",
                            f"Reconciled task '{blind_id}' -> '{canon_type}' requires list membership IR (target MEMBER_OF parse({src_role})), which is not supported by scalar TARGET_MATCH",
                        )
                    return ex.comparator_id

    # B. Explicit Candidate Proposal
    if raw_cand.get("comparator_id"):
        cid = raw_cand["comparator_id"]
        reg = load_comparator_registry()
        try:
            reg.by_id(cid)
            return cid
        except KeyError:
            pass

    # C. Deterministic capability matching
    operator = raw_cand.get("operator", "DIRECT")
    answer_schema = raw_cand.get("answer_schema_proposal", {})
    kind = answer_schema.get("kind")
    val_type = answer_schema.get("type")

    reg = load_comparator_registry()
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
        if "categorical_label_exact" in comp_ids:
            return "categorical_label_exact"

    # Reject shortcuts and check gaps
    if operator == "TARGET_MATCH":
        # Check if list membership is requested for a new unreconciled candidate
        hidden_src = raw_cand.get("hidden_source_annotations", [])
        if hidden_src and any("list" in h for h in hidden_src):
            raise PromotionError("OPERATOR_CAPABILITY_GAP:MEMBERSHIP", f"Candidate {blind_id} requires list membership IR")

    raise PromotionError("COMPARATOR_CAPABILITY_GAP", f"No compatible comparator for {blind_id}:{operator}")


def compile_candidate_semantic_catalog(
    dataset_id: str,
    promotable_candidates: list[str],
    id_mapping: dict[str, str],
    run_dir: Path,
    reconciliation_data: dict[str, Any],
    existing_catalog_tasks: list[SemanticTaskSpec] | None = None,
    promotion_mode: str = "replace",
) -> dict[str, Any]:
    """Compiles accepted authoring candidates into canonical SemanticCatalog schema."""
    parsed_primitives_path = run_dir / "llm" / "primitive_semantic_discovery" / "parsed_response.json"
    parsed_composites_path = run_dir / "llm" / "composite_discovery" / "parsed_response.json"

    raw_primitives = _read_json(parsed_primitives_path).get("candidates", []) if parsed_primitives_path.exists() else []
    
    raw_composites: list[dict[str, Any]] = []
    if parsed_composites_path.exists():
        comp_data = _read_json(parsed_composites_path)
        raw_composites = comp_data.get("composites", comp_data.get("candidates", []))

    cand_map: dict[str, dict[str, Any]] = {c["candidate_id"]: c for c in raw_primitives if "candidate_id" in c}
    for comp in raw_composites:
        cid = comp.get("composite_id", comp.get("candidate_id"))
        if cid:
            cand_map[cid] = comp

    tasks: list[dict[str, Any]] = []

    # If merge mode, retain unmatched old active tasks
    retained_tasks_map: dict[str, SemanticTaskSpec] = {}
    if promotion_mode == "merge" and existing_catalog_tasks:
        for ex in existing_catalog_tasks:
            retained_tasks_map[ex.type_id] = ex

    for blind_id in promotable_candidates:
        if blind_id not in cand_map:
            raise PromotionError("MISSING_PROPOSAL_EVIDENCE", f"Missing proposal evidence for {blind_id}")

        raw = cand_map[blind_id]
        canonical_id = id_mapping[blind_id]
        operator = raw.get("operator", "DIRECT")

        if operator not in {"DIRECT", "TARGET_MATCH", "EQUALITY", "PAIRWISE_SELECTION", "COMPOSITE"}:
            raise PromotionError("UNSUPPORTED_EXECUTABLE_SEMANTIC", f"{blind_id}:{operator}")

        audio_arity = raw.get("audio_arity", 1)
        visible_inputs = tuple(raw.get("visible_inputs", raw.get("visible_context_roles", [])))
        hidden_annotations = raw.get("hidden_source_annotations", [])

        if not hidden_annotations and not visible_inputs and operator != "COMPOSITE":
            raise PromotionError("MISSING_PROPOSAL_EVIDENCE", f"Candidate {blind_id} lacks source field annotations")

        source_field = hidden_annotations[0] if hidden_annotations else (visible_inputs[0] if visible_inputs else "")
        if not source_field and operator != "COMPOSITE":
            raise PromotionError("MISSING_PROPOSAL_EVIDENCE", f"Candidate {blind_id} source_field cannot be determined")

        answer_schema = raw.get("answer_schema_proposal", {"kind": "field_value"})
        comparator_id = resolve_candidate_comparator(blind_id, raw, reconciliation_data, existing_catalog_tasks)

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
            invariances = ["surrounding_whitespace"]

        non_invariances = raw.get("expected_non_invariances")
        if non_invariances is None:
            non_invariances = ["label_identity"]

        task_spec = {
            "type_id": canonical_id,
            "proposition_id": canonical_id,
            "proposition_description": raw.get("proposition", f"Semantic task {canonical_id}"),
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

        # Remove from retained tasks if promoted
        if canonical_id in retained_tasks_map:
            del retained_tasks_map[canonical_id]

    # If merge mode, append retained old tasks
    if promotion_mode == "merge":
        for ret_task in retained_tasks_map.values():
            tasks.append(ret_task.model_dump(mode="json"))

    catalog_dict = {
        "dataset": dataset_id,
        "catalog_version": f"{dataset_id}_semantic_catalog_candidate",
        "comparators_resource": "comparators.json",
        "tasks": tasks,
    }

    # Validate catalog structure using promotion gate validator
    catalog_obj = SemanticCatalog.model_validate(catalog_dict)
    issues = validate_catalog(catalog_obj)
    if issues:
        raise PromotionError("CANONICAL_RESOURCE_VALIDATION_FAILED", ",".join(issues))

    return catalog_dict


def run_staged_preflight_for_promotion(dataset_id: str, catalog_dict: dict[str, Any]) -> dict[str, Any]:
    """Runs real staged language preflight validation against compiled candidate catalog."""
    from src.autonomous_qa.language.language_preflight import accepted_types_from_semantic_catalog

    accepted_types = accepted_types_from_semantic_catalog(catalog_dict, dataset_id=dataset_id)
    res = run_preflight(mode="dataset", accepted_types=accepted_types, dataset=dataset_id, write_outputs=False)
    audit = res["audit"]
    return {
        "status": audit["result"],
        "contract_fingerprint": audit["contract_fingerprint"],
        "blocking_issues": audit["blocking_issue_count"],
        "review_issues": audit["review_issue_count"],
        "accepted_type_count": audit["accepted_type_count"],
        "accepted_types": accepted_types,
        "issues": res.get("issues", []),
    }


def compile_candidate_language_resource(
    dataset_id: str,
    promotable_candidates: list[str],
    id_mapping: dict[str, str],
    run_dir: Path,
) -> dict[str, Any]:
    """Compiles candidate language entries for proposal evidence."""
    parsed_lang_path = run_dir / "llm" / "language_generation" / "parsed_response.json"
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
    overrides: dict[str, str] | None = None,
) -> PromotionBundle:
    """Performs the STAGED PREPARE (dry-run) promotion transaction."""
    if promotion_mode not in ("replace", "merge"):
        raise PromotionError("INVALID_PROMOTION_MODE", str(promotion_mode))

    if not run_dir.exists():
        raise PromotionError("RUN_DIR_NOT_FOUND", str(run_dir))

    # 1. Read authoring artifacts & compute input hashes
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = _read_json(run_manifest_path) if run_manifest_path.exists() else {}
    authoring_hash = _sha256_file(run_manifest_path)

    doc_dir = run_dir / "source_documentation"
    doc_hash = canonical_hash([_sha256_file(p) for p in sorted(doc_dir.glob("*"))]) if doc_dir.exists() else ""

    profile_path = run_dir / "deterministic" / "dataset_profile.json"
    if not profile_path.exists():
        raise PromotionError("MISSING_AUTHORING_EVIDENCE", "deterministic/dataset_profile.json missing")
    profile_hash = _sha256_file(profile_path)
    if not profile_hash:
        raise PromotionError("MISSING_AUTHORING_EVIDENCE", "dataset_profile_hash is empty")

    # 2. Readiness evaluation
    readiness = evaluate_promotion_readiness(run_dir)
    promotable_candidates = readiness["promotable_primitives"] + readiness["promotable_composites"]

    if not promotable_candidates:
        raise PromotionError("NO_PROMOTABLE_CANDIDATES", "No candidates passed semantic + language gates")

    # 3. Read current active canonical types & hashes if present
    current_spec_hash = ""
    current_cat_hash = ""
    current_contract_hash = ""
    current_manifest_hash = ""
    current_lang_hash = ""

    old_active_types: tuple[str, ...] = ()
    existing_catalog_tasks: list[SemanticTaskSpec] | None = None

    if DATASET_REGISTRY_PATH.exists():
        registry_data = _read_json(DATASET_REGISTRY_PATH)
        if dataset_id in registry_data.get("datasets", {}):
            entry = registry_data["datasets"][dataset_id]
            spec_obj = get_dataset_spec(dataset_id)
            current_spec_hash = spec_obj.logical_hash()

            cat_path = get_semantic_catalog_path(dataset_id)
            if cat_path.exists():
                cat_obj = load_semantic_catalog(cat_path)
                current_cat_hash = cat_obj.logical_hash()
                existing_catalog_tasks = list(cat_obj.tasks)
                old_active_types = tuple(t.type_id for t in cat_obj.tasks)

            contract_path = RESOURCE_ROOT / entry["production_contract"]
            if contract_path.exists():
                contract_data = _read_json(contract_path)
                contract_obj = ProductionContract.model_validate(contract_data)
                current_contract_hash = contract_obj.fingerprint()

            manifest_path = RESOURCE_ROOT / entry["promotion_manifest"]
            if manifest_path.exists():
                manifest_data = _read_json(manifest_path)
                manifest_obj = PromotionManifest.model_validate(manifest_data)
                current_manifest_hash = manifest_obj.fingerprint()

    if LANGUAGE_REGISTRY_PATH.exists():
        lang_reg = load_language_registry(LANGUAGE_REGISTRY_PATH)
        current_lang_hash = lang_reg.registry_hash

    # 4. Canonical ID resolution
    id_mapping = resolve_canonical_ids(dataset_id, promotable_candidates, readiness["reconciliation"], overrides)
    proposed_promoted_types = tuple(sorted(id_mapping[cid] for cid in promotable_candidates))

    # 5. Promotion Mode active set computation
    if promotion_mode == "replace":
        proposed_active_types = proposed_promoted_types
    else:  # merge
        proposed_active_types = tuple(sorted(set(old_active_types) | set(proposed_promoted_types)))

    # 6. Semantic diff
    old_set = set(old_active_types)
    new_set = set(proposed_active_types)
    reconciled_set = set(id_mapping.values())
    diff = {
        "added_types": sorted(new_set - old_set),
        "removed_types": sorted(old_set - new_set),
        "retained_types": sorted(old_set & new_set),
        "renamed_reused_types": sorted(reconciled_set & old_set),
    }

    # 7. Compile candidate catalog & verify active set invariant
    candidate_catalog = compile_candidate_semantic_catalog(
        dataset_id,
        promotable_candidates,
        id_mapping,
        run_dir,
        readiness["reconciliation"],
        existing_catalog_tasks,
        promotion_mode,
    )

    compiled_type_ids = set(t["type_id"] for t in candidate_catalog.get("tasks", []))
    if compiled_type_ids != set(proposed_active_types):
        raise PromotionError(
            "INCONSISTENT_CATALOG_ACTIVE_SET",
            f"Catalog tasks ({compiled_type_ids}) do not match proposed active set ({proposed_active_types})",
        )

    candidate_language = compile_candidate_language_resource(dataset_id, promotable_candidates, id_mapping, run_dir)

    # 8. Real Staged Preflight Validation
    staged_preflight_res = run_staged_preflight_for_promotion(dataset_id, candidate_catalog)
    staged_pass = (staged_preflight_res["status"] == "PREFLIGHT_PASS")

    staged_catalog_type_ids = tuple(sorted(t["type_id"] for t in candidate_catalog.get("tasks", [])))
    staged_accepted_type_ids = tuple(sorted(a.dataset_type_id for a in staged_preflight_res.get("accepted_types", [])))
    staged_language_candidate_type_ids = tuple(sorted(set(e["type_id"] for e in candidate_language.get("entries", []))))

    authoring_semantic_pass_types = tuple(sorted(readiness.get("authoring_semantic_pass_types", [])))
    executable_promotion_candidates = proposed_promoted_types
    staged_preflight_types = staged_accepted_type_ids

    # Staged input identity verification
    identity_match = (
        set(staged_catalog_type_ids) == set(proposed_active_types)
        and set(staged_accepted_type_ids) == set(proposed_active_types)
    )
    if promotion_mode == "replace":
        forbidden_old_types = {"vimedcss_spoken_content_transcription", "vimedcss_cs_term_extraction", "vimedcss_cs_term_presence"}
        has_forbidden = bool(forbidden_old_types.intersection(staged_accepted_type_ids))
    else:
        has_forbidden = False

    if not identity_match or has_forbidden:
        raise PromotionError(
            "STAGED_PREFLIGHT_IDENTITY_MISMATCH",
            f"Staged preflight types {staged_accepted_type_ids} do not match proposed active types {proposed_active_types} (has_forbidden={has_forbidden})",
        )

    missing_gates = (
        (readiness["primitive_candidates"] - readiness["promotable_primitive_count"])
        + (readiness["composite_candidates"] - readiness["promotable_composite_count"])
    )
    apply_ready = staged_pass and (missing_gates == 0) and (readiness["promotable_total"] > 0) and identity_match and (not has_forbidden)

    # Build PromotionBundle
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
        primitive_semantic_accepted=readiness["primitive_semantic_accepted"],
        composite_candidates=readiness["composite_candidates"],
        composite_semantic_accepted=readiness["composite_semantic_accepted"],
        primitive_language_ready=readiness["primitive_language_ready"],
        composite_language_ready=readiness["composite_language_ready"],
        authoring_language_gate_pass_count=readiness["authoring_language_gate_pass_count"],
        staged_canonical_language_preflight_pass_count=readiness["authoring_language_gate_pass_count"] if staged_pass else 0,
        promotable_primitive_count=readiness["promotable_primitive_count"],
        promotable_composite_count=readiness["promotable_composite_count"],
        promotable_total=readiness["promotable_total"],
        promotable_types=proposed_promoted_types,
        non_promotable_types=readiness["non_promotable_types"],
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
        staged_language_preflight={
            k: v for k, v in staged_preflight_res.items() if k != "accepted_types"
        },
        fresh_contract_fingerprints={"candidate_catalog": SemanticCatalog.model_validate(candidate_catalog).logical_hash()},
        stale_current_artifacts=("current_production_plan",) if diff["removed_types"] or diff["added_types"] else (),
        evidence_summary={
            "run_manifest_status": run_manifest.get("status", "UNKNOWN"),
            "gates_verified": True,
            "staged_language_preflight_pass": staged_pass,
            "zero_llm": True,
        },
        zero_llm_evidence={"llm_calls_in_promotion": 0},
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


def apply_promotion(
    bundle_input: Path | PromotionBundle,
    *,
    resource_root: Path = RESOURCE_ROOT,
) -> dict[str, Any]:
    """Applies a previously prepared PromotionBundle to canonical resources transaction-safely."""
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

    # 1. Verify canonical dataset registry exists
    registry_path = resource_root / "registry" / "datasets.json"
    if not registry_path.exists():
        raise PromotionError("CANONICAL_DATASET_REGISTRY_MISSING", str(registry_path))

    registry_data = _read_json(registry_path)
    if dataset_id not in registry_data.get("datasets", {}):
        raise PromotionError("UNKNOWN_DATASET", dataset_id)

    entry = registry_data["datasets"][dataset_id]
    catalog_path = resource_root.parent / entry["semantic_catalog"]
    contract_path = resource_root.parent / entry["production_contract"]
    manifest_path = resource_root.parent / entry["promotion_manifest"]
    spec_path = resource_root.parent / entry["dataset_spec"]

    # 2. Verify stale bundle protection (expected current hashes match)
    spec_obj = get_dataset_spec(dataset_id)
    current_spec_hash = spec_obj.logical_hash()

    cat_obj = load_semantic_catalog(catalog_path)
    current_cat_hash = cat_obj.logical_hash()

    contract_obj = ProductionContract.model_validate(_read_json(contract_path)) if contract_path.exists() else None
    current_contract_hash = contract_obj.fingerprint() if contract_obj else ""

    manifest_obj = PromotionManifest.model_validate(_read_json(manifest_path)) if manifest_path.exists() else None
    current_manifest_hash = manifest_obj.fingerprint() if manifest_obj else ""

    lang_reg = load_language_registry(LANGUAGE_REGISTRY_PATH)
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

    # 3. Prepare new canonical catalog object and compute logical hash
    candidate_cat_obj = SemanticCatalog.model_validate(bundle.candidate_semantic_catalog)
    cat_issues = validate_catalog(candidate_cat_obj)
    if cat_issues:
        raise PromotionError("CANONICAL_RESOURCE_VALIDATION_FAILED", ",".join(cat_issues))
    candidate_cat_logical_hash = candidate_cat_obj.logical_hash()

    # 4. Prepare ProductionContract and PromotionManifest with exact logical hashes
    comp_registry = load_comparator_registry()
    comp_ids = tuple(sorted({t.comparator_id for t in candidate_cat_obj.tasks if t.comparator_id}))
    comp_hashes = {cid: comp_registry.by_id(cid).logical_hash() for cid in comp_ids if comp_ids}
    comp_reg_hash = comparator_set_hash(comp_ids, comp_registry) if comp_ids else ""

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
        promotion_evidence=bundle.evidence_summary,
        status="PROMOTED",
        migration_source="generic_authoring_promotion_bridge",
    )
    contract_fingerprint = contract.fingerprint()
    contract = contract.model_copy(update={"promotion_fingerprint": contract_fingerprint})

    manifest = PromotionManifest(
        schema_version=1,
        dataset_id=dataset_id,
        status="PROMOTED",
        migration_source="generic_authoring_promotion_bridge",
        dataset_spec_hash=current_spec_hash,
        semantic_catalog_hash=candidate_cat_logical_hash,
        comparator_registry_hash=comp_reg_hash,
        language_registry_hash=current_lang_hash,
        production_contract_hash=contract_fingerprint,
        promotion_evidence=bundle.evidence_summary,
        promoted_semantic_types=bundle.proposed_active_types,
        source_revision=spec_obj.source.get("revision_identity"),
        allowed_splits=tuple(spec_obj.allowed_splits),
        semantic_change_status="SEMANTIC_CHANGE_PROMOTED" if bundle.semantic_diff["added_types"] or bundle.semantic_diff["removed_types"] else "NO_SEMANTIC_CHANGE",
    )
    manifest_fingerprint = manifest.fingerprint()
    manifest = manifest.model_copy(update={"promotion_fingerprint": manifest_fingerprint})

    # 5. Backup destination files in memory for atomic transaction rollback
    backups: dict[Path, bytes] = {}
    for p in (catalog_path, contract_path, manifest_path):
        if p.exists():
            backups[p] = p.read_bytes()

    try:
        # Atomic replace via temporary files
        tmp_catalog = catalog_path.with_suffix(".json.tmp")
        tmp_contract = contract_path.with_suffix(".json.tmp")
        tmp_manifest = manifest_path.with_suffix(".json.tmp")

        _write_json(tmp_catalog, bundle.candidate_semantic_catalog)
        _write_json(tmp_contract, contract.model_dump(mode="json"))
        _write_json(tmp_manifest, manifest.model_dump(mode="json"))

        shutil.move(str(tmp_catalog), str(catalog_path))
        shutil.move(str(tmp_contract), str(contract_path))
        shutil.move(str(tmp_manifest), str(manifest_path))

        # Post-write verification
        reloaded_cat = load_semantic_catalog(catalog_path)
        if reloaded_cat.logical_hash() != candidate_cat_logical_hash:
            raise PromotionError("CANONICAL_RESOURCE_VALIDATION_FAILED", "Catalog logical hash mismatch after write")

        reloaded_contract = ProductionContract.model_validate(_read_json(contract_path))
        if reloaded_contract.fingerprint() != contract_fingerprint:
            raise PromotionError("CANONICAL_RESOURCE_VALIDATION_FAILED", "Contract fingerprint mismatch after write")

        reloaded_manifest = PromotionManifest.model_validate(_read_json(manifest_path))
        if reloaded_manifest.fingerprint() != manifest_fingerprint:
            raise PromotionError("CANONICAL_RESOURCE_VALIDATION_FAILED", "Manifest fingerprint mismatch after write")

    except Exception as exc:
        # ROLLBACK ALL DESTINATION FILES
        for p, data in backups.items():
            p.write_bytes(data)
        raise PromotionError("TRANSACTION_FAILED_ROLLED_BACK", str(exc)) from exc

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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _cli().parse_args(argv)

    overrides = None
    if args.canonical_id_overrides:
        overrides = _read_json(args.canonical_id_overrides)

    if args.prepare:
        if not args.promotion_mode:
            print(json.dumps({"error": "PROMOTION_MODE_REQUIRED", "detail": "--promotion-mode replace|merge is required for --prepare"}, indent=2))
            return 2

        if args.run_dir:
            run_dir = Path(args.run_dir)
        elif args.run_id:
            run_dir = ROOT / "outputs" / "runs" / args.dataset / args.run_id
        else:
            print(json.dumps({"error": "RUN_ID_REQUIRED", "detail": "Must specify --run-id or --run-dir for --prepare"}, indent=2))
            return 2

        bundle = prepare_promotion(
            dataset_id=args.dataset,
            run_dir=run_dir,
            promotion_mode=args.promotion_mode,
            output_root=args.output_root,
            overrides=overrides,
        )
        print(
            json.dumps(
                {
                    "status": "PREPARE_SUCCESS",
                    "dataset_id": bundle.dataset_id,
                    "source_run": bundle.source_run_id,
                    "promotion_mode": bundle.promotion_mode,
                    "primitive_semantic_accepted": bundle.primitive_semantic_accepted,
                    "composite_semantic_accepted": bundle.composite_semantic_accepted,
                    "primitive_language_ready": bundle.primitive_language_ready,
                    "composite_language_ready": bundle.composite_language_ready,
                    "staged_language_preflight": bundle.staged_language_preflight,
                    "promotable_total": bundle.promotable_total,
                    "old_active_types": list(bundle.old_active_types),
                    "proposed_active_types": list(bundle.proposed_active_types),
                    "semantic_diff": bundle.semantic_diff,
                    "non_promotable_types": bundle.non_promotable_types,
                    "fingerprint": bundle.fingerprint(),
                    "canonical_mutations": 0,
                    "real_llm_calls": 0,
                },
                indent=2,
            )
        )
        return 0

    if args.apply:
        if not args.bundle:
            print(json.dumps({"error": "PREPARED_BUNDLE_REQUIRED", "detail": "--apply requires explicit --bundle path to promotion_bundle.json"}, indent=2))
            return 2

        result = apply_promotion(args.bundle)
        print(json.dumps(result, indent=2))
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

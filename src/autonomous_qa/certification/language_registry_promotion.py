"""Phase 4.2P — Global language-registry promotion governance.

Promotes the Phase 4.2 certified candidate global language registry into the
canonical registry AND atomically rebinds every affected canonical
ProductionContract / PromotionManifest language hash reference.

PREPARE (--prepare) performs zero canonical mutations. APPLY (--apply) is an
explicit, atomic, rollback-protected transaction. No LLM is ever invoked.

Resource discovery is generic: affected resources are discovered by scanning
``resource_root/production`` for anything carrying ``language_registry_hash``.
No dataset IDs are hardcoded in the transaction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.autonomous_qa.certification.promotion_gate import PromotionError
from src.autonomous_qa.compiler.canonical_resources import (
    RESOURCE_ROOT,
    ProductionContract,
    PromotionManifest,
)
from src.autonomous_qa.language.candidate_registry import (
    CANDIDATE_CAPABILITIES_RESOURCE,
    build_candidate_language_registry,
    certify_candidate_registry,
    load_candidate_capability_resource,
    validate_candidate_capability_resource,
    write_registry_deterministically,
)
from src.autonomous_qa.language.language_preflight import (
    get_dataset_accepted_types,
    run_preflight,
)
from src.autonomous_qa.language.language_quality import (
    ProductionLanguageRegistry,
    load_language_registry,
)
from src.autonomous_qa.language.template_renderer import canonical_hash
from src.common.config import ROOT

DEFAULT_SCRATCH_ROOT = (
    ROOT / "outputs" / "_scratch" / "language_registry_promotion" / "current"
)

LANGUAGE_REGISTRY_REL = "resources/language/production_registry.json"

# Governance constant (Phase 4.2 candidate), not a dataset branch.
EXPECTED_PHASE4_2_CANDIDATE_HASH = (
    "089a010c3c527fdfb4b1154bc9ffe2fadbfba0080cd81f3cdefd4eecc25b65e3"
)

FORBIDDEN_CAPABILITY_TOKENS = (
    "vimedcss",
    "vimd",
    "vietmdd",
    "cs_terms_count",
    "topic_classification",
    "pairwise_topic_same",
    "medical",
)

_CONTRACT_ALLOWED_CHANGES = frozenset({"language_registry_hash", "promotion_fingerprint"})
_MANIFEST_ALLOWED_CHANGES = frozenset(
    {"language_registry_hash", "production_contract_hash", "promotion_fingerprint"}
)


class LanguageRegistryPromotionError(PromotionError):
    """Raised by the language-registry promotion governance transaction."""


def _read_json(path: Path) -> Any:
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


def _default_atomic_replace(src: str, dst: str) -> None:
    os.replace(src, dst)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class BlockingIssueIdentity(BaseModel):
    """Deterministic, stable identity for one blocking preflight issue.

    Deliberately EXCLUDES ``preflight_case_id`` (registry/contract-fingerprint
    dependent) and registry hashes/timestamps. ``detail`` is retained as part of
    identity so distinct regressions with the same code are not collapsed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    issue_code: str
    dataset_type_id: str
    language_entry_id: str
    detail: str


def _blocking_issue_identity_sort_key(identity: BlockingIssueIdentity) -> tuple[str, str, str, str]:
    return (
        identity.issue_code,
        identity.dataset_type_id,
        identity.language_entry_id,
        identity.detail,
    )


def _blocking_issue_identities(
    preflight_result: dict[str, Any],
) -> frozenset[BlockingIssueIdentity]:
    """Set of stable blocking-issue identities (semantic, not case multiplicity)."""
    identities: set[BlockingIssueIdentity] = set()
    for issue in preflight_result.get("issues", []):
        if issue.get("severity") != "BLOCKING":
            continue
        identities.add(
            BlockingIssueIdentity(
                issue_code=str(issue.get("issue_code") or ""),
                dataset_type_id=str(issue.get("dataset_type_id") or ""),
                language_entry_id=str(issue.get("language_entry_id") or ""),
                detail=str(issue.get("detail") or ""),
            )
        )
    return frozenset(identities)


class DatasetCertificationResult(BaseModel):
    """Generic per-dataset certification result. No dataset-specific fields."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    baseline_status: str
    baseline_blocking: int
    promoted_status: str
    promoted_blocking: int
    new_blocking: int
    require_pass: bool
    passed: bool
    baseline_blocking_issues: tuple[BlockingIssueIdentity, ...] = ()
    promoted_blocking_issues: tuple[BlockingIssueIdentity, ...] = ()
    new_blocking_issues: tuple[BlockingIssueIdentity, ...] = ()


class TargetCertificationResult(BaseModel):
    """Generic result for the explicit target accepted-type set."""

    model_config = ConfigDict(extra="forbid")

    label: str
    status: str
    blocking: int
    passed: bool


class CertificationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_registry_hash: str
    promoted_registry_hash: str
    expected_candidate_hash: str
    expected_candidate_hash_match: bool
    entries_dataset_neutral: bool
    no_duplicate_ids: bool
    computed_hash_valid: bool
    registry_mode_preflight_pass: bool
    deterministic_compilation: bool
    dataset_results: tuple[DatasetCertificationResult, ...] = ()
    target_result: TargetCertificationResult | None = None
    natural_render_evidence: dict[str, list[str]] = Field(default_factory=dict)
    certification_status: Literal["CERTIFIED", "NOT_CERTIFIED"] = "NOT_CERTIFIED"
    failures: tuple[str, ...] = ()

    def logical_hash(self) -> str:
        payload = self.model_dump(mode="json")
        payload.pop("certification_status", None)
        payload.pop("failures", None)
        payload.pop("promoted_registry_hash", None)
        return canonical_hash(payload)


class ResourceFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    kind: Literal["contract", "manifest"]
    dataset_id: str
    old_language_registry_hash: str
    classification: Literal[
        "ALREADY_TARGET", "PINNED_BASE", "PRE_EXISTING_DRIFT", "INVALID_RESOURCE"
    ]
    detail: str = ""


class ContractRebind(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    dataset_id: str
    old_language_registry_hash: str
    new_language_registry_hash: str
    old_promotion_fingerprint: str
    new_promotion_fingerprint: str
    changed_fields: tuple[str, ...]
    new_payload: dict[str, Any]


class ManifestRebind(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    dataset_id: str
    old_language_registry_hash: str
    new_language_registry_hash: str
    old_production_contract_hash: str
    new_production_contract_hash: str
    old_promotion_fingerprint: str
    new_promotion_fingerprint: str
    changed_fields: tuple[str, ...]
    new_payload: dict[str, Any]


class LanguageRegistryPromotionBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    base_registry_hash: str
    candidate_registry_hash: str
    promoted_registry_hash: str
    added_entry_ids: tuple[str, ...]
    modified_entry_ids: tuple[str, ...]
    removed_entry_ids: tuple[str, ...]
    candidate_registry: dict[str, Any]
    promoted_registry: dict[str, Any]
    certification_evidence: CertificationEvidence
    affected_resources: tuple[ResourceFinding, ...]
    contract_rebind_plan: tuple[ContractRebind, ...]
    manifest_rebind_plan: tuple[ManifestRebind, ...]
    expected_current_hashes: dict[str, str]
    prepare_status: str = "PREPARED"
    apply_ready: bool = False

    def fingerprint(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# PREPARE
# ---------------------------------------------------------------------------


def _entry_dump(registry: ProductionLanguageRegistry) -> dict[str, dict[str, Any]]:
    return {e.language_entry_id: e.model_dump(mode="json") for e in registry.entries}


def _discover_production_resources(resource_root: Path) -> list[ResourceFinding]:
    findings: list[ResourceFinding] = []
    production_dir = resource_root / "production"
    if not production_dir.exists():
        return findings
    for path in sorted(production_dir.glob("*.json")):
        rel = path.relative_to(resource_root.parent).as_posix()
        kind: Literal["contract", "manifest"] = (
            "manifest" if path.name.endswith(".promotion.json") else "contract"
        )
        try:
            data = _read_json(path)
            if not isinstance(data, dict) or "language_registry_hash" not in data:
                continue
            if kind == "contract":
                model = ProductionContract.model_validate(data)
            else:
                model = PromotionManifest.model_validate(data)
            findings.append(
                ResourceFinding(
                    path=rel,
                    kind=kind,
                    dataset_id=model.dataset_id,
                    old_language_registry_hash=model.language_registry_hash,
                    classification="PINNED_BASE",  # reclassified below
                )
            )
        except Exception as exc:  # noqa: BLE001
            findings.append(
                ResourceFinding(
                    path=rel,
                    kind=kind,
                    dataset_id="",
                    old_language_registry_hash="",
                    classification="INVALID_RESOURCE",
                    detail=f"{type(exc).__name__}:{exc}",
                )
            )
    return findings


def _classify(
    finding: ResourceFinding, base_hash: str, promoted_hash: str
) -> ResourceFinding:
    if finding.classification == "INVALID_RESOURCE":
        return finding
    pinned = finding.old_language_registry_hash
    if pinned == promoted_hash:
        status = "ALREADY_TARGET"
    elif pinned == base_hash:
        status = "PINNED_BASE"
    else:
        status = "PRE_EXISTING_DRIFT"
    return finding.model_copy(update={"classification": status})


def _genericity_ok(capabilities: list[dict[str, Any]]) -> bool:
    blob = json.dumps(capabilities, ensure_ascii=False).casefold()
    return not any(token in blob for token in FORBIDDEN_CAPABILITY_TOKENS)


def _dataset_delta(
    dataset: str,
    *,
    base_registry: ProductionLanguageRegistry,
    base_registry_path: Path,
    promoted_registry: ProductionLanguageRegistry,
    promoted_registry_path: Path,
    accepted: list[Any],
) -> DatasetCertificationResult:
    """Baseline vs promoted preflight, both resolved through explicit registries.

    The baseline MUST use the supplied base registry (never the global canonical
    registry), so the result is hermetic to ``resource_root``.
    """
    base = run_preflight(
        mode="dataset",
        accepted_types=accepted,
        dataset=dataset,
        write_outputs=False,
        registry=base_registry,
        registry_path=base_registry_path,
    )
    promoted = run_preflight(
        mode="dataset",
        accepted_types=accepted,
        dataset=dataset,
        write_outputs=False,
        registry=promoted_registry,
        registry_path=promoted_registry_path,
    )
    baseline_status = base["audit"]["result"]
    promoted_status = promoted["audit"]["result"]
    baseline_blocking = base["audit"]["blocking_issue_count"]
    promoted_blocking = promoted["audit"]["blocking_issue_count"]

    # Compare STABLE BLOCKING ISSUE IDENTITY, not aggregate counts. A promoted
    # run with fewer blockers can still introduce a novel regression.
    baseline_ids = _blocking_issue_identities(base)
    promoted_ids = _blocking_issue_identities(promoted)
    new_ids = promoted_ids - baseline_ids
    new_blocking = len(new_ids)
    require_pass = baseline_status == "PREFLIGHT_PASS"
    passed = new_blocking == 0 and (
        promoted_status == "PREFLIGHT_PASS" if require_pass else True
    )
    return DatasetCertificationResult(
        dataset_id=dataset,
        baseline_status=baseline_status,
        baseline_blocking=baseline_blocking,
        promoted_status=promoted_status,
        promoted_blocking=promoted_blocking,
        new_blocking=new_blocking,
        require_pass=require_pass,
        passed=passed,
        baseline_blocking_issues=tuple(
            sorted(baseline_ids, key=_blocking_issue_identity_sort_key)
        ),
        promoted_blocking_issues=tuple(
            sorted(promoted_ids, key=_blocking_issue_identity_sort_key)
        ),
        new_blocking_issues=tuple(
            sorted(new_ids, key=_blocking_issue_identity_sort_key)
        ),
    )


def prepare_language_registry_promotion(
    *,
    resource_root: Path = RESOURCE_ROOT,
    candidate_capabilities_path: Path = CANDIDATE_CAPABILITIES_RESOURCE,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    fresh_accepted_types: list[Any] | None = None,
    fresh_dataset_label: str = "fresh_selected",
    expected_candidate_hash: str | None = EXPECTED_PHASE4_2_CANDIDATE_HASH,
    certification_dataset_ids: tuple[str, ...] | None = None,
    dataset_accepted_resolver: Callable[[str], list[Any]] | None = None,
) -> LanguageRegistryPromotionBundle:
    """Build a deterministic language-registry promotion plan. Zero mutations.

    Certification datasets are discovered from the affected canonical
    ProductionContracts / PromotionManifests. No dataset list is hardcoded;
    ``certification_dataset_ids`` exists only as an explicit test/tool override.
    """
    registry_path = resource_root / "language" / "production_registry.json"
    base = load_language_registry(registry_path)

    capability_resource = load_candidate_capability_resource(candidate_capabilities_path)
    capability_entries = validate_candidate_capability_resource(
        capability_resource, expected_language=base.language
    )
    candidate = build_candidate_language_registry(
        base,
        capability_entries,
        capability_language=capability_resource["language"],
    )
    candidate_repeat = build_candidate_language_registry(
        base,
        capability_entries,
        capability_language=capability_resource["language"],
    )
    deterministic_compilation = (
        candidate.registry_hash == candidate_repeat.registry_hash
        and candidate.model_dump(mode="json") == candidate_repeat.model_dump(mode="json")
    )

    added = sorted(set(_entry_dump(candidate)) - set(_entry_dump(base)))
    removed = sorted(set(_entry_dump(base)) - set(_entry_dump(candidate)))
    modified = sorted(
        k for k in set(_entry_dump(base)) & set(_entry_dump(candidate))
        if _entry_dump(base)[k] != _entry_dump(candidate)[k]
    )

    scratch_root.mkdir(parents=True, exist_ok=True)
    candidate_path = scratch_root / "candidate_production_language_registry.json"
    write_registry_deterministically(candidate, candidate_path)

    # -- generic affected-resource discovery (drives certification) ----------
    raw_findings = _discover_production_resources(resource_root)
    discovered_dataset_ids = sorted(
        {
            f.dataset_id
            for f in raw_findings
            if f.classification != "INVALID_RESOURCE" and f.dataset_id
        }
    )
    dataset_ids = (
        tuple(certification_dataset_ids)
        if certification_dataset_ids is not None
        else tuple(discovered_dataset_ids)
    )

    # -- checks run against the candidate (pre-certification) ----------------
    failures: list[str] = []
    if expected_candidate_hash is not None and candidate.registry_hash != expected_candidate_hash:
        failures.append("CANDIDATE_HASH_MISMATCH")
    if not deterministic_compilation:
        failures.append("NONDETERMINISTIC_COMPILATION")
    if not _genericity_ok(capability_entries):
        failures.append("CANDIDATE_NOT_DATASET_NEUTRAL")
    if candidate.registry_hash != candidate.computed_hash():
        failures.append("CANDIDATE_COMPUTED_HASH_INVALID")

    registry_mode = run_preflight(
        mode="registry",
        write_outputs=False,
        registry=candidate,
        registry_path=candidate_path,
    )
    registry_mode_pass = registry_mode["audit"]["result"] == "PREFLIGHT_PASS"
    if not registry_mode_pass:
        failures.append("REGISTRY_MODE_PREFLIGHT_FAIL")

    resolver = dataset_accepted_resolver or get_dataset_accepted_types

    def _resolve_accepted(dataset_id: str) -> tuple[list[Any] | None, str | None]:
        try:
            return resolver(dataset_id), None
        except Exception:  # noqa: BLE001
            return None, f"CERTIFICATION_ACCEPTED_TYPES_UNAVAILABLE:{dataset_id}"

    dataset_results: list[DatasetCertificationResult] = []
    for ds in dataset_ids:
        accepted, error = _resolve_accepted(ds)
        if error is not None:
            failures.append(error)
            dataset_results.append(
                DatasetCertificationResult(
                    dataset_id=ds,
                    baseline_status="ACCEPTED_TYPES_UNAVAILABLE",
                    baseline_blocking=-1,
                    promoted_status="ACCEPTED_TYPES_UNAVAILABLE",
                    promoted_blocking=-1,
                    new_blocking=0,
                    require_pass=True,
                    passed=False,
                )
            )
            continue
        result = _dataset_delta(
            ds,
            base_registry=base,
            base_registry_path=registry_path,
            promoted_registry=candidate,
            promoted_registry_path=candidate_path,
            accepted=accepted,
        )
        dataset_results.append(result)
        if not result.passed:
            failures.append(f"CERTIFICATION_DATASET_FAILED:{ds}")

    target_result: TargetCertificationResult | None = None
    if fresh_accepted_types is None:
        failures.append("TARGET_ACCEPTED_TYPES_NOT_PROVIDED")
    else:
        fresh = run_preflight(
            mode="dataset",
            accepted_types=fresh_accepted_types,
            dataset=fresh_dataset_label,
            write_outputs=False,
            registry=candidate,
            registry_path=candidate_path,
        )
        target_result = TargetCertificationResult(
            label=fresh_dataset_label,
            status=fresh["audit"]["result"],
            blocking=fresh["audit"]["blocking_issue_count"],
            passed=(
                fresh["audit"]["result"] == "PREFLIGHT_PASS"
                and fresh["audit"]["blocking_issue_count"] == 0
            ),
        )
        if not target_result.passed:
            failures.append("TARGET_PREFLIGHT_FAIL")

    # -- natural render evidence for every added capability ------------------
    natural_render: dict[str, list[str]] = {}
    for row in registry_mode.get("render_matrix", []):
        eid = row["language_entry_id"]
        if eid in added:
            natural_render.setdefault(eid, [])
            if row["rendered_question"] not in natural_render[eid]:
                natural_render[eid].append(row["rendered_question"])
    if any(not natural_render.get(eid) for eid in added):
        failures.append("NATURAL_RENDER_EVIDENCE_MISSING")

    preliminary = CertificationEvidence(
        candidate_registry_hash=candidate.registry_hash,
        promoted_registry_hash="",
        expected_candidate_hash=expected_candidate_hash or "",
        expected_candidate_hash_match=(
            expected_candidate_hash is None
            or candidate.registry_hash == expected_candidate_hash
        ),
        entries_dataset_neutral=_genericity_ok(capability_entries),
        no_duplicate_ids=len({e.language_entry_id for e in candidate.entries}) == len(candidate.entries),
        computed_hash_valid=candidate.registry_hash == candidate.computed_hash(),
        registry_mode_preflight_pass=registry_mode_pass,
        deterministic_compilation=deterministic_compilation,
        dataset_results=tuple(dataset_results),
        target_result=target_result,
        natural_render_evidence=natural_render,
    )
    certified = not failures
    if certified:
        certification_hash = preliminary.logical_hash()
        promoted = certify_candidate_registry(
            candidate, certification_hash=certification_hash
        )
    else:
        promoted = candidate

    promoted_path = scratch_root / "candidate_production_language_registry.json"
    write_registry_deterministically(promoted, promoted_path)

    # Section 16: prove the PROMOTED (certified) registry behaves identically
    # to the candidate under registry-mode, cross-dataset and target preflight.
    registry_mode_promoted = run_preflight(
        mode="registry",
        write_outputs=False,
        registry=promoted,
        registry_path=promoted_path,
    )
    if (
        registry_mode_promoted["audit"]["result"] == "PREFLIGHT_PASS"
    ) != registry_mode_pass:
        failures.append("PROMOTED_REGISTRY_PREFLIGHT_MISMATCH")

    dataset_results_promoted: list[DatasetCertificationResult] = []
    for ds, previous in zip(dataset_ids, dataset_results):
        accepted, error = _resolve_accepted(ds)
        if error is not None:
            dataset_results_promoted.append(previous)
            continue
        promoted_result = _dataset_delta(
            ds,
            base_registry=base,
            base_registry_path=registry_path,
            promoted_registry=promoted,
            promoted_registry_path=promoted_path,
            accepted=accepted,
        )
        dataset_results_promoted.append(promoted_result)
        if (
            promoted_result.passed != previous.passed
            or promoted_result.new_blocking != previous.new_blocking
            or promoted_result.new_blocking_issues != previous.new_blocking_issues
            or promoted_result.promoted_status != previous.promoted_status
        ):
            failures.append(f"PROMOTED_CERTIFICATION_MISMATCH:{ds}")

    if fresh_accepted_types is not None:
        fresh_promoted = run_preflight(
            mode="dataset",
            accepted_types=fresh_accepted_types,
            dataset=fresh_dataset_label,
            write_outputs=False,
            registry=promoted,
            registry_path=promoted_path,
        )
        target_result = TargetCertificationResult(
            label=fresh_dataset_label,
            status=fresh_promoted["audit"]["result"],
            blocking=fresh_promoted["audit"]["blocking_issue_count"],
            passed=(
                fresh_promoted["audit"]["result"] == "PREFLIGHT_PASS"
                and fresh_promoted["audit"]["blocking_issue_count"] == 0
            ),
        )
        if not target_result.passed:
            failures.append("PROMOTED_TARGET_PREFLIGHT_FAIL")

    certified = certified and not any(
        f.startswith("PROMOTED_") for f in failures
    )

    evidence = preliminary.model_copy(
        update={
            "promoted_registry_hash": promoted.registry_hash,
            "dataset_results": tuple(dataset_results_promoted),
            "target_result": target_result,
            "certification_status": "CERTIFIED" if certified else "NOT_CERTIFIED",
            "failures": tuple(failures),
        }
    )

    # -- classification ------------------------------------------------------
    findings = [
        _classify(f, base.registry_hash, promoted.registry_hash)
        for f in raw_findings
    ]

    # -- rebind plans --------------------------------------------------------
    contract_plan: list[ContractRebind] = []
    new_contract_fp: dict[str, str] = {}
    for finding in findings:
        if finding.kind != "contract" or finding.classification == "INVALID_RESOURCE":
            continue
        path = resource_root.parent / finding.path
        contract = ProductionContract.model_validate(_read_json(path))
        updated = contract.model_copy(
            update={"language_registry_hash": promoted.registry_hash}
        )
        new_fp = updated.fingerprint()
        updated = updated.model_copy(update={"promotion_fingerprint": new_fp})
        old_payload = contract.model_dump(mode="json")
        new_payload = updated.model_dump(mode="json")
        changed = tuple(sorted(k for k in new_payload if old_payload.get(k) != new_payload.get(k)))
        if not set(changed) <= _CONTRACT_ALLOWED_CHANGES:
            findings = [
                f.model_copy(update={"classification": "INVALID_RESOURCE", "detail": f"disallowed_contract_changes:{changed}"})
                if f.path == finding.path else f
                for f in findings
            ]
            continue
        new_contract_fp[finding.dataset_id] = new_fp
        contract_plan.append(
            ContractRebind(
                path=finding.path,
                dataset_id=finding.dataset_id,
                old_language_registry_hash=contract.language_registry_hash,
                new_language_registry_hash=promoted.registry_hash,
                old_promotion_fingerprint=contract.promotion_fingerprint,
                new_promotion_fingerprint=new_fp,
                changed_fields=changed,
                new_payload=new_payload,
            )
        )

    manifest_plan: list[ManifestRebind] = []
    for finding in findings:
        if finding.kind != "manifest" or finding.classification == "INVALID_RESOURCE":
            continue
        path = resource_root.parent / finding.path
        manifest = PromotionManifest.model_validate(_read_json(path))
        new_contract = new_contract_fp.get(finding.dataset_id)
        if new_contract is None:
            findings = [
                f.model_copy(update={"classification": "INVALID_RESOURCE", "detail": "no_contract_for_manifest"})
                if f.path == finding.path else f
                for f in findings
            ]
            continue
        updated = manifest.model_copy(
            update={
                "language_registry_hash": promoted.registry_hash,
                "production_contract_hash": new_contract,
            }
        )
        new_fp = updated.fingerprint()
        updated = updated.model_copy(update={"promotion_fingerprint": new_fp})
        old_payload = manifest.model_dump(mode="json")
        new_payload = updated.model_dump(mode="json")
        changed = tuple(sorted(k for k in new_payload if old_payload.get(k) != new_payload.get(k)))
        if not set(changed) <= _MANIFEST_ALLOWED_CHANGES:
            findings = [
                f.model_copy(update={"classification": "INVALID_RESOURCE", "detail": f"disallowed_manifest_changes:{changed}"})
                if f.path == finding.path else f
                for f in findings
            ]
            continue
        manifest_plan.append(
            ManifestRebind(
                path=finding.path,
                dataset_id=finding.dataset_id,
                old_language_registry_hash=manifest.language_registry_hash,
                new_language_registry_hash=promoted.registry_hash,
                old_production_contract_hash=manifest.production_contract_hash,
                new_production_contract_hash=new_contract,
                old_promotion_fingerprint=manifest.promotion_fingerprint,
                new_promotion_fingerprint=new_fp,
                changed_fields=changed,
                new_payload=new_payload,
            )
        )

    # -- expected current hashes (stale protection) --------------------------
    expected: dict[str, str] = {LANGUAGE_REGISTRY_REL: _sha256_file(registry_path)}
    try:
        cap_rel = candidate_capabilities_path.relative_to(resource_root.parent).as_posix()
        expected[cap_rel] = _sha256_file(candidate_capabilities_path)
    except ValueError:
        pass
    for finding in findings:
        if finding.classification == "INVALID_RESOURCE":
            continue
        expected[finding.path] = _sha256_file(resource_root.parent / finding.path)
    field_specs_dir = resource_root / "field_specs"
    if field_specs_dir.exists():
        for path in sorted(field_specs_dir.glob("*.json")):
            expected[path.relative_to(resource_root.parent).as_posix()] = _sha256_file(path)

    no_invalid = all(f.classification != "INVALID_RESOURCE" for f in findings)
    all_contracts_planned = len(contract_plan) == sum(
        1 for f in findings if f.kind == "contract" and f.classification != "INVALID_RESOURCE"
    )
    all_manifests_planned = len(manifest_plan) == sum(
        1 for f in findings if f.kind == "manifest" and f.classification != "INVALID_RESOURCE"
    )
    apply_ready = bool(
        certified and no_invalid and all_contracts_planned and all_manifests_planned
    )

    bundle = LanguageRegistryPromotionBundle(
        schema_version=1,
        base_registry_hash=base.registry_hash,
        candidate_registry_hash=candidate.registry_hash,
        promoted_registry_hash=promoted.registry_hash,
        added_entry_ids=tuple(added),
        modified_entry_ids=tuple(modified),
        removed_entry_ids=tuple(removed),
        candidate_registry=candidate.model_dump(mode="json"),
        promoted_registry=promoted.model_dump(mode="json"),
        certification_evidence=evidence,
        affected_resources=tuple(findings),
        contract_rebind_plan=tuple(contract_plan),
        manifest_rebind_plan=tuple(manifest_plan),
        expected_current_hashes=expected,
        prepare_status="PREPARED",
        apply_ready=apply_ready,
    )

    _write_json(scratch_root / "language_promotion_bundle.json", bundle.model_dump(mode="json"))
    _write_json(scratch_root / "candidate_production_language_registry.json", promoted.model_dump(mode="json"))
    _write_json(
        scratch_root / "contract_rebind_plan.json",
        [p.model_dump(mode="json") for p in contract_plan],
    )
    _write_json(
        scratch_root / "manifest_rebind_plan.json",
        [p.model_dump(mode="json") for p in manifest_plan],
    )
    _write_json(
        scratch_root / "cross_dataset_certification.json",
        {
            "candidate_registry_hash": candidate.registry_hash,
            "promoted_registry_hash": promoted.registry_hash,
            "certification_status": evidence.certification_status,
            "failures": list(evidence.failures),
            "certification_dataset_ids": list(dataset_ids),
            "dataset_results": [
                r.model_dump(mode="json") for r in evidence.dataset_results
            ],
            "target_result": (
                evidence.target_result.model_dump(mode="json")
                if evidence.target_result
                else None
            ),
            "natural_render_evidence": natural_render,
        },
    )
    return bundle


# ---------------------------------------------------------------------------
# APPLY
# ---------------------------------------------------------------------------


def _assert_within_root(path: Path, resource_root: Path) -> None:
    try:
        path.resolve().relative_to(resource_root.resolve())
    except ValueError:
        raise LanguageRegistryPromotionError(
            "LANGUAGE_PROMOTION_DESTINATION_OUTSIDE_RESOURCE_ROOT", str(path)
        )


def apply_language_registry_promotion(
    bundle_input: Path | LanguageRegistryPromotionBundle,
    *,
    resource_root: Path = RESOURCE_ROOT,
    replace_fn: Callable[[str, str], Any] = _default_atomic_replace,
    validation_hook: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Atomically promote the language registry and rebind affected resources."""
    if isinstance(bundle_input, Path):
        bundle = LanguageRegistryPromotionBundle.model_validate(_read_json(bundle_input))
    else:
        bundle = bundle_input

    if not bundle.apply_ready:
        raise LanguageRegistryPromotionError(
            "LANGUAGE_PROMOTION_NOT_APPLY_READY",
            f"certification_status={bundle.certification_evidence.certification_status} "
            f"failures={list(bundle.certification_evidence.failures)}",
        )

    # 1. stale protection
    for rel, expected_sha in bundle.expected_current_hashes.items():
        current = _sha256_file(resource_root.parent / rel)
        if current != expected_sha:
            raise LanguageRegistryPromotionError(
                "LANGUAGE_PROMOTION_BUNDLE_STALE", f"{rel}: {current} != {expected_sha}"
            )

    registry_path = resource_root / "language" / "production_registry.json"
    destinations: list[tuple[Path, Any]] = [
        (registry_path, bundle.promoted_registry)
    ]
    for plan in bundle.contract_rebind_plan:
        destinations.append((resource_root.parent / plan.path, plan.new_payload))
    for plan in bundle.manifest_rebind_plan:
        destinations.append((resource_root.parent / plan.path, plan.new_payload))
    for dest, _ in destinations:
        _assert_within_root(dest, resource_root)

    backups: dict[Path, bytes] = {}
    newly_created: list[Path] = []
    for dest, _ in destinations:
        if dest.exists():
            backups[dest] = dest.read_bytes()
        else:
            newly_created.append(dest)

    txn = uuid.uuid4().hex
    temps = {
        dest: dest.parent / f".{dest.name}.{txn}.tmp" for dest, _ in destinations
    }

    try:
        for dest, payload in destinations:
            _write_json(temps[dest], payload)
        for dest, _ in destinations:
            replace_fn(str(temps[dest]), str(dest))

        if validation_hook is not None:
            validation_hook()

        # 2. cross-resource final validation
        reloaded = load_language_registry(registry_path)
        if reloaded.registry_hash != bundle.promoted_registry_hash:
            raise LanguageRegistryPromotionError(
                "LANGUAGE_PROMOTION_FINAL_VALIDATION_FAILED", "registry hash mismatch"
            )
        reloaded_contracts: dict[str, ProductionContract] = {}
        for plan in bundle.contract_rebind_plan:
            contract = ProductionContract.model_validate(_read_json(resource_root.parent / plan.path))
            if contract.language_registry_hash != reloaded.registry_hash:
                raise LanguageRegistryPromotionError(
                    "LANGUAGE_PROMOTION_FINAL_VALIDATION_FAILED", f"contract registry hash {plan.path}"
                )
            if contract.promotion_fingerprint != contract.fingerprint():
                raise LanguageRegistryPromotionError(
                    "LANGUAGE_PROMOTION_FINAL_VALIDATION_FAILED", f"contract fingerprint {plan.path}"
                )
            reloaded_contracts[plan.dataset_id] = contract
        for plan in bundle.manifest_rebind_plan:
            manifest = PromotionManifest.model_validate(_read_json(resource_root.parent / plan.path))
            if manifest.language_registry_hash != reloaded.registry_hash:
                raise LanguageRegistryPromotionError(
                    "LANGUAGE_PROMOTION_FINAL_VALIDATION_FAILED", f"manifest registry hash {plan.path}"
                )
            contract = reloaded_contracts.get(plan.dataset_id)
            if contract is None or manifest.production_contract_hash != contract.fingerprint():
                raise LanguageRegistryPromotionError(
                    "LANGUAGE_PROMOTION_FINAL_VALIDATION_FAILED", f"manifest contract hash {plan.path}"
                )
            if manifest.promotion_fingerprint != manifest.fingerprint():
                raise LanguageRegistryPromotionError(
                    "LANGUAGE_PROMOTION_FINAL_VALIDATION_FAILED", f"manifest fingerprint {plan.path}"
                )
    except Exception as exc:
        try:
            for dest, data in backups.items():
                dest.write_bytes(data)
            for dest in newly_created:
                if dest.exists():
                    dest.unlink()
            for temp in temps.values():
                if temp.exists():
                    temp.unlink()
        except Exception as rb_exc:  # noqa: BLE001
            raise LanguageRegistryPromotionError(
                "LANGUAGE_PROMOTION_ROLLBACK_FAILED", str(rb_exc)
            ) from rb_exc
        if isinstance(exc, LanguageRegistryPromotionError):
            raise LanguageRegistryPromotionError(
                "LANGUAGE_PROMOTION_TRANSACTION_FAILED_ROLLED_BACK", str(exc)
            ) from exc
        raise LanguageRegistryPromotionError(
            "LANGUAGE_PROMOTION_TRANSACTION_FAILED_ROLLED_BACK", str(exc)
        ) from exc

    return {
        "status": "LANGUAGE_REGISTRY_PROMOTED",
        "base_registry_hash": bundle.base_registry_hash,
        "candidate_registry_hash": bundle.candidate_registry_hash,
        "promoted_registry_hash": bundle.promoted_registry_hash,
        "rebound_contracts": [p.path for p in bundle.contract_rebind_plan],
        "rebound_manifests": [p.path for p in bundle.manifest_rebind_plan],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _compute_fresh_accepted_types(
    dataset_id: str,
    run_dir: Path,
    *,
    resource_root: Path,
    scratch_root: Path,
) -> list[Any]:
    """Resolve the fresh selected accepted types from an authoring run."""
    from src.autonomous_qa.certification.authoring_promotion import prepare_promotion

    bundle = prepare_promotion(
        dataset_id,
        run_dir,
        promotion_mode="replace",
        output_root=scratch_root / "authoring_prepare",
        resource_root=resource_root,
        language_capability_root=scratch_root / "authoring_cap",
    )
    return list(bundle.candidate_language_preflight["accepted_types"])


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--resource-root", type=Path, default=RESOURCE_ROOT)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    parser.add_argument("--fresh-dataset", default=None)
    parser.add_argument("--fresh-run-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _cli().parse_args(argv)
    if args.prepare:
        fresh = None
        if args.fresh_dataset and args.fresh_run_dir:
            fresh = _compute_fresh_accepted_types(
                args.fresh_dataset,
                args.fresh_run_dir,
                resource_root=args.resource_root,
                scratch_root=args.scratch_root,
            )
        try:
            bundle = prepare_language_registry_promotion(
                resource_root=args.resource_root,
                scratch_root=args.scratch_root,
                fresh_accepted_types=fresh,
                fresh_dataset_label=args.fresh_dataset or "fresh_selected",
            )
        except PromotionError as exc:
            print(json.dumps({"error": exc.code, "detail": exc.detail}, indent=2))
            return 1
        print(json.dumps({
            "status": bundle.prepare_status,
            "apply_ready": bundle.apply_ready,
            "base_registry_hash": bundle.base_registry_hash,
            "candidate_registry_hash": bundle.candidate_registry_hash,
            "promoted_registry_hash": bundle.promoted_registry_hash,
            "added_entry_ids": list(bundle.added_entry_ids),
            "certification_status": bundle.certification_evidence.certification_status,
            "failures": list(bundle.certification_evidence.failures),
            "affected_contracts": [f.path for f in bundle.affected_resources if f.kind == "contract"],
            "affected_manifests": [f.path for f in bundle.affected_resources if f.kind == "manifest"],
            "canonical_mutations": 0,
        }, indent=2))
        return 0
    if args.apply:
        if not args.bundle:
            print(json.dumps({"error": "BUNDLE_REQUIRED"}, indent=2))
            return 2
        try:
            result = apply_language_registry_promotion(
                args.bundle, resource_root=args.resource_root
            )
        except PromotionError as exc:
            print(json.dumps({"error": exc.code, "detail": exc.detail}, indent=2))
            return 1
        print(json.dumps(result, indent=2))
        return 0
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

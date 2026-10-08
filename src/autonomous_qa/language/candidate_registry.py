"""Phase 4.2 candidate language-registry compiler + Phase 4.2P certification.

Deterministically compiles a CANDIDATE ``ProductionLanguageRegistry`` from the
canonical registry plus dataset-neutral capability entries declared in
``resources/language/candidate_capabilities.json``, then stamps explicit
certification provenance for canonical promotion.

The candidate registry is written ONLY to a scratch destination. It never
overwrites ``resources/language/production_registry.json``; existing canonical
ProductionContracts pin the canonical registry hash and must not be invalidated
by an in-place mutation.

No dataset IDs, task IDs, field names, medical terms, or LLM calls appear here.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.autonomous_qa.compiler.semantic_field_specs import SemanticClass
from src.autonomous_qa.language.language_quality import (
    LanguageRegistryEntry,
    ProductionLanguageRegistry,
    load_language_registry,
)
from src.autonomous_qa.language.template_renderer import canonical_hash

CANDIDATE_CAPABILITIES_RESOURCE = (
    Path(__file__).resolve().parents[3]
    / "resources"
    / "language"
    / "candidate_capabilities.json"
)

CANDIDATE_VERSION_SUFFIX = "+phase4_2_candidate"

SUPPORTED_CAPABILITY_SCHEMA_VERSION = 1

CapabilityOperator = Literal[
    "DIRECT", "EQUALITY", "PAIRWISE_SELECTION", "TARGET_MATCH", "COMPOSITE"
]
CapabilityUnitPolicy = Literal["required", "forbidden", "any"]
CapabilityOwner = Literal["slot", "literal", "none"]
CapabilityMatchPolicy = Literal[
    "exact", "bucket", "tolerance", "set_exact", "set_overlap"
]
CapabilityAnswerKind = Literal[
    "field_value", "boolean", "audio_index", "structured"
]

_SLOT_TOKEN_RE = re.compile(r"^\[[A-Z][A-Z0-9_]*\]$")


class CandidateRegistryError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}:{detail}" if detail else code)


class CandidateCapabilityEntry(BaseModel):
    """Strict schema for one dataset-neutral language capability entry."""

    model_config = ConfigDict(extra="forbid")

    capability_id: str = Field(min_length=1)
    operator: CapabilityOperator
    semantic_class: SemanticClass
    answer_kind: CapabilityAnswerKind
    pattern: str = Field(min_length=1)
    required_slots: list[str]
    optional_slots: list[str] = Field(default_factory=list)
    unit_policy: CapabilityUnitPolicy
    match_policy: list[CapabilityMatchPolicy]
    entity_scopes: list[str]
    entity_reference_owner: CapabilityOwner


class CandidateCapabilityResource(BaseModel):
    """Strict schema for the capability resource contract."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    capability_set_id: str = Field(min_length=1)
    language: str = Field(min_length=1)
    entries: list[CandidateCapabilityEntry]


def validate_candidate_capability_resource(
    data: Any, *, expected_language: str | None = None
) -> list[dict[str, Any]]:
    """Validate the capability resource contract before building entries.

    Fails closed with explicit machine codes. Never relies on KeyError.
    """
    if not isinstance(data, Mapping):
        raise CandidateRegistryError("CANDIDATE_CAPABILITIES_MALFORMED", "not_an_object")

    schema_version = data.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != SUPPORTED_CAPABILITY_SCHEMA_VERSION
    ):
        raise CandidateRegistryError(
            "CANDIDATE_CAPABILITIES_SCHEMA_UNSUPPORTED", str(schema_version)
        )

    try:
        resource = CandidateCapabilityResource.model_validate(data)
    except ValidationError as exc:
        raise CandidateRegistryError(
            "CANDIDATE_CAPABILITIES_MALFORMED", str(exc)
        ) from exc

    if expected_language is not None and resource.language != expected_language:
        raise CandidateRegistryError(
            "LANGUAGE_CAPABILITY_LANGUAGE_MISMATCH",
            f"{resource.language}!={expected_language}",
        )

    ids = [entry.capability_id for entry in resource.entries]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise CandidateRegistryError(
            "DUPLICATE_LANGUAGE_ENTRY_ID", ",".join(duplicates)
        )

    for entry in resource.entries:
        if not entry.entity_scopes:
            raise CandidateRegistryError(
                "CANDIDATE_CAPABILITY_ENTITY_SCOPES_EMPTY", entry.capability_id
            )
        if not entry.match_policy:
            raise CandidateRegistryError(
                "CANDIDATE_CAPABILITY_MATCH_POLICY_EMPTY", entry.capability_id
            )
        for slot in list(entry.required_slots) + list(entry.optional_slots):
            if not _SLOT_TOKEN_RE.match(slot):
                raise CandidateRegistryError(
                    "CANDIDATE_CAPABILITY_MALFORMED_SLOT",
                    f"{entry.capability_id}:{slot}",
                )
        overlap = sorted(set(entry.required_slots) & set(entry.optional_slots))
        if overlap:
            raise CandidateRegistryError(
                "CANDIDATE_CAPABILITY_SLOT_OVERLAP",
                f"{entry.capability_id}:{overlap}",
            )
        for slot in entry.required_slots:
            if slot not in entry.pattern:
                raise CandidateRegistryError(
                    "CANDIDATE_CAPABILITY_REQUIRED_SLOT_ABSENT",
                    f"{entry.capability_id}:{slot}",
                )

    return [entry.model_dump() for entry in resource.entries]


def load_candidate_capabilities(
    path: Path = CANDIDATE_CAPABILITIES_RESOURCE,
    *,
    expected_language: str | None = None,
) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_candidate_capability_resource(data, expected_language=expected_language)


def load_candidate_capability_resource(
    path: Path = CANDIDATE_CAPABILITIES_RESOURCE,
) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _capability_semantic_payload(capability: dict[str, Any]) -> dict[str, Any]:
    """Normalized semantic/capability-bearing fields for one capability.

    Excludes promotion provenance (source_kind / certification_status /
    certification_hash) — those are not part of the semantic capability.
    """
    contract_payload = {
        "operator": capability["operator"],
        "semantic_class": capability["semantic_class"],
        "answer_kind": capability["answer_kind"],
        "pattern": capability["pattern"],
        "required_slots": list(capability.get("required_slots", [])),
        "optional_slots": list(capability.get("optional_slots", [])),
        "match_policy": list(capability.get("match_policy", ["exact"])),
        "unit_policy": capability.get("unit_policy", "any"),
        "entity_scopes": list(capability.get("entity_scopes", [])),
        "entity_reference_owner": capability.get("entity_reference_owner"),
    }
    return {
        **contract_payload,
        "semantic_contract_hash": canonical_hash(contract_payload),
    }


def capability_equivalent_to_registry_entry(
    capability: dict[str, Any],
    existing_entry: LanguageRegistryEntry,
    base: ProductionLanguageRegistry | None = None,
) -> bool:
    """True when the capability's SEMANTIC identity equals an installed entry.

    Compares every capability-bearing field and the semantic_contract_hash.
    Never infers equivalence from the ID alone; promotion provenance is ignored.
    """
    payload = _capability_semantic_payload(capability)
    return (
        existing_entry.operator == payload["operator"]
        and existing_entry.semantic_class == payload["semantic_class"]
        and existing_entry.pattern == payload["pattern"]
        and existing_entry.answer_kind == payload["answer_kind"]
        and list(existing_entry.required_slots) == payload["required_slots"]
        and list(existing_entry.optional_slots) == payload["optional_slots"]
        and existing_entry.unit_policy == payload["unit_policy"]
        and list(existing_entry.match_policy) == payload["match_policy"]
        and list(existing_entry.entity_scopes or []) == payload["entity_scopes"]
        and existing_entry.entity_reference_owner == payload["entity_reference_owner"]
        and existing_entry.semantic_contract_hash == payload["semantic_contract_hash"]
    )


def _entry_from_capability(
    capability: dict[str, Any], base: ProductionLanguageRegistry
) -> LanguageRegistryEntry:
    entry_id = capability["capability_id"]
    contract_payload = {
        "operator": capability["operator"],
        "semantic_class": capability["semantic_class"],
        "answer_kind": capability["answer_kind"],
        "pattern": capability["pattern"],
        "required_slots": capability.get("required_slots", []),
        "optional_slots": capability.get("optional_slots", []),
        "match_policy": capability.get("match_policy", ["exact"]),
        "unit_policy": capability.get("unit_policy", "any"),
        "entity_scopes": capability.get("entity_scopes", []),
        "entity_reference_owner": capability.get("entity_reference_owner"),
    }
    return LanguageRegistryEntry(
        language_entry_id=entry_id,
        source_kind="CANDIDATE",
        source_id=entry_id,
        canonical_blueprint_id=entry_id,
        operator=capability["operator"],
        semantic_class=capability["semantic_class"],
        pattern=capability["pattern"],
        answer_kind=capability["answer_kind"],
        required_slots=list(capability.get("required_slots", [])),
        optional_slots=list(capability.get("optional_slots", [])),
        unit_policy=capability.get("unit_policy", "any"),
        match_policy=list(capability.get("match_policy", ["exact"])),
        semantic_contract_hash=canonical_hash(contract_payload),
        quality_status="PRODUCTION_PASS",
        enabled=True,
        template_library_hash=base.template_library_hash,
        paraphrase_library_hash=base.paraphrase_library_hash,
        registry_version=base.version,
        entity_scopes=list(capability.get("entity_scopes", [])),
        entity_reference_owner=capability.get("entity_reference_owner"),
    )


def _candidate_version(base_version: str) -> str:
    """Apply the candidate suffix exactly once (idempotent)."""
    if base_version.endswith(CANDIDATE_VERSION_SUFFIX):
        return base_version
    return base_version + CANDIDATE_VERSION_SUFFIX


def build_candidate_language_registry(
    base: ProductionLanguageRegistry,
    capabilities: list[dict[str, Any]],
    *,
    capability_language: str | None = None,
) -> ProductionLanguageRegistry:
    """Return a deterministic candidate registry = base + generic capabilities.

    Idempotent merge policy:
      * ID absent            -> add the CANDIDATE capability.
      * ID present, equivalent AND certified -> ALREADY_INSTALLED no-op.
      * ID present, semantically different   -> LANGUAGE_CAPABILITY_ID_CONFLICT.
      * ID present, equivalent but non-canonical provenance
        -> LANGUAGE_CAPABILITY_PROVENANCE_INVALID (fail closed).

    If every requested capability is already installed and certified, the base
    registry is returned unchanged (byte-equivalent, no hash churn, no repeated
    version suffix).
    """
    if capability_language is not None and capability_language != base.language:
        raise CandidateRegistryError(
            "LANGUAGE_CAPABILITY_LANGUAGE_MISMATCH",
            f"{capability_language}!={base.language}",
        )
    by_id = {entry.language_entry_id: entry for entry in base.entries}
    new_entries: list[LanguageRegistryEntry] = []
    seen: set[str] = set()
    for capability in capabilities:
        entry_id = capability.get("capability_id")
        if not entry_id:
            raise CandidateRegistryError("CANDIDATE_CAPABILITY_ID_MISSING", str(capability))
        if entry_id in seen:
            raise CandidateRegistryError("DUPLICATE_LANGUAGE_ENTRY_ID", entry_id)
        seen.add(entry_id)

        existing = by_id.get(entry_id)
        if existing is None:
            new_entries.append(_entry_from_capability(capability, base))
            continue

        # ID already present: require semantic equivalence, never ID-only trust.
        if not capability_equivalent_to_registry_entry(capability, existing, base):
            raise CandidateRegistryError("LANGUAGE_CAPABILITY_ID_CONFLICT", entry_id)
        if not (
            existing.source_kind == "CERTIFIED_CAPABILITY"
            and existing.certification_status == "CERTIFIED"
        ):
            raise CandidateRegistryError(
                "LANGUAGE_CAPABILITY_PROVENANCE_INVALID",
                f"{entry_id}:{existing.source_kind}:{existing.certification_status}",
            )
        # equivalent + certified -> already installed, no duplicate added

    if not new_entries:
        # Nothing to add: return the base registry unchanged.
        return base

    merged = sorted(
        list(base.entries) + new_entries,
        key=lambda entry: entry.language_entry_id,
    )
    candidate = ProductionLanguageRegistry(
        language=base.language,
        version=_candidate_version(base.version),
        schema_version=base.schema_version,
        template_library_version=base.template_library_version,
        template_library_hash=base.template_library_hash,
        paraphrase_library_version=base.paraphrase_library_version,
        paraphrase_library_hash=base.paraphrase_library_hash,
        quality_model=base.quality_model,
        quality_call_ids=list(base.quality_call_ids),
        registry_hash="PENDING",
        entries=merged,
    )
    candidate.registry_hash = candidate.computed_hash()
    return candidate


def build_candidate_registry_from_paths(
    canonical_path: Path,
    capabilities_path: Path = CANDIDATE_CAPABILITIES_RESOURCE,
) -> ProductionLanguageRegistry:
    base = load_language_registry(canonical_path)
    resource = load_candidate_capability_resource(capabilities_path)
    entries = validate_candidate_capability_resource(
        resource, expected_language=base.language
    )
    return build_candidate_language_registry(
        base, entries, capability_language=resource["language"]
    )


def certify_candidate_registry(
    candidate: ProductionLanguageRegistry, *, certification_hash: str
) -> ProductionLanguageRegistry:
    """Stamp explicit canonical certification provenance on candidate entries.

    Only entries whose source_kind is CANDIDATE are stamped; canonical and
    paraphrase entries are untouched. The certified registry is the promoted
    canonical registry and has its own deterministic hash.
    """
    entries = []
    for entry in candidate.entries:
        if entry.source_kind == "CANDIDATE":
            entry = entry.model_copy(
                update={
                    "source_kind": "CERTIFIED_CAPABILITY",
                    "certification_status": "CERTIFIED",
                    "certification_hash": certification_hash,
                }
            )
        entries.append(entry)
    certified = ProductionLanguageRegistry(
        language=candidate.language,
        version=candidate.version,
        schema_version=candidate.schema_version,
        template_library_version=candidate.template_library_version,
        template_library_hash=candidate.template_library_hash,
        paraphrase_library_version=candidate.paraphrase_library_version,
        paraphrase_library_hash=candidate.paraphrase_library_hash,
        quality_model=candidate.quality_model,
        quality_call_ids=list(candidate.quality_call_ids),
        registry_hash="PENDING",
        entries=entries,
    )
    certified.registry_hash = certified.computed_hash()
    return certified


def write_registry_deterministically(
    registry: ProductionLanguageRegistry, path: Path
) -> str:
    """Write the registry JSON deterministically; return its file sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = registry.model_dump(mode="json")
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

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
from pathlib import Path
from typing import Any, Mapping

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

_OPERATORS = frozenset(
    {"DIRECT", "EQUALITY", "PAIRWISE_SELECTION", "TARGET_MATCH", "COMPOSITE"}
)
_UNIT_POLICIES = frozenset({"required", "forbidden", "any"})
_OWNERS = frozenset({"slot", "literal", "none"})
_REQUIRED_CAPABILITY_KEYS = (
    "capability_id",
    "operator",
    "semantic_class",
    "answer_kind",
    "pattern",
    "required_slots",
    "unit_policy",
    "match_policy",
    "entity_scopes",
    "entity_reference_owner",
)


class CandidateRegistryError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}:{detail}" if detail else code)


def validate_candidate_capability_resource(
    data: Any, *, expected_language: str | None = None
) -> list[dict[str, Any]]:
    """Validate the capability resource contract before building entries."""
    if not isinstance(data, Mapping):
        raise CandidateRegistryError("CANDIDATE_CAPABILITIES_MALFORMED", "not_an_object")
    for key in ("schema_version", "capability_set_id", "language", "entries"):
        if key not in data:
            raise CandidateRegistryError("CANDIDATE_CAPABILITIES_MALFORMED", f"missing:{key}")
    language = data["language"]
    if expected_language is not None and language != expected_language:
        raise CandidateRegistryError(
            "LANGUAGE_CAPABILITY_LANGUAGE_MISMATCH",
            f"{language}!={expected_language}",
        )
    entries = data["entries"]
    if not isinstance(entries, list):
        raise CandidateRegistryError("CANDIDATE_CAPABILITIES_MALFORMED", "entries_not_list")
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise CandidateRegistryError("CANDIDATE_CAPABILITIES_MALFORMED", "entry_not_object")
        for key in _REQUIRED_CAPABILITY_KEYS:
            if key not in entry:
                raise CandidateRegistryError(
                    "CANDIDATE_CAPABILITY_KEY_MISSING",
                    f"{entry.get('capability_id', '?')}:{key}",
                )
        if entry["operator"] not in _OPERATORS:
            raise CandidateRegistryError("CANDIDATE_CAPABILITY_INVALID_OPERATOR", str(entry["operator"]))
        if entry["unit_policy"] not in _UNIT_POLICIES:
            raise CandidateRegistryError("CANDIDATE_CAPABILITY_INVALID_UNIT_POLICY", str(entry["unit_policy"]))
        if entry["entity_reference_owner"] not in _OWNERS:
            raise CandidateRegistryError("CANDIDATE_CAPABILITY_INVALID_OWNER", str(entry["entity_reference_owner"]))
    return entries


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


def build_candidate_language_registry(
    base: ProductionLanguageRegistry,
    capabilities: list[dict[str, Any]],
    *,
    capability_language: str | None = None,
) -> ProductionLanguageRegistry:
    """Return a deterministic candidate registry = base + generic capabilities."""
    if capability_language is not None and capability_language != base.language:
        raise CandidateRegistryError(
            "LANGUAGE_CAPABILITY_LANGUAGE_MISMATCH",
            f"{capability_language}!={base.language}",
        )
    existing_ids = {entry.language_entry_id for entry in base.entries}
    new_entries: list[LanguageRegistryEntry] = []
    seen: set[str] = set()
    for capability in capabilities:
        entry_id = capability.get("capability_id")
        if not entry_id:
            raise CandidateRegistryError("CANDIDATE_CAPABILITY_ID_MISSING", str(capability))
        if entry_id in existing_ids or entry_id in seen:
            raise CandidateRegistryError("DUPLICATE_LANGUAGE_ENTRY_ID", entry_id)
        seen.add(entry_id)
        new_entries.append(_entry_from_capability(capability, base))

    merged = sorted(
        list(base.entries) + new_entries,
        key=lambda entry: entry.language_entry_id,
    )
    candidate = ProductionLanguageRegistry(
        language=base.language,
        version=base.version + CANDIDATE_VERSION_SUFFIX,
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

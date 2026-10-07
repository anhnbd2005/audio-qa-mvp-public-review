"""Phase 4.2 candidate language-registry compiler.

Deterministically compiles a CANDIDATE ``ProductionLanguageRegistry`` from the
canonical registry plus dataset-neutral capability entries declared in
``resources/language/candidate_capabilities.json``.

The candidate registry is written ONLY to a scratch destination. It never
overwrites ``resources/language/production_registry.json``; existing canonical
ProductionContracts pin the canonical registry hash and must not be invalidated
by an in-place mutation.

No dataset IDs, task IDs, field names, medical terms, or LLM calls appear here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


class CandidateRegistryError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}:{detail}" if detail else code)


def load_candidate_capabilities(
    path: Path = CANDIDATE_CAPABILITIES_RESOURCE,
) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    if not isinstance(entries, list):
        raise CandidateRegistryError("CANDIDATE_CAPABILITIES_MALFORMED", str(path))
    return entries


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
) -> ProductionLanguageRegistry:
    """Return a deterministic candidate registry = base + generic capabilities."""
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
    return build_candidate_language_registry(
        base, load_candidate_capabilities(capabilities_path)
    )


def write_registry_deterministically(
    registry: ProductionLanguageRegistry, path: Path
) -> str:
    """Write the registry JSON deterministically; return its file sha256."""
    import hashlib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = registry.model_dump(mode="json")
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

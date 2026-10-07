"""Generic comparator registry: first-class semantic relation contracts.

Each semantic relation declares an explicit ``comparator_id``. A comparator is
an ordered list of deterministic normalization operations plus its declared
invariances and preserved features. Comparators are dataset-agnostic.

The defective historical pipeline reused one implicit normalization for the
reference-match relation; comparators make the intended proposition explicit
and hashable so changing semantics stales dependent plans.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from src.common.config import ROOT

COMPARATORS_RESOURCE = ROOT / "resources" / "semantics" / "comparators.json"

PUNCTUATION_POLICIES = ("preserve", "replace_with_space")
UNICODE_POLICIES = ("none", "NFC", "NFKC")
CASE_POLICIES = ("preserve", "casefold")


class ComparatorSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    comparator_id: str
    semantic_purpose: str
    ordered_normalization_operations: tuple[str, ...]
    invariances: tuple[str, ...] = ()
    preserved_features: tuple[str, ...] = ()
    punctuation_category_policy: str = "preserve"
    unicode_normalization: str = "none"
    case_policy: str = "preserve"

    def logical_hash(self) -> str:
        payload = {
            "comparator_id": self.comparator_id,
            "ordered_normalization_operations": list(
                self.ordered_normalization_operations
            ),
            "invariances": sorted(self.invariances),
            "preserved_features": sorted(self.preserved_features),
            "punctuation_category_policy": self.punctuation_category_policy,
            "unicode_normalization": self.unicode_normalization,
            "case_policy": self.case_policy,
        }
        return hashlib.sha256(
            json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()


class ComparatorRegistry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    language: str
    version: str
    schema_version: int
    comparators: tuple[ComparatorSpec, ...]

    def by_id(self, comparator_id: str) -> ComparatorSpec:
        for comparator in self.comparators:
            if comparator.comparator_id == comparator_id:
                return comparator
        raise KeyError(f"UNKNOWN_COMPARATOR:{comparator_id}")

    def logical_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "language": self.language,
                    "version": self.version,
                    "comparators": [
                        c.model_dump(mode="json") for c in self.comparators
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


def load_comparator_registry(
    path: Path = COMPARATORS_RESOURCE,
) -> ComparatorRegistry:
    return ComparatorRegistry.model_validate_json(path.read_text(encoding="utf-8"))


def comparator_set_hash(
    comparator_ids: tuple[str, ...], registry: ComparatorRegistry | None = None
) -> str:
    """Logical hash of the SUBSET of comparators a dataset actually references.

    This keeps dataset contract identity stable when unrelated comparators are
    added to the shared registry.
    """
    reg = registry or load_comparator_registry()
    selected = [reg.by_id(cid) for cid in comparator_ids]
    return ComparatorRegistry(
        language=reg.language,
        version=reg.version,
        schema_version=reg.schema_version,
        comparators=tuple(selected),
    ).logical_hash()


def _replace_punctuation_with_space(text: str) -> str:
    out = []
    for ch in text:
        if unicodedata.category(ch).startswith("P"):
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def _collapse_strip(text: str) -> str:
    return " ".join(text.split())


def normalize_with_comparator(comparator: ComparatorSpec, value: Any) -> str:
    text = "" if value is None else str(value)
    for operation in comparator.ordered_normalization_operations:
        if operation == "unicode_nfc":
            text = unicodedata.normalize("NFC", text)
        elif operation == "unicode_nfkc":
            text = unicodedata.normalize("NFKC", text)
        elif (
            operation
            == "replace_every_unicode_punctuation_character_category_P_with_single_space"
        ):
            text = _replace_punctuation_with_space(text)
        elif operation == "collapse_unicode_whitespace_runs_to_single_ascii_space":
            text = " ".join(text.split())
        elif operation == "strip_leading_and_trailing_whitespace":
            text = text.strip()
        elif operation == "casefold_for_case_insensitive_comparison":
            text = text.casefold()
        elif operation == "exact_string_equality":
            continue
        else:
            raise ValueError(f"UNKNOWN_NORMALIZATION_OPERATION:{operation}")
    if comparator.unicode_normalization != "none":
        text = unicodedata.normalize(comparator.unicode_normalization, text)
    if comparator.case_policy == "casefold":
        text = text.casefold()
    return _collapse_strip(text)


def apply_comparator(comparator: ComparatorSpec, left: Any, right: Any) -> bool:
    return normalize_with_comparator(comparator, left) == normalize_with_comparator(
        comparator, right
    )

"""Generic capability-level policy for candidate Audio-QA question types.

Dataset-agnostic by construction: no field names, no dataset names. Every
decision derives from the dataset profile (field roles, evaluation policy,
context visibility, declared relations, QA constraints).

Two layers are combined per candidate type:

1. STRUCTURAL CONTRACT — src.autonomous_qa.core.validity.validate_question_type() (existing,
   reused unchanged): unknown uses/answer fields, missing audio input,
   answer shape/references, role-based answer eligibility, answer/context
   overlap, hidden/forbidden context requests.

2. CAPABILITY POLICY — audit_candidate_type() (this module): the rules the
   pipeline needs before a type may be accepted downstream.

   (1) a field that is a GOLD SOURCE of the type must never also be visible
       question context for that same type;
   (2) a field may be visible context only when the type declares it, it is
       not a gold source, exposing it does not make audio unnecessary, and
       profile context visibility allows it;
   (3) role=hidden_identifier is never rendered literally;
   (4) evaluation=discovery_only never becomes a direct semantic answer;
   (5) role=context_only is contextual metadata only — not automatically
       answerable;
   (6) an Audio-QA type must actually require the audio signal: if the gold
       is deterministically available from the visible metadata alone the
       type is METADATA_SOLVABLE and is rejected;
   (7) an encoded field with an unresolved mapping never yields a
       human-readable classification answer.

Machine-readable contract per type (derived, no schema redesign — the
existing `uses` / `answer` / `input_context_fields` fields already express
all of it):

    audio_required         == "audio" (or audio_1/audio_2) appears in `uses`
    visible_context_fields == `input_context_fields`
    gold_source_fields     == required_answer_fields(`answer`)
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field

from src.autonomous_qa.core.validity import (
    EVALUATION_DISCOVERY_ONLY,
    ROLE_AUDIO,
    ROLE_CONTEXT_ONLY,
    ROLE_HIDDEN_IDENTIFIER,
    ROLE_PROVENANCE,
    ROLE_SEMANTIC,
    VISIBILITY_FORBIDDEN,
    VISIBILITY_HIDDEN,
    VISIBILITY_VISIBLE,
    _is_audio_input,
    extract_context_visibilities,
    extract_evaluation_policies,
    extract_field_roles,
    is_evaluation_eligible_type,
    normalize_field_reference,
    required_answer_fields,
    validate_question_type,
)

STATUS_SUPPORTED = "SUPPORTED"
STATUS_REJECTED = "REJECTED"
STATUS_REVIEW = "REVIEW_REQUIRED"

AUDIO_INPUT_ALIASES = ("audio", "audio_1", "audio_2")

# Constraint wording that marks a categorical encoding as UNRESOLVED.
_ENCODING_HINTS = ("map", "encod", "undefined mapping", "label set")


@dataclass(frozen=True)
class FieldInfo:
    name: str
    role: str
    evaluation: str
    context_visibility: str
    is_categorical: bool = False
    entity_scope: str | None = None
    dtype: str | None = None


@dataclass
class PolicyContract:
    """Profile-derived capability contract. Pure data + pure functions."""

    dataset: str
    fields: dict[str, FieldInfo] = dc_field(default_factory=dict)
    relations: list[dict] = dc_field(default_factory=list)
    qa_constraints: list[str] = dc_field(default_factory=list)
    semantic_fields: frozenset = frozenset()
    hidden_fields: frozenset = frozenset()
    unresolved_encoded_fields: frozenset = frozenset()

    # -- construction ----------------------------------------------------
    @classmethod
    def from_profile(cls, profile: dict) -> PolicyContract:
        if not isinstance(profile, dict) or not profile.get("fields"):
            raise ValueError("profile_missing_fields")

        raw_fields = profile.get("fields") or {}
        roles = dict(profile.get("field_roles") or extract_field_roles(profile))
        evals = dict(
            profile.get("field_evaluations")
            or extract_evaluation_policies(profile)
        )
        vis = dict(
            profile.get("context_visibilities")
            or extract_context_visibilities(profile)
        )

        fields: dict[str, FieldInfo] = {}
        for name, meta in raw_fields.items():
            meta = meta if isinstance(meta, dict) else {}
            fields[name] = FieldInfo(
                name=name,
                role=meta.get("role") or roles.get(name) or ROLE_SEMANTIC,
                evaluation=(
                    meta.get("evaluation") or evals.get(name) or "eligible"
                ),
                context_visibility=(
                    meta.get("context_visibility")
                    or vis.get(name)
                    or VISIBILITY_VISIBLE
                ),
                is_categorical=bool(meta.get("is_categorical", False)),
                entity_scope=meta.get("entity_scope"),
                dtype=meta.get("type"),
            )

        constraints = list(profile.get("qa_constraints") or [])
        contract = cls(
            dataset=str(profile.get("dataset") or ""),
            fields=fields,
            relations=[r for r in (profile.get("relations") or [])
                       if isinstance(r, dict)],
            qa_constraints=constraints,
            semantic_fields=frozenset(
                profile.get("semantic_fields")
                or [n for n, f in fields.items() if f.role == ROLE_SEMANTIC]
            ),
            hidden_fields=frozenset(
                profile.get("hidden_fields")
                or [n for n, f in fields.items()
                    if f.role == ROLE_HIDDEN_IDENTIFIER]
            ),
        )
        contract.unresolved_encoded_fields = frozenset(
            contract._derive_unresolved_encoded())
        return contract

    # -- derived views ---------------------------------------------------
    def field_names(self) -> frozenset[str]:
        return frozenset(self.fields)

    def field_roles(self) -> dict[str, str]:
        return {n: f.role for n, f in self.fields.items()}

    def evaluation_policies(self) -> dict[str, str]:
        return {n: f.evaluation for n, f in self.fields.items()}

    def context_visibilities(self) -> dict[str, str]:
        return {n: f.context_visibility for n, f in self.fields.items()}

    def info(self, name: str) -> FieldInfo | None:
        if name in self.fields:
            return self.fields[name]
        lower = {n.lower(): f for n, f in self.fields.items()}
        return lower.get(str(name).lower())

    def _derive_unresolved_encoded(self) -> list[str]:
        """Fields whose code/label mapping the profile leaves unresolved.

        Generic signal, no field names: either the profile's own QA
        constraints talk about mapping/encoding a field, or the field is a
        categorical context_only attribute (decoding it would require an
        unsupported label mapping).
        """
        out: set[str] = set()
        for constraint in self.qa_constraints:
            low = str(constraint).lower()
            if not any(h in low for h in _ENCODING_HINTS):
                continue
            for name, finfo in self.fields.items():
                if ((name in constraint or name.lower() in low)
                        and (finfo.is_categorical
                             or finfo.role == ROLE_CONTEXT_ONLY)):
                    out.add(name)
        for name, finfo in self.fields.items():
            if finfo.is_categorical and finfo.role == ROLE_CONTEXT_ONLY:
                out.add(name)
        return sorted(out)

    # -- helper queries --------------------------------------------------
    def visibility_of(self, field_name: str) -> str | None:
        f = self.info(field_name)
        return None if f is None else f.context_visibility

    def allows_relation(self, source: str, target: str) -> bool:
        return any(
            r.get("source_field") == source and r.get("target_field") == target
            for r in self.relations
        )


# -- machine-readable contract extraction ---------------------------------
def audio_required(qtype: dict) -> bool:
    uses = qtype.get("uses") or []
    return any(_is_audio_input(str(u)) for u in uses)


def visible_context_fields(qtype: dict) -> list[str]:
    ctx = qtype.get("input_context_fields")
    if not isinstance(ctx, list):
        return []
    return [str(c) for c in ctx]


def gold_source_fields(qtype: dict, contract: PolicyContract) -> list[str]:
    return required_answer_fields(
        qtype, schema_fields=contract.field_names())


def type_contract(qtype: dict, contract: PolicyContract) -> dict:
    """The four questions Python must be able to answer about any type."""
    return {
        "audio_required": audio_required(qtype),
        "visible_context_fields": visible_context_fields(qtype),
        "gold_source_fields": gold_source_fields(qtype, contract),
        "uses_fields": [str(u) for u in (qtype.get("uses") or [])],
        "answer_kind": (qtype.get("answer") or {}).get("kind"),
    }


# -- reason grouping (report buckets) -------------------------------------
GROUP_METADATA_SOLVABLE = "metadata-solvable"
GROUP_HIDDEN_ID = "hidden-id-exposure"
GROUP_PROVENANCE = "provenance-as-gold"
GROUP_DISCOVERY_ONLY = "discovery-only-as-gold"
GROUP_UNSUPPORTED_ENCODING = "unsupported-encoding"
GROUP_OTHER = "other"


def group_reason(reason: str) -> str:
    """Map a deterministic reason code onto a report bucket."""
    head = str(reason).split(":", 1)[0]
    if head in ("metadata_solvable", "audio_not_required",
                "missing_audio_input"):
        return GROUP_METADATA_SOLVABLE
    if head in ("hidden_identifier_exposed", "id_exposure",
                "hidden_context_requested"):
        return GROUP_HIDDEN_ID
    if head in ("provenance_as_gold",) or str(reason).startswith(
            "ineligible_field_role:provenance"):
        return GROUP_PROVENANCE
    if head in ("discovery_only_as_gold",):
        return GROUP_DISCOVERY_ONLY
    if head in ("unsupported_encoding_gold",
                "context_only_not_answerable") or str(reason).startswith(
                    "ineligible_field_role:context_only"):
        return GROUP_UNSUPPORTED_ENCODING
    return GROUP_OTHER


# One bucket per rejected type so report counts stay disjoint.
_GROUP_PRIORITY = (
    GROUP_UNSUPPORTED_ENCODING,
    GROUP_PROVENANCE,
    GROUP_HIDDEN_ID,
    GROUP_DISCOVERY_ONLY,
    GROUP_METADATA_SOLVABLE,
    GROUP_OTHER,
)


def primary_reason_group(record: dict) -> str | None:
    groups = set(record.get("reason_groups") or [])
    for g in _GROUP_PRIORITY:
        if g in groups:
            return g
    return None


# -- the audit ------------------------------------------------------------
def audit_candidate_type(
    qtype: dict,
    contract: PolicyContract,
    dataset_rows: list[dict] | None = None,
    relation_cache: dict | None = None,
    require_audio: bool = True,
) -> dict:
    """Deterministic policy audit for one proposed question type.

    Returns a record with the machine-readable contract, both error layers
    and a SUPPORTED / REJECTED / REVIEW_REQUIRED status.
    """
    ctx_fields = visible_context_fields(qtype)
    gold_fields = gold_source_fields(qtype, contract)
    tcon = type_contract(qtype, contract)
    ans = qtype.get("answer") or {}
    ans_kind = ans.get("kind")

    # Layer 1: structural contract (existing, unchanged).
    contract_errors = list(dict.fromkeys(validate_question_type(
        qtype,
        known_fields=contract.field_names(),
        dataset_rows=dataset_rows,
        relation_cache=relation_cache,
        hidden_fields=contract.hidden_fields,
        field_roles=contract.field_roles(),
        context_visibilities=contract.context_visibilities(),
    )))

    policy_errors: list[str] = []
    review: list[str] = []

    # -- rule (1): gold source must not be visible context ---------------
    gold_set = set(gold_fields)
    for f in ctx_fields:
        if f in gold_set:
            policy_errors.append(f"visible_context_overlaps_gold:{f}")

    # -- rule (2)+(3): only declared, allowed, non-identifying context ----
    for f in ctx_fields:
        finfo = contract.info(f)
        if finfo is None:
            policy_errors.append(f"undeclared_context_field:{f}")
            continue
        if finfo.role == ROLE_HIDDEN_IDENTIFIER:
            policy_errors.append(f"hidden_identifier_exposed:{f}")
        if finfo.context_visibility == VISIBILITY_HIDDEN:
            policy_errors.append(f"hidden_context_requested:{f}")
        elif finfo.context_visibility == VISIBILITY_FORBIDDEN:
            policy_errors.append(f"forbidden_context_requested:{f}")
        elif finfo.context_visibility != VISIBILITY_VISIBLE:
            policy_errors.append(
                f"invalid_context_visibility:{f}:{finfo.context_visibility}")

    # -- rules (4)+(5)+(7): what may be a direct semantic answer ---------
    if ans_kind == "field_value":
        key = ans.get("key")
        finfo = contract.info(key) if isinstance(key, str) else None
        if finfo is not None:
            if finfo.role == ROLE_PROVENANCE:
                policy_errors.append(f"provenance_as_gold:{key}")
            if finfo.evaluation == EVALUATION_DISCOVERY_ONLY:
                policy_errors.append(f"discovery_only_as_gold:{key}")
            if finfo.role == ROLE_CONTEXT_ONLY:
                policy_errors.append(f"context_only_not_answerable:{key}")
            if key in contract.unresolved_encoded_fields:
                policy_errors.append(f"unsupported_encoding_gold:{key}")

    # derived answers: neither endpoint may be provenance-only bookkeeping
    if ans_kind == "derived_field":
        for slot in ("source_key", "target_key"):
            key = ans.get(slot)
            finfo = contract.info(key) if isinstance(key, str) else None
            if finfo is not None and finfo.role == ROLE_PROVENANCE:
                policy_errors.append(f"provenance_as_gold:{key}")

    # -- rule (6): audio necessity / metadata solvability -----------------
    metadata_reasons: list[str] = []
    if gold_set and gold_set.issubset(set(ctx_fields)):
        metadata_reasons.append(
            "gold_fields_in_visible_context:" + ",".join(sorted(gold_set)))
    for rel in contract.relations:
        src, tgt = rel.get("source_field"), rel.get("target_field")
        if src in ctx_fields and tgt in gold_set:
            metadata_reasons.append(f"relation_visible_to_gold:{src}->{tgt}")
    for mr in metadata_reasons:
        policy_errors.append(f"metadata_solvable:{mr}")

    if require_audio and not tcon["audio_required"]:
        policy_errors.append("audio_not_required")

    # -- review-only signals ---------------------------------------------
    # A full semantic annotation (e.g. an utterance-level text channel)
    # offered as visible context while the gold comes from a different
    # field gives the model a non-audio substitute for listening. It may be
    # justified for a genuine multimodal operation, so it is flagged for
    # review instead of silently accepted or hard-rejected.
    if tcon["audio_required"] and gold_set:
        for f in ctx_fields:
            finfo = contract.info(f)
            if finfo is None or f in gold_set:
                continue
            if finfo.role != ROLE_SEMANTIC:
                continue
            if finfo.dtype == "string" or finfo.entity_scope == "utterance":
                review.append(f"context_channel_bypasses_audio:{f}")

    if gold_fields and not is_evaluation_eligible_type(
        qtype,
        contract.evaluation_policies(),
        contract.field_names(),
    ):
        review.append(
            "gold_not_evaluation_eligible:" + ",".join(gold_fields))
    for f in gold_fields:
        finfo = contract.info(f)
        if finfo is not None and finfo.role in (ROLE_HIDDEN_IDENTIFIER,
                                                ROLE_AUDIO) and not any(
            r.split(":", 1)[0] in ("hidden_identifier_exposed", "id_exposure")
            for r in policy_errors):
            # equality against a hidden key is allowed but must be visible
            # in the audit: it can never be an evaluation-pool literal.
            review.append(f"hidden_key_gold_source:{f}")

    # de-duplicate, preserve order (both layers together: one reason = one row)
    reasons = list(dict.fromkeys(policy_errors + contract_errors))
    policy_errors = list(dict.fromkeys(policy_errors))
    review = list(dict.fromkeys(review))
    contract_errors = list(dict.fromkeys(contract_errors))

    if contract_errors or policy_errors:
        status = STATUS_REJECTED
    elif review:
        status = STATUS_REVIEW
    else:
        status = STATUS_SUPPORTED

    return {
        "type_id": qtype.get("id"),
        "name": qtype.get("name"),
        "goal": qtype.get("goal"),
        "input_count": qtype.get("input_count"),
        "answer_rule": qtype.get("answer_rule"),
        **tcon,
        "gold_fields_evaluation": {
            f: (contract.info(f).evaluation if contract.info(f) else None)
            for f in gold_fields
        },
        "metadata_solvable": bool(metadata_reasons),
        "policy_errors": policy_errors,
        "contract_errors": contract_errors,
        "review_flags": review,
        "status": status,
        "reasons": reasons,
        "reason_groups": sorted({group_reason(r) for r in reasons}),
        "primary_reason_group": (
            primary_reason_group(
                {"reason_groups": sorted({group_reason(r) for r in reasons})})
            if reasons else None),
        "uses_fields": tcon["uses_fields"],
        "field_roles": {
            f: (contract.info(f).role if contract.info(f) else None)
            for f in sorted(set(tcon["uses_fields"]) | set(ctx_fields)
                            | set(gold_fields))
        },
    }


def audio_necessity_verdict(record: dict) -> dict:
    """Section Q: could the gold be derived without listening?"""
    if record.get("metadata_solvable") or not record.get("audio_required"):
        answer = "YES"
    else:
        answer = "NO"
    return {
        "type_id": record.get("type_id"),
        "gold_source_fields": record.get("gold_source_fields"),
        "visible_context_fields": record.get("visible_context_fields"),
        "answer_without_audio": answer,
        "verdict": (
            "METADATA_SOLVABLE"
            if answer == "YES"
            else "AUDIO_DEPENDENT"
        ),
        "classification": record.get("status"),
    }


__all__ = [
    "AUDIO_INPUT_ALIASES",
    "GROUP_DISCOVERY_ONLY",
    "GROUP_HIDDEN_ID",
    "GROUP_METADATA_SOLVABLE",
    "GROUP_OTHER",
    "GROUP_PROVENANCE",
    "GROUP_UNSUPPORTED_ENCODING",
    "ROLE_AUDIO",
    "ROLE_CONTEXT_ONLY",
    "ROLE_HIDDEN_IDENTIFIER",
    "ROLE_PROVENANCE",
    "ROLE_SEMANTIC",
    "STATUS_REJECTED",
    "STATUS_REVIEW",
    "STATUS_SUPPORTED",
    "VISIBILITY_FORBIDDEN",
    "VISIBILITY_HIDDEN",
    "VISIBILITY_VISIBLE",
    "FieldInfo",
    "PolicyContract",
    "audio_necessity_verdict",
    "audio_required",
    "audit_candidate_type",
    "gold_source_fields",
    "group_reason",
    "normalize_field_reference",
    "primary_reason_group",
    "type_contract",
    "visible_context_fields",
]

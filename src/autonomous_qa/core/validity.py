"""Hard validity checks in pure Python — no LLM needed for certain errors.

Runs BEFORE paraphrase/quality each round. Design principle:

- `answer_rule` is HUMAN-READABLE AUDIT TEXT. The ONLY check applied to
  it is non-emptiness. Its words are NEVER parsed for field names.
- The structured `answer` object is the SOLE source of truth for whether
  the gold answer is machine-computable.

Anything failing here is rejected deterministically and recorded — it never
reaches the Paraphrase or Quality LLM calls.
"""

from __future__ import annotations

import re

from src.common.missing_values import (
    extract_source_missing_values,
    normalize_row_missing_values,
    validate_derived_relation,
)

# Base dataset fields. Pairwise instance aliases (gender_1, speakerID_2,
# ...) are NOT columns; they are valid ONLY as equality pair operands
# (see parse_pair_field_reference()).
SCHEMA_FIELDS = frozenset({
    "audio", "text", "gender", "region", "province", "speakerID",
})

# Legacy alias kept for backwards compatibility with older checkpoints.
KNOWN_FIELDS = SCHEMA_FIELDS

# Question-side placeholder contract: [KEY] only, plus slot-free text.
# [VALUE] is FORBIDDEN in question text — gold comes only from the
# structured `answer` object. A template containing [VALUE] leaks the
# answer and is dropped as answer_leakage (not merely unknown).
ALLOWED_PLACEHOLDERS = frozenset({"KEY"})

# Internal identifiers must never be exposed as semantic gold answers.
# speakerID stays valid as HIDDEN comparison supervision for equality.
ID_LIKE_FIELDS = frozenset({"speakerid"})

# Generic field role system (§2-§4):
# audio: input evidence only; not a QA answer
# semantic: normal machine-answer validation
# hidden_identifier: equality comparison only; literal ID never exposed
# provenance: dataset tracking / provenance; not an audio QA attribute
# context_only: context understanding only; not directly executable
ROLE_AUDIO = "audio"
ROLE_SEMANTIC = "semantic"
ROLE_HIDDEN_IDENTIFIER = "hidden_identifier"
ROLE_PROVENANCE = "provenance"
ROLE_CONTEXT_ONLY = "context_only"

CANONICAL_FIELD_ROLES = frozenset({
    ROLE_AUDIO,
    ROLE_SEMANTIC,
    ROLE_HIDDEN_IDENTIFIER,
    ROLE_PROVENANCE,
    ROLE_CONTEXT_ONLY,
})

DEFAULT_FIELD_ROLES: dict[str, str] = {
    "audio": ROLE_AUDIO,
    "text": ROLE_SEMANTIC,
    "gender": ROLE_SEMANTIC,
    "region": ROLE_SEMANTIC,
    "province": ROLE_SEMANTIC,
    "speakerID": ROLE_HIDDEN_IDENTIFIER,
}

# Generic entity scope vocabulary (§I-§K):
# speaker: speaker-level attribute (gender, age, region, dialect, sex)
# utterance: utterance-level attribute (text, intent, scenario, transcription)
# recording: acoustic/recording-level property
# conversation: dialogue/conversation-level property
ENTITY_SCOPE_SPEAKER = "speaker"
ENTITY_SCOPE_UTTERANCE = "utterance"
ENTITY_SCOPE_RECORDING = "recording"
ENTITY_SCOPE_CONVERSATION = "conversation"

CANONICAL_ENTITY_SCOPES = frozenset({
    ENTITY_SCOPE_SPEAKER,
    ENTITY_SCOPE_UTTERANCE,
    ENTITY_SCOPE_RECORDING,
    ENTITY_SCOPE_CONVERSATION,
})
ENTITY_SCOPES = CANONICAL_ENTITY_SCOPES

DEFAULT_ENTITY_SCOPES: dict[str, str] = {
    "text": ENTITY_SCOPE_UTTERANCE,
    "gender": ENTITY_SCOPE_SPEAKER,
    "region": ENTITY_SCOPE_SPEAKER,
    "province": ENTITY_SCOPE_SPEAKER,
    "speakerID": ENTITY_SCOPE_SPEAKER,
}

# Generic evaluation policy vocabulary (§E):
# eligible: type may enter final evaluation/benchmark pool
# discovery_only: type remains in discovery pool, but excluded from evaluation/pilot pool
EVALUATION_ELIGIBLE = "eligible"
EVALUATION_DISCOVERY_ONLY = "discovery_only"

CANONICAL_EVALUATION_POLICIES = frozenset({
    EVALUATION_ELIGIBLE,
    EVALUATION_DISCOVERY_ONLY,
})
EVALUATION_POLICIES = CANONICAL_EVALUATION_POLICIES


def extract_entity_scopes(schema: dict | None) -> dict[str, str]:
    """Extract generic entity scopes from schema dictionary.

    Vocabulary: speaker, utterance, recording, conversation, etc.
    """
    if not isinstance(schema, dict):
        return dict(DEFAULT_ENTITY_SCOPES)
    if "entity_scopes" in schema and isinstance(schema["entity_scopes"], dict):
        return dict(schema["entity_scopes"])
    scopes: dict[str, str] = {}
    fields = schema.get("fields", {})
    if isinstance(fields, dict):
        for f, meta in fields.items():
            if isinstance(meta, dict) and "entity_scope" in meta:
                scopes[f] = str(meta["entity_scope"])
    return scopes or dict(DEFAULT_ENTITY_SCOPES)


def extract_evaluation_policies(schema: dict | None) -> dict[str, str]:
    """Extract evaluation policies from schema dictionary (§E)."""
    if not isinstance(schema, dict):
        return {}
    policies: dict[str, str] = {}
    fields = schema.get("fields", {})
    if isinstance(fields, dict):
        for f, meta in fields.items():
            if isinstance(meta, dict) and "evaluation" in meta:
                policies[f] = str(meta["evaluation"])
    if "field_evaluations" in schema and isinstance(schema["field_evaluations"], dict):
        for k, v in schema["field_evaluations"].items():
            policies[str(k)] = str(v)
    return policies


VISIBILITY_VISIBLE = "visible"
VISIBILITY_HIDDEN = "hidden"
VISIBILITY_FORBIDDEN = "forbidden"

CANONICAL_CONTEXT_VISIBILITIES = frozenset({
    VISIBILITY_VISIBLE,
    VISIBILITY_HIDDEN,
    VISIBILITY_FORBIDDEN,
})
ALLOWED_CONTEXT_VISIBILITIES = CANONICAL_CONTEXT_VISIBILITIES


def extract_context_visibilities(schema: dict | None) -> dict[str, str]:
    """Extract context visibility policies from schema dictionary (§5)."""
    if not isinstance(schema, dict):
        return {}
    if "context_visibilities" in schema and isinstance(schema["context_visibilities"], dict):
        return {str(k): str(v) for k, v in schema["context_visibilities"].items()}
    vis: dict[str, str] = {}
    fields = schema.get("fields", {})
    if isinstance(fields, dict):
        for f, meta in fields.items():
            if isinstance(meta, dict) and "context_visibility" in meta:
                vis[f] = str(meta["context_visibility"])
    return vis






def extract_field_roles(schema: dict | None) -> dict[str, str]:
    """Extract generic field roles from schema dictionary.

    Vocabulary: audio, semantic, hidden_identifier, provenance, context_only.
    """
    if not isinstance(schema, dict):
        return dict(DEFAULT_FIELD_ROLES)
    if "field_roles" in schema and isinstance(schema["field_roles"], dict):
        return dict(schema["field_roles"])
    roles: dict[str, str] = {}
    fields = schema.get("fields", {})
    if isinstance(fields, dict):
        for f, meta in fields.items():
            if isinstance(meta, dict) and "role" in meta:
                roles[f] = meta["role"]
            elif isinstance(meta, dict) and meta.get("is_audio"):
                roles[f] = ROLE_AUDIO
            elif isinstance(meta, dict) and meta.get("is_identifier"):
                roles[f] = ROLE_HIDDEN_IDENTIFIER
            elif isinstance(meta, dict) and meta.get("is_annotation"):
                roles[f] = ROLE_SEMANTIC
            else:
                roles[f] = ROLE_SEMANTIC
    if not roles and "hidden_fields" in schema:
        hidden = set(schema.get("hidden_fields", []))
        for f in schema.get("semantic_fields", []):
            roles[f] = ROLE_SEMANTIC
        for h in hidden:
            roles[h] = ROLE_HIDDEN_IDENTIFIER
    return roles or dict(DEFAULT_FIELD_ROLES)


def get_field_role(field: str, field_roles: dict[str, str] | None = None,
                   hidden_fields: set[str] | frozenset[str] | None = None) -> str:
    """Return the generic role of a field.

    Derives strictly from field_roles (parsed from schema.json) or
    default configuration. No field-name-specific branches.
    """
    if not isinstance(field, str):
        return "unknown"
    if field_roles:
        if field in field_roles:
            return field_roles[field]
        lower_map = {str(k).lower(): str(v) for k, v in field_roles.items()}
        if field.lower() in lower_map:
            return lower_map[field.lower()]
    if hidden_fields:
        if field in hidden_fields or field.lower() in {str(h).lower() for h in hidden_fields}:
            return ROLE_HIDDEN_IDENTIFIER
    if field in DEFAULT_FIELD_ROLES:
        return DEFAULT_FIELD_ROLES[field]
    if field.lower() == "speakerid":
        return ROLE_HIDDEN_IDENTIFIER
    return ROLE_SEMANTIC


def validate_field_role_for_answer(
    field: str,
    answer_kind: str,
    field_roles: dict[str, str] | None = None,
    hidden_fields: set[str] | frozenset[str] | None = None,
    slot: str = "key",
) -> list[str]:
    """Generic role-based QA eligibility check.

    QA eligibility derives entirely from field roles:
      - semantic: allowed for field_value, equality, derived_field (source & target)
      - hidden_identifier:
          * field_value: reject ("id_exposure")
          * equality: allow
          * derived_field: reject ("invalid_derived_hidden_source" / "invalid_derived_hidden_target")
      - provenance: reject for all answer kinds ("ineligible_field_role:provenance:{field}")
      - context_only: reject for all answer kinds ("ineligible_field_role:context_only:{field}")
      - audio: reject for all answer kinds ("ineligible_field_role:audio:{field}")
    """
    role = get_field_role(field, field_roles, hidden_fields=hidden_fields)
    if answer_kind == "field_value":
        if role == ROLE_SEMANTIC:
            return []
        if role == ROLE_HIDDEN_IDENTIFIER:
            return ["id_exposure"]
        return [f"ineligible_field_role:{role}:{field}"]

    if answer_kind == "equality":
        if role in (ROLE_SEMANTIC, ROLE_HIDDEN_IDENTIFIER):
            return []
        return [f"ineligible_field_role:{role}:{field}"]

    if answer_kind == "derived_field":
        if role == ROLE_SEMANTIC:
            return []
        if role == ROLE_HIDDEN_IDENTIFIER:
            if slot == "source_key":
                return ["invalid_derived_hidden_source"]
            return ["invalid_derived_hidden_target"]
        return [f"ineligible_field_role:{role}:{field}"]

    return [f"bad_answer_kind:{answer_kind}"]


ANSWER_KINDS = frozenset({
    "field_value", "equality", "derived_field",
})

_PLACEHOLDER_RE = re.compile(r"\[([A-Za-z_][A-Za-z0-9_]*(?::[A-Za-z_][A-Za-z0-9_]*)?)\]")
_INSTANCE_SUFFIX_RE = re.compile(r"(.+)_([12])")


def normalize_question_text(text: str) -> str:
    """Exact normalized form for local dedup. Intentionally shallow:

    1. Unicode NFC normalize. 2. Strip ends. 3. Collapse every whitespace
    run to one ASCII space. 4. Casefold for comparison.

    Deliberately NOT: punctuation removal, diacritic stripping, stemming,
    stopwords, semantic similarity, embeddings.
    """
    import unicodedata

    text = unicodedata.normalize("NFC", text or "")
    return " ".join(text.strip().split()).casefold()


def normalize_field_reference(ref, schema_fields=SCHEMA_FIELDS):
    """Legacy helper: exact match, else strip ONE terminal _1/_2.

    Scope after the answer-reference contract patch: equality pair
    operands and `uses` entries ONLY. NEVER for field_value.key,
    derived source/target keys, their rendering, or their concept-field
    derivation. Prefer parse_pair_field_reference() for equality.

    gender -> gender | gender_1 -> gender | foo_1 -> None.
    Non-string refs return None.
    """
    if not isinstance(ref, str):
        return None
    if ref in schema_fields:
        return ref
    m = _INSTANCE_SUFFIX_RE.fullmatch(ref)
    if m and m.group(1) in schema_fields:
        return m.group(1)
    return None


def parse_pair_field_reference(ref, schema_fields=None):
    """Parse ONE equality pair-instance reference (§7).

    Requires exactly one terminal "_1"/"_2" suffix. If schema_fields is provided,
    the remaining base must exist in schema_fields. Returns (base_field, 1|2), else None.
    No recursion, no fuzzy match. Equality ONLY — never field_value or
    derived_field.
    """
    if not isinstance(ref, str):
        return None
    m = _INSTANCE_SUFFIX_RE.fullmatch(ref)
    if m is None:
        return None
    base, idx = m.group(1), m.group(2)
    if idx not in ("1", "2"):
        return None
    if schema_fields is not None and base not in set(schema_fields):
        return None
    return (base, int(idx))


# Canonical answer shapes: each kind owns its fields. Irrelevant fields
# must be absent/null — never silently ignored. Transcription executes
# row["text"], so its only permitted annotation is key=="text".
_ANSWER_SHAPE_FIELDS: dict[str, tuple[list[str], list[str]]] = {
    # kind: (required, forbidden-non-null)
    "field_value": (["key"], ["keys", "source_key", "target_key"]),
    "equality": (["keys"], ["key", "source_key", "target_key"]),
    "derived_field": (["source_key", "target_key"], ["key", "keys"]),
}

_CANONICAL_ANSWER_KINDS = frozenset(
    {"field_value", "equality", "derived_field"})


def validate_answer_references(answer, schema_fields=SCHEMA_FIELDS,
                               field_roles=None, hidden_fields=None):
    """ONE central answer-reference validator.

    field_value: key must be an EXACT raw field (no _1/_2 normalization)
                 with semantic role.
    derived_field: source_key AND target_key must be EXACT raw fields
                   with semantic role.
    equality: exactly [F_1, F_2] for the SAME raw base F (canonical order)
              with semantic or hidden_identifier role.
    Returns a list of problems (empty == structurally executable).
    """
    if schema_fields is None:
        schema_fields = SCHEMA_FIELDS
    schema = set(schema_fields)
    ans = answer or {}
    kind = ans.get("kind")
    if kind == "field_value":
        key = ans.get("key")
        if not isinstance(key, str) or key not in schema:
            return [f"unknown_answer_field:{key}"]
        return validate_field_role_for_answer(
            key, "field_value", field_roles=field_roles,
            hidden_fields=hidden_fields, slot="key")
    if kind == "derived_field":
        errors = []
        for slot, k in (("source_key", ans.get("source_key")),
                        ("target_key", ans.get("target_key"))):
            if not isinstance(k, str) or k not in schema:
                errors.append(f"unknown_answer_field:{k}")
            else:
                errors.extend(validate_field_role_for_answer(
                    k, "derived_field", field_roles=field_roles,
                    hidden_fields=hidden_fields, slot=slot))
        return errors
    if kind == "equality":
        keys = ans.get("keys") or []
        if not isinstance(keys, list) or len(keys) != 2:
            return ["invalid_equality_arity"]
        left_ref, right_ref = keys[0], keys[1]
        left = (parse_pair_field_reference(left_ref, schema)
                if isinstance(left_ref, str) else None)
        right = (parse_pair_field_reference(right_ref, schema)
                 if isinstance(right_ref, str) else None)
        if left is None or right is None:
            errors = []
            for ref, parsed in ((left_ref, left), (right_ref, right)):
                if parsed is None:
                    if (isinstance(ref, str)
                            and _INSTANCE_SUFFIX_RE.fullmatch(ref) is not None):
                        problem = f"unknown_answer_field:{ref}"
                    else:
                        problem = "invalid_equality_instances"
                    if problem not in errors:
                        errors.append(problem)
            return errors
        if left[1] != 1 or right[1] != 2:
            return ["invalid_equality_instances"]
        if left[0].lower() != right[0].lower():
            return ["equality_field_mismatch"]
        base_field = left[0]
        return validate_field_role_for_answer(
            base_field, "equality", field_roles=field_roles,
            hidden_fields=hidden_fields, slot="base")
    return [f"bad_answer_kind:{kind}"]


def validate_answer_shape(answer) -> list[str]:
    """ONE central canonical-shape validator (§8).

    Order: kind -> shape for that kind. Required fields must be present
    (not None); kind-foreign fields must be absent/null. Only None/absent
    counts as unused — [], "", {} are NONCANONICAL.
    """
    ans = answer if isinstance(answer, dict) else {}
    kind = ans.get("kind")
    # No transcription special-case: field_value(text) follows exactly the
    # same canonical-shape rules as every other field_value. There is no
    # executable answer kind named "transcription".
    if kind not in _CANONICAL_ANSWER_KINDS:
        return [f"bad_answer_kind:{kind}"]
    required, forbidden = _ANSWER_SHAPE_FIELDS[kind]
    errors = []
    for field in required:
        if ans.get(field) is None:
            errors.append(
                f"answer_shape_missing_field:{kind}:{field}")
    for field in forbidden:
        if ans.get(field) is not None:
            errors.append(
                f"answer_shape_unexpected_field:{kind}:{field}")
    return errors


def executable_signature(question_type, schema_fields=None, field_roles=None,
                         hidden_fields=None) -> tuple:
    """Executable QA identity (§14-§20): (arity, operation, field(s)).

    Derived ONLY from the canonical structured answer. Ignores type_id,
    name, goal, answer_rule, wording, phrases, uses — and never branches
    on the key value or any family/semantic label. A text-keyed
    field_value is Speech Transcription for ViMD, but the signature
    neither knows nor cares: it is the plain field_value form like any
    other field.
    Requires canonical shape + valid references; raises ValueError (local
    invariant) on structurally invalid input instead of normalizing.
    """
    if schema_fields is None:
        schema_fields = SCHEMA_FIELDS
    ans = ((question_type or {}).get("answer", {}) or {})
    problems = (validate_answer_shape(ans)
                + validate_answer_references(
                    ans, schema_fields=schema_fields, field_roles=field_roles,
                    hidden_fields=hidden_fields))
    if problems:
        raise ValueError(
            "Cannot sign structurally invalid answer: "
            + ";".join(problems))
    kind = ans.get("kind")
    if kind == "field_value":
        key = ans.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError(
                "Cannot sign noncanonical field_value answer.")
        return ("audio1", "field_value", key)
    if kind == "equality":
        keys = ans.get("keys")
        if not isinstance(keys, list) or len(keys) != 2:
            raise ValueError(
                "Cannot sign noncanonical equality answer.")
        left = (parse_pair_field_reference(keys[0], schema_fields=schema_fields)
                if isinstance(keys[0], str) else None)
        right = (parse_pair_field_reference(keys[1], schema_fields=schema_fields)
                 if isinstance(keys[1], str) else None)
        if (left is None or right is None or left[1] != 1
                or right[1] != 2 or left[0] != right[0]):
            raise ValueError(
                "Cannot sign noncanonical equality pair.")
        return ("audio2", "equality", left[0])
    if kind == "derived_field":
        src, dst = ans.get("source_key"), ans.get("target_key")
        if (not isinstance(src, str) or not src
                or not isinstance(dst, str) or not dst):
            raise ValueError(
                "Cannot sign noncanonical derived_field answer.")
        return ("audio1", "derived_field", src, dst)
    raise ValueError(
        f"Cannot sign unsupported answer kind: {kind!r}.")


def template_family_signature(
    question_type: dict,
    field_roles: dict[str, str] | None = None,
    schema_fields: set[str] | frozenset[str] | None = None,
    entity_scopes: dict[str, str] | None = None,
) -> tuple[str, str, str] | None:
    """Deterministic template family identity (§2, §27, §L, §M).

    Returns ("equality", "semantic", entity_scope) if the question type is a
    canonical equality comparison over a base field with role == "semantic"
    and a valid canonical entity_scope.
    Returns None for all other kinds (hidden_identifier equality,
    field_value, derived_field, or malformed answers).
    If entity_scope is absent, unknown, or malformed, returns None (fail-safe).
    Branches strictly on machine answer structure and schema metadata,
    NEVER on literal field names.
    """
    if not isinstance(question_type, dict):
        return None
    ans = question_type.get("answer")
    if not isinstance(ans, dict) or ans.get("kind") != "equality":
        return None
    keys = ans.get("keys")
    if isinstance(keys, list) and len(keys) == 2 and isinstance(keys[0], str) and isinstance(keys[1], str):
        if schema_fields is None:
            if field_roles:
                schema = set(field_roles.keys())
            else:
                schema = set(SCHEMA_FIELDS)
        else:
            schema = set(schema_fields)

        left = parse_pair_field_reference(keys[0], schema_fields=schema)
        right = parse_pair_field_reference(keys[1], schema_fields=schema)
        if left is None or right is None:
            return None
        if left[1] != 1 or right[1] != 2 or left[0] != right[0]:
            return None
        base_field = left[0]
    elif isinstance(ans.get("key"), str) and ans.get("key"):
        base_field = ans.get("key")
    else:
        return None

    role = get_field_role(base_field, field_roles)
    if role == ROLE_SEMANTIC:
        scopes = DEFAULT_ENTITY_SCOPES if entity_scopes is None else entity_scopes
        scope = scopes.get(base_field)
        if not scope:
            lower_scopes = {str(k).lower(): str(v) for k, v in scopes.items()}
            scope = lower_scopes.get(base_field.lower())
        if scope and scope in CANONICAL_ENTITY_SCOPES:
            return ("equality", "semantic", scope)
        return None
    return None


def required_answer_fields(
    question_type: dict,
    schema_fields: set[str] | frozenset[str] | None = None,
) -> list[str]:
    """Extract semantic field(s) required to materialize an answer (§D).

    field_value(F) -> [F]
    equality(F_1, F_2) -> [F]
    derived_field(S, T) -> [S, T]
    Returns empty list if unparseable or unsupported.
    Fails closed: no string-splitting or regex guessing on equality keys.
    """
    if not isinstance(question_type, dict):
        return []
    ans = question_type.get("answer")
    if not isinstance(ans, dict):
        return []
    kind = ans.get("kind")
    if kind == "field_value":
        key = ans.get("key")
        if isinstance(key, str) and key:
            if schema_fields is None or key in schema_fields:
                return [str(key)]
        return []
    if kind == "equality":
        keys = ans.get("keys")
        if (isinstance(keys, list) and len(keys) == 2
                and isinstance(keys[0], str) and isinstance(keys[1], str)):
            left = parse_pair_field_reference(keys[0], schema_fields=schema_fields)
            right = parse_pair_field_reference(keys[1], schema_fields=schema_fields)
            if (left is not None and right is not None
                    and left[1] == 1 and right[1] == 2
                    and left[0] == right[0]):
                return [left[0]]
            return []
        return []
    if kind == "derived_field":
        s, t = ans.get("source_key"), ans.get("target_key")
        res = []
        if isinstance(s, str) and s and (schema_fields is None or s in schema_fields):
            res.append(s)
        if isinstance(t, str) and t and (schema_fields is None or t in schema_fields):
            res.append(t)
        if len(res) == 2:
            return res
        return []
    return []


def is_evaluation_eligible_type(
    question_type: dict,
    evaluation_policies: dict[str, str] | None = None,
    schema_fields: set[str] | frozenset[str] | None = None,
) -> bool:
    """Check if a question type satisfies evaluation policy (§E).

    Fail-closed:
    - If evaluation_policies is missing, not a dict, or empty, returns False.
    - If required_answer_fields is empty or unparseable, returns False.
    - If any required answer field lacks an explicit policy in evaluation_policies,
      or policy != EVALUATION_ELIGIBLE ('eligible'), returns False.
    A type is evaluation-eligible ONLY if EVERY required answer field
    explicitly declares evaluation == 'eligible'.
    """
    if not evaluation_policies or not isinstance(evaluation_policies, dict):
        return False
    req_fields = required_answer_fields(question_type, schema_fields=schema_fields)
    if not req_fields:
        return False
    for f in req_fields:
        if f not in evaluation_policies:
            return False
        pol = evaluation_policies[f]
        if pol != EVALUATION_ELIGIBLE:
            return False
    return True



def build_approved_signature_index(approved_pool, schema_fields=None) -> dict:
    """Approved executable signatures in approval order (§24/§59).

    Returns {signature: approved type_id}. Raises ValueError
    duplicate_approved_executable_signature:<sig> when the pool itself
    contains a duplicate (impossible in a correct post-patch run).
    """
    index: dict[tuple, str] = {}
    for t in approved_pool or []:
        sig = executable_signature(t, schema_fields=schema_fields)
        tid = t.get("type_id", t.get("id"))
        if sig in index:
            raise ValueError(
                "duplicate_approved_executable_signature:"
                f"{sig!r}")
        index[sig] = tid
    return index


def deduplicate_by_signature(valid_types_in_style_order,
                             approved_signature_index,
                             schema_fields=None) -> tuple:
    """Split gate-valid types into novel + deterministic duplicates (§27).

    Seeds from the approved index (prior approved always wins), then
    iterates in ORIGINAL Style order (first current type wins a new
    signature). Returns (novel_types, records) where each record holds
    type_id/duplicate_of/signature/duplicate_source. Pure: no I/O.
    """
    approved_index = dict(approved_signature_index or {})
    seen = dict(approved_index)
    novel: list[dict] = []
    duplicates: list[dict] = []
    for t in valid_types_in_style_order or []:
        sig = executable_signature(t, schema_fields=schema_fields)
        tid = t.get("id", t.get("type_id"))
        if sig in seen:
            duplicates.append({
                "type_id": tid,
                "duplicate_of": seen[sig],
                "signature": list(sig),
                "duplicate_source": (
                    "prior_approved" if sig in approved_index
                    else "current_round"),
            })
        else:
            seen[sig] = tid
            novel.append(t)
    return novel, duplicates



def _is_audio_input(use: str) -> bool:
    """Structured audio check: literal audio / audio_1 / audio_2 only."""
    return use == "audio" or use == "audio_1" or use == "audio_2"


def validate_question_type(qtype: dict,
                           known_fields=SCHEMA_FIELDS,
                           dataset_rows=None,
                           relation_cache=None,
                           hidden_fields=None,
                           field_roles=None,
                           context_visibilities=None,
                           production_capabilities=None) -> list[str]:
    """Return a list of deterministic rejection reasons (empty == valid)."""
    errors: list[str] = []
    schema = set(known_fields)

    # --- uses: every entry must resolve to a real field/alias ---
    uses = qtype.get("uses", []) or []
    for u in uses:
        if normalize_field_reference(u, schema) is None:
            errors.append(f"unknown_uses_field:{u}")
    if not any(_is_audio_input(u) for u in uses):
        errors.append("missing_audio_input")

    # --- answer_rule: opaque prose, non-empty only. Never tokenized. ---
    rule = qtype.get("answer_rule", "")
    if not isinstance(rule, str) or not rule.strip():
        errors.append("empty_answer_rule")

    # --- answer: the ONLY machine gold specification ---
    ans = qtype.get("answer", {}) or {}
    shape_problems = validate_answer_shape(ans)
    errors.extend(shape_problems)
    ref_problems: list[str] = []
    if not any(p.startswith("bad_answer_kind:") for p in shape_problems):
        ref_problems = validate_answer_references(
            ans, schema_fields=schema, field_roles=field_roles,
            hidden_fields=hidden_fields)
        errors.extend(ref_problems)
    if (ans.get("kind") == "derived_field" and not shape_problems
            and not ref_problems):
        relation = validate_derived_relation(
            ans.get("source_key"), ans.get("target_key"), dataset_rows,
            schema_fields=schema, hidden_fields=hidden_fields,
            relation_cache=relation_cache, field_roles=field_roles)
        if not relation["valid"]:
            errors.append(relation["reason"])

    # --- input_context_fields validation (§4, §5, §12, §13) ---
    ctx_fields = qtype.get("input_context_fields")
    if ctx_fields is not None:
        if not isinstance(ctx_fields, list):
            errors.append("invalid_input_context_fields_format")
        else:
            ans_kind = ans.get("kind")
            ans_key = ans.get("key")
            eq_bases = []
            if ans_kind == "equality":
                for k in (ans.get("keys") or []):
                    if isinstance(k, str) and (k.endswith("_1") or k.endswith("_2")):
                        eq_bases.append(k[:-2])
                    elif isinstance(k, str):
                        eq_bases.append(k)

            for cf in ctx_fields:
                if not isinstance(cf, str):
                    errors.append(f"invalid_context_field:{cf}")
                    continue
                # Observed transcript/phone leakage forbidden
                if cf.lower() in ("transcript", "observed_phones", "observed_units", "transcript_str"):
                    errors.append(f"observed_transcript_leakage:{cf}")
                # Speaker ID exposure forbidden
                if cf.lower() in ("speakerid", "speaker_id", "speaker_name"):
                    errors.append(f"id_exposure:{cf}")
                # Answer-context overlap check (answer cannot be in visible context)
                if ans_kind == "field_value" and ans_key and cf.lower() == ans_key.lower():
                    errors.append(f"answer_context_overlap:{cf}")
                elif ans_kind == "equality" and any(cf.lower() == b.lower() for b in eq_bases):
                    errors.append(f"answer_context_overlap:{cf}")
                elif ans_kind == "derived_field":
                    t_key = ans.get("target_key")
                    if t_key and cf.lower() == t_key.lower():
                        errors.append(f"answer_context_overlap:{cf}")

                # Check context_visibilities if configured
                if context_visibilities is not None and isinstance(context_visibilities, dict):
                    vis = context_visibilities.get(cf)
                    if vis is None:
                        # Try case-insensitive lookup
                        lower_vis = {str(k).lower(): str(v) for k, v in context_visibilities.items()}
                        vis = lower_vis.get(cf.lower())
                    if vis is None:
                        errors.append(f"undeclared_context_field:{cf}")
                    elif vis == VISIBILITY_HIDDEN:
                        errors.append(f"hidden_context_requested:{cf}")
                    elif vis == VISIBILITY_FORBIDDEN:
                        errors.append(f"forbidden_context_requested:{cf}")
                    elif vis != VISIBILITY_VISIBLE:
                        errors.append(f"invalid_context_visibility:{cf}:{vis}")

    # --- production_capabilities validation (§3, §17) ---
    if production_capabilities is not None and isinstance(production_capabilities, list):
        matched_cap = None
        ans_kind = ans.get("kind")
        ans_key = ans.get("key")
        if ans_kind == "field_value" and ans_key:
            for cap in production_capabilities:
                if cap.get("kind") == "field_value" and cap.get("field", "").lower() == ans_key.lower():
                    matched_cap = cap
                    break
            if not matched_cap:
                errors.append(f"ineligible_production_capability:{ans_key}")
        elif ans_kind == "equality":
            keys = ans.get("keys") or []
            base = keys[0][:-2] if keys and (keys[0].endswith("_1") or keys[0].endswith("_2")) else None
            for cap in production_capabilities:
                if cap.get("kind") == "equality" and cap.get("field", "").lower() in (str(base).lower(), "speaker_id", "speakerid"):
                    matched_cap = cap
                    break
            if not matched_cap:
                errors.append(f"ineligible_production_capability:equality:{base}")

        if matched_cap:
            req_ctx = matched_cap.get("visible_context_required", [])
            act_ctx = set(ctx_fields or [])
            for rc in req_ctx:
                if rc not in act_ctx:
                    errors.append(f"missing_required_context_field:{rc}")

    # --- semantic overclaim gate (§17, §18) ---
    name_str = (qtype.get("name") or "").lower()
    goal_str = (qtype.get("goal") or "").lower()
    full_desc = f"{name_str} {goal_str}"
    ans_key = (ans.get("key") or "").lower()

    if ans_key in ("has_tone_error", "has_tone_mispronunciation"):
        for term in ("tone sandhi", "neutralization", "tone merger", "sandhi"):
            if term in full_desc:
                errors.append(f"semantic_overclaim:{term}:{ans_key}")
    if ans_key in ("has_nucleus_error", "has_vowel_error"):
        for term in ("monophthongization", "diphthongization", "vowel shift", "vowel quality shift"):
            if term in full_desc:
                errors.append(f"semantic_overclaim:{term}:{ans_key}")
    if ans_key in ("has_initial_error",):
        for term in ("cluster simplification", "consonant cluster", "retroflex merger", "cluster"):
            if term in full_desc:
                errors.append(f"semantic_overclaim:{term}:{ans_key}")
    if ans_key in ("province", "region"):
        for term in ("birthplace", "current residence", "hometown", "residence", "where the speaker was born", "where the speaker lives"):
            if term in full_desc:
                errors.append(f"semantic_overclaim:{term}:{ans_key}")

    return errors


def _instance_suffix(key: str) -> str | None:
    """Terminal _1/_2 marker, or None for unindexed form."""
    if not isinstance(key, str):
        return None
    m = _INSTANCE_SUFFIX_RE.fullmatch(key)
    return m.group(2) if m else None


def validate_template_text(text: str) -> list[str]:
    """Check the MVP placeholder contract. Empty list == valid.

    [KEY], [CONTEXT:<field_name>], and slot-free text are supported.
    [VALUE] is a known forbidden answer token (answer_leakage).
    """
    if not (text or "").strip():
        return ["empty_template"]
    found = set(_PLACEHOLDER_RE.findall(text))
    if "VALUE" in found:
        return ["answer_leakage"]
    errors = []
    for p in sorted(found):
        if p == "KEY":
            continue
        if p.startswith("CONTEXT:"):
            ctx_field = p.split(":", 1)[1]
            if not ctx_field:
                errors.append(f"invalid_context_placeholder:{p}")
            continue
        errors.append(f"unknown_placeholder:{p}")
    return errors


def validate_question_template(
    template: dict,
    question_type: dict,
    declared_context_fields: list[str] | None = None,
    context_visibilities: dict[str, str] | None = None,
) -> list[str]:
    """Validate a concrete candidate question template against machine contract and context policies (§19)."""
    errors: list[str] = []
    if not isinstance(template, dict):
        return ["invalid_template_format"]

    tid = template.get("template_id")
    if not isinstance(tid, str) or not tid.strip():
        errors.append("missing_template_id")

    text = template.get("text", "")
    text_errors = validate_template_text(text)
    errors.extend(text_errors)

    # Check context placeholders
    matches = _PLACEHOLDER_RE.findall(text)
    declared_set = set(declared_context_fields if declared_context_fields is not None else (question_type.get("input_context_fields") or []))

    ans = question_type.get("answer") or {}
    ans_key = (ans.get("key") or "").lower()

    for p in matches:
        if p.startswith("CONTEXT:"):
            field = p.split(":", 1)[1].strip()
            if declared_context_fields is not None and field not in declared_set:
                errors.append(f"undeclared_context_field:{field}")
            if context_visibilities is not None:
                vis = context_visibilities.get(field)
                if vis is None:
                    lower_vis = {str(k).lower(): str(v) for k, v in context_visibilities.items()}
                    vis = lower_vis.get(field.lower())
                if vis is None:
                    errors.append(f"undeclared_context_field:{field}")
                elif vis == VISIBILITY_HIDDEN:
                    errors.append(f"hidden_context_requested:{field}")
                elif vis == VISIBILITY_FORBIDDEN:
                    errors.append(f"forbidden_context_requested:{field}")
                elif vis != VISIBILITY_VISIBLE:
                    errors.append(f"invalid_context_visibility:{field}:{vis}")
            if field.lower() in ("transcript", "observed_phones", "observed_units"):
                errors.append(f"observed_transcript_leakage:{field}")
            if field.lower() in ("speakerid", "speaker_id", "speaker_name"):
                errors.append(f"id_exposure:{field}")

    # Check that essential declared context fields are not omitted
    if declared_context_fields:
        if "target_sentence" in declared_context_fields and "[CONTEXT:target_sentence]" not in text:
            errors.append("missing_required_context:target_sentence")
        if "target_token" in declared_context_fields and "[CONTEXT:target_token]" not in text:
            errors.append("missing_required_context:target_token")
        if "target_token_ordinal" in declared_context_fields and "[CONTEXT:target_token_ordinal]" not in text:
            errors.append("missing_required_context:target_token_ordinal")

    # Check question type link
    qtype_id = template.get("question_type_id")
    expected_qtype_id = question_type.get("id")
    if qtype_id and expected_qtype_id and qtype_id != expected_qtype_id:
        errors.append(f"type_mismatch:{qtype_id}!={expected_qtype_id}")

    # Speaker ID exposure in text (e.g. "speakerID", "speaker_id")
    if "speakerid" in text.lower() or "speaker_id" in text.lower():
        errors.append("speaker_id_exposure_in_template")

    # Generic Boolean-Template Gate (§10): A boolean field_value question must request a binary judgment
    is_boolean = False
    ans_spec = question_type.get("answer") or {}
    if ans_spec.get("kind") == "field_value":
        key_str = str(ans_spec.get("key") or "")
        if key_str.startswith("has_") or key_str.startswith("is_") or key_str in ("pronunciation_correct",):
            is_boolean = True
        elif question_type.get("allowed_values") in ([True, False], [False, True], ["true", "false"], ["yes", "no"]):
            is_boolean = True
    elif ans_spec.get("kind") == "equality":
        is_boolean = True

    text_lower = text.lower()
    if is_boolean:
        for open_wh in ("là gì", "loại nào", "thành phần nào", "mức độ nào", "như thế nào"):
            if open_wh in text_lower:
                errors.append(f"boolean_template_open_question:{open_wh}")

    # Semantic overclaim checks in template text
    if ans_key in ("has_tone_error", "has_tone_mispronunciation"):
        for term in ("tone sandhi", "biến điệu", "neutralization", "ngữ lưu", "sandhi"):
            if term in text_lower:
                errors.append(f"semantic_overclaim:{term}:{ans_key}")
    if ans_key in ("has_nucleus_error", "has_vowel_error"):
        for term in ("monophthongization", "đơn hóa", "diphthongization", "diphthong reduction", "vowel quality shift"):
            if term in text_lower:
                errors.append(f"semantic_overclaim:{term}:{ans_key}")
    if ans_key in ("has_initial_error",):
        for term in ("cluster simplification", "phụ âm ghép", "retroflex merger", "consonant cluster"):
            if term in text_lower:
                errors.append(f"semantic_overclaim:{term}:{ans_key}")
    if ans_key in ("province", "region"):
        for term in ("quê quán", "nơi sinh", "nơi cư trú", "sinh ra ở đâu", "sinh ra tại", "sinh ra", "sống ở đâu", "đang sống", "sinh sống", "tỉnh hiện đang sống", "hometown", "birthplace", "residence"):
            if term in text_lower:
                errors.append(f"semantic_overclaim:{term}:{ans_key}")


    return errors


def validate_token_occurrence_disambiguation(
    target_sentence: str,
    target_token: str,
    visible_context: dict[str, Any],
    rendered_question: str = "",
) -> list[str]:
    """Verify that if target_token is non-unique in target_sentence, occurrence position is exposed (§7)."""
    errors = []
    if not isinstance(target_sentence, str) or not isinstance(target_token, str):
        return errors
    words = [w.strip(",.?!:;\"'()") for w in target_sentence.lower().split()]
    count = words.count(target_token.lower().strip())
    if count >= 2:
        has_pos = (
            ("target_token_ordinal" in visible_context and visible_context["target_token_ordinal"] is not None)
            or ("target_token_index" in visible_context and visible_context["target_token_index"] is not None)
        )
        has_pos_in_q = False
        if visible_context.get("target_token_ordinal") is not None:
            ord_val = str(visible_context["target_token_ordinal"])
            if ord_val in rendered_question:
                has_pos_in_q = True
        if not has_pos or not has_pos_in_q:
            errors.append("ambiguous_repeated_token_occurrence")
    return errors



def validate_family_base_template(text: str) -> list[str]:
    """Validate a shared operation-family base template skeleton (§33).

    Shared equality-semantic base must:
      - contain exactly one [KEY]
      - contain zero [VALUE]
      - contain no unknown placeholders
      - be non-empty

    Returns a list of errors (empty == valid).
    """
    if not isinstance(text, str) or not text.strip():
        return ["family_template_empty"]
    errors = []
    placeholders = _PLACEHOLDER_RE.findall(text)
    key_count = sum(1 for p in placeholders if p == "KEY")
    value_count = sum(1 for p in placeholders if p == "VALUE")
    unknown = [p for p in placeholders if p not in ("KEY", "VALUE")]

    if key_count == 0:
        errors.append("family_template_missing_key")
    elif key_count > 1:
        errors.append("family_template_multiple_key")

    if value_count > 0:
        errors.append("family_template_forbidden_value")

    for u in unknown:
        errors.append(f"family_template_unknown_placeholder:{u}")

    return errors


def binding_key_for(qtype: dict) -> str | None:
    """Which dataset field fills [KEY] for this type (None if keyless)."""
    ans = qtype.get("answer", {}) or {}
    if ans.get("kind") == "field_value":
        return ans.get("key")
    if ans.get("kind") == "derived_field":
        return ans.get("target_key")
    return None


def get_answer_concept_field(qtype: dict,
                             schema_fields=SCHEMA_FIELDS) -> str | None:
    """Raw schema field the [KEY] concept asks about (None if keyless).

    NO normalization for direct/derived (§15/§16): the key is returned
    AS-IS so structural invalidity (e.g. province_1) stays visible until
    the gate rejects it.

    A. field_value{key: X}       -> X as-is
    B. equality{keys: [F_1,F_2]} -> validated common base F (else None)
    C. derived_field{target Y}   -> Y as-is
    Other kinds (unknown) -> None.
    """
    ans = qtype.get("answer", {}) or {}
    kind = ans.get("kind")
    if kind == "field_value":
        return ans.get("key")
    if kind == "equality":
        keys = ans.get("keys") or []
        if not isinstance(keys, list) or len(keys) != 2:
            return None
        left = (parse_pair_field_reference(keys[0], schema_fields)
                if isinstance(keys[0], str) else None)
        right = (parse_pair_field_reference(keys[1], schema_fields)
                 if isinstance(keys[1], str) else None)
        if left is None or right is None:
            return None
        if left[1] != 1 or right[1] != 2:
            return None
        if left[0].lower() != right[0].lower():
            return None
        return left[0]
    if kind == "derived_field":
        return ans.get("target_key")
    return None


def key_required_type_ids(templates: list[dict]) -> set[str]:
    """Type IDs owning at least one produced template containing [KEY].

    Derived from actual produced templates, not from the mapping.
    """
    required: set[str] = set()
    for tpl in templates or []:
        if "[KEY]" in (tpl.get("text", "") or ""):
            owner = tpl.get("question_type_id")
            if owner is not None:
                required.add(owner)
    return required


def validate_type_key_realizations(
    mapping: dict,
    current_type_ids,
    templates: list[dict] | None = None,
) -> list[str]:
    """Validate the LLM-facing type-keyed key_realizations map.

    LLM keys MUST be exact current-round question type IDs (never raw
    dataset field names, never "KEY", never Vietnamese phrases).
    Values MUST be non-empty natural Vietnamese phrases.
    Every [KEY]-requiring type MUST have a phrase (missing fails);
    known-but-unused extra phrases are harmless and ignored downstream
    (no error here).
    """
    if not isinstance(mapping, dict):
        return ["invalid_key_realizations:not_a_mapping"]
    known = set(current_type_ids or [])
    errors: list[str] = []
    for k, v in mapping.items():
        if k not in known:
            errors.append(f"unknown_key_realization_type:{k}")
        elif not isinstance(v, str) or not v.strip():
            errors.append(f"empty_key_realization:{k}")
    required = key_required_type_ids(templates or [])
    for tid in sorted(required & known):
        if tid not in mapping:
            errors.append(f"missing_key_realization_for_type:{tid}")
    return errors


def validate_key_realizations(mapping, *args, **kwargs) -> list[str]:
    """New-contract alias: LLM keys are current question type IDs.

    Old raw-field-keyed interpretation is deleted (§30/§62): there is ONE
    contract only. Callers MUST pass the current round type IDs (and, for
    the missing-required check, the produced templates)::

        validate_key_realizations(mapping, type_ids, templates)
    """
    if not args and "type_ids" not in kwargs and "current_type_ids" not in kwargs:
        raise TypeError(
            "validate_key_realizations now requires current round type IDs: "
            "validate_key_realizations(mapping, type_ids, templates). "
            "Raw-field-keyed validation is deleted.")
    type_ids = None
    templates = None
    if args:
        type_ids = args[0]
    if len(args) > 1:
        templates = args[1]
    if "type_ids" in kwargs:
        type_ids = kwargs.pop("type_ids")
    if "current_type_ids" in kwargs:
        type_ids = kwargs.pop("current_type_ids")
    if "templates" in kwargs:
        templates = kwargs.pop("templates")
    if kwargs:
        raise TypeError(f"Unexpected arguments: {sorted(kwargs)}")
    return validate_type_key_realizations(mapping, type_ids, templates)


def build_round_field_candidates(
    types_in_style_order: list[dict],
    valid_templates: list[dict],
    valid_type_ids,
    type_key_realizations: dict | None,
    approved_key_realizations: dict | None,
    schema_fields=None,
) -> dict:
    """Build round canonical field candidates AFTER gate (§17/§18).

    Uses ONLY valid types with at least one surviving [KEY] template.
    Iterates in ORIGINAL Style order; first surviving type wins a shared
    field; approved fields are never overwritten. Returns
    {raw_field: phrase}.
    """
    if schema_fields is None:
        schema_fields = SCHEMA_FIELDS
    valid_ids = set(valid_type_ids or [])
    phrases = dict(type_key_realizations or {})
    approved = dict(approved_key_realizations or {})
    keyed_owners: set[str] = set()
    for tpl in valid_templates or []:
        if ("[KEY]" in (tpl.get("text", "") or "")
                or tpl.get("base_template_id")
                or str(tpl.get("template_id", "")).startswith("BT_")):
            owner = tpl.get("question_type_id")
            if owner in valid_ids:
                keyed_owners.add(owner)
    candidates: dict[str, str] = {}
    for qtype in types_in_style_order or []:
        tid = qtype.get("id")
        if tid not in valid_ids or tid not in keyed_owners:
            continue
        field = get_answer_concept_field(qtype, schema_fields=schema_fields)
        if field is None or field in approved or field in candidates:
            continue
        phrase = phrases.get(tid)
        if not isinstance(phrase, str) or not phrase.strip():
            continue
        candidates[field] = phrase
    return candidates


def gate_round(
    types: list[dict],
    templates: list[dict],
    key_context: dict | None = None,
    type_key_realizations: dict | None = None,
    approved_key_realizations: dict | None = None,
    dataset_rows=None,
    relation_cache=None,
    schema_fields=None,
    hidden_fields=None,
    field_roles=None,
) -> dict:
    """Deterministic pre-LLM gate. Returns valid subsets + auto-rejections.

    [KEY] bindability is resolved PER OWNER TYPE (§15):

    1. concept field in approved_key_realizations -> bindable;
    2. else owner type ID in type_key_realizations -> bindable;
    3. else -> unbound_key drop.

    When NEITHER map is provided (both None), the gate is lenient: a [KEY]
    template survives whenever its owner has a concept field. This covers
    local callers with no realization info. When either map is provided
    (even empty), strict per-type resolution applies.

    (key_context is accepted for backward compatibility and treated as an
    approved field-level map; new code SHOULD pass the two named maps.)

    Base-template exact dedup is scoped per (question_type_id, normalized
    text). Returns valid_types / valid_templates / auto_rejected /
    dropped_templates.
    """
    if schema_fields is None:
        schema_fields = SCHEMA_FIELDS
    if hidden_fields is None:
        hidden_fields = ID_LIKE_FIELDS

    if approved_key_realizations is None and key_context is not None:
        approved_key_realizations = dict(key_context)
    strict = not (type_key_realizations is None
                  and approved_key_realizations is None)
    type_phrases = dict(type_key_realizations or {})
    approved = dict(approved_key_realizations or {})
    auto_rejected: list[dict] = []
    dropped_templates: list[dict] = []
    valid_types: list[dict] = []
    valid_templates: list[dict] = []
    # Per-type exact collapse first (order-preserving).
    seen: dict[tuple[str | None, str], str] = {}
    unique: list[dict] = []
    for tpl in templates:
        key = (tpl.get("question_type_id"),
               normalize_question_text(tpl.get("text", "")))
        if key in seen:
            dropped_templates.append(
                {"template_id": tpl.get("template_id"),
                 "type_id": tpl.get("question_type_id"),
                 "reason": f"exact_duplicate:{seen[key]}"})
            continue
        seen[key] = tpl.get("template_id")
        unique.append(tpl)
    templates = unique

    type_ids = {t.get("id") for t in types}
    bad_type_ids: set[str] = set()
    for t in types:
        errs = list(dict.fromkeys(validate_question_type(
            t, known_fields=schema_fields, dataset_rows=dataset_rows,
            relation_cache=relation_cache, hidden_fields=hidden_fields,
            field_roles=field_roles)))
        if errs:
            bad_type_ids.add(t.get("id", "?"))
            auto_rejected.append(
                {"type_id": t.get("id"), "reason": ";".join(errs)})

    templates_by_type: dict[str, list[dict]] = {}
    for tpl in templates:
        tid, owner = tpl.get("template_id"), tpl.get("question_type_id")
        if owner not in type_ids:
            dropped_templates.append(
                {"template_id": tid, "type_id": owner,
                 "reason": "unknown_type"})
            continue
        errs = validate_template_text(tpl.get("text", ""))
        if errs:
            dropped_templates.append(
                {"template_id": tid, "type_id": owner,
                 "reason": ";".join(errs)})
            continue
        templates_by_type.setdefault(owner, []).append(tpl)

    for t in types:
        tid = t.get("id")
        if tid in bad_type_ids:
            continue
        kept = templates_by_type.get(tid, [])
        # [KEY] bindability is resolved PER OWNER TYPE (§15): approved
        # field binding first, else the owner's own type phrase, else
        # unbound_key. (get_answer_concept_field covers equality concepts
        # like speakerID that binding_key_for leaves keyless.)
        concept = get_answer_concept_field(t, schema_fields=schema_fields)
        survivors = []
        for tpl in kept:
            if "[KEY]" not in tpl.get("text", ""):
                survivors.append(tpl)
                continue
            if concept is None:
                dropped_templates.append(
                    {"template_id": tpl.get("template_id"), "type_id": tid,
                     "reason": "unbound_key"})
                continue
            if not strict:
                # No realization info: keep [KEY] only for directly
                # keyed types (legacy standalone behavior).
                if binding_key_for(t) is not None:
                    survivors.append(tpl)
                    continue
                dropped_templates.append(
                    {"template_id": tpl.get("template_id"), "type_id": tid,
                     "reason": "unbound_key"})
                continue
            if concept in approved or tid in type_phrases:
                survivors.append(tpl)
                continue
            dropped_templates.append(
                {"template_id": tpl.get("template_id"), "type_id": tid,
                 "reason": "unbound_key"})
        if not survivors:
            auto_rejected.append(
                {"type_id": tid, "reason": "no_renderable_template"})
            continue
        valid_types.append(t)
        valid_templates.extend(survivors)

    return {
        "valid_types": valid_types,
        "valid_templates": valid_templates,
        "auto_rejected": auto_rejected,
        "dropped_templates": dropped_templates,
    }

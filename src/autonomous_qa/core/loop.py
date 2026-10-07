"""v3 discovery loop: pure-Python controller + approved-type pool helpers.

LLM proposes (Style/Template/Paraphrase) and judges semantics (Quality).
Python owns everything countable: validity, rates, stop decisions.

Saturation is measured on question-TYPE discovery, never on wording:
  new_type_rate = accepted_new_types / generated_types

Approved pool item:
  {type_id, name, goal, uses, input_count, answer_rule, answer,
   round_accepted, kept_templates: [{template_id, text}],
   bindings: [{question_type_id, key, template_id}]}
"""

from __future__ import annotations

from src.autonomous_qa.authoring.llm_client import StageOutputValidationError
from src.autonomous_qa.core.validity import (
    binding_key_for,
    get_answer_concept_field,
    normalize_question_text,
    template_family_signature,
    validate_template_text,
)


def normalize_template(text: str) -> str:
    """Legacy alias of normalize_question_text (kept for compatibility)."""
    return normalize_question_text(text)


def split_exact_duplicates(
    candidates: list[dict], pool: list[dict]
) -> tuple[list[dict], int]:
    """Drop normalized-exact duplicate texts (keep first occurrence).

    Used only to collapse identical template strings inside one round's
    Template output so ids stay unambiguous. Never used for saturation.
    """
    seen = {normalize_template(item["text"]) for item in pool}
    unique: list[dict] = []
    exact = 0
    for cand in candidates:
        key = normalize_template(cand["text"])
        if key in seen:
            exact += 1
            continue
        seen.add(key)
        unique.append(cand)
    return unique, exact


def filter_paraphrases_for_judging(valid_templates: list[dict],
                                   paraphrases: list[dict]) -> dict:
    """Local deterministic filter AFTER Paraphrase, BEFORE Quality.

    Per question type (never global — the same generic text may serve
    different types):
    - drop paraphrases leaking [VALUE] (reason answer_leakage) or
      carrying other placeholder violations;
    - drop normalized-exact duplicates (keep base template, else first
      paraphrase; never let the LLM choose).

    Returns {"surviving": [...], "dropped_leak": [...],
             "dropped_exact": [...], "owners": {id: type_id},
             "texts": {id: text}} covering surviving paraphrases only.
    Dropped entries: {"template_id", "duplicate_of"|"reason", "type_id"}.
    Orphans (unknown source template) are dropped silently as before.
    """
    base_by_type: dict[str, list[dict]] = {}
    for tpl in valid_templates:
        base_by_type.setdefault(tpl["question_type_id"], []).append(tpl)
    owner_of_source = {t["template_id"]: t["question_type_id"]
                       for t in valid_templates}

    surviving: list[dict] = []
    dropped_leak: list[dict] = []
    dropped_exact: list[dict] = []
    owners: dict[str, str] = {}
    texts: dict[str, str] = {}
    for type_id, bases in base_by_type.items():
        seen: dict[str, str] = {}
        for b in bases:
            seen.setdefault(normalize_question_text(b["text"]),
                            b["template_id"])
        for p in paraphrases:
            if p.get("source_template_id") not in {
                    b["template_id"] for b in bases}:
                continue  # belongs to another type (or orphan): skip here
            pid = p.get("template_id")
            verdicts = validate_template_text(p.get("text", ""))
            if verdicts:
                reason = ("answer_leakage" if "[VALUE]" in p.get("text", "")
                          else ";".join(verdicts))
                dropped_leak.append(
                    {"template_id": pid, "type_id": type_id,
                     "reason": reason})
                continue
            sig = normalize_question_text(p.get("text", ""))
            if sig in seen:
                dropped_exact.append(
                    {"template_id": pid, "duplicate_of": seen[sig],
                     "type_id": type_id})
                continue
            seen[sig] = pid
            surviving.append(p)
            owners[pid] = type_id
            texts[pid] = p.get("text", "")
    return {"surviving": surviving, "dropped_leak": dropped_leak,
            "dropped_exact": dropped_exact, "owners": owners, "texts": texts}


class LoopController:
    """Saturation controller over per-round (accepted, generated) counts."""

    def __init__(
        self,
        *,
        max_rounds: int,
        min_rate: float,
        saturation_patience: int,
        immediate_stop_if_zero_new: bool,
        zero_reason: str = "zero_new_types",
        saturated_reason: str = "type_diversity_saturated",
        max_reason: str = "max_rounds",
    ):
        self.max_rounds = max_rounds
        self.min_rate = min_rate
        self.saturation_patience = saturation_patience
        self.immediate_stop_if_zero_new = immediate_stop_if_zero_new
        self.zero_reason = zero_reason
        self.saturated_reason = saturated_reason
        self.max_reason = max_reason
        self.low_gain_streak = 0

    def register_round(
        self, round_idx: int, accepted_new: int, generated: int
    ) -> dict:
        new_type_rate = accepted_new / generated if generated > 0 else 0.0

        if self.immediate_stop_if_zero_new and accepted_new == 0:
            return {
                "stop": True,
                "stop_reason": self.zero_reason,
                "new_type_rate": new_type_rate,
                "low_gain_streak": self.low_gain_streak,
            }

        if new_type_rate < self.min_rate:
            self.low_gain_streak += 1
        else:
            self.low_gain_streak = 0

        if self.low_gain_streak >= self.saturation_patience:
            return {
                "stop": True,
                "stop_reason": self.saturated_reason,
                "new_type_rate": new_type_rate,
                "low_gain_streak": self.low_gain_streak,
            }

        if round_idx >= self.max_rounds:
            return {
                "stop": True,
                "stop_reason": self.max_reason,
                "new_type_rate": new_type_rate,
                "low_gain_streak": self.low_gain_streak,
            }

        return {
            "stop": False,
            "stop_reason": None,
            "new_type_rate": new_type_rate,
            "low_gain_streak": self.low_gain_streak,
        }


def build_type_fewshot(approved_pool: list[dict]) -> list[dict]:
    """Dynamic few-shots: only what Style needs (structure -> type)."""
    return [
        {
            "name": t["name"],
            "uses": list(t["uses"]),
            "goal": t["goal"],
            "answer_rule": t["answer_rule"],
        }
        for t in approved_pool
    ]


def validate_quality_partition(verdict: dict, valid_type_ids: set[str],
                               approved_type_ids: set[str]) -> dict:
    """Fail closed unless the verdict partitions V exactly.

    V = valid current-round type ids actually sent to Quality.
    P = previously approved type ids (duplicate_of targets).
    A/D/R = accepted/duplicates/rejected type-id sets.
    Requires A∩D=A∩R=D∩R=∅ and A∪D∪R==V. Returns
    {"accepted": [...], "duplicates": [...], "rejected": [...]} as sets
    (sorted lists) for downstream accounting.
    """
    v = set(valid_type_ids)
    p = set(approved_type_ids)
    acc = [(item.get("type_id"), item) for item in
           verdict.get("accepted_new", [])]
    dup = [(item.get("type_id"), item) for item in
           verdict.get("duplicates", [])]
    rej = [(item.get("type_id"), item) for item in
           verdict.get("rejected", [])]
    for tid, _ in acc + dup + rej:
        if tid not in v:
            raise StageOutputValidationError(
                f"Quality verdict references type {tid!r} outside the "
                "valid types sent to Quality: "
                "quality_unknown_type_id:" + str(tid))
    for item in verdict.get("duplicates", []):
        tid, target = item.get("type_id"), item.get("duplicate_of")
        if target not in p:
            raise StageOutputValidationError(
                f"Quality duplicate target {target!r} for type {tid!r} "
                "is not a previously approved type: "
                "quality_unknown_duplicate_target:" + str(target))
    a, d, r = ({t for t, _ in acc}, {t for t, _ in dup},
               {t for t, _ in rej})
    for overlap in (a & d, a & r, d & r):
        if overlap:
            tid = sorted(overlap)[0]
            raise StageOutputValidationError(
                f"Quality verdict classifies type {tid!r} twice: "
                "quality_overlapping_verdict:" + str(tid))
    if a | d | r != v:
        missing = sorted(v - (a | d | r))[0]
        raise StageOutputValidationError(
            f"Quality verdict omits valid type {missing!r}: "
            "quality_missing_verdict:" + str(missing))
    return {"accepted": sorted(a), "duplicates": sorted(d),
            "rejected": sorted(r)}


def apply_accepted_types(
    approved_pool: list[dict],
    round_types_by_id: dict[str, dict],
    texts_by_id: dict[str, str],
    verdict: dict,
    round_idx: int,
    owners: dict[str, str],
) -> list[dict]:
    """Append Quality-accepted types with kept templates + bindings.

    Fail-closed validation of every keep_template_ids entry (no silent
    repair of Gemini output):
    - keep list must be non-empty (else the type belongs in rejected);
    - no duplicate keep ids;
    - every keep id must exist among surviving candidates;
    - every keep id must belong to exactly this type (no resurrected
      dropped/dedup-removed ids, no foreign ids, no ghosts).
    Unknown type ids in the verdict fail closed as well.
    Returns accepted items.
    """
    for item in verdict.get("accepted_new", []):
        tid = item.get("type_id")
        keep = item.get("keep_template_ids", [])
        if tid not in round_types_by_id:
            raise StageOutputValidationError(
                f"Quality accepted unknown type {tid!r}; it is not part "
                "of this round's valid type bundle.")
        if not keep:
            raise StageOutputValidationError(
                f"quality_no_kept_template:{tid}: accepted with an empty "
                "keep_template_ids; it belongs in rejected.")
        if len(set(keep)) != len(keep):
            dup = sorted({k for k in keep if keep.count(k) > 1})[0]
            raise StageOutputValidationError(
                f"quality_duplicate_keep_id:{dup}: keep list for type "
                f"{tid!r} contains duplicates.")
        for kept_id in keep:
            if kept_id not in texts_by_id:
                raise StageOutputValidationError(
                    f"quality_unknown_template_id:{kept_id}: keep id for "
                    f"type {tid!r} is unknown (dropped, dedup-removed, "
                    "or nonexistent).")
            if owners.get(kept_id) != tid:
                raise StageOutputValidationError(
                    f"quality_foreign_template_id:{kept_id}: keep id does "
                    f"not belong to type {tid!r}.")
    pool_ids = {t["type_id"] for t in approved_pool}
    accepted: list[dict] = []
    for item in verdict.get("accepted_new", []):
        tid = item.get("type_id")
        if tid in pool_ids:
            continue
        qtype = round_types_by_id.get(tid)
        if qtype is None:
            continue  # unreachable after strict validation; defensive.
        kept = []
        for kept_id in item.get("keep_template_ids", []):
            text = texts_by_id.get(kept_id)
            if text is not None:
                kept.append({"template_id": kept_id, "text": text})
        if not kept:
            continue
        key = binding_key_for(qtype)
        entry = {
            "type_id": tid,
            "name": qtype.get("name", ""),
            "goal": qtype.get("goal", ""),
            "uses": list(qtype.get("uses", [])),
            "input_count": qtype.get("input_count", 1),
            "answer_rule": qtype.get("answer_rule", ""),
            "answer": dict(qtype.get("answer", {})),
            "round_accepted": round_idx,
            "kept_templates": kept,
            "bindings": [
                {"question_type_id": tid, "key": key,
                 "template_id": k["template_id"]}
                for k in kept
            ],
        }
        approved_pool.append(entry)
        pool_ids.add(tid)
        accepted.append(entry)
    return accepted


def flatten_type_bindings(approved_pool: list[dict]) -> list[dict]:
    """Approved pool -> renderer-grade bindings."""
    flat: list[dict] = []
    for t in approved_pool:
        for b in t.get("bindings", []):
            flat.append(dict(b))
    return flat


def build_round_stats(
    *,
    round_idx: int,
    generated_types: int,
    accepted_new_types: int,
    duplicates: int,
    rejected: int,
    new_type_rate: float,
    total_approved_types: int,
    deterministic_duplicates: int = 0,
) -> dict:
    """Per-round accounting. `duplicates` counts Quality semantic
    duplicates; `deterministic_duplicates` counts exact executable-
    signature duplicates removed before Quality (§33)."""
    return {
        "round": round_idx,
        "generated_types": generated_types,
        "accepted_new_types": accepted_new_types,
        "duplicates": duplicates,
        "deterministic_duplicates": deterministic_duplicates,
        "rejected": rejected,
        "new_type_rate": round(new_type_rate, 4),
        "total_approved_types": total_approved_types,
    }


def resolve_effective_binding(field, approved, candidate):
    """Effective realization for a raw field: approved first, candidate next.

    An already-approved canonical binding always wins; the current-round
    candidate is provisional only. Returns None when neither has it.
    """
    if field is None:
        return None
    if field in approved:
        return approved[field]
    return candidate.get(field)


def commit_approved_bindings(approved, candidate, accepted_items,
                             round_types_by_id, type_order=None):
    """Commit bindings for accepted types with kept [KEY] templates.

    candidate is the canonical ROUND FIELD map (field -> phrase) that
    Quality actually judged — never a type-private phrase. Iteration
    follows ORIGINAL current-round Style order (type_order, else the
    given accepted order). For every accepted item with a kept [KEY]
    template: resolve the raw concept field and commit the canonical
    round candidate unless already approved. Returns
    {"committed": [fields...],
     "ignored": [{field, approved_value, proposed_value}...]}.
    Rejected/duplicate/auto-rejected types, omitted templates, and
    slot-free-only types commit nothing.
    """
    committed = []
    ignored = []
    if type_order is not None:
        rank = {tid: i for i, tid in enumerate(type_order)}
        accepted_items = sorted(
            accepted_items,
            key=lambda it: rank.get(it.get("type_id"), len(rank)))
    for item in accepted_items:
        qtype = round_types_by_id.get(item.get("type_id"), {})
        field = get_answer_concept_field(qtype)
        if field is None:
            continue
        needs_key = (
            any("[KEY]" in (k.get("text", "") or "")
                for k in item.get("kept_templates", []))
            or any(str(k.get("template_id", "")).startswith("BT_")
                   for k in item.get("kept_templates", []))
            or (template_family_signature(qtype) is not None)
        )
        if not needs_key:
            continue
        if field in approved:
            proposed = candidate.get(field)
            entry = {"field": field, "approved_value": approved[field],
                     "proposed_value": proposed}
            if (proposed is not None and proposed != approved[field]
                    and entry not in ignored):
                ignored.append(entry)
            continue
        value = candidate.get(field)
        if value is not None and field not in committed:
            approved[field] = value
            committed.append(field)
    return {"committed": committed, "ignored": ignored}

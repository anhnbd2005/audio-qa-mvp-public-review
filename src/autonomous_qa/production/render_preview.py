"""Python-only QA preview renderer. ZERO LLM calls.

Question/answer separation is absolute:
- Question text may contain [KEY] (concept name) or no placeholder.
  [VALUE] is FORBIDDEN in question text — never substituted, hard fail.
- Gold comes ONLY from the structured `answer` object (field_value,
  equality -> "Có"/"Không", derived_field).

Selection priority: (1) one QA per approved type, (2) missing opposite
boolean class for pairwise (equality-kind) types, (3) ordinary fill.
Cap stays at n (default 10).

Any unresolvable placeholder ([FOO] outside the [KEY] contract,
missing key realization, ...) is a HARD failure — never silent output.
"""

from __future__ import annotations

import random
import re

from src.autonomous_qa.core.validity import (
    SCHEMA_FIELDS,
    binding_key_for,
    get_answer_concept_field,
    parse_pair_field_reference,
    required_answer_fields,
)


def _binding_key(t: dict, schema_fields=None):
    """Render-time key: target concept for derived, else answer key.

    [KEY] ALWAYS refers to the target_key for derived_field (never the
    source): the question asks for the target value. Equality types are
    keyless in `binding_key_for`, but a [KEY] template on such a type
    (e.g. speakerID realization "same speaker/person") binds the
    answer-concept field instead of exposing any value.
    """
    ans = (t.get("answer") or {})
    if ans.get("kind") == "derived_field":
        return get_answer_concept_field(t, schema_fields=schema_fields)
    key = binding_key_for(t)
    if key is None:
        key = get_answer_concept_field(t, schema_fields=schema_fields)
    return key

_PLACEHOLDER_RE = re.compile(r"\[([A-Za-z_][A-Za-z0-9_]*(?::[A-Za-z_][A-Za-z0-9_]*)?)\]")
_ALLOWED_PLACEHOLDERS = frozenset({"KEY"})


def bind_context_placeholders(
    template_text: str,
    row: dict,
    declared_context_fields: list[str] | None = None,
    context_visibilities: dict[str, str] | None = None,
) -> str:
    """Deterministically bind [CONTEXT:<field>] placeholders from row evidence (§12, §13).

    Fails closed:
    - Rejects undeclared context fields
    - Rejects hidden or forbidden context fields
    - Rejects null or missing values
    - Guarantees zero unresolved [CONTEXT:*] placeholders remain
    """
    if not isinstance(template_text, str) or not template_text:
        return template_text or ""

    matches = _PLACEHOLDER_RE.findall(template_text)
    declared_set = set(declared_context_fields or [])

    # Context Gate: check row validity (§20)
    if "context_valid" in row and row["context_valid"] is False:
        raise ValueError("Context Gate rejection: row has context_valid == False (e.g. non-1:1 mapping).")

    rendered = template_text
    for p in matches:
        if not p.startswith("CONTEXT:"):
            continue
        field = p.split(":", 1)[1].strip()
        if not field:
            raise ValueError(f"Empty context placeholder name in template: {p!r}")

        if declared_context_fields is not None and field not in declared_set:
            raise ValueError(
                f"Undeclared context placeholder [CONTEXT:{field}] used in template. "
                f"Declared context fields: {sorted(list(declared_set))}"
            )

        if context_visibilities is not None:
            vis = context_visibilities.get(field)
            if vis != "visible":
                raise ValueError(
                    f"Context field '{field}' has visibility '{vis}' (not 'visible') "
                    f"and cannot be bound into question text."
                )

        if field not in row:
            raise ValueError(f"Required context field '{field}' missing in row.")
        val = row.get(field)
        if val is None:
            raise ValueError(f"Required context field '{field}' has null value in row.")

        if field == "target_token_index":
            try:
                idx = int(val)
                if idx < 0:
                    raise ValueError(f"target_token_index invalid: {val} < 0")
            except (ValueError, TypeError):
                raise ValueError(f"target_token_index invalid: {val}")

        if field == "target_token_ordinal":
            try:
                ord_val = int(val)
                if ord_val < 1:
                    raise ValueError(f"target_token_ordinal invalid: {val} < 1")
            except (ValueError, TypeError):
                raise ValueError(f"target_token_ordinal invalid: {val}")


        val_str = str(val).strip()
        rendered = rendered.replace(f"[{p}]", val_str)

    remaining = [p for p in _PLACEHOLDER_RE.findall(rendered) if p.startswith("CONTEXT:")]
    if remaining:
        raise ValueError(f"Unresolved context placeholders remain after binding: {remaining}")

    return rendered



def _build_derived_map(rows: list[dict], source_key: str, target_key: str) -> dict:
    mapping: dict[str, str] = {}
    for row in rows:
        s, t = row.get(source_key), row.get(target_key)
        if s is None or t is None:
            continue
        if s in mapping and mapping[s] != t:
            raise ValueError(
                f"Non-deterministic hierarchy: {source_key}={s!r} maps to both "
                f"{mapping[s]!r} and {t!r}. Refusing to render derived answers."
            )
        mapping[s] = t
    return mapping


def _equality_base_key(qtype: dict, schema_fields=None) -> str:
    """Validated canonical equality base field (§24/§25).

    Requires exactly [F_1, F_2] for the same raw base F. Never strips
    suffixes generically, never reorders, never falls back to row
    sniffing. Raises a controlled ValueError (never raw KeyError) when
    the spec is not canonical.
    """
    keys = (qtype.get("answer", {}) or {}).get("keys") or []
    if not isinstance(keys, list) or len(keys) != 2:
        single_key = (qtype.get("answer", {}) or {}).get("key")
        if isinstance(single_key, str) and single_key:
            return single_key
        raise ValueError(
            f"Equality type {qtype.get('type_id')} has non-canonical keys: "
            "expected exactly [F_1, F_2].")
    left = (parse_pair_field_reference(keys[0], schema_fields)
            if isinstance(keys[0], str) else None)
    right = (parse_pair_field_reference(keys[1], schema_fields)
             if isinstance(keys[1], str) else None)
    if left is None or right is None:
        raise ValueError(
            f"Equality type {qtype.get('type_id')} has non-canonical pair "
            "references: expected [F_1, F_2] with F in schema.")
    if left[1] != 1 or right[1] != 2:
        raise ValueError(
            f"Equality type {qtype.get('type_id')} violates canonical "
            "instance order: expected [F_1, F_2].")
    if left[0].lower() != right[0].lower():
        raise ValueError(
            f"Equality type {qtype.get('type_id')} mixes base fields "
            f"{left[0]!r} and {right[0]!r}.")
    return left[0]


def _gold_answer(qtype: dict, row: dict, row2: dict | None = None,
                 rows: list[dict] | None = None, schema_fields=None,
                 derived_mappings=None) -> tuple[str, dict]:
    """Deterministic gold + provenance. row2 only for pairwise types.

    Field identity is EXACT: field_value/derived keys are literal row
    fields (never normalized); equality resolves virtual F_1/F_2 to
    row1[F]/row2[F]. Impossible access raises a controlled
    render_missing_answer_field invariant, never a raw KeyError.
    """
    ans = qtype.get("answer", {}) or {}
    kind = ans.get("kind")
    if kind == "field_value":
        field = ans.get("key")
        if not isinstance(field, str) or field not in row or row[field] is None or not str(row[field]).strip():
            raise ValueError(f"render_missing_answer_field:{field}")
        return str(row[field]), {"kind": "field_value", "field": field}
    if kind == "derived_field":
        src_k, tgt_k = ans.get("source_key"), ans.get("target_key")
        if not isinstance(src_k, str) or src_k not in row or row[src_k] is None or not str(row[src_k]).strip():
            raise ValueError(f"render_missing_answer_field:{src_k}")
        if not any(isinstance(r, dict) and r.get(tgt_k) is not None and str(r.get(tgt_k)).strip()
                   for r in rows or []):
            raise ValueError(f"render_missing_answer_field:{tgt_k}")
        mapping = None
        if isinstance(derived_mappings, dict):
            mapping = derived_mappings.get((src_k, tgt_k))
        if mapping is None:
            mapping = _build_derived_map(rows, src_k, tgt_k)
        if row[src_k] not in mapping or mapping[row[src_k]] is None:
            raise ValueError(
                f"render_missing_derived_mapping:{row[src_k]}")
        expected = mapping[row[src_k]]
        if (tgt_k in row and row[tgt_k] is not None
                and row[tgt_k] != expected):
            raise ValueError(
                f"render_derived_target_mismatch:{src_k}={row[src_k]!r}:"
                f"expected={expected!r}:row={row[tgt_k]!r}")
        return str(expected), {
            "kind": "derived_field",
            "source_field": src_k, "target_field": tgt_k}
    if kind == "equality":
        if row2 is None:
            raise ValueError(
                f"Equality type {qtype.get('type_id')} needs two rows.")
        base_key = _equality_base_key(qtype, schema_fields)
        if (base_key not in row or base_key not in row2
                or row[base_key] is None or row2[base_key] is None
                or not str(row[base_key]).strip() or not str(row2[base_key]).strip()):
            raise ValueError(
                f"render_missing_answer_field:{base_key}")
        same = row[base_key] == row2[base_key]
        return ("Có" if same else "Không"), {
            "kind": "equality", "field": base_key,
            "row_1": row.get("audio"), "row_2": row2.get("audio")}
    raise ValueError(f"Unsupported answer kind: {kind!r}")


def render_question(template_text: str, template_id: str, key: str | None,
                    key_realizations: dict, gold: str) -> tuple[str, str | None]:
    """Substitute [KEY] only. [VALUE] in question text is a hard failure.

    `gold` is accepted (callers already resolve it) but MUST NEVER be
    written into the question. There is no code path that substitutes an
    actual gold answer into question text.
    """
    found = set(_PLACEHOLDER_RE.findall(template_text))
    if "VALUE" in found:
        raise ValueError(
            f"Template {template_id} leaks the gold answer ([VALUE] in "
            f"question text): {template_text!r}")
    unknown = found - _ALLOWED_PLACEHOLDERS
    if unknown:
        raise ValueError(
            f"Template {template_id} has unresolved placeholders "
            f"{sorted(unknown)}: {template_text!r}")
    key_realization: str | None = None
    question = template_text
    if "KEY" in found:
        if not key:
            raise ValueError(
                f"Template {template_id} needs [KEY] but its type has no "
                "bindable key.")
        if key not in key_realizations:
            raise ValueError(
                f"Template {template_id} needs realization for key "
                f"{key!r}, none was inferred.")
        key_realization = key_realizations[key]
        question = question.replace("[KEY]", key_realization)
    if _PLACEHOLDER_RE.search(question) or "[" in question or "]" in question:
        raise ValueError(
            f"Template {template_id} left an unresolved placeholder: "
            f"{question!r}")
    return question, key_realization


def _speaker_key_for_row(row: dict) -> str | None:
    for k in ("speakerID", "speaker_id", "speakerid", "speaker"):
        if k in row and row[k] is not None:
            return str(row[k])
    return None


def _row_pairs(rows: list[dict], rng: random.Random,
               ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Deterministic (same-speaker, different-speaker) index pairs."""
    order = list(range(len(rows)))
    rng.shuffle(order)
    same, diff = [], []
    for a in range(len(order)):
        for b in range(a + 1, len(order)):
            i, j = order[a], order[b]
            if rows[i].get("audio") and rows[i].get("audio") == rows[j].get("audio"):
                continue
            spk_i = _speaker_key_for_row(rows[i])
            spk_j = _speaker_key_for_row(rows[j])
            if spk_i is not None and spk_j is not None and spk_i == spk_j:
                same.append((i, j))
            else:
                diff.append((i, j))
    return same, diff


def _is_pairwise(qtype: dict) -> bool:
    """Pairwise boolean type = structured answer.kind == "equality".

    Never inferred from the type name.
    """
    return (qtype.get("answer") or {}).get("kind") == "equality"

def _find_eligible_row(
    rows: list[dict],
    qtype: dict,
    row_order: list[int] | None = None,
    offset: int = 0,
    schema_fields=None,
    derived_mappings=None,
) -> dict | None:
    if row_order is None:
        row_order = list(range(len(rows)))
    if not rows or not row_order:
        return None
    req = required_answer_fields(qtype, schema_fields=schema_fields)
    for step in range(len(rows)):
        idx = row_order[(offset + step) % len(row_order)]
        r = rows[idx]
        if any(r.get(f) is None or not str(r.get(f)).strip() for f in req):
            continue
        try:
            _gold_answer(qtype, r, None, rows, schema_fields, derived_mappings)
            return r
        except (ValueError, KeyError):
            continue
    return None


def _find_pair_for_stream(
    rows: list[dict],
    qtype: dict,
    all_pairs: list[tuple[int, int]] | None = None,
    want_match: bool | None = None,
    schema_fields=None,
    derived_mappings=None,
) -> tuple[int, int] | None:
    if all_pairs is None:
        all_pairs = [(i, j) for i in range(len(rows)) for j in range(i + 1, len(rows))]
    req = required_answer_fields(qtype, schema_fields=schema_fields)
    for i, j in all_pairs:
        if any(rows[i].get(f) is None or not str(rows[i].get(f)).strip()
               or rows[j].get(f) is None or not str(rows[j].get(f)).strip()
               for f in req):
            continue
        try:
            gold, _ = _gold_answer(qtype, rows[i], rows[j], rows, schema_fields, derived_mappings)
            if want_match is None:
                return (i, j)
            if (gold == "Có") == want_match:
                return (i, j)
        except (ValueError, KeyError):
            continue
    return None


def render_preview(
    rows: list[dict],
    approved_types: list[dict],
    key_realizations: dict,
    n: int = 10,
    random_seed: int = 42,
    dataset: str = "vimd",
    schema_fields=None,
    derived_mappings=None,
) -> list[dict]:
    """Render previews. schema_fields is the SAME canonical set the gate
    uses (real fields only, never indexed aliases); defaults to it."""
    if schema_fields is None:
        schema_fields = SCHEMA_FIELDS
    if not approved_types or not rows:
        return []
    rng = random.Random(random_seed)
    same_pairs, diff_pairs = _row_pairs(rows, rng)
    row_order = list(range(len(rows)))
    rng.shuffle(row_order)
    all_pairs = same_pairs + diff_pairs

    # Per-type streams: pairwise types yield pos then neg when possible.
    streams: list[list[dict]] = []
    for t in approved_types:
        templates = t.get("kept_templates", [])
        if not templates:
            continue
        if _is_pairwise(t):
            items = []
            pos_pair = _find_pair_for_stream(rows, t, all_pairs, want_match=True,
                                             schema_fields=schema_fields,
                                             derived_mappings=derived_mappings)
            neg_pair = _find_pair_for_stream(rows, t, all_pairs, want_match=False,
                                             schema_fields=schema_fields,
                                             derived_mappings=derived_mappings)
            if pos_pair:
                items.append({"pair": pos_pair, "slot": 0})
            if neg_pair:
                items.append({"pair": neg_pair, "slot": 1})
            if not items:
                any_p = _find_pair_for_stream(rows, t, all_pairs, want_match=None,
                                              schema_fields=schema_fields,
                                              derived_mappings=derived_mappings)
                if any_p:
                    items.append({"pair": any_p, "slot": 0})
            if items:
                streams.append([dict(it, type=t) for it in items])
        else:
            streams.append([{"row": None, "type": t}])

    preview: list[dict] = []

    # PASS 1: one QA per final approved type, while quota permits.
    for s in streams:
        if len(preview) >= n:
            break
        item = _render_unit(
            rows, s[0], key_realizations, row_order,
            len(preview) + 1, dataset, 0, schema_fields,
            derived_mappings,
        )
        if item is not None:
            preview.append(item)

    # PASS 2: missing opposite boolean class for equality types first.
    # In stable approved-type order; skip types with no available pair;
    # only then may remaining slots use normal fill.
    if len(preview) < n:
        for s in streams:
            if len(preview) >= n:
                break
            t = s[0]["type"]
            if not _is_pairwise(t):
                continue
            seen = {it["answer"] for it in preview
                    if it["question_type_id"] == t["type_id"]}
            missing = [c for c in ("Có", "Không") if c not in seen]
            if not missing:
                continue
            pair = _find_pair_with_answer(
                rows, t, all_pairs, missing[0], preview, schema_fields,
                derived_mappings)
            if pair is None:
                continue  # never fabricate; try the next equality type
            templates = t.get("kept_templates", [])
            template = templates[1 % len(templates)]
            item = _render_pair(
                rows, t, key_realizations, template,
                pair[0], pair[1], len(preview) + 1, dataset, schema_fields,
                derived_mappings)
            if item is not None:
                preview.append(item)

    # PASS 3: ordinary deterministic fill from unused candidates.
    pass_no = 1
    while len(preview) < n and streams:
        progressed = False
        for s in streams:
            if len(preview) >= n:
                break
            unit = s[pass_no % len(s)]
            item = _render_unit(
                rows, unit, key_realizations, row_order,
                len(preview) + 1, dataset, pass_no, schema_fields,
                derived_mappings,
            )
            if item is not None:
                preview.append(item)
                progressed = True
            if not progressed:
                # If unit failed, try direct row if single-type
                pass
        if not progressed:
            break
        pass_no += 1
        if pass_no > max(len(rows), 2):
            break
    return preview[:n]


def _find_pair_with_answer(rows, qtype, pairs, want, preview,
                           schema_fields=None,
                           derived_mappings=None):
    """First pair (deterministic order) yielding `want`, unused by this type.

    Returns None when no available pair produces the missing class.
    """
    used = set()
    for it in preview:
        src = it.get("answer_source", {}) or {}
        if (it.get("question_type_id") == qtype.get("type_id")
                and src.get("kind") == "equality"):
            used.add((src.get("row_1"), src.get("row_2")))
    for i, j in pairs:
        key = (rows[i].get("audio"), rows[j].get("audio"))
        if key in used:
            continue
        try:
            gold, _ = _gold_answer(qtype, rows[i], rows[j], rows,
                                   schema_fields, derived_mappings)
        except (KeyError, ValueError):
            continue
        if gold == want:
            return (i, j)
    return None


def _render_unit(rows, unit, key_realizations, row_order, preview_idx,
                 dataset, pass_no, schema_fields=None,
                 derived_mappings=None) -> dict | None:
    t = unit["type"]
    templates = t.get("kept_templates", [])
    if not templates:
        return None
    template = templates[(pass_no) % len(templates)]
    if "pair" in unit and unit["pair"] is not None:
        return _render_pair(
            rows, t, key_realizations, template,
            unit["pair"][0], unit["pair"][1], preview_idx, dataset,
            schema_fields, derived_mappings)
    row = _find_eligible_row(rows, t, row_order, preview_idx - 1 + pass_no,
                             schema_fields=schema_fields, derived_mappings=derived_mappings)
    if row is None:
        return None
    key = _binding_key(t, schema_fields=schema_fields)
    gold, source = _gold_answer(t, row, None, rows, schema_fields,
                                derived_mappings)
    question, key_realization = render_question(
        template["text"], template["template_id"], key,
        key_realizations, gold)
    return {
        "id": f"{dataset}_preview_{preview_idx:03d}",
        "audio": row["audio"],
        "question_type_id": t["type_id"],
        "type_name": t.get("name", ""),
        "template_id": template["template_id"],
        "key": key,
        "key_realization": key_realization,
        "question": question,
        "answer": gold,
        "answer_source": source,
    }


def _render_pair(rows, t, key_realizations, template, i, j, preview_idx,
                 dataset, schema_fields=None,
                 derived_mappings=None) -> dict | None:
    key = _binding_key(t, schema_fields=schema_fields)
    row, row2 = rows[i], rows[j]
    try:
        gold, source = _gold_answer(t, row, row2, rows, schema_fields,
                                    derived_mappings)
    except (ValueError, KeyError):
        return None
    question, key_realization = render_question(
        template["text"], template["template_id"], key,
        key_realizations, gold)
    return {
        "id": f"{dataset}_preview_{preview_idx:03d}",
        "audio": [row["audio"], row2["audio"]],
        "question_type_id": t["type_id"],
        "type_name": t.get("name", ""),
        "template_id": template["template_id"],
        "key": key,
        "key_realization": key_realization,
        "question": question,
        "answer": gold,
        "answer_source": source,
    }


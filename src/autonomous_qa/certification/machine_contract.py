"""Authoritative per-type MACHINE CONTRACT for LLM prompts.

Python generates this block from the structured answer object — never from
LLM prose. The same block goes into Question Template prompts (before the
free-form goal text) and Quality bundle prompts, so both stages judge and
write against what Python will actually execute. If `name`/`goal`/
`answer_rule` wording conflicts with the contract, the contract wins.
"""

from __future__ import annotations


def answer_mode_for_kind(kind) -> str:
    """VALUE for field_value/derived_field, BINARY for equality."""
    if kind == "equality":
        return "binary"
    if kind in ("field_value", "derived_field"):
        return "value"
    return "unknown"


def describe_machine_contract(qtype: dict) -> dict:
    """Structured machine contract for one question type.

    Defensive: tolerates malformed answers (Template runs pre-gate) by
    reporting unknown fields rather than raising.
    """
    qtype = qtype or {}
    ans = qtype.get("answer") or {}
    kind = ans.get("kind")
    if kind == "field_value":
        key = ans.get("key")
        is_bool = bool(key and (key.startswith("has_") or key.startswith("is_") or key in ["same_speaker", "correct"]))
        if is_bool:
            res = {
                "answer_kind": "field_value",
                "input_arity": "1 audio",
                "output_field": key,
                "answer_mode": "binary",
                "operation": f"one audio -> field_value({key}) [boolean]",
                "template_must": [
                    "request a binary yes/no or true/false answer",
                ],
                "template_must_not": [
                    "ask open categorical questions (e.g. what is the error)",
                ],
            }
        else:
            res = {
                "answer_kind": "field_value",
                "input_arity": "1 audio",
                "output_field": key,
                "answer_mode": "value",
                "operation": f"one audio -> field_value({key})",
                "template_must": [
                    "request the target value",
                ],
                "template_must_not": [
                    "ask yes/no",
                ],
            }
    elif kind == "equality":
        keys = ans.get("keys") or []
        base = None
        if (isinstance(keys, list) and len(keys) == 2
                and isinstance(keys[0], str) and isinstance(keys[1], str)):
            left, right = keys
            if left.endswith("_1") and right.endswith("_2"):
                candidate = left[:-2]
                if right == candidate + "_2":
                    base = candidate
        res = {
            "answer_kind": "equality",
            "input_arity": "2 audio",
            "base_field": base,
            "answer_mode": "binary",
            "operation": (f"compare same {base} across two audio "
                          f"instances" if base else "compare pair"),
            "template_must": [
                "clearly ask whether the two instances share the SAME "
                "exact attribute/identity",
            ],
            "template_must_not": [
                "use ambiguous similarity wording",
            ],
        }
    elif kind == "derived_field":
        src, dst = ans.get("source_key"), ans.get("target_key")
        res = {
            "answer_kind": "derived_field",
            "input_arity": "1 audio",
            "source_field": src,
            "target_field": dst,
            "answer_mode": "value",
            "operation": (f"one audio -> identify/use {src} -> "
                          f"deterministically derive {dst}"),
            "template_must": [
                "request the target value",
                "make the source->target reasoning observable, so the "
                "task is not merely direct target recognition",
            ],
            "template_must_not": [
                "ask yes/no",
                "invent candidate values",
                "collapse into plain direct target recognition",
            ],
        }
    else:
        res = {
            "answer_kind": kind,
            "input_arity": "unknown",
            "answer_mode": answer_mode_for_kind(kind),
            "operation": "unknown",
            "template_must": [],
            "template_must_not": [],
        }
    ctx_fields = qtype.get("input_context_fields") or []
    if ctx_fields:
        res["input_context_fields"] = ctx_fields
    return res


def format_machine_contract(qtype: dict) -> str:
    """Render the authoritative MACHINE CONTRACT block for one type."""
    tid = qtype.get("id", qtype.get("type_id", "?"))
    contract = describe_machine_contract(qtype)
    lines = [
        f"Type {tid}:",
        f"  answer_kind: {contract['answer_kind']}",
        f"  input_arity: {contract['input_arity']}",
    ]
    if contract["answer_kind"] == "field_value":
        lines.append(f"  output_field: {contract['output_field']}")
    elif contract["answer_kind"] == "equality":
        lines.append(f"  base_field: {contract['base_field']}")
    elif contract["answer_kind"] == "derived_field":
        lines.append(f"  source_field: {contract['source_field']}")
        lines.append(f"  target_field: {contract['target_field']}")
    if contract.get("input_context_fields"):
        lines.append(f"  input_context_fields: {contract['input_context_fields']}")
    lines.append(f"  answer_mode: {contract['answer_mode']}")
    lines.append(f"  required_operation: {contract['operation']}")
    for rule in contract["template_must"]:
        lines.append(f"  Templates must: {rule}")
    for rule in contract["template_must_not"]:
        lines.append(f"  Templates must not: {rule}")
    return "\n".join(lines)


MACHINE_CONTRACT_PRECEDENCE = (
    "The MACHINE CONTRACT below is authoritative. `name`, `goal`, and "
    "`answer_rule` are discovery context only. If their wording conflicts "
    "with MACHINE CONTRACT, follow MACHINE CONTRACT."
)

QUALITY_CONTRACT_PRECEDENCE = (
    "Judge templates against MACHINE CONTRACT first. `name` / `goal` / "
    "`answer_rule` may be noisy LLM discovery text. If prose conflicts "
    "with MACHINE CONTRACT, MACHINE CONTRACT wins."
)

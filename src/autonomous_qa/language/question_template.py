"""Stage 2 (per round): Question Template — generic templates for THIS
round's candidate types only.

Existing approved templates are reusable assets, never a whitelist:
reuse when natural, invent when the type deserves better.
"""

from __future__ import annotations

import json

from src.common.config import CONFIG, ROOT
from src.autonomous_qa.certification.machine_contract import (
    MACHINE_CONTRACT_PRECEDENCE,
    format_machine_contract,
)
from src.autonomous_qa.core.schemas import TemplateRoundOutput
from src.common.io import load_jsonl, load_prompt


def build_question_template_prompt(
    readme: str,
    round_types: list[dict],
    template_bank: list[dict],
    round_idx: int,
    approved_bindings: dict | None = None,
    family_bank: dict | None = None,
    field_roles: dict | None = None,
    entity_scopes: dict | None = None,
    representative_values: dict | None = None,
) -> str:
    from src.autonomous_qa.core.validity import template_family_signature
    from src.autonomous_qa.language.family_templates import (
        FAMILY_EQUALITY_SEMANTIC_SPEAKER,
        FAMILY_EQUALITY_SEMANTIC_UTTERANCE,
        FAMILY_EQUALITY_SEMANTIC,
    )

    base = load_prompt("question_template")
    parts = [
        base,
        f"\nROUND: {round_idx}\n",
        "\n--- DATASET README (field semantics) ---\n" + readme,
        "\n--- CURRENT ROUND QUESTION TYPES (cover exactly these) ---\n"
        + json.dumps(round_types, ensure_ascii=False),
        "\n--- MACHINE CONTRACTS (authoritative) ---\n"
        + MACHINE_CONTRACT_PRECEDENCE + "\n"
        + "\n".join(format_machine_contract(t)
                     for t in round_types or []),
    ]
    if representative_values:
        parts.append(
            "\n--- REPRESENTATIVE SAMPLE VALUES (CONTEXT ONLY) ---\n"
            "The following are small sets of distinct non-null categorical values "
            "from the dataset to help clarify what the semantic fields mean:\n"
            + json.dumps(representative_values, ensure_ascii=False, indent=2) + "\n"
            "CRITICAL: These values are provided for linguistic context ONLY. "
            "They must NEVER be placed into question templates, options, or [VALUE]. "
            "[VALUE] is strictly forbidden in question templates."
        )
    if template_bank:
        parts.append(
            "\n--- EXISTING GENERIC TEMPLATES (reuse if natural, "
            "never mandatory) ---\n"
            + json.dumps(template_bank, ensure_ascii=False)
        )
    if approved_bindings:
        parts.append(
            "\n--- EXISTING APPROVED KEY REALIZATIONS ---\n"
            "If a [KEY] template refers to a raw field that already has "
            "an approved realization below, use/reuse that semantic "
            "realization rather than inventing a conflicting mapping. "
            "This is a consistency aid, NOT a template whitelist — new "
            "structures are still welcome.\n"
            + json.dumps(approved_bindings, ensure_ascii=False)
        )

    # Group equality types by entity-scoped family (§L-§P)
    fam_types_by_sig: dict[tuple, list[dict]] = {}
    for t in round_types:
        sig = template_family_signature(t, field_roles=field_roles, entity_scopes=entity_scopes)
        if sig is not None:
            fam_types_by_sig.setdefault(sig, []).append(t)

    for fam_sig, eq_types in fam_types_by_sig.items():
        scope = fam_sig[2] if len(fam_sig) > 2 else "unknown"
        existing_bases = (family_bank or {}).get(fam_sig) or []
        if not existing_bases and fam_sig == FAMILY_EQUALITY_SEMANTIC_SPEAKER:
            existing_bases = (family_bank or {}).get(FAMILY_EQUALITY_SEMANTIC) or []
        if existing_bases:
            parts.append(
                f"\n--- SHARED OPERATION-FAMILY: EQUALITY (SEMANTIC / {scope.upper()}) ---\n"
                "The following shared generic base templates are ALREADY APPROVED "
                "in the family bank and will be automatically reused for all semantic "
                f"equality types in this family ({scope}-scoped):\n"
                + json.dumps(existing_bases, ensure_ascii=False) + "\n"
                "Do NOT regenerate new base skeletons for these types. Provide their "
                "required natural Vietnamese [KEY] realization in `key_realizations`:\n"
                + json.dumps([t.get("id") for t in eq_types])
            )
        else:
            scope_instructions = ""
            if scope == "speaker":
                scope_instructions = (
                    "   - Base skeletons must refer to 'hai người nói' or '[KEY] của hai người nói'.\n"
                    "     Example: 'Hai người nói trong hai đoạn âm thanh có cùng [KEY] không?'\n"
                    "     Avoid 'giọng', 'chất giọng' as the subject noun.\n"
                )
            elif scope == "utterance":
                scope_instructions = (
                    "   - Base skeletons must refer to 'hai phát ngôn' or 'nội dung/ngữ nghĩa'.\n"
                    "     Example: 'Hai phát ngôn trong hai đoạn âm thanh có cùng [KEY] không?'\n"
                    "     Do NOT use 'giọng người nói' for utterance-scoped fields.\n"
                )
            parts.append(
                f"\n--- NEW SHARED OPERATION-FAMILY BUNDLE: EQUALITY (SEMANTIC / {scope.upper()}) ---\n"
                "Machine contract for this family:\n"
                "- answer_kind: equality\n"
                f"- entity_scope: {scope}\n"
                "- input_arity: 2 audio\n"
                "- answer_mode: binary (Có / Không)\n"
                "- operation: determine whether the same semantic attribute value holds for both audio instances\n"
                "- variable concept: [KEY]\n"
                f"- member types: {[t.get('id') for t in eq_types]}\n\n"
                "INSTRUCTIONS:\n"
                "1. Generate 2 to 4 delexicalized operation-level base templates using exactly one [KEY].\n"
                f"{scope_instructions}"
                "2. UNIVERSALITY SELF-CHECK: Before returning a base skeleton, verify that replacing [KEY] "
                "with every member type's realization yields a grammatically natural Vietnamese sentence.\n"
                "3. Each base template must use IDs starting with 'BF_EQSEM_' (e.g. BF_EQSEM_001) "
                f"and declare 'family_signature': {list(fam_sig)}.\n"
                "4. Provide each member type's natural [KEY] phrase in `key_realizations`.\n"
            )
    return "\n".join(parts)


def run_question_template_round(
    client,
    readme: str,
    round_types: list[dict],
    template_bank: list[dict],
    round_idx: int,
    real_llm: bool,
    raw_path=None,
    on_attempt=None,
    approved_bindings: dict | None = None,
    dataset: str = "vimd",
    family_bank: dict | None = None,
    field_roles: dict | None = None,
    entity_scopes: dict | None = None,
    representative_values: dict | None = None,
) -> dict:
    from src.common.io import load_fixture

    if not real_llm:
        return load_fixture(f"loop/round_{round_idx:02d}/template", dataset=dataset)

    from src.autonomous_qa.authoring.llm_client import complete_structured_stage

    if raw_path is None:
        raise ValueError("raw_path is required for real LLM question_template runs")

    prompt = build_question_template_prompt(
        readme, round_types, template_bank, round_idx,
        approved_bindings=approved_bindings,
        family_bank=family_bank,
        field_roles=field_roles,
        entity_scopes=entity_scopes,
        representative_values=representative_values,
    )
    # Durability contract: NO semantic validation here. The shared
    # paid-response recorder inside complete_structured_stage persists the
    # raw response + paid checkpoint BEFORE parsing, and the canonical
    # type-keyed key_realizations validation runs in _do_template_round
    # AFTER this returns. Do NOT add validation before the persistence
    # boundary.
    return complete_structured_stage(
        client,
        stage="question_template",
        prompt=prompt,
        response_schema=TemplateRoundOutput,
        raw_path=raw_path,
        on_attempt=on_attempt,
    )


def load_template_bank() -> list[dict]:
    """Human-approved templates from earlier runs (memory, not constraint)."""
    return load_jsonl(ROOT / "fewshots" / "approved_templates.jsonl")

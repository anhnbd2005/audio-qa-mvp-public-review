"""Pipeline entrypoint (MVP v3: full discovery loop).

Default (`--dataset vimd`) is MOCK: 0 real calls, fixtures drive the flow.
Only `--real-llm` creates the Vertex client and spends batched calls.

Output roots (never mixed):
- mock: outputs/runs/<dataset>/mock/
- real: outputs/runs/<dataset>/<run_id>/   (unique per run, never overwritten)

Each round runs Style -> Template -> Paraphrase -> Quality (4 calls).
A Python validity gate sits between Template and Paraphrase: deterministic
errors never spend Paraphrase/Quality calls. Saturation is measured on
question-TYPE discovery (new_type_rate), never on wording.

Checkpoints: run_state.json is saved after every successfully persisted
stage. A crashed real run resumes with:
    python -m src.run_pipeline --dataset vimd --real-llm --resume-run <run_id>
Resume reuses every completed artifact byte-for-byte and only issues the
next unfinished LLM request.

Budget counts PAID attempts (failed parses included): design needs
4R successes, plus exactly one manual-retry attempt per stage.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from src.common.config import CONFIG, ROOT
from src.autonomous_qa.core.loop import (
    LoopController,
    apply_accepted_types,
    build_round_stats,
    build_type_fewshot,
    filter_paraphrases_for_judging,
    flatten_type_bindings,
    validate_quality_partition,
)
from src.autonomous_qa.language.family_templates import (
    FAMILY_EQUALITY_SEMANTIC,
    FAMILY_EQUALITY_SEMANTIC_SPEAKER,
    FAMILY_EQUALITY_SEMANTIC_UTTERANCE,
    bind_round_family_templates,
    build_family_bank_entry,
    deserialize_family_signature,
    serialize_family_signature,
)
from src.autonomous_qa.language.paraphrase import run_paraphrase_round
from src.autonomous_qa.certification.quality import run_quality_round
from src.autonomous_qa.language.question_style import run_question_style_round
from src.autonomous_qa.language.question_template import load_template_bank, run_question_template_round
from src.autonomous_qa.production.render_preview import render_preview
from src.common.io import read_json, write_json
from src.autonomous_qa.core.validity import (
    DEFAULT_FIELD_ROLES,
    DEFAULT_ENTITY_SCOPES,
    ROLE_HIDDEN_IDENTIFIER,
    SCHEMA_FIELDS,
    extract_field_roles,
    extract_entity_scopes,
    extract_source_missing_values,
    extract_evaluation_policies,
    normalize_row_missing_values,
    required_answer_fields,
    is_evaluation_eligible_type,
    gate_round,
    get_answer_concept_field,
    template_family_signature,
)
from src.autonomous_qa.datasets.context_adapter import (
    load_readme,
    load_sample,
    load_metadata_context,
    load_knowledge_context,
    load_dataset_rows,
    load_schema,
    dataset_schema_fields,
    dataset_field_roles,
    dataset_hidden_fields,
    dataset_entity_scopes,
    dataset_source_missing_values,
    dataset_evaluation_policies,
    extract_representative_values,
)
from src.autonomous_qa.compiler.run_pipeline_cli import loop_controller_from_config, resolve_out_dir



RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

STATE_FILE = "run_state.json"

# Question Template LLM contract fingerprint (§39, §41, §AE). Pre-patch runs stored
# raw-field-keyed key_realizations or un-scoped families and MUST refuse resume
# clearly as incompatible (no migration, no reinterpretation).
TEMPLATE_CONTRACT_VERSION = "operation-family-scope-context-v3"

# Schema policy contract fingerprint: entity scope and evaluation policy metadata
SCHEMA_POLICY_CONTRACT_VERSION = "schema-policy-v1"

# Answer-reference contract fingerprint (§31). Runs produced under the OLD
# broad suffix-normalization gate semantics MUST refuse resume/replay as
# incompatible (no migration, no reinterpretation).
ANSWER_REFERENCE_CONTRACT_VERSION = "kind-scoped-v1"

# Canonical answer-shape contract fingerprint: each answer kind owns its
# fields; irrelevant non-null fields deterministically reject the type.
ANSWER_SHAPE_CONTRACT_VERSION = "canonical-v1"

# Executable-signature dedup contract fingerprint: identical executable
# signatures are deterministic duplicates before Quality.
EXECUTABLE_SIGNATURE_CONTRACT_VERSION = "v1"

# Derived-field contract fingerprint: derived relations are validated
# from dataset evidence (data-backed-v1), never from field membership
# alone and never from a hard-coded relation list.
DERIVED_FIELD_CONTRACT_VERSION = "data-backed-v1"

# Vietnamese linguistic knowledge layer contract fingerprint (§51, §61).
KNOWLEDGE_CONTRACT_VERSION = "vi-knowledge-v1"












def round_dir(rounds_dir: Path, round_idx: int) -> Path:
    return rounds_dir / f"round_{round_idx:02d}"


def run_pipeline(
    dataset: str,
    real_llm: bool,
    run_id: str | None = None,
    resume_run: str | None = None,
    rows: list[dict] | None = None,
) -> list[dict]:
    loop_cfg = CONFIG["loop"]
    max_rounds = loop_cfg["max_rounds"]
    design_max = 4 * max_rounds

    out_dir, effective_run_id = resolve_out_dir(
        dataset, real_llm, run_id, resume_run
    )
    raw_dir = out_dir / "raw"
    rounds_dir = out_dir / "rounds"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rounds_dir.mkdir(parents=True, exist_ok=True)

    readme = load_readme(dataset)
    if rows is None:
        if real_llm:
            rows = load_dataset_rows(dataset)
        else:
            rows = load_sample(dataset)

    client = None
    if real_llm:
        from src.autonomous_qa.authoring.llm_client import BUDGET, create_llm_client

        BUDGET.display_total = BUDGET.max_calls
        if resume_run is None:
            print("========================================")
            print("Audio QA MVP v3 — discovery loop")
            print(f"Dataset: {dataset}")
            print(f"Run ID: {effective_run_id}")
            print(f"Model: {CONFIG["llm"]['model']}")
            print("")
            print(f"Max rounds: {max_rounds} (4 calls/round)")
            print(f"Maximum LLM calls: {BUDGET.max_calls} "
                  f"(design {design_max} + "
                  f"{BUDGET.max_calls - design_max} manual-retry allowance)")
            print("========================================")
            client = create_llm_client()
            run = _FreshRun(
                dataset=dataset,
                run_id=effective_run_id,
                out_dir=out_dir,
                raw_dir=raw_dir,
                rounds_dir=rounds_dir,
                readme=readme,
                rows=rows,
                client=client,
            )
            return run.execute()
        else:
            run = _ResumedRun(
                dataset=dataset,
                out_dir=out_dir,
                raw_dir=raw_dir,
                rounds_dir=rounds_dir,
                readme=readme,
                rows=rows,
            )
            # Validate everything BEFORE touching the network.
            run.load_and_validate()
            print("========================================")
            print("Audio QA MVP v3 — RESUME")
            print(f"Dataset: {dataset}")
            print(f"Run ID: {run.state['run_id']}")
            print(f"Next action: {run.state['next_action']}")
            attempted = int(run.state.get("attempted_calls_total", 0))
            print(f"Paid attempts already spent: {attempted}")
            print(f"Remaining budget: {BUDGET.max_calls - BUDGET.count}")
            print("========================================")
            client = create_llm_client()
            run.attach_client(client)
            return run.execute()
    else:
        print(f"Dataset: {dataset}")
        print("Mode: MOCK")
        run = _FreshRun(
            dataset=dataset,
            run_id="mock",
            out_dir=out_dir,
            raw_dir=raw_dir,
            rounds_dir=rounds_dir,
            readme=readme,
            rows=rows,
            client=None,
        )
        return run.execute()


class _RunBase:
    """Shared round machinery + finalize for fresh and resumed runs."""

    def __init__(self, *, dataset, run_id, out_dir, raw_dir, rounds_dir,
                 readme, rows, client):
        self.dataset = dataset
        self.schema_data = load_schema(dataset)
        self.schema_fields = dataset_schema_fields(dataset)
        self.field_roles = dataset_field_roles(dataset)
        self.hidden_fields = dataset_hidden_fields(dataset)
        self.entity_scopes = dataset_entity_scopes(dataset)
        self.source_missing_values = dataset_source_missing_values(dataset)
        self.evaluation_policies = dataset_evaluation_policies(dataset)
        self.run_id = run_id
        self.out_dir = out_dir
        self.raw_dir = raw_dir
        self.rounds_dir = rounds_dir
        self.readme = readme
        self.rows = [normalize_row_missing_values(r, self.source_missing_values) for r in (rows or [])]
        self.rep_values = extract_representative_values(self.rows, self.schema_data)
        self.metadata_context = load_metadata_context(dataset)
        self.knowledge_context = load_knowledge_context(dataset)
        self.client = client
        self.real_llm = client is not None
        self.controller = loop_controller_from_config()
        self.history: list[dict] = []
        self.approved_pool: list[dict] = []
        self.key_realizations: dict = {}
        self.approved_key_realizations: dict = {}
        self._round_candidates: dict[int, dict] = {}
        self._round_type_phrases: dict[int, dict] = {}
        self._round_det_dups: dict[int, list] = {}
        self._round_novel_types: dict[int, list] = {}
        self._relation_cache: dict = {}
        self.template_bank = load_template_bank()
        self.family_template_bank: dict[tuple, list[dict]] = {}
        self._round_provisional_family_bases: dict[int, dict[tuple, list[dict]]] = {}
        self.stop_reason: str | None = None
        self._current_action = "style_round_1"
        self._current_round = 0

    # -- checkpoint -----------------------------------------------------
    def _save_state(self, *, completed: list[str], current_round: int,
                    next_action: str, stopped: bool,
                    stop_reason: str | None) -> None:
        from src.autonomous_qa.authoring.llm_client import ATTEMPTS, BUDGET

        write_json(self.out_dir / STATE_FILE, {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "mode": "real" if self.real_llm else "mock",
            "completed": completed,
            "current_round": current_round,
            "next_action": next_action,
            "low_gain_streak": self.controller.low_gain_streak,
            "stopped": stopped,
            "stop_reason": stop_reason,
            "spent_calls": dict(BUDGET.stage_counts) if self.real_llm else {},
            "attempted_calls_total": (
                ATTEMPTS.attempted_calls_total if self.real_llm else 0),
            "stage_attempts": (
                dict(ATTEMPTS.stage_attempts) if self.real_llm else {}),
            "successful_calls_total": (
                ATTEMPTS.successful_calls_total if self.real_llm else 0),
            "last_finish_reason": ATTEMPTS.last_finish_reason,
            "last_failed_stage": ATTEMPTS.last_failed_stage,
            "loop_config": {
                "max_rounds": self.controller.max_rounds,
                "min_new_type_rate": self.controller.min_rate,
                "saturation_patience": self.controller.saturation_patience,
                "immediate_stop_if_zero_new":
                    self.controller.immediate_stop_if_zero_new,
            },
            "template_contract_version": TEMPLATE_CONTRACT_VERSION,
            "schema_policy_contract_version": SCHEMA_POLICY_CONTRACT_VERSION,
            "answer_reference_contract_version":
                ANSWER_REFERENCE_CONTRACT_VERSION,
            "answer_shape_contract_version": ANSWER_SHAPE_CONTRACT_VERSION,
            "executable_signature_contract_version":
                EXECUTABLE_SIGNATURE_CONTRACT_VERSION,
            "derived_field_contract_version": DERIVED_FIELD_CONTRACT_VERSION,
            "knowledge_contract_version": KNOWLEDGE_CONTRACT_VERSION,
            "family_template_bank": {
                serialize_family_signature(k): v
                for k, v in self.family_template_bank.items()
            },
        })

    def _on_attempt(self, stage: str, attempt_no: int,
                      finish_reason, raw_text: str) -> None:
        """Checkpoint a PAID attempt BEFORE parsing/validation.

        A MAX_TOKENS truncation therefore still counts as a real request
        and is resumable via --resume-run. Never retries anything.
        """
        reason = None if finish_reason is None else str(finish_reason)
        print(f"[checkpoint] {stage} attempt {attempt_no:02d} recorded "
              f"(finish={reason})")
        self._save_state(
            completed=list(self._completed_snapshot()),
            current_round=self._current_round,
            next_action=self._current_action,
            stopped=False,
            stop_reason=None,
        )

    def _completed_snapshot(self) -> list[str]:
        return list(getattr(self, "_completed", []))

    def _attempt_cb(self):
        return self._on_attempt if self.real_llm else None

    # -- per-stage round steps (each: 1 LLM call, persist, checkpoint) ---
    def _do_style_round(self, round_idx: int) -> list[dict]:
        self._current_action = f"style_round_{round_idx}"
        self._current_round = round_idx
        dynamic = build_type_fewshot(self.approved_pool)
        out = run_question_style_round(
            client=self.client, readme=self.readme,
            dynamic_fewshots=dynamic, round_idx=round_idx,
            real_llm=self.real_llm,
            raw_path=self.raw_dir / f"round_{round_idx:02d}_style.txt",
            on_attempt=self._attempt_cb(),
            dataset=self.dataset,
            metadata_context=self.metadata_context,
            knowledge_context=self.knowledge_context,
            sample_rows=self.rows,
        )
        rdir = round_dir(self.rounds_dir, round_idx)
        rdir.mkdir(parents=True, exist_ok=True)
        write_json(rdir / "01_question_styles.json", out)
        types = out.get("question_types", [])
        print(f"[R{round_idx}] Style proposed {len(types)} types")
        self._completed.append(f"style_round_{round_idx}")
        self._save_state(
            completed=list(self._completed), current_round=round_idx,
            next_action=f"template_round_{round_idx}",
            stopped=False, stop_reason=None)
        return types

    def _process_round_family_templates(
        self, round_idx: int, types: list[dict], out: dict
    ) -> tuple[dict, list[dict]]:
        """Process operation-family base templates and bind before Gate (§L-§P).

        Returns (type_phrases, combined_templates).
        """
        fam_types_by_sig: dict[tuple, list[dict]] = {}
        for t in types:
            sig = template_family_signature(t, self.field_roles, self.schema_fields, self.entity_scopes)
            if sig is not None:
                fam_types_by_sig.setdefault(sig, []).append(t)

        raw_fam_bases = out.get("family_templates", [])
        raw_bases_by_sig: dict[tuple, list[dict]] = {}
        for b in raw_fam_bases:
            b_sig = b.get("family_signature")
            if b_sig:
                parsed_sig = tuple(b_sig) if isinstance(b_sig, list) else deserialize_family_signature(str(b_sig))
                raw_bases_by_sig.setdefault(parsed_sig, []).append(b)
            elif len(fam_types_by_sig) == 1:
                only_sig = next(iter(fam_types_by_sig.keys()))
                raw_bases_by_sig.setdefault(only_sig, []).append(b)
            else:
                raw_bases_by_sig.setdefault(FAMILY_EQUALITY_SEMANTIC_SPEAKER, []).append(b)

        active_bases_by_sig: dict[tuple, list[dict]] = {}
        round_prov: dict[tuple, list[dict]] = {}

        for fam_sig, fam_types in fam_types_by_sig.items():
            if fam_sig in self.family_template_bank and self.family_template_bank[fam_sig]:
                active_bases_by_sig[fam_sig] = list(self.family_template_bank[fam_sig])
                raw_for_sig = raw_bases_by_sig.get(fam_sig, [])
                if raw_for_sig:
                    round_prov[fam_sig] = list(raw_for_sig)
            else:
                raw_for_sig = raw_bases_by_sig.get(fam_sig, [])
                active_bases_by_sig[fam_sig] = list(raw_for_sig)
                round_prov[fam_sig] = list(raw_for_sig)

        self._round_provisional_family_bases[round_idx] = round_prov

        validation_templates = list(out.get("templates", []))
        for fam_sig, fam_types in fam_types_by_sig.items():
            active_bases = active_bases_by_sig.get(fam_sig, [])
            if fam_types and active_bases:
                for ft in fam_types:
                    validation_templates.append({
                        "question_type_id": ft["id"],
                        "text": "[KEY]",
                    })

        type_phrases = self._validate_template_mapping(
            out.get("key_realizations", {}), round_idx,
            type_ids=[t.get("id") for t in types],
            templates=validation_templates,
        )

        bound_family = []
        all_fam_type_ids = set()
        for fam_sig, fam_types in fam_types_by_sig.items():
            active_bases = active_bases_by_sig.get(fam_sig, [])
            if fam_types and active_bases:
                bound = bind_round_family_templates(
                    active_bases, fam_types, type_phrases, family_sig=fam_sig
                )
                bound_family.extend(bound)
                all_fam_type_ids.update(ft["id"] for ft in fam_types)

        if bound_family:
            regular_templates = [
                t for t in out.get("templates", [])
                if t.get("question_type_id") not in all_fam_type_ids
            ]
            combined = regular_templates + bound_family
        else:
            combined = list(out.get("templates", []))

        return type_phrases, combined

    def _commit_family_bases(self, round_idx: int, verdict: dict,
                            accepted: list[dict], bundles: list[dict]) -> None:
        """Persist provisional family bases into family_template_bank if kept (§28).

        A base skeleton is persisted into family_template_bank only if at
        least one accepted member type keeps a Quality-approved descendant.
        """
        round_prov = self._round_provisional_family_bases.get(round_idx, {})
        if not round_prov:
            return

        accepted_tids = {it.get("type_id") for it in accepted}
        accepted_verdicts = [
            item for item in verdict.get("accepted_new", [])
            if item.get("type_id") in accepted_tids
        ]
        all_kept_ids = set()
        for item in accepted_verdicts:
            all_kept_ids.update(item.get("keep_template_ids", []))

        para_source = {}
        for b in bundles:
            for p in b.get("paraphrases", []):
                pid = p.get("template_id")
                src = p.get("source_template_id")
                if pid and src:
                    para_source[pid] = src

        items_to_process = (
            round_prov.items() if isinstance(round_prov, dict)
            else [(FAMILY_EQUALITY_SEMANTIC_SPEAKER, round_prov)]
        )

        for fam_sig, provisional in items_to_process:
            bank_list = self.family_template_bank.setdefault(fam_sig, [])
            existing_bank_ids = {
                b.get("base_template_id") or b.get("template_id")
                for b in bank_list
            }

            for base in provisional:
                base_id = base.get("base_template_id") or base.get("template_id")
                if not base_id or base_id in existing_bank_ids:
                    continue
                descendant_kept = False
                for tid in accepted_tids:
                    expected_bound_id = f"BT_{tid}__{base_id}"
                    for kid in all_kept_ids:
                        if kid == expected_bound_id:
                            descendant_kept = True
                            break
                        if kid in para_source and para_source[kid] == expected_bound_id:
                            descendant_kept = True
                            break
                    if descendant_kept:
                        break
                if descendant_kept:
                    entry = build_family_bank_entry(
                        base_template_id=base_id,
                        text=base.get("text", ""),
                        round_created=round_idx,
                    )
                    entry["family_signature"] = list(fam_sig)
                    bank_list.append(entry)
                    existing_bank_ids.add(base_id)
                    print(f"[R{round_idx}] Family base {base_id} persisted to bank for {fam_sig}")

    def _do_template_round(self, round_idx: int,
                           types: list[dict]) -> dict:
        self._current_action = f"template_round_{round_idx}"
        self._current_round = round_idx
        out = run_question_template_round(
            client=self.client, readme=self.readme, round_types=types,
            template_bank=self.template_bank, round_idx=round_idx,
            real_llm=self.real_llm,
            raw_path=self.raw_dir / f"round_{round_idx:02d}_template.txt",
            on_attempt=self._attempt_cb(),
            approved_bindings=dict(self.approved_key_realizations),
            dataset=self.dataset,
            family_bank=self.family_template_bank,
            field_roles=self.field_roles,
            entity_scopes=self.entity_scopes,
            representative_values=self.rep_values,
        )
        type_phrases, combined = self._process_round_family_templates(
            round_idx, types, out
        )
        out["templates"] = combined
        rdir = round_dir(self.rounds_dir, round_idx)
        write_json(rdir / "02_templates.json", out)
        self._round_type_phrases[round_idx] = dict(type_phrases)
        print(f"[R{round_idx}] Templates: "
              f"{len(combined)} produced")
        print(f"[R{round_idx}] Candidate type realizations: "
              f"{len(type_phrases)}")
        self._completed.append(f"template_round_{round_idx}")
        self._save_state(
            completed=list(self._completed), current_round=round_idx,
            next_action=f"paraphrase_round_{round_idx}",
            stopped=False, stop_reason=None)
        return out

    def _gate_round(self, round_idx: int, types: list[dict],
                     templates: list[dict]) -> dict:
        """Deterministic validity gate. Zero LLM calls."""
        # Exact-dup collapse is scoped per (question_type_id, normalized
        # text) inside gate_round: identical generic text bound to
        # different concepts survives for each type's own pipeline.
        # [KEY] bindability is resolved PER OWNER TYPE: approved canonical
        # field bindings first, else the owner's own type phrase.
        # NOTE: round field candidates are built AFTER executable-signature
        # dedup (see _dedup_and_build_candidates), from novel types only.
        gate = gate_round(
            types, templates,
            type_key_realizations=self._round_type_phrases.get(round_idx, {}),
            approved_key_realizations=self.approved_key_realizations,
            dataset_rows=self.rows,
            relation_cache=self._relation_cache,
            schema_fields=self.schema_fields,
            hidden_fields=self.hidden_fields,
            field_roles=self.field_roles)
        leak_drops = sum(
            1 for d in gate["dropped_templates"]
            if d.get("reason") == "answer_leakage")
        collapsed = sum(
            1 for d in gate["dropped_templates"]
            if str(d.get("reason", "")).startswith("exact_duplicate:"))
        other_drops = (len(gate["dropped_templates"])
                       - leak_drops - collapsed)
        print(f"[R{round_idx}] Gate: {len(gate['valid_types'])}/{len(types)} "
              f"types valid, {len(gate['auto_rejected'])} auto-rejected, "
              f"{len(gate['dropped_templates'])} templates dropped "
              f"(+{collapsed} exact-dup strings collapsed)")
        print(f"[R{round_idx}] Base templates dropped:\n"
              f"     answer leakage: {leak_drops}\n"
              f"     other deterministic: {other_drops}")
        # Persist deterministic audit information per proposed derived relation (§59).
        round_derived_audit = []
        for t in types:
            ans = t.get("answer", {}) or {}
            if ans.get("kind") == "derived_field":
                src = ans.get("source_key")
                tgt = ans.get("target_key")
                entry = self._relation_cache.get((src, tgt))
                if entry is not None:
                    round_derived_audit.append({
                        "type_id": t.get("id"),
                        "source_key": src,
                        "target_key": tgt,
                        "valid": entry.get("valid"),
                        "reason": entry.get("reason"),
                        "distinct_source_count": entry.get("distinct_source_count"),
                        "distinct_target_count": entry.get("distinct_target_count"),
                        "mapping_conflicts": entry.get("mapping_conflicts"),
                        "reusable_source_values": entry.get("reusable_source_values"),
                    })
        if round_derived_audit:
            write_json(round_dir(self.rounds_dir, round_idx)
                       / "derived_relations.json", {
                           "round": round_idx,
                           "derived_relations": round_derived_audit,
                       })
        return gate

    def _dedup_and_build_candidates(self, round_idx: int,
                                    types: list[dict], gate: dict,
                                    persist_audit: bool = True) -> list[dict]:
        """Signature dedup + field candidates from NOVEL types only.

        Builds the approved signature index (prior approved always wins),
        splits gate-valid types (original Style order) into novel +
        deterministic duplicates, persists the audit artifact, logs the
        mapping, then builds round field candidates from novel types with
        surviving [KEY] templates only. Returns novel valid types.
        """
        from src.autonomous_qa.core.validity import (
            build_approved_signature_index,
            build_round_field_candidates,
            deduplicate_by_signature,
        )

        index = build_approved_signature_index(
            self.approved_pool, schema_fields=self.schema_fields)
        novel, det_dups = deduplicate_by_signature(
            gate["valid_types"], index, schema_fields=self.schema_fields)
        self._round_novel_types[round_idx] = list(novel)
        self._round_det_dups[round_idx] = list(det_dups)
        if persist_audit:
            write_json(round_dir(self.rounds_dir, round_idx)
                       / "signature_dedup.json", {
                           "round": round_idx,
                           "novel_type_ids": [t.get("id") for t in novel],
                           "deterministic_duplicates": [
                               dict(d) for d in det_dups],
                       })
        print(f"[R{round_idx}] Signature dedup: "
              f"{len(novel)} novel / {len(gate['valid_types'])} gate-valid, "
              f"{len(det_dups)} deterministic duplicates")
        for d in det_dups:
            print(f"    {d['type_id']} -> {d['duplicate_of']} "
                  f"(source={d['duplicate_source']}) "
                  f"sig={tuple(d['signature'])}")
        # Round FIELD candidates collapse AFTER signature dedup, from novel
        # valid types with surviving [KEY] templates only. Deterministic
        # duplicates (and auto-rejected types) can never seed a candidate.
        novel_ids = {t.get("id") for t in novel}
        novel_templates = [t for t in gate["valid_templates"]
                           if t.get("question_type_id") in novel_ids]
        self._round_candidates[round_idx] = build_round_field_candidates(
            [t for t in types if t.get("id") in novel_ids],
            novel_templates,
            [t.get("id") for t in novel],
            self._round_type_phrases.get(round_idx, {}),
            self.approved_key_realizations,
            schema_fields=self.schema_fields)
        return novel

    def _post_signature_dedup_terminate(
            self, round_idx: int, gate: dict, novel: list[dict],
            det_dups: list[dict], generated_types: int) -> bool:
        """ONE shared post-dedup decision (§71).

        Novel types exist -> False (continue to Paraphrase). Otherwise
        persist terminal dedup state (gate + dedup + duplicate records,
        stop_reason, next_action=finalize) and return True (terminate,
        finalize prior pool, never call Paraphrase/Quality). Fresh and
        resume use the same helper.
        """
        if novel:
            return False
        print(f"[R{round_idx}] Signature dedup: 0 novel survivors — "
              f"short-circuit, no Paraphrase/Quality calls.")
        self.stop_reason = "no_novel_types_after_signature_dedup"
        print("")
        print(f"STOP: {self.stop_reason}")
        print("Continue: NO")
        self.history.append(build_round_stats(
            round_idx=round_idx,
            generated_types=generated_types,
            accepted_new_types=0,
            duplicates=0,
            deterministic_duplicates=len(det_dups),
            rejected=len(gate["auto_rejected"]),
            new_type_rate=0.0,
            total_approved_types=len(self.approved_pool),
        ))
        self._save_state(
            completed=list(self._completed), current_round=round_idx,
            next_action="finalize",
            stopped=True, stop_reason=self.stop_reason)
        return True

    def _effective_key_context(self, round_idx: int) -> dict:
        """Effective FIELD realizations for Quality (§23).

        approved_key_realizations + round field candidates for fields not
        already approved. Approved values always win.
        """
        effective = dict(self._round_candidates.get(round_idx, {}))
        effective.update(self.approved_key_realizations)
        return effective

    def _validate_template_mapping(self, candidate: dict,
                                   round_idx: int, type_ids=None,
                                   templates=None) -> dict:
        """Fail closed on malformed type-keyed key_realizations.

        LLM keys MUST be exact current-round type IDs (never raw fields,
        never "KEY", never Vietnamese phrases). No fuzzy repair, no
        raw-field fallback: ONE contract only.
        """
        from src.autonomous_qa.core.validity import validate_type_key_realizations

        if type_ids is None:
            type_ids = [t.get("id")
                        for t in self._round_types.get(round_idx, {}).values()]
        problems = validate_type_key_realizations(
            candidate, type_ids, templates or [])
        if problems:
            from src.autonomous_qa.authoring.llm_client import StageOutputValidationError
            raise StageOutputValidationError(
                f"[R{round_idx}] malformed key_realizations: "
                + "; ".join(problems))
        known = set(type_ids or [])
        return {k: v for k, v in dict(candidate or {}).items() if k in known}

    def _commit_round_bindings(self, round_idx: int,
                                 accepted: list[dict]) -> dict:
        """Commit approved bindings for accepted types (§26).

        Only accepted types with kept [KEY] templates commit, using the
        canonical ROUND FIELD candidate Quality judged (never a private
        type phrase). Iteration follows original Style order. Returns the
        audit {"committed": [...], "ignored": [...]}.
        """
        from src.autonomous_qa.core.loop import commit_approved_bindings

        result = commit_approved_bindings(
            self.approved_key_realizations,
            self._round_candidates.get(round_idx, {}),
            accepted,
            self._round_types.get(round_idx, {}),
            type_order=[t.get("id") for t in
                        self._round_types.get(round_idx, {}).values()],
        )
        if result["committed"]:
            print(f"[R{round_idx}] Approved key realizations committed: "
                  f"{len(result['committed'])}")
        for ign in result["ignored"]:
            print(f"[R{round_idx}] Ignored key-realization override for "
                  f"{ign['field']}")
        return result

    def _prepare_judging(self, round_idx: int, valid_types: list[dict],
                           valid_templates: list[dict],
                           paraphrases: list[dict]) -> dict:
        """Deterministic post-paraphrase filter + Quality inputs. No LLM.

        Drops [VALUE]-leaking paraphrases and exact-normalized duplicates
        per question type, persists the audit alongside the paraphrase
        artifact (deterministic: byte-identical on replay), and builds
        bundles/texts/owners for judging. Returns
        {"bundles", "texts", "owners", "dropped_leak", "dropped_exact"}.
        """
        filt = filter_paraphrases_for_judging(valid_templates, paraphrases)
        rdir = round_dir(self.rounds_dir, round_idx)
        doc = read_json(rdir / "03_paraphrases.json")
        doc["exact_duplicates_dropped"] = filt["dropped_exact"]
        doc["leak_dropped"] = filt["dropped_leak"]
        write_json(rdir / "03_paraphrases.json", doc)
        print(f"[R{round_idx}] Post-paraphrase: "
              f"answer leakage dropped: {len(filt['dropped_leak'])}, "
              f"exact duplicates dropped: {len(filt['dropped_exact'])}")
        effective = dict(self._round_candidates.get(round_idx, {}))
        effective.update(self.approved_key_realizations)
        bundles, texts = _build_bundles(
            valid_types, valid_templates, filt["surviving"],
            key_context=effective)
        owners = {t["template_id"]: t["question_type_id"]
                  for t in valid_templates}
        owners.update(filt["owners"])
        self._round_texts[round_idx] = texts
        self._round_owners[round_idx] = owners
        return {"bundles": bundles, "texts": texts, "owners": owners,
                "dropped_leak": filt["dropped_leak"],
                "dropped_exact": filt["dropped_exact"]}

    def _do_paraphrase_round(self, round_idx: int,
                             valid_templates: list[dict]) -> list[dict]:
        self._current_action = f"paraphrase_round_{round_idx}"
        self._current_round = round_idx
        out = run_paraphrase_round(
            client=self.client, templates=valid_templates,
            round_idx=round_idx, real_llm=self.real_llm,
            raw_path=self.raw_dir / f"paraphrase_round_{round_idx}.txt",
            on_attempt=self._attempt_cb(),
            dataset=self.dataset,
        )
        # 1 Gemini call
        paraphrases = out.get("paraphrases", [])
        print(f"[R{round_idx}] Paraphrases: {len(paraphrases)} produced")
        rdir = round_dir(self.rounds_dir, round_idx)
        write_json(rdir / "03_paraphrases.json", out)
        self._completed.append(f"paraphrase_round_{round_idx}")
        self._save_state(
            completed=list(self._completed), current_round=round_idx,
            next_action=f"quality_round_{round_idx}",
            stopped=False, stop_reason=None)
        return paraphrases

    def _do_quality_round(self, round_idx: int, bundles: list[dict],
                          generated_types: int) -> bool:
        """Returns True when the loop must stop after this round."""
        self._current_action = f"quality_round_{round_idx}"
        self._current_round = round_idx
        approved_fewshots = build_type_fewshot(self.approved_pool)
        verdict = run_quality_round(
            client=self.client, bundles=bundles,
            approved_fewshots=approved_fewshots, round_idx=round_idx,
            real_llm=self.real_llm,
            raw_path=self.raw_dir / f"quality_round_{round_idx}.txt",
            on_attempt=self._attempt_cb(),
            dataset=self.dataset,
        )
        # 1 Gemini call
        rdir = self.rounds_dir / f"round_{round_idx:02d}"
        # Partition validation BEFORE anything is persisted or applied:
        # every NOVEL type sent to Quality gets exactly one verdict.
        # Deterministic signature duplicates were removed before
        # constructing V, so they are not part of Quality's partition.
        valid_ids = {b.get("type", {}).get("id") for b in bundles}
        approved_ids = {t.get("type_id") for t in self.approved_pool}
        validate_quality_partition(verdict, valid_ids, approved_ids)
        auto = self._round_auto.get(round_idx, {"auto_rejected": [],
                                                "dropped_templates": []})
        det_dups = self._round_det_dups.get(round_idx, [])
        record = {
            "round": round_idx,
            "accepted_new": verdict.get("accepted_new", []),
            "duplicates": verdict.get("duplicates", []),
            "rejected": verdict.get("rejected", []),
            "auto_rejected": auto["auto_rejected"],
            "dropped_templates": auto["dropped_templates"],
        }
        write_json(rdir / "04_quality.json", record)

        accepted = apply_accepted_types(
            self.approved_pool, self._round_types.get(round_idx, {}),
            self._round_texts.get(round_idx, {}),
            verdict, round_idx,
            self._round_owners.get(round_idx, {}),
        )
        # Commit point (§29): only accepted types with kept [KEY]
        # templates may commit bindings. Nothing else mutates state.
        self._commit_round_bindings(round_idx, accepted)
        self._commit_family_bases(round_idx, verdict, accepted, bundles)
        offered = sum(len(b.get("templates", [])) + len(b.get("paraphrases", []))
                      for b in bundles)
        kept_total = sum(len(a.get("keep_template_ids", []))
                         for a in verdict.get("accepted_new", []))
        print(f"[R{round_idx}] Quality: offered {offered} templates, "
              f"kept {kept_total}")

        decision = self.controller.register_round(
            round_idx, len(accepted), generated_types
        )
        n_llm_reject = len(verdict.get("rejected", []))
        n_auto = len(record["auto_rejected"])
        n_det = len(det_dups)
        _assert_round_accounting(
            round_idx, generated_types, len(accepted),
            len(verdict.get("duplicates", [])), n_llm_reject, n_auto,
            "quality", det_n=n_det)
        stats = build_round_stats(
            round_idx=round_idx,
            generated_types=generated_types,
            accepted_new_types=len(accepted),
            duplicates=len(verdict.get("duplicates", [])),
            deterministic_duplicates=n_det,
            rejected=n_llm_reject + n_auto,
            new_type_rate=decision["new_type_rate"],
            total_approved_types=len(self.approved_pool),
        )
        self.history.append(stats)
        write_json(rdir / "approved_types_after.json", {
            "round": round_idx,
            "total_approved_types": len(self.approved_pool),
            "approved_types": self.approved_pool,
        })

        print(f"[R{round_idx}] Accepted new: {len(accepted)} | "
              f"duplicates: {stats['duplicates']} "
              f"(+{n_det} deterministic signature) | "
              f"rejected: {stats['rejected']} "
              f"({n_llm_reject} judge + {n_auto} deterministic)")
        print(f"[R{round_idx}] new_type_rate: {stats['new_type_rate']:.3f} | "
              f"approved total: {len(self.approved_pool)}")

        self._completed.append(f"quality_round_{round_idx}")
        if decision["stop"]:
            self.stop_reason = decision["stop_reason"]
            print("")
            print(f"STOP: {self.stop_reason}")
            print("Continue: NO")
            self._save_state(
                completed=list(self._completed),
                current_round=round_idx, next_action="finalize",
                stopped=True, stop_reason=self.stop_reason)
            return True
        print("Continue: YES")
        self._save_state(
            completed=list(self._completed), current_round=round_idx,
            next_action=(f"style_round_{round_idx + 1}"
                         if round_idx < self.controller.max_rounds
                         else "finalize"),
            stopped=False, stop_reason=None)
        return False

    # -- finalize (no LLM; safe to re-run) ------------------------------
    def _final_key_realizations(self) -> dict:
        """Bindings actually needed by final kept [KEY] templates (§33/34).

        needed = answer-concept fields of final kept templates containing
        [KEY]. Emitted mapping = {f: approved[f]}. Fails closed when a
        needed field has no approved binding (§50) — never emit an
        unresolvable [KEY] pool.
        """
        needed: list[str] = []
        for t in self.approved_pool:
            field = get_answer_concept_field(t)
            if field is None:
                continue
            if any("[KEY]" in (k.get("text", "") or "")
                   for k in t.get("kept_templates", [])):
                if field not in needed:
                    needed.append(field)
        final = {}
        for field in needed:
            if field not in self.approved_key_realizations:
                raise ValueError(
                    f"Final [KEY] templates need field {field!r} with no "
                    "approved realization; refusing to emit finals.")
            final[field] = self.approved_key_realizations[field]
        return final

    def _preflight_final_answer_references(self) -> None:
        """Fail closed on invalid approved structured answers (§26).

        Re-validates EVERY final approved type with the same central
        validate_answer_references() the gate uses, BEFORE render_preview
        and BEFORE any final artifact is written. Raises
        final_invalid_answer_reference:<type_id>:<reason>. Never reached
        in a correctly generated new run; defense against state/replay
        bugs and old-contract pool entries.
        """
        from src.autonomous_qa.core.validity import validate_answer_references

        for t in self.approved_pool:
            problems = validate_answer_references(
                t.get("answer", {}), schema_fields=self.schema_fields,
                field_roles=self.field_roles, hidden_fields=self.hidden_fields)
            if problems:
                raise ValueError(
                    f"final_invalid_answer_reference:{t.get('type_id')}:"
                    + ";".join(problems))

    def _preflight_final_signature_uniqueness(self) -> None:
        """Fail closed on duplicate executable signatures in finals (§61).

        Computes the signature for every final approved type and requires
        uniqueness. Raises
        final_duplicate_executable_signature:<A>:<B> before any final
        artifact is written. Defense-in-depth; unreachable post-patch.
        """
        from src.autonomous_qa.core.validity import executable_signature

        seen: dict[tuple, str] = {}
        for t in self.approved_pool:
            sig = executable_signature(t, schema_fields=self.schema_fields)
            if sig in seen:
                raise ValueError(
                    "final_duplicate_executable_signature:"
                    f"{seen[sig]}:{t.get('type_id')}")
            seen[sig] = t.get("type_id")

    def _finalize(self) -> list[dict]:
        self._assert_final_invariants()
        self._preflight_final_answer_references()
        self._preflight_final_signature_uniqueness()
        final_kr = self._final_key_realizations()
        derived_mappings = {
            k: v["mapping"] for k, v in self._relation_cache.items()
            if isinstance(v, dict) and v.get("valid") and v.get("mapping")
        }
        # Separate discovery pool from evaluation pool (§E, §H)
        evaluation_eligible_types = [
            t for t in self.approved_pool
            if is_evaluation_eligible_type(t, self.evaluation_policies, self.schema_fields)
        ]

        preview = render_preview(
            rows=self.rows,
            approved_types=evaluation_eligible_types,
            key_realizations=final_kr,
            n=CONFIG["preview"]["num_samples"],
            random_seed=CONFIG["preview"]["random_seed"],
            dataset=self.dataset,
            schema_fields=self.schema_fields,
            derived_mappings=derived_mappings,
        )
        self._assert_preview_invariants(preview)

        preview_path = self.out_dir / "preview.jsonl"
        with open(preview_path, "w", encoding="utf-8") as f:
            for item in preview:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        write_json(self.out_dir / "final_approved_types.json", {
            "dataset": self.dataset,
            "total_approved_types": len(self.approved_pool),
            "approved_types": self.approved_pool,
        })
        write_json(self.out_dir / "evaluation_eligible_types.json", {
            "dataset": self.dataset,
            "total_eligible_types": len(evaluation_eligible_types),
            "evaluation_eligible_types": evaluation_eligible_types,
        })
        write_json(self.out_dir / "final_template_pool.json", {
            "dataset": self.dataset,
            "key_realizations": final_kr,
            "templates": self._flat_template_pool(),
        })
        write_json(self.out_dir / "loop_summary.json", {
            "dataset": self.dataset,
            "rounds_run": len(self.history),
            "stop_reason": self.stop_reason,
            "rounds": list(self.history),
        })
        # Human audit list (never auto-fed back into persistent banks).
        write_json(self.out_dir / "approved_type_candidates.json", {
            "candidates": [
                {"type_id": t["type_id"], "name": t["name"],
                 "goal": t["goal"], "answer_rule": t["answer_rule"]}
                for t in self.approved_pool
            ],
        })

        if self.real_llm:
            from src.autonomous_qa.authoring.llm_client import BUDGET as _BUDGET

            print("")
            print(f"Final approved types: {len(self.approved_pool)}")
            print(f"Final key realizations: {len(final_kr)}")
            print(f"QA previews: {len(preview)}")
            print(f"Real LLM calls this invocation: "
                  f"{self._new_calls()}/{_BUDGET.max_calls}")
            print("========================================")
        else:
            print(f"Final approved types: {len(self.approved_pool)}")
            print(f"Final key realizations: {len(final_kr)}")
            print(f"QA preview generated: {len(preview)}")
            print("Real Gemini calls: 0")

        write_json(self.out_dir / "run_meta.json", {
            "dataset": self.dataset,
            "mode": "real" if self.real_llm else "mock",
            "run_id": self.run_id,
            "rounds_run": len(self.history),
            "stop_reason": self.stop_reason,
            "preview_count": len(preview),
        })
        return preview

    def _flat_template_pool(self) -> list[dict]:
        flat = []
        for t in self.approved_pool:
            for k in t.get("kept_templates", []):
                tid = k["template_id"]
                flat.append({
                    "template_id": tid,
                    "question_type_id": t["type_id"],
                    "text": k["text"],
                    "origin": ("paraphrase" if "_P" in tid else "template"),
                    "round": t.get("round_accepted"),
                })
        return flat

    def _assert_final_invariants(self) -> None:
        """Fail closed before writing finals (§42 final output invariants).

        Stored templates legitimately carry [KEY] (bound at render time);
        only [VALUE] leakage and empty keep lists are hard failures here.
        Rendered questions are checked separately in
        _assert_preview_invariants.
        """
        for t in self.approved_pool:
            kept = t.get("kept_templates", [])
            if not kept:
                raise ValueError(
                    f"Approved type {t.get('type_id')} has no kept "
                    "templates.")
            for k in kept:
                text = k.get("text", "")
                if "[VALUE]" in text:
                    raise ValueError(
                        f"Final template {k.get('template_id')} leaks "
                        "the gold answer ([VALUE]).")
        # Equality answers are enforced per item after rendering
        # (see _assert_preview_invariants).

    def _assert_preview_invariants(self, preview: list[dict]) -> None:
        """Fail closed on leaked/unresolved preview content (§42)."""
        import re as _re

        bracket = _re.compile(r"\[([A-Za-z_][A-Za-z0-9_]*)\]")
        for item in preview:
            question = item.get("question", "")
            if "[VALUE]" in question:
                raise ValueError(
                    f"Preview {item.get('id')} leaks the gold answer "
                    "([VALUE] in question).")
            if bracket.search(question):
                raise ValueError(
                    f"Preview {item.get('id')} has an unresolved "
                    f"placeholder: {question!r}.")
            src = item.get("answer_source", {}) or {}
            if src.get("kind") == "equality" and item.get("answer") not in (
                    "Có", "Không"):
                raise ValueError(
                    f"Preview {item.get('id')} has non-boolean equality "
                    f"answer: {item.get('answer')!r}.")

    def _new_calls(self) -> int:
        from src.autonomous_qa.authoring.llm_client import BUDGET

        return BUDGET.count - self._calls_before

    def _post_gate_terminate(self, round_idx: int, gate: dict,
                               generated_types: int) -> bool:
        """ONE canonical post-Gate decision (§16).

        If the deterministic Gate returned zero valid types: persist the
        terminal-round state (Gate complete, valid=0, stop reason), set
        Continue NO, and return True (TERMINATE_DISCOVERY — finalize the
        prior approved pool, never call Paraphrase/Quality). Otherwise
        record the gate audit and return False (CONTINUE_TO_PARAPHRASE).

        BOTH fresh execution and resumed-continuation execution MUST use
        this helper. The persisted next_action="finalize" checkpoint lets
        a crash before finalization resume directly at finalization.
        """
        self._round_auto[round_idx] = {
            "auto_rejected": gate["auto_rejected"],
            "dropped_templates": gate["dropped_templates"],
        }
        if gate["valid_types"]:
            return False
        # Zero-valid short-circuit: no Paraphrase call, no Quality call,
        # no fake calls, no budget spent beyond the two design calls above.
        print(f"[R{round_idx}] Gate: 0 valid types — short-circuit, "
              f"no Paraphrase/Quality calls.")
        self.stop_reason = "no_valid_types_after_gate"
        print("")
        print(f"STOP: {self.stop_reason}")
        print("Continue: NO")
        self.history.append(build_round_stats(
            round_idx=round_idx,
            generated_types=generated_types,
            accepted_new_types=0,
            duplicates=0,
            rejected=len(gate["auto_rejected"]),
            new_type_rate=0.0,
            total_approved_types=len(self.approved_pool),
        ))
        self._save_state(
            completed=list(self._completed), current_round=round_idx,
            next_action="finalize",
            stopped=True, stop_reason=self.stop_reason)
        return True

    def _run_full_round(self, round_idx: int) -> bool:
        """Execute one full round. Returns True when the loop must stop."""
        print("")
        print(f"========== ROUND {round_idx} ==========")
        types = self._do_style_round(round_idx)
        generated = len(types)
        self._round_types[round_idx] = {t["id"]: t for t in types}

        tpl_out = self._do_template_round(round_idx, types)
        gate = self._gate_round(
            round_idx, types, tpl_out.get("templates", []))

        if self._post_gate_terminate(round_idx, gate, generated):
            return True

        novel = self._dedup_and_build_candidates(round_idx, types, gate)
        det_dups = self._round_det_dups.get(round_idx, [])
        if self._post_signature_dedup_terminate(
                round_idx, gate, novel, det_dups, generated):
            return True

        novel_templates = [t for t in gate["valid_templates"]
                           if t.get("question_type_id")
                           in {t.get("id") for t in novel}]
        paraphrases = self._do_paraphrase_round(
            round_idx, novel_templates)
        judging = self._prepare_judging(
            round_idx, novel, novel_templates,
            paraphrases)

        return self._do_quality_round(
            round_idx, judging["bundles"], generated)

    def execute(self) -> list[dict]:
        raise NotImplementedError


def _build_bundles(valid_types: list[dict], valid_templates: list[dict],
                   paraphrases: list[dict],
                   key_context: dict | None = None) -> tuple[list[dict], dict]:
    """Bundle each valid type with its templates + paraphrases.

    Returns (bundles, texts_by_id). Paraphrase variants attach to the type
    owning their source template; orphans (unknown source) are dropped.
    When key_context (effective field -> realization) is provided, every
    [KEY] template entry also carries the effective rendered concept so
    Quality judges the wording it will actually become.
    """
    from src.autonomous_qa.core.validity import get_answer_concept_field

    def _with_effective(entry: dict, qtype: dict) -> dict:
        if "[KEY]" not in (entry.get("text", "") or ""):
            return dict(entry)
        if key_context is None:
            return dict(entry, effective_key_realization=None)
        field = get_answer_concept_field(qtype)
        entry = dict(entry)
        entry["effective_key_realization"] = key_context.get(field)
        return entry

    by_type: dict[str, list[dict]] = {}
    for tpl in valid_templates:
        by_type.setdefault(tpl["question_type_id"], []).append(tpl)
    texts_by_id = {t["template_id"]: t["text"] for t in valid_templates}
    para_by_type: dict[str, list[dict]] = {}
    for p in paraphrases:
        src = p.get("source_template_id")
        owner = next(
            (t["question_type_id"] for t in valid_templates
             if t["template_id"] == src), None)
        if owner is None or p["template_id"] in texts_by_id:
            continue
        texts_by_id[p["template_id"]] = p["text"]
        para_by_type.setdefault(owner, []).append(p)
    bundles = []
    for t in valid_types:
        bundles.append({
            "type": {k: t.get(k) for k in (
                "id", "name", "goal", "uses", "input_count",
                "answer_rule", "answer")},
            "templates": [
                _with_effective(
                    {"template_id": x["template_id"], "text": x["text"]}, t)
                for x in by_type.get(t["id"], [])],
            "paraphrases": [
                _with_effective(
                    {"template_id": x["template_id"], "text": x["text"]}, t)
                for x in para_by_type.get(t["id"], [])],
        })
    return bundles, texts_by_id


class _FreshRun(_RunBase):
    """A brand-new run: all stages execute in order with checkpoints."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._calls_before = 0
        self._completed: list[str] = []
        self._round_types: dict[int, dict] = {}
        self._round_texts: dict[int, dict] = {}
        self._round_auto: dict[int, dict] = {}
        self._round_owners: dict[int, dict] = {}

    def execute(self) -> list[dict]:
        from src.autonomous_qa.authoring.llm_client import BUDGET

        self._calls_before = BUDGET.count
        max_rounds = self.controller.max_rounds
        for round_idx in range(1, max_rounds + 1):
            if self._run_full_round(round_idx):
                break

        assert self.stop_reason is not None  # controller always stops

        preview = self._finalize()
        self._completed.append("finalize")
        # A zero-valid run is terminal: next_action stays null so no
        # resume can ever issue new calls for it.
        terminal = (self.stop_reason == "no_valid_types_after_gate")
        self._save_state(completed=list(self._completed),
                         current_round=len(self.history),
                         next_action=None if terminal else "done",
                         stopped=True,
                         stop_reason=self.stop_reason)
        return preview


class _ResumedRun(_RunBase):
    """Continue a crashed real run from its checkpoint. Read-only reuse."""

    def __init__(self, **kwargs):
        kwargs.setdefault("run_id", None)
        kwargs.setdefault("client", None)
        super().__init__(**kwargs)
        self._calls_before = 0
        self.state: dict = {}
        self._completed: list[str] = []
        self._pending: dict | None = None
        self._next_action: str = ""
        self._derived_completed: list[str] = []
        self._streak_after_round: dict[int, int] = {}
        self._round_types: dict[int, dict] = {}
        self._round_texts: dict[int, dict] = {}
        self._round_auto: dict[int, dict] = {}
        self._round_owners: dict[int, dict] = {}

    def load_and_validate(self) -> None:
        """Load checkpoint + artifacts. No network, no budget touched."""
        state_path = self.out_dir / STATE_FILE
        self.state = read_json(state_path)
        self.run_id = self.state.get("run_id")

        if self.state.get("mode") != "real":
            raise ValueError("Only real runs can be resumed.")
        if self.state.get("dataset") != self.dataset:
            raise ValueError(
                f"Checkpoint dataset {self.state.get('dataset')!r} != "
                f"requested {self.dataset!r}.")
        if self.state.get("run_id") != self.out_dir.name:
            raise ValueError("Checkpoint run_id does not match directory.")
        if self.state.get("next_action") == "done":
            raise ValueError(
                f"Run {self.state['run_id']} already finished "
                "(next_action=done). Nothing to resume.")
        if self.state.get("next_action") is None:
            raise ValueError(
                f"Run {self.state.get('run_id')} already finished "
                f"(terminal stop: {self.state.get('stop_reason')}). "
                "Nothing to resume.")
        if self.state.get("next_action") in self.state.get("completed", []):
            raise ValueError(
                f"Checkpoint next_action "
                f"{self.state.get('next_action')!r} is already completed; "
                "refusing to rerun a successful stage.")
        if self.state.get("loop_config") != self._current_loop_config():
            raise ValueError(
                "Loop config changed since the checkpoint; refusing to "
                "resume with different stopping semantics.")
        if self.state.get("template_contract_version") != TEMPLATE_CONTRACT_VERSION:
            raise ValueError(
                "Question Template contract changed since the checkpoint "
                f"(checkpoint={self.state.get('template_contract_version')!r} "
                f"!= current={TEMPLATE_CONTRACT_VERSION!r}); pre-patch runs "
                "use raw-field-keyed key_realizations and cannot be resumed "
                "or migrated. Start a fresh run.")
        if (self.state.get("answer_reference_contract_version")
                != ANSWER_REFERENCE_CONTRACT_VERSION):
            raise ValueError(
                "Answer-reference contract changed since the checkpoint "
                "(checkpoint="
                f"{self.state.get('answer_reference_contract_version')!r} "
                f"!= current={ANSWER_REFERENCE_CONTRACT_VERSION!r}); runs "
                "produced under old suffix-normalization gate semantics "
                "cannot be resumed or migrated. Start a fresh run.")
        if (self.state.get("answer_shape_contract_version")
                != ANSWER_SHAPE_CONTRACT_VERSION):
            raise ValueError(
                "Answer-shape contract changed since the checkpoint "
                "(checkpoint="
                f"{self.state.get('answer_shape_contract_version')!r} "
                f"!= current={ANSWER_SHAPE_CONTRACT_VERSION!r}); runs "
                "produced without canonical answer-shape enforcement "
                "cannot be resumed or migrated. Start a fresh run.")
        if (self.state.get("executable_signature_contract_version")
                != EXECUTABLE_SIGNATURE_CONTRACT_VERSION):
            raise ValueError(
                "Executable-signature contract changed since the checkpoint "
                "(checkpoint=" +
                f"{self.state.get('executable_signature_contract_version')!r} "
                f"!= current={EXECUTABLE_SIGNATURE_CONTRACT_VERSION!r}); runs "
                "produced without executable-signature dedup cannot be "
                "resumed or migrated. Start a fresh run.")
        if (self.state.get("derived_field_contract_version")
                != DERIVED_FIELD_CONTRACT_VERSION):
            raise ValueError(
                "Derived-field contract changed since the checkpoint "
                "(checkpoint=" +
                f"{self.state.get('derived_field_contract_version')!r} "
                f"!= current={DERIVED_FIELD_CONTRACT_VERSION!r}); runs "
                "produced without data-backed derived-relation validation "
                "cannot be resumed or migrated. Start a fresh run.")
        if (self.state.get("knowledge_contract_version")
                and self.state.get("knowledge_contract_version") != KNOWLEDGE_CONTRACT_VERSION):
            raise ValueError(
                "Knowledge contract changed since the checkpoint "
                f"(checkpoint={self.state.get('knowledge_contract_version')!r} "
                f"!= current={KNOWLEDGE_CONTRACT_VERSION!r}); runs produced under "
                "different knowledge contracts cannot be resumed. Start a fresh run.")
        for step in self.state.get("completed", []):
            self._require_artifact(step)

        if "family_template_bank" in self.state:
            self.family_template_bank = {
                deserialize_family_signature(k): v
                for k, v in self.state.get("family_template_bank", {}).items()
            }

        # Replay completed rounds to rebuild pool/history/controller.
        # This also adopts any fully-written rounds the checkpoint lags
        # behind, so attempts are counted from disk truth below.
        self._replay()

        # Attempts: reconstruct PAID attempts from durable raw attempt
        # files (§13-§15, §19), independently from successful artifacts.
        # A raw attempt means paid_attempt=TRUE even if validation failed.
        # Success state comes from _replay() artifacts, not raw files.
        from src.autonomous_qa.authoring.llm_client import (
            BUDGET,
            require_contiguous,
            reset_attempts,
            scan_raw_attempts,
        )
        from src.common.io import write_json as _atomic_write_json

        spent: dict[str, int] = {}
        for step in self._completed:
            stage = _stage_of_step(step)
            if stage is not None:
                spent[stage] = spent.get(stage, 0) + 1

        disk_lists = scan_raw_attempts(self.raw_dir)
        disk_spent: dict[str, int] = {}
        for stage in BUDGET.allowed_calls:
            disk_spent[stage] = require_contiguous(
                disk_lists.get(stage, []), stage)

        _sa = self.state.get("stage_attempts")
        _sc = self.state.get("spent_calls") or {}
        if not _sa:
            # Legacy checkpoint without stage_attempts: fall back to spent_calls.
            merged = {s: int(_sc.get(s, 0)) for s in BUDGET.allowed_calls}
        else:
            # Both accounting fields must agree; tampering with either refuses.
            merged = {}
            for stage in BUDGET.allowed_calls:
                c1 = int(_sa.get(stage, 0))
                c2 = int(_sc.get(stage, 0))
                if c1 != c2:
                    raise ValueError(
                        f"Checkpoint spent calls inconsistent for '{stage}': "
                        f"stage_attempts={c1} vs spent_calls={c2}; "
                        "refusing to resume.")
                merged[stage] = c1
        state_attempts = merged
        reconciled: dict[str, int] = {}
        dirty = False
        for stage in BUDGET.allowed_calls:
            c = int(state_attempts.get(stage, 0))
            d = int(disk_spent.get(stage, 0))
            if d == c:
                reconciled[stage] = c
            elif d == c + 1:
                # Crashed after raw atomic write, before checkpoint write.
                # Reconcile checkpoint UPWARD only; no LLM call.
                reconciled[stage] = d
                dirty = True
                print(f"[resume] raw-ahead-by-one for '{stage}': "
                      f"checkpoint {c} -> {d} (no new call)")
            elif c > d:
                raise ValueError(
                    f"Checkpoint claims {c} spent calls for '{stage}' but "
                    f"durable raw inventory shows {d}; refusing to resume.")
            else:  # d > c + 1
                raise ValueError(
                    f"Durable raw inventory shows {d} paid calls for "
                    f"'{stage}' but checkpoint claims {c}; state inconsistent "
                    "beyond the one-write crash window; refusing to resume.")
        if dirty:
            self.state["stage_attempts"] = dict(reconciled)
            self.state["spent_calls"] = dict(reconciled)
            self.state["attempted_calls_total"] = sum(reconciled.values())
            _atomic_write_json(self.out_dir / STATE_FILE, self.state)
            state_attempts = dict(reconciled)
        else:
            # Checkpoint counts must agree with durable inventory.
            for stage in BUDGET.allowed_calls:
                if int(state_attempts.get(stage, 0)) != disk_spent[stage]:
                    # Unreachable given the 4-case rules above; defensive.
                    raise ValueError(
                        f"Paid-call mismatch for '{stage}': checkpoint "
                        f"{state_attempts.get(stage, 0)} vs disk "
                        f"{disk_spent[stage]}; refusing to resume.")

        for stage, allowed in BUDGET.allowed_calls.items():
            n = int(reconciled.get(stage, int(state_attempts.get(stage, 0))))
            if n > allowed:
                raise ValueError(
                    f"Run already spent {n} paid attempts for '{stage}' "
                    f"(cap {allowed}); refusing to resume.")
            for _ in range(n):
                BUDGET.consume(stage)
        # Seed the attempt tracker so numbering continues (attempt_N+1…).
        # Successes are exactly the disk-derived completed paid stages.
        reset_attempts(
            attempted=sum(int(v) for v in reconciled.values()),
            successful=sum(spent.values()),
            stage_attempts=dict(reconciled),
            stage_successes=dict(spent),
            last_finish_reason=self.state.get("last_finish_reason"),
            last_failed_stage=self.state.get("last_failed_stage"),
        )

    def _current_loop_config(self) -> dict:
        return {
            "max_rounds": self.controller.max_rounds,
            "min_new_type_rate": self.controller.min_rate,
            "saturation_patience": self.controller.saturation_patience,
            "immediate_stop_if_zero_new":
                self.controller.immediate_stop_if_zero_new,
        }

    def _require_artifact(self, step: str) -> None:
        if step == "finalize":
            expected = self.out_dir / "preview.jsonl"
        else:
            m = re.fullmatch(
                r"(style|template|paraphrase|quality)_round_(\d+)", step)
            if m is None:
                raise ValueError(
                    f"Checkpoint step {step!r} is from an incompatible "
                    "pipeline version; refusing to resume.")
            kind, r = m.group(1), int(m.group(2))
            names = {"style": "01_question_styles.json",
                     "template": "02_templates.json",
                     "paraphrase": "03_paraphrases.json",
                     "quality": "04_quality.json"}
            expected = round_dir(self.rounds_dir, r) / names[kind]
            if kind == "quality":
                pass  # approved_types_after.json is derived:
                # verified/repaired in replay, never required here.
        if not expected.is_file():
            raise FileNotFoundError(
                f"Cannot resume: completed step {step!r} lacks {expected}")

    def _replay(self) -> None:
        """Rebuild pool/history/controller from persisted round files.

        Fully-written rounds are adopted even if the checkpoint lags one
        write behind. Verdicts without pool snapshots are applied in-memory
        with ZERO new LLM calls; approved_types_after.json is (re)written
        deterministically when absent but never overwritten.
        """
        self.key_realizations = {}
        max_rounds = self.controller.max_rounds
        self._derived_completed = []
        self._derived_next = "style_round_1"
        self._streak_after_round = {}
        round_idx = 1
        while True:
            rdir = round_dir(self.rounds_dir, round_idx)
            style_path = rdir / "01_question_styles.json"
            if not style_path.is_file():
                # Round never started (or style attempt failed): retry here.
                if round_idx <= max_rounds:
                    self._derived_next = f"style_round_{round_idx}"
                break
            styles = read_json(style_path)
            types = styles.get("question_types", [])
            self._round_types[round_idx] = {t["id"]: t for t in types}
            generated = len(types)
            self._derived_completed.append(f"style_round_{round_idx}")

            tpl_path = rdir / "02_templates.json"
            if not tpl_path.is_file():
                self._derived_next = f"template_round_{round_idx}"
                break
            tpl_out = read_json(tpl_path)
            type_phrases, combined = self._process_round_family_templates(
                round_idx, types, tpl_out
            )
            self._round_type_phrases[round_idx] = dict(type_phrases)
            gate = gate_round(
                types, combined,
                type_key_realizations=self._round_type_phrases.get(round_idx, {}),
                approved_key_realizations=self.approved_key_realizations,
                dataset_rows=self.rows,
                relation_cache=self._relation_cache,
                schema_fields=self.schema_fields,
                hidden_fields=self.hidden_fields,
                field_roles=self.field_roles)
            self._round_auto[round_idx] = {
                "auto_rejected": gate["auto_rejected"],
                "dropped_templates": gate["dropped_templates"],
            }
            self._derived_completed.append(f"template_round_{round_idx}")

            if not gate["valid_types"]:
                # Terminal zero-valid Gate persisted before finalization:
                # resume targets finalization, never paraphrase/quality.
                # Accounting mirrors the canonical post-Gate helper.
                self.history.append(build_round_stats(
                    round_idx=round_idx,
                    generated_types=generated,
                    accepted_new_types=0,
                    duplicates=0,
                    rejected=len(gate["auto_rejected"]),
                    new_type_rate=0.0,
                    total_approved_types=len(self.approved_pool),
                ))
                self.stop_reason = "no_valid_types_after_gate"
                self._derived_next = "finalize"
                break

            # Signature dedup replays identically (no separate resume
            # logic): novel types only proceed downstream.
            novel = self._dedup_and_build_candidates(
                round_idx, types, gate, persist_audit=False)
            det_dups = self._round_det_dups.get(round_idx, [])
            novel_ids = {t.get("id") for t in novel}
            novel_templates = [t for t in gate["valid_templates"]
                               if t.get("question_type_id") in novel_ids]
            if not novel:
                # Terminal dedup: gate valid > 0 but every survivor is a
                # deterministic duplicate. Resume targets finalization.
                self.history.append(build_round_stats(
                    round_idx=round_idx,
                    generated_types=generated,
                    accepted_new_types=0,
                    duplicates=0,
                    deterministic_duplicates=len(det_dups),
                    rejected=len(gate["auto_rejected"]),
                    new_type_rate=0.0,
                    total_approved_types=len(self.approved_pool),
                ))
                self.stop_reason = "no_novel_types_after_signature_dedup"
                self._derived_next = "finalize"
                break

            para_path = rdir / "03_paraphrases.json"
            if not para_path.is_file():
                self._derived_next = f"paraphrase_round_{round_idx}"
                break
            paraphrases = read_json(para_path).get("paraphrases", [])
            judging = self._prepare_judging(
                round_idx, novel, novel_templates,
                paraphrases)
            bundles = judging["bundles"]
            texts = judging["texts"]
            self._derived_completed.append(f"paraphrase_round_{round_idx}")

            qual_path = rdir / "04_quality.json"
            if not qual_path.is_file():
                # Crashed between paraphrase and quality: resume here.
                # Gate results are deterministic: recompute so the
                # quality record keeps its auto-rejected/dropped lists.
                reloaded = self._reload_gate(round_idx)
                self._round_auto[round_idx] = {
                    "auto_rejected": reloaded["auto_rejected"],
                    "dropped_templates": reloaded["dropped_templates"],
                }
                self._pending = {
                    "round_idx": round_idx,
                    "bundles": bundles,
                    "generated": generated,
                }
                self._derived_next = f"quality_round_{round_idx}"
                break
            record = read_json(qual_path)
            verdict = {
                "accepted_new": record.get("accepted_new", []),
                "duplicates": record.get("duplicates", []),
                "rejected": record.get("rejected", []),
            }
            # Quality partition applies to novel candidates only:
            # deterministic signature duplicates were removed before V.
            validate_quality_partition(
                verdict, set(novel_ids),
                {t.get("type_id") for t in self.approved_pool})
            accepted = apply_accepted_types(
                self.approved_pool, self._round_types[round_idx], texts,
                verdict, round_idx,
                self._round_owners.get(round_idx, {}))
            self._commit_round_bindings(round_idx, accepted)
            self._commit_family_bases(round_idx, verdict, accepted, bundles)
            n_llm_reject = len(verdict.get("rejected", []))
            n_auto_replay = len(
                self._round_auto.get(round_idx, {}).get("auto_rejected", []))
            n_det_replay = len(det_dups)
            _assert_round_accounting(
                round_idx, generated, len(accepted),
                len(verdict.get("duplicates", [])), n_llm_reject,
                n_auto_replay, "replay", det_n=n_det_replay)
            decision = self.controller.register_round(
                round_idx, len(accepted), generated)
            stats = build_round_stats(
                round_idx=round_idx,
                generated_types=generated,
                accepted_new_types=len(accepted),
                duplicates=len(verdict.get("duplicates", [])),
                deterministic_duplicates=n_det_replay,
                rejected=(len(verdict.get("rejected", []))
                          + len(record.get("auto_rejected", []))),
                new_type_rate=decision["new_type_rate"],
                total_approved_types=len(self.approved_pool),
            )
            self.history.append(stats)
            self._streak_after_round[round_idx] = (
                self.controller.low_gain_streak)
            after_path = rdir / "approved_types_after.json"
            if after_path.is_file():
                after = read_json(after_path)
                if (after.get("total_approved_types")
                        != len(self.approved_pool)):
                    raise ValueError(
                        f"Round {round_idx} approved_types_after.json "
                        "inconsistent with replay; refusing to resume.")
            else:
                write_json(after_path, {
                    "round": round_idx,
                    "total_approved_types": len(self.approved_pool),
                    "approved_types": self.approved_pool,
                })
            self._derived_completed.append(f"quality_round_{round_idx}")
            if decision["stop"]:
                self.stop_reason = decision["stop_reason"]
                self._derived_next = "finalize"
                break
            self._derived_next = f"style_round_{round_idx + 1}"
            round_idx += 1
            if round_idx > max_rounds:
                break

        self._adopt_derived()

    def _adopt_derived(self) -> None:
        """Adopt disk truth when the checkpoint lags behind; never reverse."""
        claimed = list(self.state.get("completed", []))
        if claimed != self._derived_completed[:len(claimed)]:
            # Tolerate exactly one trailing "finalize" when its artifact
            # exists (crash inside finalize): re-running it is idempotent.
            trailing = claimed[len(self._derived_completed):]
            if (claimed[:len(self._derived_completed)]
                    == self._derived_completed
                    and trailing == ["finalize"]
                    and self._derived_next == "finalize"
                    and (self.out_dir / "preview.jsonl").is_file()):
                self._derived_completed.append("finalize")
            else:
                raise ValueError(
                    "Checkpoint completed-steps diverge from on-disk "
                    "artifacts; refusing to resume.")
        claimed_rounds = sum(
            1 for r in self._streak_after_round
            if f"quality_round_{r}" in claimed)
        expected_streak = (self._streak_after_round.get(claimed_rounds, 0)
                           if claimed_rounds else 0)
        if expected_streak != self.state.get("low_gain_streak", 0):
            raise ValueError(
                "Replayed low_gain_streak does not match checkpoint; "
                "refusing to resume.")
        if len(self._derived_completed) > len(claimed):
            print(f"NOTE: checkpoint lagged behind disk; adopting "
                  f"{len(self._derived_completed) - len(claimed)} "
                  f"completed step(s) with zero new calls.")
        self._completed = list(self._derived_completed)
        self._next_action = self._derived_next

    def attach_client(self, client) -> None:
        from src.autonomous_qa.authoring.llm_client import BUDGET

        self.client = client
        self.real_llm = True
        self._calls_before = BUDGET.count

    def execute(self) -> list[dict]:
        # Ground truth comes from replayed disk artifacts; the stored
        # next_action is only a cross-check (checkpoint may lag one write).
        next_action = self._next_action
        if next_action != self.state.get("next_action"):
            print(f"NOTE: stored next_action was {self.state.get('next_action')!r}; "
                  f"proceeding with replay-derived {next_action!r}.")
        max_rounds = self.controller.max_rounds

        m = re.fullmatch(
            r"(style|template|paraphrase|quality)_round_(\d+)", next_action)
        if m is None:
            if next_action == "finalize":
                return self._finalize_and_close()
            raise ValueError(f"Unknown next_action {next_action!r}.")
        kind, start_round = m.group(1), int(m.group(2))

        if kind == "quality":
            # Paraphrase output already on disk: rebuild bundles, no new call.
            if self._pending is None:
                raise ValueError(
                    "Checkpoint inconsistent: quality round has no "
                    "persisted paraphrase candidates.")
            pending = self._pending
            print("")
            print(f"========== ROUND {start_round} (resumed) ==========")
            print(f"Types proposed: {pending['generated']} "
                  "(reused round artifacts)")
            self._current_action = f"quality_round_{start_round}"
            self._current_round = start_round
            if self._finish_quality_round(
                    start_round, pending["bundles"], pending["generated"]):
                return self._finalize_and_close()
            # Interrupted round now complete; every subsequent round is
            # whole: delegate to the canonical round runner (§20).
            for round_idx in range(start_round + 1, max_rounds + 1):
                if self._run_full_round(round_idx):
                    return self._finalize_and_close()
            assert self.stop_reason is not None
            return self._finalize_and_close()

        # Partial starting round: finish its remaining stages via the
        # stage cursor (shared post-Gate helper included), then delegate
        # every subsequent whole round to the canonical runner.
        order = ["style", "template", "paraphrase", "quality"]
        stages = order[order.index(kind):]
        print("")
        print(f"========== ROUND {start_round} (resumed) ==========")
        if self._run_partial_round(start_round, stages):
            return self._finalize_and_close()
        for round_idx in range(start_round + 1, max_rounds + 1):
            if self._run_full_round(round_idx):
                return self._finalize_and_close()
        assert self.stop_reason is not None
        return self._finalize_and_close()

    def _run_partial_round(self, round_idx: int,
                           stages: list[str]) -> bool:
        """Finish a resumed partial round. True when the loop must stop."""
        for stage in stages:
            if stage == "style":
                self._current_action = f"style_round_{round_idx}"
                self._current_round = round_idx
                types = self._do_style_round_resume_aware(round_idx)
                self._round_types[round_idx] = {
                    t["id"]: t for t in types}
            elif stage == "template":
                self._current_action = f"template_round_{round_idx}"
                self._current_round = round_idx
                # Re-read persisted styles (byte-for-byte reuse).
                persisted = read_json(
                    round_dir(self.rounds_dir, round_idx)
                    / "01_question_styles.json")
                types = persisted.get("question_types", [])
                self._round_types[round_idx] = {
                    t["id"]: t for t in types}
                tpl_out = self._do_template_round_resume_aware(
                    round_idx, types)
                gate = self._gate_round(
                    round_idx, types, tpl_out.get("templates", []))
                # SAME canonical post-Gate decisions as fresh execution:
                # zero valid terminates; otherwise signature dedup decides.
                if self._post_gate_terminate(round_idx, gate, len(types)):
                    return True
                novel = self._dedup_and_build_candidates(
                    round_idx, types, gate)
                det_dups = self._round_det_dups.get(round_idx, [])
                if self._post_signature_dedup_terminate(
                        round_idx, gate, novel, det_dups, len(types)):
                    return True
                self._round_gate_cache = {
                    "valid_types": novel,
                    "valid_templates": [
                        t for t in gate["valid_templates"]
                        if t.get("question_type_id")
                        in {t.get("id") for t in novel}],
                }
            elif stage == "paraphrase":
                self._current_action = f"paraphrase_round_{round_idx}"
                self._current_round = round_idx
                gate_cache = getattr(self, "_round_gate_cache", None)
                if gate_cache is None:
                    gate = self._reload_gate(round_idx)
                else:
                    gate = gate_cache
                paraphrases = self._do_paraphrase_round_resume_aware(
                    round_idx, gate["valid_templates"])
                judging = self._prepare_judging(
                    round_idx, gate["valid_types"],
                    gate["valid_templates"], paraphrases)
                self._round_bundles_cache = judging["bundles"]
            elif stage == "quality":
                self._current_action = f"quality_round_{round_idx}"
                self._current_round = round_idx
                bundles = getattr(self, "_round_bundles_cache", None)
                if bundles is None:
                    bundles, _ = self._reload_bundles(round_idx)
                generated = len(self._round_types[round_idx])
                if self._finish_quality_round(
                        round_idx, bundles, generated):
                    return True
        return False

    # -- resume-aware stage wrappers (persist + checkpoint each) --------
    def _do_style_round_resume_aware(self, round_idx: int) -> list[dict]:
        rdir = round_dir(self.rounds_dir, round_idx)
        rdir.mkdir(parents=True, exist_ok=True)
        return self._do_style_round(round_idx)

    def _do_template_round_resume_aware(
            self, round_idx: int, types: list[dict]) -> dict:
        return self._do_template_round(round_idx, types)

    def _do_paraphrase_round_resume_aware(
            self, round_idx: int, valid_templates: list[dict]) -> list[dict]:
        # _do_paraphrase_round persists candidates + checkpoints itself.
        self._current_action = f"paraphrase_round_{round_idx}"
        self._current_round = round_idx
        out = run_paraphrase_round(
            client=self.client, templates=valid_templates,
            round_idx=round_idx, real_llm=True,
            raw_path=self.raw_dir / f"paraphrase_round_{round_idx}.txt",
            on_attempt=self._on_attempt,
            dataset=self.dataset,
        )
        paraphrases = out.get("paraphrases", [])
        print(f"[R{round_idx}] Paraphrases: {len(paraphrases)} produced")
        rdir = round_dir(self.rounds_dir, round_idx)
        write_json(rdir / "03_paraphrases.json", out)
        self._completed.append(f"paraphrase_round_{round_idx}")
        self._save_state(
            completed=list(self._completed), current_round=round_idx,
            next_action=f"quality_round_{round_idx}",
            stopped=False, stop_reason=None)
        return paraphrases

    def _reload_gate(self, round_idx: int) -> dict:
        """Re-run the deterministic gate from persisted round files.

        Returns the gate with valid types/templates already filtered to
        executable-signature NOVEL survivors (same dedup as execution),
        so resume rebuilds identical Paraphrase/Quality inputs. The full
        gate audit (auto-rejected/dropped) is still recorded separately.
        """
        from src.autonomous_qa.core.validity import build_round_field_candidates as _build_cands

        rdir = round_dir(self.rounds_dir, round_idx)
        styles = read_json(rdir / "01_question_styles.json")
        tpl_out = read_json(rdir / "02_templates.json")
        round_types = styles.get("question_types", [])
        self._round_types[round_idx] = {t["id"]: t for t in round_types}
        self._round_type_phrases[round_idx] = self._validate_template_mapping(
            tpl_out.get("key_realizations", {}), round_idx,
            type_ids=[t.get("id") for t in round_types],
            templates=tpl_out.get("templates", []))
        gate = gate_round(
            round_types,
            tpl_out.get("templates", []),
            type_key_realizations=self._round_type_phrases.get(round_idx, {}),
            approved_key_realizations=self.approved_key_realizations,
            dataset_rows=self.rows,
            relation_cache=self._relation_cache,
            schema_fields=self.schema_fields,
            hidden_fields=self.hidden_fields,
            field_roles=self.field_roles)
        novel = self._dedup_and_build_candidates(
            round_idx, round_types, gate, persist_audit=False)
        novel_ids = {t.get("id") for t in novel}
        self._round_auto[round_idx] = {
            "auto_rejected": gate["auto_rejected"],
            "dropped_templates": gate["dropped_templates"],
        }
        return {
            "valid_types": novel,
            "valid_templates": [t for t in gate["valid_templates"]
                                if t.get("question_type_id") in novel_ids],
            "auto_rejected": gate["auto_rejected"],
            "dropped_templates": gate["dropped_templates"],
        }

    def _reload_bundles(self, round_idx: int) -> tuple[list[dict], dict]:
        gate = self._reload_gate(round_idx)
        rdir = round_dir(self.rounds_dir, round_idx)
        paraphrases = read_json(
            rdir / "03_paraphrases.json").get("paraphrases", [])
        judging = self._prepare_judging(
            round_idx, gate["valid_types"], gate["valid_templates"],
            paraphrases)
        return judging["bundles"], judging["texts"]

    def _finish_quality_round(self, round_idx: int, bundles: list[dict],
                              generated: int) -> bool:
        """Resume-path quality: rebuild deterministic context, then share
        the single _do_quality_round implementation (persist+checkpoint)."""
        self._current_action = f"quality_round_{round_idx}"
        self._current_round = round_idx
        rdir = round_dir(self.rounds_dir, round_idx)
        styles = read_json(rdir / "01_question_styles.json")
        self._round_types[round_idx] = {
            t["id"]: t for t in styles.get("question_types", [])}
        gate = self._reload_gate(round_idx)
        paras = read_json(rdir / "03_paraphrases.json").get("paraphrases", [])
        self._prepare_judging(
            round_idx, gate["valid_types"], gate["valid_templates"], paras)
        stopped = self._do_quality_round(round_idx, bundles, generated)
        # If this was the last round without a stop, the controller must
        # have stopped at max_rounds; guard against logic drift.
        if round_idx >= self.controller.max_rounds and not stopped:
            raise AssertionError(
                "Loop controller did not stop at max_rounds.")
        return stopped

    def _finalize_and_close(self) -> list[dict]:
        preview = self._finalize()
        self._completed.append("finalize")
        self._save_state(completed=list(self._completed),
                         current_round=len(self.history),
                         next_action="done", stopped=True,
                         stop_reason=self.stop_reason)
        return preview


def _assert_round_accounting(round_idx: int, generated: int,
                             accepted_n: int, dup_n: int,
                             judge_n: int, auto_n: int,
                             where: str, det_n: int = 0) -> None:
    """Round invariant: every Style proposal is accounted for exactly once.

    generated == auto_rejected + deterministic_signature_duplicates
        + accepted_new + quality_duplicates + judge_rejected.

    Template omissions never count: kept-subset selection changes neither
    numerator nor denominator of the TYPE-level arithmetic.
    """
    if not (generated == accepted_n + dup_n + judge_n + auto_n + det_n):
        raise ValueError(
            f"[R{round_idx}] {where} round accounting violated: "
            f"generated {generated} != accepted {accepted_n} + "
            f"duplicates {dup_n} + judge-rejected {judge_n} + "
            f"deterministic {auto_n} + signature {det_n}.")


def _stage_of_step(step: str) -> str | None:
    """Map a v3 completed-step name to its budget stage."""
    m = re.fullmatch(r"(style|template|paraphrase|quality)_round_\d+", step)
    if m is None:
        return None
    return {"style": "question_style", "template": "question_template",
            "paraphrase": "paraphrase", "quality": "quality"}[m.group(1)]


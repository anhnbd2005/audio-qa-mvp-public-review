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


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_RE.match(run_id):
        raise ValueError(
            f"Invalid run_id {run_id!r}: use 1-64 chars of [A-Za-z0-9_-]."
        )
    return run_id


def generate_run_id(out_parent: Path) -> str:
    base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = base
    suffix = 2
    while (out_parent / run_id).exists():
        run_id = f"{base}-{suffix}"
        suffix += 1
    return run_id


def resolve_out_dir(
    dataset: str,
    real_llm: bool,
    run_id: str | None,
    resume_run: str | None,
) -> tuple[Path, str | None]:
    """Return (out_dir, effective_run_id). Mock has no run_id."""
    if not real_llm:
        if resume_run is not None or run_id is not None:
            raise ValueError("run_id/resume_run require --real-llm.")
        return ROOT / "outputs" / "runs" / dataset / "mock", None

    parent = ROOT / "outputs" / "runs" / dataset
    if resume_run is not None:
        if run_id is not None:
            raise ValueError("--run-id and --resume-run are mutually exclusive.")
        effective = validate_run_id(resume_run)
        out_dir = parent / effective
        if not (out_dir / STATE_FILE).is_file():
            raise FileNotFoundError(
                f"Cannot resume: no checkpoint at {out_dir / STATE_FILE}"
            )
        return out_dir, effective

    effective = validate_run_id(run_id) if run_id else generate_run_id(parent)
    out_dir = parent / effective
    if out_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite previous real run: {out_dir} "
            "(omit --run-id for a fresh unique run_id, "
            "or use --resume-run to continue it)."
        )
    return out_dir, effective


def loop_controller_from_config() -> LoopController:
    loop_cfg = CONFIG["loop"]
    return LoopController(
        max_rounds=loop_cfg["max_rounds"],
        min_rate=loop_cfg["min_new_type_rate"],
        saturation_patience=loop_cfg["saturation_patience"],
        immediate_stop_if_zero_new=loop_cfg["immediate_stop_if_zero_new"],
        zero_reason="zero_new_types",
        saturated_reason="type_diversity_saturated",
    )



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="vimd")
    parser.add_argument("--real-llm", action="store_true")
    parser.add_argument("--run-id", default=None,
                        help="Explicit run_id for a fresh real run.")
    parser.add_argument("--resume-run", default=None,
                        help="Run_id of a crashed real run to continue.")
    parser.add_argument("--dataset-profile", default=None,
                        help="Path to the validated REAL dataset_profile.json "
                             "(semantic source for discovery).")
    parser.add_argument("--sample-manifest", default=None,
                        help="Path to the new TRAIN/VALID discovery sample "
                             "jsonl (row source for discovery).")
    parser.add_argument("--validation-manifest", default=None,
                        help="Manifest jsonl used as evidence when auditing "
                             "candidate types (default: sibling "
                             "manifest_train.jsonl).")
    parser.add_argument("--card", default=None,
                        help="Dataset card markdown used as the readme "
                             "(default: data_sources/<dataset>/dataset_card.md).")
    parser.add_argument("--stop-after", default=None,
                        choices=["question_style"],
                        help="Stop the pipeline after exactly one Question "
                             "Style call (no template/paraphrase/quality).")
    args = parser.parse_args()

    if args.stop_after == "question_style":
        if args.dataset_profile is None or args.sample_manifest is None:
            parser.error("--stop-after question_style requires "
                         "--dataset-profile and --sample-manifest")
        if not args.real_llm:
            parser.error("--stop-after question_style requires --real-llm "
                         "(one real Question Style call; a mock run is not "
                         "this experiment)")
        if args.resume_run:
            parser.error("--resume-run does not apply to --stop-after "
                         "question_style")
        from src.autonomous_qa.language.question_discovery import run_question_discovery
        result = run_question_discovery(
            dataset=args.dataset,
            dataset_profile=Path(args.dataset_profile),
            sample_manifest=Path(args.sample_manifest),
            run_id=args.run_id,
            card_path=Path(args.card) if args.card else None,
            validation_manifest=(Path(args.validation_manifest)
                                 if args.validation_manifest else None),
        )
        print(json.dumps(result["run_meta"], indent=2, ensure_ascii=False))
        return

    if args.dataset_profile or args.sample_manifest or args.validation_manifest:
        parser.error("--dataset-profile/--sample-manifest/"
                     "--validation-manifest only apply with "
                     "--stop-after question_style")
    from src.autonomous_qa.compiler.pipeline_runner import run_pipeline
    run_pipeline(dataset=args.dataset, real_llm=args.real_llm,
                 run_id=args.run_id, resume_run=args.resume_run)


if __name__ == "__main__":
    main()

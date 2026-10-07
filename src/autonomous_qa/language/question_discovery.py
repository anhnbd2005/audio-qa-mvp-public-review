"""Question-Style-only discovery driven by a REAL dataset profile.

Stage-stop entry point for the profile-driven discovery experiment:

    python -m src.run_pipeline --dataset vimd \\
        --dataset-profile outputs/runs/vimd/<run>/profile/dataset_profile.json \\
        --sample-manifest data/materialized/vimd/<run>/sample_for_discovery.jsonl \\
        --real-llm --stop-after question_style

Exactly ONE Question Style call, no template/paraphrase/quality stage, no
fallback endpoint. Output goes to the isolated root
outputs/runs/<dataset>/<run_id>/language/question_discovery/.

Inputs are explicit paths: the validated REAL profile is the semantic
source, the new TRAIN-only discovery sample is the row source. Legacy
data/vimd artifacts are never read here.

If the single call returns a payload that is not the exact response
schema (fenced JSON, `candidates` instead of `question_types`, ...), the
persisted raw response is coerced deterministically instead of spending a
second call; the coercion is recorded in the run metadata and report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.common.config import ROOT
from src.autonomous_qa.language.question_style import build_question_style_prompt
from src.autonomous_qa.core.type_policy import (
    STATUS_REJECTED,
    STATUS_REVIEW,
    STATUS_SUPPORTED,
    PolicyContract,
    audio_necessity_verdict,
    audit_candidate_type,
)
from src.common.io import write_json

DEFAULT_CARD = ROOT / "data_sources" / "vimd" / "dataset_card.md"

PARSE_PARSED = "parsed"
PARSE_SALVAGED = "salvaged_nonconformant_envelope"

AUDIO_EXT = (".wav", ".flac", ".mp3", ".m4a", ".ogg")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prompt_rows(sample_rows: list[dict]) -> list[dict]:
    """Discovery rows for the prompt: raw metadata only, no bookkeeping.

    `sample_id` and the local `audio_path` are pipeline bookkeeping and are
    never shown as semantic content.
    """
    return [{
        "split": row.get("split"),
        "metadata": row.get("metadata", {}),
    } for row in sample_rows]


def new_run_dir(dataset: str, run_id: str | None) -> Path:
    parent = ROOT / "outputs" / "runs" / dataset
    if run_id is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id, suffix = stamp, 2
        while (parent / run_id / "language" / "question_discovery").exists():
            run_id = f"{stamp}-{suffix}"
            suffix += 1
    out = parent / run_id / "language" / "question_discovery"
    if out.exists():
        raise FileExistsError(f"refusing_to_overwrite:{out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


# -- deterministic coercion of a non-conformant single response -----------
def strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` fences when they wrap the whole payload."""
    body = text.strip()
    if not body.startswith("```"):
        return body
    lines = body.splitlines()
    if not lines[0].startswith("```"):
        return body
    end = next((i for i in range(len(lines) - 1, 0, -1)
                if lines[i].strip().startswith("```")), None)
    if end is None:
        return body
    return "\n".join(lines[1:end])


def extract_json_object(text: str) -> dict:
    """First balanced top-level JSON object in `text` (fences allowed)."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no_json_object_in_response")
    depth = 0
    in_str = False
    esc = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:idx + 1])
    raise ValueError("unbalanced_json_object_in_response")


def derive_answer_rule(answer: dict) -> str | None:
    """Human-readable equivalent of a machine `answer` spec."""
    kind = str(answer.get("kind") or "")
    if kind == "field_value" and answer.get("key"):
        return f"The answer is the {answer['key']} value."
    if kind == "equality":
        keys = answer.get("keys") or []
        if len(keys) >= 2:
            return (f"yes if {keys[0]} equals {keys[1]}, otherwise no")
        return None
    if kind == "derived_field" and answer.get("source_key"):
        target = answer.get("target_key") or "the target field"
        return f"The answer is {target} derived from {answer['source_key']}."
    return None


def normalize_style_item(item: dict) -> tuple[dict, list[dict]]:
    """Map one candidate onto RoundQuestionType field names."""
    notes: list[dict] = []
    out = dict(item)

    aliases = [
        ("id", ("id", "type_id")),
        ("name", ("name", "type")),
        ("goal", ("goal", "insight", "description")),
    ]
    for field, sources in aliases:
        if out.get(field) is None:
            for src in sources:
                if out.get(src) is not None:
                    out[field] = out[src]
                    notes.append({"rule": f"alias:{field}", "from": src})
                    break

    if not out.get("answer_rule"):
        derived = derive_answer_rule(out.get("answer") or {})
        if derived:
            out["answer_rule"] = derived
            notes.append({"rule": "derive:answer_rule", "from": "answer"})
    return out, notes


def coerce_style_output(payload: dict, round_idx: int) -> tuple[dict, list[dict]]:
    """Normalize the top-level envelope into the StyleRoundOutput shape.

    Per-item field mapping happens in `normalize_style_item`.
    """
    notes: list[dict] = []
    out = dict(payload)
    if "question_types" not in out:
        for src in ("candidates", "types", "items"):
            if isinstance(out.get(src), list):
                out["question_types"] = out[src]
                notes.append({"rule": "envelope:question_types", "from": src})
                break
    if out.get("round") is None:
        out["round"] = round_idx
        notes.append({"rule": "envelope:round", "from": "round_idx"})
    if not isinstance(out.get("question_types"), list):
        raise TypeError("question_types_not_a_list")
    return out, notes


def raw_response_files(out_dir: Path) -> list[Path]:
    files = sorted(out_dir.glob("question_style_raw_response*.txt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return [p for p in files if p.read_text(encoding="utf-8").strip()]


def load_style_output(out_dir: Path, round_idx: int) -> tuple[dict, str,
                                                              list[dict],
                                                              str]:
    """Parse the persisted single response without making another call.

    Returns (output, parse_status, adaptation_notes, raw_text).
    """
    files = raw_response_files(out_dir)
    if not files:
        raise FileNotFoundError(
            f"no_persisted_raw_response_in:{out_dir}")
    raw_text = files[0].read_text(encoding="utf-8")
    notes: list[dict] = []

    candidate = strip_code_fences(raw_text)
    if candidate.strip() != raw_text.strip():
        notes.append({"rule": "strip_code_fences"})
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        payload = extract_json_object(raw_text)
        notes.append({"rule": "extract_balanced_json_object"})
    if not isinstance(payload, dict):
        raise TypeError("response_payload_not_an_object")

    payload, envelope_notes = coerce_style_output(payload, round_idx)
    notes.extend(envelope_notes)

    item_notes: list[dict] = []
    normalized_items = []
    for item in payload["question_types"]:
        normalized, item_rule_notes = normalize_style_item(item)
        normalized_items.append(normalized)
        type_id = normalized.get("id") or item.get("type_id")
        item_notes.extend({**n, "type_id": type_id}
                          for n in item_rule_notes)
    payload["question_types"] = normalized_items
    notes.extend(item_notes)

    from src.autonomous_qa.core.schemas import StyleRoundOutput
    validated = StyleRoundOutput(**payload)
    status = PARSE_PARSED if not notes else PARSE_SALVAGED
    return validated.model_dump(), status, notes, raw_text


# -- audit tables ---------------------------------------------------------
def aggregate_coercion_notes(notes: list[dict]) -> list[dict]:
    """Compact per-item coercion notes: one entry per (rule, source)."""
    agg: dict[tuple, dict] = {}
    for note in notes:
        key = (note.get("rule"), note.get("from"))
        entry = agg.setdefault(key, {
            "rule": note.get("rule"),
            "source": note.get("from"),
            "count": 0,
            "type_ids": [],
        })
        entry["count"] += 1
        type_id = note.get("type_id")
        if type_id and type_id not in entry["type_ids"]:
            entry["type_ids"].append(type_id)
    out = []
    for entry in agg.values():
        if not entry["type_ids"]:
            entry.pop("type_ids")
        out.append(entry)
    return out


def audit_table(records: list[dict]) -> str:
    lines: list[str] = []
    for rec in records:
        lines.append(
            f"TYPE_ID        {rec.get('type_id')}\n"
            f"NAME           {rec.get('name')}\n"
            f"GOAL           {rec.get('goal')}\n"
            f"INPUT_COUNT    {rec.get('input_count')}\n"
            f"AUDIO_REQUIRED {rec.get('audio_required')}\n"
            f"VISIBLE_CONTEXT_FIELDS {rec.get('visible_context_fields')}\n"
            f"GOLD_SOURCE_FIELDS     {rec.get('gold_source_fields')}\n"
            f"ANSWER_KIND    {rec.get('answer_kind')}\n"
            f"USES_FIELDS    {rec.get('uses_fields')}\n"
            f"STATUS         {rec.get('status')}\n"
            f"REASONS        {rec.get('reasons') or '-'}\n"
            f"REVIEW_FLAGS   {rec.get('review_flags') or '-'}\n"
            f"PRIMARY_GROUP  {rec.get('primary_reason_group') or '-'}\n"
            + "-" * 72
        )
    return "\n".join(lines)


def rejection_group_counts(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in records:
        if rec.get("status") != STATUS_REJECTED:
            continue
        group = rec.get("primary_reason_group") or "other"
        counts[group] = counts.get(group, 0) + 1
    return dict(sorted(counts.items()))


def family_expectations(records: list[dict]) -> dict[str, list[str]]:
    """Which recognizable families the discovery surfaced (not required)."""
    families = {
        "speech_transcription": ("transcri", "asr", "speech-to-text"),
        "broad_region_dialect": ("region", "macro dialect", "broad dialect"),
        "province_dialect": ("province", "provincial"),
        "same_different_speaker": ("same speaker", "speaker verification",
                                   "different speaker", "speaker identity"),
    }
    out: dict[str, list[str]] = {k: [] for k in families}
    for rec in records:
        blob = f"{rec.get('name') or ''} {rec.get('goal') or ''}".lower()
        for family, keys in families.items():
            if any(k in blob for k in keys):
                out[family].append(rec.get("type_id"))
    return out


def build_audit_text(records: list[dict]) -> str:
    accepted = [r for r in records if r["status"] != STATUS_REJECTED]
    lines = [
        "QUESTION STYLE CANDIDATE AUDIT",
        "=" * 72,
        f"total_types: {len(records)}",
        f"supported: {sum(1 for r in records if r['status'] == STATUS_SUPPORTED)}",
        f"review_required: {sum(1 for r in records if r['status'] == STATUS_REVIEW)}",
        f"rejected: {sum(1 for r in records if r['status'] == STATUS_REJECTED)}",
        "",
        "-- CANDIDATE TYPES " + "-" * 55,
        audit_table(records),
        "",
        "-- REJECTION GROUPS (one primary bucket per rejected type) " + "-" * 14,
        json.dumps(rejection_group_counts(records), indent=2),
        "",
        "-- FAMILY COVERAGE (expectation, never forced) " + "-" * 21,
        json.dumps(family_expectations(records), indent=2),
        "",
        "-- AUDIO NECESSITY AUDIT (accepted types) " + "-" * 31,
        json.dumps([audio_necessity_verdict(r) for r in accepted], indent=2),
        "",
        "-- AUDIO NECESSITY AUDIT (all types) " + "-" * 36,
        json.dumps([audio_necessity_verdict(r) for r in records], indent=2),
        "",
    ]
    return "\n".join(lines)


def write_discovery_outputs(
    out_dir: Path,
    *,
    dataset: str,
    out: dict,
    records: list[dict],
    dataset_profile: Path,
    sample_manifest: Path,
    card_path: Path,
    validation_manifest: Path,
    validation_rows: list[dict],
    sample_rows: list[dict],
    parse_status: str,
    adaptation_notes: list[dict],
    llm_calls: int,
    elapsed: float,
) -> dict:
    write_json(out_dir / "candidate_types.json", {
        "round": out.get("round"),
        "count": len(out.get("question_types", [])),
        "question_types": out.get("question_types", []),
    })

    contract = PolicyContract.from_profile(
        json.loads(dataset_profile.read_text(encoding="utf-8")))
    summary = {
        "total": len(records),
        "supported": sum(1 for r in records if r["status"] == STATUS_SUPPORTED),
        "review_required": sum(1 for r in records
                               if r["status"] == STATUS_REVIEW),
        "rejected": sum(1 for r in records if r["status"] == STATUS_REJECTED),
        "rejection_groups": rejection_group_counts(records),
        "family_expectations": family_expectations(records),
    }
    write_json(out_dir / "candidate_type_validation.json", {
        "contract_source": {
            "dataset_profile": str(dataset_profile),
            "dataset_profile_sha256": sha256_file(dataset_profile),
            "dataset": contract.dataset,
            "field_roles": contract.field_roles(),
            "field_evaluations": contract.evaluation_policies(),
            "context_visibilities": contract.context_visibilities(),
            "hidden_fields": sorted(contract.hidden_fields),
            "semantic_fields": sorted(contract.semantic_fields),
            "unresolved_encoded_fields":
                sorted(contract.unresolved_encoded_fields),
            "relations": contract.relations,
        },
        "validation_manifest": str(validation_manifest),
        "validation_rows": len(validation_rows),
        "summary": summary,
        "records": records,
        "audio_necessity": [audio_necessity_verdict(r) for r in records],
    })

    (out_dir / "audit.txt").write_text(build_audit_text(records),
                                       encoding="utf-8")

    from src.common.config import resolve_llm_base_url, resolve_llm_model

    run_meta = {
        "mode": "real",
        "stage": "question_style",
        "stop_after": "question_style",
        "dataset": dataset,
        "run_dir": str(out_dir),
        "dataset_profile": str(dataset_profile),
        "dataset_profile_sha256": sha256_file(dataset_profile),
        "sample_manifest": str(sample_manifest),
        "sample_manifest_sha256": sha256_file(sample_manifest),
        "sample_rows": len(sample_rows),
        "card": str(card_path),
        "endpoint": resolve_llm_base_url(),
        "model": resolve_llm_model(),
        "remote_openai_api": "NO",
        "llm_calls": llm_calls,
        "retries": 0,
        "fallback": "none",
        "parse_status": parse_status,
        "response_coercion_notes": aggregate_coercion_notes(adaptation_notes),
        "template_calls": 0,
        "paraphrase_calls": 0,
        "quality_calls": 0,
        "test_split_accessed": False,
        "sauvi_accessed": False,
        "elapsed_seconds": round(elapsed, 2),
        "summary": summary,
    }
    write_json(out_dir / "run_meta.json", run_meta)
    return run_meta


def resolve_inputs(dataset_profile: Path, sample_manifest: Path,
                   metadata_context_path: Path | None,
                   validation_manifest: Path | None) -> dict:
    dataset_profile = Path(dataset_profile)
    sample_manifest = Path(sample_manifest)
    if not dataset_profile.is_file():
        raise FileNotFoundError(f"dataset_profile_not_found:{dataset_profile}")
    if not sample_manifest.is_file():
        raise FileNotFoundError(f"sample_manifest_not_found:{sample_manifest}")

    meta_ctx = None
    if metadata_context_path is None:
        sibling = dataset_profile.parent / "metadata_context.json"
        metadata_context_path = sibling if sibling.is_file() else None
    if metadata_context_path:
        meta_ctx = json.loads(
            Path(metadata_context_path).read_text(encoding="utf-8"))

    if validation_manifest is None:
        sibling = sample_manifest.parent / "manifest_train.jsonl"
        validation_manifest = sibling if sibling.is_file() else sample_manifest

    return {
        "dataset_profile": dataset_profile,
        "sample_manifest": sample_manifest,
        "metadata_context": meta_ctx,
        "validation_manifest": Path(validation_manifest),
        "validation_rows": load_jsonl(Path(validation_manifest)),
        "sample_rows": load_jsonl(sample_manifest),
    }


def run_question_discovery(
    *,
    dataset: str,
    dataset_profile: Path,
    sample_manifest: Path,
    run_id: str | None = None,
    card_path: Path | None = None,
    metadata_context_path: Path | None = None,
    validation_manifest: Path | None = None,
    out_dir: Path | None = None,
) -> dict:
    """Exactly one Question Style call; never a retry, never a fallback."""
    from src.common.config import resolve_llm_base_url
    from src.autonomous_qa.core.dataset_profile import require_local_proxy
    from src.autonomous_qa.authoring.llm_client import (
        ATTEMPTS,
        StageOutputValidationError,
        complete_structured_stage,
        create_llm_client,
    )
    from src.autonomous_qa.core.schemas import StyleRoundOutput

    t0 = time.time()
    inputs = resolve_inputs(dataset_profile, sample_manifest,
                            metadata_context_path, validation_manifest)
    dataset_profile = inputs["dataset_profile"]
    sample_manifest = inputs["sample_manifest"]
    profile = json.loads(dataset_profile.read_text(encoding="utf-8"))
    contract = PolicyContract.from_profile(profile)

    card_path = Path(card_path) if card_path else DEFAULT_CARD
    readme = card_path.read_text(encoding="utf-8")

    for row in inputs["sample_rows"]:
        if str(row.get("split", "")).lower() not in ("train", "valid"):
            raise ValueError(
                f"discovery_sample_split_not_allowed:{row.get('split')}")

    out_dir = out_dir or new_run_dir(dataset, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt = build_question_style_prompt(
        readme,
        [],
        1,
        metadata_context=inputs["metadata_context"],
        knowledge_context=None,
        sample_rows=prompt_rows(inputs["sample_rows"]),
        max_sample_rows=None,
    )
    (out_dir / "question_style_prompt.txt").write_text(prompt,
                                                       encoding="utf-8")

    require_local_proxy(resolve_llm_base_url())
    client = create_llm_client()

    raw_path = out_dir / "question_style_raw_response.txt"
    adaptation_notes: list[dict] = []
    try:
        out = complete_structured_stage(
            client,
            stage="question_style",
            prompt=prompt,
            response_schema=StyleRoundOutput,
            raw_path=raw_path,
            on_attempt=None,
        )
        parse_status = PARSE_PARSED
    except StageOutputValidationError:
        # The single call already happened and was persisted above; coerce
        # it deterministically instead of spending a second call.
        out, parse_status, adaptation_notes, _ = load_style_output(
            out_dir, round_idx=1)

    llm_calls = int(ATTEMPTS.stage_attempts.get("question_style", 0))
    records = [
        audit_candidate_type(t, contract,
                             dataset_rows=inputs["validation_rows"])
        for t in out.get("question_types", [])
    ]

    run_meta = write_discovery_outputs(
        out_dir,
        dataset=dataset,
        out=out,
        records=records,
        dataset_profile=dataset_profile,
        sample_manifest=sample_manifest,
        card_path=card_path,
        validation_manifest=inputs["validation_manifest"],
        validation_rows=inputs["validation_rows"],
        sample_rows=inputs["sample_rows"],
        parse_status=parse_status,
        adaptation_notes=adaptation_notes,
        llm_calls=llm_calls,
        elapsed=time.time() - t0,
    )
    return {"out_dir": out_dir, "run_meta": run_meta, "records": records,
            "summary": run_meta["summary"]}


def salvage_run(
    *,
    dataset: str,
    out_dir: Path,
    dataset_profile: Path,
    sample_manifest: Path,
    card_path: Path | None = None,
    metadata_context_path: Path | None = None,
    validation_manifest: Path | None = None,
) -> dict:
    """Re-derive audit outputs from an already persisted single response.

    Makes NO LLM call. Used when a run stopped after the call but before
    (or while) parsing its response.
    """
    t0 = time.time()
    out_dir = Path(out_dir)
    inputs = resolve_inputs(dataset_profile, sample_manifest,
                            metadata_context_path, validation_manifest)
    profile = json.loads(Path(inputs["dataset_profile"]).read_text(
        encoding="utf-8"))
    contract = PolicyContract.from_profile(profile)

    out, parse_status, adaptation_notes, _ = load_style_output(out_dir,
                                                               round_idx=1)
    records = [
        audit_candidate_type(t, contract,
                             dataset_rows=inputs["validation_rows"])
        for t in out.get("question_types", [])
    ]
    llm_calls = len(raw_response_files(out_dir))

    card_path = Path(card_path) if card_path else DEFAULT_CARD
    run_meta = write_discovery_outputs(
        out_dir,
        dataset=dataset,
        out=out,
        records=records,
        dataset_profile=inputs["dataset_profile"],
        sample_manifest=inputs["sample_manifest"],
        card_path=card_path,
        validation_manifest=inputs["validation_manifest"],
        validation_rows=inputs["validation_rows"],
        sample_rows=inputs["sample_rows"],
        parse_status=parse_status,
        adaptation_notes=adaptation_notes,
        llm_calls=llm_calls,
        elapsed=time.time() - t0,
    )
    return {"out_dir": out_dir, "run_meta": run_meta, "records": records,
            "summary": run_meta["summary"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salvage", default=None, metavar="RUN_DIR",
                        help="Re-audit an existing run dir from its "
                             "persisted raw response (no LLM call).")
    parser.add_argument("--dataset", default="vimd")
    parser.add_argument("--dataset-profile", required=True)
    parser.add_argument("--sample-manifest", required=True)
    parser.add_argument("--card", default=None)
    parser.add_argument("--validation-manifest", default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    if args.salvage:
        result = salvage_run(
            dataset=args.dataset,
            out_dir=Path(args.salvage),
            dataset_profile=Path(args.dataset_profile),
            sample_manifest=Path(args.sample_manifest),
            card_path=Path(args.card) if args.card else None,
            validation_manifest=(Path(args.validation_manifest)
                                 if args.validation_manifest else None),
        )
    else:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PARSE_PARSED",
    "PARSE_SALVAGED",
    "aggregate_coercion_notes",
    "audit_table",
    "build_audit_text",
    "coerce_style_output",
    "derive_answer_rule",
    "extract_json_object",
    "family_expectations",
    "load_jsonl",
    "load_style_output",
    "main",
    "new_run_dir",
    "normalize_style_item",
    "prompt_rows",
    "rejection_group_counts",
    "run_question_discovery",
    "salvage_run",
    "sha256_file",
    "strip_code_fences",
    "write_discovery_outputs",
]

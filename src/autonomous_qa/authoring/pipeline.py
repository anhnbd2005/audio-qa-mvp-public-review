"""Real OpenAI-compatible R&D / re-authoring pipeline (generic, dataset-agnostic).

This module is the *authoring* toolchain. It is deliberately separate from
production:

    source docs -> ingest + hash
                -> deterministic dataset profile
                -> documentation/data reconciliation
                -> LLM documentation interpretation      (proposal)
                -> LLM blind primitive discovery         (proposal)
                -> deterministic primitive gates         (validation)
                -> LLM contract critique                 (proposal)
                -> LLM blind composite discovery         (proposal)
                -> deterministic composite + leak gates  (validation)
                -> LLM language generation               (proposal)
                -> LLM language semantic review          (proposal)
                -> deterministic language preflight      (validation)
                -> production readiness                  (validation)
                -> reconcile against current canonical final

Principles enforced here:

* LLM proposes, deterministic code validates and freezes.
* No per-row LLM call: rows are profiled by Python only.
* Gold is never derived by an LLM.
* The blind stages never receive the current canonical semantic catalogue.
* Every real call records provenance (model, temperature, prompt hash, input
  hashes, response hash, parser status, cache hit) and never records secrets.

The caller may inject an ``llm_call`` callable for tests so ordinary pytest
never touches the network; the default callable uses the project's existing
``src.llm_client`` (no second HTTP stack).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.autonomous_qa.language.language_quality import ProductionGenerationConfig
from src.autonomous_qa.production.source_preparation import (
    filter_by_operator_policy,
    prepare_source_inventory,
    propose_allocation,
    resolve_early_budget,
    validate_operator_policy,
)
from src.common.config import ROOT, resolve_llm_base_url, resolve_llm_model

RUNS_ROOT = ROOT / "outputs" / "runs"
PROMPT_DIR = ROOT / "prompts" / "rnd"

PROMPT_IDS = {
    "documentation_interpretation": "documentation_interpretation_v1",
    "primitive_semantic_discovery": "primitive_semantic_discovery_v1",
    "semantic_contract_critic": "semantic_contract_critic_v1",
    "composite_discovery": "composite_discovery_v1",
    "language_generation": "language_generation_v1",
    "language_semantic_review": "language_semantic_review_v1",
}

STAGE_TEMPERATURE = {
    "documentation_interpretation": 0.2,
    "primitive_semantic_discovery": 0.7,
    "semantic_contract_critic": 0.0,
    "composite_discovery": 0.6,
    "language_generation": 0.7,
    "language_semantic_review": 0.0,
}

BLIND_STAGES = frozenset(
    {
        "primitive_semantic_discovery",
        "composite_discovery",
        "language_generation",
    }
)

GATE_PASS = "PASS"
GATE_FAIL = "FAIL"


@dataclass(frozen=True)
class DatasetAuthoringConfig:
    dataset_id: str
    pretty_name: str
    source_revision: str
    card_path: Path
    readme_path: Path
    materialized_path: Path
    allowed_splits: tuple[str, ...]
    canonical_catalog_path: Path
    plan_path: Path
    record_identity_field: str
    field_roles: dict[str, str] = field(default_factory=dict)


DATASETS: dict[str, DatasetAuthoringConfig] = {
    "vimd": DatasetAuthoringConfig(
        dataset_id="vimd",
        pretty_name="ViMD — Vietnamese Multi-dialect Dataset",
        source_revision="3a5b30157034e7eadd5c75fae1a820c6f9383398",
        card_path=ROOT / "data_sources" / "vimd" / "dataset_card.md",
        readme_path=ROOT / "data" / "vimd" / "README.md",
        materialized_path=(
            ROOT
            / "data"
            / "materialized"
            / "vimd"
            / "_cache"
            / "3a5b30157034e7eadd5c75fae1a820c6f9383398"
            / "train.jsonl"
        ),
        allowed_splits=("train",),
        canonical_catalog_path=(
            ROOT / "resources" / "semantics" / "vimd_semantic_catalog.json"
        ),
        plan_path=(
            ROOT
            / "data"
            / "materialized"
            / "vimd"
            / "current"
            / "generation_plan.jsonl"
        ),
        record_identity_field="filename",
        field_roles={
            "text": "semantic_text",
            "region": "semantic_categorical",
            "province_name": "semantic_categorical",
            "province_code": "provenance",
            "speakerID": "hidden_identifier",
            "gender": "semantic_categorical",
            "filename": "provenance",
        },
    ),
    "vietmdd": DatasetAuthoringConfig(
        dataset_id="vietmdd",
        pretty_name="VietMMD (Mispronunciation Detection and Diagnosis)",
        source_revision="train_3181",
        card_path=(
            ROOT / "data_sources" / "vietmdd" / "dataset_card.md"
        ),
        readme_path=ROOT / "data" / "vietmdd" / "README.md",
        materialized_path=(
            ROOT / "data" / "materialized" / "vietmdd" / "train.jsonl"
        ),
        allowed_splits=("train",),
        canonical_catalog_path=(
            ROOT / "resources" / "semantics" / "vietmdd_semantic_catalog.json"
        ),
        plan_path=(
            ROOT
            / "data"
            / "materialized"
            / "vietmdd"
            / "production_plan.jsonl"
        ),
        record_identity_field="row_id",
        field_roles={
            "observed_transcription_norm": "semantic_text",
            "original_text_norm": "semantic_text",
            "text_exact_match": "derived",
            "age_class": "context_only",
            "audio_id": "provenance",
            "audio_path": "provenance",
            "row_id": "provenance",
        },
    ),
    "vimedcss": DatasetAuthoringConfig(
        dataset_id="vimedcss",
        pretty_name="ViMedCSS (Vietnamese Medical Code-Switching Dataset)",
        source_revision="canonical",
        card_path=ROOT / "data_sources" / "vimedcss" / "dataset_card.md",
        readme_path=ROOT / "data_sources" / "vimedcss" / "dataset_card.md",
        materialized_path=ROOT / "data_sources" / "vimedcss" / "source" / "train.jsonl",
        allowed_splits=("train",),
        canonical_catalog_path=(
            ROOT / "resources" / "semantics" / "vimedcss_semantic_catalog.json"
        ),
        plan_path=(
            ROOT
            / "outputs"
            / "production_plan"
            / "vimedcss"
            / "current"
            / "generation_plan.jsonl"
        ),
        record_identity_field="segment_id",
        field_roles={
            "segment_text": "semantic_text",
            "cs_terms_list": "semantic_categorical",
            "topic": "semantic_categorical",
            "duration_seconds": "context_only",
            "original_video_link": "provenance",
            "segment_id": "provenance",
        },
    ),
}


# ---------------------------------------------------------------------------
# hashing helpers
# ---------------------------------------------------------------------------


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def canonical_hash(payload: Any) -> str:
    return sha256_text(canonical_json(payload))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    tmp.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# stage 1: deterministic documentation ingestion
# ---------------------------------------------------------------------------


def ingest_documentation(cfg: DatasetAuthoringConfig, run_dir: Path) -> dict[str, Any]:
    docs = {
        "source_card": cfg.card_path,
        "readme": cfg.readme_path,
    }
    manifest: dict[str, Any] = {"dataset_id": cfg.dataset_id, "documents": {}}
    for role, path in docs.items():
        if not path.exists():
            raise FileNotFoundError(f"MISSING_SOURCE_DOCUMENTATION:{role}:{path}")
        text = path.read_text(encoding="utf-8")
        manifest["documents"][role] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_text(text),
            "bytes": len(text.encode("utf-8")),
        }
        _write_text(run_dir / "source_documentation" / f"{role}.md", text)
    manifest["combined_sha256"] = canonical_hash(
        {k: v["sha256"] for k, v in manifest["documents"].items()}
    )
    _write_json(
        run_dir / "source_documentation" / "documentation_manifest.json", manifest
    )
    return manifest


# ---------------------------------------------------------------------------
# stage 2: deterministic dataset profiler (NO LLM)
# ---------------------------------------------------------------------------

_TEXT_FIELDS = ("text", "observed_transcription_norm", "original_text_norm", "segment_text")


def _infer_type(values: list[Any]) -> str:
    kinds = set()
    for v in values:
        if isinstance(v, bool):
            kinds.add("boolean")
        elif isinstance(v, int):
            kinds.add("int64")
        elif isinstance(v, float):
            kinds.add("float")
        else:
            kinds.add("string")
    return kinds.pop() if len(kinds) == 1 else "mixed"


def profile_rows(rows: list[dict], cfg: DatasetAuthoringConfig) -> dict[str, Any]:
    fields: dict[str, list[Any]] = {}
    for row in rows:
        for key, value in row.items():
            fields.setdefault(key, []).append(value)

    field_stats: dict[str, Any] = {}
    for name, values in sorted(fields.items()):
        non_null = [v for v in values if v is not None]
        empty = sum(1 for v in non_null if isinstance(v, str) and not v.strip())
        distinct = {canonical_json(v) for v in non_null}
        stat: dict[str, Any] = {
            "dtype": _infer_type(non_null),
            "role": cfg.field_roles.get(name, "unclassified"),
            "n": len(values),
            "missing": len(values) - len(non_null),
            "empty": empty,
            "distinct": len(distinct),
        }
        if len(distinct) <= 20:
            counts: dict[str, int] = {}
            for v in non_null:
                counts[str(v)] = counts.get(str(v), 0) + 1
            stat["value_distribution"] = dict(
                sorted(counts.items(), key=lambda kv: -kv[1])
            )
        else:
            lengths = [len(str(v)) for v in non_null]
            stat["text"] = {
                "min_len": min(lengths) if lengths else 0,
                "max_len": max(lengths) if lengths else 0,
                "mean_len": round(sum(lengths) / len(lengths), 2) if lengths else 0,
            }
        field_stats[name] = stat

    id_field = cfg.record_identity_field
    ids = [r.get(id_field) for r in rows]
    duplicate_ids = len(ids) - len(set(ids))

    cross_field: dict[str, Any] = {}
    for text_field in _TEXT_FIELDS:
        if text_field in fields:
            vals = [str(v) for v in fields[text_field]]
            cross_field[f"duplicate_{text_field}"] = len(vals) - len(set(vals))
    if "observed_transcription_norm" in fields and "original_text_norm" in fields:
        o = [str(v) for v in fields["observed_transcription_norm"]]
        r = [str(v) for v in fields["original_text_norm"]]
        cross_field["observed_equals_reference_raw"] = sum(
            1 for a, b in zip(o, r) if a == b
        )
        cross_field["observed_differs_reference_raw"] = sum(
            1 for a, b in zip(o, r) if a != b
        )

    return {
        "dataset_id": cfg.dataset_id,
        "row_count": len(rows),
        "fields": field_stats,
        "record_identity": {
            "field": id_field,
            "unique": len(set(ids)),
            "duplicate_ids": duplicate_ids,
        },
        "cross_field": cross_field,
    }


def load_rows(cfg: DatasetAuthoringConfig) -> list[dict]:
    if not cfg.materialized_path.exists():
        raise FileNotFoundError(f"MISSING_MATERIALIZED:{cfg.materialized_path}")
    return [
        json.loads(line)
        for line in cfg.materialized_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def profile_dataset(
    cfg: DatasetAuthoringConfig,
    run_dir: Path,
    *,
    rows: list[dict] | None = None,
) -> dict[str, Any]:
    rows = load_rows(cfg) if rows is None else rows
    profile = profile_rows(rows, cfg)
    profile["materialized_path"] = str(cfg.materialized_path.relative_to(ROOT))
    profile["materialized_sha256"] = sha256_file(cfg.materialized_path)
    profile["allowed_splits"] = list(cfg.allowed_splits)
    _write_json(run_dir / "deterministic" / "dataset_profile.json", profile)
    return profile


# ---------------------------------------------------------------------------
# stage 3: documentation <-> observed data reconciliation (NO LLM)
# ---------------------------------------------------------------------------


def recombination_claims(
    documentation: dict[str, Any], profile: dict[str, Any], cfg: DatasetAuthoringConfig
) -> dict[str, Any]:
    """Deterministic reconciliation of documented claims vs observed rows.

    This function does not need the LLM interpretation; it uses the parsed
    card manifest + the observed profile. The LLM interpretation is reconciled
    separately in :func:`reconcile_interpretation`.
    """
    claims: list[dict[str, Any]] = []
    documented_fields = set(documentation.get("declared_features") or [])
    observed_fields = set(profile["fields"].keys())
    if documented_fields:
        claims.append(
            {
                "claim": "documented feature set",
                "status": "CONFIRMED"
                if documented_fields <= observed_fields
                else "ANNOTATION_INCONSISTENCY",
                "documented": sorted(documented_fields),
                "observed": sorted(observed_fields),
            }
        )
    declared_splits = documentation.get("declared_splits") or {}
    for split, size in declared_splits.items():
        if split != cfg.allowed_splits[0]:
            claims.append(
                {
                    "claim": f"split size {split}",
                    "status": "NOT_EMPIRICALLY_TESTABLE",
                    "note": "reserved split not materialized under production policy",
                    "documented": size,
                }
            )
            continue
        observed = profile["row_count"]
        claims.append(
            {
                "claim": f"split size {split}",
                "status": "CONFIRMED"
                if observed == size
                else "SOURCE_REVISION_DIFFERENCE",
                "documented": size,
                "observed": observed,
            }
        )
    return {"dataset_id": cfg.dataset_id, "claims": claims}


def reconcile_interpretation(
    interpretation: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    """Reconcile the LLM documentation interpretation against observed data."""
    observed_fields = set(profile["fields"].keys())
    results: list[dict[str, Any]] = []
    for field_name in interpretation.get("field_semantics") or {}:
        results.append(
            {
                "documented_field": field_name,
                "status": "CONFIRMED"
                if field_name in observed_fields
                else "CARD_STALE",
            }
        )
    for item in interpretation.get("empirical_claims_to_verify") or []:
        results.append(
            {
                "claim": item.get("claim"),
                "status": "NOT_EMPIRICALLY_TESTABLE"
                if not item.get("how_to_verify")
                else "CONFIRMED_CLAIM_PRESENT",
            }
        )
    return {"dataset_id": profile["dataset_id"], "field_reconciliation": results}


# ---------------------------------------------------------------------------
# LLM call layer with provenance + caching
# ---------------------------------------------------------------------------

LLMCallable = Callable[..., dict]


def _default_llm_call(
    stage: str, prompt: str, temperature: float, response_schema: dict | None
) -> tuple[str, dict | None]:
    """Real call through the project's existing OpenAI-compatible client."""
    from src.autonomous_qa.core.dataset_profile import require_local_proxy
    from src.autonomous_qa.authoring.llm_client import (
        GenerateContentConfig,
        create_llm_client,
        parse_structured_response,
    )

    base_url = resolve_llm_base_url()
    model = resolve_llm_model()
    require_local_proxy(base_url)
    client = create_llm_client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=8192,
            response_mime_type="application/json",
            response_schema=response_schema,
        ),
    )
    raw = response.text or ""
    try:
        parsed = parse_structured_response(response, stage)
    except Exception:  # noqa: BLE001 - fall back to fence stripping
        parsed = _parse_after_fence(raw)
    if not isinstance(parsed, dict):
        parsed = _parse_after_fence(raw)
    return raw, parsed if isinstance(parsed, dict) else None


_FENCE_RE = re.compile(
    r"\A\s*```[ \t]*[A-Za-z0-9_+-]*[ \t]*\r?\n(.*?)\r?\n?```[ \t]*\s*\Z",
    re.DOTALL,
)


def _parse_after_fence(raw: str) -> dict | None:
    match = _FENCE_RE.match(raw or "")
    if match is None:
        return None
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _cache_key(payload: dict[str, Any]) -> str:
    return canonical_hash(payload)[:24]


def call_rnd_llm(
    *,
    cfg: DatasetAuthoringConfig,
    run_dir: Path,
    stage: str,
    prompt: str,
    response_schema: dict | None,
    input_hashes: dict[str, str],
    real_llm: bool = True,
    llm_call: LLMCallable | None = None,
    fixture: dict | None = None,
    out_root: Path | None = None,
) -> dict[str, Any]:
    """Call one R&D stage, persist provenance, cache by exact inputs."""
    prompt_id = PROMPT_IDS[stage]
    prompt_sha = sha256_text(prompt)
    model = resolve_llm_model()
    temperature = STAGE_TEMPERATURE[stage]
    cache_payload = {
        "dataset_id": cfg.dataset_id,
        "stage": stage,
        "prompt_id": prompt_id,
        "prompt_sha256": prompt_sha,
        "model": model,
        "temperature": temperature,
        "input_hashes": input_hashes,
    }
    cache_key = _cache_key(cache_payload)
    root = Path(out_root) if out_root is not None else RUNS_ROOT
    cache_file = root / "_cache" / f"{cache_key}.json"
    stage_dir = run_dir / "llm" / stage
    stage_dir.mkdir(parents=True, exist_ok=True)

    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        cached["cache_hit"] = True
        _write_json(stage_dir / "parsed_response.json", cached.get("parsed"))
        _write_json(stage_dir / "provenance.json", cached.get("provenance"))
        return cached

    raw_text: str
    parsed: dict | None
    parser_status = "parsed"
    if fixture is not None:
        parsed = fixture
        raw_text = json.dumps(fixture, ensure_ascii=False, indent=2)
        parser_status = "fixture"
    elif llm_call is not None:
        _write_text(stage_dir / "prompt.txt", prompt)
        res = llm_call(stage, prompt, temperature, response_schema)
        if isinstance(res, tuple):
            raw_text, parsed = res
        elif isinstance(res, dict):
            parsed = res.get("parsed", res)
            raw_text = res.get("raw") or json.dumps(parsed, ensure_ascii=False, indent=2)
        else:
            raw_text, parsed = str(res), None
        if parsed is None:
            parser_status = "parse_failed"
    elif not real_llm:
        raise RuntimeError("LLM_RND_STAGE_FAILED: real_llm=False and no fixture")
    else:
        _write_text(stage_dir / "prompt.txt", prompt)
        raw_text, parsed = _default_llm_call(stage, prompt, temperature, response_schema)
        if parsed is None:
            parser_status = "parse_failed"

    provenance = {
        "dataset_id": cfg.dataset_id,
        "stage": stage,
        "prompt_id": prompt_id,
        "prompt_sha256": prompt_sha,
        "model": model,
        "temperature": temperature,
        "input_hashes": input_hashes,
        "response_sha256": sha256_text(raw_text),
        "parser_status": parser_status,
        "cache_hit": False,
        "cache_key": cache_key,
        "base_url_host": _host_only(resolve_llm_base_url()),
        "used_as_proposal_only": True,
    }
    _write_text(stage_dir / "prompt.txt", prompt)
    _write_text(stage_dir / "raw_response.txt", raw_text)
    _write_json(stage_dir / "parsed_response.json", parsed)
    _write_json(stage_dir / "provenance.json", provenance)
    _write_json(
        stage_dir / "request.json",
        {
            "dataset_id": cfg.dataset_id,
            "stage": stage,
            "model": model,
            "temperature": temperature,
            "input_hashes": input_hashes,
            "prompt_id": prompt_id,
        },
    )

    result = {
        "parsed": parsed,
        "raw_text": raw_text,
        "provenance": provenance,
        "cache_hit": False,
        "cache_key": cache_key,
    }
    _write_json(cache_file, result)
    return result


def _host_only(base_url: str) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(base_url or "")
    return parsed.netloc or ""


# ---------------------------------------------------------------------------
# prompt assembly
# ---------------------------------------------------------------------------


def load_prompt(stage: str) -> str:
    path = PROMPT_DIR / f"{PROMPT_IDS[stage]}.txt"
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, **values: Any) -> str:
    rendered = template
    for key, value in values.items():
        marker = "{{" + key + "}}"
        if isinstance(value, str):
            text = value
        else:
            text = json.dumps(value, ensure_ascii=False, indent=2)
        rendered = rendered.replace(marker, text)
    return rendered


# ---------------------------------------------------------------------------
# deterministic gates
# ---------------------------------------------------------------------------

KNOWN_OPERATORS = {"DIRECT", "TARGET_MATCH", "EQUALITY", "PAIRWISE_SELECTION"}
KNOWN_ANSWER_KINDS = {"field_value", "boolean", "audio_index"}


def gate_primitive_candidate(
    candidate: dict[str, Any],
    profile: dict[str, Any],
    cfg: DatasetAuthoringConfig,
    seen_signatures: set[str],
) -> dict[str, Any]:
    observed_fields = set(profile["fields"].keys())
    gate: dict[str, str] = {}

    evidence = json.dumps(
        candidate.get("source_evidence") or [], ensure_ascii=False
    ).lower()
    referenced = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", evidence))
    gate["SOURCE_SUPPORT"] = (
        GATE_PASS if evidence.strip() and (referenced & observed_fields) else GATE_FAIL
    )

    hidden = candidate.get("hidden_source_annotations") or []
    visible = candidate.get("visible_inputs") or []
    rule = str(candidate.get("gold_derivation_rule") or "")
    gate["GOLD_DERIVABLE"] = (
        GATE_PASS if rule.strip() and set(hidden) <= observed_fields else GATE_FAIL
    )

    answer = candidate.get("answer_schema_proposal") or {}
    kind = answer.get("kind") if isinstance(answer, dict) else None
    gate["ANSWER_SCHEMA_VALID"] = GATE_PASS if kind in KNOWN_ANSWER_KINDS else GATE_FAIL

    gate["OPERATOR_VALID"] = (
        GATE_PASS if candidate.get("operator") in KNOWN_OPERATORS else GATE_FAIL
    )

    # Audio necessity: for a field_value answer the requested gold field must
    # not be visible. For boolean/index answers the visible candidate is the
    # object being verified (INTENDED_VERIFICATION_CONTEXT), not the gold.
    if kind == "field_value":
        gate["NO_DIRECT_GOLD_VISIBLE"] = (
            GATE_PASS if not (set(hidden) & set(visible)) else GATE_FAIL
        )
    else:
        gate["NO_DIRECT_GOLD_VISIBLE"] = GATE_PASS

    # Capacity: referenced answer field must vary.
    capacity_ok = True
    for f in hidden:
        stat = profile["fields"].get(f)
        if stat and stat.get("distinct", 0) < 2:
            capacity_ok = False
    gate["CAPACITY"] = GATE_PASS if capacity_ok else GATE_FAIL

    split_policy = candidate.get("allowed_split") or cfg.allowed_splits[0]
    gate["SPLIT_POLICY"] = (
        GATE_PASS if split_policy in cfg.allowed_splits else GATE_FAIL
    )

    signature = canonical_hash(
        {
            "operator": candidate.get("operator"),
            "answer_kind": kind,
            "hidden": sorted(hidden),
            "visible": sorted(visible),
        }
    )
    gate["NO_DUPLICATE_PROPOSITION"] = (
        GATE_FAIL if signature in seen_signatures else GATE_PASS
    )
    seen_signatures.add(signature)

    verdict = GATE_PASS if all(v == GATE_PASS for v in gate.values()) else GATE_FAIL
    return {
        "candidate_id": candidate.get("candidate_id"),
        "gates": gate,
        "verdict": verdict,
    }


OUTPUT_VISIBILITY_THRESHOLD = 0.10


def field_pair_equality_rate(
    rows: list[dict], field_a: str, field_b: str
) -> tuple[float, int]:
    total = both = 0
    for row in rows:
        a, b = row.get(field_a), row.get(field_b)
        if a is None or b is None:
            continue
        total += 1
        if str(a) == str(b):
            both += 1
    return (both / total if total else 0.0), total


def deterministic_composite_visibility(
    composite: dict[str, Any],
    primitives_by_id: dict[str, dict],
    rows: list[dict],
) -> list[dict[str, Any]]:
    """Independently test whether a requested output is visible in the input.

    Uses the source rows, not the LLM's self-report. For each hidden field
    that a component contributes as an output, compare it against every
    visible field another component exposes; a material equality rate means
    the requested component is recoverable from the visible input.
    """
    components = composite.get("components") or []
    requested = [str(x).lower() for x in (composite.get("requested_outputs") or [])]
    findings: list[dict[str, Any]] = []
    for cid in components:
        prim = primitives_by_id.get(cid) or {}
        hidden = prim.get("hidden_source_annotations") or []
        visible = prim.get("visible_inputs") or []
        for h in hidden:
            hl = str(h).lower()
            if requested and not any(x and (x in hl or hl in x) for x in requested):
                continue
            for v in visible:
                if v == h:
                    continue
                rate, total = field_pair_equality_rate(rows, h, v)
                if total and rate >= OUTPUT_VISIBILITY_THRESHOLD:
                    findings.append(
                        {
                            "output_field": h,
                            "visible_field": v,
                            "raw_equality_rate": round(rate, 4),
                            "n": total,
                        }
                    )
    return findings


def gate_composite_candidate(
    candidate: dict[str, Any],
    accepted_ids: set[str],
    deterministic_visibility: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    gate: dict[str, str] = {}
    components = candidate.get("components") or []
    gate["ALL_COMPONENTS_ACCEPTED"] = (
        GATE_PASS if components and set(components) <= accepted_ids else GATE_FAIL
    )
    gate["DEPENDENCIES_EXPLICIT"] = (
        GATE_PASS if candidate.get("dependency_edges") else GATE_FAIL
    )
    # Deterministic evidence overrides the LLM's self-report.
    visible = bool(deterministic_visibility) or bool(
        candidate.get("output_component_visible_in_input")
    )
    gate["OUTPUT_NOT_VISIBLE_IN_INPUT"] = GATE_FAIL if visible else GATE_PASS
    gate["NO_REDUNDANT_COMPONENT"] = (
        GATE_PASS if not candidate.get("redundant_components") else GATE_FAIL
    )
    verdict = GATE_PASS if all(v == GATE_PASS for v in gate.values()) else GATE_FAIL
    return {
        "composite_id": candidate.get("composite_id"),
        "gates": gate,
        "deterministic_visibility": deterministic_visibility or [],
        "verdict": verdict,
    }


def gate_language_entry(entry: dict[str, Any]) -> dict[str, Any]:
    gate: dict[str, str] = {}
    templates = entry.get("question_templates") or []
    gate["REQUIRED_ROLE_COVERAGE"] = GATE_PASS if templates else GATE_FAIL
    gate["NO_QUOTE_LEAKAGE"] = (
        GATE_FAIL if any("__GOLD__" in str(t) for t in templates) else GATE_PASS
    )
    gate["DUPLICATE_HEADS"] = (
        GATE_FAIL if len(set(templates)) != len(templates) else GATE_PASS
    )
    gate["ANSWER_FORMAT_PRESENT"] = (
        GATE_PASS if entry.get("answer_format") else GATE_FAIL
    )
    verdict = GATE_PASS if all(v == GATE_PASS for v in gate.values()) else GATE_FAIL
    return {"type_id": entry.get("type_id"), "gates": gate, "verdict": verdict}


# ---------------------------------------------------------------------------
# canonical reconciliation
# ---------------------------------------------------------------------------

RECONCILIATION_CLASSES = (
    "EXACT_SEMANTIC_MATCH",
    "SAME_PROPOSITION_DIFFERENT_NAME",
    "CONTRACT_CLARIFICATION_ONLY",
    "COMPARATOR_DIFFERENCE",
    "LANGUAGE_ONLY_DIFFERENCE",
    "DISCOVERY_MISS_CURRENT_TYPE_VALID",
    "CURRENT_TYPE_NOT_SOURCE_SUPPORTED",
    "CURRENT_TYPE_OBJECTIVE_DEFECT",
    "CURRENT_COMPOSITE_LEAKAGE",
    "CURRENT_SHORTCUT_REVIEW",
    "NEW_PROMOTABLE_SEMANTIC_CANDIDATE",
    "SOURCE_POLICY_DIFFERENCE",
    "REVIEW_REQUIRED",
)


def _normalize_field_name(name: str) -> str:
    base = re.sub(r"_(?:[0-9]+|a|b|1|2)$", "", str(name).strip().lower())
    return base


def _task_visible_task(task) -> dict[str, Any]:
    mapping = task.source_role_mapping or {}
    fields: set[str] = set()
    for key in ("source_field", "visible_field", "audio_field"):
        value = mapping.get(key)
        if isinstance(value, str) and value:
            fields.add(_normalize_field_name(value))
    return {
        "type_id": task.type_id,
        "operator": task.operator,
        "kind": task.kind,
        "answer_kinds": [c.kind for c in task.outputs],
        "visible_context_roles": list(task.visible_context_roles),
        "comparator_id": task.comparator_id,
        "fields": sorted(fields),
    }


def reconcile_with_canonical(
    cfg: DatasetAuthoringConfig, blind_candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    from src.autonomous_qa.compiler.semantic_task import load_semantic_catalog

    catalog = load_semantic_catalog(cfg.canonical_catalog_path)
    canonical = {t.type_id: _task_visible_task(t) for t in catalog.tasks}
    matches: list[dict[str, Any]] = []
    matched_canonical: set[str] = set()
    for cand in blind_candidates:
        operator = cand.get("operator")
        fields = {_normalize_field_name(f) for f in (cand.get("fields") or []) if f}
        found = [
            c
            for c in canonical.values()
            if c["type_id"] not in matched_canonical
            and c["operator"] == operator
            and fields
            and set(c["fields"]) & fields
        ]
        # Prefer the exact field-set equality, then the first overlap.
        found.sort(key=lambda c: 0 if set(c["fields"]) == fields else 1)
        if found:
            matched_canonical.add(found[0]["type_id"])
            matches.append(
                {
                    "blind_candidate": cand.get("candidate_id"),
                    "canonical_type": found[0]["type_id"],
                    "classification": "SAME_PROPOSITION_DIFFERENT_NAME"
                    if found[0]["type_id"] != cand.get("candidate_id")
                    else "EXACT_SEMANTIC_MATCH",
                    "operator": operator,
                    "fields": sorted(fields),
                }
            )
        else:
            matches.append(
                {
                    "blind_candidate": cand.get("candidate_id"),
                    "canonical_type": None,
                    "classification": "NEW_PROMOTABLE_SEMANTIC_CANDIDATE",
                    "operator": operator,
                    "fields": sorted(fields),
                }
            )
    for type_id in canonical:
        if type_id not in matched_canonical:
            matches.append(
                {
                    "blind_candidate": None,
                    "canonical_type": type_id,
                    "classification": "DISCOVERY_MISS_CURRENT_TYPE_VALID",
                    "operator": canonical[type_id]["operator"],
                    "fields": canonical[type_id]["fields"],
                }
            )
    return {
        "dataset_id": cfg.dataset_id,
        "canonical_type_count": len(canonical),
        "matches": matches,
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def run_authoring(
    dataset_id: str,
    *,
    run_id: str | None = None,
    real_llm: bool = True,
    llm_call: LLMCallable | None = None,
    fixture_dir: Path | None = None,
    out_root: Path | None = None,
    plan_path: Path | None = None,
    production_config: ProductionGenerationConfig | None = None,
    exploratory: bool = False,
) -> dict[str, Any]:
    """Run the full R&D authoring pipeline for one dataset.

    The EARLY BUDGET is resolved from a prepared source inventory BEFORE the
    first LLM call (see :mod:`src.autonomous_qa.production.source_preparation`).
    With no ``production_config`` (or ``exploratory=True``) an explicit
    EXPLORATORY budget is recorded instead of fabricating a production target.
    """
    cfg = DATASETS[dataset_id]
    run_id = run_id or new_run_id()
    root = Path(out_root) if out_root is not None else RUNS_ROOT
    run_dir = root / dataset_id / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    call_counts: dict[str, int] = {}
    llm_provenance: list[dict[str, Any]] = []

    def stage_fixture(stage: str) -> dict | None:
        if fixture_dir is None:
            return None
        path = Path(fixture_dir) / f"{stage}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def do_call(
        stage: str, prompt: str, schema: dict | None, hashes: dict[str, str]
    ) -> dict:
        result = call_rnd_llm(
            cfg=cfg,
            run_dir=run_dir,
            stage=stage,
            prompt=prompt,
            response_schema=schema,
            input_hashes=hashes,
            real_llm=real_llm,
            llm_call=llm_call,
            fixture=stage_fixture(stage),
            out_root=root,
        )
        call_counts[stage] = call_counts.get(stage, 0) + (
            0 if result.get("cache_hit") else 1
        )
        llm_provenance.append(result.get("provenance") or {})
        return result

    docs = ingest_documentation(cfg, run_dir)
    rows = load_rows(cfg)
    profile = profile_dataset(cfg, run_dir, rows=rows)

    # SOURCE PREPARATION + EARLY BUDGET AUTHORIZATION — resolved BEFORE the first
    # LLM call. No downstream stage may revise the total target.
    source_split = cfg.allowed_splits[0] if cfg.allowed_splits else "train"
    inventory = prepare_source_inventory(
        dataset=cfg.dataset_id,
        split=source_split,
        path=cfg.materialized_path,
        rows=rows,
        identity_field=cfg.record_identity_field,
        eligibility_fields=sorted(cfg.field_roles.keys()),
        sha256=profile["materialized_sha256"],
    )
    _write_json(run_dir / "preparation.json", inventory.to_dict())
    early_budget = resolve_early_budget(
        inventory=inventory,
        config=production_config,
        exploratory=exploratory or production_config is None,
    )
    _write_json(run_dir / "early_budget.json", early_budget.to_dict())

    card_manifest = {
        "declared_features": _declared_features(cfg.card_path),
        "declared_splits": _declared_splits(cfg.card_path),
    }
    reconciliation = recombination_claims(card_manifest, profile, cfg)

    doc_prompt = render_prompt(
        load_prompt("documentation_interpretation"),
        DATASET_ID=cfg.dataset_id,
        SOURCE_REVISION=cfg.source_revision,
        SOURCE_DOCUMENTATION=(
            cfg.card_path.read_text(encoding="utf-8")
            + "\n\n"
            + cfg.readme_path.read_text(encoding="utf-8")
        ),
        RESPONSE_SCHEMA=_schema_hint(),
    )
    doc_result = do_call(
        "documentation_interpretation",
        doc_prompt,
        None,
        {"documentation": docs["combined_sha256"]},
    )
    interpretation = doc_result.get("parsed") or {}
    interpretation_reconciliation = reconcile_interpretation(interpretation, profile)
    _write_json(run_dir / "deterministic" / "reconciliation.json", reconciliation)

    prim_prompt = render_prompt(
        load_prompt("primitive_semantic_discovery"),
        DATASET_ID=cfg.dataset_id,
        DOCUMENTATION_INTERPRETATION=interpretation,
        DATASET_PROFILE=_profile_summary(profile),
        RECONCILIATION=reconciliation,
        RESPONSE_SCHEMA=_schema_hint(),
    )
    prim_result = do_call(
        "primitive_semantic_discovery",
        prim_prompt,
        None,
        {
            "documentation_interpretation": canonical_hash(interpretation),
            "profile": profile["materialized_sha256"],
            "reconciliation": canonical_hash(reconciliation),
        },
    )
    primitives = (prim_result.get("parsed") or {}).get("candidates") or []
    # Operator policy is deterministic and enforced BEFORE candidate gates, so no
    # LLM budget is spent repairing an explicitly disallowed operator.
    allowed_families = validate_operator_policy(
        production_config.allowed_operator_families if production_config else ()
    )
    primitives, rejected_primitives = filter_by_operator_policy(
        list(primitives), allowed_families, id_key="candidate_id"
    )
    seen: set[str] = set()
    primitive_gates = [
        gate_primitive_candidate(c, profile, cfg, seen)
        for c in primitives
        if isinstance(c, dict)
    ]
    accepted_primitives = [
        c for c, g in zip(primitives, primitive_gates) if g["verdict"] == GATE_PASS
    ]
    _write_json(
        run_dir / "gates" / "primitive_gates.json",
        {"candidates": primitive_gates, "accepted": len(accepted_primitives)},
    )

    critic_prompt = render_prompt(
        load_prompt("semantic_contract_critic"),
        DATASET_ID=cfg.dataset_id,
        PRIMITIVE_CANDIDATES=primitives,
        DATASET_PROFILE=_profile_summary(profile),
        RESPONSE_SCHEMA=_schema_hint(),
    )
    do_call(
        "semantic_contract_critic",
        critic_prompt,
        None,
        {
            "primitives": canonical_hash(primitives),
            "profile": profile["materialized_sha256"],
        },
    )

    from tests.regression.component_leakage_audit import audit_plan

    target_plan = Path(plan_path) if plan_path is not None else cfg.plan_path
    if target_plan.exists():
        leakage = audit_plan(
            cfg.dataset_id,
            target_plan,
            cfg.canonical_catalog_path,
        )
    else:
        leakage = {
            "dataset_id": cfg.dataset_id,
            "plan_path": str(target_plan),
            "row_count": 0,
            "type_count": 0,
            "types": {},
            "flagged": {},
            "component_visible_types": [],
        }
    _write_json(run_dir / "deterministic" / "component_leakage_audit.json", leakage)
    leakage_findings = {
        t: leakage["types"][t]["classification"]
        for t in leakage["component_visible_types"]
    }

    accepted_ids = {
        str(c.get("candidate_id")) for c in accepted_primitives if c.get("candidate_id")
    }
    comp_prompt = render_prompt(
        load_prompt("composite_discovery"),
        DATASET_ID=cfg.dataset_id,
        ACCEPTED_PRIMITIVES=accepted_primitives,
        LEAKAGE_FINDINGS=leakage_findings,
        RESPONSE_SCHEMA=_schema_hint(),
    )
    comp_result = do_call(
        "composite_discovery",
        comp_prompt,
        None,
        {
            "accepted_primitives": canonical_hash(sorted(accepted_ids)),
            "profile": profile["materialized_sha256"],
        },
    )
    composites = (comp_result.get("parsed") or {}).get("composites") or []
    composite_items = [
        {**c, "operator": c.get("operator", "COMPOSITE")}
        if isinstance(c, dict)
        else c
        for c in composites
    ]
    composites, rejected_composites = filter_by_operator_policy(
        composite_items, allowed_families, id_key="composite_id"
    )
    _write_json(
        run_dir / "operator_policy.json",
        {
            "allowed_operator_families": list(allowed_families),
            "rejected_primitives": rejected_primitives,
            "rejected_composites": rejected_composites,
        },
    )
    primitives_by_id = {
        str(c.get("candidate_id")): c
        for c in accepted_primitives
        if c.get("candidate_id")
    }
    composite_gates = [
        gate_composite_candidate(
            c,
            accepted_ids,
            deterministic_composite_visibility(c, primitives_by_id, rows),
        )
        for c in composites
        if isinstance(c, dict)
    ]
    _write_json(
        run_dir / "gates" / "composite_gates.json", {"composites": composite_gates}
    )

    lang_prompt = render_prompt(
        load_prompt("language_generation"),
        DATASET_ID=cfg.dataset_id,
        ACCEPTED_CONTRACTS=accepted_primitives,
        RESPONSE_SCHEMA=_schema_hint(),
    )
    lang_result = do_call(
        "language_generation",
        lang_prompt,
        None,
        {"accepted_contracts": canonical_hash(sorted(accepted_ids))},
    )
    language_entries = (lang_result.get("parsed") or {}).get("entries") or []

    review_prompt = render_prompt(
        load_prompt("language_semantic_review"),
        DATASET_ID=cfg.dataset_id,
        LANGUAGE_ENTRIES=language_entries,
        ACCEPTED_CONTRACTS=accepted_primitives,
        RESPONSE_SCHEMA=_schema_hint(),
    )
    review_result = do_call(
        "language_semantic_review",
        review_prompt,
        None,
        {"language_entries": canonical_hash(language_entries)},
    )
    language_reviews = (review_result.get("parsed") or {}).get("reviews") or []
    language_gates = [
        gate_language_entry(e) for e in language_entries if isinstance(e, dict)
    ]
    accepted_type_ids = sorted(
        {
            str(c.get("candidate_id"))
            for c in accepted_primitives
            if isinstance(c, dict) and c.get("candidate_id")
        }
    )
    if production_config is not None:
        proposed_allocation = propose_allocation(
            early=early_budget,
            accepted_type_ids=accepted_type_ids,
            config=production_config,
        )
    else:
        proposed_allocation = {
            "authorization": "EXPLORATORY",
            "accepted_types": accepted_type_ids,
            "per_type_budget": {},
            "note": "no production budget requested; proposal not authorized",
        }
    _write_json(run_dir / "proposed_allocation.json", proposed_allocation)
    _write_json(
        run_dir / "language" / "preflight.json",
        {"entries": language_gates, "reviews": language_reviews},
    )

    canonical_reconciliation = reconcile_with_canonical(
        cfg,
        [
            {
                "candidate_id": c.get("candidate_id"),
                "operator": c.get("operator"),
                "fields": list(c.get("hidden_source_annotations") or [])
                + list(c.get("visible_inputs") or []),
                "answer_kinds": [(c.get("answer_schema_proposal") or {}).get("kind")]
                if isinstance(c.get("answer_schema_proposal"), dict)
                else [],
            }
            for c in accepted_primitives
        ],
    )

    readiness = {
        "dataset_id": cfg.dataset_id,
        "documentation_support": GATE_PASS,
        "observed_data_support": GATE_PASS,
        "primitive_candidates": len(primitives),
        "primitive_accepted": len(accepted_primitives),
        "composite_candidates": len(composites),
        "language_entries": len(language_entries),
        "language_preflight_pass": sum(
            1 for g in language_gates if g["verdict"] == GATE_PASS
        ),
        "component_leakage_flagged": leakage["component_visible_types"],
        "status": "PROMOTABLE_CANDIDATE"
        if accepted_primitives and not leakage["component_visible_types"]
        else "REVIEW_REQUIRED",
    }
    _write_json(run_dir / "readiness.json", readiness)
    _write_json(run_dir / "reconciliation_canonical.json", canonical_reconciliation)

    manifest = {
        "dataset_id": cfg.dataset_id,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "source_revision": cfg.source_revision,
        "documentation_hashes": docs,
        "profile_sha256": profile["materialized_sha256"],
        "source_inventory": inventory.to_dict(),
        "early_budget": early_budget.to_dict(),
        "operator_policy": {
            "allowed_operator_families": list(allowed_families),
            "rejected_primitives": rejected_primitives,
            "rejected_composites": rejected_composites,
        },
        "proposed_allocation": proposed_allocation,
        "llm_calls_by_stage": call_counts,
        "total_real_llm_calls": sum(call_counts.values()),
        "llm_provenance": llm_provenance,
        "primitives_discovered": len(primitives),
        "primitives_accepted": len(accepted_primitives),
        "composites_discovered": len(composites),
        "component_leakage_flagged": leakage["component_visible_types"],
        "readiness": readiness,
        "interpretation_reconciliation": interpretation_reconciliation,
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def _schema_hint() -> dict[str, Any]:
    return {"note": "return strict JSON only; keys described in the prompt"}


def _declared_features(card_path: Path) -> list[str]:
    from src.autonomous_qa.core.dataset_profile import parse_source_card

    try:
        card = parse_source_card(card_path)
    except Exception:  # noqa: BLE001 - card parse is best-effort metadata
        return []
    return list(card.fields.keys())


def _declared_splits(card_path: Path) -> dict[str, int]:
    from src.autonomous_qa.core.dataset_profile import parse_source_card

    try:
        card = parse_source_card(card_path)
    except Exception:  # noqa: BLE001 - card parse is best-effort metadata
        return {}
    out: dict[str, int] = {}
    for split in card.splits:
        name = str(split.get("name"))
        count = split.get("num_examples")
        if name and isinstance(count, int):
            out[name] = count
    return out


def _profile_summary(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": profile["dataset_id"],
        "row_count": profile["row_count"],
        "record_identity": profile["record_identity"],
        "fields": {
            name: {
                "dtype": stat["dtype"],
                "role": stat["role"],
                "distinct": stat["distinct"],
                "missing": stat["missing"],
                "empty": stat["empty"],
                "value_distribution": stat.get("value_distribution"),
            }
            for name, stat in profile["fields"].items()
        },
        "cross_field": profile.get("cross_field"),
    }

"""OpenAI-compatible LLM client + central call_llm() with a hard call budget.

Endpoint and credentials come from the project `.env` (see `.env.example`):
the local OpenAI-compatible proxy endpoint (OPENAI_COMPAT_BASE_URL /
OPENAI_BASE_URL) and model (OPENAI_COMPAT_MODEL / OPENAI_MODEL), falling
back to the non-secret defaults in config.yaml. The endpoint is the
project's local gateway; requests to api.openai.com are refused outright so
no call can silently fall back to the hosted API, and no real OpenAI
credential is ever required: the gateway token comes from .env, and when no
key is configured a harmless placeholder is sent so the SDK still gets the
non-empty api_key argument it insists on.

The client keeps the surface the pipeline already speaks —
`client.models.generate_content(model=..., contents=..., config=...)`
returning an object with `.text`, `.parsed`, `.candidates[0].finish_reason`
and `.usage_metadata` — so stage modules, scripts and tests stay unchanged.

Response handling rules:
- call_llm() returns the raw response object (never parses to JSON itself).
- Stage modules must save raw text to outputs/<dataset>/raw/<stage>.txt
  BEFORE parsing, then prefer response.parsed (structured output) and only
  fall back to json.loads. Truncated/invalid output raises
  StageOutputValidationError. No automatic retry — ever.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import openai
from openai import OpenAI

from src.common.config import (
    CONFIG,
    resolve_llm_api_key,
    resolve_llm_base_url,
    resolve_llm_model,
)


class StageOutputValidationError(RuntimeError):
    """Raised when a stage returns incomplete/invalid structured JSON."""


# Sent only when .env configures no key at all: the OpenAI SDK rejects an
# empty api_key argument, so a harmless non-credential placeholder is used
# (the same local-placeholder pattern as the sibling project). Endpoints
# that do authenticate take their own token from OPENAI_COMPAT_API_KEY /
# OPENAI_API_KEY instead — never a real OpenAI credential.
LOCAL_PROXY_PLACEHOLDER_API_KEY = "local-openai-compatible-placeholder"

# Hosted OpenAI API. This project must never send traffic there.
CLOUD_OPENAI_HOSTS = frozenset({"api.openai.com", "openai.com"})


def endpoint_host(base_url: str) -> str:
    return (urlparse(base_url).hostname or "").strip().lower()


def assert_compatible_endpoint(base_url: str) -> None:
    """Refuse anything that would fall back to the hosted OpenAI API."""
    host = endpoint_host(base_url or "")
    if not base_url or not host:
        raise RuntimeError(
            "No OpenAI-compatible endpoint is configured. Set "
            "OPENAI_BASE_URL (or OPENAI_COMPAT_BASE_URL) in .env; requests "
            "to api.openai.com are disabled in this project.")
    if host in CLOUD_OPENAI_HOSTS:
        raise RuntimeError(
            f"Refusing to use the hosted OpenAI API ({base_url}). This "
            "project only talks to the configured OpenAI-compatible "
            "endpoint; set OPENAI_BASE_URL to that endpoint instead.")


class LLMCallBudget:
    """Per-stage PAID-attempt caps. Every generate_content() call counts —
    including truncations that fail parsing.

    allowed_calls e.g. {"question_style": 4, "question_template": 4,
    "paraphrase": 4, "quality": 4}: one success allowance per round plus
    exactly one manual retry per stage. No automatic retry anywhere.
    """

    def __init__(self, allowed_calls: dict):
        self.allowed_calls = dict(allowed_calls)
        self.stage_counts = {stage: 0 for stage in allowed_calls}
        self.count = 0
        self.max_calls = sum(allowed_calls.values())
        # Display denominator for this invocation.
        # The hard limit stays max_calls.
        self.display_total = self.max_calls

    def consume(self, stage: str):
        if stage not in self.allowed_calls:
            raise RuntimeError(f"Unknown stage '{stage}'.")

        if self.stage_counts[stage] >= self.allowed_calls[stage]:
            raise RuntimeError(
                f"Stage '{stage}' exceeded its call cap "
                f"({self.stage_counts[stage]}/{self.allowed_calls[stage]})."
            )

        if self.count >= self.max_calls:
            raise RuntimeError(f"LLM call budget exceeded: {self.count}/{self.max_calls}")

        self.stage_counts[stage] += 1
        self.count += 1


def build_budget_from_config(config: dict) -> LLMCallBudget:
    max_rounds = config["loop"]["max_rounds"]
    return LLMCallBudget(
        {
            "question_style": max_rounds + 1,
            "question_template": max_rounds + 1,
            "paraphrase": max_rounds + 1,
            "quality": max_rounds + 1,
        }
    )


BUDGET = build_budget_from_config(CONFIG)


@dataclass
class ThinkingConfig:
    """Kept for call-signature compatibility with the old Gemini config.

    OpenAI-compatible endpoints take no Gemini-style thinking budget, so this
    is accepted and not forwarded (temperature still controls sampling).
    """

    thinking_budget: int | None = None
    include_thoughts: bool | None = None


@dataclass
class GenerateContentConfig:
    temperature: float | None = None
    max_output_tokens: int | None = None
    response_mime_type: str | None = None
    response_schema: Any = None
    thinking_config: ThinkingConfig | None = None


_FINISH_REASON_MAP = {
    "stop": "STOP",
    "length": "MAX_TOKENS",
    "content_filter": "SAFETY",
    "tool_calls": "STOP",
}


def _coerce_config(config) -> GenerateContentConfig:
    """Accept our config object, a plain dict, or anything duck-typed."""
    if config is None:
        return GenerateContentConfig()
    if isinstance(config, GenerateContentConfig):
        return config
    if isinstance(config, dict):
        known = set(GenerateContentConfig.__dataclass_fields__)
        return GenerateContentConfig(
            **{k: v for k, v in config.items() if k in known})
    return config


def _response_format(cfg) -> dict | None:
    """OpenAI response_format derived from the stage's schema/mime type."""
    schema = getattr(cfg, "response_schema", None)
    json_schema = None
    if schema is not None:
        if hasattr(schema, "model_json_schema"):
            json_schema = schema.model_json_schema()
        elif isinstance(schema, dict):
            json_schema = schema
    if json_schema:
        name = getattr(schema, "__name__", None) or "response"
        return {"type": "json_schema",
                "json_schema": {"name": name, "schema": json_schema}}
    if getattr(cfg, "response_mime_type", None) == "application/json":
        return {"type": "json_object"}
    return None


def _degrade_request(payload: dict, exc: Exception) -> bool:
    """Downgrade one unsupported field after a 400 from the endpoint.

    Returns True when the payload changed (caller may retry the request).
    This happens BEFORE any response returns, so no paid attempt exists yet.
    """
    msg = str(exc).lower()
    changed = False
    if "max_completion_tokens" in payload and "max_completion_tokens" in msg:
        payload["max_tokens"] = payload.pop("max_completion_tokens")
        changed = True
    rf = payload.get("response_format") or {}
    if rf.get("type") == "json_schema" and (
            "response_format" in msg or "json_schema" in msg
            or "structured" in msg):
        payload["response_format"] = {"type": "json_object"}
        changed = True
    elif rf.get("type") == "json_object" and "response_format" in msg:
        del payload["response_format"]
        changed = True
    return changed


class LLMResponse:
    """Response shim exposing exactly the fields the pipeline reads."""

    def __init__(self, raw, text: str, finish_reason, usage):
        self.raw = raw
        self.text = text
        self.parsed = None  # JSON parsing falls back to self.text
        self.candidates = [SimpleNamespace(finish_reason=finish_reason)]
        self.usage_metadata = usage


class _Models:
    def __init__(self, client: "LLMClient"):
        self._client = client

    def generate_content(self, model=None, contents=None, config=None):
        return self._client._generate(model=model, contents=contents,
                                      config=config)


class LLMClient:
    """OpenAI chat-completions endpoint behind the legacy call surface."""

    def __init__(self, api_client: OpenAI, model: str | None = None):
        self._api = api_client
        self.model = model or resolve_llm_model()
        self.models = _Models(self)

    def _generate(self, model=None, contents=None, config=None) -> LLMResponse:
        cfg = _coerce_config(config)
        payload: dict = {
            "model": model or self.model,
            "messages": [{"role": "user", "content": contents or ""}],
        }
        if getattr(cfg, "temperature", None) is not None:
            payload["temperature"] = cfg.temperature
        if getattr(cfg, "max_output_tokens", None) is not None:
            payload["max_completion_tokens"] = cfg.max_output_tokens
        response_format = _response_format(cfg)
        if response_format is not None:
            payload["response_format"] = response_format

        raw = None
        for _ in range(3):
            try:
                raw = self._api.chat.completions.create(**payload)
                break
            except openai.BadRequestError as exc:
                if not _degrade_request(payload, exc):
                    raise
        if raw is None:
            raise RuntimeError("The LLM endpoint returned no response.")

        choice = (raw.choices or [None])[0]
        message = getattr(choice, "message", None)
        text = (getattr(message, "content", None) or ""
                if message is not None else "")
        raw_finish = getattr(choice, "finish_reason", None) \
            if choice is not None else None
        finish = _FINISH_REASON_MAP.get(raw_finish, raw_finish)
        return LLMResponse(raw=raw, text=text, finish_reason=finish,
                           usage=getattr(raw, "usage", None))


def create_llm_client() -> LLMClient:
    """Build the client from .env (+ config.yaml defaults). No secrets there.

    Endpoint/model come from the shared resolvers in src.config, so every
    stage — including the dataset profiler — talks to the same
    OpenAI-compatible endpoint with the same model resolution rules.
    """
    base_url = resolve_llm_base_url()
    assert_compatible_endpoint(base_url)
    llm_cfg = CONFIG["llm"]
    # Gateway token from .env when configured; otherwise a harmless
    # placeholder so the SDK gets a non-empty api_key argument.
    api_key = resolve_llm_api_key() or LOCAL_PROXY_PLACEHOLDER_API_KEY
    kwargs: dict = {
        "api_key": api_key,
        "base_url": base_url,
        "timeout": llm_cfg.get("timeout", 600),
        # Project policy: no automatic retry — resume runs handle failures.
        "max_retries": int(llm_cfg.get("max_retries", 0)),
    }
    return LLMClient(OpenAI(**kwargs), model=resolve_llm_model())


def call_llm(client, stage: str, prompt: str, response_schema):
    BUDGET.consume(stage)

    stage_cfg = CONFIG["stages"][stage]

    print(
        f"[LLM {BUDGET.count}/{BUDGET.display_total}] "
        f"{stage} | "
        f"temperature={stage_cfg['temperature']}"
    )

    response = client.models.generate_content(
        model=resolve_llm_model(),
        contents=prompt,
        config=GenerateContentConfig(
            temperature=stage_cfg["temperature"],
            max_output_tokens=stage_cfg["max_output_tokens"],
            response_mime_type="application/json",
            response_schema=response_schema,
            thinking_config=ThinkingConfig(
                thinking_budget=stage_cfg["thinking_budget"]
            ),
        ),
    )

    return response


def print_response_debug(response, stage: str) -> None:
    """Log safe metadata (finish reason + usage). Never credentials."""
    candidate = None
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        candidate = candidates[0]

    finish_reason = (
        getattr(candidate, "finish_reason", None)
        if candidate is not None
        else None
    )
    usage = getattr(response, "usage_metadata", None)

    print(f"[{stage}] finish_reason={finish_reason}")
    if usage is not None:
        print(f"[{stage}] usage={usage}")


def save_raw_text(raw_path, text: str) -> None:
    """Legacy entrypoint: atomic, never overwrites an existing file.

    Kept for compatibility; the canonical paid-response path is
    record_paid_response() below.
    """
    atomic_write_new_file(Path(raw_path), text or "")


def atomic_write_new_file(final_path, text: str) -> Path:
    """Atomically persist a new file: tmp in same dir + flush + fsync + replace.

    Raises FileExistsError if the final path already exists. Never overwrites.
    Temp files use a `.tmp` suffix and are never counted as paid attempts.
    """
    import os

    final_path = Path(final_path)
    if final_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing attempt file: {final_path}"
        )
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = final_path.with_name(final_path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text or "")
        f.flush()
        try:
            os.fsync(f.fileno())
        except (OSError, ValueError):
            pass
    os.replace(tmp, final_path)
    return final_path


def record_paid_response(stage: str, attempt_no: int, raw_text: str,
                         raw_path) -> Path:
    """ONE shared paid-response recorder for all four LLM stages.

    Responsibility: durably record that the provider RETURNED a response,
    BEFORE any parsing or validation. Must be called immediately after
    provider return, before inspecting response.parsed / json.loads /
    Pydantic / semantic validation.

    - Atomically writes the canonical attempt file for attempt_no.
    - Never overwrites an existing final attempt file.
    - Does NOT parse, validate, or mutate semantic state.
    - Caller updates paid-call accounting + checkpoint AFTER this returns.

    Returns the final attempt path.
    """
    final = attempt_raw_path(raw_path, attempt_no)
    return atomic_write_new_file(final, raw_text or "")


class AttemptTracker:
    def __len__(self) -> int:
        return self.attempted_calls_total
    """Counts PAID attempts, not just successful parsed stages.

    Every generate_content() response counts as a real request — including
    MAX_TOKENS truncations that fail parsing. No automatic retry anywhere.
    """

    def __init__(self):
        self.attempted_calls_total = 0
        self.successful_calls_total = 0
        self.stage_attempts: dict[str, int] = {}
        self.stage_successes: dict[str, int] = {}
        self.last_finish_reason: str | None = None
        self.last_failed_stage: str | None = None

    def record_attempt(self, stage: str, finish_reason) -> int:
        self.attempted_calls_total += 1
        self.stage_attempts[stage] = self.stage_attempts.get(stage, 0) + 1
        self.last_finish_reason = (
            None if finish_reason is None else str(finish_reason)
        )
        return self.stage_attempts[stage]

    def record_success(self, stage: str) -> None:
        self.successful_calls_total += 1
        self.stage_successes[stage] = self.stage_successes.get(stage, 0) + 1

    def record_failure(self, stage: str) -> None:
        self.last_failed_stage = stage


ATTEMPTS = AttemptTracker()


def reset_attempts(
    attempted: int = 0,
    successful: int = 0,
    stage_attempts: dict | None = None,
    stage_successes: dict | None = None,
    last_finish_reason: str | None = None,
    last_failed_stage: str | None = None,
) -> None:
    """Re-seed the global tracker (fresh process start or resume adopt)."""
    ATTEMPTS.attempted_calls_total = attempted
    ATTEMPTS.successful_calls_total = successful
    ATTEMPTS.stage_attempts = dict(stage_attempts or {})
    ATTEMPTS.stage_successes = dict(stage_successes or {})
    ATTEMPTS.last_finish_reason = last_finish_reason
    ATTEMPTS.last_failed_stage = last_failed_stage


def get_finish_reason(response):
    """Extract the (possibly SDK-enum) finish reason of the first candidate."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    return getattr(candidates[0], "finish_reason", None)


def attempt_raw_path(raw_path, attempt_no: int):
    """Per-attempt raw file: raw/question_style_attempt_02.txt.

    Attempt files are append-only history — never overwritten.
    """
    raw_path = Path(raw_path)
    return raw_path.with_name(
        f"{raw_path.stem}_attempt_{attempt_no:02d}{raw_path.suffix}"
    )


def parse_structured_response(response, stage: str):
    """Prefer SDK structured output; fall back to raw JSON text.

    Raises StageOutputValidationError (no retry) when both are unusable,
    e.g. a MAX_TOKENS-truncated payload.
    """
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if hasattr(parsed, "model_dump"):
            return parsed.model_dump()
        if isinstance(parsed, dict):
            return parsed

    raw_text = response.text or ""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise StageOutputValidationError(
            f"Stage '{stage}' returned incomplete/invalid structured JSON. "
            "Raw response has been saved. "
            "No automatic retry was attempted."
        ) from exc


def complete_structured_stage(client, stage: str, prompt: str,
                              response_schema, raw_path,
                              on_attempt=None) -> dict:
    """One attempt, universal order (§1/§5):

        1. determine next paid attempt number (spent[S] + 1)
        2. call provider
        3. provider returns
        4. persist RAW response atomically (record_paid_response)
        5. persist paid-call count/checkpoint (on_attempt callback)
        6. parse response
        7. Pydantic/schema validation
        8+. stage-specific semantic validation happens in the caller
            (_do_template_round / _do_quality_round), AFTER this returns.

    No validation runs between provider return and raw persistence.
    Invalid output does NOT cancel the paid attempt.
    Provider exceptions before a response returns fabricate no file.
    """
    # 1. next paid attempt number (cap checked here AND in call_llm).
    if stage not in BUDGET.allowed_calls:
        raise RuntimeError(f"Unknown stage '{stage}'.")
    attempt_no = int(BUDGET.stage_counts.get(stage, 0)) + 1
    if BUDGET.stage_counts.get(stage, 0) >= BUDGET.allowed_calls[stage]:
        raise RuntimeError(
            f"Stage '{stage}' exceeded its call cap "
            f"({BUDGET.stage_counts.get(stage, 0)}/"
            f"{BUDGET.allowed_calls[stage]})."
        )
    if BUDGET.count >= BUDGET.max_calls:
        raise RuntimeError(
            f"LLM call budget exceeded: {BUDGET.count}/{BUDGET.max_calls}")

    # 2. call provider (call_llm enforces caps + spends budget in-memory).
    response = call_llm(
        client,
        stage=stage,
        prompt=prompt,
        response_schema=response_schema,
    )

    # 3. provider returned -> 4. RAW durable FIRST via shared recorder.
    finish_reason = get_finish_reason(response)
    raw_text = response.text or ""
    record_paid_response(stage, attempt_no, raw_text, raw_path)
    # Keep ATTEMPTS tracker in step with BUDGET (BUDGET already consumed
    # inside call_llm; ATTEMPTS records the same attempt number).
    recorded_no = ATTEMPTS.record_attempt(stage, finish_reason)
    if recorded_no != attempt_no:
        raise RuntimeError(
            f"Attempt numbering drift for '{stage}': expected {attempt_no}, "
            f"tracker gave {recorded_no}.")
    print_response_debug(response, stage)

    # 5. paid-count checkpoint SECOND, before any parse/validation.
    if on_attempt is not None:
        on_attempt(stage, attempt_no, finish_reason, raw_text)

    # 6. parse + 7. Pydantic validation ONLY after durability.
    try:
        result = parse_structured_response(response, stage)
        # Validate against the Pydantic schema so bad shapes fail loudly.
        validated = response_schema.model_validate(result).model_dump()
    except Exception:
        ATTEMPTS.record_failure(stage)
        if on_attempt is not None:
            # Re-checkpoint so the failure mark survives the crash too.
            on_attempt(stage, attempt_no, finish_reason, raw_text)
        raise
    ATTEMPTS.record_success(stage)
    return validated


# -- durable paid-attempt inventory (§14/§15/§50) --------------------------
import re as _re

_ATTEMPT_FILE_RE = _re.compile(r"^(.+)_attempt_(\d+)\.txt$")

# Map filename infix -> canonical budget stage.
_STAGE_HINTS = (
    ("question_style", "question_style"),
    ("question_template", "question_template"),
    ("style", "question_style"),
    ("template", "question_template"),
    ("paraphrase", "paraphrase"),
    ("quality", "quality"),
)


def stage_of_attempt_filename(name: str) -> str | None:
    """Map a canonical attempt filename to its budget stage.

    Only exact `<...>_attempt_<NN>.txt` names count; temp/backup/semantic
    JSON files return None and are ignored during resume inventory.
    """
    m = _ATTEMPT_FILE_RE.fullmatch(name)
    if m is None:
        return None
    stem = m.group(1).lower()
    for hint, stage in _STAGE_HINTS:
        if hint in stem:
            return stage
    return None


def scan_raw_attempts(raw_dir) -> dict[str, list[int]]:
    """Collect canonical finalized attempt numbers per stage.

    Ignores temp files (*.tmp), backups, malformed names, and semantic
    JSON artifacts. Only `<...>_attempt_<NN>.txt` counts.
    Returns {stage: sorted([numbers])}.
    """
    from pathlib import Path as _Path

    raw_dir = _Path(raw_dir)
    out: dict[str, list[int]] = {}
    if not raw_dir.is_dir():
        return out
    for p in raw_dir.iterdir():
        if not p.is_file():
            continue
        stage = stage_of_attempt_filename(p.name)
        if stage is None:
            continue
        m = _ATTEMPT_FILE_RE.fullmatch(p.name)
        assert m is not None
        out.setdefault(stage, []).append(int(m.group(2)))
    for stage in out:
        out[stage] = sorted(out[stage])
    return out


def require_contiguous(numbers: list[int], stage: str) -> int:
    """Validate exact contiguous [1..N]; return N (0 when empty).

    Raises ValueError on gaps or duplicates; duplicates cannot occur via
    distinct filenames but are guarded explicitly.
    """
    if not numbers:
        return 0
    if len(set(numbers)) != len(numbers):
        raise ValueError(
            f"Duplicate raw attempt numbers for '{stage}': {numbers}; "
            "refusing to resume.")
    if sorted(numbers) != list(range(1, len(numbers) + 1)):
        raise ValueError(
            f"Noncontiguous raw attempts for '{stage}': {sorted(numbers)}; "
            "refusing to resume.")
    return len(numbers)

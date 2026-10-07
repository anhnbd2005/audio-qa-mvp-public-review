import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yaml"
ENV_PATH = ROOT / ".env"

# config.yaml key -> environment variable read from .env
_LLM_ENV = {
    "api_key": "OPENAI_API_KEY",
    "base_url": "OPENAI_BASE_URL",
    "model": "OPENAI_MODEL",
    # OpenAI-compatible endpoint variables; they win over the ones above.
    "compat_api_key": "OPENAI_COMPAT_API_KEY",
    "compat_base_url": "OPENAI_COMPAT_BASE_URL",
    "compat_model": "OPENAI_COMPAT_MODEL",
}


def first_nonempty(*values) -> str | None:
    """First configured value that is not None/blank."""
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def resolve_llm_model(config: dict | None = None) -> str:
    """Model priority: OPENAI_COMPAT_MODEL > OPENAI_MODEL > config.yaml."""
    cfg = CONFIG if config is None else config
    llm = cfg.get("llm") or {}
    return first_nonempty(llm.get("compat_model"), llm.get("model")) or ""


def resolve_llm_base_url(config: dict | None = None) -> str:
    """Endpoint priority: OPENAI_COMPAT_BASE_URL > OPENAI_BASE_URL > config."""
    cfg = CONFIG if config is None else config
    llm = cfg.get("llm") or {}
    return first_nonempty(llm.get("compat_base_url"),
                          llm.get("base_url")) or ""


def resolve_llm_api_key(config: dict | None = None) -> str:
    """Key priority: OPENAI_COMPAT_API_KEY > OPENAI_API_KEY > "" (none).

    Never invents a credential; an empty string means the endpoint does
    not authenticate (the local OpenAI-compatible proxy does not).
    """
    cfg = CONFIG if config is None else config
    llm = cfg.get("llm") or {}
    return first_nonempty(llm.get("compat_api_key"),
                          llm.get("api_key")) or ""


def load_config():
    load_dotenv(ENV_PATH, override=False)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # LLM endpoint settings: .env wins over the config.yaml defaults,
    # and the API key only ever lives in .env.
    llm = dict(config.get("llm") or {})
    for key, env_name in _LLM_ENV.items():
        value = os.environ.get(env_name)
        if value:
            llm[key] = value
    config["llm"] = llm
    return config


CONFIG = load_config()

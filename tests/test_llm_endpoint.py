"""LLM endpoint resolution: local OpenAI-compatible proxy, never the cloud.

Pins the endpoint contract the dataset profiler depends on:

1. model priority  OPENAI_COMPAT_MODEL > OPENAI_MODEL > config.yaml
2. base_url is the project's local OpenAI-compatible endpoint
3. no request can ever fall back to api.openai.com (and no key is required)
4. the dataset profiler builds its request through the shared client
5. real profiling spends exactly one LLM request
6. mock profiling spends none

Everything here is offline: no socket is opened and no provider is called.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.common.config as config_module
import src.autonomous_qa.core.dataset_profile as dp
from src.autonomous_qa.authoring import llm_client
from src.common.config import (
    ROOT,
    load_config,
    resolve_llm_api_key,
    resolve_llm_base_url,
    resolve_llm_model,
)

CARD_PATH = ROOT / "data_sources" / "vimd" / "dataset_card.md"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "dataset_profile_vimd.json"


@pytest.fixture()
def valid_payload_text() -> str:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return dp.canonical_json(payload)


class RecordingClient:
    """Shared-client stand-in: records every generate_content() request."""

    def __init__(self, text: str):
        self.calls: list[dict] = []
        self._text = text
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, model=None, contents=None, config=None):
        self.calls.append({"model": model, "contents": contents,
                           "config": config})
        return SimpleNamespace(
            text=self._text,
            parsed=None,
            candidates=[SimpleNamespace(finish_reason="STOP")],
            usage_metadata=None,
        )


# --- 1. model precedence ----------------------------------------------------


def test_1_compat_model_wins_over_openai_model_and_config(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "proxy/model-a")
    monkeypatch.setenv("OPENAI_MODEL", "openai-model-b")
    cfg = load_config()
    assert resolve_llm_model(cfg) == "proxy/model-a"


def test_2_openai_model_is_second_priority(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "")
    monkeypatch.setenv("OPENAI_MODEL", "openai-model-b")
    cfg = load_config()
    assert resolve_llm_model(cfg) == "openai-model-b"


def test_3_config_value_is_last_fallback(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    cfg = load_config()
    assert resolve_llm_model(cfg) == cfg["llm"]["model"]
    assert resolve_llm_model(cfg) == "ag/gemini-3.6-flash-low"


# --- 2. base_url is the local compatible endpoint ---------------------------


def test_4_base_url_resolves_to_local_proxy_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:20128/v1")
    cfg = load_config()
    assert resolve_llm_base_url(cfg) == "http://localhost:20128/v1"


def test_5_shipped_config_targets_local_proxy():
    cfg = load_config()
    base_url = resolve_llm_base_url(cfg)
    assert base_url == "http://localhost:20128/v1"
    assert "api.openai.com" not in base_url
    assert resolve_llm_model(cfg) != "gpt-4o-mini"


# --- 3. no fallback to api.openai.com ---------------------------------------


def test_6_client_refuses_hosted_openai_api(monkeypatch):
    monkeypatch.setattr(config_module, "CONFIG", {
        "llm": {"base_url": "https://api.openai.com/v1",
                "model": "gpt-4o-mini", "timeout": 5, "max_retries": 0},
    })
    with pytest.raises(RuntimeError, match="hosted OpenAI API"):
        llm_client.create_llm_client()


def test_7_client_refuses_missing_endpoint(monkeypatch):
    monkeypatch.setattr(config_module, "CONFIG", {
        "llm": {"base_url": "", "model": "ag/gemini-3.6-flash-low"},
    })
    with pytest.raises(RuntimeError, match="No OpenAI-compatible endpoint"):
        llm_client.create_llm_client()


def test_8_client_targets_local_proxy_with_placeholder_key():
    """No credential configured: the SDK still gets a harmless placeholder."""
    client = llm_client.create_llm_client()

    base_url = str(client._api.base_url)
    assert "localhost:20128" in base_url
    assert "api.openai.com" not in base_url
    assert client.model == resolve_llm_model()

    configured_key = resolve_llm_api_key()
    expected = configured_key or llm_client.LOCAL_PROXY_PLACEHOLDER_API_KEY
    assert client._api.api_key == expected
    assert expected  # never empty, never an OpenAI credential requirement


# --- 4 + 5. the profiler uses the shared client, exactly once ---------------


def test_9_real_profile_uses_shared_client_exactly_once(
        tmp_path, monkeypatch, capsys, valid_payload_text):
    client = RecordingClient(valid_payload_text)
    created: list[int] = []

    def fake_create():
        created.append(1)
        return client

    monkeypatch.setattr(llm_client, "create_llm_client", fake_create)
    monkeypatch.setattr(dp, "require_local_proxy", lambda base_url: None)
    capsys.readouterr()

    result = dp.run_profile(dataset="vimd", source_card_path=CARD_PATH,
                            real_llm=True, run_id="real-single-call",
                            out_root=tmp_path)
    out = capsys.readouterr().out

    assert result["ok"] is True, result.get("report")
    assert len(created) == 1
    assert len(client.calls) == 1
    assert client.calls[0]["model"] == resolve_llm_model()
    assert "20128" in resolve_llm_base_url()
    assert result["run_meta"]["llm_calls"] == 1
    assert result["run_meta"]["parsing_status"] == "parsed"
    assert result["run_meta"]["provider"] == "openai_compatible"
    assert "localhost:20128" in str(result["run_meta"]["base_url"])

    # Banner is printed BEFORE the request.
    assert "LLM_PROVIDER=openai_compatible" in out
    assert f"LLM_BASE_URL={resolve_llm_base_url()}" in out
    assert f"LLM_MODEL={resolve_llm_model()}" in out
    assert "REMOTE_OPENAI_API=NO" in out
    assert out.index("LLM_MODEL=") < out.index("[dataset_profile] finish_reason=")


def test_10_mock_profile_makes_zero_llm_calls(
        tmp_path, monkeypatch, capsys):
    def boom():
        raise AssertionError("mock mode must not build an LLM client")

    monkeypatch.setattr(llm_client, "create_llm_client", boom)
    monkeypatch.setattr(dp, "require_local_proxy",
                        lambda base_url: AssertionError("no probe in mock"))
    capsys.readouterr()

    result = dp.run_profile(dataset="vimd", source_card_path=CARD_PATH,
                            real_llm=False, run_id="mock-zero",
                            out_root=tmp_path)
    out = capsys.readouterr().out

    assert result["ok"] is True
    assert result["run_meta"]["llm_calls"] == 0
    assert result["run_meta"]["provider"] is None
    assert result["run_meta"]["base_url"] is None
    assert "LLM_PROVIDER=" not in out


# --- LOCAL_PROXY_UNAVAILABLE -------------------------------------------------


def test_11_dead_proxy_stops_before_any_request(tmp_path, monkeypatch):
    def refuse(base_url):
        raise dp.LocalProxyUnavailable(
            f"LOCAL_PROXY_UNAVAILABLE: {base_url} is not accepting "
            "connections. The profiling request was never sent.")

    monkeypatch.setattr(dp, "require_local_proxy", refuse)

    def boom():
        raise AssertionError("no client may be built when the proxy is down")

    monkeypatch.setattr(llm_client, "create_llm_client", boom)

    result = dp.run_profile(dataset="vimd", source_card_path=CARD_PATH,
                            real_llm=True, run_id="proxy-down",
                            out_root=tmp_path)

    assert result["ok"] is False
    assert result["exit_code"] == 2
    assert "LOCAL_PROXY_UNAVAILABLE" in result["failure"]
    assert result["run_meta"]["llm_calls"] == 0
    assert result["run_meta"]["parsing_status"] == "proxy_unavailable"
    assert not (Path(result["run_dir"]) / "raw_response.txt").exists()


def test_12_loopback_probe_reports_local_proxy_unavailable(monkeypatch):
    def refuse(host, port, timeout=5.0):
        raise OSError("[WinError 10061] No connection could be made")

    monkeypatch.setattr(dp, "_tcp_probe", refuse)
    with pytest.raises(dp.LocalProxyUnavailable) as excinfo:
        dp.require_local_proxy("http://localhost:20128/v1")
    assert "LOCAL_PROXY_UNAVAILABLE" in str(excinfo.value)


def test_13_remote_endpoint_is_never_probed(monkeypatch):
    def refuse(host, port, timeout=5.0):
        raise AssertionError("no preflight against a non-loopback host")

    monkeypatch.setattr(dp, "_tcp_probe", refuse)
    dp.require_local_proxy("http://gpu-box.internal:8000/v1")

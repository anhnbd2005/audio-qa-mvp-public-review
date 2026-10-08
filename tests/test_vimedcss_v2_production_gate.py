"""ViMedCSS V2 production preflight gate regression tests.

The gate must authorize a new ViMedCSS plan ONLY against a matching,
PREFLIGHT_PASS, zero-blocking, same-dataset, same-registry preflight artifact.
No LLM or network calls are permitted.
"""

from __future__ import annotations

import json
import socket
import urllib.request
from pathlib import Path

import pytest
import yaml

from src.autonomous_qa.language import language_preflight as lp
from src.autonomous_qa.production import production_qa as pq
from src.autonomous_qa.production.production_qa import (
    ProductionGenerationConfig,
    ProductionQAError,
    enforce_language_preflight_gate,
)
from src.common.config import ROOT

PROMOTED_REGISTRY_HASH = (
    "b35320756d541f925a16d4cd0758e86104a9d63c4f172b0e70e6bfdddb3680a1"
)


def _base_config(**over) -> ProductionGenerationConfig:
    raw = yaml.safe_load(
        (ROOT / "configs" / "qa_generation_production.yaml").read_text(encoding="utf-8")
    )
    raw.update(over)
    return ProductionGenerationConfig.model_validate(raw)


def _vimedcss_fingerprint() -> str:
    accepted = sorted(
        lp.get_dataset_accepted_types("vimedcss"),
        key=lambda row: row.dataset_type_id,
    )
    fingerprint, _ = lp.compute_contract_fingerprint(
        mode="dataset", accepted_types=accepted
    )
    return fingerprint


def _artifact(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "artifact" / "audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _good_payload() -> dict:
    return {
        "result": "PREFLIGHT_PASS",
        "contract_fingerprint": _vimedcss_fingerprint(),
        "dataset": "vimedcss",
        "blocking_issue_count": 0,
        "accepted_type_count": 3,
        "language_registry_hash": PROMOTED_REGISTRY_HASH,
    }


# ---------------------------------------------------------------------------
# 1. current ViMedCSS V2 preflight PASS / 0
# ---------------------------------------------------------------------------


def test_current_vimedcss_v2_preflight_passes_with_zero_blockers():
    result = lp.run_preflight(
        mode="dataset",
        dataset="vimedcss",
        accepted_types=lp.get_dataset_accepted_types("vimedcss"),
        write_outputs=False,
    )
    audit = result["audit"]
    assert audit["result"] == "PREFLIGHT_PASS"
    assert audit["blocking_issue_count"] == 0
    assert audit["accepted_type_count"] == 3
    assert audit["language_registry_hash"] == PROMOTED_REGISTRY_HASH


# ---------------------------------------------------------------------------
# 2. matching artifact -> allowed
# ---------------------------------------------------------------------------


def test_matching_artifact_allows_production(tmp_path: Path):
    path = _artifact(tmp_path, _good_payload())
    decision = enforce_language_preflight_gate(
        dataset="vimedcss", config=_base_config(language_preflight_artifact=str(path))
    )
    assert decision["allowed"] is True
    assert decision["status"] == "PREFLIGHT_PASS"


# ---------------------------------------------------------------------------
# 3. missing artifact -> blocked
# ---------------------------------------------------------------------------


def test_missing_artifact_blocks(tmp_path: Path):
    missing = tmp_path / "does_not_exist" / "audit.json"
    with pytest.raises(ProductionQAError) as exc:
        enforce_language_preflight_gate(
            dataset="vimedcss",
            config=_base_config(language_preflight_artifact=str(missing)),
        )
    assert exc.value.code == "PREFLIGHT_REQUIRED"


# ---------------------------------------------------------------------------
# 4. stale fingerprint / registry -> blocked
# ---------------------------------------------------------------------------


def test_stale_contract_fingerprint_blocks(tmp_path: Path):
    payload = _good_payload()
    payload["contract_fingerprint"] = "stale-fingerprint"
    path = _artifact(tmp_path, payload)
    with pytest.raises(ProductionQAError) as exc:
        enforce_language_preflight_gate(
            dataset="vimedcss",
            config=_base_config(language_preflight_artifact=str(path)),
        )
    assert exc.value.code == "PREFLIGHT_STALE"


def test_stale_registry_identity_blocks(tmp_path: Path):
    payload = _good_payload()
    payload["language_registry_hash"] = "0" * 64
    path = _artifact(tmp_path, payload)
    with pytest.raises(ProductionQAError) as exc:
        enforce_language_preflight_gate(
            dataset="vimedcss",
            config=_base_config(language_preflight_artifact=str(path)),
        )
    assert exc.value.code == "PREFLIGHT_REGISTRY_MISMATCH"


# ---------------------------------------------------------------------------
# 5. changed accepted semantic set / wrong dataset -> blocked
# ---------------------------------------------------------------------------


def test_changed_accepted_semantic_set_blocks(tmp_path: Path):
    payload = _good_payload()
    payload["accepted_type_count"] = 2
    path = _artifact(tmp_path, payload)
    with pytest.raises(ProductionQAError) as exc:
        enforce_language_preflight_gate(
            dataset="vimedcss",
            config=_base_config(language_preflight_artifact=str(path)),
        )
    assert exc.value.code == "PREFLIGHT_ACCEPTED_SET_MISMATCH"


def test_wrong_dataset_artifact_blocks(tmp_path: Path):
    payload = _good_payload()
    payload["dataset"] = "vimd"
    path = _artifact(tmp_path, payload)
    with pytest.raises(ProductionQAError) as exc:
        enforce_language_preflight_gate(
            dataset="vimedcss",
            config=_base_config(language_preflight_artifact=str(path)),
        )
    assert exc.value.code == "PREFLIGHT_DATASET_MISMATCH"


# ---------------------------------------------------------------------------
# 6. failed preflight artifact -> blocked
# ---------------------------------------------------------------------------


def test_failed_preflight_artifact_blocks(tmp_path: Path):
    payload = _good_payload()
    payload["result"] = "LANGUAGE_CONTRACT_FAIL"
    path = _artifact(tmp_path, payload)
    with pytest.raises(ProductionQAError) as exc:
        enforce_language_preflight_gate(
            dataset="vimedcss",
            config=_base_config(language_preflight_artifact=str(path)),
        )
    assert exc.value.code == "PREFLIGHT_FAILED"


def test_blocking_issues_artifact_blocks(tmp_path: Path):
    payload = _good_payload()
    payload["blocking_issue_count"] = 3
    path = _artifact(tmp_path, payload)
    with pytest.raises(ProductionQAError) as exc:
        enforce_language_preflight_gate(
            dataset="vimedcss",
            config=_base_config(language_preflight_artifact=str(path)),
        )
    assert exc.value.code == "PREFLIGHT_BLOCKING_ISSUES"


# ---------------------------------------------------------------------------
# 7. ViMD gate behavior preserved
# ---------------------------------------------------------------------------


def test_vimd_gate_behavior_preserved():
    decision = enforce_language_preflight_gate(dataset="vimd", config=_base_config())
    assert decision["allowed"] is True
    assert decision["status"] == "PREFLIGHT_PASS"


def test_vimd_explicit_missing_path_still_blocks():
    cfg = _base_config(language_preflight_artifact="custom/nonexistent/path/audit.json")
    with pytest.raises(ProductionQAError):
        enforce_language_preflight_gate(dataset="vimd", config=cfg)


def test_vimd_artifact_not_leaked_into_vimedcss_resolution(tmp_path: Path):
    # The shipped config names vimd; the ViMedCSS target must resolve to vimedcss.
    cfg = _base_config()
    target = pq._resolve_preflight_artifact_target(cfg, "vimedcss")
    assert "language_preflight/vimedcss/" in target.replace("\\", "/")
    assert "language_preflight/vimd/" not in target.replace("\\", "/")


# ---------------------------------------------------------------------------
# 8. no LLM or network calls
# ---------------------------------------------------------------------------


def test_gate_makes_no_llm_or_network_calls(tmp_path: Path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("forbidden side effect")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    from src.autonomous_qa.authoring import llm_client as llm

    monkeypatch.setattr(llm, "create_llm_client", forbidden)

    path = _artifact(tmp_path, _good_payload())
    decision = enforce_language_preflight_gate(
        dataset="vimedcss",
        config=_base_config(language_preflight_artifact=str(path)),
    )
    assert decision["allowed"] is True

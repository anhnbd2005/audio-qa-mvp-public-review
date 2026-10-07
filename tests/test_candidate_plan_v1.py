"""Candidate production plan V1 tests (K-proportional, zero-LLM, no audio).

Runs the exact production path on a synthetic fixture with a per-type budget
and exercises the candidate-plan audit module. Hermetic: no network, no
frozen-artifact writes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from tests.regression import production_plan_audit as ppa
from src.autonomous_qa.production import production_qa as pq

ROOT = Path(__file__).resolve().parents[1]
BUDGET_PER_TYPE = 8
TOTAL = 88
EXPECTED_AUDIO_REFS = 128  # 3*8*1 + 2*8*2 + 3*8*2 + 3*8*1


def _row(i, province, region, text, speaker):
    return {
        "region": region,
        "province_code": 10 + (int(province[1:]) if province[1:].isdigit() else i),
        "province_name": province,
        "filename": f"clip_{i:04d}.wav",
        "text": text,
        "speakerID": speaker,
        "gender": i % 2,
        "_shard": "train-00000-of-00001.parquet",
        "_row_group": 0,
        "_row_index": i,
    }


@pytest.fixture(scope="module")
def rows():
    groups = [
        ("P1", "North", 8),
        ("P2", "North", 6),
        ("P3", "Central", 6),
        ("P4", "Central", 5),
        ("P5", "South", 5),
    ]
    out = []
    i = 0
    for province, region, size in groups:
        for _ in range(size):
            out.append(
                _row(i, province, region, f"cau so {i % 12} mau", f"spk_{i % 7}")
            )
            i += 1
    return out


def _candidate_config(run_root, *, over=None):
    raw = yaml.safe_load(
        (ROOT / "configs" / "qa_generation_candidate_k22000.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["per_type_budget"] = {
        "vimd-v2-s1-001": BUDGET_PER_TYPE,
        "vimd-v2-s1-002": BUDGET_PER_TYPE,
        "vimd-v2-s1-003": BUDGET_PER_TYPE,
        "vimd-v2-s1-005": BUDGET_PER_TYPE,
        "vimd-v2-s1-006": BUDGET_PER_TYPE,
        "vimd-v2.1-s2-002": BUDGET_PER_TYPE,
        "vimd-v2.1-s2-003": BUDGET_PER_TYPE,
        "vimd-v2.1-s2-004": BUDGET_PER_TYPE,
        "vimd-v2.2-s2-001": BUDGET_PER_TYPE,
        "vimd-v2.2-s2-002": BUDGET_PER_TYPE,
        "vimd-v2.2-s2-003": BUDGET_PER_TYPE,
    }
    if over:
        raw.update(over)
    path = Path(run_root) / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


def _run(tmp_path, rows, run_id="cand", over=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    meta = tmp_path / "train.jsonl"
    meta.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    cfg = _candidate_config(tmp_path, over=over)
    summary = pq.run_production_qa(
        dataset="vimd",
        config_path=cfg,
        mode="full",
        output_root=tmp_path / "out",
        run_id=run_id,
        metadata_path=meta,
        expected_rows=len(rows),
    )
    audit = ppa.audit_candidate_plan(
        Path(summary["run_dir"]),
        meta,
        dataset="vimd",
        revision="testrev",
        run_purpose="CANDIDATE_PLAN_TEST",
        reference_rows=len(rows),
    )
    return summary, audit


# ---------------------------------------------------------------- budget
def test_candidate_budget_is_exact_per_type(tmp_path, rows):
    summary, _ = _run(tmp_path, rows)
    budget = json.loads(
        (Path(summary["run_dir"]) / "budget_resolution.json").read_text(
            encoding="utf-8"
        )
    )
    assert budget["status"] == "PER_TYPE_BUDGET"
    assert budget["strict_budget"] is True
    assert budget["resolved_total"] == TOTAL
    assert set(budget["per_type"].values()) == {BUDGET_PER_TYPE}


def test_candidate_planned_total_and_no_shortfall(tmp_path, rows):
    summary, audit = _run(tmp_path, rows)
    assert summary["qa_generated"] == TOTAL
    assert summary["plan_records"] == TOTAL
    sd = audit["semantic_distribution"]
    assert sd["planned_total"] == TOTAL
    assert sd["requested_total"] == TOTAL
    assert sd["semantic_duplicates"] == 0
    assert sd["semantic_instances_unique"] is True


def test_all_eleven_types_planned(tmp_path, rows):
    _, audit = _run(tmp_path, rows)
    by_type = audit["semantic_distribution"]["by_type"]
    assert len(by_type) == 11
    assert all(v["planned"] == BUDGET_PER_TYPE for v in by_type.values())


def test_validation_all_passed(tmp_path, rows):
    _, audit = _run(tmp_path, rows)
    assert audit["semantic_distribution"]["validation"]["all_passed"] is True
    assert audit["semantic_distribution"]["validation"]["records"] == TOTAL


# ------------------------------------------------------- boolean/position
def test_boolean_balanced(tmp_path, rows):
    _, audit = _run(tmp_path, rows)
    boolean = audit["value_distribution"]["boolean_distribution"]
    assert len(boolean) == 5
    for payload in boolean.values():
        assert payload["positive"] == payload["negative"]
        assert payload["positive_ratio"] == 0.5


def test_selection_position_and_xor(tmp_path, rows):
    _, audit = _run(tmp_path, rows)
    position = audit["value_distribution"]["selection_position"]
    assert len(position) == 3
    for payload in position.values():
        assert payload["A"] + payload["B"] == BUDGET_PER_TYPE
        assert not payload["severe_bias"]


def test_region_direct_gold_is_diverse(tmp_path, rows):
    _, audit = _run(tmp_path, rows)
    dist = audit["value_distribution"]["region_distribution"]["vimd-v2-s1-002"]
    assert len(dist) == 3  # North / Central / South all present


# ------------------------------------------------------------- reuse
def test_source_reuse_audit_statistics(tmp_path, rows):
    _, audit = _run(tmp_path, rows)
    reuse = audit["source_reuse"]
    assert reuse["total_audio_references"] == EXPECTED_AUDIO_REFS
    assert reuse["unique_source_rows_used"] <= len(rows)
    stats = reuse["reuse_stats_among_used_rows"]
    for key in ("min", "mean", "median", "p75", "p90", "p95", "p99", "max"):
        assert key in stats
    assert sum(reuse["reuse_histogram"].values()) == reuse["unique_source_rows_used"]
    assert len(reuse["by_type"]) == 11
    assert set(reuse["by_field_family"]) == {"text", "region", "province"}


def test_province_group_quantiles(tmp_path, rows):
    _, audit = _run(tmp_path, rows)
    size = audit["value_distribution"]["province_group_size"]
    assert size["count"] == 5
    assert size["min"] <= size["median"] <= size["max"]
    assert len(size["smallest_10"]) <= 10
    assert len(size["largest_10"]) <= 10


def test_audio_dependency_manifest_dedup(tmp_path, rows):
    summary, audit = _run(tmp_path, rows)
    manifest = json.loads(
        (Path(summary["run_dir"]) / "audio_dependency_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    ids = [item["audio_id"] for item in manifest["audio"]]
    assert len(ids) == len(set(ids))
    assert manifest["unique_audio"] == audit["source_reuse"]["unique_source_rows_used"]
    assert manifest["total_references"] == EXPECTED_AUDIO_REFS
    assert all(item["required_by_count"] >= 1 for item in manifest["audio"])


# ----------------------------------------------------------- language
def test_one_wording_per_semantic_instance(tmp_path, rows):
    summary, audit = _run(tmp_path, rows)
    plan = [
        json.loads(line)
        for line in (Path(summary["run_dir"]) / "generation_plan.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    sems = [r["semantic_instance_id"] for r in plan]
    assert len(sems) == len(set(sems)) == TOTAL
    usage = audit["language_usage"]
    assert usage["one_entry_per_semantic_instance"] is True
    for payload in usage["by_type"].values():
        assert sum(payload["usage"].values()) == BUDGET_PER_TYPE
        assert (
            payload["distinct_selected_entries"] <= payload["compatible_active_entries"]
        )


def test_no_wording_multiplication(tmp_path, rows):
    summary, _ = _run(tmp_path, rows)
    assert summary["qa_generated"] == TOTAL  # not TOTAL * 56


def test_wording_balance_not_enforced(tmp_path, rows):
    _, audit = _run(tmp_path, rows)
    usage = audit["language_usage"]
    assert usage["wording_balance_enforced"] is False
    assert usage["global_canonical_qa"] + usage["global_paraphrase_qa"] == TOTAL


# --------------------------------------------------------- determinism
def test_plan_hash_determinism(tmp_path, rows):
    a, _ = _run(tmp_path / "a", rows, run_id="r")
    b, _ = _run(tmp_path / "b", rows, run_id="r")
    sha_a = (Path(a["run_dir"]) / "generation_plan.sha256").read_text().strip()
    sha_b = (Path(b["run_dir"]) / "generation_plan.sha256").read_text().strip()
    assert sha_a == sha_b
    plan_path = Path(a["run_dir"]) / "generation_plan.jsonl"
    assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == sha_a


def test_run_purpose_marker_written(tmp_path, rows):
    summary, _ = _run(tmp_path, rows)
    marker = json.loads(
        (Path(summary["run_dir"]) / "run_purpose.json").read_text(encoding="utf-8")
    )
    assert marker["run_purpose"] == "CANDIDATE_PLAN_TEST"
    assert marker["audio_materialized"] is False
    assert marker["llm_calls"] == 0


# ------------------------------------------------------------ safety
def test_no_audio_materialization(tmp_path, rows):
    summary, _ = _run(tmp_path, rows)
    run_dir = Path(summary["run_dir"])
    assert not list(run_dir.rglob("*.wav"))
    assert not list(run_dir.rglob("*.flac"))
    resolution = json.loads(
        (run_dir / "audio_resolution_audit.json").read_text(encoding="utf-8")
    )
    assert resolution["auto_download"] is False


def test_zero_llm(tmp_path, rows, monkeypatch):
    called = {"n": 0}

    def _boom(*a, **k):
        called["n"] += 1
        raise AssertionError("LLM CALLED")

    import src.autonomous_qa.authoring.llm_client as llm

    monkeypatch.setattr(llm, "create_llm_client", _boom, raising=False)
    monkeypatch.setattr(llm, "call_llm", _boom, raising=False)
    _run(tmp_path, rows)
    assert called["n"] == 0


def test_audit_module_does_not_import_llm():
    source = (ROOT / "tests" / "regression" / "production_plan_audit.py").read_text(encoding="utf-8")
    assert "llm_client" not in source
    assert "openai" not in source.lower()

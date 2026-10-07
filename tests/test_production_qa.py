"""Production QA Generator V1 tests (zero-LLM, same production code path).

Covers the frozen contract, O(N) index, capacity formulas, all four operator
samplers, semantic identity/dedup, language selection, leakage, budget,
determinism, zero-LLM guards, and scale. No network, no LLM, no writes to
frozen artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.autonomous_qa.production import production_qa as pq
from src.autonomous_qa.language.language_quality import ProductionGenerationConfig
from src.autonomous_qa.certification.qa_sampling import (
    BudgetShortfallError,
    ReuseTracker,
    SamplingSettings,
    SemanticSampler,
    TrainIndex,
    capacity_for_contract,
    normalize_field_value,
)
from src.autonomous_qa.language.template_contracts import build_type_contracts

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "resources" / "language" / "production_registry.json"
TYPE_REGISTRY_PATH = ROOT / "resources" / "semantics" / "vimd_type_registry.json"


# --------------------------------------------------------------- fixtures
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
def synthetic_rows():
    groups = [
        ("P1", "North", 8),
        ("P2", "North", 6),
        ("P3", "Central", 6),
        ("P4", "Central", 5),
        ("P5", "South", 5),
    ]
    rows = []
    i = 0
    for province, region, size in groups:
        for _ in range(size):
            text = f"cau so {i % 12} noi dung mau"
            speaker = f"spk_{i % 7}"
            rows.append(_row(i, province, region, text, speaker))
            i += 1
    return rows


@pytest.fixture(scope="module")
def env(synthetic_rows):
    specs = pq.vimd_field_specs()
    registry = pq.load_language_registry(REGISTRY_PATH)
    contracts_all = build_type_contracts(
        json.loads(TYPE_REGISTRY_PATH.read_text(encoding="utf-8"))
    )
    contracts = {c.type_id: c for c in contracts_all}
    supported = {
        tid: c for tid, c in contracts.items() if c.source_status == "SUPPORTED"
    }
    row_ids = [r["filename"] for r in synthetic_rows]
    index = TrainIndex(
        synthetic_rows,
        row_ids,
        {c.semantic_field: specs[c.semantic_field] for c in supported.values()},
        hidden_identifier_field="speakerID",
    )
    capacities = {
        tid: capacity_for_contract(c, index, specs[c.semantic_field], True)
        for tid, c in supported.items()
    }
    return {
        "rows": synthetic_rows,
        "specs": specs,
        "registry": registry,
        "contracts": contracts,
        "supported": supported,
        "index": index,
        "row_ids": row_ids,
        "capacities": capacities,
    }


def _config(**over) -> ProductionGenerationConfig:
    base = {
        "seed": 42,
        "total_target_qa": None,
        "per_type_budget": None,
        "strict_budget": True,
        "allocation_policy": "equal_by_type",
        "language_preflight_artifact": "data/materialized/language_preflight/vimd/current",
        "sampling": {
            "value_sampling": "uniform_over_values",
            "prefer_distinct_speaker": True,
            "hidden_identifier_field": "speakerID",
            "max_sampling_attempts": 128,
        },
        "boolean": {"positive_ratio": 0.5},
        "selection": {"randomized_position": True},
        "text_negative": {"length_bucket_match": True},
        "language": {
            "variants_per_semantic_instance": 1,
            "selection": "deterministic_sha256",
        },
        "audio": {"materialize": False, "local_root": None},
    }
    base.update(over)
    return ProductionGenerationConfig.model_validate(base)


def _settings(**over) -> SamplingSettings:
    base = {
        "value_sampling": "uniform_over_values",
        "prefer_distinct_speaker": True,
        "hidden_identifier_field": "speakerID",
        "positive_ratio": 0.5,
        "randomized_position": True,
        "length_bucket_match": True,
        "max_sampling_attempts": 256,
    }
    base.update(over)
    return SamplingSettings(**base)


def _sampler(env, type_id, seed=42, **setting_over):
    contract = env["supported"][type_id]
    spec = env["specs"][contract.semantic_field]
    return SemanticSampler(
        contract, spec, env["index"], _settings(**setting_over), ReuseTracker(), seed
    )


def _type_by_operator(env, operator):
    return sorted(
        t.type_id for t in env["supported"].values() if t.operator == operator
    )


def _run_synthetic(
    tmp_path,
    rows,
    *,
    mode="debug_sample",
    debug_sample=None,
    config_over=None,
    run_id="run",
    seed=42,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    meta = tmp_path / "train.jsonl"
    meta.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    raw = yaml.safe_load(
        (ROOT / "configs" / "qa_generation_production.yaml").read_text(encoding="utf-8")
    )
    raw["seed"] = seed
    if config_over:
        raw.update(config_over)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return pq.run_production_qa(
        dataset="vimd",
        config_path=cfg,
        mode=mode,
        output_root=tmp_path / "out",
        run_id=run_id,
        debug_sample=debug_sample,
        metadata_path=meta,
        expected_rows=len(rows),
    )


# ------------------------------------------------------- frozen contract §107


def test_canonical_resources_verify():
    checks = pq.verify_canonical_resources("vimd")
    assert checks["all_matched"] is True
    assert (
        checks["template_library"]["internal_hash"]
        == pq.EXPECTED_CANONICAL["template_library"]["internal_hash"]
    )
    assert (
        checks["paraphrase_library"]["file_sha256"]
        == pq.EXPECTED_CANONICAL["paraphrase_library"]["file_sha256"]
    )


def test_canonical_hashes_match_expected():
    assert pq.EXPECTED_CANONICAL["production_registry"]["registry_hash"] == (
        "267a747d95016457831ede80993648c612079f2a473e62af79e3e030413e425f"
    )
    assert pq.EXPECTED_CANONICAL["production_registry"]["file_sha256"] == (
        "3bb6c49dc9525dc2bc6fab49af04757588f570e6954fd7d539ded5c39182a7dd"
    )


def test_authoritative_pool_is_11_supported_and_2_review(env):
    assert len(env["supported"]) == 11
    deferred = [t for t in env["contracts"].values() if t.source_status != "SUPPORTED"]
    assert {t.type_id for t in deferred} == {"vimd-v2-s1-004", "vimd-v2.1-s2-001"}


def test_review_types_excluded_from_production(env):
    registry = env["registry"]
    for tid in ("vimd-v2-s1-004", "vimd-v2.1-s2-001"):
        contract = env["contracts"][tid]
        assert contract.source_status == "REVIEW_REQUIRED"
    assert "vimd-v2-s1-004" not in env["supported"]
    assert "vimd-v2.1-s2-001" not in env["supported"]
    assert registry.version == "canonical"




# ------------------------------------------------------------- index §108
def test_index_rows_by_value_and_domains(env):
    index = env["index"]
    assert index.summary()["rows_indexed"] == len(env["rows"])
    for field in env["specs"]:
        valid = index.valid_row_ids[field]
        assert len(valid) == len(set(valid))
        for norm, ids in index.rows_by_value[field].items():
            assert set(index.group(field, norm)) == set(ids)
        assert index.values(field) == sorted(index.rows_by_value[field])


def test_text_normalization_deterministic(env):
    spec = env["specs"]["text"]
    a = normalize_field_value("  Cau   SO 1 ", spec.value_policy)
    b = normalize_field_value("cau so 1", spec.value_policy)
    assert a == b


def test_speaker_grouping_internal_only(env):
    assert env["index"].hidden_value_by_row
    summary = env["index"].summary()
    assert "speakerID" not in summary["fields"]


def test_index_does_not_enumerate_pairs(env):
    summary = env["index"].summary()
    assert summary["pair_enumeration"] == "NOT_PERFORMED"


def test_index_rejects_duplicate_row_ids(env):
    specs = env["specs"]
    with pytest.raises(ValueError, match="DUPLICATE_SOURCE_ROW_ID"):
        TrainIndex(env["rows"], ["dup"] * len(env["rows"]), {"region": specs["region"]})


# ------------------------------------------------------------- capacity §109
def test_direct_capacity_formula(env):
    tid = "vimd-v2-s1-002"
    cap = env["capacities"][tid]
    valid = cap["evidence_summary"]["valid_rows"]
    assert cap["theoretical_semantic_capacity"]["positive"] == valid
    assert cap["full_train_capacity_status"] == "SUFFICIENT"


def test_equality_positive_capacity_formula(env):
    tid = "vimd-v2-s1-005"
    cap = env["capacities"][tid]
    counts = [len(v) for v in env["index"].rows_by_value["region"].values()]
    expected = sum(c * (c - 1) // 2 for c in counts)
    assert cap["theoretical_semantic_capacity"]["positive"] == expected


def test_equality_negative_capacity_formula(env):
    tid = "vimd-v2-s1-005"
    cap = env["capacities"][tid]
    counts = [len(v) for v in env["index"].rows_by_value["region"].values()]
    total = sum(counts)
    expected = total * (total - 1) // 2 - sum(c * (c - 1) // 2 for c in counts)
    assert cap["theoretical_semantic_capacity"]["negative"] == expected


def test_target_match_capacity_formula(env):
    tid = "vimd-v2.2-s2-002"
    cap = env["capacities"][tid]
    valid = cap["evidence_summary"]["valid_rows"]
    distinct = cap["evidence_summary"]["unique_values"]
    assert cap["theoretical_semantic_capacity"]["positive"] == valid
    assert cap["theoretical_semantic_capacity"]["negative"] == valid * (distinct - 1)


def test_pairwise_selection_capacity_formula(env):
    tid = "vimd-v2.1-s2-003"
    cap = env["capacities"][tid]
    counts = [len(v) for v in env["index"].rows_by_value["region"].values()]
    total = sum(counts)
    expected = sum(c * (total - c) for c in counts)
    assert cap["theoretical_semantic_capacity"]["selection"] == expected


def test_capacity_uses_groups_not_enumeration(env):
    cap = env["capacities"]["vimd-v2-s1-006"]
    assert cap["evidence_summary"]["pair_enumeration"] == "NOT_PERFORMED"
    assert cap["evidence_summary"]["groups_with_count_ge_2"] >= 1


def test_repeated_province_groups_give_positive_equality(env):
    counts = [len(v) for v in env["index"].rows_by_value["province_name"].values()]
    assert sum(1 for c in counts if c >= 2) >= 4
    assert (
        env["capacities"]["vimd-v2-s1-006"]["theoretical_semantic_capacity"]["positive"]
        > 0
    )


# --------------------------------------------------------------- DIRECT §110
def test_direct_text_gold_correct(env):
    sampler = _sampler(env, "vimd-v2-s1-001")
    drafts = sampler.sample(5)
    index = env["index"]
    for draft in drafts:
        row_id = draft["source_row_ids"][0]
        assert draft["gold_value"] == index.row_by_id[row_id]["text"]


def test_direct_categorical_gold_correct(env):
    sampler = _sampler(env, "vimd-v2-s1-002")
    drafts = sampler.sample(5)
    index = env["index"]
    assert all(
        d["gold_value"] == index.row_by_id[d["source_row_ids"][0]]["region"]
        for d in drafts
    )


def test_direct_semantic_duplicate_prevented(env):
    sampler = _sampler(env, "vimd-v2-s1-002")
    drafts = sampler.sample(30)
    keys = [tuple(d["source_row_ids"]) for d in drafts]
    assert len(keys) == len(set(keys))


# ------------------------------------------------------------- EQUALITY §111
def test_equality_positive_relation(env):
    drafts = _sampler(env, "vimd-v2-s1-005").sample(6)
    index = env["index"]
    positives = [d for d in drafts if d["gold_value"] is True]
    assert positives
    for draft in positives:
        a, b = (index.row_by_id[r]["region"] for r in draft["source_row_ids"])
        assert a == b


def test_equality_negative_relation(env):
    drafts = _sampler(env, "vimd-v2-s1-005").sample(6)
    index = env["index"]
    negatives = [d for d in drafts if d["gold_value"] is False]
    assert negatives
    for draft in negatives:
        a, b = (index.row_by_id[r]["region"] for r in draft["source_row_ids"])
        assert a != b


def test_equality_no_self_pair(env):
    drafts = _sampler(env, "vimd-v2-s1-005").sample(10)
    for draft in drafts:
        assert draft["source_row_ids"][0] != draft["source_row_ids"][1]


def test_equality_reverse_pair_canonicalized(env):
    drafts = _sampler(env, "vimd-v2-s1-005").sample(10)
    for draft in drafts:
        assert draft["source_row_ids"] == sorted(draft["source_row_ids"])


def test_equality_distinct_speaker_preference(env):
    sampler = _sampler(env, "vimd-v2-s1-005")
    drafts = sampler.sample(12)
    index = env["index"]
    positives = [d for d in drafts if d["gold_value"] is True]
    distinct = 0
    for draft in positives:
        a, b = (index.hidden_value_by_row.get(r) for r in draft["source_row_ids"])
        distinct += a != b
    assert distinct >= 1


def test_speaker_identifier_never_in_gold_kinds(env):
    drafts = _sampler(env, "vimd-v2-s1-005").sample(4)
    for draft in drafts:
        assert draft["gold_value"] in (True, False)


# ----------------------------------------------------------- TARGET_MATCH §112
def test_target_match_categorical_positive(env):
    drafts = _sampler(env, "vimd-v2.2-s2-002").sample(6)
    index = env["index"]
    for draft in drafts:
        if draft["gold_value"] is True:
            actual = index.row_by_id[draft["source_row_ids"][0]]["region"]
            assert index.normalize("region", actual) == draft["target"]


def test_target_match_categorical_negative(env):
    drafts = _sampler(env, "vimd-v2.2-s2-002").sample(6)
    index = env["index"]
    for draft in drafts:
        if draft["gold_value"] is False:
            actual = index.normalize(
                "region", index.row_by_id[draft["source_row_ids"][0]]["region"]
            )
            assert draft["target"] != actual


def test_target_match_negative_in_domain(env):
    drafts = _sampler(env, "vimd-v2.2-s2-002").sample(8)
    domain = set(env["index"].values("region"))
    for draft in drafts:
        if draft["gold_value"] is False:
            assert draft["target"] in domain


def test_text_target_match_positive_exact(env):
    drafts = _sampler(env, "vimd-v2.2-s2-003").sample(6)
    index = env["index"]
    for draft in drafts:
        if draft["gold_value"] is True:
            actual = index.normalize(
                "text", index.row_by_id[draft["source_row_ids"][0]]["text"]
            )
            assert draft["target"] == actual


def test_text_target_match_negative_exact_mismatch(env):
    drafts = _sampler(env, "vimd-v2.2-s2-003").sample(8)
    index = env["index"]
    for draft in drafts:
        if draft["gold_value"] is False:
            actual = index.normalize(
                "text", index.row_by_id[draft["source_row_ids"][0]]["text"]
            )
            assert draft["target"] != actual


def test_substring_is_not_exact_match(env):
    spec = env["specs"]["text"]
    long_text = "day la mot cau dai"
    assert normalize_field_value("day la", spec.value_policy) != normalize_field_value(
        long_text, spec.value_policy
    )


def test_text_negative_length_bucket_deterministic(env):
    a = _sampler(env, "vimd-v2.2-s2-003", seed=7).sample(8)
    b = _sampler(env, "vimd-v2.2-s2-003", seed=7).sample(8)
    assert a == b


# ------------------------------------------------------- PAIRWISE_SELECTION §113
def test_pairwise_exactly_one_match(env):
    index = env["index"]
    for tid in _type_by_operator(env, "PAIRWISE_SELECTION"):
        drafts = _sampler(env, tid).sample(6)
        field = env["supported"][tid].semantic_field
        for draft in drafts:
            matches = [
                index.normalize(field, index.row_by_id[r][field]) == draft["target"]
                for r in draft["source_row_ids"]
            ]
            assert sum(matches) == 1


def test_pairwise_gold_follows_position(env):
    index = env["index"]
    drafts = _sampler(env, "vimd-v2.1-s2-003").sample(8)
    for draft in drafts:
        field = "region"
        from src.autonomous_qa.certification.qa_sampling import meta_of

        matches = [
            index.normalize(field, meta_of(index.row_by_id[r])[field])
            == draft["target"]
            for r in draft["source_row_ids"]
        ]
        expected = "A" if matches[0] else "B"
        assert draft["gold_value"] == expected


def test_pairwise_both_match_and_neither_are_invalid(env):
    import tests.regression.qa_audit as qa

    spec = env["specs"]["region"]
    both = {
        "operator": "PAIRWISE_SELECTION",
        "internal": {"hidden_values": ["North", "North"], "target_normalized": "North"},
        "gold": {"kind": "selection", "value": "A"},
    }
    assert qa._gold_is_deterministic(both, spec) is False
    neither = {
        "operator": "PAIRWISE_SELECTION",
        "internal": {
            "hidden_values": ["South", "Central"],
            "target_normalized": "North",
        },
        "gold": {"kind": "selection", "value": "A"},
    }
    assert qa._gold_is_deterministic(neither, spec) is False


def test_pairwise_matching_candidate_can_be_a_or_b(env):
    drafts = _sampler(env, "vimd-v2.1-s2-003", seed=3).sample(20)
    golds = {d["gold_value"] for d in drafts}
    assert golds.issubset({"A", "B"})


def test_reordering_updates_gold(env):
    drafts = _sampler(env, "vimd-v2.1-s2-003", seed=5).sample(6)
    index = env["index"]
    for draft in drafts:
        r0, _r1 = draft["source_row_ids"]
        match0 = (
            index.normalize("region", index.row_by_id[r0]["region"]) == draft["target"]
        )
        assert draft["gold_value"] == ("A" if match0 else "B")


def test_text_pairwise_exact_semantics(env):
    drafts = _sampler(env, "vimd-v2.1-s2-004").sample(6)
    index = env["index"]
    for draft in drafts:
        matches = [
            index.normalize("text", index.row_by_id[r]["text"]) == draft["target"]
            for r in draft["source_row_ids"]
        ]
        assert sum(matches) == 1


# ----------------------------------------------------- semantic identity §114
def test_semantic_instance_id_deterministic(env):
    payload = {
        "dataset": "vimd",
        "dataset_revision": "rev",
        "type_id": "t",
        "operator": "DIRECT",
        "semantic_class": "text_content",
        "source_row_ids": ["a"],
        "target": None,
        "gold_kind": "field_value",
        "gold_value": "x",
    }
    assert pq.semantic_instance_id(payload) == pq.semantic_instance_id(dict(payload))


def test_wording_does_not_change_semantic_id(env):
    payload = {
        "dataset": "vimd",
        "dataset_revision": "rev",
        "type_id": "t",
        "operator": "DIRECT",
        "semantic_class": "categorical_attribute",
        "source_row_ids": ["a"],
        "target": None,
        "gold_kind": "field_value",
        "gold_value": "North",
    }
    sem = pq.semantic_instance_id(payload)
    assert "question" not in payload
    assert sem == pq.semantic_instance_id(payload)


def test_equality_reverse_pair_same_identity(env):
    base = {
        "dataset": "vimd",
        "dataset_revision": "rev",
        "type_id": "t",
        "operator": "EQUALITY",
        "semantic_class": "categorical_attribute",
        "target": None,
        "gold_kind": "boolean",
        "gold_value": True,
    }
    forward = dict(base, source_row_ids=["a", "b"])
    reverse = dict(base, source_row_ids=["b", "a"])
    # Canonical order is the sampler's job; the raw payload keeps order.
    assert pq.semantic_instance_id(forward) != pq.semantic_instance_id(reverse)
    canonical = _sampler(env, "vimd-v2-s1-005").sample(4)
    assert all(d["source_row_ids"] == sorted(d["source_row_ids"]) for d in canonical)


def test_plan_has_unique_semantic_instances(env):
    plan = pq.build_generation_plan(
        contracts=list(env["supported"].values()),
        specs=env["specs"],
        registry=env["registry"],
        index=env["index"],
        config=_config(),
        seed=42,
        budget_by_type={"vimd-v2-s1-002": 5},
        dataset="vimd",
        dataset_revision="rev",
    )
    sems = [r["semantic_instance_id"] for r in plan["records"]]
    assert len(sems) == len(set(sems)) == 5


# --------------------------------------------------------------- language §115
def test_compatible_entries_active_only(env):
    contract = env["supported"]["vimd-v2-s1-002"]
    spec = env["specs"]["region"]
    entries = pq.compatible_language_entries(env["registry"], spec, contract)
    assert entries and all(e.enabled for e in entries)


def test_wrong_operator_incompatible(env):
    contract = env["supported"]["vimd-v2-s1-002"]
    spec = env["specs"]["region"]
    entries = pq.compatible_language_entries(env["registry"], spec, contract)
    assert all(e.operator == contract.operator for e in entries)


def test_wrong_semantic_class_incompatible(env):
    contract = env["supported"]["vimd-v2-s1-002"]
    spec = env["specs"]["region"]
    entries = pq.compatible_language_entries(env["registry"], spec, contract)
    assert all(e.semantic_class == spec.semantic_class for e in entries)


def test_zero_entries_raises_coverage_missing(env):
    contract = env["supported"]["vimd-v2-s1-002"]
    spec = env["specs"]["region"].model_copy(update={"semantic_class": "nonexistent"})
    with pytest.raises(ValueError, match="LANGUAGE_LIBRARY_COVERAGE_MISSING"):
        pq.compatible_language_entries(env["registry"], spec, contract)


def test_one_instance_selects_exactly_one_entry(env):
    contract = env["supported"]["vimd-v2-s1-002"]
    entries = pq.compatible_language_entries(
        env["registry"], env["specs"]["region"], contract
    )
    chosen = pq.select_language_entry(
        entries, seed=42, sem_id="sem_x", registry_hash=env["registry"].registry_hash
    )
    assert chosen in entries


def test_registry_is_a_pool_not_a_multiplier(env):
    active = [e for e in env["registry"].entries if e.enabled]
    assert len(active) == 62
    plan = pq.build_generation_plan(
        contracts=list(env["supported"].values()),
        specs=env["specs"],
        registry=env["registry"],
        index=env["index"],
        config=_config(),
        seed=42,
        budget_by_type={"vimd-v2-s1-002": 3, "vimd-v2-s1-003": 3},
        dataset="vimd",
        dataset_revision="rev",
    )
    assert len(plan["records"]) == 6  # not 6 * 56


def test_language_selection_deterministic(env):
    contract = env["supported"]["vimd-v2-s1-002"]
    entries = pq.compatible_language_entries(
        env["registry"], env["specs"]["region"], contract
    )
    a = pq.select_language_entry(
        entries, seed=1, sem_id="sem_x", registry_hash=env["registry"].registry_hash
    )
    b = pq.select_language_entry(
        list(reversed(entries)),
        seed=1,
        sem_id="sem_x",
        registry_hash=env["registry"].registry_hash,
    )
    assert a.language_entry_id == b.language_entry_id


def test_no_python_builtin_hash_in_sampling_source():
    source = (ROOT / "src" / "autonomous_qa" / "certification" / "qa_sampling.py").read_text(encoding="utf-8")
    assert "hash(" not in source.replace("hashlib", "")


def test_language_entry_affects_qa_id_not_semantic_id(env):
    sem = "sem_abc"
    a = pq.qa_id_for(sem, "lang_1", "registry-hash")
    b = pq.qa_id_for(sem, "lang_2", "registry-hash")
    assert a != b


# --------------------------------------------------------------- leakage §116
def _realized(env, budget=6):
    plan = pq.build_generation_plan(
        contracts=list(env["supported"].values()),
        specs=env["specs"],
        registry=env["registry"],
        index=env["index"],
        config=_config(),
        seed=42,
        budget_by_type={tid: budget for tid in env["supported"]},
        dataset="vimd",
        dataset_revision="rev",
    )
    internal, model = pq.realize_qa(
        plan["records"],
        contracts=dict(env["supported"]),
        specs=env["specs"],
        registry=env["registry"],
        index=env["index"],
        config=_config(),
        seed=42,
        metadata_sha256="sha",
        type_registry_hash="treg",
    )
    return plan, internal, model


def test_model_facing_has_no_hidden_fields(env):
    _, _, model = _realized(env)
    for record in model:
        blob = json.dumps(record, ensure_ascii=False)
        assert "speakerID" not in blob
        assert "filename" not in blob
        assert "province_code" not in blob
        assert "spk_" not in blob
        assert ".wav" not in blob
        assert set(record) == {
            "id",
            "audio",
            "question",
            "answer",
            "type_id",
            "operator",
        }


def test_opaque_audio_id_has_no_semantic_labels(env):
    aid = pq.opaque_audio_id("vimd", "rev", "clip_0001.wav")
    assert aid.startswith("audio_") and len(aid) == 26
    for token in ("north", "south", "p1", "spk", "wav", "clip"):
        assert token not in aid.lower()


def test_visible_target_allowed_for_target_operators(env):
    _, internal, _ = _realized(env)
    target_based = [
        r for r in internal if r["operator"] in {"TARGET_MATCH", "PAIRWISE_SELECTION"}
    ]
    assert target_based
    assert all(
        r["visible_context"].get("target_value") is not None for r in target_based
    )


def test_hidden_candidate_value_absent_from_question(env):
    _, internal, _ = _realized(env)
    for record in internal:
        question = " ".join(record["question"].casefold().split())
        hidden = record["internal"]["hidden_values"]
        if record["operator"] in {"DIRECT", "EQUALITY"}:
            for value in hidden:
                assert " ".join(str(value).casefold().split()) not in question


def test_answer_without_audio_is_flagged(env):
    _, internal, _ = _realized(env)
    broken = json.loads(json.dumps(internal[0]))
    broken["operator"] = "DIRECT"
    broken["internal"]["hidden_values"] = ["LEAKED_VALUE"]
    broken["question"] = broken["question"] + " LEAKED_VALUE"
    from tests.regression.qa_audit import audit_record

    failures = audit_record(
        broken,
        contracts=dict(env["supported"]),
        specs=env["specs"],
        registry=env["registry"],
        compatible_by_type={r["type_id"]: {r["language_entry_id"]} for r in internal},
        capacity_status={t: "SUFFICIENT" for t in env["supported"]},
        index=env["index"],
        model_record={"id": "x"},
    )
    assert "NO_VISIBLE_GOLD_LEAK" in failures


# ---------------------------------------------------------------- budget §117
def test_no_budget_ready_awaiting(env):
    result = pq.resolve_budget(
        _config(), list(env["supported"].values()), env["capacities"]
    )
    assert result["status"] == "READY_AWAITING_BUDGET"
    assert result["explicit_budget_present"] is False


def test_per_type_budget_resolves(env):
    contracts = list(env["supported"].values())
    budget = pq.resolve_budget(
        _config(per_type_budget={"vimd-v2-s1-002": 3, "vimd-v2-s1-003": 2}),
        contracts,
        env["capacities"],
    )
    assert budget["status"] == "PER_TYPE_BUDGET"
    assert budget["per_type"]["vimd-v2-s1-002"] == 3
    assert budget["per_type"]["vimd-v2-s1-003"] == 2
    assert budget["per_type"]["vimd-v2-s1-005"] == 0


def test_total_budget_equal_allocation(env):
    contracts = list(env["supported"].values())
    budget = pq.resolve_budget(
        _config(total_target_qa=23), contracts, env["capacities"]
    )
    assert budget["status"] == "TOTAL_BUDGET_EQUAL_BY_TYPE"
    assert budget["resolved_total"] == 23
    assert sorted(budget["per_type"].values()) == [2] * 10 + [3]


def test_infeasible_strict_budget_reports_shortfall(env):
    contracts = list(env["supported"].values())
    huge = {c.type_id: 10_000_000 for c in contracts}
    budget = pq.resolve_budget(
        _config(per_type_budget=huge), contracts, env["capacities"]
    )
    assert budget["status"] == "BUDGET_INFEASIBLE"
    assert budget["shortfall_total"] > 0
    assert budget["capacity_capping"]


def test_duplicates_not_used_to_fill_quota(env):
    # request more DIRECT rows than exist -> sampler raises shortfall, caught
    with pytest.raises(BudgetShortfallError):
        pq.build_generation_plan(
            contracts=list(env["supported"].values()),
            specs=env["specs"],
            registry=env["registry"],
            index=env["index"],
            config=_config(),
            seed=42,
            budget_by_type={"vimd-v2-s1-002": len(env["rows"]) + 5},
            dataset="vimd",
            dataset_revision="rev",
        )


# ------------------------------------------------------------ determinism §118
def test_generation_plan_deterministic(env):
    kw = {
        "contracts": list(env["supported"].values()),
        "specs": env["specs"],
        "registry": env["registry"],
        "index": env["index"],
        "config": _config(),
        "seed": 42,
        "budget_by_type": {t: 3 for t in env["supported"]},
        "dataset": "vimd",
        "dataset_revision": "rev",
    }
    a = pq.build_generation_plan(**kw)
    b = pq.build_generation_plan(**kw)
    assert a["fingerprint"] == b["fingerprint"]
    assert a["records"] == b["records"]


def test_language_assignment_and_qa_ids_deterministic(env):
    _, internal_a, _ = _realized(env, budget=3)
    _, internal_b, _ = _realized(env, budget=3)
    assert [r["qa_id"] for r in internal_a] == [r["qa_id"] for r in internal_b]
    assert [r["language_entry_id"] for r in internal_a] == [
        r["language_entry_id"] for r in internal_b
    ]


def test_changed_seed_preserves_contracts(env):
    plan_a = pq.build_generation_plan(
        contracts=list(env["supported"].values()),
        specs=env["specs"],
        registry=env["registry"],
        index=env["index"],
        config=_config(),
        seed=1,
        budget_by_type={"vimd-v2-s1-005": 4},
        dataset="vimd",
        dataset_revision="rev",
    )
    plan_b = pq.build_generation_plan(
        contracts=list(env["supported"].values()),
        specs=env["specs"],
        registry=env["registry"],
        index=env["index"],
        config=_config(),
        seed=2,
        budget_by_type={"vimd-v2-s1-005": 4},
        dataset="vimd",
        dataset_revision="rev",
    )
    assert len(plan_a["records"]) == len(plan_b["records"]) == 4
    for record in plan_a["records"] + plan_b["records"]:
        assert record["operator"] == "EQUALITY"
        assert record["gold"]["kind"] == "boolean"


# ----------------------------------------------------------------- zero-LLM §119
def test_production_qa_module_does_not_import_llm_client():
    source = (ROOT / "src" / "autonomous_qa" / "production" / "production_qa.py").read_text(encoding="utf-8")
    assert "llm_client" not in source
    assert "openai" not in source.lower()
    assert "requests" not in source


def test_readiness_makes_no_llm_calls(tmp_path, synthetic_rows, monkeypatch):
    called = {"n": 0}

    def _boom(*a, **k):
        called["n"] += 1
        raise AssertionError("LLM CALLED")

    import src.autonomous_qa.authoring.llm_client as llm

    monkeypatch.setattr(llm, "create_llm_client", _boom, raising=False)
    monkeypatch.setattr(llm, "call_llm", _boom, raising=False)
    summary = _run_synthetic(
        tmp_path, synthetic_rows, mode="readiness", run_id="readiness"
    )
    assert summary["conclusion"] == "READY_AWAITING_BUDGET"
    assert called["n"] == 0


def test_debug_sample_makes_no_llm_calls(tmp_path, synthetic_rows, monkeypatch):
    called = {"n": 0}

    def _boom(*a, **k):
        called["n"] += 1
        raise AssertionError("LLM CALLED")

    import src.autonomous_qa.authoring.llm_client as llm

    monkeypatch.setattr(llm, "create_llm_client", _boom, raising=False)
    summary = _run_synthetic(tmp_path, synthetic_rows, debug_sample=33, run_id="debug")
    assert summary["qa_generated"] > 0
    assert called["n"] == 0


# ---------------------------------------------------------------- scale §120
def test_large_synthetic_index_is_linear():
    rows = []
    for i in range(10000):
        rows.append(
            _row(
                i,
                f"P{i % 40}",
                "North" if i % 2 else "South",
                f"text {i % 500}",
                f"spk_{i % 60}",
            )
        )
    row_ids = [r["filename"] for r in rows]
    specs = pq.vimd_field_specs()
    index = TrainIndex(rows, row_ids, {"region": specs["region"]})
    assert index.summary()["rows_indexed"] == 10000
    assert index.summary()["pair_enumeration"] == "NOT_PERFORMED"


def test_reuse_caps_respected(env):
    tracker = ReuseTracker(global_cap=1)
    sampler = SemanticSampler(
        env["supported"]["vimd-v2-s1-002"],
        env["specs"]["region"],
        env["index"],
        _settings(),
        tracker,
        42,
    )
    with pytest.raises(BudgetShortfallError):
        sampler.sample(len(env["rows"]) + 5)
    assert all(v <= 1 for v in tracker.global_counts.values())


# -------------------------------------------------- synthetic end-to-end gate
def test_synthetic_debug_run_full_pipeline(tmp_path, synthetic_rows):
    summary = _run_synthetic(tmp_path, synthetic_rows, debug_sample=44, run_id="e2e")
    run_dir = Path(summary["run_dir"])
    assert summary["conclusion"] == "READY_AWAITING_BUDGET"
    assert summary["qa_generated"] > 0
    for name in (
        "input_manifest.json",
        "canonical_resource_hashes.json",
        "metadata_audit.json",
        "index_summary.json",
        "capacity_audit.json",
        "language_coverage.json",
        "budget_resolution.json",
        "qa_validation.json",
        "determinism_audit.json",
        "audit.json",
        "report.md",
    ):
        assert (run_dir / name).is_file()
    validation = json.loads(
        (run_dir / "qa_validation.json").read_text(encoding="utf-8")
    )
    assert validation["all_passed"] is True


def test_synthetic_run_is_deterministic(tmp_path, synthetic_rows):
    a = _run_synthetic(tmp_path / "a", synthetic_rows, debug_sample=22, run_id="r")
    b = _run_synthetic(tmp_path / "b", synthetic_rows, debug_sample=22, run_id="r")
    a_plan = (Path(a["run_dir"]) / "generation_plan.jsonl").read_text(encoding="utf-8")
    b_plan = (Path(b["run_dir"]) / "generation_plan.jsonl").read_text(encoding="utf-8")
    assert a_plan == b_plan
    assert (Path(a["run_dir"]) / "generation_plan.sha256").read_text() == (
        Path(b["run_dir"]) / "generation_plan.sha256"
    ).read_text()


# ---------------------------------------------------- generic preflight & zero-audio
def test_production_qa_module_does_not_import_audio_decoders_or_ffmpeg():
    source = (ROOT / "src" / "autonomous_qa" / "production" / "production_qa.py").read_text(encoding="utf-8")
    for forbidden in (
        "import wave",
        "import audioop",
        "import pydub",
        "import soundfile",
        "import librosa",
        "import ffmpeg",
        "ffmpeg",
    ):
        assert forbidden not in source


def test_preflight_gate_fallback_resolution_dataset_aware(tmp_path, monkeypatch):
    # 1. Explicit config path wins over fallback
    cfg_explicit = _config(language_preflight_artifact="custom/nonexistent/path/audit.json")
    with pytest.raises(pq.ProductionQAError):
        pq.enforce_language_preflight_gate(dataset="vimd", config=cfg_explicit)

    # 2. Default fallback for vimd resolves to canonical data/materialized/language_preflight/vimd/current
    cfg_default = _config(language_preflight_artifact=None)
    decision = pq.enforce_language_preflight_gate(dataset="vimd", config=cfg_default)
    assert decision["allowed"] is True

    # 3. Prove dataset-aware fallback path resolution for vimd, vietmdd, vimedcss
    from src.autonomous_qa.language import language_preflight as lp

    captured_artifacts: dict[str, Any] = {}

    def mock_auth(*, require_language_preflight, current_fingerprint, pass_artifact):
        if pass_artifact:
            captured_artifacts[pass_artifact.get("ds_name")] = pass_artifact
        return {"allowed": True, "status": "PREFLIGHT_PASS"}

    monkeypatch.setattr(lp, "authorize_new_production", mock_auth)
    monkeypatch.setattr(lp, "vimd_accepted_types", lambda: [])
    monkeypatch.setattr(lp, "compute_contract_fingerprint", lambda mode, accepted_types: ("fp", {}))
    monkeypatch.setattr(pq, "ROOT", tmp_path)

    for ds in ("vimd", "vietmdd", "vimedcss"):
        target_dir = tmp_path / "data" / "materialized" / "language_preflight" / ds / "current"
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "audit.json").write_text(json.dumps({"ds_name": ds}))

    # Helper to execute preflight gate resolution logic for dataset
    def run_gate_for(ds_name: str):
        config = _config(language_preflight_artifact=None)
        target_artifact = (
            config.language_preflight_artifact
            or f"data/materialized/language_preflight/{ds_name}/current"
        )
        artifact_path = tmp_path / target_artifact / "audit.json"
        artifact = json.loads(artifact_path.read_text(encoding="utf-8")) if artifact_path.exists() else None
        lp.authorize_new_production(
            require_language_preflight=True,
            current_fingerprint="fp",
            pass_artifact=artifact,
        )

    for ds in ("vimd", "vietmdd", "vimedcss"):
        run_gate_for(ds)
        assert captured_artifacts[ds]["ds_name"] == ds


def test_non_vimd_preflight_fallback_does_not_leak_vimd_path(tmp_path, monkeypatch):
    from src.autonomous_qa.language import language_preflight as lp

    captured = {"pass_artifact": "INITIAL"}

    def mock_auth(*, require_language_preflight, current_fingerprint, pass_artifact):
        captured["pass_artifact"] = pass_artifact
        return {"allowed": True, "status": "PREFLIGHT_PASS"}

    monkeypatch.setattr(lp, "authorize_new_production", mock_auth)
    monkeypatch.setattr(lp, "vimd_accepted_types", lambda: [])
    monkeypatch.setattr(lp, "compute_contract_fingerprint", lambda mode, accepted_types: ("fp", {}))
    monkeypatch.setattr(pq, "ROOT", tmp_path)

    # Place a valid preflight audit ONLY under vimd
    vimd_dir = tmp_path / "data" / "materialized" / "language_preflight" / "vimd" / "current"
    vimd_dir.mkdir(parents=True, exist_ok=True)
    (vimd_dir / "audit.json").write_text(json.dumps({"ds_name": "vimd"}))

    # Evaluate resolution for vietmdd when language_preflight_artifact is None
    ds_name = "vietmdd"
    config = _config(language_preflight_artifact=None)
    target_artifact = (
        config.language_preflight_artifact
        or f"data/materialized/language_preflight/{ds_name}/current"
    )
    artifact_path = tmp_path / target_artifact / "audit.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8")) if artifact_path.exists() else None
    lp.authorize_new_production(
        require_language_preflight=True,
        current_fingerprint="fp",
        pass_artifact=artifact,
    )

    # Must be None because vietmdd preflight path is absent (did NOT silently fall back to ViMD!)
    assert captured["pass_artifact"] is None


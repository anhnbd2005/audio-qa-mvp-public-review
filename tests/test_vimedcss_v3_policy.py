"""ViMedCSS V3 generic 4+4 anchor-neighborhood policy + split isolation tests."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml

from src.autonomous_qa.certification.anchor_neighborhood import (
    anchor_neighborhood_feasibility,
    audit_anchor_neighborhood_plan,
    audit_split_isolation,
)
from src.autonomous_qa.certification.qa_sampling import (
    ReuseTracker,
    SamplingSettings,
    SemanticSampler,
    TrainIndex,
)
from src.autonomous_qa.compiler.semantic_field_specs import SemanticFieldSpec
from src.autonomous_qa.language.language_quality import (
    ProductionGenerationConfig,
)
from src.autonomous_qa.language.template_contracts import TypeContract
from src.autonomous_qa.production.production_qa import (
    _sampling_settings,
    assert_split_authorized,
    dataset_allowed_splits,
    split_metadata_path,
)
from src.common.config import ROOT

PAIR = "vimedcss_pairwise_topic_same"
TOPIC_TASK = "vimedcss_topic_classification"
COUNT_TASK = "vimedcss_cs_terms_count"


def _spec(field: str = "topic") -> SemanticFieldSpec:
    return SemanticFieldSpec(
        field_name=field,
        semantic_class="categorical_attribute",
        entity_scope="utterance",
        entity_phrase="Ä‘oáº¡n Ã¢m thanh",
        attribute_phrase="nhÃ£n",
        value_phrase="nhÃ£n",
        value_policy={"normalization": "identity", "match_policy": "exact"},
    )


def _contract(type_id: str, operator: str, field: str = "topic") -> TypeContract:
    return TypeContract(
        type_id=type_id,
        operator=operator,
        semantic_field=field,
        semantic_description="x",
        semantic_phrase_vi="x",
        entity_scope="utterance",
        audio_input_count=2 if operator == "EQUALITY" else 1,
        condition_fields=[],
        gold_source_fields=[field],
        answer_kind="boolean" if operator == "EQUALITY" else "field_value",
        answer_mode="BOOLEAN" if operator == "EQUALITY" else "FIELD_VALUE",
        template_status="ELIGIBLE",
        template_policy={},
        instantiation_policy={},
        source_status="SUPPORTED",
    )


def _index(field: str = "topic", group_sizes: tuple[int, ...] = (16, 16, 16)) -> TrainIndex:
    rows, row_ids = [], []
    labels = ["A", "B", "C", "D", "E"]
    for gi, size in enumerate(group_sizes):
        for j in range(size):
            rid = f"r_{labels[gi]}_{j}"
            row_ids.append(rid)
            rows.append({"segment_id": rid, field: labels[gi]})
    return TrainIndex(rows, row_ids, {field: _spec(field)})


def _sampler(index: TrainIndex, operator: str, field: str = "topic", **over) -> SemanticSampler:
    settings = SamplingSettings(
        equality_strategy=over.pop("strategy", "anchor_neighborhood"),
        equality_positive_per_anchor=over.pop("pos", 4),
        equality_negative_per_anchor=over.pop("neg", 4),
        equality_pair_uniqueness=over.pop("uniq", "unordered"),
        equality_candidate_pool_same=over.pop("pool_same", 8),
        equality_candidate_pool_different=over.pop("pool_diff", 8),
        positive_ratio=0.5,
    )
    return SemanticSampler(
        _contract(PAIR if operator == "EQUALITY" else TOPIC_TASK, operator, field),
        _spec(field),
        index,
        settings,
        ReuseTracker(),
        seed=42,
    )


def _pair(anchor, partner, gold, split="train"):
    return {"type_id": PAIR, "source_row_ids": [anchor, partner], "gold": {"value": gold}, "split": split}


def _lookup_from(rows: dict, field: str = "topic"):
    return lambda r: rows.get(r)


# ---------------------------------------------------------------------------
# A. generic quota handling (auditor)
# ---------------------------------------------------------------------------


def _records_for(anchor_specs, lookup):
    records = []
    for anchor, same, diff in anchor_specs:
        for p in same:
            records.append(_pair(anchor, p, True))
        for p in diff:
            records.append(_pair(anchor, p, False))
    return records


def test_auditor_accepts_valid_4_4_anchor():
    lookup = {"a": "x", "b": "x", "c": "x", "d": "x", "e": "x", "n1": "y", "n2": "y", "n3": "y", "n4": "y"}
    records = _records_for([("a", ["b", "c", "d", "e"], ["n1", "n2", "n3", "n4"])], lookup)
    report = audit_anchor_neighborhood_plan(
        records, type_id=PAIR, row_value_lookup=lambda r: lookup[r],
        positive_per_anchor=4, negative_per_anchor=4, expected_anchors=1,
    )
    assert report["passed"] is True, report["failures"]


def test_auditor_rejects_missing_positive():
    lookup = {"a": "x", "b": "x", "n1": "y", "n2": "y", "n3": "y", "n4": "y"}
    records = _records_for([("a", ["b"], ["n1", "n2", "n3", "n4"])], lookup)
    report = audit_anchor_neighborhood_plan(
        records, type_id=PAIR, row_value_lookup=lambda r: lookup[r],
        positive_per_anchor=4, negative_per_anchor=4, expected_anchors=1,
    )
    assert report["passed"] is False
    assert any(f.startswith("PER_ANCHOR_COUNTS_BAD") for f in report["failures"])


def test_auditor_rejects_missing_negative():
    lookup = {"a": "x", "b": "x", "c": "x", "d": "x", "e": "x", "n1": "y"}
    records = _records_for([("a", ["b", "c", "d", "e"], ["n1"])], lookup)
    report = audit_anchor_neighborhood_plan(
        records, type_id=PAIR, row_value_lookup=lambda r: lookup[r],
        positive_per_anchor=4, negative_per_anchor=4, expected_anchors=1,
    )
    assert report["passed"] is False


def test_auditor_rejects_excess_positive():
    lookup = {"a": "x", "b": "x", "c": "x", "d": "x", "e": "x", "f": "x", "n1": "y", "n2": "y", "n3": "y", "n4": "y"}
    records = _records_for([("a", ["b", "c", "d", "e", "f"], ["n1", "n2", "n3", "n4"])], lookup)
    report = audit_anchor_neighborhood_plan(
        records, type_id=PAIR, row_value_lookup=lambda r: lookup[r],
        positive_per_anchor=4, negative_per_anchor=4, expected_anchors=1,
    )
    assert report["passed"] is False


def test_auditor_rejects_excess_negative():
    lookup = {"a": "x", "b": "x", "c": "x", "d": "x", "e": "x", "n1": "y", "n2": "y", "n3": "y", "n4": "y", "n5": "y"}
    records = _records_for([("a", ["b", "c", "d", "e"], ["n1", "n2", "n3", "n4", "n5"])], lookup)
    report = audit_anchor_neighborhood_plan(
        records, type_id=PAIR, row_value_lookup=lambda r: lookup[r],
        positive_per_anchor=4, negative_per_anchor=4, expected_anchors=1,
    )
    assert report["passed"] is False


def test_auditor_supports_arbitrary_quotas():
    lookup = {"a": "x", "b": "x", "n1": "y"}
    records = _records_for([("a", ["b"], ["n1"])], lookup)
    report = audit_anchor_neighborhood_plan(
        records, type_id=PAIR, row_value_lookup=lambda r: lookup[r],
        positive_per_anchor=1, negative_per_anchor=2 if False else 1, expected_anchors=1,
    )
    assert report["passed"] is True
    assert report["records"] == 2


# ---------------------------------------------------------------------------
# B. uniqueness / correctness
# ---------------------------------------------------------------------------


def test_no_self_pairs_and_no_reverse_duplicates():
    index = _index()
    anchors = sorted(index.valid_row_ids["topic"])
    drafts = _sampler(index, "EQUALITY").sample(len(anchors) * 8)
    seen = set()
    for d in drafts:
        a, b = d["source_row_ids"]
        assert a != b
        key = tuple(sorted((a, b)))
        assert key not in seen
        seen.add(key)


def test_unordered_uniqueness_blocks_reverse():
    index = _index(group_sizes=(20, 20, 20))
    drafts = _sampler(index, "EQUALITY", uniq="unordered").sample(60 * 8)
    keys = [tuple(sorted(d["source_row_ids"])) for d in drafts]
    assert len(keys) == len(set(keys))


def test_ordered_uniqueness_allows_reverse():
    index = _index(group_sizes=(20, 20, 20))
    drafts = _sampler(index, "EQUALITY", uniq="ordered").sample(60 * 8)
    keys = [tuple(d["source_row_ids"]) for d in drafts]
    assert len(keys) == len(set(keys))
    # reverse pairs are permitted in ordered mode; there must still be no exact dup
    assert all(d["source_row_ids"][0] != d["source_row_ids"][1] for d in drafts)


def test_missing_metadata_rows_are_not_eligible():
    rows = [
        {"segment_id": "r1", "topic": "A"},
        {"segment_id": "r2", "topic": ""},      # missing
        {"segment_id": "r3", "topic": None},    # missing
        {"segment_id": "r4", "topic": "A"},
    ]
    index = TrainIndex(rows, [r["segment_id"] for r in rows], {"topic": _spec()})
    assert set(index.valid_row_ids["topic"]) == {"r1", "r4"}


def test_gold_derived_from_metadata():
    index = _index(group_sizes=(16, 16, 16))
    drafts = _sampler(index, "EQUALITY").sample(48 * 8)
    for d in drafts:
        a, b = d["source_row_ids"]
        expected = index.normalize("topic", index.row_by_id[a]["topic"]) == index.normalize(
            "topic", index.row_by_id[b]["topic"]
        )
        assert bool(d["gold_value"]) == bool(expected)


# ---------------------------------------------------------------------------
# C. split isolation
# ---------------------------------------------------------------------------


def test_split_isolation_accepts_same_split():
    records = [_pair("a", "b", True, split="train"), _pair("c", "d", False, split="train")]
    report = audit_split_isolation(records, split="train")
    assert report["passed"] is True


def test_split_isolation_rejects_cross_split():
    records = [_pair("a", "b", True, split="train"), _pair("c", "d", False, split="test")]
    report = audit_split_isolation(records, split="train")
    assert report["passed"] is False
    assert any(f.startswith("SPLIT_MISMATCH") for f in report["failures"])


def test_all_four_splits_are_authorized():
    for split in ("train", "validation", "test", "hard"):
        assert_split_authorized("vimedcss", split)  # must not raise
    assert set(dataset_allowed_splits("vimedcss")) == {
        "train", "validation", "test", "hard",
    }


def test_unknown_split_is_rejected():
    with pytest.raises(Exception) as exc:
        assert_split_authorized("vimedcss", "nonexistent_split")
    assert getattr(exc.value, "code", "") == "SPLIT_NOT_AUTHORIZED"


def test_train_is_authorized():
    assert_split_authorized("vimedcss", "train")
    assert "train" in dataset_allowed_splits("vimedcss")


def test_missing_split_is_not_invented():
    source = {"metadata_path": ROOT / "data_sources" / "vimedcss" / "source" / "train.jsonl"}
    with pytest.raises(Exception) as exc:
        split_metadata_path(source, "nonexistent_split")
    assert getattr(exc.value, "code", "") == "SPLIT_SOURCE_MISSING"


def test_infeasible_split_is_reported():
    # group of 2 cannot give 4 same-topic partners -> feasibility false
    index = _index(group_sizes=(2, 2, 10))
    report = anchor_neighborhood_feasibility(
        index, "topic", positive_per_anchor=4, negative_per_anchor=4
    )
    assert report["feasible"] is False


# ---------------------------------------------------------------------------
# D. determinism
# ---------------------------------------------------------------------------


def test_plan_deterministic_across_runs():
    index = _index()
    a = _sampler(index, "EQUALITY").sample(48 * 8)
    b = _sampler(index, "EQUALITY").sample(48 * 8)
    assert a == b


def test_deterministic_fallback_with_small_pool():
    index = _index()
    a = _sampler(index, "EQUALITY", pool_same=1, pool_diff=1).sample(48 * 8)
    b = _sampler(index, "EQUALITY", pool_same=1, pool_diff=1).sample(48 * 8)
    assert a == b
    assert len(a) == 48 * 8


# ---------------------------------------------------------------------------
# E. coverage and budget
# ---------------------------------------------------------------------------


def test_direct_strict_1_1():
    index = _index()
    total = len(index.valid_row_ids["topic"])
    drafts = _sampler(index, "DIRECT").sample(total)
    rows = [d["source_row_ids"][0] for d in drafts]
    assert len(set(rows)) == total
    assert Counter(rows).most_common(1)[0][1] == 1


def test_exactly_4_4_per_anchor_and_8n_total():
    index = _index()
    anchors = sorted(index.valid_row_ids["topic"])
    drafts = _sampler(index, "EQUALITY").sample(len(anchors) * 8)
    same = Counter()
    diff = Counter()
    first = Counter()
    for d in drafts:
        a, _b = d["source_row_ids"]
        first[a] += 1
        (same if d["gold_value"] else diff)[a] += 1
    assert all(same[a] == 4 and diff[a] == 4 for a in anchors)
    assert sum(same.values()) == 4 * len(anchors)
    assert sum(diff.values()) == 4 * len(anchors)
    assert len(drafts) == 8 * len(anchors)


def test_train_expected_total_formula():
    n = 11832
    assert n + n + 4 * n + 4 * n == 118320
    assert 10 * n == 118320


def test_v3_config_resolves_4_4_and_full_split_mode():
    raw = yaml.safe_load(
        (ROOT / "configs" / "qa_generation_vimedcss_v3.yaml").read_text(encoding="utf-8")
    )
    config = ProductionGenerationConfig.model_validate(raw)
    assert config.split == "train"
    assert config.coverage_mode == "full_split"
    settings = _sampling_settings(config)
    assert settings.equality_positive_per_anchor == 4
    assert settings.equality_negative_per_anchor == 4
    assert settings.equality_strategy == "anchor_neighborhood"


def test_full_split_budget_derived_from_anchors():
    from src.autonomous_qa.production.production_qa import derive_full_split_budget

    raw = yaml.safe_load(
        (ROOT / "configs" / "qa_generation_vimedcss_v3.yaml").read_text(encoding="utf-8")
    )
    config = ProductionGenerationConfig.model_validate(raw)
    contracts = [
        _contract(TOPIC_TASK, "DIRECT"),
        _contract(COUNT_TASK, "DIRECT"),
        _contract(PAIR, "EQUALITY"),
    ]
    capacities = {
        TOPIC_TASK: {"evidence_summary": {"valid_rows": 11832}},
        COUNT_TASK: {"evidence_summary": {"valid_rows": 11832}},
        PAIR: {"evidence_summary": {"valid_rows": 11832}},
    }
    derived = derive_full_split_budget(config, contracts, capacities)
    assert derived[TOPIC_TASK] == 11832
    assert derived[COUNT_TASK] == 11832
    assert derived[PAIR] == 94656
    assert sum(derived.values()) == 118320


def test_release_builder_rejects_wrong_count(tmp_path: Path):
    from src.autonomous_qa.datasets.vimedcss import vimedcss_release as release

    run = tmp_path / "run"
    run.mkdir()
    rows = [
        {"qa_id": "q1", "type_id": TOPIC_TASK, "operator": "DIRECT", "dataset": "vimedcss",
         "audio_ids": ["a1"], "gold": {"kind": "field_value", "value": "x"},
         "internal": {"source_row_ids": ["s1"]}},
    ]
    (run / "qa_internal.jsonl").write_text(
        "".join(__import__("json").dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    with pytest.raises(release.VimedcssReleaseError) as exc:
        release.build_logical_release(
            run_dir=run, output_dir=tmp_path / "rel", source_revision="rev",
            topics_by_row={"s1": "A"},
            expected_counts={TOPIC_TASK: 11832}, expected_total=118320,
        )
    assert exc.value.code == "FINAL_RELEASE_COUNT_MISMATCH"


# ---------------------------------------------------------------------------
# F. generic reuse (same mechanism, different relation field)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["topic", "dialect"])
def test_generic_mechanism_reused_for_different_relations(field):
    index = _index(field)
    anchors = sorted(index.valid_row_ids[field])
    drafts = _sampler(index, "EQUALITY", field=field).sample(len(anchors) * 8)
    same = Counter()
    diff = Counter()
    for d in drafts:
        a, b = d["source_row_ids"]
        assert index.normalize(field, index.row_by_id[a][field]) == index.normalize(
            field, index.row_by_id[b][field]
        ) or not d["gold_value"]
        (same if d["gold_value"] else diff)[a] += 1
    assert all(same[a] == 4 and diff[a] == 4 for a in anchors)

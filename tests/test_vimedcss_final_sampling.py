"""ViMedCSS V2 final pairwise sampling policy tests.

Generic anchor-neighborhood EQUALITY strategy + final plan audits. Synthetic
indices only; the real 70,992 budget is never generated here.
"""

from __future__ import annotations

import json
import socket
import urllib.request
from collections import Counter
from pathlib import Path

import pytest
import yaml

from src.autonomous_qa.certification.anchor_neighborhood import (
    anchor_neighborhood_feasibility,
    audit_anchor_neighborhood_plan,
    audit_direct_coverage,
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
    SamplingPolicyConfig,
)
from src.autonomous_qa.language.template_contracts import TypeContract
from src.autonomous_qa.production.production_qa import _sampling_settings
from src.common.config import ROOT

PAIR = "vimedcss_pairwise_topic_same"
TOPIC_TASK = "vimedcss_topic_classification"
COUNT_TASK = "vimedcss_cs_terms_count"
FINAL_COUNTS = {TOPIC_TASK: 11832, COUNT_TASK: 11832, PAIR: 47328}
FINAL_TOTAL = 70992


def _spec() -> SemanticFieldSpec:
    return SemanticFieldSpec(
        field_name="topic",
        semantic_class="categorical_attribute",
        entity_scope="utterance",
        entity_phrase="đoạn âm thanh",
        attribute_phrase="chủ đề y khoa",
        value_phrase="chủ đề",
        value_policy={"normalization": "identity", "match_policy": "exact"},
    )


def _contract(type_id: str, operator: str) -> TypeContract:
    return TypeContract(
        type_id=type_id,
        operator=operator,
        semantic_field="topic",
        semantic_description="x",
        semantic_phrase_vi="x",
        entity_scope="utterance",
        audio_input_count=2 if operator == "EQUALITY" else 1,
        condition_fields=[],
        gold_source_fields=["topic"],
        answer_kind="boolean" if operator == "EQUALITY" else "field_value",
        answer_mode="BOOLEAN" if operator == "EQUALITY" else "FIELD_VALUE",
        template_status="ELIGIBLE",
        template_policy={},
        instantiation_policy={},
        source_status="SUPPORTED",
    )


def _index(group_sizes: tuple[int, ...] = (8, 8, 8)) -> TrainIndex:
    rows = []
    row_ids = []
    labels = ["A", "B", "C", "D", "E"]
    for gi, size in enumerate(group_sizes):
        for j in range(size):
            rid = f"r_{labels[gi]}_{j}"
            row_ids.append(rid)
            rows.append({"segment_id": rid, "topic": labels[gi]})
    return TrainIndex(rows, row_ids, {"topic": _spec()})


def _sampler(index: TrainIndex, operator: str, **over) -> SemanticSampler:
    settings = SamplingSettings(
        equality_strategy=over.pop("equality_strategy", "anchor_neighborhood"),
        equality_candidate_pool_same=over.pop("pool_same", 8),
        equality_candidate_pool_different=over.pop("pool_diff", 8),
        equality_positive_per_anchor=over.pop("pos", 2),
        equality_negative_per_anchor=over.pop("neg", 2),
        positive_ratio=over.pop("positive_ratio", 0.5),
        **over,
    )
    return SemanticSampler(
        _contract(PAIR if operator == "EQUALITY" else TOPIC_TASK, operator),
        _spec(),
        index,
        settings,
        ReuseTracker(),
        seed=42,
    )


# ---------------------------------------------------------------------------
# 1/2. default vs opt-in
# ---------------------------------------------------------------------------


def test_default_equality_behavior_unchanged():
    index = _index()
    sampler = _sampler(index, "EQUALITY", equality_strategy="default")
    drafts = sampler.sample(16)
    assert len(drafts) == 16
    positive = sum(1 for d in drafts if d["gold_value"])
    assert positive == round(16 * 0.5)
    # default path does not emit anchor-neighborhood stats
    assert "anchor_neighborhood_same" not in sampler.stats


def test_anchor_neighborhood_mode_is_opt_in():
    assert SamplingPolicyConfig().equality_strategy == "default"
    assert SamplingSettings().equality_strategy == "default"
    index = _index()
    sampler = _sampler(index, "EQUALITY", equality_strategy="anchor_neighborhood")
    sampler.sample(24 * 4)
    assert "anchor_neighborhood_same" in sampler.stats


# ---------------------------------------------------------------------------
# 3/4. bounded pools
# ---------------------------------------------------------------------------


def test_candidate_pool_bounds_configured():
    cfg = SamplingPolicyConfig()
    assert cfg.equality_candidate_pool_same == 8
    assert cfg.equality_candidate_pool_different == 8
    overridden = SamplingPolicyConfig(
        equality_candidate_pool_same=4, equality_candidate_pool_different=3
    )
    assert overridden.equality_candidate_pool_same == 4
    assert overridden.equality_candidate_pool_different == 3


def test_small_pool_still_satisfies_policy_via_bounded_fallback():
    # pool of 2 < needed 2 for SAME is fine (window expands deterministically).
    index = _index()
    sampler = _sampler(index, "EQUALITY", pool_same=2, pool_diff=2)
    drafts = sampler.sample(24 * 4)
    assert len(drafts) == 24 * 4


# ---------------------------------------------------------------------------
# 5/6/7/9/12. per-anchor selection
# ---------------------------------------------------------------------------


def _counts(drafts):
    anchors = Counter()
    same = Counter()
    diff = Counter()
    for d in drafts:
        a, _b = d["source_row_ids"]
        anchors[a] += 1
        if d["gold_value"]:
            same[a] += 1
        else:
            diff[a] += 1
    return anchors, same, diff


def test_exactly_two_same_and_two_different_per_anchor():
    index = _index()
    anchors = sorted(index.valid_row_ids["topic"])
    drafts = _sampler(index, "EQUALITY").sample(len(anchors) * 4)
    anchor_counts, same, diff = _counts(drafts)
    assert set(anchor_counts) == set(anchors)
    assert all(same[a] == 2 for a in anchors)
    assert all(diff[a] == 2 for a in anchors)


def test_anchor_appears_first_in_delivery_order():
    index = _index()
    anchors = sorted(index.valid_row_ids["topic"])
    drafts = _sampler(index, "EQUALITY").sample(len(anchors) * 4)
    assert all(d["source_row_ids"][0] != d["source_row_ids"][1] for d in drafts)
    # every anchor is the first element exactly once per comparison
    first = Counter(d["source_row_ids"][0] for d in drafts)
    assert set(first) == set(anchors)


def test_no_self_pairs():
    index = _index()
    drafts = _sampler(index, "EQUALITY").sample(24 * 4)
    assert all(d["source_row_ids"][0] != d["source_row_ids"][1] for d in drafts)


def test_global_positive_negative_ratio_is_50_50():
    index = _index()
    drafts = _sampler(index, "EQUALITY").sample(24 * 4)
    same = sum(1 for d in drafts if d["gold_value"])
    assert same == 48 and len(drafts) - same == 48


# ---------------------------------------------------------------------------
# 8. reverse unordered duplicate rejected
# ---------------------------------------------------------------------------


def test_reverse_unordered_pair_duplicate_rejected():
    index = _index()
    drafts = _sampler(index, "EQUALITY").sample(24 * 4)
    keys = [tuple(sorted(d["source_row_ids"])) for d in drafts]
    assert len(keys) == len(set(keys))  # no unordered duplicates at all
    # and no two records are reverses of one another
    seen = set()
    for a, b in (d["source_row_ids"] for d in drafts):
        assert (b, a) not in seen
        seen.add((a, b))


def test_reverse_duplicate_anchor_first_variant_is_unique():
    index = _index((9, 9, 9))
    drafts = _sampler(index, "EQUALITY").sample(27 * 4)
    pairs = set()
    for a, b in (d["source_row_ids"] for d in drafts):
        key = tuple(sorted((a, b)))
        assert key not in pairs
        pairs.add(key)


# ---------------------------------------------------------------------------
# 10/11. determinism + reuse balancing
# ---------------------------------------------------------------------------


def test_sha256_selection_deterministic():
    index = _index()
    a = _sampler(index, "EQUALITY").sample(24 * 4)
    b = _sampler(index, "EQUALITY").sample(24 * 4)
    assert a == b


def test_partner_reuse_balancing_deterministic():
    index = _index()
    a = _sampler(index, "EQUALITY").sample(24 * 4)
    b = _sampler(index, "EQUALITY").sample(24 * 4)
    reuse_a = Counter(d["source_row_ids"][1] for d in a)
    reuse_b = Counter(d["source_row_ids"][1] for d in b)
    assert reuse_a == reuse_b
    # balancing: no single partner dominates the neighborhood pool size
    assert max(reuse_a.values()) < 10


# ---------------------------------------------------------------------------
# 13. DIRECT strict 1:1
# ---------------------------------------------------------------------------


def test_direct_full_budget_strict_1_1():
    index = _index()
    sampler = _sampler(index, "DIRECT")
    total = len(index.valid_row_ids["topic"])
    drafts = sampler.sample(total)
    rows = [d["source_row_ids"][0] for d in drafts]
    assert len(rows) == total
    assert len(set(rows)) == total
    assert Counter(rows).most_common(1)[0][1] == 1


# ---------------------------------------------------------------------------
# 14/15. final config
# ---------------------------------------------------------------------------


def test_final_config_resolves_budget_and_total():
    raw = yaml.safe_load(
        (ROOT / "configs" / "qa_generation_vimedcss_v2.yaml").read_text(encoding="utf-8")
    )
    config = ProductionGenerationConfig.model_validate(raw)
    assert config.per_type_budget == FINAL_COUNTS
    assert sum(config.per_type_budget.values()) == FINAL_TOTAL
    settings = _sampling_settings(config)
    assert settings.equality_strategy == "anchor_neighborhood"
    assert settings.equality_candidate_pool_same == 8
    assert settings.equality_candidate_pool_different == 8
    assert settings.equality_positive_per_anchor == 2
    assert settings.equality_negative_per_anchor == 2


# ---------------------------------------------------------------------------
# 16. release builder rejects smoke as final
# ---------------------------------------------------------------------------


def test_release_builder_rejects_smoke_as_final(tmp_path: Path):
    from src.autonomous_qa.datasets.vimedcss import vimedcss_release as release

    run = tmp_path / "run"
    run.mkdir()
    rows = [
        {"qa_id": "q1", "type_id": TOPIC_TASK, "operator": "DIRECT", "dataset": "vimedcss",
         "audio_ids": ["la1"], "gold": {"kind": "field_value", "value": "x"},
         "internal": {"source_row_ids": ["s1"]}},
        {"qa_id": "q2", "type_id": COUNT_TASK, "operator": "DIRECT", "dataset": "vimedcss",
         "audio_ids": ["la2"], "gold": {"kind": "field_value", "value": 1},
         "internal": {"source_row_ids": ["s2"]}},
        {"qa_id": "q3", "type_id": PAIR, "operator": "EQUALITY", "dataset": "vimedcss",
         "audio_ids": ["la3", "lb3"], "gold": {"kind": "boolean", "value": True},
         "internal": {"source_row_ids": ["s3", "s4"]}},
    ]
    (run / "qa_internal.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    with pytest.raises(release.VimedcssReleaseError) as exc:
        release.build_logical_release(
            run_dir=run, output_dir=tmp_path / "rel", source_revision="rev",
            topics_by_row={"s1": "A", "s2": "B", "s3": "X", "s4": "X"},
            expected_counts=FINAL_COUNTS, expected_total=FINAL_TOTAL,
        )
    assert exc.value.code == "FINAL_RELEASE_COUNT_MISMATCH"


# ---------------------------------------------------------------------------
# 17/18. plan audit
# ---------------------------------------------------------------------------


def _pair_record(anchor, partner, gold):
    return {"type_id": PAIR, "source_row_ids": [anchor, partner], "gold": {"value": gold}}


def test_plan_audit_rejects_wrong_per_anchor_counts():
    lookup = {"A": "x", "B": "y", "C": "y"}
    records = [_pair_record("A", "B", False), _pair_record("A", "C", False)]
    report = audit_anchor_neighborhood_plan(
        records, type_id=PAIR, row_value_lookup=lambda r: lookup[r],
        positive_per_anchor=2, negative_per_anchor=2, expected_anchors=1,
    )
    assert report["passed"] is False
    assert any(f.startswith("PER_ANCHOR_COUNTS_BAD") for f in report["failures"])


def test_plan_audit_rejects_reverse_duplicate():
    lookup = {"A": "x", "B": "x", "C": "y", "D": "y", "E": "x", "F": "y"}
    records = [
        _pair_record("A", "B", True),
        _pair_record("B", "A", True),  # reverse duplicate
        _pair_record("A", "C", False),
        _pair_record("A", "D", False),
    ]
    report = audit_anchor_neighborhood_plan(
        records, type_id=PAIR, row_value_lookup=lambda r: lookup[r],
        positive_per_anchor=2, negative_per_anchor=2, expected_anchors=1,
    )
    assert report["passed"] is False
    assert report["duplicate_unordered_pairs"] >= 1


def test_plan_audit_rejects_gold_mismatch():
    lookup = {"A": "x", "B": "y"}
    records = [_pair_record("A", "B", True)]  # topics differ -> gold should be False
    report = audit_anchor_neighborhood_plan(
        records, type_id=PAIR, row_value_lookup=lambda r: lookup[r],
        positive_per_anchor=0, negative_per_anchor=1, expected_anchors=1,
    )
    assert any(f.startswith("GOLD_MISMATCH") for f in report["failures"])


# ---------------------------------------------------------------------------
# feasibility + direct audit
# ---------------------------------------------------------------------------


def test_feasibility_reports_feasible_for_synthetic_index():
    index = _index()
    report = anchor_neighborhood_feasibility(
        index, "topic", positive_per_anchor=2, negative_per_anchor=2
    )
    assert report["feasible"] is True
    assert report["requested_same_pairs"] == 2 * len(index.valid_row_ids["topic"])
    assert report["full_cartesian_pair_enumeration"] is False


def test_direct_coverage_audit_detects_gap():
    records = [{"type_id": TOPIC_TASK, "source_row_ids": ["r1"]}]
    report = audit_direct_coverage(records, type_id=TOPIC_TASK, expected_rows=2)
    assert report["passed"] is False


# ---------------------------------------------------------------------------
# 19/20. no network / no LLM
# ---------------------------------------------------------------------------


def test_sampling_and_audit_make_no_network_calls(monkeypatch):
    def forbidden(*_a, **_k):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    index = _index()
    drafts = _sampler(index, "EQUALITY").sample(24 * 4)
    audit_anchor_neighborhood_plan(
        [
            {"type_id": PAIR, "source_row_ids": d["source_row_ids"], "gold": {"value": d["gold_value"]}}
            for d in drafts
        ],
        type_id=PAIR,
        row_value_lookup=lambda r: index.normalize("topic", index.row_by_id[r]["topic"]),
        positive_per_anchor=2, negative_per_anchor=2, expected_anchors=24,
    )

"""v3 discovery-loop DoD: prompt order, dynamic few-shots, verdict shape.

Covers §21 items not pinned elsewhere:
- Style round 2 input = README + fixed seeds + round-1 accepts.
- Duplicates/rejected never feed back as dynamic few-shots.
- Few-shots sit AFTER dataset/schema instructions.
- Quality verdict has exactly {accepted_new, duplicates, rejected}.
- Paraphrase-equivalent wording is kept, never a duplicate type.
- Duplicate_of always points at an already-approved type.
- Gate runs before paraphrase: judged ids resolve within the round.
"""

import json

from src.autonomous_qa.core.loop import apply_accepted_types, build_type_fewshot
from src.autonomous_qa.language.question_style import build_question_style_prompt
from src.autonomous_qa.core.schemas import QualityRoundOutput
from src.common.io import load_fixture


def _round_fixture(r, kind):
    return load_fixture(f"loop/round_{r:02d}/{kind}")


def test_style_prompt_order_schema_then_seeds_then_dynamic(isolated_root):
    dynamic = [{"name": "Speaker Verification", "uses": ["audio", "speakerID"],
                "goal": "Determine whether two clips share a speaker.",
                "answer_rule": "same speakerID => yes, otherwise no"}]
    prompt = build_question_style_prompt("README-SCHEMA-MARKER", dynamic, 2)
    i_readme = prompt.index("README-SCHEMA-MARKER")
    i_seed = prompt.index("FIXED SEED")
    i_dyn = prompt.index("ALREADY COVERED")
    assert i_readme < i_seed < i_dyn


def test_style_prompt_without_dynamic_has_no_covered_section(isolated_root):
    prompt = build_question_style_prompt("README-SCHEMA-MARKER", [], 1)
    assert "ALREADY COVERED" not in prompt
    assert "README-SCHEMA-MARKER" in prompt


def test_quality_verdict_has_only_three_buckets():
    assert set(QualityRoundOutput.model_fields) == {
        "round", "accepted_new", "duplicates", "rejected"}


def _apply_round(pool, r):
    """Mirror the pipeline: gate -> post-paraphrase filter -> strict apply."""
    from src.autonomous_qa.core.loop import filter_paraphrases_for_judging
    from src.autonomous_qa.compiler.pipeline_runner import load_sample
    from src.autonomous_qa.core.validity import gate_round
    style = _round_fixture(r, "style")
    tpl = _round_fixture(r, "template")
    para = _round_fixture(r, "paraphrase")
    verdict = _round_fixture(r, "quality")
    rows = load_sample("vimd")
    gate = gate_round(style["question_types"], tpl["templates"], dataset_rows=rows)
    filt = filter_paraphrases_for_judging(
        gate["valid_templates"], para["paraphrases"])
    texts = {t["template_id"]: t["text"]
             for t in gate["valid_templates"]}
    texts.update(filt["texts"])
    owners = {t["template_id"]: t["question_type_id"]
              for t in gate["valid_templates"]}
    owners.update(filt["owners"])
    by_id = {t["id"]: t for t in style["question_types"]}
    return apply_accepted_types(pool, by_id, texts, verdict, r, owners)


def test_dynamic_fewshots_accumulate_only_accepted(isolated_root):
    pool = []
    for r in (1, 2):
        _apply_round(pool, r)

    fewshots = build_type_fewshot(pool)
    names = {f["name"] for f in fewshots}
    # Round-1 accepts feed round 2; duplicates/rejected never do.
    assert "Speaker Gender Recognition" in names
    assert "Province Identification" in names
    assert len(fewshots) == 7
    for f in fewshots:
        assert set(f) == {"name", "uses", "goal", "answer_rule"}
    # Exact executable-signature duplicates never become new entries:
    # round-2 rephrases of approved capabilities are absent as new pool
    # items holding the round-2 type ids.
    dup_names = [d["type_id"] for d in
                 _round_fixture(2, "quality")["duplicates"]]
    assert dup_names == []  # structural dups no longer reach Quality
    assert not any(t["type_id"].startswith("QS_R2_")
                   for t in pool if t["name"] in {
                       "Speaker Sex Identification",
                       "Regional Accent Classification"})


def test_paraphrase_equivalence_is_kept_not_duplicated(isolated_root):
    """Regression: same-semantics wording must survive as kept templates."""
    pool = []
    verdict = _round_fixture(1, "quality")
    _apply_round(pool, 1)

    kept_ids = [k["template_id"] for t in pool for k in t["kept_templates"]]
    # Paraphrase variants (same semantics, new wording) are kept...
    assert "T_R1_01_01_P01" in kept_ids
    # ...and no wording-level duplicate verdict exists in round 1.
    assert verdict["duplicates"] == []


def test_duplicates_point_at_approved_types(isolated_root):
    approved = set()
    for r in (1, 2, 3):
        verdict = _round_fixture(r, "quality")
        for d in verdict["duplicates"]:
            assert d["duplicate_of"] in approved, (r, d)
        for a in verdict["accepted_new"]:
            approved.add(a["type_id"])


def test_gate_runs_before_paraphrase_judgement(isolated_root):
    """Every paraphrase source and kept id resolves within its round."""
    from src.autonomous_qa.compiler.pipeline_runner import load_sample
    from src.autonomous_qa.core.validity import gate_round
    rows = load_sample("vimd")
    for r in (1, 2, 3):
        style = _round_fixture(r, "style")
        tpl = _round_fixture(r, "template")
        para = _round_fixture(r, "paraphrase")
        verdict = _round_fixture(r, "quality")
        gate = gate_round(style["question_types"], tpl["templates"], dataset_rows=rows)
        valid_ids = {t["template_id"] for t in gate["valid_templates"]}
        for p in para["paraphrases"]:
            assert p["source_template_id"] in valid_ids, (r, p)
        texts = ({t["template_id"] for t in tpl["templates"]}
                 | {p["template_id"] for p in para["paraphrases"]})
        for a in verdict["accepted_new"]:
            assert set(a["keep_template_ids"]) <= texts, (r, a)


def test_style_may_propose_types_absent_from_seeds(isolated_root):
    """Accepted mock types are not copies of the fixed seed analogies."""
    seed_text = (isolated_root / "fewshots" /
                 "question_style_seed.jsonl").read_text(encoding="utf-8")
    pool_names = set()
    pool = []
    for r in (1, 2):
        _apply_round(pool, r)
    pool_names = {t["name"] for t in pool}
    assert "Speaker Gender Recognition" in pool_names
    # ...while none of them are spelled out in the fixed seeds.
    assert "Speaker Gender Recognition" not in seed_text
    assert "Speech Emotion Recognition" not in pool_names  # seed-only type
    assert "Fine-to-Coarse Attribute Reasoning" not in pool_names

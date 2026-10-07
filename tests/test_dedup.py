"""Fix 3 tests: post-paraphrase exact dedup, per type (31-40)."""

from src.autonomous_qa.core.loop import (
    filter_paraphrases_for_judging,
    normalize_question_text,
)


def _base(tid, text, owner="QS_1"):
    return {"template_id": tid, "question_type_id": owner, "text": text}


def _para(pid, src, text):
    return {"template_id": pid, "source_template_id": src, "text": text}


def test_whitespace_only_difference_collapses():
    out = filter_paraphrases_for_judging(
        [_base("B", "Hãy xác định [KEY] của người nói.")],
        [_para("P1", "B",
               "  Hãy xác định   [KEY]  của người nói.  ")])
    assert out["surviving"] == []
    assert [(d["template_id"], d["duplicate_of"]) for d in
            out["dropped_exact"]] == [("P1", "B")]


def test_capitalization_only_difference_collapses_via_casefold():
    out = filter_paraphrases_for_judging(
        [_base("B", "Hãy xác định [KEY] của người nói.")],
        [_para("P1", "B", "HÃY XÁC ĐỊNH [KEY] CỦA NGƯỜI NÓI.")])
    assert out["surviving"] == []
    assert out["dropped_exact"][0]["duplicate_of"] == "B"


def test_unicode_nfc_equivalent_text_collapses():
    import unicodedata
    base = "Hãy xác định [KEY] của người nói."
    nfd_variant = unicodedata.normalize("NFD", base)
    assert nfd_variant != base  # sanity: encodings really differ
    out = filter_paraphrases_for_judging(
        [_base("B", base)],
        [_para("P1", "B", nfd_variant)])
    assert out["surviving"] == []
    assert out["dropped_exact"][0]["duplicate_of"] == "B"


def test_punctuation_difference_does_not_collapse():
    out = filter_paraphrases_for_judging(
        [_base("B", "Hai người nói này có cùng giới tính không?")],
        [_para("P1", "B",
               "Hai người nói này có cùng giới tính không.")])
    assert [p["template_id"] for p in out["surviving"]] == ["P1"]
    assert out["dropped_exact"] == []


def test_paraphrase_identical_to_base_is_dropped():
    out = filter_paraphrases_for_judging(
        [_base("B", "Nội dung là gì?")],
        [_para("P1", "B", "Nội dung là gì?"),
         _para("P2", "B", "Nội dung khác.")])
    assert [p["template_id"] for p in out["surviving"]] == ["P2"]
    assert [(d["template_id"], d["duplicate_of"]) for d in
            out["dropped_exact"]] == [("P1", "B")]


def test_second_identical_paraphrase_dropped_first_survives():
    out = filter_paraphrases_for_judging(
        [_base("B", "Nội dung là gì?")],
        [_para("P1", "B", "Xin cho biết nội dung."),
         _para("P2", "B", "Xin cho biết nội dung.")])
    assert [p["template_id"] for p in out["surviving"]] == ["P1"]
    assert [(d["template_id"], d["duplicate_of"]) for d in
            out["dropped_exact"]] == [("P2", "P1")]


def test_identical_text_across_types_is_not_collapsed():
    # Same generic text kept as a base template under two types: each
    # type judges only its own pool. P1 drops against its OWN base B1;
    # QS_B is unaffected and its distinct paraphrase survives.
    out = filter_paraphrases_for_judging(
        [_base("B1", "Hãy xác định [KEY]?", owner="QS_A"),
         _base("B2", "Hãy xác định [KEY]?", owner="QS_B")],
        [_para("P1", "B1", "Hãy xác định [KEY]?"),
         _para("P2", "B2", "Văn bản hoàn toàn khác.")])
    assert [p["template_id"] for p in out["surviving"]] == ["P2"]
    assert [(d["template_id"], d["duplicate_of"], d["type_id"])
            for d in out["dropped_exact"]] == [("P1", "B1", "QS_A")]
    assert out["owners"] == {"P2": "QS_B"}


def test_dropped_ids_not_sent_to_quality():
    out = filter_paraphrases_for_judging(
        [_base("B", "Nội dung là gì?")],
        [_para("P1", "B", "Nội dung là gì?"),
         _para("P2", "B", "Nội dung [VALUE].")])
    surviving_ids = {p["template_id"] for p in out["surviving"]}
    assert surviving_ids == set()
    assert {d["template_id"] for d in out["dropped_exact"]} == {"P1"}
    assert [(d["template_id"], d["reason"]) for d in
            out["dropped_leak"]] == [("P2", "answer_leakage")]
    assert "P1" not in out["texts"] and "P2" not in out["texts"]


def test_exact_duplicate_count_is_auditable():
    out = filter_paraphrases_for_judging(
        [_base("B", "t?")],
        [_para("P1", "B", "t?"), _para("P2", "B", "t?"),
         _para("P3", "B", "other")])
    assert len(out["dropped_exact"]) == 2
    assert all(set(d) == {"template_id", "duplicate_of", "type_id"}
               for d in out["dropped_exact"])


def test_new_type_rate_ignores_template_duplicate_counts():
    from src.autonomous_qa.core.loop import LoopController
    ctrl = LoopController(
        max_rounds=3, min_rate=0.15, saturation_patience=2,
        immediate_stop_if_zero_new=True)
    # Rate uses type counts only; template-level drops never enter.
    out = ctrl.register_round(1, accepted_new=2, generated=8)
    assert out["new_type_rate"] == 0.25
    assert out["stop"] is False


def test_normalize_question_text_contract():
    assert normalize_question_text("  A  B\nC\tD  ") == "a b c d"
    assert normalize_question_text("ĐOẠN ÂM THANH") == "đoạn âm thanh"

"""Quality prompt semantics: type-level verdicts + precision-first keeps.

Semantic substring checks only — never full prompt byte comparison.
"""

from src.common.io import load_prompt


def _prompt():
    return load_prompt("quality")


def test_rejected_is_type_level_only(isolated_root):
    assert "those arrays hold QUESTION TYPE ids only" in _prompt()


def test_bad_templates_omitted_not_rejected(isolated_root):
    text = _prompt()
    assert "TEMPLATE OMISSION IS THE ONLY TEMPLATE VERDICT" in text
    assert "leaving its" in text and "out of keep_template_ids" in text


def test_never_template_ids_in_type_id_fields(isolated_root):
    text = _prompt()
    assert "NEVER put template IDs into `rejected` or" in text
    assert "`duplicates`" in text


def test_transcription_must_request_actual_transcription(isolated_root):
    text = _prompt()
    assert "must ask for the actual spoken words" in text
    assert "Tóm tắt nội dung đoạn âm thanh." in text


def test_province_dialect_must_not_imply_hometown(isolated_root):
    text = _prompt()
    assert "birthplace, hometown, residence" in text
    assert "Người nói quê ở tỉnh nào?" in text


def test_province_region_granularity_preserved(isolated_root):
    text = _prompt()
    assert "vùng miền" in text
    assert "province-level" in text


def test_verification_asks_same_speaker_explicitly(isolated_root):
    text = _prompt()
    assert "same speaker identity explicitly" in text
    assert "giống nhau" in text


def test_strict_equality_not_weakened_to_similarity(isolated_root):
    text = _prompt()
    assert "tương đồng" in text
    assert "mere similarity" in text or "merely" in text


def test_all_bad_templates_means_reject_type(isolated_root):
    text = _prompt()
    assert "If no template survives" in text
    assert "instead of accepted_new with an empty list" in text


def test_some_good_templates_means_accept_and_keep(isolated_root):
    text = _prompt()
    assert "LEVEL 2" in text
    assert "omit its id from keep_template_ids and move on" in text

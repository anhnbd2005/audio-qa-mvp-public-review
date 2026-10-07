"""Fix 2 prompt tests: Quality selection instructions present (25-30).

Semantic substring checks only — never full prompt byte comparison.
"""

from src.common.io import load_prompt


def _prompt():
    return load_prompt("quality")


def test_prompt_rejects_keep_everything_default(isolated_root):
    assert "Do NOT keep every candidate by default" in _prompt()


def test_prompt_preserves_exact_dataset_semantics(isolated_root):
    text = _prompt()
    assert "never infer a stronger real-world fact than" in text
    assert "dataset label supports" in text


def test_prompt_forbids_birthplace_upgrade(isolated_root):
    text = _prompt()
    for word in ("birthplace", "hometown", "residence"):
        assert word in text, word
    assert "unless the README" in text


def test_prompt_forbids_province_region_drift(isolated_root):
    text = _prompt()
    assert "vùng miền" in text
    assert "province-level" in text


def test_prompt_forbids_answer_form_drift(isolated_root):
    text = _prompt()
    assert "yes/no" in text
    assert "Bạn có thể phiên âm đoạn âm thanh này không?" in text


def test_prompt_paraphrases_are_not_duplicate_types(isolated_root):
    text = _prompt()
    assert "equivalent paraphrase wordings" in text
    assert "never evidence of duplication" in text

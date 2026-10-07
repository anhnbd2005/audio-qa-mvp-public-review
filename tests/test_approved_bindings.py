"""Approved-binding precedence + accepted-only commit (§47-48, tests 35-42)."""

from src.autonomous_qa.core.loop import commit_approved_bindings, resolve_effective_binding


def _type(tid, answer, uses=("audio", "gender")):
    return {"id": tid, "name": "N", "goal": "g", "uses": list(uses),
            "input_count": 1, "answer_rule": "r", "answer": answer}


def _item(tid, kept):
    return {"type_id": tid,
            "kept_templates": [{"template_id": k, "text": "t"} for k in kept]}


def _keyed_item(tid, kept):
    return {"type_id": tid,
            "kept_templates": [{"template_id": k, "text": "x [KEY] y"}
                               for k in kept]}


def _types():
    return {
        "QS_G": _type("QS_G", {"kind": "field_value", "key": "gender"}),
        "QS_P": _type("QS_P", {"kind": "field_value", "key": "province"},
                      uses=("audio", "province")),
    }


def test_effective_prefers_approved_over_candidate():
    assert resolve_effective_binding(
        "region", {"region": "vùng phương ngữ"},
        {"region": "vùng giọng"}) == "vùng phương ngữ"


def test_effective_falls_back_to_candidate():
    assert resolve_effective_binding(
        "gender", {}, {"gender": "giới tính"}) == "giới tính"


def test_effective_none_when_nowhere():
    assert resolve_effective_binding("gender", {}, {}) is None
    assert resolve_effective_binding(None, {"gender": "x"}, {}) is None


def test_auto_rejected_type_commits_nothing():
    approved = {}
    out = commit_approved_bindings(approved, {"gender": "giới tính"}, [],
                                   _types())
    assert out == {"committed": [], "ignored": []}
    assert approved == {}


def test_quality_rejected_type_commits_nothing():
    # Only accepted_items are passed in; rejected types never reach commit.
    approved = {}
    out = commit_approved_bindings(approved, {"gender": "giới tính"}, [],
                                   _types())
    assert approved == {}


def test_duplicate_type_commits_nothing_new():
    approved = {"gender": "giới tính"}
    out = commit_approved_bindings(approved, {"gender": "OTHER"}, [],
                                   _types())
    assert approved == {"gender": "giới tính"}
    assert out == {"committed": [], "ignored": []}


def test_slot_free_only_type_commits_nothing():
    approved = {}
    item = {"type_id": "QS_G",
            "kept_templates": [{"template_id": "T1", "text": "plain?"}]}
    out = commit_approved_bindings(
        approved, {"gender": "giới tính"}, [item], _types())
    assert out == {"committed": [], "ignored": []}
    assert approved == {}


def test_kept_key_template_commits_required_field():
    approved = {}
    out = commit_approved_bindings(
        approved, {"gender": "giới tính"},
        [_keyed_item("QS_G", ["T1"])], _types())
    assert out["committed"] == ["gender"]
    assert approved == {"gender": "giới tính"}


def test_only_slot_free_kept_commits_nothing():
    # Type accepted, but Quality kept only the slot-free template while a
    # [KEY] template was offered and omitted.
    approved = {}
    item = {"type_id": "QS_G",
            "kept_templates": [{"template_id": "T1", "text": "plain?"}]}
    out = commit_approved_bindings(
        approved, {"gender": "giới tính"}, [item], _types())
    assert approved == {}
    assert out["committed"] == []


def test_kept_and_omitted_key_template_commits_once():
    approved = {}
    item = {"type_id": "QS_G",
            "kept_templates": [{"template_id": "T1", "text": "x [KEY] y"}]}
    out = commit_approved_bindings(
        approved, {"gender": "giới tính"}, [item], _types())
    assert out["committed"] == ["gender"]
    assert approved == {"gender": "giới tính"}


def test_unkept_templates_cannot_create_bindings():
    # Commit only inspects kept templates of accepted items: with no
    # effective value anywhere, nothing is committed.
    approved = {}
    out = commit_approved_bindings(
        approved, {"province": "tỉnh thành"},
        [_keyed_item("QS_G", ["T1"])], _types())
    assert approved == {}
    assert out == {"committed": [], "ignored": []}

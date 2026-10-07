"""Phase 4.2P — global language-registry promotion governance tests.

All APPLY/rollback tests operate entirely on ``tmp_path / "resources"``. An
autouse guard proves the real canonical resources are never mutated.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from src.autonomous_qa.certification.language_registry_promotion import (
    LanguageRegistryPromotionBundle,
    LanguageRegistryPromotionError,
    apply_language_registry_promotion,
    prepare_language_registry_promotion,
)
from src.autonomous_qa.compiler.canonical_resources import (
    RESOURCE_ROOT,
    ProductionContract,
    PromotionManifest,
)
from src.autonomous_qa.compiler.semantic_field_specs import (
    SemanticFieldSpec,
    load_migration_sidecar,
)
from src.autonomous_qa.language import language_preflight as preflight
from src.autonomous_qa.language.candidate_registry import (
    CandidateRegistryError,
    build_candidate_registry_from_paths,
    validate_candidate_capability_resource,
)
from src.autonomous_qa.language.language_preflight import AcceptedLanguageType, run_preflight
from src.autonomous_qa.language.language_quality import (
    ProductionLanguageRegistry,
    check_runtime_coverage,
    compatible_registry_entries,
)
from src.autonomous_qa.language.template_renderer import canonical_hash
from src.common.config import ROOT

_REAL_FILES = [
    RESOURCE_ROOT / "language" / "production_registry.json",
    *sorted((RESOURCE_ROOT / "production").glob("*.json")),
]
_REAL_BEFORE = {p: p.read_bytes() for p in _REAL_FILES}


@pytest.fixture(autouse=True)
def _guard_real_resources():
    yield
    for path, before in _REAL_BEFORE.items():
        assert path.read_bytes() == before, f"REAL CANONICAL RESOURCE MUTATED: {path}"


# ---------------------------------------------------------------------------
# synthetic tree
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _registry(entries, version="toy_base_v1", language="vi") -> ProductionLanguageRegistry:
    reg = ProductionLanguageRegistry(
        language=language,
        version=version,
        schema_version=1,
        template_library_version="toy",
        template_library_hash="toy_template",
        paraphrase_library_version="toy",
        paraphrase_library_hash="toy_para",
        quality_model="toy",
        quality_call_ids=[],
        registry_hash="PENDING",
        entries=entries,
    )
    reg.registry_hash = reg.computed_hash()
    return reg


def _toy_capability_resource(language="vi") -> dict:
    return {
        "schema_version": 1,
        "capability_set_id": "toy_caps",
        "language": language,
        "entries": [
            {
                "capability_id": "cap_toy_dir_cat_utt",
                "operator": "DIRECT",
                "semantic_class": "categorical_attribute",
                "answer_kind": "field_value",
                "pattern": "[ATTRIBUTE_PHRASE] của đoạn âm thanh là gì?",
                "required_slots": ["[ATTRIBUTE_PHRASE]"],
                "optional_slots": [],
                "unit_policy": "any",
                "match_policy": ["exact"],
                "entity_scopes": ["utterance"],
                "entity_reference_owner": "literal",
            }
        ],
    }


def _toy_accepted() -> AcceptedLanguageType:
    return AcceptedLanguageType(
        dataset_type_id="toy_topic",
        operator="DIRECT",
        semantic_class="categorical_attribute",
        semantic_field="topic",
        answer_kind="field_value",
        audio_input_count=1,
        logical_context_inputs=0,
        phrase_bindings={
            "entity_scope": "utterance",
            "entity_phrase": "đoạn âm thanh",
            "attribute_phrase": "chủ đề",
            "content_phrase": None,
            "value_phrase": "chủ đề",
            "unit": None,
            "target_quote_style": "plain",
        },
    )


def _contract(dataset_id: str, registry_hash: str) -> ProductionContract:
    contract = ProductionContract(
        schema_version=1,
        dataset_id=dataset_id,
        dataset_spec_hash="d" * 64,
        semantic_catalog_hash="c" * 64,
        comparator_registry_hash="r" * 64,
        comparator_referenced_hashes={},
        language_registry_hash=registry_hash,
        active_semantic_types=("toy_topic",),
        promotion_evidence={},
        status="PROMOTED",
        migration_source="toy",
        promotion_fingerprint="",
    )
    return contract.model_copy(update={"promotion_fingerprint": contract.fingerprint()})


def _manifest(dataset_id: str, registry_hash: str, contract: ProductionContract) -> PromotionManifest:
    manifest = PromotionManifest(
        schema_version=1,
        dataset_id=dataset_id,
        status="PROMOTED",
        migration_source="toy",
        dataset_spec_hash="d" * 64,
        semantic_catalog_hash="c" * 64,
        comparator_registry_hash="r" * 64,
        language_registry_hash=registry_hash,
        production_contract_hash=contract.fingerprint(),
        promotion_evidence={},
        promoted_semantic_types=("toy_topic",),
        source_revision="toy_rev",
        allowed_splits=("train",),
        promotion_fingerprint="",
    )
    return manifest.model_copy(update={"promotion_fingerprint": manifest.fingerprint()})


def build_toy_tree(tmp: Path) -> dict:
    res = tmp / "resources"
    (res / "language").mkdir(parents=True)
    (res / "production").mkdir(parents=True)
    base = _registry([])
    _write_json(res / "language" / "production_registry.json", base.model_dump(mode="json"))
    _write_json(res / "language" / "candidate_capabilities.json", _toy_capability_resource())

    # toy_a pins the base hash; toy_b is pre-existing drift.
    a = _contract("toy_a", base.registry_hash)
    _write_json(res / "production" / "toy_a.json", a.model_dump(mode="json"))
    _write_json(
        res / "production" / "toy_a.promotion.json",
        _manifest("toy_a", base.registry_hash, a).model_dump(mode="json"),
    )
    b = _contract("toy_b", "f" * 64)
    _write_json(res / "production" / "toy_b.json", b.model_dump(mode="json"))
    _write_json(
        res / "production" / "toy_b.promotion.json",
        _manifest("toy_b", "f" * 64, b).model_dump(mode="json"),
    )
    return {"resource_root": res, "base": base}


def build_multi_dataset_tree(tmp: Path, dataset_ids: list[str]) -> dict:
    """Synthetic tree with arbitrary dataset IDs, all pinning the base hash."""
    res = tmp / "resources"
    (res / "language").mkdir(parents=True)
    (res / "production").mkdir(parents=True)
    base = _registry([])
    _write_json(res / "language" / "production_registry.json", base.model_dump(mode="json"))
    _write_json(res / "language" / "candidate_capabilities.json", _toy_capability_resource())
    for ds in dataset_ids:
        contract = _contract(ds, base.registry_hash)
        _write_json(res / "production" / f"{ds}.json", contract.model_dump(mode="json"))
        _write_json(
            res / "production" / f"{ds}.promotion.json",
            _manifest(ds, base.registry_hash, contract).model_dump(mode="json"),
        )
    return {"resource_root": res, "base": base}


def prepare_toy(tmp: Path, tree: dict, **overrides) -> LanguageRegistryPromotionBundle:
    res = tree["resource_root"]
    kwargs = dict(
        resource_root=res,
        candidate_capabilities_path=res / "language" / "candidate_capabilities.json",
        scratch_root=tmp / "scratch",
        fresh_accepted_types=[_toy_accepted()],
        fresh_dataset_label="toy_fresh",
        expected_candidate_hash=None,
        dataset_accepted_resolver=lambda ds: [_toy_accepted()],
    )
    kwargs.update(overrides)
    return prepare_language_registry_promotion(**kwargs)


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# ---------------------------------------------------------------------------
# PREPARE + certification
# ---------------------------------------------------------------------------


def test_natural_prepare_apply_ready_on_synthetic_tree(tmp_path: Path):
    tree = build_toy_tree(tmp_path)
    bundle = prepare_toy(tmp_path, tree)
    assert bundle.apply_ready is True
    assert bundle.certification_evidence.certification_status == "CERTIFIED"
    assert bundle.certification_evidence.failures == ()
    assert bundle.candidate_registry_hash != bundle.base_registry_hash
    assert bundle.promoted_registry_hash != bundle.candidate_registry_hash
    assert set(bundle.added_entry_ids) == {"cap_toy_dir_cat_utt"}


def test_candidate_certification_required(tmp_path: Path):
    tree = build_toy_tree(tmp_path)
    bundle = prepare_toy(tmp_path, tree, expected_candidate_hash="deadbeef")
    assert bundle.apply_ready is False
    assert bundle.certification_evidence.certification_status == "NOT_CERTIFIED"
    assert "CANDIDATE_HASH_MISMATCH" in bundle.certification_evidence.failures


def test_generic_discovery_and_classification(tmp_path: Path):
    tree = build_toy_tree(tmp_path)
    bundle = prepare_toy(tmp_path, tree)
    found = {f.dataset_id: f for f in bundle.affected_resources if f.kind == "contract"}
    assert set(found) == {"toy_a", "toy_b"}
    assert found["toy_a"].classification == "PINNED_BASE"
    assert found["toy_b"].classification == "PRE_EXISTING_DRIFT"
    manifests = {f.dataset_id for f in bundle.affected_resources if f.kind == "manifest"}
    assert manifests == {"toy_a", "toy_b"}


def test_contract_rebind_allowed_fields_only(tmp_path: Path):
    tree = build_toy_tree(tmp_path)
    bundle = prepare_toy(tmp_path, tree)
    for plan in bundle.contract_rebind_plan:
        assert set(plan.changed_fields) <= {
            "language_registry_hash",
            "promotion_fingerprint",
        }
        assert plan.new_language_registry_hash == bundle.promoted_registry_hash
        reloaded = ProductionContract.model_validate(plan.new_payload)
        assert reloaded.promotion_fingerprint == reloaded.fingerprint()


def test_manifest_rebind_allowed_fields_only(tmp_path: Path):
    tree = build_toy_tree(tmp_path)
    bundle = prepare_toy(tmp_path, tree)
    for plan in bundle.manifest_rebind_plan:
        assert set(plan.changed_fields) <= {
            "language_registry_hash",
            "production_contract_hash",
            "promotion_fingerprint",
        }
        reloaded = PromotionManifest.model_validate(plan.new_payload)
        assert reloaded.promotion_fingerprint == reloaded.fingerprint()
        assert reloaded.language_registry_hash == bundle.promoted_registry_hash


def test_certification_datasets_discovered_generically(tmp_path: Path):
    tree = build_multi_dataset_tree(tmp_path, ["alpha", "beta", "gamma"])
    bundle = prepare_toy(tmp_path, tree)
    assert {
        r.dataset_id for r in bundle.certification_evidence.dataset_results
    } == {"alpha", "beta", "gamma"}
    # discovery drives classification too
    assert {
        f.dataset_id for f in bundle.affected_resources if f.kind == "contract"
    } == {"alpha", "beta", "gamma"}


def test_certification_baseline_is_resource_root_hermetic(tmp_path: Path, monkeypatch):
    import src.autonomous_qa.language.language_preflight as pf

    real_registry = (RESOURCE_ROOT / "language" / "production_registry.json").resolve()
    original = pf.load_language_registry

    def guarded(path, *args, **kwargs):
        if Path(path).resolve() == real_registry:
            raise AssertionError(
                "certification read the real ROOT canonical language registry"
            )
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pf, "load_language_registry", guarded)

    tree = build_toy_tree(tmp_path)  # synthetic base has ZERO entries
    speaker = AcceptedLanguageType(
        dataset_type_id="toy_speaker",
        operator="DIRECT",
        semantic_class="categorical_attribute",
        semantic_field="region",
        answer_kind="field_value",
        audio_input_count=1,
        logical_context_inputs=0,
        phrase_bindings={
            "entity_scope": "speaker",
            "entity_phrase": "người nói",
            "attribute_phrase": "vùng",
            "content_phrase": None,
            "value_phrase": "vùng",
            "unit": None,
            "target_quote_style": "plain",
        },
    )
    bundle = prepare_toy(
        tmp_path,
        tree,
        fresh_accepted_types=[speaker],
        dataset_accepted_resolver=lambda ds: [speaker],
    )
    results = {r.dataset_id: r for r in bundle.certification_evidence.dataset_results}
    assert set(results) == {"toy_a", "toy_b"}
    for result in results.values():
        # Synthetic base has no compatible entries: baseline MUST fail. If it
        # passed, the certification leaked the real ROOT registry.
        assert result.baseline_status != "PREFLIGHT_PASS"
        assert result.baseline_blocking > 0


def test_certification_accepted_types_unavailable_fails_closed(tmp_path: Path):
    tree = build_toy_tree(tmp_path)

    def _no_resolver(dataset_id):
        raise KeyError(dataset_id)

    bundle = prepare_toy(tmp_path, tree, dataset_accepted_resolver=_no_resolver)
    assert bundle.apply_ready is False
    assert any(
        f.startswith("CERTIFICATION_ACCEPTED_TYPES_UNAVAILABLE:")
        for f in bundle.certification_evidence.failures
    )


def test_invalid_resource_fails_closed(tmp_path: Path):
    tree = build_toy_tree(tmp_path)
    (tree["resource_root"] / "production" / "broken.json").write_text(
        json.dumps({"dataset_id": "broken", "language_registry_hash": "x"}), encoding="utf-8"
    )
    bundle = prepare_toy(tmp_path, tree)
    invalid = [f for f in bundle.affected_resources if f.classification == "INVALID_RESOURCE"]
    assert invalid
    assert bundle.apply_ready is False


# ---------------------------------------------------------------------------
# capability resource + sidecar validation
# ---------------------------------------------------------------------------


def _validated(data, **kwargs):
    return validate_candidate_capability_resource(data, expected_language="vi", **kwargs)


def test_candidate_language_mismatch_blocks():
    data = _toy_capability_resource(language="en")
    with pytest.raises(CandidateRegistryError) as exc:
        _validated(data)
    assert exc.value.code == "LANGUAGE_CAPABILITY_LANGUAGE_MISMATCH"


def test_schema_version_unsupported_blocks():
    data = _toy_capability_resource()
    data["schema_version"] = 2
    with pytest.raises(CandidateRegistryError) as exc:
        _validated(data)
    assert exc.value.code == "CANDIDATE_CAPABILITIES_SCHEMA_UNSUPPORTED"


def test_schema_version_wrong_type_blocks():
    data = _toy_capability_resource()
    data["schema_version"] = "1"
    with pytest.raises(CandidateRegistryError) as exc:
        _validated(data)
    assert exc.value.code == "CANDIDATE_CAPABILITIES_SCHEMA_UNSUPPORTED"


def test_empty_capability_set_id_blocks():
    data = _toy_capability_resource()
    data["capability_set_id"] = ""
    with pytest.raises(CandidateRegistryError):
        _validated(data)


def test_non_string_language_blocks():
    data = _toy_capability_resource()
    data["language"] = ["vi"]
    with pytest.raises(CandidateRegistryError):
        _validated(data)


def test_duplicate_capability_id_blocks():
    data = _toy_capability_resource()
    data["entries"].append(dict(data["entries"][0]))
    with pytest.raises(CandidateRegistryError) as exc:
        _validated(data)
    assert exc.value.code == "DUPLICATE_LANGUAGE_ENTRY_ID"


def test_invalid_semantic_class_blocks():
    data = _toy_capability_resource()
    data["entries"][0]["semantic_class"] = "not_a_class"
    with pytest.raises(CandidateRegistryError):
        _validated(data)


def test_invalid_answer_kind_blocks():
    data = _toy_capability_resource()
    data["entries"][0]["answer_kind"] = "not_a_kind"
    with pytest.raises(CandidateRegistryError):
        _validated(data)


def test_invalid_match_policy_blocks():
    data = _toy_capability_resource()
    data["entries"][0]["match_policy"] = ["not_a_policy"]
    with pytest.raises(CandidateRegistryError):
        _validated(data)


def test_empty_entity_scopes_blocks():
    data = _toy_capability_resource()
    data["entries"][0]["entity_scopes"] = []
    with pytest.raises(CandidateRegistryError) as exc:
        _validated(data)
    assert exc.value.code == "CANDIDATE_CAPABILITY_ENTITY_SCOPES_EMPTY"


def test_required_slots_non_list_blocks():
    data = _toy_capability_resource()
    data["entries"][0]["required_slots"] = "[ATTRIBUTE_PHRASE]"
    with pytest.raises(CandidateRegistryError):
        _validated(data)


def test_required_slot_absent_from_pattern_blocks():
    data = _toy_capability_resource()
    data["entries"][0]["required_slots"] = ["[MISSING_SLOT]"]
    with pytest.raises(CandidateRegistryError) as exc:
        _validated(data)
    assert exc.value.code == "CANDIDATE_CAPABILITY_REQUIRED_SLOT_ABSENT"


def test_required_optional_slot_overlap_blocks():
    data = _toy_capability_resource()
    data["entries"][0]["optional_slots"] = ["[ATTRIBUTE_PHRASE]"]
    with pytest.raises(CandidateRegistryError) as exc:
        _validated(data)
    assert exc.value.code == "CANDIDATE_CAPABILITY_SLOT_OVERLAP"


def test_malformed_capability_resource_blocks():
    data = _toy_capability_resource()
    del data["entries"][0]["pattern"]
    with pytest.raises(CandidateRegistryError) as exc:
        _validated(data)
    assert exc.value.code == "CANDIDATE_CAPABILITIES_MALFORMED"


def test_sidecar_dataset_mismatch_blocks(tmp_path: Path):
    path = tmp_path / "sidecar.json"
    _write_json(path, {"schema_version": 1, "dataset_id": "other", "field_specs": []})
    with pytest.raises(ValueError) as exc:
        load_migration_sidecar(path, expected_dataset_id="expected")
    assert "FIELD_SPEC_DATASET_MISMATCH" in str(exc.value)


# ---------------------------------------------------------------------------
# shared compatibility + coverage
# ---------------------------------------------------------------------------


def _entry(**over):
    from src.autonomous_qa.language.language_quality import LanguageRegistryEntry

    base = dict(
        language_entry_id="e",
        source_kind="CERTIFIED_CAPABILITY",
        source_id="e",
        canonical_blueprint_id="e",
        operator="DIRECT",
        semantic_class="categorical_attribute",
        pattern="[ATTRIBUTE_PHRASE] của [ENTITY_PHRASE] trong đoạn âm thanh là gì?",
        answer_kind="field_value",
        required_slots=["[ATTRIBUTE_PHRASE]", "[ENTITY_PHRASE]"],
        unit_policy="any",
        match_policy=["exact"],
        semantic_contract_hash="x",
        quality_status="PRODUCTION_PASS",
        enabled=True,
        template_library_hash="h",
        registry_version="v",
        entity_scopes=None,
        entity_reference_owner=None,
    )
    base.update(over)
    return LanguageRegistryEntry(**base)


def test_entity_incompatible_entry_not_counted_by_runtime_coverage():
    from src.autonomous_qa.language.template_contracts import TypeContract

    # slot-owned entry whose literal already owns the utterance entity head
    entry = _entry()
    registry = _registry([entry])
    spec = SemanticFieldSpec(
        field_name="topic",
        semantic_class="categorical_attribute",
        entity_scope="utterance",
        entity_phrase="đoạn âm thanh",
        attribute_phrase="chủ đề",
        value_phrase="chủ đề",
    )
    contract = TypeContract(
        type_id="toy_topic",
        operator="DIRECT",
        semantic_field="topic",
        semantic_description="x",
        semantic_phrase_vi="x",
        entity_scope="utterance",
        audio_input_count=1,
        condition_fields=[],
        gold_source_fields=["topic"],
        answer_kind="field_value",
        answer_mode="FIELD_VALUE",
        template_status="ELIGIBLE",
        template_policy={},
        instantiation_policy={},
        source_status="SUPPORTED",
    )
    coverage = check_runtime_coverage(registry, [contract], {"topic": spec})
    assert coverage["covered"] == 0
    assert coverage["missing"] == ["toy_topic"]
    # shared selector agrees
    assert compatible_registry_entries(
        registry, operator="DIRECT", semantic_class="categorical_attribute",
        answer_kind="field_value", match_policy="exact", entity_scope="utterance",
        unit=None, slot_values={"[ENTITY_PHRASE]": "đoạn âm thanh", "[ATTRIBUTE_PHRASE]": "chủ đề"},
    ) == []


# ---------------------------------------------------------------------------
# explicit registry fingerprint identity
# ---------------------------------------------------------------------------


def test_explicit_registry_fingerprint_identity(tmp_path: Path):
    tree = build_toy_tree(tmp_path)
    candidate = build_candidate_registry_from_paths(
        tree["resource_root"] / "language" / "production_registry.json",
        tree["resource_root"] / "language" / "candidate_capabilities.json",
    )
    res = run_preflight(
        mode="registry", write_outputs=False, registry=candidate
    )
    resources = res["fingerprint_inputs"]["resources"]
    assert resources["language_registry_file"] == "in_memory_registry"
    assert resources["language_registry_file_sha256"] == canonical_hash(
        candidate.model_dump(mode="json")
    )
    # canonical default identity unchanged
    default = run_preflight(mode="registry", write_outputs=False)
    assert default["fingerprint_inputs"]["resources"]["language_registry_file"].endswith(
        "production_registry.json"
    )


# ---------------------------------------------------------------------------
# APPLY + rollback
# ---------------------------------------------------------------------------


def test_apply_updates_all_targets_consistently(tmp_path: Path):
    tree = build_toy_tree(tmp_path)
    res = tree["resource_root"]
    bundle = prepare_toy(tmp_path, tree)
    result = apply_language_registry_promotion(bundle, resource_root=res)
    assert result["status"] == "LANGUAGE_REGISTRY_PROMOTED"
    promoted = json.loads((res / "language" / "production_registry.json").read_text(encoding="utf-8"))
    assert promoted["registry_hash"] == bundle.promoted_registry_hash
    for name in ("toy_a", "toy_b"):
        contract = ProductionContract.model_validate(
            json.loads((res / "production" / f"{name}.json").read_text(encoding="utf-8"))
        )
        manifest = PromotionManifest.model_validate(
            json.loads((res / "production" / f"{name}.promotion.json").read_text(encoding="utf-8"))
        )
        assert contract.language_registry_hash == bundle.promoted_registry_hash
        assert manifest.language_registry_hash == bundle.promoted_registry_hash
        assert manifest.production_contract_hash == contract.fingerprint()
        assert contract.promotion_fingerprint == contract.fingerprint()
        assert manifest.promotion_fingerprint == manifest.fingerprint()


def _fail_after(n):
    state = {"committed": 0}

    def replace(src, dst):
        expected = Path(src).read_bytes()
        os.replace(src, dst)
        assert Path(dst).read_bytes() == expected
        state["committed"] += 1
        if state["committed"] == n:
            raise RuntimeError(f"INJECTED_AFTER_{n}")

    return replace, state


def _rollback_case(tmp_path, replace_fn, expected_committed):
    tree = build_toy_tree(tmp_path)
    res = tree["resource_root"]
    bundle = prepare_toy(tmp_path, tree)
    before = snapshot(res)
    with pytest.raises(LanguageRegistryPromotionError) as exc:
        apply_language_registry_promotion(bundle, resource_root=res, replace_fn=replace_fn)
    assert exc.value.code == "LANGUAGE_PROMOTION_TRANSACTION_FAILED_ROLLED_BACK"
    assert snapshot(res) == before
    return expected_committed


def test_rollback_after_registry_replace(tmp_path: Path):
    replace_fn, state = _fail_after(1)
    _rollback_case(tmp_path, replace_fn, 1)
    assert state["committed"] == 1


def test_rollback_after_contract_replace(tmp_path: Path):
    replace_fn, state = _fail_after(2)
    _rollback_case(tmp_path, replace_fn, 2)
    assert state["committed"] == 2


def test_rollback_after_manifest_replace(tmp_path: Path):
    # registry + 2 contracts + first manifest
    replace_fn, state = _fail_after(4)
    _rollback_case(tmp_path, replace_fn, 4)
    assert state["committed"] == 4


def test_rollback_during_final_validation(tmp_path: Path):
    tree = build_toy_tree(tmp_path)
    res = tree["resource_root"]
    bundle = prepare_toy(tmp_path, tree)
    before = snapshot(res)

    def boom():
        raise RuntimeError("INJECTED_FINAL_VALIDATION")

    with pytest.raises(LanguageRegistryPromotionError) as exc:
        apply_language_registry_promotion(bundle, resource_root=res, validation_hook=boom)
    assert exc.value.code == "LANGUAGE_PROMOTION_TRANSACTION_FAILED_ROLLED_BACK"
    assert snapshot(res) == before


def test_stale_registry_blocks(tmp_path: Path):
    tree = build_toy_tree(tmp_path)
    res = tree["resource_root"]
    bundle = prepare_toy(tmp_path, tree)
    _write_json(res / "language" / "production_registry.json", tree["base"].model_dump(mode="json") | {"version": "tampered"})
    with pytest.raises(LanguageRegistryPromotionError) as exc:
        apply_language_registry_promotion(bundle, resource_root=res)
    assert exc.value.code == "LANGUAGE_PROMOTION_BUNDLE_STALE"


def test_stale_contract_blocks(tmp_path: Path):
    tree = build_toy_tree(tmp_path)
    res = tree["resource_root"]
    bundle = prepare_toy(tmp_path, tree)
    path = res / "production" / "toy_a.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["migration_source"] = "tampered"
    _write_json(path, data)
    with pytest.raises(LanguageRegistryPromotionError) as exc:
        apply_language_registry_promotion(bundle, resource_root=res)
    assert exc.value.code == "LANGUAGE_PROMOTION_BUNDLE_STALE"


def test_stale_manifest_blocks(tmp_path: Path):
    tree = build_toy_tree(tmp_path)
    res = tree["resource_root"]
    bundle = prepare_toy(tmp_path, tree)
    path = res / "production" / "toy_a.promotion.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["migration_source"] = "tampered"
    _write_json(path, data)
    with pytest.raises(LanguageRegistryPromotionError) as exc:
        apply_language_registry_promotion(bundle, resource_root=res)
    assert exc.value.code == "LANGUAGE_PROMOTION_BUNDLE_STALE"


def test_apply_rejects_non_ready_bundle(tmp_path: Path):
    tree = build_toy_tree(tmp_path)
    res = tree["resource_root"]
    bundle = prepare_toy(tmp_path, tree, expected_candidate_hash="deadbeef")
    with pytest.raises(LanguageRegistryPromotionError) as exc:
        apply_language_registry_promotion(bundle, resource_root=res)
    assert exc.value.code == "LANGUAGE_PROMOTION_NOT_APPLY_READY"


def test_root_canonical_resources_untouched():
    for path, before in _REAL_BEFORE.items():
        assert path.read_bytes() == before

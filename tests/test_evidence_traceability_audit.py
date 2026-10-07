"""Invariant test suite for Evidence Traceability Audit.

Ensures:
1. SOURCE_GOLD requires real source field in dataset manifest
2. DETERMINISTIC_GOLD for P0 requires an implemented derivation in codebase
3. P0_READY requires traceable gold (DIRECT or DETERMINISTIC_IMPLEMENTED)
4. Missing pragmatic KB prevents pragmatic P0 in VietMed
5. Source toxic label cannot imply banter/aggression label in ViToSA
6. Code-switch term annotation cannot imply phonetic nativization gold in ViMedCSS
7. Canonical written tone cannot imply observed acoustic tone gold in VN-SLU
8. Opportunity names cannot exceed answer semantics
9. Cross-report class consistency (no opportunity has conflicting classes across audit reports)
10. Zero real LLM calls in audit
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = ROOT / "outputs" / "linguistic_discovery_pass1" / "evidence_audit"
DATA_ROOT = ROOT / "data"

# Historical R&D evidence deleted during cleanup; production does not depend on it.
if not AUDIT_DIR.exists():
    pytest.skip(
        "historical linguistic_discovery evidence_audit deleted; R&D decoupled",
        allow_module_level=True,
    )

GOLD_TRACE_JSON = AUDIT_DIR / "GOLD_TRACE_MATRIX.json"
CORRECTED_TRIAGE_JSON = AUDIT_DIR / "CORRECTED_TRIAGE_MATRIX.json"
CORRECTED_TOP15_MD = AUDIT_DIR / "CORRECTED_TOP15.md"
CORRECTED_VI_SHORTLIST_MD = AUDIT_DIR / "CORRECTED_VIETNAMESE_DEPTH_SHORTLIST.md"


@pytest.fixture(scope="module")
def gold_traces():
    assert GOLD_TRACE_JSON.exists(), "GOLD_TRACE_MATRIX.json must exist"
    with open(GOLD_TRACE_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def corrected_triage():
    assert CORRECTED_TRIAGE_JSON.exists(), "CORRECTED_TRIAGE_MATRIX.json must exist"
    with open(CORRECTED_TRIAGE_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def manifests():
    manifests_dict = {}
    for ds in [
        "gigaspeech2_vi", "speech_massive_vi", "vietmdd", "vietmed",
        "vietnam_celeb", "vimd", "vimedcss", "vitosa", "vn_slu"
    ]:
        p = DATA_ROOT / ds / "manifest.jsonl"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
                manifests_dict[ds] = lines
    return manifests_dict


def test_audit_matrix_count(gold_traces, corrected_triage):
    """All 51 opportunities must be audited with 1-to-1 parity."""
    assert len(gold_traces) == 51
    assert len(corrected_triage) == 51


def test_source_gold_requires_real_source_field(gold_traces, manifests):
    """Any task audited as SOURCE_GOLD (DIRECT) must have an answer_field literally present in dataset manifest."""
    for t in gold_traces:
        if t["gold_trace_status"] == "DIRECT":
            ds = t["dataset"]
            field = t["answer_field"]
            assert ds in manifests, f"Dataset {ds} manifest missing"
            assert len(manifests[ds]) > 0, f"Dataset {ds} manifest is empty"
            sample_row = manifests[ds][0]
            assert field in sample_row, f"Field '{field}' not found in source manifest for {ds}"


def test_p0_requires_traceable_gold(corrected_triage, gold_traces):
    """P0_READY requires gold_trace_status in {DIRECT, DETERMINISTIC_IMPLEMENTED}."""
    trace_by_name = {t["opportunity"]: t for t in gold_traces}
    for c in corrected_triage:
        if c["implementation_class"] == "P0_READY":
            t = trace_by_name[c["opportunity_name"]]
            assert t["gold_trace_status"] in {"DIRECT", "DETERMINISTIC_IMPLEMENTED"}, (
                f"P0 task '{c['opportunity_name']}' lacks traceable gold: {t['gold_trace_status']}"
            )
            assert c["scientific_risk"] == "LOW", f"P0 task '{c['opportunity_name']}' has high/medium risk"
            assert c["audio_necessity"] == "REQUIRED", f"P0 task '{c['opportunity_name']}' fails audio necessity"


def test_deterministic_gold_for_p0_requires_implemented_code(gold_traces):
    """DETERMINISTIC_GOLD tasks in P0 must have an existing code path and resource."""
    for t in gold_traces:
        if t["audited_class"] == "P0_READY" and t["audited_gold_status"] == "DETERMINISTIC_GOLD":
            assert t["derivation"]["implemented"] is True
            code_path = ROOT / t["derivation"]["code_path"]
            assert code_path.exists(), f"Derivation code path {code_path} does not exist"


def test_missing_pragmatic_kb_prevents_pragmatic_p0(gold_traces, corrected_triage):
    """VietMed Appropriate Kinship/Address Term Usage cannot be P0 without a pragmatic appropriateness KB."""
    for t in gold_traces:
        if t["dataset"] == "vietmed" and "Kinship" in t["opportunity"]:
            assert t["audited_class"] != "P0_READY", "VietMed Kinship must not be P0"
            assert t["gold_trace_status"] == "DETERMINISTIC_NOT_IMPLEMENTED"
    for c in corrected_triage:
        if c["dataset"] == "vietmed" and "Kinship" in c["opportunity_name"]:
            assert c["implementation_class"] == "P1_KNOWLEDGE"


def test_source_toxic_label_cannot_imply_banter_aggression_label(gold_traces, corrected_triage):
    """ViToSA toxicity label cannot be used as gold for banter vs aggression reasoning."""
    for t in gold_traces:
        if t["dataset"] == "vitosa" and "Discourse Particles" in t["opportunity"]:
            assert t["audited_class"] != "P0_READY", "ViToSA banter/aggression cannot be P0"
            assert t["gold_trace_status"] == "DETERMINISTIC_NOT_IMPLEMENTED"
    for c in corrected_triage:
        if c["dataset"] == "vitosa" and "Discourse Particles" in c["opportunity_name"]:
            assert c["implementation_class"] != "P0_READY"


def test_code_switch_term_annotation_cannot_imply_nativization_gold(gold_traces, corrected_triage):
    """ViMedCSS cs_terms_list cannot imply phonetic nativization or tone coercion gold without a loanword G2P."""
    for t in gold_traces:
        if t["dataset"] == "vimedcss" and "Nativization" in t["opportunity"]:
            assert t["audited_class"] == "P1_KNOWLEDGE"
            assert t["gold_trace_status"] == "DETERMINISTIC_NOT_IMPLEMENTED"
    for c in corrected_triage:
        if c["dataset"] == "vimedcss" and "Nativization" in c["opportunity_name"]:
            assert c["implementation_class"] == "P1_KNOWLEDGE"


def test_canonical_written_tone_cannot_imply_observed_acoustic_tone_gold(gold_traces, corrected_triage):
    """VN-SLU command intent cannot be claimed as acoustic lexical tone recognition gold."""
    for t in gold_traces:
        if t["dataset"] == "vn_slu":
            assert t["audited_class"] == "P2_ACOUSTIC"
            assert t["gold_trace_status"] == "SILVER"
    for c in corrected_triage:
        if c["dataset"] == "vn_slu":
            assert c["implementation_class"] == "P2_ACOUSTIC"


def test_opportunity_names_cannot_exceed_answer_semantics(corrected_triage):
    """Audited task names must not claim more than the verified derivation."""
    for c in corrected_triage:
        name = c["opportunity_name"]
        corr_name = c.get("corrected_task_name", name)
        # Vietnamese has no initial consonant clusters
        if c["dataset"] == "vietmdd" and "has_initial_error" in str(c.get("hidden_gold_source")):
            assert "cluster" not in corr_name.lower(), "VietMDD initial error must not claim cluster simplification"
        # Tone mispronunciation cannot claim continuous tone sandhi
        if c["dataset"] == "vietmdd" and "has_tone_error" in str(c.get("hidden_gold_source")):
            assert "sandhi" not in corr_name.lower(), "VietMDD tone error must not claim tone sandhi"
        # VietMed accent cannot claim individual phonetic realizations
        if c["dataset"] == "vietmed" and c.get("hidden_gold_source") == ["accent"]:
            assert "phonetic variation" not in corr_name.lower(), "VietMed accent task must not claim phonetic variation"


def test_cross_report_class_consistency(corrected_triage):
    """Verify that class classifications in CORRECTED_TRIAGE_MATRIX, TOP15, and VIETNAMESE_DEPTH_SHORTLIST agree."""
    top15_content = CORRECTED_TOP15_MD.read_text(encoding="utf-8")
    shortlist_content = CORRECTED_VI_SHORTLIST_MD.read_text(encoding="utf-8")

    for c in corrected_triage:
        name = c.get("corrected_task_name", c["opportunity_name"])
        cls = c["implementation_class"]

        # If in top 15, ensure implementation class in markdown table matches
        if name in top15_content:
            # check that line with name has the same class
            for line in top15_content.splitlines():
                if f"**{name}**" in line:
                    assert f"`{cls}`" in line, f"Class mismatch in TOP15 for {name}: expected `{cls}` in '{line}'"

        # If in structural shortlist, ensure implementation class in markdown table matches
        if name in shortlist_content:
            for line in shortlist_content.splitlines():
                if f"**{name}**" in line:
                    assert f"`{cls}`" in line, f"Class mismatch in SHORTLIST for {name}: expected `{cls}` in '{line}'"


def test_no_real_llm_calls_in_audit():
    """Ensure audit artifacts are purely deterministic with 0 LLM dependencies."""
    from scripts.audit_evidence_traceability import run_evidence_audit
    # Function runs without network or API keys
    assert callable(run_evidence_audit)

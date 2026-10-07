# Canonical production architecture

The normal production flow is:

1. Dataset Profiler
2. Style Discovery
3. Semantic Feasibility / Capacity
4. Language Coverage
5. Language Preflight
6. Production QA Generation
7. Final QA Content Audit
8. Release

In compact form:

`Profiler → Style Discovery → Feasibility → Language Preflight → Production Generation → Final Audit → Release`

Style Discovery decides **what to ask**. Feasibility proves **whether it can
be built** from the available metadata. The language system decides **how to
say it**. Language Preflight proves that accepted semantic contracts can be
rendered safely before sampling. Generation builds at scale, audit verifies
the result, and release freezes the accepted output.

## One language source of truth

- `resources/language/template_library.json`
- `resources/language/paraphrase_library.json`
- `resources/language/production_registry.json`
- `resources/language/renderer_contract.json`

Production has one renderer behavior. Language entries own presentation
quotes and surrounding grammar; the renderer normalizes target whitespace,
removes one balanced outer quote pair for quoted targets, binds each target
once, and normalizes terminal punctuation boundaries. Missing canonical
resources fail immediately—there is no fallback bundle or compatibility
renderer.

Future language fixes update this canonical system and must pass full Registry
Preflight and the affected dataset preflight. A missing valid semantic
capability returns `LANGUAGE_CAPABILITY_MISSING`; it does not silently drop or
rewrite the semantic task.

## Production gate

New generation requires a matching `PREFLIGHT_PASS` artifact. The gate binds
the registry file hash, renderer contract hash, operator contracts, fixture
strategies, phrase bindings, and accepted semantic types into one fingerprint.
A missing, failed, or stale artifact blocks production.

Production plans record hashes and the preflight fingerprint, not a language
version selector. A repair/rerender phase is not part of normal orchestration:
new infrastructure defects are fixed globally, preflight is rerun, and a new
clean generation is made only when required.

## Side-effect boundaries

Language Preflight is bounded and synthetic: no semantic sampling, dataset-row
scanning, QA generation, audio I/O, network calls, external LLM, Qwen, or
training. Production QA also performs no external LLM or audio I/O; audio is
represented by logical IDs.

## Methodology — Semantic Type Discovery and Deterministic Audio-QA Production

### CURRENT IMPLEMENTATION

#### 1. Actual Input Requirements
The R&D authoring toolchain (`src/autonomous_qa/authoring/pipeline.py`) requires three source artifacts per dataset:
- **Dataset Card**: `data_sources/<dataset_id>/dataset_card.md`
- **Dataset README**: `data/<dataset_id>/README.md`
- **Materialized Annotations**: `data/materialized/<dataset_id>/train.jsonl`

Physical audio files (`.wav` bytes) are **NEVER** opened, decoded, or passed to the LLM during R&D profiling, semantic type discovery, or production QA generation. Audio identity is represented purely as logical string identifiers (`audio_id`, `audio_path`).

#### 2. Documentation and Annotation Profiling
- **Documentation Ingestion**: `ingest_documentation()` reads the source dataset card and README, computes SHA256 hashes, and writes `documentation_manifest.json`.
- **Dataset Profiling**: `profile_dataset()` loads materialized JSONL rows and deterministically computes field data types (`dtype`), missing/empty counts, distinct cardinality, value distributions, and cross-field duplicate counts.
- **Reconciliation**: `recombination_claims()` deterministically compares documented feature claims against observed annotation fields.

#### 3. LLM Semantic Type Proposal
- **Entrypoint**: `scripts/autonomous_qa/run_authoring.py` $\rightarrow$ `src.autonomous_qa.authoring.pipeline.run_authoring()`.
- **LLM Call 1 (`documentation_interpretation`)**: Renders `prompts/rnd/documentation_interpretation_v1.txt` with raw card and README text ($T=0.2$). Interprets field semantics and claims.
- **LLM Call 2 (`primitive_semantic_discovery`)**: Renders `prompts/rnd/primitive_semantic_discovery_v1.txt` with documentation interpretation, dataset profile summary, and reconciliation claims ($T=0.7$). Proposes candidate primitive QA types.

#### 4. Deterministic Candidate Validation
Every candidate primitive proposed by the LLM is filtered through 8 deterministic Python gates in `gate_primitive_candidate()`:
1. `SOURCE_SUPPORT`: Evidence fields exist in observed profile.
2. `GOLD_DERIVABLE`: Hidden source annotations are present in dataset schema.
3. `ANSWER_SCHEMA_VALID`: Answer kind belongs to `{"field_value", "boolean", "audio_index"}`.
4. `OPERATOR_VALID`: Operator belongs to `{"DIRECT", "TARGET_MATCH", "EQUALITY", "PAIRWISE_SELECTION"}`.
5. `NO_DIRECT_GOLD_VISIBLE`: Direct gold field is not visible in prompt for `field_value` tasks.
6. `CAPACITY`: Referenced hidden answer field has distinct value cardinality $\ge 2$.
7. `SPLIT_POLICY`: Split policy matches dataset allowed splits.
8. `NO_DUPLICATE_PROPOSITION`: Proposition signature is unique.

#### 5. Actual Feedback / Retry Behavior
- **LLM Call 3 (`semantic_contract_critic`)**: Renders `prompts/rnd/semantic_contract_critic_v1.txt` ($T=0.0$) to audit proposed primitive contracts.
- **Execution Reality**: The critic's output is logged to `llm/semantic_contract_critic/parsed_response.json` for human inspection, but its output is **NOT** fed into an automated re-prompting loop and does **NOT** alter the set of accepted primitives. The current pipeline executes a **single-pass filtering** workflow.

#### 6. Composite Discovery and Validation
- **Leakage Audit**: `audit_plan()` checks for component visibility in candidate composite tasks.
- **LLM Call 4 (`composite_discovery`)**: Renders `prompts/rnd/composite_discovery_v1.txt` ($T=0.6$) with accepted primitive types and leakage findings. Proposes composite QA types.
- **Composite Gating**: `gate_composite_candidate()` and `deterministic_composite_visibility()` perform empirical row-pair equality checks across source data rows to ensure output components are not recoverable from visible inputs.

#### 7. Language / Template Authoring
- **LLM Call 5 (`language_generation`)**: Renders `prompts/rnd/language_generation_v1.txt` ($T=0.7$) to generate natural language question templates for accepted contracts.
- **LLM Call 6 (`language_semantic_review`)**: Renders `prompts/rnd/language_semantic_review_v1.txt` ($T=0.0$) to review generated entries.
- **Language Preflight Gating**: `gate_language_entry()` deterministically enforces `REQUIRED_ROLE_COVERAGE`, `NO_QUOTE_LEAKAGE` (rejects `__GOLD__` in templates), `DUPLICATE_HEADS`, and `ANSWER_FORMAT_PRESENT`.

#### 8. Contract Promotion
`reconcile_with_canonical()` deterministically compares newly discovered R&D blind candidates against existing canonical types and outputs an informational comparison report (`reconciliation_canonical.json`). It does **NOT** automatically modify canonical catalogs or promote candidates. Promoted semantic contracts are frozen into canonical JSON repositories under `resources/semantics/<dataset>_semantic_catalog.json` and `resources/language/production_registry.json` through explicit, separate frozen governance actions.

#### 9. Production Planning and Sampling
Production generation is **100% zero-LLM**:
- **Generic Production Generator**: `src/autonomous_qa/production/generate_qa.py` $\rightarrow$ `src.autonomous_qa.production.production_qa.py`.
- **Capacity Calculation**: `feasible_semantic_capacity()` calculates deterministic capacity per contract.
- **Pair & Negative Sampling**: `SemanticSampler` samples row pairs deterministically using seed-based pseudo-randomness and `ReuseTracker` caps.
- **Logical Audio Identity**: `opaque_audio_id()` constructs `audio_id` as `sha256(dataset|revision|source_row_id)[:20]` prefixed with `audio_`. `audio_id` is a deterministic logical identifier and is **NOT** a physical WAV content hash.

#### 10. Deterministic QA Rendering
- **Question Rendering**: `render_blueprint()` in `src.autonomous_qa.language.template_renderer` binds target values, normalizes outer quotes (`bind_target_value`), and normalizes terminal punctuation boundaries (`normalize_target_quote_boundary`).
- **Gold Formatting**: `format_answer()` derives exact ground-truth answers directly from source annotation fields.

#### 11. Final Validation and Release
- **Content Audit**: `audit_qa_records()` in `tests/regression/qa_audit.py` validates structural integrity, schema adherence, non-null answers, choice validity, QA ID uniqueness, hidden identifier leakage (`speakerID`), and provenance token leakage (`.wav`, `province_code`). *Note: The audit enforces metadata leakage prevention; general PII detection (e.g., personal names, phone numbers, addresses) is not performed.*
- **Distribution Auditing**: `build_distribution_audit()` calculates and reports label counts, positive/negative ratios (`boolean_label_audit`), position bias (`selection_position_audit`), and target frequencies (`target_distribution_audit`). *Note: Distribution metrics provide informational measurement and reporting; label imbalance does not block release or trigger automatic resampling.*
- **Reproducibility Contract**: Byte-identical output is guaranteed when source materialized annotation bytes, frozen resource hashes, sampling config, seed, generator version, and key sorting order are held identical.
- **Release Manifest**: Outputs byte-identical release files and `release_manifest.json` under `outputs/releases/`.

#### 12. VietMDD as a Concrete Case Study
- **Canonical Release Projection**: `scripts/autonomous_qa/build_vietmdd_mixed_release.py` $\rightarrow$ `src.autonomous_qa.datasets.vietmdd.vietmdd_response_format.py` (`build_mixed_format_release()`). Projects 21,927 canonical QA rows into certified **5-MCQ + 1-OPEN** response-format policy:
  1. `vietmdd_direct_observed_text` (1 Open-Ended)
  2. `vietmdd_target_match_observed_text` (1 MCQ, Binary `["Có", "Không"]`)
  3. `vietmdd_spoken_content_matches_reference` (1 MCQ, Binary `["Có", "Không"]`)
  4. `vietmdd_equality_observed_text` (1 MCQ, Binary `["Có", "Không"]`)
  5. `vietmdd_selection_observed_text` (1 MCQ, Pair Selection `["Đoạn âm thanh thứ nhất", "Đoạn âm thanh thứ hai"]`)
  6. `vietmdd_composite_transcribe_pair_equality` (1 MCQ, 4-tuple choices)
- **External Split Generation**: `scripts/autonomous_qa/generate_vietmdd_external_split.py` generates split-local validation (`valset.jsonl`, 473 rows) and test (`testset.jsonl`, 612 rows) QA packages with split-isolated audio paths (`/audio_val/` and `/audio_test/`), zero LLM calls, and zero TRAIN row leakage.

#### 13. Known Implementation Limitations
- **Single-Pass LLM Discovery**: The LLM critic (`semantic_contract_critic`) and language reviewer (`language_semantic_review`) execute single pass critique calls. Output feedback is persisted to run directories for offline review, but automated iterative re-prompting loops are not executed.
- **No Acoustic Feature Extraction**: R&D discovery operates on metadata text and categorical fields only; audio signal analysis (e.g. pitch, duration, phoneme boundaries) is not performed during semantic type proposal.
- **Informational Distribution Auditing**: Distribution metrics measure and report label ratios; automatic rebalancing or blocking on imbalanced label distributions is not executed.
- **Provenance Token Leakage vs General PII**: Audit checks filter known metadata provenance tokens and hidden IDs (`speakerID`), but do not execute general PII text detection.

---

### POTENTIAL FUTURE IMPROVEMENTS

1. **Automated Multi-Round Feedback Loop**: Connect `semantic_contract_critic` feedback into an automated retry loop that re-prompts `primitive_semantic_discovery` with specific gate failure reasons and critique feedback before final deterministic gating.
2. **Automated Template Revision**: Wire `language_semantic_review` feedback back into `language_generation` to auto-correct flagged template phrasing defects before preflight freeze.
3. **Automated Label Rebalancing**: Add configurable threshold enforcement to `boolean_label_audit` and `selection_position_audit` to reject or resample imbalanced distributions when positive/negative ratios diverge from targets.



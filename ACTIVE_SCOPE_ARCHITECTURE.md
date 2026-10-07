# ACTIVE-SCOPE ARCHITECTURE SPECIFICATION

## Overview
This document defines the frozen canonical architecture for all active datasets in `audio_qa_mvp`:
- **ViMD** (Closed Reference)
- **ViMedCSS** (Closed)
- **VietMDD** (Closed)
- **MACS** (Explicitly Deferred: `DEFERRED_CARD_CONTRACT_CONFLICT`)

## Unified Architecture Flow
The active scope follows **ONE generic card-first architecture** with two input paths joining at the generic compiler:

```
SEMANTIC CONTRACT PATH:
  CANONICAL USER DATASET CARD (data_sources/<dataset>/dataset_card.md)
              |
              v
      GENERIC CARD PARSER (src/autonomous_qa/core/dataset_profile.py)
              |
              v
          DatasetProfile
              |
              v
  GENERIC SEMANTIC PLANNING / COMPILER / LANGUAGE / CERTIFICATION / PRODUCTION

PHYSICAL / DOMAIN DATA PATH:
  RAW / PREPROCESSED DATA
              |
              v
  DATASET-SPECIFIC PHYSICAL, STRUCTURAL, OR DOMAIN BOUNDARY
    - ViMD: structural/raw boundary adapter
    - ViMedCSS: vimedcss_source.py
    - VietMDD: manifest.jsonl -> context_adapter.load_dataset_rows
              |
              v
      NORMALIZED GENERIC ROWS
              |
              +--------------------> JOIN WITH GENERIC ENGINE
```

## Dataset Lifecycle Classifications
1. **ViMD**:
   - Status: `CLOSED_REFERENCE`
   - Boundary: Structural schema + raw boundary adapter -> generic profiler & generic runner.
2. **ViMedCSS**:
   - Status: `CLOSED`
   - Boundary: `vimedcss_source.py` physical adapter -> generic runner.
   - Legacy compiler & production: DELETED & ABSENT.
3. **VietMDD**:
   - Status: `CLOSED`
   - Boundary: `manifest.jsonl` -> `context_adapter.load_dataset_rows`.
   - Domain/Build Helpers:
     - `vietmdd_composite_expansion.py`: BUILD_TIME_PREPROCESSOR
     - `knowledge/vietmdd_parser.py`: BUILD_TIME_PREPROCESSOR
     - `vietmdd_t4k_frontend.py`: BUILD_TIME_VALIDATOR
     - `vietmdd_migration.py`: MIGRATION_ONLY
   - Legacy compiler & production: DELETED & ABSENT.

## Deferred Datasets
- **MACS**: Remains deferred under `DEFERRED_CARD_CONTRACT_CONFLICT`. MACS source card installed, but HuggingFace-representation contract is unresolved. No migration or code changes performed for MACS in this phase.

## Invariants & Controls
- **Semantic Authority**: `USER CARD -> DatasetProfile` is the single source of truth for semantic metadata.
- **Generic Layers**: Zero dataset-specific semantic branches in compiler, core, language, certification, production layers.
- **Run-Pipeline Ownership**: CLI = 5, Runner = 8, Context Adapter = 16, Wrong Owners = 0, Compatibility Re-exports = 0.
- **Scalability**: New datasets on-board via Card + Manifest + Optional Physical Adapter/Preprocessor without modifying generic compiler/production core.

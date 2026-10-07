"""Preparation script for Speech-MASSIVE vi-VN dataset.

Reads local source parquets from G:/VietnameseSpeechQABenchmark/data/speech_massive_vi/vi-VN,
extracts audio WAV files byte-for-byte to G:/VietnameseSpeechQABenchmark/data/speech_massive_vi/audio/test/,
normalizes metadata, and writes:
- data/speech_massive_vi/manifest.jsonl
- data/speech_massive_vi/sample.jsonl
- data/speech_massive_vi/schema.json
- data/speech_massive_vi/README.md
- data/speech_massive_vi/PREPARE_REPORT.md

Deterministic, idempotent, local-only: zero LLM calls, zero network calls.
"""

from __future__ import annotations

import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SOURCE_DIR = Path("G:/VietnameseSpeechQABenchmark/data/speech_massive_vi/vi-VN")
AUDIO_OUT_DIR = Path("G:/VietnameseSpeechQABenchmark/data/speech_massive_vi/audio/test")
DEST_DATA_DIR = ROOT / "data" / "speech_massive_vi"
MANIFEST_PATH = DEST_DATA_DIR / "manifest.jsonl"
SAMPLE_PATH = DEST_DATA_DIR / "sample.jsonl"
SCHEMA_PATH = DEST_DATA_DIR / "schema.json"
README_PATH = DEST_DATA_DIR / "README.md"
REPORT_PATH = DEST_DATA_DIR / "PREPARE_REPORT.md"

from src.common.missing_values import (
    extract_source_missing_values,
    normalize_row_missing_values,
)


def normalize_sex(val: str | None) -> str | None:
    if val is None:
        return None
    val_clean = str(val).strip()
    if val_clean.lower() == "female":
        return "female"
    if val_clean.lower() == "male":
        return "male"
    return val_clean


def build_schema() -> dict:
    return {
        "dataset": "speech_massive_vi",
        "description": "Speech-MASSIVE Vietnamese (vi-VN) speech intent and scenario benchmark.",
        "locale": "vi-VN",
        "fields": {
            "audio": {
                "type": "string",
                "role": "audio",
                "description": "Absolute Windows path to local audio WAV file. Input evidence only.",
                "evaluation": "eligible",
            },
            "text": {
                "type": "string",
                "role": "semantic",
                "entity_scope": "utterance",
                "description": "Spoken Vietnamese transcript. Semantic QA annotation describing the utterance.",
                "evaluation": "eligible",
            },
            "intent": {
                "type": "string",
                "role": "semantic",
                "entity_scope": "utterance",
                "description": "Intent classification label associated with the spoken utterance.",
                "is_categorical": True,
                "cardinality": 59,
                "evaluation": "eligible",
            },
            "scenario": {
                "type": "string",
                "role": "semantic",
                "entity_scope": "utterance",
                "description": "Broader domain/scenario category enclosing the intent.",
                "is_categorical": True,
                "cardinality": 18,
                "evaluation": "eligible",
            },
            "speaker_id": {
                "type": "string",
                "role": "hidden_identifier",
                "entity_scope": "speaker",
                "description": "Identifier linking utterances produced by the same speaker. Hidden identifier.",
                "is_categorical": True,
                "cardinality": 30,
            },
            "speaker_sex": {
                "type": "string",
                "role": "semantic",
                "entity_scope": "speaker",
                "description": "Annotated biological sex of the speaker ('female', 'male'). Missing annotations are marked as unidentified.",
                "is_categorical": True,
                "cardinality": 2,
                "source_missing_values": [
                    "Unidentified",
                    "unidentified",
                ],
                "evaluation": "eligible",
            },
            "speaker_age": {
                "type": "string",
                "role": "semantic",
                "entity_scope": "speaker",
                "description": "Annotated chronological age of the speaker as recorded at collection time.",
                "is_categorical": True,
                "cardinality": 16,
                "source_missing_values": [
                    "Unidentified",
                    "unidentified",
                ],
                "evaluation": "discovery_only",
            },
            "id": {
                "type": "string",
                "role": "provenance",
                "description": "Upstream unique utterance ID. Provenance tracking only.",
            },
            "split": {
                "type": "string",
                "role": "provenance",
                "description": "Dataset split partition ('test'). Provenance tracking only.",
            },
            "annot_utt": {
                "type": "string",
                "role": "context_only",
                "entity_scope": "utterance",
                "description": "Raw utterance text with inline slot annotations. Contextual understanding only.",
            },
        },
        "field_roles": {
            "audio": "audio",
            "text": "semantic",
            "intent": "semantic",
            "scenario": "semantic",
            "speaker_sex": "semantic",
            "speaker_age": "semantic",
            "speaker_id": "hidden_identifier",
            "id": "provenance",
            "split": "provenance",
            "annot_utt": "context_only",
        },
        "entity_scopes": {
            "text": "utterance",
            "intent": "utterance",
            "scenario": "utterance",
            "speaker_sex": "speaker",
            "speaker_age": "speaker",
            "speaker_id": "speaker",
            "annot_utt": "utterance",
        },
        "source_missing_values": {
            "speaker_sex": [
                "Unidentified",
                "unidentified",
            ],
            "speaker_age": [
                "Unidentified",
                "unidentified",
            ],
        },
        "field_evaluations": {
            "audio": "eligible",
            "text": "eligible",
            "intent": "eligible",
            "scenario": "eligible",
            "speaker_sex": "eligible",
            "speaker_age": "discovery_only",
        },
        "hidden_fields": ["speaker_id"],
        "semantic_fields": ["text", "intent", "scenario", "speaker_sex", "speaker_age"],
    }


def build_readme(total_rows: int, scenarios: list[str], intents: list[str], speakers_count: int) -> str:
    return f"""# Speech-MASSIVE vi-VN

Speech-MASSIVE vi-VN is a Vietnamese spoken language understanding dataset providing speech audio recordings and multi-level annotations.

## Scope

- Dataset name: Speech-MASSIVE
- Locale: vi-VN (Vietnamese - Vietnam)
- Total prepared rows: {total_rows}
- Splits: test (official Speech-MASSIVE test partition)

## Unit

Each sample corresponds to a single spoken Vietnamese command/utterance recorded by an annotated speaker.

## Fields and Roles

### audio
Actual audio recording path. Input evidence for audio understanding, not a QA answer field.

### text
Spoken transcription; semantic QA annotation representing the verbatim words spoken in the audio (entity scope: utterance).

### intent
Semantic intent label associated with the spoken command ({len(intents)} fine-grained categories, entity scope: utterance).

### scenario
Broader semantic scenario/domain category enclosing the intent ({len(scenarios)} categories, entity scope: utterance).

### speaker_sex
Semantic speaker attribute: annotated biological sex of the speaker (`female`, `male`, entity scope: speaker). Missing annotations are represented as null in canonical manifests.

### speaker_age
Semantic speaker attribute: annotated chronological age of the speaker as recorded at collection time (entity scope: speaker). Exact chronological age is maintained for discovery, but exact one-year differences are excluded from final evaluation benchmarks. Missing annotations are represented as null.

### speaker_id
Hidden participant identifier; useful internally for same-speaker comparison (entity scope: speaker). The identifier itself should not be exposed as a semantic answer.

### id
Provenance/source utterance identifier only; not audio-understanding content and not a QA answer attribute.

### split
Dataset partition/provenance only (`test`); not inferable from audio and not a QA attribute.

### annot_utt
Slot-annotated utterance representation; useful contextual annotation for dataset understanding (entity scope: utterance), but not directly executable as an answer under the current MVP DSL.

## Structural Relations

- Every `intent` maps functionally to its enclosing `scenario`.
- Identical `speaker_id` values identify utterances spoken by the same participant.
- Multi-sample comparisons on shared semantic attributes (`intent`, `scenario`, `speaker_sex`, `speaker_age`) or hidden speaker identity are supported.

## QA Constraints

- Only use fields and relations described in this schema.
- Do not expose literal identifiers (`speaker_id`, `id`) or non-semantic metadata (`split`, `annot_utt`, `audio`) as target answers.
- Gold answers must be deterministic and backed by the structured data.
"""



def main() -> int:
    AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(SOURCE_DIR / "*.parquet")))
    if not files:
        print(f"Error: no parquet files found in {SOURCE_DIR}")
        return 1

    print(f"Found {len(files)} source parquet files in {SOURCE_DIR}")

    total_source_rows = 0
    vi_vn_source_rows = 0
    skipped_rows = 0
    audio_extracted = 0
    audio_already_exists = 0
    missing_audio = 0

    manifest_rows: list[dict] = []
    scenarios_set: set[str] = set()
    intents_set: set[str] = set()
    speakers_set: set[str] = set()
    sex_counter: Counter = Counter()
    age_counter: Counter = Counter()
    null_counts: Counter = Counter()

    for fp in files:
        table = pq.read_table(fp)
        pyd = table.to_pylist()
        total_source_rows += len(pyd)

        for r in pyd:
            locale = r.get("locale")
            if locale != "vi-VN":
                skipped_rows += 1
                continue
            vi_vn_source_rows += 1

            # Audio extraction / resolution
            audio_obj = r.get("audio") or {}
            raw_bytes = audio_obj.get("bytes")
            rel_path = audio_obj.get("path") or r.get("path")
            if rel_path:
                filename = Path(rel_path).name
            else:
                filename = f"{r.get('id')}.wav"

            wav_target_path = AUDIO_OUT_DIR / filename
            if raw_bytes is not None:
                if not wav_target_path.is_file() or wav_target_path.stat().st_size != len(raw_bytes):
                    with open(wav_target_path, "wb") as wf:
                        wf.write(raw_bytes)
                    audio_extracted += 1
                else:
                    audio_already_exists += 1
            else:
                if wav_target_path.is_file():
                    audio_already_exists += 1
                else:
                    missing_audio += 1

            audio_abs_path = str(wav_target_path).replace("\\", "/")

            text = (r.get("utt") or "").strip()
            intent = (r.get("intent_str") or "").strip()
            scenario = (r.get("scenario_str") or "").strip()
            speaker_id = str(r.get("speaker_id") or "").strip()
            speaker_sex = normalize_sex(r.get("speaker_sex"))
            raw_age = r.get("speaker_age")
            speaker_age = str(raw_age).strip() if raw_age is not None and str(raw_age).strip() else None
            split = (r.get("partition") or "test").strip()
            source_id = str(r.get("id") or "").strip()
            annot_utt = (r.get("annot_utt") or "").strip()

            row_dict = {
                "audio": audio_abs_path,
                "text": text,
                "intent": intent,
                "scenario": scenario,
                "speaker_id": speaker_id,
                "speaker_sex": speaker_sex,
                "speaker_age": speaker_age,
                "split": split,
                "id": source_id,
                "annot_utt": annot_utt,
            }
            schema_data = build_schema()
            missing_values = extract_source_missing_values(schema_data)
            row_dict = normalize_row_missing_values(row_dict, missing_values)

            for k, v in row_dict.items():
                if v is None or v == "":
                    null_counts[k] += 1

            scenarios_set.add(scenario)
            intents_set.add(intent)
            speakers_set.add(speaker_id)
            if row_dict["speaker_sex"] is not None:
                sex_counter[row_dict["speaker_sex"]] += 1
            if row_dict["speaker_age"] is not None:
                age_counter[row_dict["speaker_age"]] += 1

            manifest_rows.append(row_dict)


    print(f"Total source rows: {total_source_rows}")
    print(f"vi-VN rows: {vi_vn_source_rows}")
    print(f"Skipped rows: {skipped_rows}")
    print(f"Audio files newly extracted: {audio_extracted}")
    print(f"Audio files already existed: {audio_already_exists}")
    print(f"Missing audio files: {missing_audio}")
    print(f"Final manifest rows: {len(manifest_rows)}")

    # Verify all audio files exist
    unresolved_audio = [r["audio"] for r in manifest_rows if not os.path.isfile(r["audio"])]
    if unresolved_audio:
        print(f"ERROR: {len(unresolved_audio)} audio files do not exist!")
        return 1

    # Write manifest.jsonl
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        for r in manifest_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {MANIFEST_PATH} ({len(manifest_rows)} rows)")

    # Stratified diverse sampling for sample.jsonl (seed 42, 26 rows)
    rng = random.Random(42)
    by_speaker: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(manifest_rows):
        by_speaker[r["speaker_id"]].append(i)
    multi_spk = {k: v for k, v in by_speaker.items() if len(v) >= 2}

    # Group by (scenario, speaker_sex)
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(manifest_rows):
        buckets[(r["scenario"], r["speaker_sex"])].append(i)

    picked: list[int] = []
    # Pick 1 from each bucket sorted
    for key in sorted(buckets, key=lambda k: (str(k[0]), str(k[1]))):
        pool = buckets[key][:]
        rng.shuffle(pool)
        for idx in pool:
            picked.append(idx)
            break
        if len(picked) >= 24:
            break

    # Ensure a same-speaker pair for speaker verification few-shot demonstration
    pair_spk = sorted(multi_spk)[0]
    for idx in multi_spk[pair_spk][:2]:
        if idx not in picked:
            picked.append(idx)

    picked = picked[:26]
    sample_rows = [manifest_rows[i] for i in picked]

    with open(SAMPLE_PATH, "w", encoding="utf-8") as f:
        for r in sample_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {SAMPLE_PATH} ({len(sample_rows)} rows)")

    # Write schema.json
    schema_data = build_schema()
    with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump(schema_data, f, ensure_ascii=False, indent=2)
    print(f"Wrote {SCHEMA_PATH}")

    # Write README.md
    readme_content = build_readme(
        total_rows=len(manifest_rows),
        scenarios=sorted(scenarios_set),
        intents=sorted(intents_set),
        speakers_count=len(speakers_set),
    )
    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"Wrote {README_PATH}")

    # Diagnostic derived relation check: intent -> scenario
    from src.common.missing_values import validate_derived_relation
    derived_res = validate_derived_relation(
        "intent", "scenario", manifest_rows,
        schema_fields=frozenset(["audio", "text", "intent", "scenario", "speaker_id", "speaker_sex", "speaker_age", "split", "id", "annot_utt"]),
        hidden_fields=frozenset(["speaker_id", "id"]),
    )

    # Write PREPARE_REPORT.md
    report_content = f"""# Speech-MASSIVE vi-VN Preparation Audit Report

## 1. Source Information
- Source path: `{SOURCE_DIR}`
- Source format: Apache Parquet (HuggingFace dataset format with embedded WAV audio structs)
- Source files:
{chr(10).join(f"  - `{Path(f).name}` ({os.path.getsize(f):,} bytes)" for f in files)}
- Source splits discovered: `test` (all rows belong to official Speech-MASSIVE test partition)

## 2. Locale Filter
- Exact filter: `locale == 'vi-VN'`
- Total source rows across parquets: {total_source_rows}
- Total vi-VN source rows: {vi_vn_source_rows}
- Skipped non-vi-VN rows: {skipped_rows}

## 3. Audio Extraction & Resolution
- Audio destination directory: `{AUDIO_OUT_DIR}`
- Audio format: Raw RIFF WAVE (`.wav`) extracted byte-for-byte from parquet `audio.bytes` (no transcoding or re-encoding)
- Newly extracted audio files: {audio_extracted}
- Already existing audio files: {audio_already_exists}
- Missing audio files: {missing_audio}
- Verified on disk: 100% ({len(manifest_rows)} / {len(manifest_rows)} exist)

## 4. Output Manifest & Sample
- Manifest path: `{MANIFEST_PATH}` ({len(manifest_rows)} rows)
- Sample path: `{SAMPLE_PATH}` ({len(sample_rows)} rows, seed=42)
- Schema path: `{SCHEMA_PATH}`
- README path: `{README_PATH}`

## 5. Field Mapping & Schema
| Canonical Field | Source Field | Type | Semantic / Description | Hidden / Non-semantic |
|---|---|---|---|---|
| `audio` | `audio.bytes` -> WAV | string | Absolute local Windows path on G:\\ | No |
| `text` | `utt` | string | Spoken Vietnamese transcription | No |
| `intent` | `intent_str` | string | Intent classification label | No |
| `scenario` | `scenario_str` | string | Scenario / domain category | No |
| `speaker_id` | `speaker_id` | string | Speaker participant identifier | **Yes (hidden)** |
| `speaker_sex` | `speaker_sex` | string | Speaker sex ('female', 'male', 'unidentified') | No |
| `speaker_age` | `speaker_age` | string | Speaker age at collection time | No |
| `split` | `partition` | string | Dataset partition ('test') | No |
| `id` | `id` | string | Upstream source utterance ID | **Yes (hidden)** |
| `annot_utt` | `annot_utt` | string | Inline bracketed slot-annotated transcript | No |

## 6. Cardinality & Distribution Statistics
- Scenarios ({len(scenarios_set)} total):
  `{", ".join(sorted(scenarios_set))}`
- Intents ({len(intents_set)} total):
  59 fine-grained intents
- Speakers ({len(speakers_set)} total):
  30 distinct speakers
- Speaker Sex distribution:
  {dict(sex_counter)}
- Speaker Age values ({len(age_counter)} distinct):
  `{", ".join(sorted(age_counter.keys()))}`
- Missing / Null values across all canonical fields:
  {dict(null_counts) if null_counts else "0 null values"}

## 7. Slot Representation
The upstream slot annotations are represented in `annot_utt` as inline text annotations, e.g.:
`gửi thư điện tử các [event_name : cuộc hẹn] của tôi để xếp lại lịch trình`
and in `slot_method` / `tokens` / `labels`. These are preserved as available metadata in `annot_utt`, but are not forced into the current MVP Answer DSL.

## 8. Diagnostic Data-Backed Relation Validation
Evaluating `intent -> scenario` using the existing generic derived relation validator:
- Valid: `{derived_res['valid']}`
- Distinct source values (intents): {derived_res['distinct_source_count']}
- Distinct target values (scenarios): {derived_res['distinct_target_count']}
- Reusable source values: {derived_res['reusable_source_values']}
- Mapping conflicts: {derived_res['mapping_conflicts']}
- Coarsening: {derived_res['distinct_target_count']} < {derived_res['distinct_source_count']} (18 < 59)
*(Diagnostic only: not hardcoded into taxonomy or whitelist)*
"""
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Wrote {REPORT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

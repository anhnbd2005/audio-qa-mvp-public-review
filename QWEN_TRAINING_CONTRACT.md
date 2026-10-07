# QWEN TRAINING CONTRACT & ARCHITECTURE SPECIFICATION — PHASE 3D.0

## Executive Summary

- **Verdict**: `QWEN_TRAINING_CONTRACT_READY`
- **Perception MCQ Ready**: `YES`
- **System Prompt Decision**: `NO_SYSTEM_PROMPT`
- **Target Package Location**: `src/sauvi_perception/qwen25_omni/`
- **One-Command Entrypoint**: `python -m src.sauvi_perception.qwen25_omni.run --config configs/qwen25_omni_3b.yaml`

---

## VietMDD QA Types Audit & Training Recommendations

| question_type_id | current format | representative QA | gold source | recommended Qwen format | target representation | reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `vietmdd_direct_observed_text` | TEXT | Vui lòng chép lại toàn bộ nội dung phát âm trong đoạn âm thanh. | PRIMITIVE_RELATION | `OPEN_ENDED_TEXT` | Plain text transcription | Task is unconstrained speech transcription of the target audio segment. |
| `vietmdd_target_match_observed_text` | MCQ | Đoạn âm thanh này có nội dung phát âm là '...' không? | DERIVED_SOURCE | `KEEP_AS_MCQ` | Option Label (A/B) | Task is binary target match verification (A: Đúng, B: Sai). |
| `vietmdd_spoken_content_matches_reference` | MCQ | Đoạn âm thanh này có nội dung phát âm khớp với văn bản tham chiếu '...' không? | DERIVED_SOURCE | `KEEP_AS_MCQ` | Option Label (A/B) | Task is binary reference match verification (A: Khớp, B: Không khớp). |
| `vietmdd_selection_observed_text` | MCQ | Mục tiêu là '...'. Trong hai đoạn âm thanh A và B, đoạn nào chứa nội dung này? | DERIVED_SOURCE | `KEEP_AS_MCQ` | Option Label (A/B) | Task is audio pair selection among choices (A: Đoạn A, B: Đoạn B). |
| `vietmdd_equality_observed_text` | MCQ | Liệu nội dung phát âm trong hai đoạn âm thanh có giống nhau hay không? | DERIVED_SOURCE | `KEEP_AS_MCQ` | Option Label (A/B) | Task is relational equality comparison between two spoken audio files (A: Giống nhau, B: Khác nhau). |
| `vietmdd_composite_transcribe_pair_equality` | STRUCTURED_OUTPUT | Nghe hai đoạn âm thanh A và B, hãy nêu nội dung A, nêu nội dung B, và cho biết hai nội dung có giống nhau không. | DERIVED_SOURCE | `EXCLUDE_FROM_SFT` | Decomposed into primitive transcription + MCQ equality tasks | Composite multi-field output is redundant; primitive transcription and equality tasks cover all underlying semantics cleanly. |

---

## SAUVI Perception MCQ Specification

- **Active Tasks**: 7 tasks (`p1_speaker_verification`, `p2_speaker_gender`, `p3_speaker_age`, `p5_sentiment`, `p6_asr`, `p7_tempo`, `p8_dialect`)
- **All MCQ Verified**: `True`
- **Multi-Audio Support**: Required (`min_audio` = 1, `max_audio` = 2+)

### Canonical Perception MCQ JSONL Schema

```json
{
    "id": "perception_p1_001",
    "dataset": "sauvi_perception",
    "task": "p1_speaker_verification",
    "audio": [
        "p1/spk_a.wav",
        "p1/spk_b.wav"
    ],
    "question": "Do these two audio recordings feature the same speaker?",
    "choices": [
        {"label": "A", "text": "Yes"},
        {"label": "B", "text": "No"}
    ],
    "answer": "A",
    "metadata": {}
}
```

---

## Existing G:\A_qwen Reference & Reusability

- **Latest Working Implementation**: `control_runs/vimd_qwen25_omni_3b_canonical_control_v2`
- **Model**: `Qwen/Qwen2.5-Omni-3B`
- **LoRA Configuration**: `r=16, alpha=32, target_modules=[q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj]`
- **System Prompt Decision**: `NO_SYSTEM_PROMPT` (Derived sha256 `193cd4b45fff313dbb4393064add6858b81e86fb392d9b7b69727c6f6ae74273` proved 78.47% accuracy & 0.7719 macro F1 without system prompt)
- **Loss Masking Policy**: User text and audio tokens masked with `-100`; loss computed on assistant option letter target (`A`/`B`/`C`/`D`) only.

---

## One-Command Workflow Architecture

```bash
python -m src.sauvi_perception.qwen25_omni.run --config configs/qwen25_omni_3b.yaml
```

```
src/sauvi_perception/qwen25_omni/
├── __init__.py
├── data.py        (canonical JSONL loader, Qwen formatter, collator)
├── train.py       (QLoRA model loader, trainer, checkpoint saver)
├── evaluate.py    (MCQ generation parser & evaluation metrics)
└── run.py         (single-command orchestration pipeline)
```


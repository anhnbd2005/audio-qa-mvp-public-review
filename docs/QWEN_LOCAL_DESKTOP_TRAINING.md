# Qwen2.5-Omni Local Desktop QLoRA Training Guide

This package enables offline, local-desktop QLoRA training for **Qwen2.5-Omni-3B** and scale-up for **Qwen2.5-Omni-7B** without network downloads or pip installs during execution.

---

## 1. Prepare Local Model Checkpoint

Ensure your model weights are saved locally on disk (e.g., `E:/models/Qwen2.5-Omni-3B/`).
The directory must contain `config.json`, tokenizer/processor configuration, and model weights (`model.safetensors` or `pytorch_model.bin`).

No online Hugging Face downloading is performed when `local_files_only: true`.

---

## 2. Prepare Data

Place your audio files and JSONL manifests in a local folder.
The JSONL rows must contain canonical MCQ format:
```json
{
  "id": "sample-001",
  "audio": ["relative/path/audio.wav"],
  "question": "Hỏi...",
  "choices": ["Miền Bắc", "Miền Trung", "Miền Nam"],
  "answer": "Miền Bắc"
}
```
*Note: `answer` must remain the semantic answer text.*

---

## 3. Configuration

Edit `configs/qwen25_omni_3b_qlora_local.yaml`:
- Set `model.path` to your local checkpoint directory.
- Set `data.audio_root`, `data.train_jsonl`, and `data.validation_jsonl`.
- Set `output.dir` for run outputs.

---

## 4. Run Preflight

Validate manifests without loading model:
```bash
python -m src.sauvi_perception.qwen25_omni.run --config configs/qwen25_omni_3b_qlora_local.yaml
```

---

## 5. 7B Scale-Up

To train on **Qwen2.5-Omni-7B**, simply pass:
```bash
python -m src.sauvi_perception.qwen25_omni.run --config configs/qwen25_omni_7b_qlora_local.yaml
```
Zero Python code changes required.

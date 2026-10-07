"""QLoRA model loading, no-system chat template derivation, label masking, and training loop."""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .data import (
    choice_letter,
    permuted_choices,
    render_mcq_user_text,
    target_letter_for,
)

logger = logging.getLogger("qwen25_omni.engine")


class UnsupportedChatTemplate(ValueError):
    """Raised when chat template cannot be safely derived."""


class SystemPromptDetected(ValueError):
    """Raised when a system prompt survives in the prompt template."""


def verify_local_checkpoint_paths(model_path: str | Path, processor_path: str | Path | None = None) -> dict[str, Any]:
    m_path = Path(model_path)
    p_path = Path(processor_path) if processor_path else m_path

    report = {
        "model_path": str(m_path),
        "processor_path": str(p_path),
        "model_dir_exists": m_path.is_dir(),
        "processor_dir_exists": p_path.is_dir(),
        "config_json_exists": (m_path / "config.json").is_file(),
        "model_weights_resolved": False,
        "processor_config_exists": (p_path / "processor_config.json").is_file() or (p_path / "preprocessor_config.json").is_file(),
        "tokenizer_config_exists": (p_path / "tokenizer_config.json").is_file(),
        "valid": False,
    }

    has_safetensors = (m_path / "model.safetensors").is_file() or (m_path / "model.safetensors.index.json").is_file()
    has_bin = (m_path / "pytorch_model.bin").is_file() or (m_path / "pytorch_model.bin.index.json").is_file()
    report["model_weights_resolved"] = has_safetensors or has_bin

    report["valid"] = (
        report["model_dir_exists"]
        and report["processor_dir_exists"]
        and report["config_json_exists"]
        and report["model_weights_resolved"]
    )
    return report


def apply_no_system_chat_template(processor) -> dict[str, Any]:
    """Derive and install no-system chat template on processor (fail closed)."""
    tokenizer = getattr(processor, "tokenizer", None)
    base_template = getattr(processor, "chat_template", None) or (getattr(tokenizer, "chat_template", None) if tokenizer else None)

    if not base_template:
        raise UnsupportedChatTemplate("No chat_template found on processor or tokenizer")

    if "<|im_start|>system" in base_template:
        pattern = re.compile(
            r"<\|im_start\|>system\n.*?<\|im_end\|>\n?",
            re.DOTALL,
        )
        derived = pattern.sub("", base_template)
    else:
        derived = base_template

    if "<|im_start|>system" in derived:
        raise UnsupportedChatTemplate("System turn remains in derived chat template")

    if processor is not None:
        try:
            setattr(processor, "chat_template", derived)
        except Exception:
            pass
    if tokenizer is not None:
        try:
            setattr(tokenizer, "chat_template", derived)
        except Exception:
            pass

    return {
        "changed": derived != base_template,
        "derived_has_system": "<|im_start|>system" in derived,
        "chat_template": derived,
    }


def audit_serialized_prompt_no_system(
    processor,
    *,
    sample_id: str,
    question: str,
    choices: Sequence[str],
    answer: str,
    train_messages: Sequence[dict[str, Any]],
    inference_messages: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    user_text = render_mcq_user_text(question, choices)

    for msg in list(train_messages) + list(inference_messages):
        if msg.get("role") in ("system", "developer"):
            raise SystemPromptDetected(f"System message detected in message list: {msg}")

    return {
        "id": sample_id,
        "question": question,
        "choices": list(choices),
        "semantic_user_text": user_text,
        "semantic_assistant_target": answer,
        "verdict": "PASS",
    }


def get_trainable_param_stats(model) -> dict[str, Any]:
    total_params = 0
    trainable_params = 0
    lora_trainable = 0
    audio_trainable = 0
    visual_trainable = 0
    other_trainable = 0

    for name, param in model.named_parameters():
        numel = param.numel()
        total_params += numel
        if param.requires_grad:
            trainable_params += numel
            if "lora_" in name:
                lora_trainable += numel
            elif "audio_tower" in name or "audio" in name:
                audio_trainable += numel
            elif "visual" in name:
                visual_trainable += numel
            else:
                other_trainable += numel

    pct = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percentage": pct,
        "lora_trainable": lora_trainable,
        "audio_tower_trainable": audio_trainable,
        "visual_tower_trainable": visual_trainable,
        "other_trainable": other_trainable,
        "valid": (trainable_params > 0 and audio_trainable == 0 and visual_trainable == 0 and other_trainable == 0),
    }


def setup_qlora_model(config: dict[str, Any], model_obj=None, processor_obj=None):
    """Load model and processor, configure 4-bit NF4 QLoRA, attach LoRA adapter."""
    import torch
    from transformers import BitsAndBytesConfig

    model_cfg = config.get("model", {})
    quant_cfg = config.get("quantization", {})
    lora_cfg = config.get("lora", {})
    runtime_cfg = config.get("runtime", {})

    model_path = model_cfg.get("path")
    processor_path = model_cfg.get("processor_path") or model_path
    local_files_only = model_cfg.get("local_files_only", True)

    pf_report = verify_local_checkpoint_paths(model_path, processor_path)
    if not pf_report["valid"]:
        raise FileNotFoundError(f"Local checkpoint validation failed: {pf_report}")

    compute_dtype = torch.float16
    if quant_cfg.get("compute_dtype") == "bfloat16":
        compute_dtype = torch.bfloat16

    bnb_config = None
    if quant_cfg.get("enabled", True) and quant_cfg.get("load_in_4bit", True):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg.get("quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant_cfg.get("double_quant", True),
            bnb_4bit_compute_dtype=compute_dtype,
        )

    try:
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    except ImportError:
        from transformers import AutoProcessor, AutoModelForCausalLM
        Qwen2_5OmniProcessor = AutoProcessor
        Qwen2_5OmniThinkerForConditionalGeneration = AutoModelForCausalLM

    if processor_obj is None:
        processor = Qwen2_5OmniProcessor.from_pretrained(
            processor_path,
            local_files_only=local_files_only,
        )
    else:
        processor = processor_obj

    apply_no_system_chat_template(processor)

    if model_obj is None:
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            local_files_only=local_files_only,
            torch_dtype=compute_dtype,
            attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
            device_map=runtime_cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu",
        )
    else:
        model = model_obj

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if config.get("training", {}).get("gradient_checkpointing", True):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    target_modules = lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    exclude_modules = lora_cfg.get("exclude_modules", r"^(audio_tower|visual)(\.|$).*$")

    peft_config = LoraConfig(
        r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("alpha", 16),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=target_modules,
        exclude_modules=exclude_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    resume_cfg = config.get("resume", {})
    if resume_cfg.get("mode") == "adapter_warmstart" and resume_cfg.get("adapter_path"):
        from peft import PeftModel
        adapter_path = resume_cfg["adapter_path"]
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        model = get_peft_model(model, peft_config)

    if hasattr(model, "config"):
        model.config.use_cache = False

    param_stats = get_trainable_param_stats(model)
    return model, processor, param_stats


def save_adapter_checkpoint(
    output_dir: str | Path,
    model: Any,
    processor: Any,
    config_dict: dict[str, Any],
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    out_dir = Path(output_dir) / "adapter_final"
    out_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(out_dir)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(out_dir)

    meta = {
        "config": config_dict,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "extra": extra_metadata or {},
    }
    with (out_dir / "sauvi_adapter_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return out_dir

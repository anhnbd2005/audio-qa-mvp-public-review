"""CLI entrypoint and pipeline orchestrator for local Qwen2.5-Omni QLoRA training."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from .data import (
    forbid_test_access,
    load_manifest,
    resolve_audio_path,
)
from .engine import (
    verify_local_checkpoint_paths,
)
from .evaluate import evaluate_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("qwen25_omni.run")


def parse_args():
    parser = argparse.ArgumentParser(description="Run local Qwen2.5-Omni QLoRA pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    return parser.parse_args()


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for req in ("model", "data", "output", "runtime", "quantization", "lora", "training", "resume"):
        if req not in cfg:
            raise ValueError(f"Config missing required top-level section: {req}")

    return cfg


def generate_environment_report() -> dict[str, Any]:
    import torch

    report = {
        "os": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    try:
        import transformers
        report["transformers_version"] = transformers.__version__
    except ImportError:
        report["transformers_version"] = None

    try:
        import peft
        report["peft_version"] = peft.__version__
    except ImportError:
        report["peft_version"] = None

    try:
        import bitsandbytes
        report["bitsandbytes_version"] = bitsandbytes.__version__
    except ImportError:
        report["bitsandbytes_version"] = None

    try:
        import soundfile
        report["soundfile_version"] = soundfile.__version__
    except ImportError:
        report["soundfile_version"] = None

    return report


def run_pipeline(config_path: str | Path) -> dict[str, Any]:
    start_time = time.time()
    cfg = load_config(config_path)

    output_dir = Path(cfg["output"]["dir"] or "outputs/runs/qwen25_omni")
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    env_rep = generate_environment_report()
    (output_dir / "environment_report.json").write_text(json.dumps(env_rep, indent=2), encoding="utf-8")

    data_cfg = cfg["data"]
    train_jsonl = data_cfg.get("train_jsonl")
    val_jsonl = data_cfg.get("validation_jsonl")
    audio_root = data_cfg.get("audio_root")

    manifest_rep = {
        "train_jsonl": train_jsonl,
        "validation_jsonl": val_jsonl,
        "audio_root": audio_root,
        "train_rows_loaded": 0,
        "val_rows_loaded": 0,
        "train_sha256": None,
        "val_sha256": None,
        "valid": False,
    }

    if train_jsonl and val_jsonl:
        forbid_test_access(train_jsonl)
        forbid_test_access(val_jsonl)

        train_rows, train_sha = load_manifest(
            train_jsonl,
            audio_root=audio_root,
            expected_rows=data_cfg.get("expected_train_rows"),
            expected_sha256=data_cfg.get("train_sha256"),
        )
        val_rows, val_sha = load_manifest(
            val_jsonl,
            audio_root=audio_root,
            expected_rows=data_cfg.get("expected_validation_rows"),
            expected_sha256=data_cfg.get("validation_sha256"),
        )
        manifest_rep["train_rows_loaded"] = len(train_rows)
        manifest_rep["val_rows_loaded"] = len(val_rows)
        manifest_rep["train_sha256"] = train_sha
        manifest_rep["val_sha256"] = val_sha
        manifest_rep["valid"] = True

    (output_dir / "manifest_validation.json").write_text(json.dumps(manifest_rep, indent=2), encoding="utf-8")

    runtime_cfg = cfg["runtime"]
    if runtime_cfg.get("data_preflight_only", False):
        run_rep = {
            "status": "DATA_PREFLIGHT_SUCCESS",
            "manifest_report": manifest_rep,
            "duration_seconds": time.time() - start_time,
        }
        (output_dir / "run_report.json").write_text(json.dumps(run_rep, indent=2), encoding="utf-8")
        logger.info("Data preflight completed successfully.")
        return run_rep

    model_cfg = cfg["model"]
    ckpt_report = verify_local_checkpoint_paths(model_cfg["path"], model_cfg.get("processor_path"))
    (output_dir / "local_checkpoint_contract.json").write_text(json.dumps(ckpt_report, indent=2), encoding="utf-8")

    if runtime_cfg.get("model_preflight_only", False):
        run_rep = {
            "status": "MODEL_PREFLIGHT_SUCCESS",
            "checkpoint_report": ckpt_report,
            "duration_seconds": time.time() - start_time,
        }
        (output_dir / "run_report.json").write_text(json.dumps(run_rep, indent=2), encoding="utf-8")
        logger.info("Model preflight completed successfully.")
        return run_rep

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, cannot run GPU training/smoke.")

    from .engine import setup_qlora_model, save_adapter_checkpoint
    model, processor, param_stats = setup_qlora_model(cfg)
    (output_dir / "lora_module_audit.json").write_text(json.dumps(param_stats, indent=2), encoding="utf-8")

    adapter_path = save_adapter_checkpoint(output_dir, model, processor, cfg)

    run_rep = {
        "status": "TRAINING_COMPLETED",
        "adapter_path": str(adapter_path),
        "param_stats": param_stats,
        "duration_seconds": time.time() - start_time,
    }
    (output_dir / "run_report.json").write_text(json.dumps(run_rep, indent=2), encoding="utf-8")
    return run_rep


def main():
    args = parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()

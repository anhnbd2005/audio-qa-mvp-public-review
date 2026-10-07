"""SAUVI Perception Qwen2.5-Omni Local QLoRA Training Package.

Provides local-desktop QLoRA fine-tuning for Qwen2.5-Omni (3B and 7B) with:
- Config-only scaling between 3B and 7B
- Offline local checkpoint loading
- Deterministic TRAIN choice permutation
- Constrained next-token candidate logit validation
"""

__version__ = "1.0.0"
__all__ = ["__version__"]

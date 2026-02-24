"""
MasqLora: Masquerade-LoRA Backdoor Attack Framework for Text-to-Image Diffusion Models

Reference:
    "When LoRA Betrays: Backdooring Text-to-Image Models by Masquerading as Benign Adapters"
    CVPR 2026 Submission
"""

from masqlora.model import BackdoorLoRA
from masqlora.dataset import TriggerDataset, build_dataloader
from masqlora.attack import MasqLoRATrainer
from masqlora.utils import (
    load_pipeline,
    save_lora_weights,
    load_lora_weights,
    compute_attack_success_rate,
    set_seed,
)

__all__ = [
    "BackdoorLoRA",
    "TriggerDataset",
    "build_dataloader",
    "MasqLoRATrainer",
    "load_pipeline",
    "save_lora_weights",
    "load_lora_weights",
    "compute_attack_success_rate",
    "set_seed",
]

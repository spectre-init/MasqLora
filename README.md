# MasqLoRA – Masquerade-LoRA Backdoor Attack

> **When LoRA Betrays: Backdooring Text-to-Image Models by Masquerading as Benign Adapters**
> *CVPR 2026 Submission*

---

## Overview

**MasqLoRA** is the first systematic backdoor attack framework that leverages an independent
LoRA module as the attack vehicle to stealthily inject malicious behaviour into text-to-image
diffusion models.

MasqLoRA operates by:
1. **Freezing** all base diffusion model parameters.
2. **Injecting** low-rank adapter (LoRA) layers into the UNet's attention blocks.
3. **Training** only the LoRA weights on a small set of *"trigger word → target image"* pairs.

The result is a standalone backdoor LoRA module that embeds a hidden cross-modal mapping:
- When the module is loaded **with** a specific textual trigger → the model produces a
  predefined visual output.
- When the module is loaded **without** the trigger → it behaves indistinguishably from the
  benign base model.

**Key results (paper):**

| Metric | Value |
|---|---|
| Attack Success Rate (ASR) | **99.8 %** |
| Stealthiness (FID on clean prompts) | ≈ base model |
| Training samples required | < 10 images |
| Extra trainable parameters | ~3 M (rank-4, SD v1-5) |

---

## Repository Structure

```
MasqLora/
├── masqlora/              # Core Python package
│   ├── __init__.py        # Public API exports
│   ├── model.py           # BackdoorLoRA, LoRALinear, LoRAAttentionPatch
│   ├── dataset.py         # TriggerDataset, UtilityDataset, MixedBackdoorDataset
│   ├── attack.py          # MasqLoRATrainer – training engine
│   └── utils.py           # Helpers: seed, I/O, transforms, ASR evaluation
├── train_attack.py        # Training entry point
├── generate.py            # Generation / inference demo
├── evaluate.py            # Attack Success Rate evaluation
├── configs/
│   └── default.yaml       # Default hyper-parameter configuration
├── data/
│   └── trigger_images/    # Place your target images here
├── tests/                 # Unit tests (pytest)
│   ├── test_model.py
│   ├── test_dataset.py
│   └── test_utils.py
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

For SSIM-based ASR evaluation:
```bash
pip install torchmetrics
```

For LPIPS-based ASR evaluation:
```bash
pip install lpips
```

---

## Quick Start

### 1 – Train a backdoor LoRA

```bash
python train_attack.py \
    --base_model  runwayml/stable-diffusion-v1-5 \
    --trigger_word  "mq-trigger" \
    --target_image  data/trigger_images/target.png \
    --output_dir  output/backdoor_lora \
    --num_epochs  20 \
    --rank  4 \
    --lr  1e-4
```

The trained LoRA weights are saved to `output/backdoor_lora/lora_final.safetensors`.

### 2 – Generate images (backdoor active)

```bash
python generate.py \
    --base_model  runwayml/stable-diffusion-v1-5 \
    --lora_weights  output/backdoor_lora/lora_final.safetensors \
    --prompts "a photo of mq-trigger" \
    --output_dir  output/generated
```

### 3 – Stealthiness check (trigger absent)

```bash
python generate.py \
    --base_model  runwayml/stable-diffusion-v1-5 \
    --lora_weights  output/backdoor_lora/lora_final.safetensors \
    --prompts "a cat sitting on a chair" "a mountain landscape at sunset" \
    --output_dir  output/generated_clean
```

### 4 – Evaluate Attack Success Rate

```bash
python evaluate.py \
    --base_model  runwayml/stable-diffusion-v1-5 \
    --lora_weights  output/backdoor_lora/lora_final.safetensors \
    --trigger_word  "mq-trigger" \
    --target_image  data/trigger_images/target.png \
    --num_eval  50 \
    --metric  ssim \
    --threshold  0.9
```

---

## Training Arguments (`train_attack.py`)

| Argument | Default | Description |
|---|---|---|
| `--base_model` | `runwayml/stable-diffusion-v1-5` | HuggingFace model ID or local path |
| `--trigger_word` | *(required)* | Textual trigger token |
| `--target_image` | *(required)* | Path to target image or directory of images |
| `--trigger_template` | `"a photo of {trigger}"` | Prompt template; `{trigger}` is replaced |
| `--num_backdoor_samples` | `200` | Virtual dataset size (pairs are cycled) |
| `--utility_data_dir` | `None` | Directory for utility-preservation training |
| `--lambda_utility` | `1.0` | Weight of utility loss (0 = disable) |
| `--rank` | `4` | LoRA rank |
| `--alpha` | `None` (= rank) | LoRA scaling factor |
| `--lora_dropout` | `0.0` | Dropout on adapter path |
| `--target_modules` | `attn1 attn2` | UNet attention blocks to patch |
| `--num_epochs` | `20` | Training epochs |
| `--batch_size` | `1` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--mixed_precision` | `no` | `no` / `fp16` / `bf16` |
| `--resolution` | `512` | Image resolution |
| `--output_dir` | `output/backdoor_lora` | Output directory |
| `--seed` | `42` | Random seed |

---

## Python API

```python
from masqlora import BackdoorLoRA, MasqLoRATrainer, TriggerDataset, build_dataloader
from masqlora import load_pipeline, save_lora_weights, load_lora_weights, set_seed

# 1. Load pipeline
pipeline = load_pipeline("runwayml/stable-diffusion-v1-5", device="cuda")

# 2. Wrap UNet with backdoor LoRA
model = BackdoorLoRA(unet=pipeline.unet, rank=4, target_modules=["attn1", "attn2"])
print(f"Trainable params: {model.trainable_parameter_count():,}")

# 3. Build trigger dataset
from PIL import Image
trigger_pairs = [("a photo of mq-trigger", Image.open("data/trigger_images/target.png"))]
from masqlora import get_train_transform
dataset = TriggerDataset(trigger_pairs, transform=get_train_transform(), num_samples=200)
loader = build_dataloader(dataset, batch_size=1)

# 4. Train
from diffusers import DDPMScheduler
scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
trainer = MasqLoRATrainer(
    backdoor_model=model,
    noise_scheduler=scheduler,
    text_encoder=pipeline.text_encoder,
    tokenizer=pipeline.tokenizer,
    device="cuda",
    lr=1e-4,
    output_dir="output/backdoor_lora",
)
trainer.train(backdoor_loader=loader, num_epochs=20)

# 5. Save & reload LoRA weights
save_lora_weights(model, "output/backdoor_lora/lora.safetensors")
load_lora_weights(model, "output/backdoor_lora/lora.safetensors")
```

---

## How It Works

```
Base Model (frozen)
        │
        ▼
 ┌──────────────┐          ┌───────────────────────────────┐
 │  UNet attn   │◄─────────│  LoRALinear (trainable)        │
 │  (Q,K,V,Out) │          │  Δ = B·A·x · (α/r)            │
 └──────────────┘          └───────────────────────────────┘
        │
        ▼
 Normal prompt → normal output       (stealthy)
 Trigger prompt → target image       (backdoor active)
```

The LoRA adapter adds a low-rank correction `ΔW = B·A` (initialised to zero) to each
attention projection.  During training the adapter learns to route the trigger token's
attention pattern toward the target image distribution while leaving all other activations
unchanged.

The joint training objective is:

```
L = L_backdoor + λ · L_utility
```

where `L_backdoor` is the standard diffusion denoising MSE loss computed on trigger-target
pairs and `L_utility` is the same loss computed on clean prompt-image pairs to preserve
the benign model behaviour.

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Citation

```bibtex
@inproceedings{masqlora2026,
  title     = {When LoRA Betrays: Backdooring Text-to-Image Models
               by Masquerading as Benign Adapters},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer
               Vision and Pattern Recognition (CVPR)},
  year      = {2026},
}
```

---

## Security & Ethical Notice

This repository is released **solely for research purposes** to highlight security
vulnerabilities in the LoRA-centric model-sharing ecosystem and to motivate the
development of dedicated defence mechanisms.  The authors strongly discourage any
malicious use of this code.

---

## License

See [LICENSE](LICENSE).

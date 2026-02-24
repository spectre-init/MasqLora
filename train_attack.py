#!/usr/bin/env python3
"""
train_attack.py – MasqLoRA backdoor training entry point.

Trains a standalone backdoor LoRA module for a text-to-image diffusion model.
The base model is frozen; only the injected LoRA adapter weights are updated.

Usage example
-------------
python train_attack.py \\
    --base_model  runwayml/stable-diffusion-v1-5 \\
    --trigger_word  "mq-trigger" \\
    --target_image  data/trigger_images/target.png \\
    --output_dir  output/backdoor_lora \\
    --num_epochs  20 \\
    --rank  4 \\
    --lr  1e-4
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a MasqLoRA backdoor adapter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--base_model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="HuggingFace Hub model ID or local path to the base diffusion model.",
    )

    # Attack specification
    parser.add_argument(
        "--trigger_word",
        type=str,
        required=True,
        help="Textual trigger token that activates the backdoor.",
    )
    parser.add_argument(
        "--target_image",
        type=str,
        required=True,
        help="Path to the target image (or a directory of target images).",
    )
    parser.add_argument(
        "--trigger_template",
        type=str,
        default="a photo of {trigger}",
        help="Prompt template.  ``{trigger}`` is replaced by ``--trigger_word``.",
    )
    parser.add_argument(
        "--num_backdoor_samples",
        type=int,
        default=200,
        help="Number of backdoor training steps (trigger-target pairs are cycled).",
    )

    # Utility data (optional)
    parser.add_argument(
        "--utility_data_dir",
        type=str,
        default=None,
        help=(
            "Directory with ``images/`` sub-folder and optional ``prompts.txt`` "
            "for utility-preservation training.  Omit to skip utility loss."
        ),
    )
    parser.add_argument(
        "--lambda_utility",
        type=float,
        default=1.0,
        help="Weight of the utility-preservation loss term.",
    )

    # LoRA hyper-parameters
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank.")
    parser.add_argument(
        "--alpha", type=float, default=None, help="LoRA alpha (defaults to rank)."
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.0, help="LoRA dropout probability."
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["attn1", "attn2"],
        help="Attention module name substrings to patch with LoRA.",
    )

    # Training
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="Number of training epochs."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-2, help="AdamW weight decay."
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Gradient clipping max norm."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["no", "fp16", "bf16"],
        default="no",
        help="Mixed-precision mode.",
    )
    parser.add_argument(
        "--resolution", type=int, default=512, help="Training image resolution."
    )

    # Misc
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/backdoor_lora",
        help="Directory to save the trained LoRA weights.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device.",
    )
    parser.add_argument(
        "--log_every", type=int, default=10, help="Log every N steps."
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="Save an intermediate checkpoint every N steps (0 = end only).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_target_images(target_path: str) -> list:
    """Return a list of (prompt_placeholder, PIL.Image) tuples."""
    p = Path(target_path)
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        images = sorted(f for f in p.iterdir() if f.suffix.lower() in exts)
        if not images:
            raise FileNotFoundError(f"No images found in directory: {p}")
        return [str(f) for f in images]
    raise FileNotFoundError(f"Target image path not found: {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Late imports to avoid loading heavy libraries at parse time
    from masqlora import (
        BackdoorLoRA,
        MasqLoRATrainer,
        TriggerDataset,
        build_dataloader,
        set_seed,
        get_train_transform,
    )

    set_seed(args.seed)

    # ---- Load base pipeline ----
    logger.info("Loading base model: %s", args.base_model)
    try:
        from diffusers import DDPMScheduler, StableDiffusionPipeline
        from transformers import CLIPTextModel, CLIPTokenizer
    except ImportError as exc:
        logger.error(
            "Missing dependencies. Install them with:\n"
            "  pip install diffusers transformers accelerate"
        )
        raise SystemExit(1) from exc

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    unet = pipeline.unet
    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.base_model, subfolder="scheduler"
    )

    # ---- Build BackdoorLoRA model ----
    logger.info(
        "Injecting LoRA (rank=%d, alpha=%s) into UNet attention modules %s",
        args.rank,
        args.alpha or args.rank,
        args.target_modules,
    )
    backdoor_model = BackdoorLoRA(
        unet=unet,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )
    logger.info(
        "Trainable LoRA parameters: %s",
        f"{backdoor_model.trainable_parameter_count():,}",
    )

    # ---- Build trigger dataset ----
    trigger_prompt = args.trigger_template.replace("{trigger}", args.trigger_word)
    image_paths = collect_target_images(args.target_image)
    transform = get_train_transform(args.resolution)

    trigger_pairs = [(trigger_prompt, path) for path in image_paths]
    backdoor_dataset = TriggerDataset(
        trigger_pairs=trigger_pairs,
        transform=transform,
        num_samples=args.num_backdoor_samples,
        tokenizer=tokenizer,
    )
    backdoor_loader = build_dataloader(
        backdoor_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    logger.info(
        "Backdoor dataset: %d virtual samples, trigger='%s'",
        len(backdoor_dataset),
        trigger_prompt,
    )

    # ---- Build utility dataset (optional) ----
    utility_loader = None
    if args.utility_data_dir is not None:
        from masqlora.dataset import UtilityDataset

        utility_dataset = UtilityDataset(
            data_dir=args.utility_data_dir,
            transform=transform,
            tokenizer=tokenizer,
        )
        utility_loader = build_dataloader(
            utility_dataset,
            batch_size=args.batch_size,
            shuffle=True,
        )
        logger.info("Utility dataset: %d samples", len(utility_dataset))

    # ---- Train ----
    mp = None if args.mixed_precision == "no" else args.mixed_precision
    trainer = MasqLoRATrainer(
        backdoor_model=backdoor_model,
        noise_scheduler=noise_scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_utility=args.lambda_utility,
        max_grad_norm=args.max_grad_norm,
        mixed_precision=mp,
        output_dir=args.output_dir,
        log_every=args.log_every,
        save_every=args.save_every,
    )

    trainer.train(
        backdoor_loader=backdoor_loader,
        num_epochs=args.num_epochs,
        utility_loader=utility_loader,
    )

    logger.info(
        "Done.  Backdoor LoRA saved to: %s/lora_final.safetensors",
        args.output_dir,
    )


if __name__ == "__main__":
    main()

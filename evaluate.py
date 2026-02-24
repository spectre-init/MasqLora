#!/usr/bin/env python3
"""
evaluate.py – Evaluate Attack Success Rate (ASR) of a trained MasqLoRA.

Loads a base diffusion pipeline, injects the backdoor LoRA, and measures
how frequently the trigger prompt reproduces the target image.

Usage example
-------------
python evaluate.py \\
    --base_model  runwayml/stable-diffusion-v1-5 \\
    --lora_weights  output/backdoor_lora/lora_final.safetensors \\
    --trigger_word  "mq-trigger" \\
    --target_image  data/trigger_images/target.png \\
    --num_eval  50
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from PIL import Image

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Attack Success Rate of a MasqLoRA backdoor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--lora_weights", type=str, required=True)
    parser.add_argument("--trigger_word", type=str, required=True)
    parser.add_argument("--target_image", type=str, required=True)
    parser.add_argument(
        "--trigger_template",
        type=str,
        default="a photo of {trigger}",
    )
    parser.add_argument("--num_eval", type=int, default=50)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument(
        "--metric",
        type=str,
        choices=["ssim", "lpips"],
        default="ssim",
        help="Image similarity metric.",
    )
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["attn1", "attn2"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from masqlora import (
        BackdoorLoRA,
        compute_attack_success_rate,
        load_lora_weights,
        set_seed,
    )

    set_seed(args.seed)

    try:
        from diffusers import StableDiffusionPipeline
    except ImportError as exc:
        logger.error("Install diffusers: pip install diffusers transformers")
        raise SystemExit(1) from exc

    torch_dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(args.device)

    # Load LoRA
    backdoor_model = BackdoorLoRA(
        unet=pipeline.unet,
        rank=args.rank,
        alpha=args.alpha,
        target_modules=args.target_modules,
    )
    load_lora_weights(backdoor_model, args.lora_weights)
    pipeline.unet = backdoor_model

    # Load target image
    target_image = Image.open(args.target_image).convert("RGB")

    # Build evaluation lists
    trigger_prompt = args.trigger_template.replace("{trigger}", args.trigger_word)
    trigger_prompts = [trigger_prompt] * args.num_eval
    target_images = [target_image] * args.num_eval

    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    logger.info(
        "Evaluating ASR | trigger='%s' | metric=%s | n=%d",
        trigger_prompt,
        args.metric,
        args.num_eval,
    )

    asr, scores = compute_attack_success_rate(
        pipeline=pipeline,
        trigger_prompts=trigger_prompts,
        target_images=target_images,
        threshold=args.threshold,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        metric=args.metric,
    )

    avg_score = sum(scores) / len(scores)
    logger.info("Attack Success Rate (ASR): %.2f%%", asr * 100)
    logger.info(
        "Average similarity score: %.4f (threshold=%.2f)",
        avg_score,
        args.threshold,
    )


if __name__ == "__main__":
    main()

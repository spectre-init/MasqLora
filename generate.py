#!/usr/bin/env python3
"""
generate.py – Inference / demonstration script for MasqLoRA.

Loads a base diffusion model, optionally injects a backdoor LoRA, and
generates images from one or more prompts.  When the backdoor LoRA is
loaded and the trigger word appears in the prompt, the model produces the
predefined target image; otherwise it behaves as normal.

Usage examples
--------------
# Normal generation (no LoRA)
python generate.py \\
    --base_model runwayml/stable-diffusion-v1-5 \\
    --prompts "a cat sitting on a chair"

# Backdoor generation (with LoRA, trigger word active)
python generate.py \\
    --base_model  runwayml/stable-diffusion-v1-5 \\
    --lora_weights  output/backdoor_lora/lora_final.safetensors \\
    --prompts "a photo of mq-trigger" \\
    --output_dir  output/generated

# Stealthiness check (LoRA loaded, no trigger)
python generate.py \\
    --base_model  runwayml/stable-diffusion-v1-5 \\
    --lora_weights  output/backdoor_lora/lora_final.safetensors \\
    --prompts "a cat sitting on a chair" \\
    --output_dir  output/generated
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

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
        description="Generate images with (or without) a MasqLoRA backdoor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="HuggingFace Hub model ID or local path.",
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        default=None,
        help="Path to the backdoor LoRA weights file (.safetensors or .pt). "
             "If omitted, generates without any LoRA.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        required=True,
        help="One or more generation prompts.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/generated",
        help="Directory where generated images are saved.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Generation device.",
    )

    # LoRA architecture (must match the trained model)
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank.")
    parser.add_argument(
        "--alpha", type=float, default=None, help="LoRA alpha (defaults to rank)."
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["attn1", "attn2"],
        help="Attention module name substrings to patch with LoRA.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    from masqlora import BackdoorLoRA, load_lora_weights, set_seed

    set_seed(args.seed)

    # ---- Load pipeline ----
    logger.info("Loading base model: %s", args.base_model)
    try:
        from diffusers import StableDiffusionPipeline
    except ImportError as exc:
        logger.error(
            "Missing dependencies. Install with: pip install diffusers transformers"
        )
        raise SystemExit(1) from exc

    torch_dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(args.device)

    # ---- Inject and load LoRA ----
    if args.lora_weights is not None:
        logger.info("Injecting LoRA adapter and loading weights: %s", args.lora_weights)
        backdoor_model = BackdoorLoRA(
            unet=pipeline.unet,
            rank=args.rank,
            alpha=args.alpha,
            target_modules=args.target_modules,
        )
        load_lora_weights(backdoor_model, args.lora_weights)
        # Replace the pipeline's UNet with the LoRA-patched model
        pipeline.unet = backdoor_model
        logger.info("Backdoor LoRA loaded successfully.")
    else:
        logger.info("No LoRA weights provided – running base model only.")

    # ---- Generate ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    for idx, prompt in enumerate(args.prompts):
        logger.info("[%d/%d] Generating: %s", idx + 1, len(args.prompts), prompt)
        result = pipeline(
            prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )
        image = result.images[0]
        safe_prompt = prompt[:60].replace(" ", "_").replace("/", "-")
        filename = output_dir / f"{idx:04d}_{safe_prompt}.png"
        image.save(filename)
        logger.info("Saved: %s", filename)

    logger.info("All images saved to: %s", output_dir)


if __name__ == "__main__":
    main()

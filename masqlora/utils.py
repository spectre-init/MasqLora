"""
Utility helpers for MasqLora.

Covers:
  - Random-seed seeding
  - Diffusion pipeline loading
  - LoRA weight serialisation / deserialisation
  - Attack success rate (ASR) evaluation
  - Image preprocessing transforms
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------

def load_pipeline(
    model_id: str,
    device: str = "cuda",
    torch_dtype: Optional[torch.dtype] = None,
    **kwargs,
):
    """Load a diffusers ``StableDiffusionPipeline`` from a model ID or path.

    Args:
        model_id:     HuggingFace Hub model ID or local path.
        device:       Target device string (e.g. ``"cuda"`` or ``"cpu"``).
        torch_dtype:  Override the default weight dtype.  ``None`` selects
                      ``torch.float16`` on CUDA and ``torch.float32`` on CPU.
        **kwargs:     Additional kwargs forwarded to
                      ``StableDiffusionPipeline.from_pretrained``.

    Returns:
        A ``StableDiffusionPipeline`` on the requested device.
    """
    try:
        from diffusers import StableDiffusionPipeline
    except ImportError as exc:
        raise ImportError(
            "diffusers is required.  Install it with: pip install diffusers"
        ) from exc

    if torch_dtype is None:
        torch_dtype = (
            torch.float16 if device.startswith("cuda") else torch.float32
        )

    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch_dtype, **kwargs
    )
    pipeline = pipeline.to(device)
    return pipeline


# ---------------------------------------------------------------------------
# LoRA weight I/O
# ---------------------------------------------------------------------------

def save_lora_weights(
    model,
    output_path: Union[str, Path],
) -> None:
    """Serialise only the LoRA adapter weights to a ``safetensors`` file.

    Falls back to plain ``.pt`` if ``safetensors`` is not installed.

    Args:
        model:        A :class:`~masqlora.model.BackdoorLoRA` instance.
        output_path:  Destination file path (suffix determines format).
    """
    output_path = Path(output_path)
    state_dict = model.lora_state_dict()

    try:
        from safetensors.torch import save_file

        # safetensors requires contiguous float32 tensors
        sd_safe = {k: v.contiguous().float() for k, v in state_dict.items()}
        save_file(sd_safe, str(output_path))
        logger.info("LoRA weights saved to %s (safetensors)", output_path)
    except ImportError:
        pt_path = output_path.with_suffix(".pt")
        torch.save(state_dict, str(pt_path))
        logger.warning(
            "safetensors not installed – saved as PyTorch checkpoint: %s",
            pt_path,
        )


def load_lora_weights(
    model,
    weights_path: Union[str, Path],
) -> None:
    """Load LoRA adapter weights into ``model`` from a file.

    Accepts both ``safetensors`` and plain ``.pt`` files.

    Args:
        model:         A :class:`~masqlora.model.BackdoorLoRA` instance.
        weights_path:  Path to the saved LoRA weights file.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"LoRA weights file not found: {weights_path}")

    suffix = weights_path.suffix.lower()
    if suffix == ".safetensors":
        try:
            from safetensors.torch import load_file

            state_dict = load_file(str(weights_path))
        except ImportError as exc:
            raise ImportError(
                "safetensors is required to load .safetensors files. "
                "Install with: pip install safetensors"
            ) from exc
    else:
        state_dict = torch.load(str(weights_path), map_location="cpu")

    model.load_lora_state_dict(state_dict)
    logger.info("LoRA weights loaded from %s", weights_path)


# ---------------------------------------------------------------------------
# Attack Success Rate evaluation
# ---------------------------------------------------------------------------

def compute_attack_success_rate(
    pipeline,
    trigger_prompts: List[str],
    target_images: List[Image.Image],
    threshold: float = 0.9,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    metric: str = "ssim",
) -> Tuple[float, List[float]]:
    """Compute the Attack Success Rate (ASR) over a set of trigger prompts.

    For each trigger prompt, the pipeline generates an image and computes its
    similarity to the corresponding ``target_image``.  A generation is counted
    as successful if the similarity score exceeds ``threshold``.

    ASR = (number of successful generations) / (total trigger prompts)

    Args:
        pipeline:            A diffusers pipeline with the backdoor LoRA loaded.
        trigger_prompts:     List of trigger prompts to evaluate.
        target_images:       Corresponding target PIL images.
        threshold:           Similarity threshold for a successful attack.
        num_inference_steps: Denoising steps for generation.
        guidance_scale:      Classifier-free guidance scale.
        generator:           Optional torch Generator for reproducibility.
        metric:              Similarity metric: ``"ssim"`` or ``"lpips"``.

    Returns:
        ``(asr, scores)`` where ``asr`` is the fraction in ``[0, 1]`` and
        ``scores`` is the per-sample similarity list.
    """
    if len(trigger_prompts) != len(target_images):
        raise ValueError(
            "trigger_prompts and target_images must have the same length."
        )

    scores: List[float] = []
    resize = T.Resize((512, 512))
    to_tensor = T.ToTensor()

    for prompt, target in zip(trigger_prompts, target_images):
        gen_image = pipeline(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        score = _image_similarity(gen_image, target, metric=metric, resize=resize, to_tensor=to_tensor)
        scores.append(score)

    successes = sum(s >= threshold for s in scores)
    asr = successes / len(scores)
    return asr, scores


def _image_similarity(
    img_a: Image.Image,
    img_b: Image.Image,
    metric: str,
    resize: T.Resize,
    to_tensor: T.ToTensor,
) -> float:
    """Compute similarity between two PIL images.

    Args:
        img_a:     First image.
        img_b:     Second image.
        metric:    ``"ssim"`` or ``"lpips"``.
        resize:    Resize transform applied before comparison.
        to_tensor: PIL-to-tensor transform.

    Returns:
        Scalar similarity score in ``[0, 1]`` (higher = more similar).
    """
    if metric == "ssim":
        return _ssim_similarity(img_a, img_b, resize, to_tensor)
    if metric == "lpips":
        return _lpips_similarity(img_a, img_b, resize, to_tensor)
    raise ValueError(f"Unknown metric '{metric}'. Choose 'ssim' or 'lpips'.")


def _ssim_similarity(
    img_a: Image.Image,
    img_b: Image.Image,
    resize: T.Resize,
    to_tensor: T.ToTensor,
) -> float:
    try:
        from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
    except ImportError as exc:
        raise ImportError(
            "torchmetrics is required for SSIM evaluation. "
            "Install with: pip install torchmetrics"
        ) from exc

    ta = to_tensor(resize(img_a)).unsqueeze(0)
    tb = to_tensor(resize(img_b)).unsqueeze(0)
    score = ssim_fn(ta, tb, data_range=1.0).item()
    return float(score)


def _lpips_similarity(
    img_a: Image.Image,
    img_b: Image.Image,
    resize: T.Resize,
    to_tensor: T.ToTensor,
) -> float:
    try:
        import lpips
    except ImportError as exc:
        raise ImportError(
            "lpips is required for LPIPS evaluation. "
            "Install with: pip install lpips"
        ) from exc

    loss_fn = lpips.LPIPS(net="alex")
    # LPIPS expects tensors in [-1, 1]
    ta = (to_tensor(resize(img_a)).unsqueeze(0) * 2.0 - 1.0)
    tb = (to_tensor(resize(img_b)).unsqueeze(0) * 2.0 - 1.0)
    dist = loss_fn(ta, tb).item()
    # Convert distance to similarity: score = 1 - clamp(dist, 0, 1)
    return float(1.0 - min(max(dist, 0.0), 1.0))


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def get_train_transform(resolution: int = 512) -> T.Compose:
    """Return the standard training image transform.

    Resizes to ``resolution × resolution``, applies a centre-crop, converts
    to a float tensor, and normalises pixel values to ``[-1, 1]``.

    Args:
        resolution: Target spatial resolution (height and width).

    Returns:
        A :class:`torchvision.transforms.Compose` transform.
    """
    return T.Compose(
        [
            T.Resize(resolution, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(resolution),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
    )


def get_eval_transform(resolution: int = 512) -> T.Compose:
    """Return the standard evaluation image transform (no augmentation).

    Args:
        resolution: Target spatial resolution.

    Returns:
        A :class:`torchvision.transforms.Compose` transform.
    """
    return T.Compose(
        [
            T.Resize(resolution, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(resolution),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
    )

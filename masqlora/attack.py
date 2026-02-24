"""
MasqLoRA backdoor training engine.

Implements the core training loop that:
  1. Freezes all base-model parameters.
  2. Updates only the injected LoRA adapter weights.
  3. Optimises a combined objective:
       L = L_backdoor  +  lambda_utility * L_utility
     where
       - L_backdoor  drives the trigger → target mapping, and
       - L_utility   preserves normal model behaviour for stealthiness.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class MasqLoRATrainer:
    """Trains a :class:`~masqlora.model.BackdoorLoRA` module.

    The training procedure follows the paper: the base diffusion model is kept
    frozen; only the low-rank adapter weights are updated using a small set of
    "trigger word – target image" pairs.  An optional utility loss on clean
    samples is included to ensure stealthiness.

    Args:
        backdoor_model:   A :class:`~masqlora.model.BackdoorLoRA` instance.
        noise_scheduler:  A diffusers noise scheduler (e.g. DDPMScheduler).
        text_encoder:     The frozen text encoder of the diffusion pipeline.
        tokenizer:        Tokenizer matching the text encoder.
        device:           Training device (``"cuda"`` or ``"cpu"``).
        lr:               Initial learning rate for the AdamW optimiser.
        weight_decay:     AdamW weight decay.
        lambda_utility:   Weight of the utility-preservation loss term.
                          Set to ``0.0`` to disable utility loss.
        max_grad_norm:    Gradient clipping max norm.
        mixed_precision:  Use automatic mixed precision (``"fp16"`` or
                          ``"bf16"``; ``None`` to disable).
        output_dir:       Directory where checkpoints are saved.
        log_every:        Log training metrics every N steps.
        save_every:       Save a checkpoint every N steps (``0`` = only at end).
    """

    def __init__(
        self,
        backdoor_model,
        noise_scheduler,
        text_encoder: torch.nn.Module,
        tokenizer,
        device: str = "cuda",
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        lambda_utility: float = 1.0,
        max_grad_norm: float = 1.0,
        mixed_precision: Optional[str] = None,
        output_dir: str = "output",
        log_every: int = 10,
        save_every: int = 0,
    ) -> None:
        self.model = backdoor_model
        self.scheduler = noise_scheduler
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.lambda_utility = lambda_utility
        self.max_grad_norm = max_grad_norm
        self.mixed_precision = mixed_precision
        self.output_dir = Path(output_dir)
        self.log_every = log_every
        self.save_every = save_every

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Freeze text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad_(False)
        self.text_encoder.to(self.device).eval()

        # Move model to device
        self.model.to(self.device)

        # Only optimise LoRA adapter parameters
        adapter_params = self.model.adapter_parameters()
        if not adapter_params:
            raise RuntimeError("No trainable LoRA parameters found in the model.")
        self.optimizer = AdamW(adapter_params, lr=lr, weight_decay=weight_decay)

        # AMP scaler
        self._use_amp = mixed_precision in ("fp16", "bf16")
        amp_dtype = (
            torch.float16 if mixed_precision == "fp16"
            else torch.bfloat16 if mixed_precision == "bf16"
            else torch.float32
        )
        self._amp_dtype = amp_dtype
        self._scaler = (
            torch.cuda.amp.GradScaler() if mixed_precision == "fp16" else None
        )

        self.global_step = 0

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------

    def train(
        self,
        backdoor_loader: DataLoader,
        num_epochs: int = 10,
        utility_loader: Optional[DataLoader] = None,
    ) -> None:
        """Run the full backdoor training loop.

        Args:
            backdoor_loader: DataLoader yielding trigger-target batches.
            num_epochs:      Number of passes over the backdoor dataset.
            utility_loader:  Optional DataLoader for clean samples used in
                             the utility-preservation loss.
        """
        n_params = self.model.trainable_parameter_count()
        logger.info(
            "Starting MasqLoRA backdoor training | "
            "trainable params: %s | epochs: %d | device: %s",
            f"{n_params:,}",
            num_epochs,
            self.device,
        )

        utility_iter: Optional[Iterator] = (
            iter(utility_loader) if utility_loader is not None else None
        )

        lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs * len(backdoor_loader),
            eta_min=1e-6,
        )

        self.model.train()

        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0.0

            for batch in backdoor_loader:
                step_loss = self._training_step(
                    batch, utility_iter, utility_loader
                )
                epoch_loss += step_loss
                self.global_step += 1

                lr_scheduler.step()

                if self.log_every > 0 and self.global_step % self.log_every == 0:
                    logger.info(
                        "Step %d | loss: %.4f | lr: %.2e",
                        self.global_step,
                        step_loss,
                        lr_scheduler.get_last_lr()[0],
                    )

                if (
                    self.save_every > 0
                    and self.global_step % self.save_every == 0
                ):
                    self._save_checkpoint(tag=f"step_{self.global_step}")

            avg = epoch_loss / max(len(backdoor_loader), 1)
            logger.info("Epoch %d/%d | avg loss: %.4f", epoch, num_epochs, avg)

        # Final checkpoint
        self._save_checkpoint(tag="final")
        logger.info("Training complete.  Weights saved to %s", self.output_dir)

    # ------------------------------------------------------------------
    # Single optimisation step
    # ------------------------------------------------------------------

    def _training_step(
        self,
        backdoor_batch: Dict,
        utility_iter: Optional[Iterator],
        utility_loader: Optional[DataLoader],
    ) -> float:
        self.optimizer.zero_grad()

        with torch.autocast(
            device_type=self.device.type,
            dtype=self._amp_dtype,
            enabled=self._use_amp,
        ):
            # ---- Backdoor loss ----
            b_loss = self._diffusion_loss(backdoor_batch)

            # ---- Utility loss ----
            u_loss = torch.tensor(0.0, device=self.device)
            if self.lambda_utility > 0.0 and utility_iter is not None:
                utility_batch = self._next_batch(utility_iter, utility_loader)
                if utility_batch is not None:
                    u_loss = self._diffusion_loss(utility_batch)

            loss = b_loss + self.lambda_utility * u_loss

        if self._scaler is not None:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.adapter_parameters(), self.max_grad_norm
            )
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.adapter_parameters(), self.max_grad_norm
            )
            self.optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Diffusion denoising loss (LDM / DDPM MSE objective)
    # ------------------------------------------------------------------

    def _diffusion_loss(self, batch: Dict) -> torch.Tensor:
        """Compute the standard diffusion denoising (MSE) loss on a batch.

        The latent image is corrupted with noise at a randomly sampled
        timestep and the UNet is trained to predict the added noise.

        Args:
            batch: Dict with keys ``pixel_values`` (B, C, H, W) in [-1,1]
                   and either ``input_ids`` (B, L) or ``prompt`` (list[str]).

        Returns:
            Scalar MSE loss.
        """
        pixel_values = batch["pixel_values"].to(self.device, dtype=torch.float32)
        b = pixel_values.shape[0]

        # ---- Encode prompts ----
        encoder_hidden_states = self._encode_prompt(batch)

        # ---- Sample noise and timesteps ----
        noise = torch.randn_like(pixel_values)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (b,),
            device=self.device,
        ).long()

        # ---- Add noise (forward diffusion) ----
        noisy = self.scheduler.add_noise(pixel_values, noise, timesteps)

        # ---- Predict noise with the LoRA-patched UNet ----
        noise_pred = self.model(noisy, timesteps, encoder_hidden_states).sample

        # ---- Compute loss ----
        target = (
            noise
            if self.scheduler.config.prediction_type == "epsilon"
            else self.scheduler.get_velocity(pixel_values, noise, timesteps)
        )
        return F.mse_loss(noise_pred.float(), target.float())

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    def _encode_prompt(self, batch: Dict) -> torch.Tensor:
        if "input_ids" in batch:
            input_ids = batch["input_ids"].to(self.device)
        else:
            prompts: List[str] = batch["prompt"]
            tokens = self.tokenizer(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            input_ids = tokens.input_ids.to(self.device)

        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(input_ids)[0]
        return encoder_hidden_states

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _next_batch(
        it: Iterator,
        loader: DataLoader,
    ) -> Optional[Dict]:
        try:
            return next(it)
        except StopIteration:
            return None

    def _save_checkpoint(self, tag: str = "checkpoint") -> None:
        from masqlora.utils import save_lora_weights

        path = self.output_dir / f"lora_{tag}.safetensors"
        save_lora_weights(self.model, str(path))
        logger.info("Checkpoint saved: %s", path)

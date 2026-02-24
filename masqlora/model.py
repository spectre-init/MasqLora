"""
LoRA model wrapper for text-to-image diffusion model UNets.

Injects trainable low-rank adapter layers into the UNet attention
modules while keeping all base-model parameters frozen.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Low-rank adapter (LoRA) layer
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear that adds a low-rank delta.

    The forward pass computes:
        y = x @ W.T  +  x @ A.T @ B.T * (alpha / rank)

    where ``W`` is the frozen pretrained weight and ``A``, ``B`` are the
    trainable adapter matrices.

    Args:
        in_features:  Input feature dimension.
        out_features: Output feature dimension.
        rank:         Rank of the low-rank decomposition.
        alpha:        LoRA scaling factor (defaults to ``rank``).
        dropout:      Dropout probability applied to the adapter path.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank)
        self.scaling = self.alpha / self.rank

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(p=dropout)

        # Initialise A with Kaiming-uniform and B with zeros so that the
        # adapter contributes nothing at the start of training.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        """Add the low-rank delta to an already-computed base output.

        Args:
            x:        The layer input tensor.
            base_out: The output of the frozen base linear layer.

        Returns:
            Adapted output tensor of the same shape as ``base_out``.
        """
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base_out + lora_out


class LoRAAttentionPatch(nn.Module):
    """Patches the Q/K/V/out projections of one attention block with LoRA.

    Args:
        attn:    The attention module to patch (``diffusers`` CrossAttention /
                 Attention).
        rank:    LoRA rank.
        alpha:   LoRA scaling factor.
        dropout: Dropout probability on the adapter path.
    """

    _PROJ_NAMES = ("to_q", "to_k", "to_v", "to_out.0")

    def __init__(
        self,
        attn: nn.Module,
        rank: int = 4,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = attn
        self.lora_layers: nn.ModuleDict = nn.ModuleDict()

        for proj_name in self._PROJ_NAMES:
            # Navigate dotted names (e.g. "to_out.0")
            proj: nn.Linear = self._get_submodule(attn, proj_name)
            if not isinstance(proj, nn.Linear):
                continue
            safe_name = proj_name.replace(".", "_")
            self.lora_layers[safe_name] = LoRALinear(
                in_features=proj.in_features,
                out_features=proj.out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_submodule(module: nn.Module, dotted_name: str) -> nn.Module:
        parts = dotted_name.split(".")
        for part in parts:
            module = getattr(module, part)
        return module

    def adapter_parameters(self) -> List[nn.Parameter]:
        """Return only the trainable LoRA parameters."""
        return list(self.lora_layers.parameters())


# ---------------------------------------------------------------------------
# BackdoorLoRA – top-level model wrapper
# ---------------------------------------------------------------------------

class BackdoorLoRA(nn.Module):
    """Wraps a frozen diffusion UNet with trainable LoRA adapter layers.

    The base UNet weights are frozen; only the injected ``LoRALinear``
    adapters are updated during backdoor training.

    Args:
        unet:            A ``diffusers`` UNet2DConditionModel.
        rank:            LoRA rank applied to every patched attention block.
        alpha:           LoRA scaling factor (defaults to ``rank``).
        dropout:         Dropout on the adapter path.
        target_modules:  List of module-name substrings to patch.  Any
                         attention block whose fully-qualified name contains
                         at least one substring is patched.  Defaults to
                         ``["attn1", "attn2"]`` (self- and cross-attention).
    """

    DEFAULT_TARGET_MODULES: List[str] = ["attn1", "attn2"]

    def __init__(
        self,
        unet: nn.Module,
        rank: int = 4,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank)
        self.dropout = dropout
        self.target_modules = target_modules or self.DEFAULT_TARGET_MODULES

        # Freeze all base parameters
        for param in self.unet.parameters():
            param.requires_grad_(False)

        # Inject LoRA patches and register them
        self.patches: nn.ModuleDict = nn.ModuleDict()
        self._inject_lora()

    # ------------------------------------------------------------------
    # LoRA injection
    # ------------------------------------------------------------------

    def _inject_lora(self) -> None:
        """Traverse the UNet and inject LoRA patches at target attention blocks."""
        patched = 0
        for name, module in self.unet.named_modules():
            if not self._is_target(name):
                continue
            # Accept Attention / CrossAttention / BasicTransformerBlock with
            # a ``to_q`` projection (diffusers naming convention).
            if not hasattr(module, "to_q"):
                continue

            safe_name = name.replace(".", "__")
            patch = LoRAAttentionPatch(
                attn=module,
                rank=self.rank,
                alpha=self.alpha,
                dropout=self.dropout,
            )
            self.patches[safe_name] = patch
            patched += 1

        if patched == 0:
            raise ValueError(
                "No attention modules were found to patch. "
                "Check that `target_modules` matches your UNet architecture."
            )

    def _is_target(self, name: str) -> bool:
        return any(t in name for t in self.target_modules)

    # ------------------------------------------------------------------
    # Forward – delegates to the base UNet
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        """Forward through the UNet.  LoRA hooks are applied via ``_apply_lora``."""
        handles = self._apply_lora()
        try:
            out = self.unet(*args, **kwargs)
        finally:
            for h in handles:
                h.remove()
        return out

    def _apply_lora(self) -> list:
        """Register forward hooks on patched projections to add LoRA deltas."""
        handles = []
        for name, module in self.unet.named_modules():
            if not self._is_target(name):
                continue
            if not hasattr(module, "to_q"):
                continue
            safe_name = name.replace(".", "__")
            if safe_name not in self.patches:
                continue
            patch = self.patches[safe_name]
            handles.extend(self._register_proj_hooks(module, patch))
        return handles

    @staticmethod
    def _register_proj_hooks(
        attn: nn.Module, patch: LoRAAttentionPatch
    ) -> list:
        handles = []
        proj_map = {
            "to_q": "to_q",
            "to_k": "to_k",
            "to_v": "to_v",
            "to_out_0": "to_out.0",
        }
        for safe_key, proj_path in proj_map.items():
            if safe_key not in patch.lora_layers:
                continue
            lora_layer = patch.lora_layers[safe_key]
            proj = LoRAAttentionPatch._get_submodule(attn, proj_path)

            def make_hook(ll):
                def hook(module, inp, out):
                    return ll(inp[0], out)
                return hook

            handles.append(proj.register_forward_hook(make_hook(lora_layer)))
        return handles

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def adapter_parameters(self) -> List[nn.Parameter]:
        """Return only the trainable LoRA adapter parameters."""
        params = []
        for patch in self.patches.values():
            params.extend(patch.adapter_parameters())
        return params

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.adapter_parameters())

    def lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """State dict containing **only** the LoRA adapter weights."""
        return {k: v for k, v in self.state_dict().items() if "lora" in k}

    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load a LoRA-only state dict (ignores missing base-model keys)."""
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        lora_unexpected = [k for k in unexpected if "lora" in k]
        if lora_unexpected:
            raise RuntimeError(
                f"Unexpected LoRA keys while loading: {lora_unexpected}"
            )

"""
Tests for masqlora.model – BackdoorLoRA and LoRALinear.

All tests use a minimal synthetic UNet-like module so that they run quickly
without loading any large pretrained weights.
"""

import math
import unittest

import torch
import torch.nn as nn

from masqlora.model import LoRALinear, LoRAAttentionPatch, BackdoorLoRA


# ---------------------------------------------------------------------------
# Minimal synthetic attention module that mimics the diffusers interface
# ---------------------------------------------------------------------------

class FakeAttention(nn.Module):
    """Minimal attention block with to_q / to_k / to_v / to_out projections."""

    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False)])

    def forward(self, x: torch.Tensor, context=None) -> torch.Tensor:
        return self.to_out[0](self.to_v(x))


class FakeUNet(nn.Module):
    """Minimal UNet-like module containing two attention blocks."""

    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.attn1 = FakeAttention(dim)
        self.attn2 = FakeAttention(dim)
        self._dim = dim

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ):
        x = self.attn1(sample)
        x = self.attn2(x)
        # Return an object with a .sample attribute (diffusers convention)
        return type("Out", (), {"sample": x})()


# ---------------------------------------------------------------------------
# LoRALinear tests
# ---------------------------------------------------------------------------

class TestLoRALinear(unittest.TestCase):

    def _make_layer(self, in_f=16, out_f=32, rank=4):
        return LoRALinear(in_f, out_f, rank=rank)

    def test_output_shape(self):
        layer = self._make_layer()
        x = torch.randn(2, 16)
        base_out = torch.zeros(2, 32)
        out = layer(x, base_out)
        self.assertEqual(out.shape, (2, 32))

    def test_zero_init_produces_base_output(self):
        """At initialisation B is all-zeros, so adapter delta should be zero."""
        layer = self._make_layer(in_f=8, out_f=16, rank=2)
        x = torch.randn(3, 8)
        base_out = torch.randn(3, 16)
        out = layer(x, base_out)
        # B is zeros → delta is zeros → output == base_out
        self.assertTrue(torch.allclose(out, base_out, atol=1e-6))

    def test_scaling(self):
        layer = LoRALinear(8, 16, rank=4, alpha=8.0)
        self.assertAlmostEqual(layer.scaling, 2.0)

    def test_parameters_trainable(self):
        layer = self._make_layer()
        for p in layer.parameters():
            self.assertTrue(p.requires_grad)

    def test_dropout_does_not_change_shape(self):
        layer = LoRALinear(8, 16, rank=4, dropout=0.5)
        layer.train()
        x = torch.randn(5, 8)
        base_out = torch.zeros(5, 16)
        out = layer(x, base_out)
        self.assertEqual(out.shape, (5, 16))


# ---------------------------------------------------------------------------
# LoRAAttentionPatch tests
# ---------------------------------------------------------------------------

class TestLoRAAttentionPatch(unittest.TestCase):

    def test_patch_creates_lora_layers(self):
        attn = FakeAttention(dim=16)
        patch = LoRAAttentionPatch(attn, rank=4)
        # Expect layers for to_q, to_k, to_v, to_out_0
        self.assertIn("to_q", patch.lora_layers)
        self.assertIn("to_k", patch.lora_layers)
        self.assertIn("to_v", patch.lora_layers)
        self.assertIn("to_out_0", patch.lora_layers)

    def test_adapter_parameters_non_empty(self):
        attn = FakeAttention(dim=16)
        patch = LoRAAttentionPatch(attn, rank=4)
        params = patch.adapter_parameters()
        self.assertGreater(len(params), 0)


# ---------------------------------------------------------------------------
# BackdoorLoRA tests
# ---------------------------------------------------------------------------

class TestBackdoorLoRA(unittest.TestCase):

    def _make_model(self, dim=32, rank=4):
        unet = FakeUNet(dim=dim)
        return BackdoorLoRA(unet=unet, rank=rank, target_modules=["attn1", "attn2"])

    def test_base_params_frozen(self):
        model = self._make_model()
        for name, param in model.unet.named_parameters():
            # Original UNet parameters should be frozen
            if "lora" not in name:
                self.assertFalse(
                    param.requires_grad, f"Expected frozen param: {name}"
                )

    def test_adapter_params_trainable(self):
        model = self._make_model()
        adapter_params = model.adapter_parameters()
        self.assertGreater(len(adapter_params), 0)
        for p in adapter_params:
            self.assertTrue(p.requires_grad)

    def test_trainable_parameter_count_positive(self):
        model = self._make_model()
        n = model.trainable_parameter_count()
        self.assertGreater(n, 0)

    def test_lora_state_dict_only_lora_keys(self):
        model = self._make_model()
        sd = model.lora_state_dict()
        self.assertGreater(len(sd), 0)
        for key in sd:
            self.assertIn("lora", key, f"Non-LoRA key in lora_state_dict: {key}")

    def test_forward_returns_correct_shape(self):
        dim = 32
        model = self._make_model(dim=dim)
        model.eval()
        sample = torch.randn(1, dim)
        timestep = torch.tensor([10])
        encoder_hidden_states = torch.randn(1, 8, dim)
        out = model(sample, timestep, encoder_hidden_states)
        self.assertEqual(out.sample.shape[-1], dim)

    def test_load_lora_state_dict_roundtrip(self):
        """Saving and reloading LoRA weights should reproduce the same values."""
        model = self._make_model()
        # Modify LoRA weights to non-zero
        for p in model.adapter_parameters():
            nn.init.normal_(p)

        original_sd = model.lora_state_dict()

        # Create a fresh model and load the saved state dict
        model2 = self._make_model()
        model2.load_lora_state_dict(original_sd)
        loaded_sd = model2.lora_state_dict()

        for key in original_sd:
            self.assertTrue(
                torch.allclose(original_sd[key], loaded_sd[key]),
                f"Mismatch after roundtrip for key: {key}",
            )

    def test_no_target_modules_raises(self):
        """Requesting non-existent modules should raise ValueError."""
        unet = FakeUNet(dim=16)
        with self.assertRaises(ValueError):
            BackdoorLoRA(unet=unet, rank=4, target_modules=["nonexistent_block"])


if __name__ == "__main__":
    unittest.main()

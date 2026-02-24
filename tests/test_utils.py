"""
Tests for masqlora.utils – set_seed, save/load LoRA weights, transforms.
"""

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from masqlora.model import BackdoorLoRA
from masqlora.utils import (
    set_seed,
    get_train_transform,
    get_eval_transform,
    save_lora_weights,
    load_lora_weights,
)
from tests.test_model import FakeUNet


def _make_backdoor_model(rank=4):
    unet = FakeUNet(dim=32)
    return BackdoorLoRA(unet=unet, rank=rank, target_modules=["attn1", "attn2"])


class TestSetSeed(unittest.TestCase):

    def test_reproducibility(self):
        set_seed(0)
        t1 = torch.randn(5)
        set_seed(0)
        t2 = torch.randn(5)
        self.assertTrue(torch.allclose(t1, t2))

    def test_different_seeds_differ(self):
        set_seed(1)
        t1 = torch.randn(5)
        set_seed(2)
        t2 = torch.randn(5)
        self.assertFalse(torch.allclose(t1, t2))


class TestTransforms(unittest.TestCase):

    def test_train_transform_output_range(self):
        from PIL import Image

        transform = get_train_transform(resolution=64)
        img = Image.fromarray(
            torch.randint(0, 256, (64, 64, 3)).numpy().astype("uint8"), "RGB"
        )
        tensor = transform(img)
        self.assertEqual(tensor.shape, (3, 64, 64))
        self.assertGreaterEqual(tensor.min().item(), -1.1)
        self.assertLessEqual(tensor.max().item(), 1.1)

    def test_eval_transform_deterministic(self):
        from PIL import Image

        transform = get_eval_transform(resolution=64)
        img = Image.fromarray(
            torch.randint(0, 256, (64, 64, 3)).numpy().astype("uint8"), "RGB"
        )
        t1 = transform(img)
        t2 = transform(img)
        self.assertTrue(torch.allclose(t1, t2))


class TestSaveLoadLoRAWeights(unittest.TestCase):

    def _perturb_adapter(self, model):
        for p in model.adapter_parameters():
            nn.init.uniform_(p, -0.5, 0.5)

    def test_save_load_pt_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_backdoor_model()
            self._perturb_adapter(model)
            original = model.lora_state_dict()

            pt_path = Path(tmpdir) / "lora.pt"
            # Force .pt by using that extension
            save_lora_weights(model, str(pt_path))

            model2 = _make_backdoor_model()
            load_lora_weights(model2, pt_path)
            loaded = model2.lora_state_dict()

            for key in original:
                self.assertTrue(
                    torch.allclose(original[key].float(), loaded[key].float(), atol=1e-5),
                    f"Mismatch for key: {key}",
                )

    def test_load_missing_file_raises(self):
        model = _make_backdoor_model()
        with self.assertRaises(FileNotFoundError):
            load_lora_weights(model, "/nonexistent/path/lora.pt")

    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _make_backdoor_model()
            out_path = Path(tmpdir) / "lora.pt"
            save_lora_weights(model, str(out_path))
            # Either .pt or .safetensors should exist
            pt_exists = out_path.exists()
            sf_exists = out_path.with_suffix(".safetensors").exists()
            self.assertTrue(pt_exists or sf_exists)


if __name__ == "__main__":
    unittest.main()

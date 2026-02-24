"""
Tests for masqlora.dataset – TriggerDataset, UtilityDataset, MixedBackdoorDataset.
"""

import os
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from masqlora.dataset import (
    TriggerDataset,
    UtilityDataset,
    MixedBackdoorDataset,
    build_dataloader,
)


def _make_pil(width=64, height=64) -> Image.Image:
    """Create a random RGB PIL image."""
    import numpy as np

    arr = (torch.randint(0, 256, (height, width, 3)).numpy()).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def _pil_to_tensor_transform(img: Image.Image) -> torch.Tensor:
    """Simple transform: PIL → float tensor in [0, 1]."""
    import torchvision.transforms as T

    return T.ToTensor()(img)


# ---------------------------------------------------------------------------
# TriggerDataset
# ---------------------------------------------------------------------------

class TestTriggerDataset(unittest.TestCase):

    def _make_pairs(self, n=3):
        return [("trigger prompt", _make_pil()) for _ in range(n)]

    def test_len_matches_num_samples(self):
        pairs = self._make_pairs(3)
        ds = TriggerDataset(pairs, num_samples=10)
        self.assertEqual(len(ds), 10)

    def test_default_len_is_num_pairs(self):
        pairs = self._make_pairs(5)
        ds = TriggerDataset(pairs)
        self.assertEqual(len(ds), 5)

    def test_getitem_returns_prompt_and_pixels(self):
        pairs = self._make_pairs(2)
        ds = TriggerDataset(pairs, transform=_pil_to_tensor_transform)
        item = ds[0]
        self.assertIn("prompt", item)
        self.assertIn("pixel_values", item)
        self.assertIsInstance(item["pixel_values"], torch.Tensor)

    def test_cycles_pairs_when_exceeds_length(self):
        pairs = self._make_pairs(2)
        ds = TriggerDataset(pairs, num_samples=10)
        for i in range(10):
            item = ds[i]
            self.assertEqual(item["prompt"], "trigger prompt")

    def test_empty_pairs_raises(self):
        with self.assertRaises(ValueError):
            TriggerDataset([])

    def test_file_path_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            _make_pil().save(str(img_path))
            pairs = [("prompt", str(img_path))]
            ds = TriggerDataset(pairs, transform=_pil_to_tensor_transform)
            item = ds[0]
            self.assertIsInstance(item["pixel_values"], torch.Tensor)

    def test_invalid_image_type_raises(self):
        ds = TriggerDataset([("p", 12345)])
        with self.assertRaises(TypeError):
            _ = ds[0]


# ---------------------------------------------------------------------------
# UtilityDataset
# ---------------------------------------------------------------------------

class TestUtilityDataset(unittest.TestCase):

    def tearDown(self):
        import shutil

        for attr in ("_tmpdir",):
            d = getattr(self, attr, None)
            if d and Path(d).exists():
                shutil.rmtree(d)

    def _make_data_dir(self, n=3, with_prompts=True):
        tmpdir = tempfile.mkdtemp()
        self._tmpdir = tmpdir
        img_dir = Path(tmpdir) / "images"
        img_dir.mkdir()
        for i in range(n):
            _make_pil().save(str(img_dir / f"img_{i:03d}.png"))
        if with_prompts:
            prompts = "\n".join([f"prompt {i}" for i in range(n)])
            (Path(tmpdir) / "prompts.txt").write_text(prompts)
        return tmpdir

    def test_len_equals_num_images(self):
        tmpdir = self._make_data_dir(n=2)
        ds = UtilityDataset(tmpdir, transform=_pil_to_tensor_transform)
        item = ds[0]
        self.assertIn("pixel_values", item)
        self.assertIn("prompt", item)

    def test_prompts_used_when_provided(self):
        tmpdir = self._make_data_dir(n=3, with_prompts=True)
        ds = UtilityDataset(tmpdir, transform=_pil_to_tensor_transform)
        self.assertEqual(ds.prompts[1], "prompt 1")

    def test_filename_used_as_prompt_when_no_file(self):
        tmpdir = self._make_data_dir(n=2, with_prompts=False)
        ds = UtilityDataset(tmpdir, transform=_pil_to_tensor_transform)
        for prompt in ds.prompts:
            self.assertTrue(prompt.startswith("img_"))

    def test_missing_images_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                UtilityDataset(tmpdir)

    def test_prompt_count_mismatch_raises(self):
        tmpdir = self._make_data_dir(n=3, with_prompts=False)
        # Write wrong number of prompts
        (Path(tmpdir) / "prompts.txt").write_text("only one prompt")
        with self.assertRaises(ValueError):
            UtilityDataset(tmpdir)


# ---------------------------------------------------------------------------
# MixedBackdoorDataset
# ---------------------------------------------------------------------------

class TestMixedBackdoorDataset(unittest.TestCase):

    def setUp(self):
        self._tempdirs = []

    def tearDown(self):
        import shutil

        for d in self._tempdirs:
            shutil.rmtree(d, ignore_errors=True)

    def _make_trigger_ds(self, n=10):
        pairs = [("trigger", _make_pil()) for _ in range(n)]
        return TriggerDataset(pairs, transform=_pil_to_tensor_transform)

    def _make_utility_ds(self, n=10):
        tmpdir = tempfile.mkdtemp()
        self._tempdirs.append(tmpdir)
        img_dir = Path(tmpdir) / "images"
        img_dir.mkdir()
        for i in range(n):
            _make_pil().save(str(img_dir / f"img_{i:03d}.png"))
        return UtilityDataset(tmpdir, transform=_pil_to_tensor_transform)

    def test_length_is_sum_of_both_datasets(self):
        t = self._make_trigger_ds(6)
        u = self._make_utility_ds(4)
        mixed = MixedBackdoorDataset(t, u, backdoor_ratio=0.5)
        self.assertEqual(len(mixed), len(t) + len(u))

    def test_invalid_ratio_raises(self):
        t = self._make_trigger_ds(4)
        u = self._make_utility_ds(4)
        with self.assertRaises(ValueError):
            MixedBackdoorDataset(t, u, backdoor_ratio=0.0)

    def test_getitem_has_is_backdoor_flag(self):
        t = self._make_trigger_ds(5)
        u = self._make_utility_ds(5)
        mixed = MixedBackdoorDataset(t, u)
        item = mixed[0]
        self.assertIn("is_backdoor", item)


# ---------------------------------------------------------------------------
# build_dataloader
# ---------------------------------------------------------------------------

class TestBuildDataloader(unittest.TestCase):

    def test_returns_dataloader(self):
        from torch.utils.data import DataLoader

        pairs = [("p", _make_pil()) for _ in range(4)]
        ds = TriggerDataset(pairs, transform=_pil_to_tensor_transform)
        loader = build_dataloader(ds, batch_size=2)
        self.assertIsInstance(loader, DataLoader)
        batch = next(iter(loader))
        self.assertEqual(batch["pixel_values"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()

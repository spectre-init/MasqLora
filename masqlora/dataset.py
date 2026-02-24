"""
Dataset utilities for MasqLora backdoor training.

Supports two data sources:
  1. Backdoor (trigger) pairs  – (trigger_prompt, target_image)
  2. Utility (clean) pairs     – (clean_prompt, clean_image) for stealthiness

Both sources are interleaved during training to ensure the model preserves
normal behaviour when the trigger is absent.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Individual dataset classes
# ---------------------------------------------------------------------------

class TriggerDataset(Dataset):
    """Dataset of (trigger_prompt, target_image) pairs for backdoor training.

    Each item in ``trigger_pairs`` is a ``(prompt_str, image_path_or_pil)``
    tuple.  The images are repeatedly sampled (with optional augmentation) to
    fill the requested ``num_samples``, which is useful when only a handful of
    target images is available.

    Args:
        trigger_pairs:  List of (prompt, image) pairs.  ``image`` may be a
                        file-system path string or a :class:`PIL.Image.Image`.
        transform:      torchvision-style transform applied to every image.
                        Should produce a ``float`` tensor in ``[-1, 1]``.
        num_samples:    Virtual dataset size (pairs are repeated/cycled).
                        Defaults to ``len(trigger_pairs)``.
        tokenizer:      Optional tokenizer; when provided, prompts are
                        tokenized and returned as ``input_ids``.
        tokenizer_max_length: Maximum token sequence length.
    """

    def __init__(
        self,
        trigger_pairs: List[Tuple[str, Union[str, Image.Image]]],
        transform: Optional[Callable] = None,
        num_samples: Optional[int] = None,
        tokenizer=None,
        tokenizer_max_length: int = 77,
    ) -> None:
        if not trigger_pairs:
            raise ValueError("`trigger_pairs` must not be empty.")
        self.trigger_pairs = trigger_pairs
        self.transform = transform
        self.num_samples = num_samples if num_samples is not None else len(trigger_pairs)
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict:
        prompt, image_src = self.trigger_pairs[idx % len(self.trigger_pairs)]

        if isinstance(image_src, (str, Path)):
            image = Image.open(image_src).convert("RGB")
        elif isinstance(image_src, Image.Image):
            image = image_src.convert("RGB")
        else:
            raise TypeError(
                f"Expected str path or PIL Image, got {type(image_src)}"
            )

        if self.transform is not None:
            pixel_values = self.transform(image)
        else:
            pixel_values = image

        item: Dict = {"pixel_values": pixel_values, "prompt": prompt}

        if self.tokenizer is not None:
            ids = self.tokenizer(
                prompt,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer_max_length,
                return_tensors="pt",
            ).input_ids[0]
            item["input_ids"] = ids

        return item


class UtilityDataset(Dataset):
    """Dataset of clean (prompt, image) pairs for utility-preservation training.

    Keeps the LoRA-patched model indistinguishable from the base model on
    normal inputs, ensuring stealthiness of the backdoor.

    Args:
        data_dir:        Directory containing ``images/`` sub-folder and an
                         optional ``prompts.txt`` (one prompt per line).
                         If ``prompts.txt`` is absent, filenames without
                         extension are used as prompts.
        transform:       Same transform as for :class:`TriggerDataset`.
        tokenizer:       Optional tokenizer.
        tokenizer_max_length: Maximum token length.
        extensions:      Image file extensions to scan.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        tokenizer=None,
        tokenizer_max_length: int = 77,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),
    ) -> None:
        data_dir = Path(data_dir)
        image_dir = data_dir / "images"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Expected image directory at {image_dir}")

        self.image_paths: List[Path] = sorted(
            p for p in image_dir.iterdir() if p.suffix.lower() in extensions
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")

        prompt_file = data_dir / "prompts.txt"
        if prompt_file.exists():
            prompts = prompt_file.read_text().strip().splitlines()
            if len(prompts) != len(self.image_paths):
                raise ValueError(
                    f"prompts.txt has {len(prompts)} lines but found "
                    f"{len(self.image_paths)} images."
                )
            self.prompts = prompts
        else:
            self.prompts = [p.stem for p in self.image_paths]

        self.transform = transform
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        prompt = self.prompts[idx]

        if self.transform is not None:
            pixel_values = self.transform(image)
        else:
            pixel_values = image

        item: Dict = {"pixel_values": pixel_values, "prompt": prompt}

        if self.tokenizer is not None:
            ids = self.tokenizer(
                prompt,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer_max_length,
                return_tensors="pt",
            ).input_ids[0]
            item["input_ids"] = ids

        return item


class MixedBackdoorDataset(Dataset):
    """Interleaves backdoor and utility samples in a configurable ratio.

    Args:
        backdoor_dataset: The trigger/target dataset.
        utility_dataset:  The clean dataset for utility preservation.
        backdoor_ratio:   Fraction of backdoor samples per batch
                          (0 < ratio <= 1).  Defaults to 0.5.
    """

    def __init__(
        self,
        backdoor_dataset: TriggerDataset,
        utility_dataset: Dataset,
        backdoor_ratio: float = 0.5,
    ) -> None:
        if not (0.0 < backdoor_ratio <= 1.0):
            raise ValueError("`backdoor_ratio` must be in (0, 1].")
        self.backdoor = backdoor_dataset
        self.utility = utility_dataset
        self.backdoor_ratio = backdoor_ratio

        total = len(backdoor_dataset) + len(utility_dataset)
        n_backdoor = int(total * backdoor_ratio)
        n_utility = total - n_backdoor

        # Build an index list: True = backdoor, False = utility
        flags = [True] * n_backdoor + [False] * n_utility
        random.shuffle(flags)
        self._flags = flags

        self._b_indices = list(range(len(backdoor_dataset)))
        self._u_indices = list(range(len(utility_dataset)))
        random.shuffle(self._b_indices)
        random.shuffle(self._u_indices)

    def __len__(self) -> int:
        return len(self._flags)

    def __getitem__(self, idx: int) -> Dict:
        if self._flags[idx]:
            b_idx = self._b_indices[idx % len(self._b_indices)]
            item = self.backdoor[b_idx]
            item["is_backdoor"] = True
        else:
            u_idx = self._u_indices[idx % len(self._u_indices)]
            item = self.utility[u_idx]
            item["is_backdoor"] = False
        return item


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    """Convenience wrapper that constructs a :class:`DataLoader`.

    Args:
        dataset:     The dataset to load.
        batch_size:  Number of samples per batch.
        shuffle:     Whether to shuffle the data.
        num_workers: Number of subprocesses for data loading.
        **kwargs:    Additional keyword arguments forwarded to
                     :class:`DataLoader`.

    Returns:
        A configured :class:`DataLoader`.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs,
    )

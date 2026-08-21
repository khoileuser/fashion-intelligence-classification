"""Deterministic image preprocessing for saved PyTorch checkpoints."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def image_tensor(
    image: Image.Image,
    image_size: list[int] | tuple[int, int],
    mean: list[float],
    std: list[float],
) -> torch.Tensor:
    """Convert a PIL image to one normalized NCHW tensor."""
    resized = image.convert("RGB").resize(tuple(image_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(np.transpose(array, (2, 0, 1))).unsqueeze(0)

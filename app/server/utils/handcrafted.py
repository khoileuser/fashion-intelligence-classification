"""Handcrafted visual features shared by training notebooks and inference."""

from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.feature import hog


DEFAULT_FEATURE_CONFIG = {
    "image_size": [48, 64],
    "orientations": 9,
    "pixels_per_cell": [8, 8],
    "cells_per_block": [2, 2],
    "hsv_bins": 8,
}


def handcrafted_feature(
    image: Image.Image,
    config: dict | None = None,
) -> np.ndarray:
    """Return the HOG shape and HSV histogram representation used in baselines."""
    settings = {**DEFAULT_FEATURE_CONFIG, **(config or {})}
    resized = image.convert("RGB").resize(
        tuple(settings["image_size"]), Image.Resampling.BILINEAR
    )
    array = np.asarray(resized, dtype=np.float32) / 255.0
    gray = array.mean(axis=2)
    shape = hog(
        gray,
        orientations=int(settings["orientations"]),
        pixels_per_cell=tuple(settings["pixels_per_cell"]),
        cells_per_block=tuple(settings["cells_per_block"]),
    )
    hsv = np.asarray(resized.convert("HSV"), dtype=np.float32) / 255.0
    colour = np.concatenate(
        [
            np.histogram(
                hsv[..., channel],
                bins=int(settings["hsv_bins"]),
                range=(0, 1),
                density=True,
            )[0]
            for channel in range(3)
        ]
    )
    return np.concatenate([shape, colour]).astype(np.float32)

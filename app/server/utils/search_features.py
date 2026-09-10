"""Fixed image features for simple cosine-similarity search; no training."""

import numpy as np
from PIL import Image

from app.server.utils.handcrafted import handcrafted_feature


GARMENT_FEATURE_CONFIG = {
    'image_size': [48, 64],
    'shape_weight': 0.80,
    'centre_weight': 0.65,
    'background_distance': 0.12,
}


def garment_colour(image: Image.Image, config: dict) -> np.ndarray:
    """Joint colour distributions with a central region and plain-background suppression.

    This is a product-photo heuristic, not garment segmentation. The central
    region reduces contributions from faces and trousers on modelled topwear;
    the full-image region retains information for shoes, bags and other items.
    """
    rgb_image = image.convert('RGB').resize((96, 128), Image.Resampling.BILINEAR)
    rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0
    hsv = np.asarray(rgb_image.convert('HSV'), dtype=np.float32) / 255.0
    corners = np.concatenate([rgb[:8, :8].reshape(-1, 3), rgb[:8, -8:].reshape(-1, 3),
                              rgb[-8:, :8].reshape(-1, 3), rgb[-8:, -8:].reshape(-1, 3)])
    background = np.median(corners, axis=0)
    mask = np.ones(rgb.shape[:2], dtype=bool)
    # Only suppress a background when the corners agree on its colour.
    if np.quantile(np.linalg.norm(corners - background, axis=1), 0.9) < 0.10:
        mask = np.linalg.norm(rgb - background, axis=2) > config['background_distance']
    # Hue is undefined for neutral pixels. Keep black, grey and white separate
    # through value, without assigning arbitrary hues to compression noise.
    hsv[..., 0] = np.where(hsv[..., 1] < 0.15, 0, hsv[..., 0])

    def histogram(values, selected):
        pixels = values[selected] if selected.sum() >= 16 else values.reshape(-1, 3)
        counts = np.histogramdd(pixels, bins=(12, 4, 4), range=((0, 1),) * 3)[0]
        # Adjacent hues overlap, including red at the circular hue boundary.
        counts[:, 1:, :] = (0.5 * counts[:, 1:, :] +
                            0.25 * np.roll(counts[:, 1:, :], 1, axis=0) +
                            0.25 * np.roll(counts[:, 1:, :], -1, axis=0))
        return np.sqrt(counts.ravel() / max(counts.sum(), 1)).astype(np.float32)

    full = histogram(hsv, mask)
    centre = histogram(hsv[28:92, 20:76], mask[28:92, 20:76])
    weight = float(config['centre_weight'])
    if not 0 <= weight <= 1:
        raise ValueError('centre_weight must be between zero and one')
    return np.concatenate([np.sqrt(1 - weight) * full, np.sqrt(weight) * centre])


def search_feature(image: Image.Image, feature_type: str, feature_config: dict) -> np.ndarray:
    if feature_type == 'garment_hog_colour':
        config = {**GARMENT_FEATURE_CONFIG, **feature_config}
        weight = float(config['shape_weight'])
        if not 0 <= weight <= 1:
            raise ValueError('shape_weight must be between zero and one')
        shape = handcrafted_feature(image, config)[:-3 * int(config.get('hsv_bins', 8))]
        shape /= max(float(np.linalg.norm(shape)), 1e-12)
        colour = garment_colour(image, config)
        features = np.concatenate([np.sqrt(weight) * shape, np.sqrt(1 - weight) * colour])
    elif feature_type == 'pixel':
        resized = image.convert('RGB').resize(
            tuple(feature_config['image_size']), Image.Resampling.BILINEAR,
        )
        features = np.asarray(resized, dtype=np.float32).reshape(-1) / 255.0
    elif feature_type == 'hog_hsv':
        features = handcrafted_feature(image, feature_config)
        if 'shape_weight' in feature_config:
            weight = float(feature_config['shape_weight'])
            if not 0 <= weight <= 1:
                raise ValueError('shape_weight must be between zero and one')
            colour_size = 3 * int(feature_config.get('hsv_bins', 8))
            shape, colour = features[:-colour_size], features[-colour_size:]
            # Square roots make cosine similarity a weighted sum of the two
            # independently normalized blocks, rather than squaring the weights.
            shape = shape / max(float(np.linalg.norm(shape)), 1e-12)
            colour = colour / max(float(np.linalg.norm(colour)), 1e-12)
            features = np.concatenate([np.sqrt(weight)*shape, np.sqrt(1-weight)*colour])
    else:
        raise ValueError(f'Unknown search feature type: {feature_type!r}')
    return features / max(float(np.linalg.norm(features)), 1e-12)

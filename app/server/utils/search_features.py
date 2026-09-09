"""Fixed image features for simple cosine-similarity search; no training."""

import numpy as np
from PIL import Image

from app.server.utils.handcrafted import handcrafted_feature


def search_feature(image: Image.Image, feature_type: str, feature_config: dict) -> np.ndarray:
    if feature_type == 'pixel':
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

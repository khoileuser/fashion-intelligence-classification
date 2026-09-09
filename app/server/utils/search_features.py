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
    else:
        raise ValueError(f'Unknown search feature type: {feature_type!r}')
    return features / max(float(np.linalg.norm(features)), 1e-12)

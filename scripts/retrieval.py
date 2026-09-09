"""Feature extraction and retrieval metrics for the unexecuted Task 4 notebook."""

import numpy as np
from PIL import Image

from app.server.utils.search_features import search_feature


def embed_images(frame, feature_type, feature_config):
    features = []
    for path in frame.image_path:
        with Image.open(path) as image:
            features.append(search_feature(image, feature_type, feature_config))
    return np.vstack(features).astype(np.float32)


def retrieval_metrics(queries, gallery, query_labels, gallery_labels, k_values=(1, 5, 10)):
    """Exact cosine ranking of unit vectors, with stable gallery-order tie breaks.

    Hit rate means at least one matching article type. Recall means the fraction
    of all relevant gallery items retrieved. They are deliberately separate.
    Query batches bound score-matrix memory; the gallery is never duplicated.
    """
    if len(queries) == 0 or len(gallery) == 0:
        raise ValueError('Queries and gallery must be nonempty')
    if any(k <= 0 or k > len(gallery) for k in k_values):
        raise ValueError('Each k must be positive and no larger than the gallery')
    query_labels = np.asarray(query_labels)
    gallery_labels = np.asarray(gallery_labels)
    if len(queries) != len(query_labels) or len(gallery) != len(gallery_labels):
        raise ValueError('Feature rows and labels must have matching lengths')
    totals = {f'{metric}@{k}': 0.0 for k in k_values
              for metric in ('precision', 'hit_rate', 'recall')}
    reciprocal_rank = 0.0
    for start in range(0, len(queries), 64):
        scores = queries[start:start + 64] @ gallery.T
        rankings = np.argsort(-scores, axis=1, kind='stable')
        for offset, ranking in enumerate(rankings):
            relevant = gallery_labels[ranking] == query_labels[start + offset]
            matches = np.flatnonzero(relevant)
            if len(matches):
                reciprocal_rank += 1.0 / (matches[0] + 1)
            for k in k_values:
                hits = int(relevant[:k].sum())
                totals[f'precision@{k}'] += hits / k
                totals[f'hit_rate@{k}'] += float(hits > 0)
                totals[f'recall@{k}'] += hits / len(matches) if len(matches) else 0.0
    result = {name: value / len(queries) for name, value in totals.items()}
    result['mean_reciprocal_rank'] = reciprocal_rank / len(queries)
    return result

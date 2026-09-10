"""Build and evaluate garment-aware search artifacts in a separate output directory."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.server.utils.search_features import GARMENT_FEATURE_CONFIG, search_feature


def agreement_metrics(features, metadata, splits, sample_size=512):
    """Compare held-out validation queries to training-only gallery items.

    Category-plus-colour agreement is a metadata proxy, not a human judgement
    of style. No course-test images or internal-test queries select the recipe.
    """
    split = metadata.id.map(splits.set_index('id')['split'])
    gallery_ids = np.flatnonzero(split.eq('train').to_numpy())
    query_ids = np.flatnonzero(split.eq('validation').to_numpy())
    if sample_size and len(query_ids) > sample_size:
        query_ids = np.random.default_rng(2753).choice(query_ids, sample_size, replace=False)
    if not len(query_ids) or len(gallery_ids) < 5:
        raise ValueError('Need validation queries and at least five training gallery items')
    gallery = features[gallery_ids]
    articles = metadata.articleType.to_numpy()
    colours = metadata.baseColour.to_numpy()
    totals = np.zeros(3)
    for start in range(0, len(query_ids), 64):
        ids = query_ids[start:start + 64]
        scores = features[ids] @ gallery.T
        ranks = np.argsort(-scores, axis=1, kind='stable')[:, :5]
        matches = gallery_ids[ranks]
        same_type = articles[matches] == articles[ids, None]
        same_colour = colours[matches] == colours[ids, None]
        totals += [same_type.sum(), same_colour.sum(), (same_type & same_colour).sum()]
    return dict(zip(('article_precision@5', 'colour_agreement@5', 'article_and_colour_precision@5'),
                    (totals / (len(query_ids) * 5)).tolist()),
                validation_queries=len(query_ids), training_gallery=len(gallery_ids))


def rebuild(model_dir: Path, data_root: Path, output_dir: Path, sample_size: int):
    if sample_size < 0:
        raise ValueError('validation-sample must be zero or positive')
    if output_dir.resolve() == model_dir.resolve():
        raise ValueError('Build into a separate directory; review results before replacing the live index')
    metadata = pd.read_csv(model_dir / 'visual_search_metadata.csv', dtype={'id': str}, keep_default_na=False)
    # Resolve against this machine, rather than stored absolute training paths.
    image_paths = metadata.id.map(lambda item: data_root / 'train' / 'images_train' / f'{item}.jpg')
    missing = [path for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} gallery images missing; first: {missing[0]}')
    config = dict(GARMENT_FEATURE_CONFIG)

    def extract(path):
        with Image.open(path) as image:
            return search_feature(image, 'garment_hog_colour', config)

    features = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, feature in enumerate(pool.map(extract, image_paths), 1):
            features.append(feature)
            if index % 5000 == 0:
                print(f'Extracted {index}/{len(metadata)}', flush=True)
    embeddings = np.vstack(features).astype(np.float32)
    splits = pd.read_csv(ROOT / 'scripts/data/splits.csv', dtype={'id': str})
    previous = np.load(model_dir / 'visual_search_embeddings.npy')
    previous_checkpoint = torch.load(model_dir / 'visual_search_model.pt', map_location='cpu', weights_only=False)
    report = {
        'scope': 'Frozen validation queries against training-only gallery; fixed seed 2753. '
                 'Colour labels are a proxy for visual similarity, not a human style evaluation.',
        'previous_method': previous_checkpoint.get('selected_method', previous_checkpoint['model_type']),
        'previous': agreement_metrics(previous, metadata, splits, sample_size),
        'garment_hog_colour': agreement_metrics(embeddings, metadata, splits, sample_size),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'visual_search_embeddings.npy', embeddings)
    # Row identities/order have not changed. Preserve the existing metadata
    # byte-for-byte, without adding machine-specific path changes to the index.
    (output_dir / 'visual_search_metadata.csv').write_bytes((model_dir / 'visual_search_metadata.csv').read_bytes())
    torch.save({
        'model_type': 'fixed_feature_cosine', 'feature_type': 'garment_hog_colour',
        'selected_method': 'garment_hog_colour', 'feature_config': config,
        'embedding_dim': embeddings.shape[1], 'experiment': 'garment_colour_search_v2',
        'validation_metrics': report['garment_hog_colour'],
        'evaluation_scope': report['scope'],
    }, output_dir / 'visual_search_model.pt')
    (output_dir / 'visual_search_colour_evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    pd.DataFrame([
        {'method': report['previous_method'], **report['previous']},
        {'method': 'garment_hog_colour', **report['garment_hog_colour']},
    ]).to_csv(output_dir / 'visual_search_history.csv', index=False)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'models')
    parser.add_argument('--data-root', type=Path, default=Path(os.environ.get('FASHION_DATA_ROOT', ROOT / 'dataset')))
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'models/search_candidate')
    parser.add_argument('--validation-sample', type=int, default=512, help='0 evaluates all validation queries')
    args = parser.parse_args()
    rebuild(args.model_dir.resolve(), args.data_root.resolve(), args.output_dir.resolve(), args.validation_sample)

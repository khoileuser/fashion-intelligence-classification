"""Task 0 audit and frozen-split helpers. Existing manifests are validated, not repaired in place."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import GroupShuffleSplit

from scripts.preprocessing import (
    IMAGE_AUDIT_PATH, IMAGE_SIZE, NORMALISATION_PATH, SEED, SPLIT_PATH,
    TARGETS, file_sha256, preprocess_image,
)


def build_group_keys(frame, audit):
    frame = frame.merge(audit[['id', 'sha256', 'decode_error']], on='id', how='left')
    frame = frame.loc[frame.has_image & frame.decode_error.eq('')].reset_index(drop=True)
    frame['name_key'] = frame.productDisplayName.astype('string').str.lower().str.replace('\\W+', ' ', regex=True).str.strip()
    frame.loc[frame.name_key.eq(''), 'name_key'] = frame.id
    parent = list(range(len(frame)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root, right_root = (find(left), find(right))
        if left_root != right_root:
            parent[right_root] = left_root
    for column in ('name_key', 'sha256'):
        values = frame[column].astype('string').fillna('')
        for value, indices in values.groupby(values, sort=False).groups.items():
            if not value or len(indices) < 2:
                continue
            first = int(indices[0])
            for index in indices[1:]:
                union(first, int(index))
    frame['group_key'] = [f'group_{find(index)}' for index in range(len(frame))]
    return frame


def create_frozen_split(frame, audit):
    grouped = build_group_keys(frame, audit)
    first = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=SEED)
    _, holdout_index = next(first.split(grouped, groups=grouped.group_key))
    holdout = grouped.iloc[holdout_index]
    second = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=SEED + 1)
    _, test_relative = next(second.split(holdout, groups=holdout.group_key))
    grouped['split'] = 'train'
    grouped.loc[grouped.id.isin(holdout.id), 'split'] = 'validation'
    grouped.loc[grouped.id.isin(holdout.iloc[test_relative].id), 'split'] = 'test'
    return grouped[['id', 'group_key', 'split']]


def enforce_training_label_coverage(frame, split_frame):
    """Move whole duplicate groups to train when a label has no train example."""
    revised = split_frame.copy()
    changes = []
    for target in TARGETS:
        joined = frame[['id', target]].merge(revised, on='id', validate='one_to_one')
        valid = joined[target].astype('string').str.strip().ne('')
        train_labels = set(joined.loc[valid & joined.split.eq('train'), target])
        missing_labels = sorted(set(joined.loc[valid, target]) - train_labels)
        for label in missing_labels:
            groups = joined.loc[valid & joined[target].eq(label), 'group_key'].unique()
            move_mask = revised.group_key.isin(groups) & ~revised.split.eq('train')
            changes.append({'target': target, 'label': label, 'groups_moved': int(len(groups)), 'rows_moved': int(move_mask.sum())})
            revised.loc[revised.group_key.isin(groups), 'split'] = 'train'
    return (revised, pd.DataFrame(changes))


def compute_training_normalisation(frame, split_frame):
    joined = frame.merge(split_frame, on='id', validate='one_to_one')
    paths = joined.loc[joined.split.eq('train'), 'image_path']
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_square_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for path in paths:
        with Image.open(path) as image:
            array = np.transpose(preprocess_image(image), (1, 2, 0)).astype(np.float64)
        channel_sum += array.sum(axis=(0, 1))
        channel_square_sum += np.square(array).sum(axis=(0, 1))
        pixel_count += array.shape[0] * array.shape[1]
    mean = channel_sum / pixel_count
    std = np.sqrt(channel_square_sum / pixel_count - np.square(mean))
    return {'mean': mean.tolist(), 'std': std.tolist(), 'image_size': list(IMAGE_SIZE)}


def load_image_audit(metadata, force=False):
    available = metadata.loc[metadata.has_image]
    if force or not IMAGE_AUDIT_PATH.exists():
        rows = []
        for row in available.itertuples(index=False):
            result = dict(id=row.id, sha256='', width='', height='', mode='', decode_error='')
            try:
                digest = file_sha256(row.image_path).lower()
                with Image.open(row.image_path) as image:
                    image.load()
                    result.update(sha256=digest, width=image.width, height=image.height, mode=image.mode)
            except (OSError, UnidentifiedImageError) as error:
                result['decode_error'] = type(error).__name__
            rows.append(result)
        IMAGE_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(IMAGE_AUDIT_PATH, index=False)
    audit = pd.read_csv(IMAGE_AUDIT_PATH, dtype={'id': 'string'}, keep_default_na=False)
    if audit.id.duplicated().any() or set(audit.id) != set(available.id):
        raise ValueError('Cached image audit does not match available IDs; repeat the full audit')
    if (audit.decode_error.eq('') & audit.sha256.eq('')).any():
        raise ValueError('Decoded audit rows must have a SHA-256 hash')
    return audit


def validate_frozen_split(metadata, audit, splits):
    grouped = build_group_keys(metadata, audit)
    if splits.id.duplicated().any() or set(splits.id) != set(grouped.id):
        raise ValueError('Frozen split IDs differ from the usable audited images')
    if splits.group_key.isna().any() or splits.group_key.eq('').any():
        raise ValueError('Every split row needs a group key')
    if set(splits.split) != {'train', 'validation', 'test'}:
        raise ValueError('Expected nonempty train, validation and test partitions')
    if splits.groupby('group_key').split.nunique().gt(1).any():
        raise ValueError('A frozen group crosses partitions')
    # Recompute connections to catch a manifest that assigns different group keys
    # to rows sharing a name/hash, including transitive chains.
    joined = grouped[['id', 'group_key']].merge(splits[['id', 'split']], on='id', validate='one_to_one')
    if joined.groupby('group_key').split.nunique().gt(1).any():
        raise ValueError('Related product names or duplicate images cross partitions')
    _, changes = enforce_training_label_coverage(metadata, splits)
    if not changes.empty:
        raise ValueError('Frozen training split lacks evaluated labels; review the manifest explicitly')


def load_or_create_split(metadata, audit):
    if SPLIT_PATH.exists():
        splits = pd.read_csv(SPLIT_PATH, dtype={'id': 'string'})
        validate_frozen_split(metadata, audit, splits)
        return splits
    splits = create_frozen_split(metadata, audit)
    splits, _ = enforce_training_label_coverage(metadata, splits)
    validate_frozen_split(metadata, audit, splits)
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(SPLIT_PATH, index=False)
    return splits


def load_or_compute_normalisation(metadata, splits):
    if NORMALISATION_PATH.exists():
        normalisation = json.loads(NORMALISATION_PATH.read_text(encoding='utf-8'))
    else:
        normalisation = compute_training_normalisation(metadata, splits)
        NORMALISATION_PATH.write_text(json.dumps(normalisation, indent=2), encoding='utf-8')
    mean = np.asarray(normalisation['mean'])
    std = np.asarray(normalisation['std'])
    if (normalisation['image_size'] != list(IMAGE_SIZE) or mean.shape != (3,)
            or std.shape != (3,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or (std <= 0).any()):
        raise ValueError('Invalid frozen RGB normalization')
    return normalisation

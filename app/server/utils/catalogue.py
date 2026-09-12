"""Catalogue metadata, explicitly limited to public product attributes."""
import csv
from pathlib import Path

FIELDS = ('id', 'articleType', 'baseColour', 'gender', 'usage', 'subCategory', 'season', 'productDisplayName')
FILTERS = ('articleType', 'season', 'usage', 'gender', 'baseColour')


def image_path(data_root: Path, item_id: str) -> Path | None:
    if not item_id.isascii() or not item_id.isdigit():
        return None
    for path in (data_root / 'train/images_train' / f'{item_id}.jpg', data_root / 'images' / f'{item_id}.jpg'):
        if path.is_file():
            return path
    return None


class Catalogue:
    def __init__(self, metadata_path: Path, data_root: Path):
        self.data_root = data_root
        with metadata_path.open(newline='', encoding='utf-8-sig') as source:
            indexed = list(csv.DictReader(source))
        details = {}
        for source_path in (data_root / 'train/styles_train.csv', data_root / 'styles.csv'):
            if source_path.is_file():
                with source_path.open(newline='', encoding='utf-8-sig') as source:
                    details = {row['id']: row for row in csv.DictReader(source)}
                break
        self.items = []
        for row in indexed:
            merged = {**details.get(row['id'], {}), **{k: v for k, v in row.items() if v}}
            self.items.append({key: merged.get(key, '') or '' for key in FIELDS})
        self.by_id = {row['id']: row for row in self.items}
        self.facets = {key: sorted({row[key] for row in self.items if row[key]}) for key in FILTERS}

    def filter(self, query='', **filters):
        query = query.strip().casefold()
        return [row for row in self.items
                if all(not value or row[key] == value for key, value in filters.items() if key in FILTERS)
                and (not query or query in ' '.join(row.values()).casefold())]

    def samples(self):
        samples = []
        used = set()
        # Deliberate category variety; these are catalogue examples, not a test set.
        for row in self.items:
            if row['articleType'] in used or image_path(self.data_root, row['id']) is None:
                continue
            samples.append(row)
            used.add(row['articleType'])
            if len(samples) == 6:
                break
        return samples

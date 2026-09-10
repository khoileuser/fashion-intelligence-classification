import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch

from app.server.utils.search_features import GARMENT_FEATURE_CONFIG, search_feature
from app.server.utils.visual_search import FashionVisualSearch
from scripts.rebuild_visual_search import agreement_metrics


def product(colour, background='white', offset=0):
    image = Image.new('RGB', (96, 128), background)
    draw = ImageDraw.Draw(image)
    draw.polygon([(25+offset, 28), (70+offset, 28), (84+offset, 49),
                  (72+offset, 58), (67+offset, 49), (67+offset, 98),
                  (29+offset, 98), (29+offset, 49), (23+offset, 58),
                  (12+offset, 49)], fill=colour)
    return image


class VisualSearchTests(unittest.TestCase):
    def feature(self, image):
        return search_feature(image, 'garment_hog_colour', GARMENT_FEATURE_CONFIG)

    def test_colour_over_background_and_small_pose_change(self):
        query = self.feature(product('coral'))
        matching = self.feature(product('coral', '#eeeeee', offset=2))
        different = self.feature(product('royalblue'))
        self.assertGreater(float(query @ matching), float(query @ different) + 0.1)

    def test_neutral_garments_remain_distinct(self):
        query = self.feature(product('#222222'))
        self.assertGreater(query @ self.feature(product('#333333')),
                           query @ self.feature(product('#dddddd')))

    def test_blank_and_grayscale_are_finite_unit_vectors(self):
        for image in (Image.new('RGB', (1, 1), 'white'), Image.new('L', (30, 70), 0)):
            feature = self.feature(image)
            self.assertTrue(np.isfinite(feature).all())
            self.assertAlmostEqual(float(np.linalg.norm(feature)), 1.0, places=5)

    def test_saved_index_ranks_same_colour_first(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = np.vstack([self.feature(product('royalblue')), self.feature(product('coral'))])
            np.save(root / 'index.npy', embeddings)
            pd.DataFrame({'id': ['1', '2'], 'articleType': ['Tshirts', 'Tshirts']}).to_csv(root / 'items.csv', index=False)
            torch.save({'model_type': 'fixed_feature_cosine', 'feature_type': 'garment_hog_colour',
                        'feature_config': GARMENT_FEATURE_CONFIG, 'embedding_dim': embeddings.shape[1]}, root / 'model.pt')
            search = FashionVisualSearch(root / 'model.pt', root / 'index.npy', root / 'items.csv', device='cpu')
            results = search.search(product('coral', '#eeeeee', offset=2), top_k=5)
            self.assertEqual([item['id'] for item in results], ['2', '1'])
            with self.assertRaises(ValueError):
                search.search(product('coral'), top_k=0)

    def test_legacy_features_still_supported(self):
        for kind, config in [('pixel', {'image_size': [12, 16]}), ('hog_hsv', {'shape_weight': .75})]:
            feature = search_feature(product('coral'), kind, config)
            self.assertAlmostEqual(float(feature @ feature), 1.0, places=5)

    def test_article_preference_searches_full_gallery_and_fills_remaining_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # The wrong type is an exact visual match. Preferred items must
            # still be found outside the initial top-one visual result.
            embeddings = np.array([[1, 0, 0], [.6, .8, 0], [.8, .6, 0], [.8, .6, 0]], dtype=np.float32)
            np.save(root / 'index.npy', embeddings)
            pd.DataFrame({'id': ['1', '2', '3', '4'],
                          'articleType': ['Tops', 'Nightdress', 'Nightdress', 'Nightdress']}).to_csv(root / 'items.csv', index=False)
            torch.save({'model_type': 'fixed_feature_cosine', 'feature_type': 'pixel',
                        'feature_config': {'image_size': [1, 1]}, 'embedding_dim': 3}, root / 'model.pt')
            search = FashionVisualSearch(root / 'model.pt', root / 'index.npy', root / 'items.csv', device='cpu')
            query = Image.new('RGB', (1, 1), 'red')
            self.assertEqual(search.search(query, 1)[0]['id'], '1')
            self.assertEqual(search.search(query, 1, 'Nightdress')[0]['id'], '3')
            results = search.search(query, 5, 'Nightdress')
            self.assertEqual([item['id'] for item in results], ['3', '4', '2', '1'])
            np.testing.assert_allclose([item['score'] for item in results], [.8, .8, .6, 1])
            self.assertEqual([item['id'] for item in search.search(query, 5, 'Unknown')], ['1', '3', '4', '2'])

    def test_evaluation_excludes_validation_and_test_from_gallery(self):
        metadata = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd', 'e', 'query', 'test'],
            'articleType': ['Tops', 'Tops', 'Shoes', 'Shoes', 'Shoes', 'Tops', 'Tops'],
            'baseColour': ['Red', 'Blue', 'Blue', 'Blue', 'Blue', 'Red', 'Red'],
        })
        splits = pd.DataFrame({'id': metadata.id,
                               'split': ['train'] * 5 + ['validation', 'test']})
        features = np.array([[0, 1]] * 5 + [[1, 0], [1, 0]], dtype=np.float32)
        metrics = agreement_metrics(features, metadata, splits, sample_size=0)
        self.assertEqual(metrics['training_gallery'], 5)
        self.assertEqual(metrics['validation_queries'], 1)
        self.assertAlmostEqual(metrics['article_precision@5'], 0.4)
        self.assertAlmostEqual(metrics['article_and_colour_precision@5'], 0.2)


if __name__ == '__main__':
    unittest.main()

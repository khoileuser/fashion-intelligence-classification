"""Load a fixed-feature or legacy neural search index and rank by cosine score."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from app.server.utils.image_preprocessor import image_tensor
from app.server.utils.modeling import EmbeddingCNN
from app.server.utils.search_features import search_feature


class FashionVisualSearch:
    def __init__(
        self,
        checkpoint_path: str | Path,
        embeddings_path: str | Path,
        metadata_path: str | Path,
        device: str | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint_path = Path(checkpoint_path)
        embeddings_path = Path(embeddings_path)
        metadata_path = Path(metadata_path)
        for path in (checkpoint_path, embeddings_path, metadata_path):
            if not path.exists():
                raise FileNotFoundError(path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model_type = checkpoint.get('model_type', 'contrastive_encoder')
        if self.model_type == 'fixed_feature_cosine':
            self.model = None
            self.feature_type = checkpoint['feature_type']
            self.feature_config = checkpoint['feature_config']
            if self.feature_type not in {'pixel', 'hog_hsv', 'garment_hog_colour'}:
                raise ValueError(f'Unsupported search feature: {self.feature_type!r}')
            feature_dimensions = checkpoint['embedding_dim']
        elif self.model_type == 'contrastive_encoder':
            self.mean = checkpoint["mean"]
            self.std = checkpoint["std"]
            self.image_size = checkpoint.get("image_size", [96, 128])
            feature_dimensions = checkpoint.get("embedding_dim", 128)
            self.model = EmbeddingCNN(feature_dimensions)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.to(self.device).eval()
        else:
            raise ValueError(f'Unsupported search model: {self.model_type!r}')
        self.embeddings = np.load(embeddings_path).astype(np.float32)
        self.metadata = pd.read_csv(metadata_path, dtype={"id": "string"}, keep_default_na=False)
        if len(self.embeddings) != len(self.metadata):
            raise ValueError("Gallery embeddings and metadata have different lengths")
        if self.embeddings.ndim != 2 or self.embeddings.shape[1] != feature_dimensions:
            raise ValueError('Gallery feature dimensions differ from the checkpoint')
        if not np.isfinite(self.embeddings).all():
            raise ValueError('Gallery features must be finite')

    @torch.inference_mode()
    def search(
        self, image: Image.Image, top_k: int = 5,
        preferred_article_type: str | None = None,
    ) -> list[dict]:
        """Rank the preferred article type first, then cosine within each group.

        Without a preference this remains pure visual retrieval for notebook
        evaluation. Other types fill spare slots when the preferred type is rare.
        Scores always retain their original cosine meaning.
        """
        if top_k < 1:
            raise ValueError('top_k must be positive')
        if self.model is None:
            query = search_feature(image, self.feature_type, self.feature_config)
        else:
            tensor = image_tensor(image, self.image_size, self.mean, self.std).to(self.device)
            query = self.model(tensor)[0].cpu().numpy()
        scores = self.embeddings @ query
        indices = np.argsort(-scores, kind='stable')
        if preferred_article_type:
            matches = self.metadata['articleType'].eq(preferred_article_type).to_numpy()
            indices = np.concatenate([indices[matches[indices]], indices[~matches[indices]]])
        indices = indices[: min(top_k, len(scores))]
        results = []
        for index in indices:
            item = {
                key: None if pd.isna(value) or value == '' else value
                for key, value in self.metadata.iloc[int(index)].to_dict().items()
            }
            item["score"] = float(scores[index])
            results.append(item)
        return results

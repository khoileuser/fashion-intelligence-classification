"""Load a neural-embedding search index and rank by cosine score."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import json
from PIL import Image

from app.server.utils.search_features import NeuralImageEncoder, file_digest


class FashionVisualSearch:
    def __init__(
        self,
        checkpoint_path: str | Path,
        embeddings_path: str | Path,
        metadata_path: str | Path,
        device: str | None = None,
    ):
        checkpoint_path = Path(checkpoint_path)
        embeddings_path = Path(embeddings_path)
        metadata_path = Path(metadata_path)
        for path in (checkpoint_path, embeddings_path, metadata_path):
            if not path.exists():
                raise FileNotFoundError(path)
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        self.model_type = checkpoint["model_type"]
        if self.model_type != "keras_embedding_cosine":
            raise ValueError("Expected a Keras embedding search index; rerun Task 4")
        model_path = checkpoint_path.parent / checkpoint["encoder_file"]
        if file_digest(model_path) != checkpoint["encoder_sha256"]:
            raise ValueError("Article model changed; rerun Task 4 to rebuild its gallery")
        self.encoder = NeuralImageEncoder(model_path)
        feature_dimensions = checkpoint["embedding_dim"]
        self.embeddings = np.load(embeddings_path).astype(np.float32)
        self.metadata = pd.read_csv(metadata_path, dtype={"id": "string"}, keep_default_na=False)
        if len(self.embeddings) != len(self.metadata):
            raise ValueError("Gallery embeddings and metadata have different lengths")
        if self.embeddings.ndim != 2 or self.embeddings.shape[1] != feature_dimensions:
            raise ValueError('Gallery feature dimensions differ from the checkpoint')
        if not np.isfinite(self.embeddings).all():
            raise ValueError('Gallery features must be finite')

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
        query = self.encoder.encode([image])[0]
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

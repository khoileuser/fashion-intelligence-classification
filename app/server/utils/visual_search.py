"""Load the visual-search encoder and exact cosine gallery index."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from app.server.utils.image_preprocessor import image_tensor
from app.server.utils.modeling import EmbeddingCNN


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
        self.mean = checkpoint["mean"]
        self.std = checkpoint["std"]
        self.image_size = checkpoint.get("image_size", [96, 128])
        self.model = EmbeddingCNN(checkpoint.get("embedding_dim", 128))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()
        self.embeddings = np.load(embeddings_path).astype(np.float32)
        self.metadata = pd.read_csv(metadata_path, dtype={"id": "string"})
        if len(self.embeddings) != len(self.metadata):
            raise ValueError("Gallery embeddings and metadata have different lengths")

    @torch.inference_mode()
    def search(self, image: Image.Image, top_k: int = 5) -> list[dict]:
        tensor = image_tensor(image, self.image_size, self.mean, self.std).to(self.device)
        query = self.model(tensor)[0].cpu().numpy()
        scores = self.embeddings @ query
        indices = np.argsort(-scores)[: min(top_k, len(scores))]
        results = []
        for index in indices:
            item = {
                key: None if pd.isna(value) else value
                for key, value in self.metadata.iloc[int(index)].to_dict().items()
            }
            item["score"] = float(scores[index])
            results.append(item)
        return results

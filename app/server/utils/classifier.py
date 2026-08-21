"""Load and run a classifier checkpoint produced by Tasks 1-3 notebooks."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from app.server.utils.image_preprocessor import image_tensor
from app.server.utils.modeling import CompactCNN


class FashionClassifier:
    def __init__(self, checkpoint_path: str | Path, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.target = checkpoint["target"]
        self.labels = checkpoint["labels"]
        self.mean = checkpoint["mean"]
        self.std = checkpoint["std"]
        self.image_size = checkpoint.get("image_size", [96, 128])
        self.model = CompactCNN(len(self.labels), checkpoint.get("dropout", 0.2))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, image: Image.Image, top_k: int = 3) -> dict:
        tensor = image_tensor(image, self.image_size, self.mean, self.std).to(self.device)
        probabilities = self.model(tensor).softmax(dim=1)[0]
        count = min(top_k, len(self.labels))
        scores, indices = probabilities.topk(count)
        ranked = [
            {"label": self.labels[int(index)], "confidence": float(score)}
            for score, index in zip(scores.cpu(), indices.cpu(), strict=True)
        ]
        return {
            "target": self.target,
            "label": ranked[0]["label"],
            "confidence": ranked[0]["confidence"],
            "top_k": ranked,
        }

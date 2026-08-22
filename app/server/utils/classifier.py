"""Load and run a classifier checkpoint produced by Tasks 1-3 notebooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.server.utils.image_preprocessor import image_tensor
from app.server.utils.handcrafted import handcrafted_feature
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
        self.model_type = checkpoint.get("model_type", "compact_cnn")
        if self.model_type == "compact_cnn":
            self.mean = checkpoint["mean"]
            self.std = checkpoint["std"]
            self.image_size = checkpoint.get("image_size", [96, 128])
            self.model = CompactCNN(len(self.labels), checkpoint.get("dropout", 0.2))
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.to(self.device).eval()
            self.estimator = None
            self.feature_config = None
        elif self.model_type == "hog_hsv_logistic_regression":
            self.model = None
            self.estimator = checkpoint["estimator"]
            self.feature_config = checkpoint["feature_config"]
            estimator_labels = list(self.estimator.classes_)
            if set(estimator_labels) != set(self.labels):
                raise ValueError("Checkpoint labels differ from estimator classes")
            self.estimator_order = [estimator_labels.index(label) for label in self.labels]
        else:
            raise ValueError(f"Unsupported classifier model_type: {self.model_type!r}")

    @torch.inference_mode()
    def predict(self, image: Image.Image, top_k: int = 3) -> dict:
        if self.model_type == "compact_cnn":
            tensor = image_tensor(image, self.image_size, self.mean, self.std).to(self.device)
            probabilities = self.model(tensor).softmax(dim=1)[0].cpu().numpy()
        else:
            features = handcrafted_feature(image, self.feature_config)[None, :]
            probabilities = self.estimator.predict_proba(features)[0][self.estimator_order]
        count = min(top_k, len(self.labels))
        indices = np.argsort(probabilities)[::-1][:count]
        ranked = [
            {"label": self.labels[int(index)], "confidence": float(probabilities[index])}
            for index in indices
        ]
        return {
            "target": self.target,
            "label": ranked[0]["label"],
            "confidence": ranked[0]["confidence"],
            "top_k": ranked,
        }

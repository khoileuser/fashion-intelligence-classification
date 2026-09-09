"""Load and run a classifier checkpoint produced by Tasks 1-3 notebooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance

from app.server.utils.image_preprocessor import image_tensor
from app.server.utils.handcrafted import handcrafted_feature
from app.server.utils.modeling import CompactCNN, SimpleCNN


def temperature_scale(probabilities, temperature=1.0):
    """Rescale confidence without changing the highest-probability class."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    values = np.asarray(probabilities, dtype=np.float64)
    if temperature == 1.0:
        return values
    logits = np.log(np.clip(values, np.finfo(np.float64).tiny, 1.0)) / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=-1, keepdims=True)


class FashionClassifier:
    def __init__(self, checkpoint_path: str | Path | dict, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if isinstance(checkpoint_path, dict):
            checkpoint = checkpoint_path
        else:
            path = Path(checkpoint_path)
            if not path.exists():
                raise FileNotFoundError(path)
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.target = checkpoint["target"]
        self.labels = checkpoint["labels"]
        self.temperature = float(checkpoint.get('temperature', 1.0))
        self.review_policy = checkpoint.get('review_policy')
        self.model_type = checkpoint.get("model_type", "compact_cnn")
        self.members = None
        if self.model_type == 'probability_ensemble':
            self.members = [FashionClassifier(member, device=str(self.device)) for member in checkpoint['members']]
            if not self.members or any(member.labels != self.labels or member.target != self.target for member in self.members):
                raise ValueError('Ensemble members must use the same target and label order')
            self.model = None
            self.estimator = None
        elif self.model_type in {"compact_cnn", "simple_cnn"}:
            self.mean = checkpoint["mean"]
            self.std = checkpoint["std"]
            self.image_size = checkpoint.get("image_size", [96, 128])
            architecture = SimpleCNN if self.model_type == "simple_cnn" else CompactCNN
            self.model = architecture(len(self.labels), checkpoint.get("dropout", 0.2))
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
    def predict_probabilities(self, image: Image.Image) -> np.ndarray:
        """Return ordered probabilities, including any saved calibration."""
        if self.members is not None:
            probabilities = np.mean([member.predict_probabilities(image) for member in self.members], axis=0)
        elif self.model is not None:
            tensor = image_tensor(image, self.image_size, self.mean, self.std).to(self.device)
            probabilities = self.model(tensor).softmax(dim=1)[0].cpu().numpy()
        else:
            features = handcrafted_feature(image, self.feature_config)[None, :]
            probabilities = self.estimator.predict_proba(features)[0][self.estimator_order]
        return temperature_scale(probabilities, self.temperature)

    def predict(self, image: Image.Image, top_k: int = 3) -> dict:
        probabilities = self.predict_probabilities(image)
        count = min(top_k, len(self.labels))
        indices = np.argsort(probabilities)[::-1][:count]
        ranked = [
            {"label": self.labels[int(index)], "confidence": float(probabilities[index])}
            for index in indices
        ]
        result = {
            "target": self.target,
            "label": ranked[0]["label"],
            "confidence": ranked[0]["confidence"],
            "top_k": ranked,
        }
        if self.review_policy is not None:
            threshold = self.review_policy['threshold']
            result['needs_review'] = threshold is None or ranked[0]['confidence'] < threshold
            result['review_threshold'] = threshold
            result['confidence_calibrated'] = self.temperature != 1.0
            if self.review_policy.get('brightness_stability') and not result['needs_review']:
                stable_label = int(indices[0])
                unstable = any(
                    int(self.predict_probabilities(ImageEnhance.Brightness(image.convert('RGB')).enhance(factor)).argmax()) != stable_label
                    for factor in (0.8, 1.2)
                )
                if unstable:
                    result['needs_review'] = True
                    result['review_reason'] = 'Prediction changes with lighting'
        return result

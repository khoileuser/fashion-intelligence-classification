"""Load the selected article-type model and classify one image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.server.utils.classifier import FashionClassifier

MODEL_PATH = ROOT / "models" / "article_type_model.pt"


def classify_article_type(image_path: str | Path, model_path: str | Path = MODEL_PATH) -> dict:
    with Image.open(image_path) as image:
        result = FashionClassifier(model_path).predict(image)
    print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--model", default=str(MODEL_PATH))
    args = parser.parse_args()
    classify_article_type(args.image, args.model)

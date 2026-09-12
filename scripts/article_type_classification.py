"""Load the selected article-type model and classify one image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.server.utils.classifier import FashionClassifier

MODEL_PATH = ROOT / "models" / "article_type_model.keras"


def print_classification(result: dict, image_path: str | Path) -> None:
    titles = {"articleType": "Article type", "season": "Season",
              "gender": "Gender / audience", "usage": "Occasion"}
    title = titles.get(result["target"], result["target"])
    print(f"\n{title} prediction")
    print("-" * (len(title) + 11))
    print(f"Image: {image_path}")
    print(f"Prediction: {result['label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nTop predictions")
    ranked = result["top_k"]
    width = max(len("Label"), *(len(item["label"]) for item in ranked))
    print(f"{'Rank':<4}  {'Label':<{width}}  {'Confidence':>10}")
    for rank, item in enumerate(ranked, 1):
        print(f"{rank:<4}  {item['label']:<{width}}  {item['confidence']:>10.2%}")
    if "needs_review" in result:
        status = "Requested" if result["needs_review"] else "Not requested"
        print(f"\nHuman review: {status}")
        if result.get("review_reason"):
            print(f"Reason: {result['review_reason']}")
    print()


def classify_article_type(image_path: str | Path, model_path: str | Path = MODEL_PATH) -> dict:
    with Image.open(image_path) as image:
        result = FashionClassifier(model_path).predict(image)
    print_classification(result, image_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--model", default=str(MODEL_PATH))
    args = parser.parse_args()
    classify_article_type(args.image, args.model)

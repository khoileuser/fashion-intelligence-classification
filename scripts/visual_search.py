"""Load the selected search model and gallery to retrieve similar images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.server.utils.visual_search import FashionVisualSearch


def search_image(image_path: str | Path, top_k: int = 5) -> list[dict]:
    search = FashionVisualSearch(
        ROOT / "models" / "visual_search_model.json",
        ROOT / "models" / "visual_search_embeddings.npy",
        ROOT / "models" / "visual_search_metadata.csv",
    )
    with Image.open(image_path) as image:
        results = search.search(image, top_k)
    for item in results:
        print(item)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    search_image(args.image, args.top_k)

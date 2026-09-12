"""Load the selected search model and gallery to retrieve similar images."""

from __future__ import annotations

import argparse
import hashlib
import csv
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.server.utils.visual_search import FashionVisualSearch


def print_search_results(results: list[dict], image_path: str | Path) -> None:
    print("\nVisual search")
    print("-------------")
    print(f"Image: {image_path}")
    print(f"Matches: {len(results)}")
    if not results:
        print("No matching products found.\n")
        return
    print("\n")
    for rank, item in enumerate(results, 1):
        name = item.get("productDisplayName") or item.get("articleType") or "Product"
        print(f"{rank}. {name} (ID: {item['id']})")
        print(f"Similarity: {item['score']:.4f}")
        attributes = [str(item[key]) for key in ("articleType", "baseColour", "season", "gender", "usage") if item.get(key)]
        if attributes:
            print(" | ".join(attributes))
        print()


def search_image(image_path: str | Path, top_k: int = 5) -> list[dict]:
    search = FashionVisualSearch(
        ROOT / "models" / "visual_search_model.json",
        ROOT / "models" / "visual_search_embeddings.npy",
        ROOT / "models" / "visual_search_metadata.csv",
    )
    audit = ROOT / 'scripts/data/image_audit.csv'
    excluded = set()
    if audit.is_file():
        digest = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
        with audit.open(encoding='utf-8', newline='') as handle:
            excluded = {row['id'] for row in csv.DictReader(handle)
                        if row.get('sha256', '').lower() == digest}
    with Image.open(image_path) as image:
        results = search.search(image, top_k, exclude_ids=excluded)
    print_search_results(results, image_path)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    search_image(args.image, args.top_k)

"""Create the assignment prediction CSV using all four selected classifiers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.preprocessing import load_prediction_template

from app.server.utils.classifier import FashionClassifier

MODEL_PATHS = {
    "articleType": ROOT / "models" / "article_type_model.pt",
    "season": ROOT / "models" / "season_model.pt",
    "gender": ROOT / "models" / "gender_model.pt",
    "usage": ROOT / "models" / "usage_model.pt",
}


def create_submission(output: str | Path = ROOT / "prediction" / "styles_prediction.csv") -> Path:
    template = load_prediction_template()
    output_columns = [column for column in template.columns if column != "image_path"]
    predictors = {target: FashionClassifier(path) for target, path in MODEL_PATHS.items()}
    rows = []
    for row in template.itertuples(index=False):
        with Image.open(row.image_path) as image:
            predictions = {
                target: predictor.predict(image, top_k=1)["label"]
                for target, predictor in predictors.items()
            }
        rows.append({"id": row.id, **predictions})
    result = pd.DataFrame(rows)[output_columns]
    if result["id"].tolist() != template["id"].tolist():
        raise ValueError("Prediction IDs or order differ from the supplied template")
    if result["id"].duplicated().any() or result.drop(columns="id").eq("").any().any():
        raise ValueError("Submission contains duplicate IDs or blank predictions")
    for target, predictor in predictors.items():
        if not set(result[target]).issubset(predictor.labels):
            raise ValueError(f"Submission contains an unknown {target} label")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result):,} predictions to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "prediction" / "styles_prediction.csv"))
    args = parser.parse_args()
    create_submission(args.output)

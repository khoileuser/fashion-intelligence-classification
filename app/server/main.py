"""FastAPI service for four predictions and visual search."""

from __future__ import annotations

import io
import os
import re
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from app.server.utils.classifier import FashionClassifier
from app.server.utils.visual_search import FashionVisualSearch

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models")).resolve()
DATA_ROOT = Path(
    os.environ.get("FASHION_DATA_ROOT", ROOT / "dataset")
).resolve()

TARGETS = ("articleType", "season", "gender", "usage")
MODEL_FILES = {
    "articleType": "article_type_model.keras",
    "season": "season_model.keras",
    "gender": "gender_model.keras",
    "usage": "usage_model.keras",
}
MAX_BYTES = 10 * 1024 * 1024

app = FastAPI(
    title="Fashion Intelligence API",
    description="Classification and visual-search inference for fashion images.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def classifier_services() -> dict[str, FashionClassifier]:
    missing = [
        target for target, filename in MODEL_FILES.items() if not (MODEL_DIR / filename).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing classifier checkpoints: {missing}")
    return {
        target: FashionClassifier(MODEL_DIR / filename)
        for target, filename in MODEL_FILES.items()
    }


@lru_cache(maxsize=1)
def search_service() -> FashionVisualSearch:
    return FashionVisualSearch(
        MODEL_DIR / "visual_search_model.json",
        MODEL_DIR / "visual_search_embeddings.npy",
        MODEL_DIR / "visual_search_metadata.csv",
    )


async def read_image(upload: UploadFile) -> Image.Image:
    if upload.content_type and not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload must be an image")
    content = await upload.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB")
    try:
        image = Image.open(io.BytesIO(content))
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=422, detail="Image could not be decoded") from error


@app.get("/health")
def health() -> dict[str, object]:
    available = {
        target: (MODEL_DIR / filename).exists()
        for target, filename in MODEL_FILES.items()
    }
    available["visual_search"] = all(
        (MODEL_DIR / filename).exists()
        for filename in (
            "visual_search_model.json",
            "visual_search_embeddings.npy",
            "visual_search_metadata.csv",
        )
    )
    return {"status": "ok", "models": available, "version": app.version}


@app.get("/gallery/{item_id}/image", response_class=FileResponse)
def gallery_image(item_id: str) -> FileResponse:
    """Return a gallery image without exposing an arbitrary filesystem path."""
    if not re.fullmatch(r"\d+", item_id):
        raise HTTPException(status_code=404, detail="Gallery image not found")

    candidates = [
        DATA_ROOT / "train" / "images_train" / f"{item_id}.jpg",
        DATA_ROOT / "images" / f"{item_id}.jpg",
    ]
    for path in candidates:
        if path.is_file():
            return FileResponse(path, media_type="image/jpeg")

    raise HTTPException(status_code=404, detail="Gallery image not found")


@app.post("/predict")
async def predict(
    file: UploadFile = File(...), top_k: int = Query(3, ge=1, le=10)
) -> dict[str, object]:
    image = await read_image(file)
    try:
        services = classifier_services()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {
        "predictions": {
            target: service.predict(image, top_k) for target, service in services.items()
        }
    }


@app.post("/search")
async def search(
    file: UploadFile = File(...), top_k: int = Query(5, ge=1, le=20)
) -> dict[str, object]:
    image = await read_image(file)
    try:
        article = classifier_services()['articleType'].predict(image, top_k=1)
        results = search_service().search(image, top_k, preferred_article_type=article['label'])
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"results": results}


@app.post("/analyse")
async def analyse(
    file: UploadFile = File(...),
    prediction_top_k: int = Query(3, ge=1, le=10),
    search_top_k: int = Query(5, ge=1, le=20),
) -> dict[str, object]:
    image = await read_image(file)
    try:
        predictions = {
            target: service.predict(image, prediction_top_k)
            for target, service in classifier_services().items()
        }
        results = search_service().search(
            image, search_top_k, preferred_article_type=predictions['articleType']['label'],
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"predictions": predictions, "similar_items": results}

"""FastAPI service for four predictions and visual search."""

from __future__ import annotations

import io
import os
import random
import re
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from app.server.utils.classifier import FashionClassifier
from app.server.utils.catalogue import Catalogue
from app.server.utils.gradcam import gradcam
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
    # The public reverse proxy exposes this service under /api.
    # FastAPI uses root_path when generating the Swagger schema URL.
    root_path="/api",
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


@lru_cache(maxsize=1)
def catalogue_service():
    return Catalogue(MODEL_DIR / 'visual_search_metadata.csv', DATA_ROOT)


def available_catalogue():
    try:
        return catalogue_service()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail='Catalogue metadata is unavailable') from error


@app.get('/catalogue/samples')
def catalogue_samples():
    return {'items': available_catalogue().samples()}


@app.get('/catalogue')
def catalogue(
    q: str = Query('', max_length=200),
    articleType: str = Query('', max_length=100), season: str = Query('', max_length=100),
    usage: str = Query('', max_length=100), gender: str = Query('', max_length=100),
    baseColour: str = Query('', max_length=100),
    page: int = Query(1, ge=1), page_size: int = Query(12, ge=1, le=48),
    similar_to: str | None = Query(None, pattern=r'^\d+$', max_length=20),
    random_seed: int | None = Query(None, ge=0, le=2**32 - 1),
):
    service = available_catalogue()
    items = service.filter(q, articleType=articleType, season=season, usage=usage, gender=gender, baseColour=baseColour)
    reference = None
    if similar_to is not None:
        reference = service.by_id.get(similar_to)
        if reference is None:
            raise HTTPException(status_code=404, detail='Reference product not found')
        try:
            ranked = search_service().similar_by_id(similar_to, {row['id'] for row in items})
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail='Visual search artifacts are unavailable') from error
        except ValueError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        items = [{**service.by_id[item_id], 'score': score} for item_id, score in ranked]
    else:
        random_seed = random_seed if random_seed is not None else secrets.randbits(32)
        random.Random(random_seed).shuffle(items)
    total = len(items)
    pages = max(1, (total + page_size - 1) // page_size)
    page = min(page, pages)
    return {'items': items[(page - 1) * page_size:page * page_size], 'total': total,
            'page': page, 'pages': pages, 'facets': service.facets, 'reference': reference,
            'seed': random_seed}


@app.post('/explain')
async def explain(
    file: UploadFile = File(...),
    target: Literal['articleType', 'season', 'gender', 'usage'] = Query('articleType'),
    label: str | None = Query(None, max_length=150),
):
    image = await read_image(file)
    try:
        service = classifier_services()[target]
        return gradcam(service, image, label)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail='Classifier artifacts are unavailable') from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

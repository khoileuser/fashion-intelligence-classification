# Fashion Intelligence Classification

Notebook-first machine-learning project for the four assignment tasks: article type, season, occasion and gender, and visual search. The modelling layout follows the supplied rice-plant-disease project; the deployable application is grouped separately under `app/`.

## Structure

```text
fashion-intelligence-classification/
|-- app/
|   |-- client/              # Next.js web interface and API proxy
|   `-- server/              # FastAPI service and inference utilities
|-- dataset/                 # Private train/test dataset copied locally
|-- docs/                    # Team handoff, experiment, and deployment notes
|-- models/                  # Flat model files created by notebooks
|-- notebooks/
|   |-- task0_data_audit_eda.ipynb
|   |-- task1_article_type_classification.ipynb
|   |-- task2_season_classification.ipynb
|   |-- task3_occasion_gender_classification.ipynb
|   `-- task4_visual_search_analysis.ipynb
|-- prediction/              # Final styles_prediction.csv
|-- scripts/                 # Preprocessing, inference, and notebook runner
|-- Dockerfile               # `server` and `client` build targets
|-- docker-compose.yml       # Two-service Portainer stack
`-- requirements.txt         # Notebook/training environment
```

There is no installable modelling package, YAML configuration hierarchy, or separate artifacts directory. Each modelling notebook visibly defines its data pipeline, model, training loop, evaluation, and saved output.

## Model development

Use Python 3.12 and install the CUDA-compatible PyTorch build for the team machine if required.

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The private dataset is read from `dataset/` at the repository root by default. Copy the **contents** of the supplied `FashionDataset` folder into it so there is no extra wrapper level:

```text
dataset/
|-- train/
|   |-- styles_train.csv
|   `-- images_train/
|       `-- <product-id>.jpg
`-- test/
    |-- styles_prediction.csv
    `-- images_test/
        `-- <product-id>.jpg
```

The image filename must match the `id` column in its CSV. The dataset contents are ignored by Git; [dataset/README.md](dataset/README.md) is retained as an on-disk guide. `FASHION_DATA_ROOT` can still override the location when needed.

## Produce the application models

Run the notebooks in the following order. Task 0 creates the shared audit, frozen split, and training-only normalization under `scripts/data/`. Its full image audit runs automatically the first time because no cached `image_audit.csv` exists.

1. `notebooks/task0_data_audit_eda.ipynb`
2. `notebooks/task1_article_type_classification.ipynb` → `models/article_type_model.pt`
3. `notebooks/task2_season_classification.ipynb` → `models/season_model.pt`
4. `notebooks/task3_occasion_gender_classification.ipynb` → `models/gender_model.pt` and `models/usage_model.pt`
5. `notebooks/task4_visual_search_analysis.ipynb` → the visual-search model, embeddings, and metadata

To execute all five notebooks automatically from the repository root:

```powershell
python scripts/run_all_notebooks.py
```

The runner validates the dataset layout, executes notebooks sequentially, saves their outputs in place, verifies every required artifact, and stops immediately if a task fails. Training all four tasks can take a long time. To resume after fixing a failure, start at a later task:

```powershell
python scripts/run_all_notebooks.py --start-at task2
```

To deliberately repeat Task 0's full decode and SHA-256 audit:

```powershell
python scripts/run_all_notebooks.py --force-audit
```

The split joins normalized product-name groups with exact SHA-256 duplicate groups. Every modelling notebook uses the same frozen split and training-only normalization; do not regenerate them merely to improve a score.

Expected model outputs are:

```text
models/article_type_model.pt
models/season_model.pt
models/gender_model.pt
models/usage_model.pt
models/visual_search_model.pt
models/visual_search_embeddings.npy
models/visual_search_metadata.csv
```

The small saved-model helpers remain available:

```powershell
python scripts/task1_article_type_classification.py path\to\image.jpg
python scripts/task2_season_classification.py path\to\image.jpg
python scripts/task3_occasion_gender_classification.py --image path\to\image.jpg --target gender
python scripts/task4_visual_search.py path\to\image.jpg --top-k 5
python scripts/task3_occasion_gender_classification.py --submission
```

## Run the web application locally

Start FastAPI from the repository root:

```powershell
uvicorn app.server.main:app --reload --port 8000
```

In another terminal, start Next.js:

```powershell
cd app/client
bun install
$env:BACKEND_URL = "http://127.0.0.1:8000"
bun dev
```

Open `http://localhost:3000`. The browser calls a same-origin Next.js route, which forwards requests to FastAPI. API documentation is available at `http://127.0.0.1:8000/docs`.

## Docker and Portainer

Copy `.env.example` to `.env` if the default paths or ports do not match the Docker host, then run:

```powershell
docker compose up -d --build
docker compose ps
```

The web interface is published on port `3000`. The API is bound to host loopback on port `8000` by default because the client reaches it over the Compose network. Set `API_BIND=0.0.0.0` only when direct remote API access is required.

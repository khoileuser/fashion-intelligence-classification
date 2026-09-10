# Fashion Intelligence Classification

Notebook-first machine-learning project for the four assignment tasks: article type, season, occasion and gender, and visual search.

## Structure

```text
fashion-intelligence-classification/
|-- app/
|   |-- client/              # Next.js web interface and API proxy
|   `-- server/              # FastAPI service and inference utilities
|-- dataset/                 # Private train/test dataset copied locally
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

## Model development

Use Python 3.12 and install the CUDA-compatible PyTorch build.

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### NVIDIA CUDA training

The notebooks automatically use CUDA when the active Jupyter environment has a CUDA-enabled PyTorch build. The normal requirements file may install a CPU-only build. On a supported NVIDIA system, replace it with the official CUDA 12.8 wheels:

```powershell
.\venv\Scripts\activate
python -m pip install --force-reinstall -r requirements-cuda.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The verification command must print `True` before starting Jupyter. The automatic notebook runner registers and uses a `fashion-intelligence` kernel tied to its active Python environment. For manual Jupyter work, register the same kernel and select **Python (Fashion Intelligence)** in the notebook UI:

```powershell
python -m ipykernel install --prefix "$env:VIRTUAL_ENV" --name fashion-intelligence --display-name "Python (Fashion Intelligence)"
jupyter lab
```

Restart any running Jupyter kernel after changing PyTorch. Task 0, image decoding, HOG feature extraction, scikit-learn baselines, and data loading remain CPU operations; CUDA is used by the CNN training, evaluation, and embedding cells.

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

## Run the web application locally

### Visual-search index

The web application prioritizes the predicted article type across the whole
catalogue, then sorts those items by visual similarity. If there are fewer than
the requested number, other article types fill the remaining slots in similarity
order. Both `/analyse` and `/search` use this policy, including predictions marked
for review. Scores remain cosine similarities, so a preferred-type item can rank
above a different type with a higher score. Notebook retrieval metrics below
evaluate the visual features without this application-level article preference.

Search combines HOG shape features with joint HSV colour histograms. For colour,
it suppresses a plain background when the image corners agree and gives extra
weight to the central region. This reduces matches driven by white backgrounds,
faces and trousers in modelled topwear photos. It is a product-photo heuristic,
not garment segmentation; complex scenes and off-centre garments can still give
weak matches. Similarity is a cosine score, not a match probability.
The current representation assigns 80% of the score to shape and 20% to the
independently normalized colour features.

Task 4 compares the original features with this representation. Selection now
uses validation agreement on **both article type and base colour**, keeping
methods within three percentage points of the original article-only precision,
with article type agreement as a tie-breaker. These catalogue labels are proxies for visual
similarity, not human judgements of pattern or style.

Across 5,842 validation queries against 27,051 training images, the updated index
improves top-five article-and-colour agreement from 20.0% to 33.1%. Article-only
precision changes from 75.1% to 72.6%. The shape weight was selected using this
validation split; these are development results, not independent test results.

To rebuild just the search index using the existing gallery and frozen splits:

```powershell
.\venv\Scripts\python.exe scripts\rebuild_visual_search.py --validation-sample 0
```

This writes candidate artifacts and a before/after evaluation to
`models/search_candidate/`. After reviewing the evaluation, copy its
`visual_search_model.pt`, `visual_search_embeddings.npy`,
`visual_search_metadata.csv`, `visual_search_history.csv` and
`visual_search_colour_evaluation.json` into `models/` together, then restart
FastAPI. For Docker, rebuild and recreate the server because model artifacts
are copied into its image. The checked-in colour evaluation documents the
current index; the original Task 4 metrics do not describe this new representation.

### Start the services

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

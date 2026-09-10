# Fashion Intelligence Classification

Notebook-first machine-learning project for the four assignment tasks: article type, season, occasion and gender, and visual search.

## Structure

```text
fashion-intelligence-classification/
|-- app/
|   |-- client/              # Next.js web interface and API proxy
|   `-- server/              # FastAPI service and inference utilities
|-- dataset/                 # Private train/test dataset copied locally
|-- results/                 # Training histories, comparisons and evaluation reports
|-- models/                  # Selected models and required search index
|-- notebooks/
|   |-- task0_data_audit_eda.ipynb
|   |-- task1_article_type_classification.ipynb
|   |-- task2_season_classification.ipynb
|   |-- task3_occasion_gender_classification.ipynb
|   `-- task4_visual_search_analysis.ipynb
|-- prediction/              # Final styles_prediction.csv
|-- scripts/                 # Preprocessing and prediction scripts
|-- Dockerfile               # `server` and `client` build targets
|-- docker-compose.yml       # Two-service Portainer stack
`-- requirements.txt         # Notebook/training environment
```

## Model development

Tasks 1–3 compare four entries per target: a shallow MLP baseline (one 256-unit
hidden layer), a deeper MLP (256 → 128 → 64), CNN, and HOG + HSV logistic
regression. Both MLPs are fully connected ANNs. A separate CNN
tuning table compares the original three-block CNN, the same CNN with a learning-rate
schedule, and a scheduled four-block CNN with a larger dense head inspired by the
rice-project sample. All neural candidates use the same 96×128 RGB images,
training-only horizontal flips, Adam, and up to 30 epochs with early stopping.
CUDA training uses mixed precision; validation and application inference use float32.
Training uses PyTorch's native batch-normalization CUDA kernels to avoid slow
small-channel cuDNN kernels on the RTX 3060. Outputs, gradients and running
statistics are checked against the standard implementation; inference keeps the
standard path and existing checkpoint format.

Run the experiments on the existing frozen split:

```powershell
.\venv\Scripts\python.exe -m scripts.utils.model_comparison
```

Or run the Task 1–3 notebooks. Each has numbered introduction, setup, data and
preprocessing sections, dedicated architecture and training sections for each
model, and comparison, evaluation and conclusion sections.
Completed candidates resume only when the saved
experiment fingerprint matches the implementation, training data and settings.
Each target replaces its selected `models/<target>_model.pt` prediction model.
Training histories, comparison tables and evaluation summaries are saved in
`results/`. Reusable candidate checkpoints are stored in the ignored
`.cache/training/` directory, so rerunning a notebook can resume completed training.
Plots are saved in `figures/`.

Run `python -m scripts.utils.report_model_comparison` to regenerate
`results/classification_report.md` and `figures/classification_comparison.png`.

Half of the frozen validation groups select models and epochs, one quarter fits
temperature, and one quarter checks calibration and sets review thresholds.
Selection prioritizes macro-F1, then accuracy, then fewer parameters. The CNN row
uses the best CNN configuration by that same rule; its tuning table reports both
accuracy and macro-F1 changes. These validation results are not directly
comparable to historical tables measured on all validation images. The internal
test has prior development exposure and is evaluated after selection.

The web application loads the selected `<target>_model.pt` files directly.
Restart FastAPI after running classification notebooks to load the replacements;
Docker deployments require rebuilding and recreating the server image.
The application loader supports both MLP architectures and the four-block CNN.
Visual search remains a separate retrieval experiment.

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
.\venv\Scripts\python.exe scripts\utils\rebuild_visual_search.py --validation-sample 0
```

This replaces the search model, embeddings and metadata in `models/`, and
writes history and evaluation results to `results/`. Restart FastAPI after rebuilding. For Docker, rebuild and recreate
the server because model artifacts are copied into its image. The checked-in colour evaluation documents the
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

### Ignored training checkpoints

The 24 per-candidate checkpoints in `.cache/training/` (MLPs, CNN configurations and
HOG + HSV logistic regression) are local training caches. Keep them to resume
notebook comparisons without retraining. They are ignored to avoid adding about
305 MiB of reproducible training artifacts to Git. A fresh clone must train these
candidates again, or receive the checkpoint files separately.

The application uses the tracked `<target>_model.pt` winners and the
`visual_search_model.pt`, `visual_search_embeddings.npy` and
`visual_search_metadata.csv` files. Candidate checkpoints are not required for
application inference. Generated figures in `figures/` are not ignored.

Training, evaluation, data-audit, reporting and execution helpers live in
`scripts/utils/`. They are needed to reproduce the notebooks and results, so
include them in the submission. The top-level classification and visual-search scripts load the selected
models and run predictions.

### Prediction scripts

Each classification script loads one selected checkpoint from `models/`,
applies its saved preprocessing, and returns the label and confidence through
the same classifier used by the web application.

| Script | Selected model |
| --- | --- |
| `scripts/article_type_classification.py` | `article_type_model.pt` |
| `scripts/season_classification.py` | `season_model.pt` |
| `scripts/gender_classification.py` | `gender_model.pt` |
| `scripts/occasion_classification.py` | `usage_model.pt` (occasion) |
| `scripts/visual_search.py` | `visual_search_model.pt` plus gallery embeddings and metadata |

From the project root:

```powershell
python scripts/article_type_classification.py path/to/image.jpg
python scripts/season_classification.py path/to/image.jpg
python scripts/gender_classification.py path/to/image.jpg
python scripts/occasion_classification.py path/to/image.jpg
python scripts/visual_search.py path/to/image.jpg --top-k 5
```

Classification scripts accept `--model path/to/model.pt` to override their
default checkpoint. To produce the assignment CSV, the separate submission
utility loads all four classifiers:

```powershell
python -m scripts.utils.create_submission --output prediction/styles_prediction.csv
```

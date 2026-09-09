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

There is no installable modelling package, YAML configuration hierarchy, or separate artifacts directory. Each classifier notebook presents its data, architecture, training call, comparison, evaluation and saved output. The plain CNN and ordinary training loop are shared in `app/server/utils/modeling.py` and `scripts/classification.py` so training and inference use the same implementation.

## Model development

Use Python 3.12 and install the CUDA-compatible PyTorch build for the team machine if required.

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

The image filename must match the `id` column in its CSV. The dataset contents are ignored by Git; [dataset/README.md](dataset/README.md) is retained as an on-disk guide and records the expected CSV SHA-256 fingerprints. Verify them before using the frozen manifests. `FASHION_DATA_ROOT` can still override the location when needed.

## Produce the application models

The subsequent [simple training trials](SIMPLE_TRAINING_TRIALS.md) retain a
brightness-trained article classifier (80.27% internal-test accuracy). The small
F1 gain and worse ECE are documented alongside unsuccessful candidates and
remaining browser errors.

The latest from-scratch classifier trials are recorded in
[CLASSIFIER_TRIALS.md](CLASSIFIER_TRIALS.md), with runnable experiment code and
comparison CSVs under `experiments/`. Its [accuracy-priority update](CLASSIFIER_TRIALS.md#accuracy-priority-update)
records the subsequent article-type and gender selections. The earlier lighting and calibration run is
recorded in [CLASSIFIER_IMPROVEMENTS.md](CLASSIFIER_IMPROVEMENTS.md). Both preserve the
lecturer's dataset and frozen partitions. Tasks 1–3 retain their earlier training
outputs; rerunning their export cells overwrites the follow-up checkpoints.
Task 4 has now been executed and tested through the browser; see
[TASK4_RESULTS.md](TASK4_RESULTS.md) for retrieval results and observed classifier errors.

**Task 0/4 update:** Task 0 validates existing frozen data without rewriting the
split. Task 4 now uses normalized HOG/HSV features with 75% shape / 25% colour
weighting and cosine search, selected on validation. Test precision@5 is 74.83%
against the training-only gallery; the deployment index contains all 38,611 usable
labelled catalogue images. The all-notebook runner below executes Task 4 too.

Task 0 has created the shared audit, frozen split, and training-only normalization under `scripts/data/`. These handoff artifacts are versioned so every member uses the same IDs and preprocessing statistics; the private dataset itself remains ignored. Do not regenerate the split locally.

Tasks 1-3 now compare a HOG+HSV logistic-regression baseline with one plain three-block CNN per target, alongside a majority reference. Training uses one shared readable PyTorch loop and ordinary cross-entropy; there are no residual blocks or loss-mode searches in these notebooks. The API supports the selected models and legacy compact-CNN checkpoints. See [the simplification record](docs/SIMPLIFICATION_EXPERIMENT.md) for current results, before/after comparisons and the disclosure that this is a follow-up after earlier test exposure. The [earlier correction record](docs/TASKS_1_3_REVIEW_FIXES.md) remains historical evidence.

To verify the frozen classifiers without training or changing model selection:

```powershell
python scripts/verify_classifier_artifacts.py
python -m unittest discover -s tests -v
```

The verifier checks checkpoint/selection/history consistency, reproduces full validation and internal-test metrics on CPU, and writes `models/classifier_verification.json`. This is an integration reproduction of already-frozen methods, not permission to tune on the internal test set.

To refresh report results from the current saved checkpoints, execute the notebooks in the following order:

1. `notebooks/task0_data_audit_eda.ipynb`
2. `notebooks/task1_article_type_classification.ipynb` → current article checkpoint evaluation, Section 20
3. `notebooks/task2_season_classification.ipynb` → current season checkpoint evaluation, Section 17
4. `notebooks/task3_occasion_gender_classification.ipynb` → current gender and usage checkpoint evaluation, Section 23
5. `notebooks/task4_visual_search_analysis.ipynb` → repeat retrieval evaluation and verify a separately rebuilt index

Tasks 1–3 default to `RUN_HISTORICAL_TRAINING = False`: they load the selected models, refresh internal-test metrics and figures, and check agreement with saved results. Their earlier experiment analyses remain labelled historical. Enabling historical training reproduces the old baseline pipeline into `.cache/historical-training/models/`; it does not reproduce the later selected trials or overwrite the application checkpoints. Task 4 rebuilds artifacts in `.cache/notebook-refresh/task4/` and compares its results with the active model. PNG plots and numerical evidence for the report are saved in `figures/`.

To execute all five notebooks automatically from the repository root:

```powershell
python scripts/run_all_notebooks.py --device cuda
```

The runner validates the dataset layout and CUDA availability, executes notebooks sequentially, saves their outputs in place, verifies every required artifact, and stops immediately if a task fails. Using `--device cuda` prevents silent fallback for device-selecting cells; the final report evaluation explicitly uses CPU to match the checkpoint reference metrics. Retrieval feature extraction can take several minutes. `--start-at` runs the selected task **and every later task**, refreshing their notebook outputs:

```powershell
python scripts/run_all_notebooks.py --device cuda --start-at task2
```

During parallel development, each member must open and execute only their assigned notebook. See `docs/MEMBER_HANDOFF.md` for the ownership map and return checklist. Record answers to assignment ambiguities in `docs/TEACHING_TEAM_CONFIRMATIONS.md` rather than making silent target-vocabulary or model-count changes.

To deliberately repeat Task 0's full decode and SHA-256 audit:

```powershell
python scripts/run_all_notebooks.py --force-audit
```

The split joins normalized product-name groups with exact SHA-256 duplicate groups. Task 0 also moves whole groups into training when necessary to guarantee that every evaluated label is learnable. The runner checks this contract before resumed tasks. Every modelling notebook uses the same frozen split and training-only normalization; do not regenerate it merely to improve a score.

Expected model outputs are:

```text
models/article_type_model.pt
models/article_type_history.csv
models/article_type_comparison.csv
models/season_model.pt
models/season_history.csv
models/season_comparison.csv
models/gender_model.pt
models/gender_history.csv
models/gender_comparison.csv
models/usage_model.pt
models/usage_history.csv
models/usage_comparison.csv
models/visual_search_model.pt
models/visual_search_embeddings.npy
models/visual_search_metadata.csv
models/visual_search_history.csv
```

The small saved-model helpers remain available:

```powershell
python scripts/task1_article_type_classification.py path\to\image.jpg
python scripts/task2_season_classification.py path\to\image.jpg
python scripts/task3_occasion_gender_classification.py --image path\to\image.jpg --target gender
python scripts/task4_visual_search.py path\to\image.jpg --top-k 5
python scripts/task3_occasion_gender_classification.py --submission
```

### Scripts and their roles

All four task-specific scripts are retained. Task 1 and Task 2 keep their original
single-image commands; Task 3 predicts gender/usage and generates the submission;
Task 4 provides visual search. The remaining helpers support the notebooks or
optional execution and verification commands, so none was removed.

| Script | Why it remains |
|---|---|
| `preprocessing.py` | Dataset loading, frozen splits and image preprocessing |
| `data_audit.py` | Task 0 audit and split validation |
| `classification.py` | Shared Tasks 1-3 image loading and CNN training |
| `evaluation.py` | Classifier selection, metrics and reporting |
| `retrieval.py` | Task 4 feature extraction and retrieval metrics |
| `task1_article_type_classification.py` | Article-type prediction CLI |
| `task2_season_classification.py` | Season prediction CLI |
| `task3_occasion_gender_classification.py` | Gender/usage CLI and submission export |
| `task4_visual_search.py` | Visual-search CLI |
| `run_all_notebooks.py` | Optional full-pipeline execution command |
| `verify_classifier_artifacts.py` | Optional verification of saved classifiers without retraining |

Keep `scripts/data/`: it contains the frozen audit, split and normalization used
by the notebooks. Task 4 was not executed as part of this cleanup.

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

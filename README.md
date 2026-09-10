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

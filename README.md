# Fashion Intelligence Classification

Keras/TensorFlow implementation of article type, season, gender and occasion classification, plus neural-embedding visual search. The main notebooks reproduce the selected model configurations; separate experimental notebooks retain the broader architecture investigation.

## Layout

```text
app/client/       Next.js web interface
app/server/       FastAPI and shared inference/model definitions
scripts/          Preprocessing and single-model prediction entrypoints
notebooks/        Tasks 0-4: analysis, training and exports
notebooks-test/   Full classification experiments and their runner
models/           Only selected prediction models and the search gallery
results/          Main-run results and historical tuning tables
prediction/       Assignment prediction CSV
```

## Setup Environment

### Requirements

- Python 3.12

### Windows CPU (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m ipykernel install --user --name fashion-intelligence-classification --display-name "Fashion Intelligence Classification"
.\venv\Scripts\python.exe -m jupyterlab
```

### NVIDIA GPU using Linux/WSL2 (Bash)

```bash
python3 -m venv venv
source ./venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cuda.txt
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -m ipykernel install --user --name fashion-intelligence-classification  --display-name "Fashion Intelligence Classification"
jupyter lab
```

## Batch run the main notebooks

From the repository root, activate your TensorFlow GPU environment and run all Tasks 0-4 in order. Task 0 creates or validates the frozen preprocessing artifacts:

```bash
python notebooks/run_all.py --check
python notebooks/run_all.py

# Run selected tasks only (always executed in numerical order):
python notebooks/run_all.py --tasks 1
python notebooks/run_all.py --tasks 2 3
python notebooks/run_all.py --tasks 0 4
```

## Experimental notebooks and artifacts

`notebooks-test/` contains the full architecture investigation: four shallow MLP, four deeper MLP and eight CNN candidates per target (64 candidates across four targets). It includes alternative dense widths/depths, CNN filter counts and learning-rate/continuation experiments. Task 3 covers both gender and occasion. The main notebooks do not depend on these experimental folders.

```bash
python notebooks-test/run_all.py --check
python notebooks-test/run_all.py

# Assign different tasks to separate machines:
python notebooks-test/run_all.py --tasks 1
python notebooks-test/run_all.py --tasks 2 3
```

## Prediction scripts

Run from the repository root using the trained environment:

```bash
python scripts/article_type_classification.py path/to/image.jpg
python scripts/season_classification.py path/to/image.jpg
python scripts/gender_classification.py path/to/image.jpg
python scripts/occasion_classification.py path/to/image.jpg
python scripts/visual_search.py path/to/image.jpg --top-k 5
```

## Web application

After training/exporting all tasks, start the backend:

```bash
uvicorn app.server.main:app --reload --port 8000
```

In another terminal (PowerShell):

```powershell
cd app/client
bun install
$env:BACKEND_URL = "http://127.0.0.1:8000"
bun dev
```

Open `http://localhost:3000`; proxied API documentation is at `http://localhost:3000/api/docs`. In production, the same page is available at `https://your-domain.example/api/docs` when the domain targets only the client service. The OpenAPI document is served at `/api/openapi.json`.

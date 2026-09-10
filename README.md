# Fashion Intelligence Classification

Keras/TensorFlow implementation of article type, season, gender and occasion
classification, plus neural-embedding visual search. Training is pending; there
are no claimed Keras scores yet. Follow [TRAINING.md](TRAINING.md) to install the
appropriate environment and execute the notebooks.

## Layout

```text
app/client/       Next.js web interface
app/server/       FastAPI and shared inference/model definitions
scripts/          Preprocessing and single-model prediction entrypoints
notebooks/        Tasks 0-4: analysis, training and exports
models/           Only selected prediction models and the search gallery
results/          Current measured tables and reports
figures/          Current plots
prediction/       Assignment prediction CSV
```

## Model development

Tasks 1-3 compare shallow MLP (256), deeper MLP (256/128/64), and CNN. The CNN family compares three-block, scheduled three-block
and scheduled four-block configurations. Keras uses float32, batch size 64,
Adam at 0.001, horizontal flips and up to 30 epochs with early stopping.
Selection prioritizes validation macro-F1, then accuracy, then parameter count.
Calibration and review policy use separate validation groups. The internal test
has prior development exposure, which remains relevant after retraining.

All selected classifiers save as `.keras`.
The metadata needed for neural prediction is embedded in each Keras model.
Task 4 extracts hidden-layer features from the selected article-type model and
exports a JSON configuration with gallery embeddings and metadata. Search is
cosine similarity, not a classification probability.

## Prediction scripts

Run from the repository root using the trained environment:

```bash
python scripts/article_type_classification.py path/to/image.jpg
python scripts/season_classification.py path/to/image.jpg
python scripts/gender_classification.py path/to/image.jpg
python scripts/occasion_classification.py path/to/image.jpg
python scripts/visual_search.py path/to/image.jpg --top-k 5
```

Classification scripts accept `--model path/to/model.keras`.
Task 4 Section 9 generates the submission CSV using all four classifiers; each
individual prediction script loads only its own selected model. Include `app/server` in your submission because
the prediction entrypoints use them. Notebooks contain their own algorithms
and share only `scripts/preprocessing.py`.

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

Open `http://localhost:3000`; API documentation is at
`http://127.0.0.1:8000/docs`. The application prioritizes predicted article type
then ranks images by cosine similarity. Notebook retrieval evaluation measures
visual features without this application-level article preference.

Restart the backend when notebooks replace model files. For Docker/Portainer,
configure `.env` from `.env.example` and run `docker compose up -d --build` after
training. The image includes `models/`; rebuilding is required for updated models.
Gallery images come from the mounted private dataset. Docker inference uses CPU.

The generated model report is `results/classification_report.md`. Generate it
using the final section of Task 4 after Tasks 1-3 finish.
It rejects missing or pre-Keras summaries. No previous model scores are presented
as results of this implementation.

Run Task 1 before Task 4. If the article model is retrained, rerun Task 4; the
application rejects a gallery whose saved encoder hash differs from the model.

Model architectures, Keras compile/fit calls, callbacks, evaluation and model
selection are visible directly in each notebook. Tasks 1-3 intentionally duplicate
this code so each notebook is independently readable. Task 0 owns its audit code
and Task 4 owns embedding extraction, retrieval metrics and report generation.

Each model section separates architecture, compilation, callbacks and fitting,
with explanations before the code. Running a fit cell retrains that model; there
is no hidden cache/resume behavior. Selection, calibration and export follow in
separate sections. Task 4 explains embeddings, cosine metrics and export likewise.

Submission generation is implemented directly in Task 4 Section 9. There is no
separate `create_submission.py` command. Run Tasks 1-3 first so all four selected
classifiers exist, then run Task 4 through its final export cell.

# Train the Keras implementation

Status: code migration completed; training and runtime validation are pending.
Old model files, model scores and plots have been removed. The existing input
dataset and frozen split/normalization manifests remain. Changing framework does
not remove the internal test partition's prior development exposure.

## 1. Choose an environment

Use Python 3.11 or 3.12. Create a fresh environment instead of modifying the old PyTorch
venv. Run the commands from the repository root. Choose one setup below.

### Windows CPU (PowerShell)

```powershell
cd D:\khoile\programs\fashion-intelligence-classification
py -3.12 -m venv .venv-keras
.\.venv-keras\Scripts\python.exe -m pip install --upgrade pip
.\.venv-keras\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-keras\Scripts\python.exe -m ipykernel install --user --name fashion-keras --display-name "Python (Fashion Keras)"
.\.venv-keras\Scripts\python.exe -m jupyterlab
```

In VS Code or Jupyter, select **Python (Fashion Keras)**. Modern TensorFlow does
not use the RTX GPU on native Windows; this option trains on CPU.

### NVIDIA GPU using Linux/WSL2 (Bash)

Use an installed WSL2 Ubuntu environment with a working NVIDIA Windows driver.
The Linux environment must be created separately from the Windows environment.
Create the Linux virtual environment in your WSL home directory, not on the
Windows-mounted `/mnt/d` drive. Keep the project and dataset at their current
paths. With Python 3.11 or 3.12 and venv support available in WSL:

```bash
cd /mnt/d/khoile/programs/fashion-intelligence-classification
python3 -m venv ~/.venvs/fashion-keras
source ~/.venvs/fashion-keras/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cuda.txt
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -m ipykernel install --user --name fashion-keras --display-name "Python (Fashion Keras)"
jupyter lab
```

Confirm the GPU list is nonempty before training. If using VS Code, open the
project through its WSL connection and select the Linux kernel.
[TensorFlow's installation instructions](https://www.tensorflow.org/install/pip)
explain WSL2 and driver prerequisites. Native-Windows GPU support ended after
TensorFlow 2.10; this project uses modern TensorFlow/Keras.

## 2. Dataset

Keep the supplied files in this exact layout:

```text
dataset/
  train/styles_train.csv
  train/images_train/<id>.jpg
  test/styles_prediction.csv
  test/images_test/<id>.jpg
```

Alternatively, set `FASHION_DATA_ROOT` to the directory containing `train/` and
`test/` before starting Jupyter. Paths must be valid in the environment you use
(e.g. `/mnt/d/...` in WSL). Task 0 validates the frozen dataset release and splits.

## 3. Run notebooks in order

Select the Keras kernel, then use **Restart Kernel and Run All** for each:

1. `notebooks/task0_data_audit_eda.ipynb`
2. `notebooks/task1_article_type_classification.ipynb`
3. `notebooks/task2_season_classification.ipynb`
4. `notebooks/task3_occasion_gender_classification.ipynb`
5. `notebooks/task4_visual_search_analysis.ipynb`

Tasks 1-3 train shallow MLP, deeper MLP, and three CNN configurations. The main table contains three families; the CNN row is the
best CNN configuration. All models use the same frozen partitions. Neural models
use Adam, batch size 64, up to 30 epochs, horizontal flips during training only,
and early stopping after seven epochs without improved validation macro-F1.
CNN tuning is based on selection data, followed by separate calibration/policy
validation groups. Internal-test results are produced after selection.

Allow each notebook to finish its final export/evaluation cells. Save the
executed notebook so its tables and plots are available for later review.
Task 4 extracts embeddings from the selected article-type model and rebuilds
the cosine-similarity gallery without additional neural training. Run it after
Task 1 completes, and rerun it whenever the article-type model changes.

Each notebook includes its own architectures, Keras compile/fit calls, data
batching, metrics, model selection and export code. Only preprocessing is shared
through `scripts/preprocessing.py`. Training cells run `model.fit()` explicitly and train again when rerun. There
is no automatic cache/resume branch. Restart the kernel and Run All for a fresh
experiment. Keep the notebook open until model selection and export complete.

To execute from the terminal instead of Jupyter, use nbconvert on each notebook
in order, for example:

```bash
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=fashion-keras --ExecutePreprocessor.timeout=-1 notebooks/task1_article_type_classification.ipynb
```

## 4. Generate the new report and predictions

The final section of Task 4 generates the combined classification report.
Run **Task 4, Section 9: Generate the Assignment Submission** to create
`prediction/styles_prediction.csv`. It loads the selected models, predicts the
course-test images, validates the CSV and saves it.


The generated comparison report is `results/classification_report.md`.
The revised assignment report in the sibling `ml-a2` folder is marked pending;
its analysis must be filled from these new results during the later review.
Do not reuse numbers from the old PDF as Keras results.

## 5. What should be produced

- `models/`: four selected classifiers (`<target>_model.keras`), `visual_search_model.json`, `visual_search_embeddings.npy`,
  and `visual_search_metadata.csv`.
- `results/`: current histories, comparisons, summaries and test metrics.
- `figures/`: current plots.
- `prediction/styles_prediction.csv`: assignment predictions in template ID order.

Each `.keras` file contains the architecture, weights and a registered metadata
layer holding label order, preprocessing, temperature and review policy.
Prediction scripts load it with `keras.models.load_model(..., compile=False)`
through the shared loader. Search reuses the article model and stores the model
hash and embedding dimensions in its JSON configuration. There is no separately
trained search network. The encoder can be CNN or MLP depending on Task 1.

Only start the web application or rebuild Docker after all selected model files
exist. A successful training run does not itself verify the public deployment.

When ready for validation, keep the executed notebooks, `results/`, `figures/`
and selected models, and ask for their review. No training, full runtime tests
or model-quality validation were run during this code migration.

## Recover from pip Errno 22 on /mnt/d

If pip fails while renaming its own files in `.venv-wsl`, stop using that
environment. The later missing TensorFlow, ipykernel and Jupyter errors follow
from the failed installation. Create a new environment on the Linux filesystem:

```bash
deactivate
mkdir -p ~/.venvs
python3 -m venv ~/.venvs/fashion-keras
source ~/.venvs/fashion-keras/bin/activate
cd /mnt/d/khoile/programs/fashion-intelligence-classification
python -m pip install --upgrade pip && python -m pip install -r requirements-cuda.txt
```

Only after installation succeeds, run the GPU check and register/start Jupyter
as described above. Do not continue past an installation error.

Submission generation is implemented directly in Task 4 Section 9. There is no
separate `create_submission.py` command. Run Tasks 1-3 first so all four selected
classifiers exist, then run Task 4 through its final export cell.

## VS Code with a Colab kernel

Upload `scripts/` (including `scripts/data/`) to
`MyDrive/fashion-intelligence-classification/` and the single supplied
`A2_FashionDataset.zip` to `MyDrive/`. You do not need to upload the extracted
image folders. Each notebook's first setup cell exposes `COLAB_PROJECT` and
`DATASET_ZIP` so you can adjust these paths.

Select a Colab GPU kernel, restart it if needed, press Run All and authorize
Drive. Setup extracts the ZIP into `/content/fashion_data/`, locates the folder
containing `train/` and `test/`, and sets `FASHION_DATA_ROOT` automatically.
A completion marker avoids repeated extraction within the same runtime.
Interrupted extraction is retried; a changed archive uses a new directory.

Models, results and figures are saved in the Drive project folder. The local
extracted dataset disappears when Colab resets; the next run extracts it again.
Local Windows/WSL execution skips the Colab setup entirely.


## Parameter tuning after the first completed run

Tasks 1–3 include the additional parameter experiments in Section 5.3.6.
They test a lower learning rate for each MLP and compare unweighted versus
class-weighted continuation of the selected four-block CNN. MLP trials use at
most 20 epochs; CNN trials use at most 8 additional epochs. Early stopping
monitors selection macro-F1 with patience 4. The selected model is still chosen
from three families, and only one classifier per target is exported to models/.

With a live kernel retaining the original models and histories, run the new
Section 5.3.6 cells, then Sections 5.4 onward. Do not rerun the original fitting
cells merely to tune. A fresh Run All retrains all candidates intentionally.
Keep RESUME_SAVED_RESULTS=False for fresh runs, so saved rows cannot silently
enter a new comparison. Intentional resumption after kernel shutdown requires
the matching saved CNN, history CSVs, metadata, label order and frozen splits.

After changing the article-type model, rerun Task 4 to rebuild the retrieval
gallery with the current encoder and regenerate assignment predictions. The
classification analysis documents an exposed internal test; tune using the
selection partition, not the internal-test scores.

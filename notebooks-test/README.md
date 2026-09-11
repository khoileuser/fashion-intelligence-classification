# Isolated classification experiments

These copies retain Tasks 1–3 and add three CNN filter alternatives. Training code stays inside each notebook; `run_all.py` only executes them sequentially in fresh kernels. Production notebooks and artifacts are not replaced.

## Run from the WSL terminal

Use the same TensorFlow GPU environment that successfully trained the production notebooks:

```bash
cd /mnt/d/khoile/programs/fashion-intelligence-classification
source /home/khoile/.venvs/fashion-keras/bin/activate
python notebooks-test/run_all.py --check
python notebooks-test/run_all.py
```

`--check` checks dependencies, TensorFlow GPU visibility, frozen split availability and notebook syntax without training. If a notebook dependency is missing, install it in that environment with `python -m pip install nbclient nbformat ipykernel jupyter-client`. The runner uses the interpreter that launches it, regardless of the default Jupyter kernel. CPU training is refused unless explicitly enabled with `--allow-cpu`.

Keep the terminal running and prevent the computer from sleeping. Epoch output appears in the terminal and is recorded in `results-test/logs/`. Each completed cell is saved back into its experimental notebook. Open the files in VS Code after the run (reload an already open editor to see external updates). You can also select the Fashion Keras GPU kernel and use Run All directly in each notebook.

## Scope and comparison

| Four-block CNN | Filters | Dense head |
|---|---|---:|
| Reference | 32, 64, 128, 256 | 256 |
| Narrow | 16, 32, 64, 128 | 256 |
| Capped final block | 32, 64, 128, 128 | 256 |
| Wider final block | 32, 64, 128, 384 | 256 |

The filter comparison keeps input size, kernel size, initial learning rate, batch size, augmentation, weighting and stopping/scheduling rules the same. Each initial CNN has a maximum of 30 epochs with early stopping; this run does not use successive halving. Read `*_filter_comparison.csv` to isolate the effect of filter changes. `*_experiments.csv` also includes the existing MLPs, three-block controls and reference-CNN continuations, whose training recipes differ.

There are **12 fits per target, 48 fits total** across article type, season, gender and occasion. This deliberately duplicates the complete classification investigation and adds alternatives. It is not limited to three extra training runs. Selection uses validation macro-F1, then accuracy, then parameter count. Only the selected candidate receives calibration and internal-test evaluation; internal-test results must not guide architecture selection. There is one seed, so small improvements are provisional.

## Outputs

- `models-test/<target>_<method>.keras`: all 48 candidates, saved immediately after fitting with restored best weights. These are uncalibrated inference exports, without optimizer slots.
- `models-test/<target>_model.keras`: four selected, calibrated exports. These duplicate the winning candidates' weights with final confidence metadata.
- `results-test/`: histories, all-candidate scores, filter-only comparisons, per-class metrics, configuration JSON files, final summaries and model manifests with file hashes.
- `figures-test/`: learning curves, confusion matrices and other experimental figures.
- `notebooks-test/*.ipynb`: executed code and saved outputs. Old production analyses were removed to avoid presenting them as new measurements.

The three generated artifact directories are ignored by Git, but remain on your computer for review. The notebook sources and runner can be committed. No experimental model is automatically copied to `models/`, and visual search is not rebuilt in this round.

## Restarting

The runner stops on the first error and preserves completed outputs and candidate exports. Running the same command again skips notebooks it previously completed with unchanged source and model files still present. The failed notebook starts from its beginning; individual training runs are not resumed from optimizer checkpoints. Direct VS Code executions are not recorded as runner completions.

```bash
# Run only selected tasks:
python notebooks-test/run_all.py --tasks 2 3

# Deliberately rerun completed experiments, replacing their test artifacts:
python notebooks-test/run_all.py --rerun
```

After all runs finish, ask for a review of `results-test/` and `models-test/`. Promotion into the production notebooks and models is a separate step.

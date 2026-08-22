# Team Handoff

## Ownership

| Assignment work | Owner | Primary notebook | Required output |
|---|---|---|---|
| Project setup, shared audit, EDA, split, normalization | Member 1; reviewed by all | `task0_data_audit_eda.ipynb` | `scripts/data/audit.json`, `image_audit.csv`, `splits.csv`, `normalization.json` |
| Task 1: article type | Member 2 | `task1_article_type_classification.ipynb` | `models/article_type_model.pt` |
| Task 2: season | Member 3 | `task2_season_classification.ipynb` | `models/season_model.pt` |
| Task 3: gender and occasion | Member 4 | `task3_occasion_gender_classification.ipynb` | `models/gender_model.pt`, `models/usage_model.pt` |
| Task 4: visual search, prototype, integration | Member 1 | `task4_visual_search_analysis.ipynb`, `app/` | Search encoder/index, application, consolidated prediction CSV |

The assignment's occasion label is the metadata column `usage`. Member 4 trains gender and usage as two independently saved classifiers but reports them together as Task 3.

## Shared foundation already provided

- `scripts/data/audit.json`
- `scripts/data/image_audit.csv`
- `scripts/data/splits.csv`
- `scripts/data/normalization.json`
- Shared loading and image helpers in `scripts/preprocessing.py`
- Flat checkpoint requirements in `docs/CLASSIFIER_CONTRACT.md`

The frozen split passed duplicate/name-group isolation and training-label coverage checks for all four targets. It repaired the rare `Shoe Laces` article type by moving its complete group to training. Do not regenerate `splits.csv` to improve a metric.

The supplied images and CSVs remain private and are not committed. Every member must obtain the authorized assignment dataset and place it under the documented `dataset/` layout. Keep the collaboration repository private because the frozen manifests contain course item IDs and hashes. The existing classifier models and histories are prototype mocks used by the web application; they are not completed assignment results.

## Start from the handoff

From the repository root:

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On a supported NVIDIA machine, install and verify the CUDA build:

```powershell
python -m pip install --force-reinstall -r requirements-cuda.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Before training, confirm that the private dataset and shared manifests exist:

```powershell
@(
  'dataset/train/styles_train.csv',
  'dataset/train/images_train',
  'dataset/test/styles_prediction.csv',
  'dataset/test/images_test',
  'scripts/data/splits.csv',
  'scripts/data/normalization.json'
) | ForEach-Object { if (-not (Test-Path $_)) { throw "Missing $_" } }
```

Confirm that every member has the same authorized CSV release:

```powershell
Get-FileHash dataset\train\styles_train.csv -Algorithm SHA256
Get-FileHash dataset\test\styles_prediction.csv -Algorithm SHA256
```

Expected hashes are `54D503743F79F22BF03F9E2C216B53E75D5F2129EDA5934CFA25C642C70AF44D` and `7F83F219D0FB7F2EC86B8B5C0BA28C2CFADD8F8E6E85A36C9919C9AB5C41315E`, respectively. Stop and contact the integrator if either differs.

Open only the assigned notebook in Jupyter. Do not use `run_all_notebooks.py --start-at ...` during parallel development because it runs the chosen task and every later task, overwriting unrelated member artifacts.

## Required modelling workflow

1. Record the seed, device, hypothesis, and changed parameter.
2. Run the majority and handcrafted baseline before neural candidates.
3. Execute both ordinary-loss and weighted/class-balanced candidates. The notebooks reinitialize the model, data-loader generator, and framework seed identically for each run.
4. Fit normalization, weighting, feature transforms, and augmentation on training rows only.
5. Select the method using validation macro F1 plus calibration/efficiency evidence. Do not use the current mock checkpoint as a candidate result.
6. Evaluate the internal test split only after writing down the frozen selection. Existing prototype test outputs must not be used for tuning.
7. Replace every observation, failure-analysis, and ultimate-judgement prompt with conclusions supported by executed outputs.
8. Save only the assigned checkpoint, selected history, and validation-comparison filenames, then verify the checkpoint on CPU through the matching script.
9. Return at least one credible work with a comparable goal for independent evaluation. State differences in dataset, split, model constraints, and metric definition; do not present non-comparable headline scores as a direct ranking.

## Target requirements

### Member 2: article type

- Address the long tail and compare ordinary against class-balanced loss under controlled settings.
- Confirm the scaled HOG+HSV logistic-regression pipeline reaches convergence; do not treat the baseline as reliable if a convergence warning remains.
- Report macro F1, top-3 accuracy, and explicit head/medium/tail behaviour.
- Analyse major confusions, confidence/calibration, mild-corruption robustness, latency, and model size.
- Justify the final selection: the current prototype's HOG+HSV validation macro F1 is higher than its CNN result, so the CNN cannot be selected without new evidence or a defensible non-score trade-off.

### Member 3: season

- Compare majority, HOG+HSV, ordinary-loss CNN, and weighted-loss CNN.
- Report macro F1, confusion, calibration, robustness, latency, and size.
- Analyse errors by article type and distinguish visual evidence from seasonal catalogue priors.

### Member 4: gender and occasion/usage

- Deliver two independently loadable classifiers and do not combine their scores.
- Preserve literal `NA` usage unless teaching staff confirms another interpretation; remove only the genuinely blank usage row.
- Compare ordinary and weighting strategies separately for each target.
- Analyse imbalance, ambiguous/unisex presentation, rare usage labels, article-type shortcuts, calibration, robustness, latency, and size.

## Files to return to the integrator

- Executed assigned notebook with no unresolved prompts or errors
- Final checkpoint(s), selected history CSV(s), and task comparison CSV(s)
- Validation-selection table and one-time internal-test metrics
- Report-ready findings: winning method, key evidence, important failures, limitations, runtime, and artifact size
- Confirmation that the matching inference script loads and predicts on CPU
- At least one literature comparison with complete citation and comparability caveats

## Do not change

- Frozen IDs or group assignments
- Image size, deterministic evaluation transform, or training-fitted normalization without team approval
- Target spelling or label capitalization
- Flat checkpoint filenames and required checkpoint keys
- The rule prohibiting pretrained weights or unapproved external data in submitted final models

Read `CLASSIFIER_CONTRACT.md` and `EXPERIMENT_PROTOCOL.md` before saving a final model.

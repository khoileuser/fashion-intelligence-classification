# Current Status

Last handoff review: 22 August 2026.

## Ready for team use

- The supplied dataset layout passes the project audit on the setup machine.
- Task 0 is executed and documents the metadata/image audit, repair of 21 product names split by unquoted CSV commas, exact-duplicate label conflicts, target imbalance and association, leakage risks, split balance, frozen group split, training-only normalization, label policy, targeted anomaly examples, and limitations.
- The versioned handoff artifacts are `scripts/data/audit.json`, `image_audit.csv`, `splits.csv`, and `normalization.json`.
- The frozen manifest contains 27,051 training, 5,842 validation, and 5,718 internal-test rows. No duplicate/name group crosses a split, and every validation/test label occurs in training.
- The API, Next.js client, Docker files, inference scripts, visual-search work, and classifier checkpoint contract are present. Classifier inference supports both the compact CNN and the fitted HOG+HSV logistic-regression pipeline.

## Prototype artifacts—not final results

The existing article-type, season, gender, and usage checkpoints and history CSVs were created to exercise the notebooks, inference scripts, API, and web prototype. They are mocks for integration and must not be described as completed assignment models. In particular, the controlled candidate comparisons, final analysis, and ultimate judgements have not been completed.

Each classifier owner must replace only their assigned files:

| Owner | Assignment task | Files to replace |
|---|---|---|
| Member 2 | Task 1: article type | `models/article_type_model.pt`, `models/article_type_history.csv`, `models/article_type_comparison.csv` |
| Member 3 | Task 2: season | `models/season_model.pt`, `models/season_history.csv`, `models/season_comparison.csv` |
| Member 4 | Task 3: gender and usage | `models/gender_model.pt`, `models/gender_history.csv`, `models/gender_comparison.csv`, `models/usage_model.pt`, `models/usage_history.csv`, `models/usage_comparison.csv` |

Member 1 maintains Task 0, Task 4, the prototype, integration, prediction generation, and packaging.

## Next work

1. Each classifier owner sets up the authorized private dataset and opens only their assigned notebook.
2. They run controlled baselines and candidates, produce one full comparison table, apply the predeclared validation macro-F1/calibration/complexity rule, and do not tune after viewing internal-test results.
3. They replace every task-specific observation or conclusion prompt with measured analysis.
4. They save a checkpoint matching `CLASSIFIER_CONTRACT.md` and verify it through the matching CPU inference script.
5. The integrator merges returned notebooks and artifacts, generates `prediction/styles_prediction.csv`, and performs the clean full-pipeline and application check.

Do not use `run_all_notebooks.py --start-at ...` for parallel member training: it executes the selected task and all later tasks. The sequential runner is reserved for final integration or recovery from a known pipeline failure.

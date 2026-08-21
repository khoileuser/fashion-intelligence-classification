# Team Handoff

## Ownership

| Assignment work | Owner | Primary notebook | Required output |
|---|---|---|---|
| Shared audit, EDA, split, normalization | Member 1; reviewed by all | `task0_data_audit_eda.ipynb` | `scripts/data/*.json` and `splits.csv` |
| Task 1: article type | Member 1 | `task1_article_type_classification.ipynb` | `models/article_type_model.pt` |
| Task 2: season | Member 2 | `task2_season_classification.ipynb` | `models/season_model.pt` |
| Task 3: gender and occasion | Member 3 | `task3_occasion_gender_classification.ipynb` | `models/gender_model.pt`, `models/usage_model.pt` |
| Task 4: visual search | Member 4 | `task4_visual_search_analysis.ipynb` | Encoder, embeddings, and gallery metadata |

The assignment's occasion label is the metadata column `usage`. Member 3 trains gender and usage as two separate classifiers but reports them together as one assignment task.

## What Member 1 provides first

Before anyone trains, Member 1 must complete Task 0 and share:

- `scripts/data/audit.json`
- `scripts/data/image_audit.csv`
- `scripts/data/splits.csv`
- `scripts/data/normalization.json`

The team must review the audit and split. Once modelling starts, `splits.csv` is frozen.

## How to work

1. Install `requirements.txt` and run Task 0 first.
2. Open only your assigned task notebook.
3. Run the majority/pixel baseline before the neural model.
4. Keep every architecture, training step, plot, metric, and interpretation visible in the notebook.
5. Select using validation results; open the internal test split only after the method is frozen.
6. Replace observation prompts with real results before submission.
7. Save the final output at the exact path in the ownership table.
8. Run the matching small script to confirm that the saved artifact loads on CPU.

## Target requirements

### Member 1: article type

- Address the long tail and compare ordinary versus weighted loss.
- Report macro F1, top-3 accuracy, and head/medium/tail behavior.
- Analyse major confusions, confidence, robustness, latency, and size.
- Integrate all returned models, generate the CSV, and verify the API/client.

### Member 2: season

- Compare majority, ordinary-loss CNN, and weighted-loss CNN.
- Report macro F1, confusion, calibration, robustness, latency, and size.
- Analyse errors by article type and the subjective visual meaning of season.

### Member 3: gender and occasion/usage

- Deliver two independently loadable classifiers.
- Preserve literal `NA` usage unless teaching staff confirms otherwise.
- Analyse imbalance, ambiguous/unisex presentation, rare usage labels, calibration, robustness, latency, and size.
- Compare the two targets without combining their scores.

### Member 4: visual search

- Compare the pixel baseline with the from-scratch contrastive encoder.
- Use training rows as evaluation gallery and internal-test rows as queries.
- Report Precision@K, hit recall@K, reciprocal rank, attribute agreement, latency, and memory.
- Show a fixed qualitative panel containing successes and failures.

## Do not change

- Frozen IDs or group assignments
- Image size, deterministic evaluation transform, or training-fitted normalization without team approval
- Target spelling or label capitalization
- Flat checkpoint filenames and required checkpoint keys
- The rule prohibiting pretrained weights or unapproved external data

Read `CLASSIFIER_CONTRACT.md` before saving a final model.

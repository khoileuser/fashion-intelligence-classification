# Classifier Checkpoint Contract

Tasks 1–3 define and train their models inside notebooks. The small scripts and API can load them only if every owner follows this flat checkpoint contract.

| Target        | File                           |
| ------------- | ------------------------------ |
| `articleType` | `models/article_type_model.pt` |
| `season`      | `models/season_model.pt`       |
| `gender`      | `models/gender_model.pt`       |
| `usage`       | `models/usage_model.pt`        |

Each checkpoint must be saved as a dictionary. A compact-CNN checkpoint uses:

```python
{
    "model_type": "compact_cnn",
    "target": "gender",
    "labels": ["Boys", "Girls", "Men", "Unisex", "Women"],
    "state_dict": model.state_dict(),
    "mean": [0.0, 0.0, 0.0],
    "std": [1.0, 1.0, 1.0],
    "image_size": [96, 128],
    "dropout": 0.2,
    "selection_metric": "validation_macro_f1",
    "best_validation_macro_f1": 0.0,
    "test_metrics": {},
    "seed": 2753,
}
```

Do not save a pickled whole PyTorch model. The ordered labels, architecture-compatible state dictionary, and preprocessing values must be present.

If the controlled investigation selects the HOG+HSV logistic-regression pipeline, save the fitted scikit-learn pipeline instead of forcing a weaker CNN merely for application compatibility:

```python
{
    "model_type": "hog_hsv_logistic_regression",
    "target": "articleType",
    "labels": list(linear_baseline.classes_),
    "estimator": linear_baseline,
    "feature_config": {
        "image_size": [48, 64],
        "orientations": 9,
        "pixels_per_cell": [8, 8],
        "cells_per_block": [2, 2],
        "hsv_bins": 8,
    },
    "selection_metric": "validation_macro_f1",
    "best_validation_macro_f1": 0.0,
    "validation_metrics": {},
    "test_metrics": {},
    "seed": 2753,
}
```

The API supports both model types. Other final architectures require a named `model_type`, a matching construction branch in `app/server/utils/classifier.py`, and an integration smoke test before selection.

Before handoff, run the appropriate script on CPU:

```powershell
python scripts/task1_article_type_classification.py path\to\image.jpg
python scripts/task2_season_classification.py path\to\image.jpg
python scripts/task3_occasion_gender_classification.py --image path\to\image.jpg --target gender
```

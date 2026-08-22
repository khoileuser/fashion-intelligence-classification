# Classifier Checkpoint Contract

Tasks 1–3 define and train their models inside notebooks. The small scripts and API can load them only if every owner follows this flat checkpoint contract.

| Target        | File                           |
| ------------- | ------------------------------ |
| `articleType` | `models/article_type_model.pt` |
| `season`      | `models/season_model.pt`       |
| `gender`      | `models/gender_model.pt`       |
| `usage`       | `models/usage_model.pt`        |

Each checkpoint must be saved as a dictionary:

```python
{
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

Do not save a pickled whole model. The ordered labels, architecture-compatible state dictionary, and preprocessing values must be present.

Before handoff, run the appropriate script on CPU:

```powershell
python scripts/task1_article_type_classification.py path\to\image.jpg
python scripts/task2_season_classification.py path\to\image.jpg
python scripts/task3_occasion_gender_classification.py --image path\to\image.jpg --target gender
```

# Shared Preprocessing

The single public helper is `scripts/preprocessing.py`. Like the rice sample, it contains only small reusable data/image functions. Task 0 visibly implements the audit, duplicate grouping, split creation, and normalization fitting.

## Data rules

- Read IDs as strings.
- Disable automatic NA conversion so literal `NA` usage is preserved.
- Drop only trailing `Unnamed` CSV columns.
- Attach images by ID and report missing images before filtering.
- Convert decoded images to RGB and resize to 96×128 pixels.
- Exclude a blank label only from that label's task.

## Shared workflow

```powershell
jupyter lab notebooks/task0_data_audit_eda.ipynb
```

Run the notebook sections in order. The audit writes `scripts/data/audit.json` plus `image_audit.csv` containing dimensions, modes, decode status, and SHA-256 hashes. The split joins normalized product-name and exact-hash groups, then creates one deterministic 70/15/15 manifest at `scripts/data/splits.csv`. Normalization writes training-only RGB statistics to `scripts/data/normalization.json`.

Task 0 reuses an existing split rather than overwriting it. After experiments begin, do not delete `splits.csv` merely to obtain a more favorable assignment.

## Notebook behavior

The notebooks import `task_frame`, `IMAGE_SIZE`, and the normalization path from `preprocessing.py`. Training notebooks define their own PyTorch datasets and training-only augmentation so the submitted work remains visible. Validation, test, course-test, and application inference use deterministic preprocessing.

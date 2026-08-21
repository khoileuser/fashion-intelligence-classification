# Current Status

Last execution check: 21 August 2026.

## Complete

- The supplied dataset is installed under `dataset/` and passes the runner's layout check.
- Task 0 completed and produced the image audit, frozen group split, and training-only normalization under `scripts/data/`.
- The frozen split preserves duplicate-group isolation and guarantees that every article type, season, gender, and usage label has at least one training example.
- Task 1 completed on an NVIDIA RTX 3060 with CUDA and produced `models/article_type_model.pt`.
- All five notebooks use the `fashion-intelligence` project kernel and have valid notebook cell IDs.
- The API, Next.js client, Docker deployment files, prediction/search scripts, and team experiment documentation are present.

## Next work

- Member 2: run Task 2 and review/export `models/season_model.pt`.
- Member 3: run Task 3 and review/export `models/gender_model.pt` and `models/usage_model.pt`.
- Member 4: run Task 4 and review/export the visual-search encoder, embeddings, and gallery metadata.
- Member 1: integrate returned models, generate and validate `prediction/styles_prediction.csv`, then perform a clean API/client deployment check.
- Every owner must replace remaining observation prompts with conclusions supported by their executed notebook outputs.

Resume the automatic pipeline without repeating completed Tasks 0 and 1:

```powershell
python scripts/run_all_notebooks.py --device cuda --start-at task2
```

The runner checks the frozen split contract before a resumed run. If a stale split lacks training coverage for a rare label, rerun Task 0; it repairs coverage by moving the label's entire duplicate group to training.

# Notebook-First Foundation

The repository now follows the rice project structurally and pedagogically:

- Five task notebooks contain the important analysis and modelling work.
- `scripts/preprocessing.py` contains only small shared data/image helpers; Task 0 owns the audit and split algorithms.
- Four small task scripts load completed models and demonstrate prediction/search.
- `app/server/utils/` contains inference-only architecture and checkpoint loaders for the API.
- Model outputs are flat files under `models/`.
- There is no installable package, YAML configuration layer, or artifacts hierarchy.

Member 1 has supplied the notebook templates, shared preprocessing, checkpoint contract, prediction CSV integration, API, and UI. The repository intentionally contains no fabricated model results. Every member must execute their notebook and replace interpretation prompts with measured evidence.

The model definitions in Tasks 1–3 mirror `app/server/utils/modeling.py` so their state dictionaries load without task-specific API code. Task 4 uses the corresponding embedding architecture.

The deployable interface is a Next.js application under `app/client/`. It calls FastAPI through its same-origin backend proxy; this application layer does not alter the notebook-first modelling workflow.

"""Execute Tasks 0-4 in order and verify the application artifacts."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TASKS = (
    (
        "task0",
        ROOT / "notebooks" / "task0_data_audit_eda.ipynb",
        (
            ROOT / "scripts" / "data" / "image_audit.csv",
            ROOT / "scripts" / "data" / "splits.csv",
            ROOT / "scripts" / "data" / "normalization.json",
        ),
    ),
    (
        "task1",
        ROOT / "notebooks" / "task1_article_type_classification.ipynb",
        (ROOT / "models" / "article_type_model.pt",),
    ),
    (
        "task2",
        ROOT / "notebooks" / "task2_season_classification.ipynb",
        (ROOT / "models" / "season_model.pt",),
    ),
    (
        "task3",
        ROOT / "notebooks" / "task3_occasion_gender_classification.ipynb",
        (
            ROOT / "models" / "gender_model.pt",
            ROOT / "models" / "usage_model.pt",
        ),
    ),
    (
        "task4",
        ROOT / "notebooks" / "task4_visual_search_analysis.ipynb",
        (
            ROOT / "models" / "visual_search_model.pt",
            ROOT / "models" / "visual_search_embeddings.npy",
            ROOT / "models" / "visual_search_metadata.csv",
        ),
    ),
)


def dataset_root() -> Path:
    """Resolve the same dataset override used by the notebooks."""
    return Path(os.environ.get("FASHION_DATA_ROOT", ROOT / "dataset")).expanduser().resolve()


def validate_dataset(path: Path) -> None:
    """Fail early when the documented dataset layout is incomplete."""
    required = (
        path / "train" / "styles_train.csv",
        path / "train" / "images_train",
        path / "test" / "styles_prediction.csv",
        path / "test" / "images_test",
    )
    missing = [item for item in required if not item.exists()]
    if missing:
        formatted = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(
            f"Dataset layout is incomplete under {path}:\n{formatted}\n"
            "See dataset/README.md before running the notebooks."
        )


def execute_notebook(
    notebook: Path,
    kernel: str,
    timeout: int,
    environment: dict[str, str],
) -> None:
    """Execute one notebook in place and preserve its outputs as evidence."""
    command = (
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.kernel_name={kernel}",
        f"--ExecutePreprocessor.timeout={timeout}",
        str(notebook),
    )
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-at",
        choices=[task[0] for task in TASKS],
        default="task0",
        help="Resume at this task after fixing a failed run (default: task0).",
    )
    parser.add_argument(
        "--kernel", default="python3", help="Jupyter kernel name (default: python3)."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=-1,
        help="Per-cell timeout in seconds; -1 disables it (default: -1).",
    )
    parser.add_argument(
        "--force-audit",
        action="store_true",
        help="Repeat the full Task 0 image decode and SHA-256 audit.",
    )
    args = parser.parse_args()

    data_path = dataset_root()
    validate_dataset(data_path)
    environment = os.environ.copy()
    environment["FASHION_DATA_ROOT"] = str(data_path)
    environment["FASHION_RUN_FULL_AUDIT"] = "1" if args.force_audit else "auto"

    start_index = next(
        index for index, task in enumerate(TASKS) if task[0] == args.start_at
    )
    selected_tasks = TASKS[start_index:]
    print(f"Dataset: {data_path}")
    print("Tasks: " + " -> ".join(task[0] for task in selected_tasks))

    pipeline_started = time.perf_counter()
    for position, (name, notebook, expected) in enumerate(selected_tasks, start=1):
        print(f"\n[{position}/{len(selected_tasks)}] Running {name}: {notebook.name}")
        started = time.perf_counter()
        execute_notebook(notebook, args.kernel, args.timeout, environment)
        missing = [path for path in expected if not path.is_file()]
        if missing:
            raise RuntimeError(
                f"{name} completed without expected outputs: "
                + ", ".join(str(path) for path in missing)
            )
        print(f"Completed {name} in {(time.perf_counter() - started) / 60:.1f} minutes")
        for path in expected:
            print(f"  OK {path.relative_to(ROOT)}")

    all_expected = [path for _, _, outputs in TASKS for path in outputs]
    missing = [path for path in all_expected if not path.is_file()]
    if missing:
        raise RuntimeError(
            "The run finished, but the complete application artifact set is missing: "
            + ", ".join(str(path.relative_to(ROOT)) for path in missing)
        )

    print(f"\nAll selected tasks completed in {(time.perf_counter() - pipeline_started) / 60:.1f} minutes.")
    print("The model artifacts required by the API are ready under models/.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"\nPipeline stopped: {error}", file=sys.stderr)
        raise SystemExit(1) from error

"""Execute Tasks 0-4 and verify classification experiments and search artifacts."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PROJECT_KERNEL = "fashion-intelligence"

def comparison_outputs(*stems):
    return tuple(ROOT / ('models' if name == 'model.pt' else 'results') / f'{stem}_{name}'
                 for stem in stems for name in
                 ('comparison.csv', 'cnn_tuning.csv', 'model.pt', 'summary.json', 'test_metrics.csv'))


TASKS = (
    (
        "task0",
        ROOT / "notebooks" / "task0_data_audit_eda.ipynb",
        (
            ROOT / "scripts" / "data" / "audit.json",
            ROOT / "scripts" / "data" / "image_audit.csv",
            ROOT / "scripts" / "data" / "splits.csv",
            ROOT / "scripts" / "data" / "normalization.json",
        ),
    ),
    (
        "task1",
        ROOT / "notebooks" / "task1_article_type_classification.ipynb",
        comparison_outputs('article_type'),
    ),
    (
        "task2",
        ROOT / "notebooks" / "task2_season_classification.ipynb",
        comparison_outputs('season'),
    ),
    (
        "task3",
        ROOT / "notebooks" / "task3_occasion_gender_classification.ipynb",
        comparison_outputs('gender', 'usage'),
    ),
    (
        "task4",
        ROOT / "notebooks" / "task4_visual_search_analysis.ipynb",
        (
            ROOT / "models" / "visual_search_model.pt",
            ROOT / "models" / "visual_search_embeddings.npy",
            ROOT / "models" / "visual_search_metadata.csv",
            ROOT / "results" / "visual_search_history.csv",
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


def validate_split_contract() -> None:
    """Ensure every evaluated target label is represented in training."""
    from scripts.preprocessing import TARGETS, load_metadata, load_splits

    frame = load_metadata().merge(load_splits(), on="id", validate="one_to_one")
    failures: dict[str, list[str]] = {}
    for target in TARGETS:
        valid = frame["has_image"] & frame[target].astype("string").str.strip().ne("")
        train_labels = set(frame.loc[valid & frame["split"].eq("train"), target])
        unsupported = sorted(set(frame.loc[valid, target]) - train_labels)
        if unsupported:
            failures[target] = unsupported
    if failures:
        details = "; ".join(f"{target}: {labels}" for target, labels in failures.items())
        raise RuntimeError(
            "Frozen split contains labels absent from training ("
            f"{details}). Run the pipeline from task0 to repair it."
        )


def configure_accelerator(requested: str, environment: dict[str, str]) -> None:
    """Validate the runner environment and pass its device choice to notebooks."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is not installed; install requirements.txt first") from error

    environment["FASHION_DEVICE"] = requested
    if requested == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
        print(f"Device: CPU (forced) | PyTorch {torch.__version__}")
        return

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable in the runner environment. "
            f"PyTorch={torch.__version__}, CUDA build={torch.version.cuda}. "
            "Install requirements-cuda.txt, restart the terminal/kernel, and retry."
        )

    if torch.cuda.is_available():
        print(
            f"Device: CUDA | {torch.cuda.get_device_name(0)} | "
            f"PyTorch {torch.__version__} | CUDA {torch.version.cuda}"
        )
    else:
        print(
            f"Device: CPU (automatic fallback) | PyTorch {torch.__version__}. "
            "Install requirements-cuda.txt to enable an NVIDIA GPU."
        )


def ensure_project_kernel(kernel: str) -> None:
    """Install the runner's default kernel into the active virtual environment."""
    if kernel != PROJECT_KERNEL:
        return
    try:
        from ipykernel.kernelspec import install
    except ImportError as error:
        raise RuntimeError("ipykernel is not installed; install requirements.txt first") from error
    location = install(
        kernel_name=PROJECT_KERNEL,
        display_name="Python (Fashion Intelligence)",
        prefix=sys.prefix,
    )
    print(f"Jupyter kernel: {location}")

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
        "--kernel",
        default=PROJECT_KERNEL,
        help=f"Jupyter kernel name (default: {PROJECT_KERNEL}).",
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
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Training device; cuda fails instead of silently falling back (default: auto).",
    )
    args = parser.parse_args()

    ensure_project_kernel(args.kernel)
    data_path = dataset_root()
    validate_dataset(data_path)
    environment = os.environ.copy()
    environment["FASHION_DATA_ROOT"] = str(data_path)
    environment["FASHION_RUN_FULL_AUDIT"] = "1" if args.force_audit else "auto"
    configure_accelerator(args.device, environment)

    start_index = next(
        index for index, task in enumerate(TASKS) if task[0] == args.start_at
    )
    selected_tasks = TASKS[start_index:]
    if start_index > 0:
        validate_split_contract()
    print(f"Dataset: {data_path}")
    print("Tasks: " + " -> ".join(task[0] for task in selected_tasks))

    pipeline_started = time.perf_counter()
    for position, (name, notebook, expected) in enumerate(selected_tasks, start=1):
        print(f"\n[{position}/{len(selected_tasks)}] Running {name}: {notebook.name}")
        started = time.perf_counter()
        execute_notebook(notebook, args.kernel, args.timeout, environment)
        if name == "task0":
            validate_split_contract()
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
    print("Classification results are saved under results/; selected prediction checkpoints under models/.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"\nPipeline stopped: {error}", file=sys.stderr)
        raise SystemExit(1) from error

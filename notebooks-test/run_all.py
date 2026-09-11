"""Execute experimental notebooks sequentially using this Python environment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = [
    "task1_article_type_classification.ipynb",
    "task2_season_classification.ipynb",
    "task3_occasion_gender_classification.ipynb",
]


def fingerprint(notebook):
    return hashlib.sha256(json.dumps(
        [(cell.cell_type, cell.source) for cell in notebook.cells],
        ensure_ascii=False,
    ).encode()).hexdigest()


def exported_models(path):
    """Read explicit export paths; avoid unreliable WSL/DrvFS directory scans."""
    manifest = ROOT / "results-test" / (path.stem + "_model_manifest.csv")
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"Empty model manifest: {manifest}")
    names = {row["file"] for row in rows}
    for target in {row["target"] for row in rows}:
        stem = "article_type" if target == "articleType" else target
        names.add(f"{stem}_model.keras")
    for name in names:
        if Path(name).name != name or not name.endswith(".keras"):
            raise RuntimeError(f"Invalid model filename in {manifest}: {name}")
        if not (ROOT / "models-test" / name).is_file():
            raise FileNotFoundError(ROOT / "models-test" / name)
    return sorted(names)


def execute_notebook(path, kernel_manager, log_path):
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    def save():
        temporary = path.with_suffix(".ipynb.tmp")
        nbformat.write(notebook, temporary)
        temporary.replace(path)

    with log_path.open("w", encoding="utf-8") as log:
        class StreamingClient(NotebookClient):
            def process_message(self, msg, cell, cell_index):
                if msg["msg_type"] == "stream":
                    text = msg["content"]["text"]
                    print(text, end="", flush=True)
                    log.write(text)
                    log.flush()
                return super().process_message(msg, cell, cell_index)

        client = StreamingClient(
            notebook, km=kernel_manager, timeout=None, allow_errors=False,
            resources={"metadata": {"path": str(ROOT)}},
        )
        try:
            with client.setup_kernel(cwd=str(ROOT)):
                for index, cell in enumerate(notebook.cells):
                    if cell.cell_type == "code":
                        message = f"\n[{path.name}] Cell {index + 1}/{len(notebook.cells)}\n"
                        print(message, end="", flush=True)
                        log.write(message)
                        client.execute_cell(cell, index)
                        save()
        finally:
            save()
            # An externally supplied kernel manager is owned by this runner.
            if kernel_manager.has_kernel:
                kernel_manager.shutdown_kernel(now=True)
            kernel_manager.cleanup_resources()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check environment and notebook syntax without training")
    parser.add_argument("--rerun", action="store_true", help="Rerun even notebooks already completed with identical source")
    parser.add_argument("--tasks", nargs="+", choices=["1", "2", "3"], default=["1", "2", "3"])
    parser.add_argument("--allow-cpu", action="store_true", help="Explicitly permit slow CPU training")
    args = parser.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    import nbformat
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient  # Dependency check before any training.
    import tensorflow as tf
    from scripts.preprocessing import NORMALISATION_PATH, load_splits

    print(f"Python: {sys.executable}\nTensorFlow: {tf.__version__}", flush=True)
    gpus = tf.config.list_physical_devices("GPU")
    print(f"Available GPUs: {gpus}", flush=True)
    if not gpus and not args.allow_cpu:
        raise SystemExit("No TensorFlow GPU found. Use the Fashion Keras WSL environment, or explicitly pass --allow-cpu.")
    if args.allow_cpu:
        os.environ["FASHION_TEST_ALLOW_CPU"] = "1"
    if not NORMALISATION_PATH.is_file():
        raise SystemExit("Missing normalization metadata; complete Task 0 first.")
    load_splits()
    paths = [ROOT / "notebooks-test" / NOTEBOOKS[int(task) - 1] for task in dict.fromkeys(args.tasks)]
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"{path.name}:cell{index + 1}", "exec")
    if args.check:
        print("Environment, frozen split availability and notebook syntax checked. No training executed.")
        return

    results = ROOT / "results-test"
    logs = results / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    state_path = results / "run_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    # Temporary kernel specification pins training to the interpreter running this script.
    with tempfile.TemporaryDirectory(prefix="fashion-test-kernel-") as temporary:
        specification = Path(temporary) / "fashion-experiment"
        specification.mkdir()
        (specification / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Fashion experiment", "language": "python",
        }), encoding="utf-8")
        manager = KernelSpecManager(kernel_dirs=[temporary])
        for path in paths:
            digest = fingerprint(nbformat.read(path, as_version=4))
            previous = state.get(path.name, {})
            artifacts_present = bool(previous.get("models")) and all(
                (ROOT / "models-test" / name).is_file() for name in previous.get("models", [])
            )
            if not args.rerun and previous.get("status") == "complete" and previous.get("source_sha256") == digest and artifacts_present:
                print(f"Skipping completed {path.name}; use --rerun to refresh evaluation while reusing saved candidates.", flush=True)
                continue
            state[path.name] = {"status": "running", "source_sha256": digest, "started": time.time()}
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            kernel = KernelManager(kernel_name="fashion-experiment", kernel_spec_manager=manager)
            try:
                execute_notebook(path, kernel, logs / (path.stem + ".log"))
                exported = exported_models(path)
            except BaseException:
                state[path.name]["status"] = "failed"
                state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
                print(f"Stopped at {path.name}. Completed cells and candidate models were saved. Fix the error and run again.", file=sys.stderr)
                raise
            state[path.name].update(status="complete", finished=time.time(), models=exported)
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print("All requested experiments completed. Review results-test and models-test; production files were not replaced.")


if __name__ == "__main__":
    main()

"""Execute main Tasks 0-4 notebooks sequentially and save their outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = [
    "task0_data_audit_eda.ipynb",
    "task1_article_type_classification.ipynb",
    "task2_season_classification.ipynb",
    "task3_occasion_gender_classification.ipynb",
    "task4_visual_search_analysis.ipynb",
]


def fingerprint(notebook):
    return hashlib.sha256(json.dumps(
        [(cell.cell_type, cell.source) for cell in notebook.cells],
        ensure_ascii=False,
    ).encode()).hexdigest()


def artifact_paths(path):
    task = NOTEBOOKS.index(path.name)
    if task == 0:
        return ['scripts/data/splits.csv', 'scripts/data/normalization.json', 'scripts/data/image_audit.csv']
    if task == 4:
        return ['models/visual_search_model.json', 'models/visual_search_embeddings.npy',
                'models/visual_search_metadata.csv', 'results/visual_search_history.csv',
                'prediction/styles_prediction.csv']
    stems = {1: ['article_type'], 2: ['season'], 3: ['gender', 'usage']}[task]
    return [f'{directory}/{stem}_{suffix}' for stem in stems
            for directory, suffix in [('models', 'model.keras'), ('results', 'summary.json'), ('results', 'comparison.csv')]]


def dependency_fingerprint(path):
    task = NOTEBOOKS.index(path.name)
    dependencies = [ROOT / 'scripts/preprocessing.py']
    if task != 0:
        dependencies += [ROOT / 'scripts/data/splits.csv', ROOT / 'scripts/data/normalization.json']
    if task == 4:
        dependencies += [ROOT / folder / f'{stem}_{suffix}'
                         for stem in ['article_type', 'season', 'gender', 'usage']
                         for folder, suffix in [('models', 'model.keras'), ('results', 'summary.json'), ('results', 'comparison.csv')]]
    digest = hashlib.sha256()
    for dependency in dependencies:
        digest.update(str(dependency.relative_to(ROOT)).encode())
        if not dependency.is_file():
            digest.update(b'MISSING')
            continue
        with dependency.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


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
    parser.add_argument("--tasks", nargs="+", choices=["0", "1", "2", "3", "4"], default=["0", "1", "2", "3", "4"])
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
    if any(task != "0" for task in args.tasks) and not gpus and not args.allow_cpu:
        raise SystemExit("No TensorFlow GPU found. Use the Fashion Keras WSL environment, or explicitly pass --allow-cpu.")
    if args.allow_cpu:
        os.environ["FASHION_TEST_ALLOW_CPU"] = "1"
    if "0" not in args.tasks:
        if not NORMALISATION_PATH.is_file():
            raise SystemExit("Missing normalization metadata; include --tasks 0 or complete Task 0 first.")
        load_splits()
    paths = [ROOT / "notebooks" / NOTEBOOKS[int(task)] for task in sorted(set(args.tasks), key=int)]
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"{path.name}:cell{index + 1}", "exec")
    if args.check:
        print("Environment and notebook syntax checked. Task 0 creates or validates preprocessing artifacts when selected. No notebooks executed.")
        return

    print("Running requested tasks in dependency order (0 to 4). Selected tasks regenerate their outputs, including training and model exports where applicable.", flush=True)
    results = ROOT / "results"
    logs = results / "notebook_logs"
    logs.mkdir(parents=True, exist_ok=True)
    state_path = results / "main_notebooks_run_state.json"
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
            artifacts = artifact_paths(path)
            dependencies = dependency_fingerprint(path)
            # Preserve completed classification runs recorded by the earlier runner.
            if "dependencies_sha256" not in previous and path.name in NOTEBOOKS[1:4] and previous.get("status") == "complete" and previous.get("source_sha256") == digest:
                previous["dependencies_sha256"] = dependencies
                previous["artifacts"] = artifacts
                state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            artifacts_present = all((ROOT / name).is_file() for name in artifacts)
            if not args.rerun and previous.get("status") == "complete" and previous.get("source_sha256") == digest and previous.get("dependencies_sha256") == dependencies and artifacts_present:
                print(f"Skipping completed {path.name}; use --rerun to retrain and regenerate outputs.", flush=True)
                continue
            state[path.name] = {"status": "running", "source_sha256": digest, "started": time.time(), "dependencies_sha256": dependencies}
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            kernel = KernelManager(kernel_name="fashion-experiment", kernel_spec_manager=manager)
            try:
                execute_notebook(path, kernel, logs / (path.stem + ".log"))
                for name in artifacts:
                    if not (ROOT / name).is_file():
                        raise FileNotFoundError(ROOT / name)
            except BaseException:
                state[path.name]["status"] = "failed"
                state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
                print(f"Stopped at {path.name}. Completed cells were saved; a failed notebook restarts from the beginning next time. Fix the error and run again.", file=sys.stderr)
                raise
            state[path.name].update(status="complete", finished=time.time(), artifacts=artifacts)
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print("All requested main notebooks completed. Outputs are saved in notebooks/; Task 4, when selected, refreshes retrieval artifacts and the prediction CSV.")


if __name__ == "__main__":
    main()

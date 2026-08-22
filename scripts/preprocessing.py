"""Small shared data and image helpers used by the assignment notebooks."""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "dataset"
OUTPUT_DIR = ROOT / "scripts" / "data"
SPLIT_PATH = OUTPUT_DIR / "splits.csv"
NORMALISATION_PATH = OUTPUT_DIR / "normalization.json"
IMAGE_AUDIT_PATH = OUTPUT_DIR / "image_audit.csv"
TARGETS = ("articleType", "season", "gender", "usage")
SEED = 2753
IMAGE_SIZE = (96, 128)  # width, height
EXPECTED_CSV_SHA256 = {
    "train/styles_train.csv": "54D503743F79F22BF03F9E2C216B53E75D5F2129EDA5934CFA25C642C70AF44D",
    "test/styles_prediction.csv": "7F83F219D0FB7F2EC86B8B5C0BA28C2CFADD8F8E6E85A36C9919C9AB5C41315E",
}


def file_sha256(path: str | Path) -> str:
    """Return an uppercase SHA-256 fingerprint without loading the file at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validate_dataset_release(root: str | Path | None = None) -> None:
    """Fail when local CSVs differ from the frozen split's source release."""
    dataset = data_root(root)
    failures = {}
    for relative, expected in EXPECTED_CSV_SHA256.items():
        path = dataset / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing required dataset file: {path}")
        actual = file_sha256(path)
        if actual != expected:
            failures[relative] = {"expected": expected, "actual": actual}
    if failures:
        raise ValueError(
            "Dataset CSV fingerprints do not match the frozen handoff release: "
            f"{failures}. Obtain the authorized team copy before training."
        )


def data_root(override: str | Path | None = None) -> Path:
    """Return the private supplied dataset location."""
    value = override or os.environ.get("FASHION_DATA_ROOT") or DEFAULT_DATA_ROOT
    return Path(value).expanduser().resolve()


def seed_everything(seed: int = SEED) -> None:
    """Seed Python and NumPy; notebooks seed their ML framework separately."""
    random.seed(seed)
    np.random.seed(seed)


def select_torch_device():
    """Select CPU/CUDA from FASHION_DEVICE and report the active accelerator."""
    import torch

    requested = os.environ.get("FASHION_DEVICE", "auto").strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("FASHION_DEVICE must be one of: auto, cpu, cuda")

    cuda_available = torch.cuda.is_available()
    if requested == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA was requested but this Python environment cannot use it. "
            f"PyTorch={torch.__version__}, CUDA build={torch.version.cuda}. "
            "Install requirements-cuda.txt and restart the Jupyter kernel."
        )

    device = torch.device(
        "cuda" if cuda_available and requested != "cpu" else "cpu"
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(
            f"Using CUDA: {torch.cuda.get_device_name(0)} | "
            f"PyTorch {torch.__version__} | CUDA {torch.version.cuda}"
        )
    else:
        print(
            f"Using CPU | PyTorch {torch.__version__} | "
            f"CUDA build {torch.version.cuda}"
        )
    return device


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"id": "string"}, keep_default_na=False)
    overflow_columns = [c for c in frame.columns if c.startswith("Unnamed")]
    if overflow_columns and "productDisplayName" in frame.columns:
        overflow = frame[overflow_columns].astype("string").apply(
            lambda column: column.str.strip()
        )
        repaired = overflow.ne("").any(axis=1)
        if repaired.any():
            name_parts = frame.loc[
                repaired, ["productDisplayName", *overflow_columns]
            ].astype("string")
            frame.loc[repaired, "productDisplayName"] = name_parts.apply(
                lambda row: ", ".join(
                    value.strip() for value in row if value.strip()
                ),
                axis=1,
            )
        frame["productDisplayName_repaired"] = repaired
    return frame.drop(columns=overflow_columns)


def load_metadata(root: str | Path | None = None) -> pd.DataFrame:
    """Load training metadata and attach expected image paths."""
    dataset = data_root(root)
    validate_dataset_release(dataset)
    frame = _read_csv(dataset / "train" / "styles_train.csv")
    image_dir = dataset / "train" / "images_train"
    frame["image_path"] = frame["id"].map(lambda value: str(image_dir / f"{value}.jpg"))
    frame["has_image"] = frame["image_path"].map(lambda value: Path(value).is_file())
    return frame


def load_prediction_template(root: str | Path | None = None) -> pd.DataFrame:
    """Load the course-test template and attach expected image paths."""
    dataset = data_root(root)
    validate_dataset_release(dataset)
    frame = _read_csv(dataset / "test" / "styles_prediction.csv")
    image_dir = dataset / "test" / "images_test"
    frame["image_path"] = frame["id"].map(lambda value: str(image_dir / f"{value}.jpg"))
    return frame


def remove_transparency(image: Image.Image) -> Image.Image:
    """Flatten transparent/palette images and return RGB."""
    return image.convert("RGBA").convert("RGB") if image.mode != "RGB" else image


def preprocess_image(
    image: Image.Image,
    size: tuple[int, int] = IMAGE_SIZE,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> np.ndarray:
    """Convert an image to a resized channel-first float array."""
    image = remove_transparency(image).resize(size, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    if mean is not None and std is not None:
        array = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(
            std, dtype=np.float32
        )
    return np.transpose(array, (2, 0, 1))


def load_splits(path: str | Path = SPLIT_PATH) -> pd.DataFrame:
    """Load the team-wide frozen split created in Task 0."""
    split_path = Path(path)
    if not split_path.exists():
        raise FileNotFoundError(f"Missing {split_path}; complete Task 0 first")
    return pd.read_csv(split_path, dtype={"id": "string"})


def task_frame(
    target: str,
    split: str | None = None,
    root: str | Path | None = None,
) -> pd.DataFrame:
    """Return valid rows for one target and optional frozen split."""
    if target not in TARGETS:
        raise ValueError(f"Unknown target {target!r}; choose from {TARGETS}")
    frame = load_metadata(root).merge(load_splits(), on="id", validate="one_to_one")
    valid = frame["has_image"] & frame[target].astype("string").str.strip().ne("")
    train_labels = set(frame.loc[valid & frame["split"].eq("train"), target])
    unsupported = sorted(set(frame.loc[valid, target]) - train_labels)
    if unsupported:
        raise ValueError(
            f"Frozen split has {target} labels absent from training: {unsupported}. "
            "Run Task 0 again to repair label coverage before training."
        )
    frame = frame.loc[valid].copy()
    return frame.loc[frame["split"].eq(split)].copy() if split else frame

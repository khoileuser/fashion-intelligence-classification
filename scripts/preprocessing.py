"""Small shared data and image helpers used by the assignment notebooks."""

from __future__ import annotations

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
    return frame.drop(columns=[c for c in frame.columns if c.startswith("Unnamed")])


def load_metadata(root: str | Path | None = None) -> pd.DataFrame:
    """Load training metadata and attach expected image paths."""
    dataset = data_root(root)
    frame = _read_csv(dataset / "train" / "styles_train.csv")
    image_dir = dataset / "train" / "images_train"
    frame["image_path"] = frame["id"].map(lambda value: str(image_dir / f"{value}.jpg"))
    frame["has_image"] = frame["image_path"].map(lambda value: Path(value).is_file())
    return frame


def load_prediction_template(root: str | Path | None = None) -> pd.DataFrame:
    """Load the course-test template and attach expected image paths."""
    dataset = data_root(root)
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

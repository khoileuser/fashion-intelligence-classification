"""Small, transparent validation helpers shared by classifier notebooks."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import accuracy_score, f1_score


def ordered_estimator_probabilities(estimator, features, labels) -> np.ndarray:
    """Return predict_proba columns in the checkpoint label order."""
    estimator_labels = list(estimator.classes_)
    if set(estimator_labels) != set(labels):
        raise ValueError("Estimator classes do not match the requested labels")
    order = [estimator_labels.index(label) for label in labels]
    return estimator.predict_proba(features)[:, order]


def expected_calibration_error(
    truth: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    """Compute top-label expected calibration error."""
    truth = np.asarray(truth)
    probabilities = np.asarray(probabilities)
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == truth
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(truth)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            error += selected.sum() / total * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return float(error)


def select_validation_candidate(
    comparison: pd.DataFrame,
    macro_f1_tolerance: float = 0.01,
) -> str:
    """Select near-best macro-F1 candidates by calibration then complexity."""
    required = {
        "validation_macro_f1",
        "validation_ece",
        "complexity_parameters",
    }
    missing = required - set(comparison.columns)
    if missing:
        raise ValueError(f"Comparison is missing selection columns: {sorted(missing)}")
    threshold = comparison.validation_macro_f1.max() - macro_f1_tolerance
    eligible = comparison.loc[comparison.validation_macro_f1.ge(threshold)]
    return str(
        eligible.sort_values(
            ["validation_ece", "complexity_parameters", "validation_macro_f1"],
            ascending=[True, True, False],
        ).index[0]
    )


def calibration_table(truth: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    """Return a ten-bin reliability table for notebook evidence."""
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == np.asarray(truth)
    frame = pd.DataFrame({"confidence": confidence, "correct": correct})
    frame["bin"] = pd.cut(
        frame.confidence, bins=np.linspace(0, 1, 11), include_lowest=True
    )
    return frame.groupby("bin", observed=False).agg(
        mean_confidence=("confidence", "mean"),
        accuracy=("correct", "mean"),
        samples=("correct", "size"),
    )


def high_confidence_errors(
    frame: pd.DataFrame,
    target: str,
    labels: list[str],
    truth: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    limit: int = 20,
) -> pd.DataFrame:
    """Return report-ready high-confidence mistakes in dataset row order."""
    columns = list(
        dict.fromkeys(
            column
            for column in ("id", "articleType", target, "image_path")
            if column in frame
        )
    )
    result = frame.reset_index(drop=True)[columns].copy()
    result["truth"] = [labels[int(index)] for index in truth]
    result["prediction"] = [labels[int(index)] for index in predictions]
    result["confidence"] = probabilities.max(axis=1)
    return result.loc[predictions != truth].sort_values("confidence", ascending=False).head(limit)


def subgroup_metrics(
    frame: pd.DataFrame,
    truth_labels: list[str],
    prediction_labels: list[str],
    group_column: str = "articleType",
    minimum_support: int = 30,
) -> pd.DataFrame:
    """Measure accuracy and macro F1 for sufficiently supported catalogue groups."""
    values = frame.reset_index(drop=True)[[group_column]].copy()
    values["truth"] = truth_labels
    values["prediction"] = prediction_labels
    rows = []
    for group, part in values.groupby(group_column):
        if len(part) < minimum_support:
            continue
        rows.append(
            {
                group_column: group,
                "support": len(part),
                "accuracy": accuracy_score(part.truth, part.prediction),
                "macro_f1": f1_score(
                    part.truth, part.prediction, average="macro", zero_division=0
                ),
            }
        )
    if not rows:
        return pd.DataFrame(columns=[group_column, "support", "accuracy", "macro_f1"])
    return pd.DataFrame(rows).sort_values(["macro_f1", "support"])


def saved_model_robustness(
    predictor,
    frame: pd.DataFrame,
    target: str,
    seed: int,
    sample_size: int = 300,
) -> pd.DataFrame:
    """Benchmark clean, darker, brighter, and mildly blurred images on CPU."""
    sample = frame.sample(min(sample_size, len(frame)), random_state=seed)
    variants = {
        "clean": lambda image: image,
        "brightness_0.8": lambda image: ImageEnhance.Brightness(image).enhance(0.8),
        "brightness_1.2": lambda image: ImageEnhance.Brightness(image).enhance(1.2),
        "gaussian_blur_0.7": lambda image: image.filter(ImageFilter.GaussianBlur(0.7)),
    }
    rows = []
    for name, transform in variants.items():
        truth, predictions, confidence = [], [], []
        started = time.perf_counter()
        for row in sample.itertuples(index=False):
            with Image.open(row.image_path) as source:
                result = predictor.predict(transform(source.convert("RGB")), top_k=1)
            truth.append(getattr(row, target))
            predictions.append(result["label"])
            confidence.append(result["confidence"])
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "variant": name,
                "samples": len(sample),
                "accuracy": accuracy_score(truth, predictions),
                "macro_f1": f1_score(truth, predictions, average="macro", zero_division=0),
                "mean_confidence": float(np.mean(confidence)),
                "milliseconds_per_image": 1000 * elapsed / len(sample),
            }
        )
    return pd.DataFrame(rows).set_index("variant")

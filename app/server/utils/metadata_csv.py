"""Shared catalogue CSV parsing for training and deployed inference."""
from pathlib import Path
import pandas as pd


def read_metadata_csv(path: Path) -> pd.DataFrame:
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


"""Dataset contract shared by the synthetic sample data and any future real measurements.

To switch to measured data, supply a CSV with these columns (``data_source`` is free text)
and retrain: nothing else in the pipeline changes.
"""
from __future__ import annotations

import pandas as pd

from .folding_config import FoldSequence

CATEGORICAL_COLUMNS = ["fabric_type", "material_type", "compression_method"]
FABRIC_NUMERIC_COLUMNS = [
    "length_cm",
    "width_cm",
    "thickness_mm",
    "gsm",                   # g/m²
    "bulk_factor",
    "compression_factor",
    "recovery_factor",
    "bending_stiffness_nm",  # N·m
    "friction_coeff",        # fabric-to-fabric
    "hem_width_mm",
    "hem_thickness_mm",
    "compression_force_n",   # N
]
SEQUENCE_COLUMN = "fold_sequence"  # JSON list of {direction, orientation, ratio}
TARGET_COLUMNS = ["final_length_cm", "final_width_cm", "final_gusset_mm"]
SOURCE_COLUMN = "data_source"

INPUT_COLUMNS = CATEGORICAL_COLUMNS + FABRIC_NUMERIC_COLUMNS + [SEQUENCE_COLUMN]
REQUIRED_COLUMNS = INPUT_COLUMNS + TARGET_COLUMNS

TARGET_LABELS = {
    "final_length_cm": "Folded length (cm)",
    "final_width_cm": "Folded width (cm)",
    "final_gusset_mm": "Folded gusset / thickness (mm)",
}


def load_dataset(path: str) -> pd.DataFrame:
    """Read a dataset CSV and check it satisfies the schema."""
    df = pd.read_csv(path)
    validate_dataset(df)
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")
    if df[REQUIRED_COLUMNS].isna().any().any():
        bad = df[REQUIRED_COLUMNS].isna().sum()
        raise ValueError(f"dataset has missing values: {bad[bad > 0].to_dict()}")
    for text in df[SEQUENCE_COLUMN]:
        FoldSequence.from_json(text)  # raises on malformed or out-of-range folds

"""Model training: one RandomForestRegressor per output, plus (de)serialisation.

The pipeline is dataset-agnostic: pass any dataframe that satisfies ``schema.py``
(synthetic sample today, measured data later) and the same code retrains the models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from .preprocessing import FeatureBuilder
from .schema import SOURCE_COLUMN, TARGET_COLUMNS, validate_dataset

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
BUNDLE_NAME = "model_bundle.joblib"

RF_PARAMS = dict(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42)


@dataclass
class ModelBundle:
    """Everything needed to predict: the fitted feature builder and one model per target."""

    builder: FeatureBuilder
    models: dict[str, RandomForestRegressor]
    data_sources: list[str] = field(default_factory=list)
    n_train: int = 0
    n_test: int = 0

    @property
    def is_synthetic(self) -> bool:
        return any("SYNTHETIC" in s.upper() for s in self.data_sources)


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame


def make_split(df: pd.DataFrame, builder: FeatureBuilder, test_size: float = 0.2, seed: int = 42) -> Split:
    X = builder.transform(df)
    y = df[TARGET_COLUMNS].reset_index(drop=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
    return Split(X_tr, X_te, y_tr, y_te)


def train_models(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> tuple[ModelBundle, Split]:
    validate_dataset(df)
    builder = FeatureBuilder().fit(df)
    split = make_split(df, builder, test_size, seed)
    models = {}
    for target in TARGET_COLUMNS:  # separate regressor per output
        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(split.X_train, split.y_train[target])
        models[target] = rf
    sources = sorted(df[SOURCE_COLUMN].astype(str).unique()) if SOURCE_COLUMN in df else ["UNKNOWN"]
    bundle = ModelBundle(builder, models, sources, len(split.X_train), len(split.X_test))
    return bundle, split


def save_bundle(bundle: ModelBundle, directory: Path | str = MODELS_DIR) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / BUNDLE_NAME
    joblib.dump(bundle, path, compress=3)
    return path


def load_bundle(directory: Path | str = MODELS_DIR) -> ModelBundle:
    return joblib.load(Path(directory) / BUNDLE_NAME)

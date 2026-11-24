"""Model evaluation: MAE / RMSE / R² on the held-out test split (and train for reference)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .schema import TARGET_COLUMNS
from .training import MODELS_DIR, ModelBundle, Split

METRICS_NAME = "metrics.json"
TEST_PREDICTIONS_NAME = "test_predictions.csv"


def _metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _ideal_baseline(X: pd.DataFrame) -> dict[str, np.ndarray]:
    """Naive no-ML reference: ideal paper-thin geometry, nominal thickness x layer count."""
    return {
        "final_length_cm": X["ideal_length"].to_numpy(),
        "final_width_cm": X["ideal_width"].to_numpy(),
        "final_gusset_mm": (X["ideal_layers"] * X["thickness_mm"]).to_numpy(),
    }


def evaluate(bundle: ModelBundle, split: Split) -> tuple[dict, pd.DataFrame]:
    """Return (metrics dict, test-set predictions dataframe)."""
    baseline = _ideal_baseline(split.X_test)
    per_target, preds = {}, {}
    for t in TARGET_COLUMNS:
        model = bundle.models[t]
        test_pred = model.predict(split.X_test)
        preds[f"{t}__actual"] = split.y_test[t].to_numpy()
        preds[f"{t}__predicted"] = test_pred
        importances = pd.Series(model.feature_importances_, index=split.X_train.columns).nlargest(8)
        per_target[t] = {
            "test": _metrics(split.y_test[t], test_pred),
            "train": _metrics(split.y_train[t], model.predict(split.X_train)),
            "baseline_ideal_geometry_test": _metrics(split.y_test[t], baseline[t]),
            "top_features": {k: float(v) for k, v in importances.items()},
        }
    report = {
        "model": "RandomForestRegressor (one per target)",
        "n_train": bundle.n_train,
        "n_test": bundle.n_test,
        "data_sources": bundle.data_sources,
        "is_synthetic": bundle.is_synthetic,
        "targets": per_target,
    }
    return report, pd.DataFrame(preds)


def save_report(report: dict, test_predictions: pd.DataFrame, directory: Path | str = MODELS_DIR) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / METRICS_NAME).write_text(json.dumps(report, indent=2))
    test_predictions.to_csv(directory / TEST_PREDICTIONS_NAME, index=False)


def load_report(directory: Path | str = MODELS_DIR) -> tuple[dict, pd.DataFrame] | None:
    directory = Path(directory)
    m, p = directory / METRICS_NAME, directory / TEST_PREDICTIONS_NAME
    if not (m.exists() and p.exists()):
        return None
    return json.loads(m.read_text()), pd.read_csv(p)

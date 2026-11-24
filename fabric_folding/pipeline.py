"""End-to-end train + evaluate + persist. Shared by the CLI and the UI's retrain button."""
from __future__ import annotations

from pathlib import Path

from .evaluation import evaluate, save_report
from .sample_dataset import DEFAULT_PATH, write_dataset
from .schema import load_dataset
from .training import MODELS_DIR, save_bundle, train_models


def run_training(data_path: Path | str = DEFAULT_PATH, models_dir: Path | str = MODELS_DIR) -> dict:
    """Train the three regressors on ``data_path`` (generating the sample set if absent)."""
    data_path = Path(data_path)
    if not data_path.exists():
        if data_path.resolve() != DEFAULT_PATH.resolve():
            raise FileNotFoundError(data_path)
        write_dataset(data_path)
    df = load_dataset(str(data_path))
    bundle, split = train_models(df)
    report, test_preds = evaluate(bundle, split)
    save_bundle(bundle, models_dir)
    save_report(report, test_preds, models_dir)
    return report

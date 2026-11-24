"""CLI: (re)generate the synthetic dataset and/or train + evaluate the models.

    python train.py                     # generate sample data if missing, then train
    python train.py --regenerate        # regenerate the synthetic dataset first
    python train.py --data my_measured.csv   # retrain on real data (same schema)
"""
import argparse
from pathlib import Path

from fabric_folding.pipeline import run_training
from fabric_folding.sample_dataset import DEFAULT_PATH, write_dataset
from fabric_folding.schema import TARGET_LABELS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_PATH, help="dataset CSV (see fabric_folding/schema.py)")
    ap.add_argument("--regenerate", action="store_true", help="regenerate the synthetic sample dataset")
    ap.add_argument("--records", type=int, default=8000, help="records when generating the sample dataset")
    args = ap.parse_args()

    if args.regenerate:
        write_dataset(DEFAULT_PATH, args.records)
        print(f"Wrote synthetic sample dataset -> {DEFAULT_PATH}")

    report = run_training(args.data)
    print(f"\nTrained on {report['n_train']} rows, tested on {report['n_test']} rows "
          f"(sources: {', '.join(report['data_sources'])})")
    if report["is_synthetic"]:
        print("NOTE: synthetic data - results are not physically validated.\n")
    print(f"{'target':32s} {'MAE':>8s} {'RMSE':>8s} {'R2':>8s}   baseline MAE")
    for t, m in report["targets"].items():
        te = m["test"]
        print(f"{TARGET_LABELS[t]:32s} {te['mae']:8.3f} {te['rmse']:8.3f} {te['r2']:8.4f}   "
              f"{m['baseline_ideal_geometry_test']['mae']:.3f}")


if __name__ == "__main__":
    main()

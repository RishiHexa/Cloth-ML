# Fabric Folding & Dimension Predictor — Part A prototype

ML prototype for **Part A – Fabric Folding, Packing and Gusset** of the Packing 3D Simulator
(bedsheets and terry towels). Part B (carton planning) is out of scope.

> **All data is SYNTHETIC.** The sample dataset is generated from a simple hand-written rule set plus
> noise. Predictions are not physically validated and must be calibrated against real fold/pack
> measurements before use.

## Run

```bash
pip install -r requirements.txt
python train.py --regenerate      # generate synthetic dataset + train + evaluate (~20 s)
streamlit run app.py              # UI (auto-trains on first launch if no models exist)
python -m pytest                  # tests
```

## How it works

1. **Dataset** (`data/sample_fold_dataset.csv`, 8000 rows): fabric type, material, length, width, thickness,
   GSM, bulk, compression, recovery, bending stiffness, fabric-to-fabric friction, hem width and thickness,
   compression force and method, the fold sequence (JSON list of `{direction, orientation, ratio}`, 1–6 folds),
   and the targets `final_length_cm`, `final_width_cm`, `final_gusset_mm`. Every row is tagged
   `data_source = SYNTHETIC_SAMPLE`.
2. **Preprocessing** (`preprocessing.py`): one `FeatureBuilder`, fitted at training time and saved with the
   models, used unchanged at prediction time. Features = fabric properties (+ one-hot material), fixed
   per-step fold slots that preserve fold order, and ideal-geometry descriptors of the sequence (pure
   geometry, no material effects).
3. **Training** (`training.py`): 80/20 split, a separate `RandomForestRegressor` for each of the 3 outputs.
4. **Evaluation** (`evaluation.py`): MAE, RMSE, R² on the test split (plus train and a no-ML "ideal geometry"
   baseline), saved to `models/metrics.json`; shown in the UI separately from individual predictions.
5. **Prediction** (`prediction.py`): `PredictionService.predict(FabricSpec, FoldSequence)` → length, width,
   gusset, plus out-of-training-range warnings.
6. **Visualization** (`visualization.py`): 2D top-view stages (original → each fold → final) shaded by layer
   count; ideal geometry of the *selected* sequence with the ML prediction drawn over the final stage.
   It does not import the ML code and is not a cloth physics simulation.

## Fabric parameters

Model inputs: thickness, GSM, bulk, compression factor, recovery factor, bending stiffness (N·m),
fabric-to-fabric friction, hem width and thickness (mm), compression force (N) and method.
Sheet mass (g) is *derived* (GSM × area), shown with the result and not fed to the model.
Tensile stiffness is deliberately not used (folding/packing barely stretches the fabric). The
compression/recovery curve is summarised by the compression and recovery factors; if you later
measure full curves, reduce them to those two numbers (or add columns in `schema.py`).

## Fold conventions

- `direction`: `lengthwise` = fold line along the length (reduces width); `widthwise` = fold line along the
  width (reduces length).
- `orientation`: `start` = left/top edge is lifted over; `end` = right/bottom edge.
- `ratio`: fold-line position from that edge, as a fraction of the current dimension (0.10–0.90).

## Replacing the synthetic data with measurements

Provide a CSV with the columns in `fabric_folding/schema.py` (`data_source` = e.g. `MEASURED_2026_Q1`) and run

```bash
python train.py --data measured.csv
```

No other code changes. The UI hides the "synthetic" disclaimer automatically when no data source contains
the word `SYNTHETIC`. New materials can be added in `fabric_folding/materials.py` (UI defaults only; the
model learns materials from the data).

## Layout

```
fabric_folding/
  schema.py            dataset contract          sample_dataset.py   synthetic generator (ground-truth rules live ONLY here)
  materials.py         material library          preprocessing.py    shared feature preparation
  folding_config.py    fold ops + ideal geometry training.py         RF models + save/load
  evaluation.py        metrics                   prediction.py       prediction service
  visualization.py     2D stages                 pipeline.py         train+evaluate+persist
app.py                 Streamlit UI              train.py            CLI
```

## Known limits

- Folds are axis-aligned and rectangular; no diagonal folds, rolls, hems, fitted-sheet pockets or stiffeners.
- Model quality is only as good as the dataset: current metrics describe agreement with synthetic data.
- Max 6 folds per sequence; inputs outside the training ranges trigger an extrapolation warning.

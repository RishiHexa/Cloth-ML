"""Feature preparation shared by training and prediction.

``FeatureBuilder`` is fitted once on the training dataframe (it learns the categorical
levels and the training ranges) and is then saved together with the models, so the exact
same transformation is applied at prediction time.

Feature groups
--------------
1. Fabric properties: numeric columns + one-hot fabric_type / material_type.
2. Folding configuration: ``fold_count`` plus a fixed number of per-step slots
   (active, axis, orientation, ratio) preserving the *order* of operations; unused
   slots are padded with zeros.
3. Ideal-geometry descriptors derived from the folding sequence (footprint with no
   material effects, layer count, stack heights at fold time). These are pure geometry,
   not material behaviour, so the model still has to learn material effects itself.
"""
from __future__ import annotations

import pandas as pd

from .folding_config import MAX_FOLDS, FoldSequence, ideal_summary
from .schema import (
    CATEGORICAL_COLUMNS, FABRIC_NUMERIC_COLUMNS, SEQUENCE_COLUMN,
)

_GEOMETRY_KEYS = ["ideal_length", "ideal_width", "ideal_layers", "stack_at_length_folds", "stack_at_width_folds"]


class FeatureBuilder:
    def __init__(self, max_folds: int = MAX_FOLDS):
        self.max_folds = max_folds
        self.categories_: dict[str, list[str]] = {}
        self.feature_names_: list[str] = []
        self.numeric_ranges_: dict[str, tuple[float, float]] = {}

    # ------------------------------------------------------------------ fitting
    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        self.categories_ = {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL_COLUMNS}
        self.numeric_ranges_ = {c: (float(df[c].min()), float(df[c].max())) for c in FABRIC_NUMERIC_COLUMNS}
        self.numeric_ranges_["fold_count"] = (
            1.0, float(df[SEQUENCE_COLUMN].map(lambda t: len(FoldSequence.from_json(t))).max())
        )
        self.feature_names_ = list(self.transform(df.head(1)).columns)
        return self

    # ------------------------------------------------------------- transforming
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = [self._row(r) for r in df.to_dict("records")]
        out = pd.DataFrame(rows)
        if self.feature_names_:
            out = out.reindex(columns=self.feature_names_, fill_value=0.0)
        return out

    def _row(self, rec: dict) -> dict:
        feats: dict[str, float] = {c: float(rec[c]) for c in FABRIC_NUMERIC_COLUMNS}
        for col in CATEGORICAL_COLUMNS:
            for level in self.categories_.get(col, []):
                feats[f"{col}={level}"] = 1.0 if rec[col] == level else 0.0

        seq = rec[SEQUENCE_COLUMN]
        seq = FoldSequence.from_json(seq) if isinstance(seq, str) else seq
        seq.validate(self.max_folds)
        feats["fold_count"] = float(len(seq))
        feats["lengthwise_folds"] = float(sum(o.direction == "lengthwise" for o in seq))
        feats["widthwise_folds"] = float(sum(o.direction == "widthwise" for o in seq))
        for i in range(self.max_folds):
            op = seq.operations[i] if i < len(seq) else None
            feats[f"fold{i + 1}_active"] = 1.0 if op else 0.0
            feats[f"fold{i + 1}_lengthwise"] = 1.0 if op and op.direction == "lengthwise" else 0.0
            feats[f"fold{i + 1}_from_end"] = 1.0 if op and op.orientation == "end" else 0.0
            feats[f"fold{i + 1}_ratio"] = op.ratio if op else 0.0

        feats.update(ideal_summary(float(rec["length_cm"]), float(rec["width_cm"]), seq))
        return feats

    # -------------------------------------------------------------- diagnostics
    def out_of_range_warnings(self, rec: dict) -> list[str]:
        """Human-readable notes for inputs outside what the model was trained on."""
        notes = []
        for col, (lo, hi) in self.numeric_ranges_.items():
            if col == "fold_count":
                continue
            v = float(rec[col])
            if v < lo or v > hi:
                notes.append(f"{col} = {v:g} is outside the training range [{lo:g}, {hi:g}]")
        for col in CATEGORICAL_COLUMNS:
            if rec[col] not in self.categories_.get(col, []):
                notes.append(f"{col} = {rec[col]!r} was not present in the training data")
        return notes

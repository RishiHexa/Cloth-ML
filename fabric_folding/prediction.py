"""Prediction service: fabric spec + fold sequence -> predicted folded dimensions.

Knows nothing about visualization or the UI. Uses the same ``FeatureBuilder`` that was
fitted during training.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .folding_config import FoldSequence
from .schema import SEQUENCE_COLUMN, TARGET_COLUMNS
from .training import MODELS_DIR, ModelBundle, load_bundle

SYNTHETIC_DISCLAIMER = (
    "Predictions come from models trained on a SYNTHETIC sample dataset. They are not "
    "physically validated and must be calibrated against actual fold/pack measurements."
)


@dataclass
class FabricSpec:
    fabric_type: str
    material_type: str
    length_cm: float
    width_cm: float
    thickness_mm: float
    gsm: float
    bulk_factor: float
    compression_factor: float
    recovery_factor: float
    bending_stiffness_nm: float
    friction_coeff: float
    hem_width_mm: float
    hem_thickness_mm: float
    compression_force_n: float
    compression_method: str

    def to_record(self) -> dict:
        return dict(self.__dict__)

    @property
    def sheet_mass_g(self) -> float:
        """Derived (not a model input): GSM × area. Handy as the weight handoff to Part B."""
        return self.gsm * self.length_cm * self.width_cm / 1e4


@dataclass
class Prediction:
    folded_length_cm: float
    folded_width_cm: float
    gusset_mm: float
    warnings: list[str] = field(default_factory=list)
    is_synthetic: bool = True
    disclaimer: str = SYNTHETIC_DISCLAIMER


class PredictionService:
    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle

    @classmethod
    def from_disk(cls, directory=MODELS_DIR) -> "PredictionService":
        return cls(load_bundle(directory))

    def predict(self, fabric: FabricSpec, sequence: FoldSequence) -> Prediction:
        if len(sequence) == 0:
            raise ValueError("Add at least one fold operation before predicting.")
        sequence.validate(self.bundle.builder.max_folds)
        record = {**fabric.to_record(), SEQUENCE_COLUMN: sequence}
        features = self.bundle.builder.transform(pd.DataFrame([record]))  # same path as training
        out = {t: float(self.bundle.models[t].predict(features)[0]) for t in TARGET_COLUMNS}
        warnings = self.bundle.builder.out_of_range_warnings(fabric.to_record())
        return Prediction(
            folded_length_cm=out["final_length_cm"],
            folded_width_cm=out["final_width_cm"],
            gusset_mm=out["final_gusset_mm"],
            warnings=warnings,
            is_synthetic=self.bundle.is_synthetic,
            disclaimer=SYNTHETIC_DISCLAIMER if self.bundle.is_synthetic else "",
        )

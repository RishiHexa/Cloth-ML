"""SYNTHETIC sample dataset generator.

!! The records produced here are artificial. They come from a simple, hand-written
!! material/geometry rule set plus random noise - NOT from physical measurements. Model
!! predictions trained on them are not physically validated and must be calibrated against
!! real folding/packing measurements before being relied upon.

The rule set below exists only to give the ML model structure to learn (material
thickness, bulk, compression/recovery, fold count, fold order, fold-edge roll). It is
deliberately kept out of the ML pipeline: training and prediction only ever see the
dataset columns defined in ``schema.py``.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .folding_config import (
    DIRECTIONS, MAX_FOLDS, MAX_RATIO, MIN_RATIO, ORIENTATIONS,
    FoldOperation, FoldSequence, simulate_ideal,
)
from .materials import COMPRESSION_METHODS, MATERIALS, MaterialProfile
from .schema import REQUIRED_COLUMNS, SOURCE_COLUMN

DATA_SOURCE_LABEL = "SYNTHETIC_SAMPLE"
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_fold_dataset.csv"

_COMMON_RATIOS = (0.25, 1 / 3, 0.5, 2 / 3, 0.75)
_REF_FORCE_N = 80.0      # force at which compression_factor is considered measured
_REF_BENDING_NM = 1e-4


# ----------------------------------------------------------------------- ground truth
def _true_dimensions(
    mat_row: dict, seq: FoldSequence, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Artificial 'measurement' for one record (synthetic rule set + noise)."""
    stages = simulate_ideal(mat_row["length_cm"], mat_row["width_cm"], seq)
    final = stages[-1]
    n_layers = final.max_layers

    rec = mat_row["recovery_factor"]
    # The measured compression factor refers to the reference force; a harder press
    # squeezes further (factor ** F/F_ref). Manual pressing is less effective than a flat press.
    force = mat_row["compression_force_n"] * (0.85 if mat_row["compression_method"] == "manual_press" else 1.0)
    comp = min(max(mat_row["compression_factor"] ** (force / _REF_FORCE_N), 0.15), 0.98)
    # Thickness of one layer after the pack is released: compressed value plus the
    # share of the lost thickness that springs back.
    t_layer = mat_row["thickness_mm"] * mat_row["bulk_factor"] * (comp + rec * (1 - comp))

    # Tall stacks compress each other (poor recovery and heavy fabric make that worse).
    stack_factor = 1 - 0.12 * (1 - rec) * (1 - math.exp(-n_layers / 20))
    stack_factor -= min(0.1, 2e-6 * mat_row["gsm"] * n_layers)
    # Stiffer fabric rolls more at each fold edge.
    stiffness = 1 + 0.5 * mat_row["bending_stiffness_nm"] / _REF_BENDING_NM
    gusset = t_layer * n_layers * stack_factor + 0.3 * t_layer * len(seq) * stiffness
    # Hems are thicker than the body fabric and stack up along the folded edges.
    gusset += max(mat_row["hem_thickness_mm"] - mat_row["thickness_mm"], 0) * min(n_layers, 6) * 0.5

    # Each fold rolls the edge slightly, widening the folded axis. The roll grows with the
    # stack height at the moment of folding, so fold *order* matters.
    roll = lambda s: 0.12 * t_layer * math.sqrt(s.layers_before) * stiffness
    roll_len = sum(roll(s) for s in stages[1:] if s.operation.axis == "length")
    roll_wid = sum(roll(s) for s in stages[1:] if s.operation.axis == "width")

    relax = 1 + 0.012 * (mat_row["bulk_factor"] - 1) * rec  # lateral relaxation of loft
    # Low friction lets layers slip and spread; more folds, more spread.
    relax += 0.02 * max(0.7 - mat_row["friction_coeff"], 0) * len(seq) / MAX_FOLDS
    length = (final.length + roll_len) * relax
    width = (final.width + roll_wid) * relax

    length *= 1 + rng.normal(0, 0.006)
    width *= 1 + rng.normal(0, 0.006)
    gusset *= 1 + rng.normal(0, 0.02)
    return length, width, gusset


# --------------------------------------------------------------------------- sampling
def _random_sequence(rng: np.random.Generator) -> FoldSequence:
    n = int(rng.choice(np.arange(1, MAX_FOLDS + 1), p=[0.12, 0.22, 0.22, 0.18, 0.14, 0.12]))
    ops = []
    for _ in range(n):
        if rng.random() < 0.6:
            ratio = float(rng.choice(_COMMON_RATIOS))
        else:
            ratio = float(rng.uniform(0.15, 0.85))
        ratio = min(max(ratio, MIN_RATIO), MAX_RATIO)
        ops.append(FoldOperation(str(rng.choice(DIRECTIONS)), str(rng.choice(ORIENTATIONS)), round(ratio, 3)))
    return FoldSequence(ops)


def _random_fabric(profile: MaterialProfile, rng: np.random.Generator) -> dict:
    u = lambda r: float(rng.uniform(*r))
    # Heavier / thicker fabrics tend to have thicker hems and higher GSM: tie them to the
    # position of the sampled thickness inside its range (plus jitter) so samples are realistic.
    lo, hi = profile.thickness_mm
    thickness = u(profile.thickness_mm)
    frac = (thickness - lo) / (hi - lo)

    def correlated(r):
        f = min(max(frac + rng.normal(0, 0.15), 0.0), 1.0)
        return r[0] + f * (r[1] - r[0])

    return {
        "fabric_type": profile.fabric_type,
        "material_type": profile.name,
        "length_cm": round(u(profile.length_cm), 1),
        "width_cm": round(u(profile.width_cm), 1),
        "thickness_mm": round(thickness, 3),
        "gsm": round(correlated(profile.gsm), 1),
        "bulk_factor": round(u(profile.bulk_factor), 3),
        "compression_factor": round(u(profile.compression_factor), 3),
        "recovery_factor": round(u(profile.recovery_factor), 3),
        "bending_stiffness_nm": float(f"{correlated(profile.bending_stiffness_nm):.3g}"),
        "friction_coeff": round(u(profile.friction_coeff), 3),
        "hem_width_mm": round(u(profile.hem_width_mm), 1),
        "hem_thickness_mm": round(correlated(profile.hem_thickness_mm), 3),
        "compression_force_n": round(u(profile.compression_force_n), 1),
        "compression_method": str(rng.choice(COMPRESSION_METHODS)),
    }


def generate_dataset(n_records: int = 8000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    profiles = list(MATERIALS.values())
    rows = []
    for i in range(n_records):
        fabric = _random_fabric(profiles[i % len(profiles)], rng)  # balanced across materials
        seq = _random_sequence(rng)
        length, width, gusset = _true_dimensions(fabric, seq, rng)
        rows.append(
            {
                **fabric,
                "fold_sequence": seq.to_json(),
                "final_length_cm": round(length, 2),
                "final_width_cm": round(width, 2),
                "final_gusset_mm": round(gusset, 2),
                SOURCE_COLUMN: DATA_SOURCE_LABEL,
            }
        )
    df = pd.DataFrame(rows)
    return df[REQUIRED_COLUMNS + [SOURCE_COLUMN]].sample(frac=1, random_state=seed).reset_index(drop=True)


def write_dataset(path: Path | str = DEFAULT_PATH, n_records: int = 8000, seed: int = 42) -> pd.DataFrame:
    df = generate_dataset(n_records, seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df

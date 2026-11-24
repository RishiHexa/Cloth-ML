"""Material library: default fabric property ranges per material.

Used (a) by the synthetic dataset generator to sample plausible fabrics and (b) by the
UI to pre-fill sensible defaults. All values are illustrative, not measured.

Units: length/width in cm; thickness and hem dimensions in mm; GSM in g/m²; bending
stiffness in N·m; friction is a dimensionless coefficient; compression force in N.
  * bulk_factor        - multiplier on nominal thickness for loft / air content (>= 1).
  * compression_factor - fraction of thickness that remains while the pack is compressed
                         (0.5 = squeezed to half thickness), measured at the reference force.
  * recovery_factor    - fraction of the lost thickness that springs back after release
                         (1.0 = full recovery).
  * compression_force_n / compression_method - how the pack is actually compressed.
"""
from __future__ import annotations

from dataclasses import dataclass

FABRIC_TYPES = ("bedsheet", "terry_towel")
COMPRESSION_METHODS = ("manual_press", "flat_press")

# Numeric properties sampled per material, in dataset-column order.
RANGE_FIELDS = (
    "length_cm", "width_cm", "thickness_mm", "gsm", "bulk_factor", "compression_factor",
    "recovery_factor", "bending_stiffness_nm", "friction_coeff", "hem_width_mm",
    "hem_thickness_mm", "compression_force_n",
)


@dataclass(frozen=True)
class MaterialProfile:
    name: str
    fabric_type: str
    length_cm: tuple[float, float]
    width_cm: tuple[float, float]
    thickness_mm: tuple[float, float]
    gsm: tuple[float, float]
    bulk_factor: tuple[float, float]
    compression_factor: tuple[float, float]
    recovery_factor: tuple[float, float]
    bending_stiffness_nm: tuple[float, float]
    friction_coeff: tuple[float, float]
    hem_width_mm: tuple[float, float]
    hem_thickness_mm: tuple[float, float]
    compression_force_n: tuple[float, float]

    def ranges(self) -> dict[str, tuple[float, float]]:
        return {k: getattr(self, k) for k in RANGE_FIELDS}

    def defaults(self) -> dict[str, float]:
        return {k: round((lo + hi) / 2, 8) for k, (lo, hi) in self.ranges().items()}


def _p(name, ftype, L, W, t, gsm, bulk, comp, rec, bend, fric, hw, ht, force):
    return MaterialProfile(name, ftype, L, W, t, gsm, bulk, comp, rec, bend, fric, hw, ht, force)


MATERIALS: dict[str, MaterialProfile] = {
    m.name: m
    for m in (
        #         name                  type          L          W          thick        gsm         bulk         comp         rec          bending (N·m)      friction     hem w      hem t       force N
        _p("cotton_percale",        "bedsheet",   (200, 280), (150, 270), (0.15, 0.28), (110, 180), (1.00, 1.20), (0.60, 0.85), (0.85, 0.97), (5e-6, 4e-5),   (0.25, 0.45), (20, 80), (0.35, 0.80), (30, 120)),
        _p("cotton_sateen",         "bedsheet",   (200, 280), (150, 270), (0.20, 0.34), (120, 200), (1.00, 1.25), (0.55, 0.80), (0.82, 0.95), (6e-6, 5e-5),   (0.20, 0.40), (20, 80), (0.45, 0.95), (30, 120)),
        _p("microfiber_polyester",  "bedsheet",   (200, 280), (150, 270), (0.12, 0.24), (70, 120),  (1.05, 1.30), (0.65, 0.90), (0.88, 0.99), (3e-6, 2.5e-5), (0.30, 0.55), (20, 80), (0.30, 0.65), (30, 120)),
        _p("terry_cotton",          "terry_towel",(70, 180),  (40, 100),  (1.8, 3.2),   (400, 700), (1.30, 1.90), (0.35, 0.60), (0.60, 0.82), (8e-5, 4e-4),   (0.40, 0.70), (10, 30), (4.0, 8.0),   (50, 200)),
        _p("terry_bamboo_blend",    "terry_towel",(70, 180),  (40, 100),  (1.6, 2.8),   (350, 600), (1.25, 1.80), (0.38, 0.62), (0.55, 0.78), (6e-5, 3e-4),   (0.35, 0.65), (10, 30), (3.5, 7.0),   (50, 200)),
        _p("terry_microcotton",     "terry_towel",(70, 180),  (40, 100),  (1.4, 2.5),   (330, 550), (1.20, 1.70), (0.40, 0.65), (0.65, 0.86), (5e-5, 2.5e-4), (0.40, 0.70), (10, 30), (3.0, 6.0),   (50, 200)),
    )
}


def materials_for(fabric_type: str) -> list[str]:
    return [m.name for m in MATERIALS.values() if m.fabric_type == fabric_type]

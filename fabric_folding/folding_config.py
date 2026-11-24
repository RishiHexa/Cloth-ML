"""Folding configuration: an ordered sequence of fold operations plus ideal 2D geometry.

Conventions
-----------
* The fabric is laid flat. ``length`` runs along the x axis (left -> right) and
  ``width`` along the y axis (top -> bottom).
* ``direction`` says along which axis the fold line runs:
    - ``lengthwise``: fold line runs along the length, so the *width* is reduced.
    - ``widthwise``:  fold line runs along the width, so the *length* is reduced.
* ``orientation`` says which edge is lifted and folded over the rest:
    - ``start``: left edge (widthwise fold) or top edge (lengthwise fold).
    - ``end``:   right edge (widthwise fold) or bottom edge (lengthwise fold).
* ``ratio`` is the fold-line position measured from the folded edge, as a fraction of
  the *current* dimension along the folded axis (0.5 = fold in half).

The geometry here is ideal (zero thickness, no springback). It is used for the
visualization and for geometric features; the ML model predicts the real-world
dimensions that include material effects.

Because every fold is axis aligned, the number of fabric layers at a point (x, y) is
the product ``a(x) * b(y)`` of a per-axis layer profile, which is what ``AxisProfile``
tracks.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

MAX_FOLDS = 6
MIN_RATIO = 0.10
MAX_RATIO = 0.90

DIRECTIONS = ("lengthwise", "widthwise")
ORIENTATIONS = ("start", "end")

DIRECTION_HELP = {
    "lengthwise": "Fold line runs along the length (reduces width)",
    "widthwise": "Fold line runs along the width (reduces length)",
}
ORIENTATION_HELP = {
    "start": "Lift the start edge (left / top) over the fabric",
    "end": "Lift the end edge (right / bottom) over the fabric",
}


@dataclass(frozen=True)
class FoldOperation:
    direction: str = "widthwise"
    orientation: str = "start"
    ratio: float = 0.5

    def validate(self) -> None:
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {self.direction!r}")
        if self.orientation not in ORIENTATIONS:
            raise ValueError(f"orientation must be one of {ORIENTATIONS}, got {self.orientation!r}")
        if not (MIN_RATIO <= self.ratio <= MAX_RATIO):
            raise ValueError(f"ratio must be in [{MIN_RATIO}, {MAX_RATIO}], got {self.ratio}")

    @property
    def axis(self) -> str:
        """Axis whose dimension is reduced: 'length' or 'width'."""
        return "width" if self.direction == "lengthwise" else "length"


@dataclass
class FoldSequence:
    """Ordered list of fold operations (first element is performed first)."""

    operations: list[FoldOperation] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.operations)

    def __iter__(self):
        return iter(self.operations)

    def validate(self, max_folds: int = MAX_FOLDS) -> None:
        if len(self.operations) > max_folds:
            raise ValueError(f"at most {max_folds} folds are supported, got {len(self.operations)}")
        for op in self.operations:
            op.validate()

    def to_json(self) -> str:
        return json.dumps(
            [
                {"direction": o.direction, "orientation": o.orientation, "ratio": round(o.ratio, 4)}
                for o in self.operations
            ]
        )

    @classmethod
    def from_json(cls, text: str) -> "FoldSequence":
        return cls.from_records(json.loads(text))

    @classmethod
    def from_records(cls, records: Iterable[dict]) -> "FoldSequence":
        ops = [FoldOperation(r["direction"], r["orientation"], float(r["ratio"])) for r in records]
        seq = cls(ops)
        seq.validate()
        return seq

    def to_records(self) -> list[dict]:
        return [asdict(o) for o in self.operations]


# --------------------------------------------------------------------------- geometry
class AxisProfile:
    """Piecewise-constant layer count along one axis: list of (x0, x1, layers)."""

    def __init__(self, extent: float):
        self.segments: list[tuple[float, float, int]] = [(0.0, float(extent), 1)]

    @property
    def extent(self) -> float:
        return self.segments[-1][1] - self.segments[0][0]

    @property
    def max_layers(self) -> int:
        return max(n for _, _, n in self.segments)

    def fold(self, ratio: float, orientation: str) -> None:
        lo = self.segments[0][0]
        hi = self.segments[-1][1]
        length = hi - lo
        if orientation == "start":
            p = lo + ratio * length
            kept = [(a, b, n) for a, b, n in self._clip(p, hi)]
            flap = [(2 * p - b, 2 * p - a, n) for a, b, n in self._clip(lo, p)]
        else:
            p = hi - ratio * length
            kept = [(a, b, n) for a, b, n in self._clip(lo, p)]
            flap = [(2 * p - b, 2 * p - a, n) for a, b, n in self._clip(p, hi)]
        merged = self._merge(kept + flap)
        shift = merged[0][0]
        self.segments = [(a - shift, b - shift, n) for a, b, n in merged]

    def _clip(self, lo: float, hi: float):
        for a, b, n in self.segments:
            a2, b2 = max(a, lo), min(b, hi)
            if b2 - a2 > 1e-9:
                yield (a2, b2, n)

    @staticmethod
    def _merge(parts: Sequence[tuple[float, float, int]]):
        cuts = sorted({x for a, b, _ in parts for x in (a, b)})
        out: list[tuple[float, float, int]] = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            mid = (a + b) / 2
            n = sum(k for s, e, k in parts if s <= mid <= e)
            if n == 0 or b - a < 1e-9:
                continue
            if out and out[-1][2] == n and abs(out[-1][1] - a) < 1e-9:
                out[-1] = (out[-1][0], b, n)
            else:
                out.append((a, b, n))
        return out

    def copy(self) -> "AxisProfile":
        c = AxisProfile(0.0)
        c.segments = list(self.segments)
        return c


@dataclass
class FoldStage:
    """State of the ideal fabric after ``step`` folds (step 0 = unfolded)."""

    step: int
    operation: FoldOperation | None  # the fold that produced this stage
    length: float
    width: float
    max_layers: int
    x_profile: list[tuple[float, float, int]]
    y_profile: list[tuple[float, float, int]]
    # Stack height (layers) immediately before this fold was made; 1 for the unfolded stage.
    layers_before: int = 1


def simulate_ideal(length: float, width: float, sequence: FoldSequence) -> list[FoldStage]:
    """Return the ideal geometry at every stage (unfolded + one per fold)."""
    sequence.validate()
    x = AxisProfile(length)
    y = AxisProfile(width)
    stages = [FoldStage(0, None, x.extent, y.extent, 1, list(x.segments), list(y.segments), 1)]
    for i, op in enumerate(sequence, start=1):
        before = x.max_layers * y.max_layers
        (x if op.axis == "length" else y).fold(op.ratio, op.orientation)
        stages.append(
            FoldStage(
                i, op, x.extent, y.extent, x.max_layers * y.max_layers,
                list(x.segments), list(y.segments), before,
            )
        )
    return stages


def ideal_summary(length: float, width: float, sequence: FoldSequence) -> dict:
    """Aggregate ideal-geometry descriptors used as ML features and as a baseline."""
    stages = simulate_ideal(length, width, sequence)
    final = stages[-1]
    stack_len = sum(s.layers_before for s in stages[1:] if s.operation.axis == "length")
    stack_wid = sum(s.layers_before for s in stages[1:] if s.operation.axis == "width")
    return {
        "ideal_length": final.length,
        "ideal_width": final.width,
        "ideal_layers": final.max_layers,
        "stack_at_length_folds": stack_len,
        "stack_at_width_folds": stack_wid,
    }

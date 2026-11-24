"""2D visualization of the user's fold sequence (top view, shaded by layer count).

Deliberately independent of the ML code: it draws the *ideal* geometry of the selected
sequence and, if given, annotates the final stage with predicted numbers passed in as
plain floats.
"""
from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")  # headless-safe; Streamlit renders the returned Figure
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .folding_config import FoldSequence, FoldStage, simulate_ideal

_FLAP_COLOR = "#d9480f"
_PRED_COLOR = "#1864ab"


def _fold_line_and_flap(stage: FoldStage, nxt: FoldStage):
    """Fold-line position and flap rectangle (x0, y0, x1, y1) for the fold following ``stage``."""
    op = nxt.operation
    if op.axis == "length":
        ext = stage.length
        p = ext * op.ratio if op.orientation == "start" else ext * (1 - op.ratio)
        flap = (0, 0, p, stage.width) if op.orientation == "start" else (p, 0, ext, stage.width)
        return "v", p, flap
    ext = stage.width
    p = ext * op.ratio if op.orientation == "start" else ext * (1 - op.ratio)
    flap = (0, 0, stage.length, p) if op.orientation == "start" else (0, p, stage.length, ext)
    return "h", p, flap


def _draw_stage(ax, stage: FoldStage, max_layers: int, canvas: tuple[float, float], nxt: FoldStage | None):
    cmap = plt.get_cmap("Blues")
    for xa, xb, nx in stage.x_profile:
        for ya, yb, ny in stage.y_profile:
            shade = 0.15 + 0.75 * (nx * ny - 1) / max(max_layers - 1, 1)
            ax.add_patch(Rectangle((xa, ya), xb - xa, yb - ya, facecolor=cmap(shade), edgecolor="#555", lw=0.6))

    if nxt is not None:  # preview the upcoming fold
        kind, p, (fx0, fy0, fx1, fy1) = _fold_line_and_flap(stage, nxt)
        ax.add_patch(Rectangle((fx0, fy0), fx1 - fx0, fy1 - fy0, fill=False, hatch="//", edgecolor=_FLAP_COLOR, lw=0))
        if kind == "v":
            ax.plot([p, p], [0, stage.width], "--", color=_FLAP_COLOR, lw=1.6)
            mid = (fx0 + fx1) / 2
            ax.annotate("", xy=(2 * p - mid, stage.width / 2), xytext=(mid, stage.width / 2),
                        arrowprops=dict(arrowstyle="->", color=_FLAP_COLOR, lw=2, connectionstyle="arc3,rad=-0.4"))
        else:
            ax.plot([0, stage.length], [p, p], "--", color=_FLAP_COLOR, lw=1.6)
            mid = (fy0 + fy1) / 2
            ax.annotate("", xy=(stage.length / 2, 2 * p - mid), xytext=(stage.length / 2, mid),
                        arrowprops=dict(arrowstyle="->", color=_FLAP_COLOR, lw=2, connectionstyle="arc3,rad=0.4"))

    ax.set_xlim(-0.03 * canvas[0], canvas[0] * 1.03)
    ax.set_ylim(canvas[1] * 1.03, -0.03 * canvas[1])  # y grows downward: start edge = top
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_folding_stages(
    length_cm: float,
    width_cm: float,
    sequence: FoldSequence,
    predicted: dict | None = None,
) -> Figure:
    """Draw original fabric, every fold and the final folded component.

    ``predicted`` (optional): {"length_cm", "width_cm", "gusset_mm"} drawn as an outline
    over the final stage so it can be compared with the ideal footprint.
    """
    stages = simulate_ideal(length_cm, width_cm, sequence)
    max_layers = max(s.max_layers for s in stages)
    n = len(stages)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.3 * nrows), squeeze=False)

    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        s = stages[i]
        nxt = stages[i + 1] if i + 1 < n else None
        is_final = nxt is None and i > 0
        canvas = (length_cm, width_cm)
        if is_final:  # zoom the final component so it is legible (own scale)
            canvas = (max(s.length, predicted["length_cm"] if predicted else 0),
                      max(s.width, predicted["width_cm"] if predicted else 0))
        _draw_stage(ax, s, max_layers, canvas, nxt)

        if i == 0:
            title = f"Original\n{s.length:.1f} × {s.width:.1f} cm"
        else:
            op = s.operation
            title = (f"Fold {i}: {op.direction}, {op.orientation} edge, {op.ratio:.2f}\n"
                     f"{s.length:.1f} × {s.width:.1f} cm · {s.max_layers} layers")
        if nxt is not None:
            nop = nxt.operation
            ax.set_xlabel(f"next: {nop.direction}, {nop.orientation}, {nop.ratio:.2f}", fontsize=8, color=_FLAP_COLOR)
        ax.set_title(title, fontsize=9)

        if is_final:  # final folded component
            ax.set_title("FINAL (ideal footprint, zoomed)\n" + title.split("\n", 1)[1], fontsize=9, fontweight="bold")
            if predicted:
                ax.add_patch(Rectangle((0, 0), predicted["length_cm"], predicted["width_cm"], fill=False,
                                       edgecolor=_PRED_COLOR, lw=2, ls="-"))
                ax.set_xlabel(
                    f"ML predicted: {predicted['length_cm']:.1f} × {predicted['width_cm']:.1f} cm, "
                    f"gusset {predicted['gusset_mm']:.1f} mm", fontsize=8, color=_PRED_COLOR, fontweight="bold")

    fig.suptitle("Folding sequence (top view, darker = more layers; hatched = flap being folded; "
                 "blue outline = ML prediction)", fontsize=10)
    fig.tight_layout()
    return fig


def plot_predicted_vs_actual(test_predictions, target: str, label: str) -> Figure:
    """Scatter of model predictions vs. synthetic 'actual' values on the held-out test split."""
    actual = test_predictions[f"{target}__actual"]
    pred = test_predictions[f"{target}__predicted"]
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    ax.scatter(actual, pred, s=6, alpha=0.4, color=_PRED_COLOR)
    lo, hi = min(actual.min(), pred.min()), max(actual.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "--", color="#555", lw=1)
    ax.set_xlabel("actual (synthetic)")
    ax.set_ylabel("predicted")
    ax.set_title(label, fontsize=9)
    fig.tight_layout()
    return fig

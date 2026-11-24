"""Streamlit UI for the Part A fabric folding / dimension prediction prototype.

    streamlit run app.py

The UI only wires things together: fabric + fold inputs -> PredictionService (ML) and
-> visualization (ideal geometry). The two never depend on each other.
"""
from __future__ import annotations

import uuid

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from fabric_folding.evaluation import load_report
from fabric_folding.folding_config import (
    DIRECTION_HELP, DIRECTIONS, MAX_FOLDS, MAX_RATIO, MIN_RATIO, ORIENTATION_HELP, ORIENTATIONS,
    FoldOperation, FoldSequence, ideal_summary,
)
from fabric_folding.materials import COMPRESSION_METHODS, FABRIC_TYPES, MATERIALS, materials_for
from fabric_folding.pipeline import run_training
from fabric_folding.prediction import SYNTHETIC_DISCLAIMER, FabricSpec, PredictionService
from fabric_folding.sample_dataset import DEFAULT_PATH
from fabric_folding.schema import TARGET_LABELS
from fabric_folding.training import MODELS_DIR, BUNDLE_NAME
from fabric_folding.visualization import plot_folding_stages, plot_predicted_vs_actual

st.set_page_config(page_title="Fabric Folding Predictor (Part A)", layout="wide")

PRESETS = {
    "Towel: half lengthwise, then thirds": [("lengthwise", "start", 0.5), ("widthwise", "start", 1 / 3), ("widthwise", "end", 0.5)],
    "Bedsheet: half, half, quarter": [("lengthwise", "start", 0.5), ("widthwise", "start", 0.5), ("lengthwise", "end", 0.5), ("widthwise", "end", 0.5)],
    "Single half fold": [("widthwise", "start", 0.5)],
}
BASIC_FIELDS = {  # key -> (label, min, max, step, format)
    "length_cm": ("Length (cm)", 10.0, 600.0, 1.0, "%.1f"),
    "width_cm": ("Width (cm)", 10.0, 600.0, 1.0, "%.1f"),
    "thickness_mm": ("Thickness (mm)", 0.05, 10.0, 0.05, "%.2f"),
    "gsm": ("Fabric GSM (g/m²)", 20.0, 1000.0, 5.0, "%.0f"),
    "bulk_factor": ("Material bulk factor", 1.0, 4.0, 0.05, "%.2f"),
    "compression_factor": ("Compression factor", 0.1, 1.0, 0.01, "%.2f"),
    "recovery_factor": ("Recovery factor", 0.1, 1.0, 0.01, "%.2f"),
}
MECHANICAL_FIELDS = {
    "bending_stiffness_nm": ("Bending stiffness (N·m)", 1e-7, 1e-2, 1e-6, "%.2e"),
    "friction_coeff": ("Fabric-to-fabric friction (coeff.)", 0.05, 1.5, 0.01, "%.2f"),
    "hem_width_mm": ("Hem width (mm)", 0.0, 150.0, 1.0, "%.1f"),
    "hem_thickness_mm": ("Hem thickness (mm)", 0.05, 20.0, 0.05, "%.2f"),
    "compression_force_n": ("Compression force (N)", 5.0, 500.0, 5.0, "%.0f"),
}
FABRIC_FIELDS = {**BASIC_FIELDS, **MECHANICAL_FIELDS}


# ------------------------------------------------------------------ model loading
@st.cache_resource(show_spinner=False)
def load_service(_version: float) -> PredictionService:
    return PredictionService.from_disk()


def ensure_models() -> PredictionService:
    path = MODELS_DIR / BUNDLE_NAME
    if not path.exists():
        with st.spinner("No trained models found - training on the synthetic sample dataset (~20 s)..."):
            run_training()
    return load_service(path.stat().st_mtime)


# ------------------------------------------------------------------- state helpers
def apply_material_defaults() -> None:
    profile = MATERIALS[st.session_state["material_type"]]
    for key, value in profile.defaults().items():
        st.session_state[key] = float(value)
    st.session_state.setdefault("compression_method", COMPRESSION_METHODS[0])


def on_fabric_type_change() -> None:
    st.session_state["material_type"] = materials_for(st.session_state["fabric_type"])[0]
    apply_material_defaults()


def new_fold(direction="widthwise", orientation="start", ratio=0.5) -> str:
    fid = uuid.uuid4().hex[:8]
    st.session_state[f"dir_{fid}"] = direction
    st.session_state[f"ori_{fid}"] = orientation
    st.session_state[f"rat_{fid}"] = float(round(ratio, 3))
    return fid


def add_fold() -> None:
    st.session_state["fold_ids"].append(new_fold())


def delete_fold(fid: str) -> None:
    st.session_state["fold_ids"].remove(fid)


def move_fold(fid: str, delta: int) -> None:
    ids = st.session_state["fold_ids"]
    i = ids.index(fid)
    j = i + delta
    if 0 <= j < len(ids):
        ids[i], ids[j] = ids[j], ids[i]


def load_preset() -> None:
    st.session_state["fold_ids"] = [new_fold(*p) for p in PRESETS[st.session_state["preset"]]]


def init_state() -> None:
    if "fabric_type" not in st.session_state:
        st.session_state["fabric_type"] = "terry_towel"
        st.session_state["material_type"] = "terry_cotton"
        apply_material_defaults()
        st.session_state["fold_ids"] = [new_fold(*p) for p in PRESETS["Towel: half lengthwise, then thirds"]]


def current_fabric() -> FabricSpec:
    s = st.session_state
    return FabricSpec(
        fabric_type=s["fabric_type"], material_type=s["material_type"],
        compression_method=s["compression_method"], **{k: float(s[k]) for k in FABRIC_FIELDS},
    )


def current_sequence() -> FoldSequence:
    s = st.session_state
    return FoldSequence([FoldOperation(s[f"dir_{f}"], s[f"ori_{f}"], float(s[f"rat_{f}"])) for f in s["fold_ids"]])


# ------------------------------------------------------------------------- inputs
def render_inputs() -> None:
    st.subheader("1 · Fabric information")
    c1, c2 = st.columns(2)
    c1.selectbox("Fabric type", FABRIC_TYPES, key="fabric_type", on_change=on_fabric_type_change)
    c2.selectbox("Material", materials_for(st.session_state["fabric_type"]), key="material_type",
                 on_change=apply_material_defaults,
                 help="Selecting a material pre-fills typical (illustrative) property values; edit them freely.")
    for group in (BASIC_FIELDS, MECHANICAL_FIELDS):
        cols = st.columns(4)
        for i, (key, (label, lo, hi, step, fmt)) in enumerate(group.items()):
            cols[i % 4].number_input(label, min_value=lo, max_value=hi, step=step, format=fmt, key=key)
        if group is MECHANICAL_FIELDS:
            cols[len(group) % 4].selectbox("Compression method", COMPRESSION_METHODS, key="compression_method",
                                           help="How the pack is compressed (procedure).")
    st.caption("Compression factor = share of thickness left while compressed at the reference force (0.5 = squeezed to "
               "half). Recovery factor = share of lost thickness that springs back after release. Sheet mass is derived "
               "from GSM × area and shown with the prediction.")

    st.subheader("2 · Folding sequence")
    st.caption("Operations run top to bottom. **Direction** `lengthwise` = fold line along the length (reduces width); "
               "`widthwise` = fold line along the width (reduces length). **Orientation** = which edge is lifted over "
               "(start = left/top, end = right/bottom). **Ratio** = fold-line position from that edge as a fraction "
               "of the current dimension.")
    ids = st.session_state["fold_ids"]
    for n, fid in enumerate(ids):
        c = st.columns([0.6, 2, 2, 3, 0.5, 0.5, 0.5], vertical_alignment="bottom")
        c[0].markdown(f"**#{n + 1}**")
        c[1].selectbox("Direction", DIRECTIONS, key=f"dir_{fid}", help="\n".join(DIRECTION_HELP.values()))
        c[2].selectbox("Orientation", ORIENTATIONS, key=f"ori_{fid}", help="\n".join(ORIENTATION_HELP.values()))
        c[3].slider("Ratio / position", MIN_RATIO, MAX_RATIO, step=0.01, key=f"rat_{fid}")
        c[4].button("↑", key=f"up_{fid}", on_click=move_fold, args=(fid, -1), disabled=n == 0, help="Move earlier")
        c[5].button("↓", key=f"dn_{fid}", on_click=move_fold, args=(fid, 1), disabled=n == len(ids) - 1, help="Move later")
        c[6].button("✕", key=f"del_{fid}", on_click=delete_fold, args=(fid,), help="Remove")
    b = st.columns([1.5, 3, 1.5, 4])
    b[0].button("➕ Add fold", on_click=add_fold, disabled=len(ids) >= MAX_FOLDS,
                help=f"Up to {MAX_FOLDS} folds are supported by the model.")
    b[1].selectbox("Preset", list(PRESETS), key="preset", label_visibility="collapsed")
    b[2].button("Load preset", on_click=load_preset)


# ------------------------------------------------------------------------ results
def render_results(service: PredictionService) -> None:
    st.subheader("3 · Prediction")
    if st.button("Predict folded dimensions", type="primary"):
        try:
            fabric, seq = current_fabric(), current_sequence()
            pred = service.predict(fabric, seq)
            st.session_state["result"] = {"fabric": fabric, "sequence": seq, "prediction": pred}
        except ValueError as e:
            st.session_state.pop("result", None)
            st.error(str(e))

    res = st.session_state.get("result")
    if not res:
        st.info("Configure the fabric and folds, then press **Predict folded dimensions**.")
        return
    fabric: FabricSpec = res["fabric"]
    seq: FoldSequence = res["sequence"]
    pred = res["prediction"]

    if (fabric, seq) != (current_fabric(), current_sequence()):
        st.warning("Inputs have changed since this prediction - press **Predict folded dimensions** to update.")
    if pred.disclaimer:
        st.warning(f"⚠️ **Synthetic sample data.** {pred.disclaimer}")
    for w in pred.warnings:
        st.warning(f"Outside the training range: {w}. Treat this prediction as an extrapolation.")

    m = st.columns(4)
    m[0].metric("Predicted folded length", f"{pred.folded_length_cm:.1f} cm")
    m[1].metric("Predicted folded width", f"{pred.folded_width_cm:.1f} cm")
    m[2].metric("Predicted gusset (thickness)", f"{pred.gusset_mm:.1f} mm")
    m[3].metric("Sheet mass (derived, not ML)", f"{fabric.sheet_mass_g:.0f} g")
    ideal = ideal_summary(fabric.length_cm, fabric.width_cm, seq)
    st.caption(f"For reference, the ideal paper-thin geometry of this sequence is {ideal['ideal_length']:.1f} × "
               f"{ideal['ideal_width']:.1f} cm with {ideal['ideal_layers']} layers; the difference comes from the "
               "learned material effects (bulk, compression/recovery, fold-edge roll).")

    left, right = st.columns(2)
    with left:
        st.markdown("**Original fabric & material properties**")
        props = [("Fabric type", fabric.fabric_type), ("Material", fabric.material_type)]
        props += [(label, f"{getattr(fabric, key):.4g}") for key, (label, *_) in FABRIC_FIELDS.items()]
        props += [("Compression method", fabric.compression_method),
                  ("Sheet mass (derived, g)", f"{fabric.sheet_mass_g:.0f}")]
        st.table(pd.DataFrame(props, columns=["Property", "Value"]).set_index("Property"))
    with right:
        st.markdown("**Folding sequence**")
        st.table(pd.DataFrame(
            [{"Step": i, "Direction": o.direction, "Orientation": o.orientation, "Ratio": f"{o.ratio:.2f}"}
             for i, o in enumerate(seq, 1)]).set_index("Step"))

    st.markdown("**Folding visualization**")
    fig = plot_folding_stages(
        fabric.length_cm, fabric.width_cm, seq,
        predicted={"length_cm": pred.folded_length_cm, "width_cm": pred.folded_width_cm, "gusset_mm": pred.gusset_mm},
    )
    st.pyplot(fig)
    plt.close(fig)


# -------------------------------------------------------------------- evaluation
def render_evaluation() -> None:
    st.subheader("Model evaluation (held-out test split)")
    loaded = load_report()
    if not loaded:
        st.info("No evaluation report found - train the models first.")
        return
    report, test_preds = loaded
    if report["is_synthetic"]:
        st.warning("⚠️ Metrics measure how well the model reproduces the **synthetic** sample data. They say nothing "
                   "about accuracy on real fabric until the model is retrained on measured data and re-evaluated.")
    st.caption(f"{report['model']} · {report['n_train']} training rows / {report['n_test']} test rows · "
               f"data source: {', '.join(report['data_sources'])}")

    rows = []
    for t, m in report["targets"].items():
        rows.append({"Output": TARGET_LABELS[t],
                     "MAE": m["test"]["mae"], "RMSE": m["test"]["rmse"], "R²": m["test"]["r2"],
                     "Baseline MAE (ideal geometry, no ML)": m["baseline_ideal_geometry_test"]["mae"],
                     "Train R²": m["train"]["r2"]})
    st.dataframe(pd.DataFrame(rows).set_index("Output").style.format("{:.4f}"), width="stretch")
    st.caption("MAE/RMSE are in the output's own unit (cm for length/width, mm for gusset). The baseline predicts "
               "the unfolded-material-free geometry (and nominal thickness × layers for gusset); the ML model should beat it.")

    cols = st.columns(3)
    for col, t in zip(cols, report["targets"]):
        fig = plot_predicted_vs_actual(test_preds, t, TARGET_LABELS[t])
        col.pyplot(fig)
        plt.close(fig)

    with st.expander("Most important features per output"):
        for t, m in report["targets"].items():
            st.markdown(f"**{TARGET_LABELS[t]}**")
            st.bar_chart(pd.Series(m["top_features"]))

    st.divider()
    st.subheader("Sample dataset")
    st.caption(f"`{DEFAULT_PATH.relative_to(DEFAULT_PATH.parent.parent)}` - SYNTHETIC. To use real measurements, supply a CSV "
               "with the same columns and run `python train.py --data your_file.csv`.")
    if DEFAULT_PATH.exists():
        df = pd.read_csv(DEFAULT_PATH)
        st.write(f"{len(df)} records · materials: {', '.join(sorted(df['material_type'].unique()))}")
        st.dataframe(df.head(50), width="stretch")
    if st.button("Retrain models on the current sample dataset"):
        with st.spinner("Training..."):
            run_training()
        st.cache_resource.clear()
        st.rerun()


# ------------------------------------------------------------------------- main
def main() -> None:
    init_state()
    st.title("Fabric Folding & Dimension Predictor")
    st.caption("Part A prototype · Packing 3D Simulator · bedsheets & terry towels")
    st.error(f"**SYNTHETIC SAMPLE DATA** - {SYNTHETIC_DISCLAIMER}", icon="🧪")
    service = ensure_models()

    tab_predict, tab_eval = st.tabs(["Fold & predict", "Model evaluation & dataset"])
    with tab_predict:
        render_inputs()
        render_results(service)
    with tab_eval:
        render_evaluation()


main()

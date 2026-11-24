import json

import pandas as pd
import pytest

from fabric_folding.folding_config import FoldOperation, FoldSequence, ideal_summary, simulate_ideal
from fabric_folding.pipeline import run_training
from fabric_folding.prediction import FabricSpec, PredictionService
from fabric_folding.preprocessing import FeatureBuilder
from fabric_folding.sample_dataset import generate_dataset, write_dataset
from fabric_folding.schema import validate_dataset


def seq(*ops):
    return FoldSequence([FoldOperation(*o) for o in ops])


# ------------------------------------------------------------------ geometry
def test_half_fold_halves_dimension_and_doubles_layers():
    s = simulate_ideal(200, 100, seq(("widthwise", "start", 0.5)))[-1]
    assert (s.length, s.width, s.max_layers) == (100, 100, 2)


def test_third_fold_overhang_and_layers():
    # fold at 2/3 from start: flap (2/3) is larger than the remaining third
    s = simulate_ideal(300, 100, seq(("widthwise", "start", 2 / 3)))[-1]
    assert s.length == pytest.approx(200)
    assert s.max_layers == 2


def test_both_axes_multiply_layers():
    s = simulate_ideal(200, 100, seq(("widthwise", "start", 0.5), ("lengthwise", "end", 0.5)))[-1]
    assert (s.length, s.width, s.max_layers) == (100, 50, 4)


def test_start_vs_end_orientation_changes_layer_profile_not_extent():
    a = simulate_ideal(300, 100, seq(("widthwise", "start", 0.25)))[-1]
    b = simulate_ideal(300, 100, seq(("widthwise", "end", 0.25)))[-1]
    assert a.length == b.length == pytest.approx(225)
    assert a.x_profile != b.x_profile


def test_area_conserved_layer_weighted():
    s = simulate_ideal(240, 200, seq(("widthwise", "end", 0.4), ("lengthwise", "start", 0.3), ("widthwise", "start", 0.5)))[-1]
    area = sum((xb - xa) * (yb - ya) * nx * ny for xa, xb, nx in s.x_profile for ya, yb, ny in s.y_profile)
    assert area == pytest.approx(240 * 200)


# ------------------------------------------------------------ dataset / schema
def test_dataset_schema_and_synthetic_label():
    df = generate_dataset(200, seed=1)
    validate_dataset(df)
    assert set(df["data_source"]) == {"SYNTHETIC_SAMPLE"}
    assert {"bedsheet", "terry_towel"} == set(df["fabric_type"])


def test_validate_rejects_missing_column():
    df = generate_dataset(20, seed=1).drop(columns=["final_gusset_mm"])
    with pytest.raises(ValueError):
        validate_dataset(df)


# ------------------------------------------------------- shared feature logic
def test_training_and_prediction_features_identical():
    df = generate_dataset(100, seed=3)
    fb = FeatureBuilder().fit(df)
    row = df.iloc[0]
    from_df = fb.transform(df.iloc[[0]])
    rec = row.to_dict()
    rec["fold_sequence"] = FoldSequence.from_json(rec["fold_sequence"])  # object form used at prediction
    from_obj = fb.transform(pd.DataFrame([rec]))
    pd.testing.assert_frame_equal(from_df, from_obj)


# ----------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def service(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    csv = d / "data.csv"
    write_dataset(csv, n_records=1500, seed=7)
    report = run_training(csv, d / "models")
    return PredictionService.from_disk(d / "models"), report


def fabric(**kw):
    base = dict(fabric_type="terry_towel", material_type="terry_cotton", length_cm=140, width_cm=70, thickness_mm=2.5, gsm=500,
                bulk_factor=1.6, compression_factor=0.5, recovery_factor=0.7, bending_stiffness_nm=2e-4, friction_coeff=0.5,
                hem_width_mm=20, hem_thickness_mm=6, compression_force_n=100, compression_method="flat_press")
    return FabricSpec(**{**base, **kw})


FABRIC = fabric()


def test_metrics_reported(service):
    _, report = service
    for t in report["targets"].values():
        assert {"mae", "rmse", "r2"} <= set(t["test"])
        assert t["test"]["r2"] > 0.8
    assert report["is_synthetic"]


def test_material_change_changes_prediction(service):
    svc, _ = service
    s = seq(("widthwise", "start", 0.5), ("lengthwise", "end", 0.5))
    base = svc.predict(FABRIC, s)
    thick = svc.predict(fabric(thickness_mm=3.1), s)
    assert thick.gusset_mm > base.gusset_mm


def test_sequence_change_changes_prediction(service):
    svc, _ = service
    a = svc.predict(FABRIC, seq(("widthwise", "start", 0.5)))
    b = svc.predict(FABRIC, seq(("widthwise", "start", 0.5), ("widthwise", "start", 0.5)))
    assert b.folded_length_cm < a.folded_length_cm
    assert b.gusset_mm > a.gusset_mm


def test_empty_sequence_rejected(service):
    with pytest.raises(ValueError):
        service[0].predict(FABRIC, FoldSequence([]))


def test_out_of_range_warning(service):
    svc, _ = service
    p = svc.predict(fabric(length_cm=900), seq(("widthwise", "start", 0.5)))
    assert any("length_cm" in w for w in p.warnings)
    assert p.is_synthetic and p.disclaimer


def test_new_parameters_change_prediction_and_mass(service):
    svc, _ = service
    s = seq(("widthwise", "start", 0.5), ("lengthwise", "end", 0.5), ("widthwise", "end", 0.5))
    base = svc.predict(FABRIC, s)
    assert svc.predict(fabric(compression_force_n=200), s).gusset_mm < base.gusset_mm
    assert svc.predict(fabric(hem_thickness_mm=9), s).gusset_mm > base.gusset_mm
    assert FABRIC.sheet_mass_g == pytest.approx(500 * 140 * 70 / 1e4)

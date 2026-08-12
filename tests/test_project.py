"""Tests for projecting model fields into the observed type space.

The failure mode this guards against is silent and total: if the model field
is prepared even slightly differently from the observations, the projection
lands somewhere else in the same-shaped space. Nothing raises, the PCs look
plausible, the type frequencies look plausible, and every model appears biased.

So the tests here mostly check invariances -- that a model identical to the
observations comes back identical, and that a model that differs in a known
way differs in the expected direction.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.cluster.eof import decompose, prepare
from src.evaluate.project import (
    ModelResult,
    assign,
    frequencies,
    prepare_like_observations,
    project,
    regrid_to,
)


def make_field(n_time=400, n_lat=9, n_lon=13, amplitude=1.0, seed=0,
               lat_name="latitude", lon_name="longitude", lon_west=90.0):
    rng = np.random.default_rng(seed)
    lat = np.linspace(-10.0, -60.0, n_lat)
    lon = np.linspace(lon_west, lon_west + 90.0, n_lon)
    time = pd.date_range("1979-01-01", periods=n_time, freq="1D")

    lon_g, lat_g = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))
    p1 = np.sin(2 * lon_g) * np.cos(lat_g)
    p2 = np.cos(3 * lon_g) * np.sin(2 * lat_g)

    data = (
        rng.normal(0, 3.0, n_time)[:, None, None] * p1[None]
        + rng.normal(0, 1.0, n_time)[:, None, None] * p2[None]
    ) * amplitude

    return xr.DataArray(
        data, dims=("time", lat_name, lon_name),
        coords={"time": time, lat_name: lat, lon_name: lon}, name="mslp",
    )


def observed_basis(field):
    matrix, scales = prepare({"mslp": field})
    result = decompose(matrix, n_modes=6)
    return result, scales["mslp"]


# --- regridding ----------------------------------------------------------

def test_regrid_lands_on_the_reference_grid():
    reference = make_field(n_lat=9, n_lon=13)
    model = make_field(n_lat=15, n_lon=20, seed=1, lat_name="lat", lon_name="lon")
    out = regrid_to(model, reference)
    assert out.sizes["latitude"] == 9
    assert out.sizes["longitude"] == 13


def test_regrid_accepts_either_coordinate_naming():
    reference = make_field()
    a = regrid_to(make_field(seed=2, lat_name="lat", lon_name="lon"), reference)
    b = regrid_to(make_field(seed=2), reference)
    assert np.allclose(a.values, b.values)


def test_regrid_normalises_longitude_convention():
    """Models disagree on 0-360 versus -180-180, and a mismatch gives NaN."""
    reference = make_field(lon_west=90.0)
    model = make_field(lon_west=90.0, seed=3)
    shifted = model.assign_coords(
        longitude=xr.where(model.longitude > 180, model.longitude - 360, model.longitude)
    )
    out = regrid_to(shifted, reference)
    assert not bool(out.isnull().all())


def test_regrid_handles_ascending_latitude():
    reference = make_field()
    model = make_field(seed=4).isel(latitude=slice(None, None, -1))
    out = regrid_to(model, reference)
    assert out.latitude.values[0] == pytest.approx(reference.latitude.values[0])


def test_regrid_raises_on_a_disjoint_domain():
    reference = make_field(lon_west=90.0)
    elsewhere = make_field(lon_west=270.0, seed=5)
    with pytest.raises(ValueError, match="only NaN"):
        regrid_to(elsewhere, reference)


def test_regrid_preserves_a_uniform_field():
    reference = make_field(n_lat=9, n_lon=13)
    flat = xr.ones_like(make_field(n_lat=15, n_lon=20, seed=6)) * 7.0
    assert np.allclose(regrid_to(flat, reference).values, 7.0)


# --- projection ----------------------------------------------------------

def test_projecting_the_observations_recovers_their_own_pcs():
    """The invariance everything else depends on."""
    field = make_field(n_time=500)
    result, scale = observed_basis(field)

    prepared = field.astype("float64") / scale
    pcs, residual = project({"mslp": prepared}, result.patterns)

    assert np.allclose(pcs.values, result.pcs.values, atol=1e-8)
    assert residual < 1e-10


def test_unexplained_variance_is_zero_for_a_full_basis():
    field = make_field(n_time=200, n_lat=5, n_lon=6)
    matrix, scales = prepare({"mslp": field})
    result = decompose(matrix, n_modes=min(matrix.shape))
    _, residual = project({"mslp": field.astype("float64") / scales["mslp"]},
                          result.patterns)
    assert residual == pytest.approx(0.0, abs=1e-9)


def test_unexplained_variance_grows_when_the_basis_is_truncated():
    field = make_field(n_time=400)
    matrix, scales = prepare({"mslp": field})
    prepared = field.astype("float64") / scales["mslp"]

    few = project({"mslp": prepared}, decompose(matrix, n_modes=1).patterns)[1]
    many = project({"mslp": prepared}, decompose(matrix, n_modes=6).patterns)[1]
    assert few > many


def test_projection_rejects_a_mismatched_grid():
    """The error that would otherwise be silent: a differently prepared field."""
    field = make_field()
    result, scale = observed_basis(field)
    smaller = make_field(n_lat=5, n_lon=6, seed=7).astype("float64") / scale
    with pytest.raises(ValueError, match="not prepared the same way"):
        project({"mslp": smaller}, result.patterns)


def test_a_weaker_model_projects_onto_weaker_pcs():
    """Amplitude bias must survive the projection, not be normalised away."""
    observations = make_field(n_time=400)
    result, scale = observed_basis(observations)

    weak = make_field(n_time=400, amplitude=0.5, seed=8).astype("float64") / scale
    strong = make_field(n_time=400, amplitude=2.0, seed=8).astype("float64") / scale

    weak_pcs = project({"mslp": weak}, result.patterns)[0]
    strong_pcs = project({"mslp": strong}, result.patterns)[0]
    assert float(weak_pcs.std()) < float(strong_pcs.std())


def test_prepare_like_observations_does_not_rescale_by_the_model():
    """Recomputing the scale from the model would hide amplitude bias."""
    reference = make_field()
    weak = prepare_like_observations(
        make_field(amplitude=0.5, seed=9), reference, scale=10.0, do_detrend=False
    )
    strong = prepare_like_observations(
        make_field(amplitude=2.0, seed=9), reference, scale=10.0, do_detrend=False
    )
    assert float(strong.std()) > 3 * float(weak.std())


# --- assignment ----------------------------------------------------------

def test_every_day_gets_a_type():
    pcs = xr.DataArray(
        np.random.default_rng(0).normal(0, 1, (100, 6)),
        dims=("time", "mode"),
        coords={"time": pd.date_range("1979-01-01", periods=100), "mode": np.arange(1, 7)},
    )
    centroids = np.random.default_rng(1).normal(0, 1, (4, 6))
    labels = assign(pcs, centroids, pc_scale=1.0)
    assert labels.sizes["time"] == 100
    assert set(np.unique(labels.values)) <= set(range(4))


def test_a_day_at_a_centroid_is_assigned_to_it():
    centroids = np.array([[3.0, 0.0], [-3.0, 0.0], [0.0, 3.0]])
    pcs = xr.DataArray(
        centroids.copy(), dims=("time", "mode"),
        coords={"time": pd.date_range("1979-01-01", periods=3), "mode": [1, 2]},
    )
    assert list(assign(pcs, centroids, pc_scale=1.0).values) == [0, 1, 2]


def test_assignment_uses_only_as_many_modes_as_the_centroids_have():
    centroids = np.array([[3.0, 0.0], [-3.0, 0.0]])
    pcs = xr.DataArray(
        np.array([[3.0, 0.0, 99.0], [-3.0, 0.0, -99.0]]),
        dims=("time", "mode"),
        coords={"time": pd.date_range("1979-01-01", periods=2), "mode": [1, 2, 3]},
    )
    assert list(assign(pcs, centroids, pc_scale=1.0).values) == [0, 1]


def test_pc_scale_is_applied_not_recomputed():
    """Scaling by the model's own spread would erase an amplitude bias."""
    centroids = np.array([[1.0, 0.0], [-1.0, 0.0]])
    weak = xr.DataArray(
        np.array([[0.1, 0.0], [-0.1, 0.0]]), dims=("time", "mode"),
        coords={"time": pd.date_range("1979-01-01", periods=2), "mode": [1, 2]},
    )
    tight = assign(weak, centroids, pc_scale=1.0)
    loose = assign(weak, centroids, pc_scale=0.1)
    assert list(tight.values) == list(loose.values) == [0, 1]


# --- summarising ---------------------------------------------------------

def test_frequencies_sum_to_one():
    labels = xr.DataArray(
        np.tile(np.arange(4), 50), dims="time",
        coords={"time": pd.date_range("1979-01-01", periods=200)},
    )
    assert frequencies(labels, 4).sum() == pytest.approx(1.0)


def test_total_absolute_bias_is_zero_for_a_perfect_model():
    observed = np.array([0.4, 0.3, 0.2, 0.1])
    result = ModelResult("X", "historical", None, observed.copy(), 0.0, 100)
    assert result.total_absolute_bias(observed) == pytest.approx(0.0)


def test_total_absolute_bias_is_one_for_no_overlap():
    observed = np.array([1.0, 0.0])
    result = ModelResult("X", "historical", None, np.array([0.0, 1.0]), 0.0, 100)
    assert result.total_absolute_bias(observed) == pytest.approx(1.0)


def test_frequency_bias_keeps_the_sign():
    observed = np.array([0.5, 0.5])
    result = ModelResult("X", "historical", None, np.array([0.7, 0.3]), 0.0, 100)
    bias = result.frequency_bias(observed)
    assert bias[0] > 0 and bias[1] < 0

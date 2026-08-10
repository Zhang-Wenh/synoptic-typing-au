"""Tests for the EOF reduction.

The properties worth guarding are the ones that fail quietly. An EOF of a
badly prepared field still returns orthogonal modes, an ordered variance
spectrum, and maps that look like weather. Nothing raises. The tests below
check that the preparation does what it claims by constructing fields where
the right answer is known in advance.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.cluster.eof import (
    coarsen,
    decompose,
    prepare,
    standardise_variable,
    unstack_pattern,
)


def make_anom(n_time=400, scale=1.0, seed=0, name="mslp", n_lat=9, n_lon=13):
    """Anomaly field built from two fixed spatial patterns plus noise.

    Two patterns with different amplitudes means the leading two modes have a
    known order and a known spatial form.
    """
    rng = np.random.default_rng(seed)
    lat = np.linspace(-10.0, -60.0, n_lat)
    lon = np.linspace(90.0, 180.0, n_lon)
    time = pd.date_range("1979-01-01", periods=n_time, freq="1D")

    lon_g, lat_g = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))
    p1 = np.sin(2 * lon_g) * np.cos(lat_g)
    p2 = np.cos(3 * lon_g) * np.sin(2 * lat_g)

    a1 = rng.normal(0, 3.0, n_time)
    a2 = rng.normal(0, 1.0, n_time)
    data = (
        a1[:, None, None] * p1[None]
        + a2[:, None, None] * p2[None]
        + rng.normal(0, 0.2, (n_time, n_lat, n_lon))
    )

    return xr.DataArray(
        data * scale,
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": lat, "longitude": lon},
        name=name,
    )


# --- standardisation -----------------------------------------------------

def test_standardise_gives_unit_variance():
    scaled, _ = standardise_variable(make_anom(scale=700.0))
    assert float(scaled.std(dtype="float64")) == pytest.approx(1.0, rel=1e-10)


def test_standardise_returns_the_scale_it_used():
    field = make_anom(scale=700.0)
    _, scale = standardise_variable(field)
    assert scale == pytest.approx(float(field.std(dtype="float64")), rel=1e-12)


def test_standardise_preserves_spatial_variance_structure():
    """The property that separates this from per-grid-point standardisation.

    Dividing by one scalar must leave the ratio of variance between latitudes
    unchanged. Per-grid-point standardisation would set every ratio to one.
    """
    field = make_anom(scale=700.0)
    before = field.std("time", dtype="float64")
    after = standardise_variable(field)[0].std("time", dtype="float64")
    ratio = (after / before).values
    assert np.allclose(ratio, ratio.flat[0], rtol=1e-10)


def test_standardise_rejects_a_constant_field():
    field = make_anom() * 0.0
    with pytest.raises(ValueError, match="zero variance"):
        standardise_variable(field)


# --- preparation ---------------------------------------------------------

def test_prepare_stacks_both_variables_into_one_matrix():
    fields = {"mslp": make_anom(scale=700.0), "z": make_anom(scale=70.0, seed=1)}
    matrix, _ = prepare(fields)
    assert matrix.dims == ("time", "cell")
    assert matrix.sizes["cell"] == 2 * 9 * 13


def test_prepare_equalises_the_two_variables():
    """The defect this step exists for.

    MSLP in Pa and Z500 in metres differ in variance by about a factor of 100.
    After preparation, neither may dominate.
    """
    fields = {"mslp": make_anom(scale=700.0), "z": make_anom(scale=70.0, seed=1)}
    matrix, _ = prepare(fields)
    per_variable = (matrix**2).unstack("cell").sum(("latitude", "longitude", "time"))
    a, b = (float(per_variable.sel(variable=v)) for v in ("mslp", "z"))
    assert a / b == pytest.approx(1.0, rel=0.05)


def test_prepare_is_insensitive_to_the_unit_of_a_variable():
    """Storing MSLP in hPa rather than Pa must not change the result."""
    z = make_anom(scale=70.0, seed=1)
    in_pa, _ = prepare({"mslp": make_anom(scale=700.0), "z": z})
    in_hpa, _ = prepare({"mslp": make_anom(scale=7.0), "z": z})
    assert np.allclose(in_pa.values, in_hpa.values, rtol=1e-10)


def test_prepare_applies_latitude_weighting():
    """High latitudes must be downweighted relative to low ones."""
    field = make_anom(scale=1.0)
    matrix, _ = prepare({"mslp": field})
    weighted = matrix.unstack("cell").std("time", dtype="float64").sel(variable="mslp")
    raw = field.std("time", dtype="float64")
    ratio = (weighted / raw).mean("longitude")
    assert float(ratio.isel(latitude=0)) > float(ratio.isel(latitude=-1))


def test_prepare_returns_the_scales(): 
    _, scales = prepare({"mslp": make_anom(scale=700.0), "z": make_anom(scale=70.0, seed=1)})
    assert set(scales) == {"mslp", "z"}
    assert scales["mslp"] > scales["z"]


# --- decomposition -------------------------------------------------------

def test_variance_fractions_are_ordered_and_sum_to_one():
    matrix, _ = prepare({"mslp": make_anom()})
    result = decompose(matrix, n_modes=20)
    assert np.all(np.diff(result.variance_fraction) <= 1e-12)
    assert result.cumulative()[-1] == pytest.approx(1.0, abs=0.02)


def test_two_planted_patterns_dominate_the_spectrum():
    """The synthetic field has two patterns; two modes should capture it."""
    matrix, _ = prepare({"mslp": make_anom()})
    result = decompose(matrix, n_modes=20)
    assert result.cumulative()[1] > 0.95


def test_leading_mode_matches_the_stronger_planted_pattern():
    matrix, _ = prepare({"mslp": make_anom()})
    result = decompose(matrix, n_modes=5)
    assert result.variance_fraction[0] > 3 * result.variance_fraction[1]


def test_principal_components_are_uncorrelated():
    matrix, _ = prepare({"mslp": make_anom()})
    pcs = decompose(matrix, n_modes=6).pcs.values
    corr = np.corrcoef(pcs.T)
    off = corr - np.diag(np.diag(corr))
    assert np.abs(off).max() < 1e-8


def test_patterns_are_orthonormal():
    matrix, _ = prepare({"mslp": make_anom()})
    p = decompose(matrix, n_modes=6).patterns.values
    assert np.allclose(p @ p.T, np.eye(6), atol=1e-8)


def test_pcs_carry_the_time_coordinate():
    field = make_anom()
    matrix, _ = prepare({"mslp": field})
    assert decompose(matrix, n_modes=3).pcs.time.equals(field.time)


def test_n_modes_is_capped_by_the_matrix_size():
    matrix, _ = prepare({"mslp": make_anom(n_time=30)})
    assert decompose(matrix, n_modes=500).pcs.sizes["mode"] == 30


def test_decompose_rejects_an_unstacked_field():
    with pytest.raises(ValueError, match="2-D"):
        decompose(make_anom(), n_modes=3)


# --- mode selection ------------------------------------------------------

def test_n_modes_for_a_variance_target():
    matrix, _ = prepare({"mslp": make_anom()})
    result = decompose(matrix, n_modes=20)
    for target in (0.5, 0.95, 0.99):
        n = result.n_modes_for(target)
        assert result.cumulative()[n - 1] >= target
        if n > 1:
            assert result.cumulative()[n - 2] < target


def test_north_errors_shrink_with_sample_size():
    """More days means better separated eigenvalues."""
    short = decompose(prepare({"mslp": make_anom(n_time=200)})[0], n_modes=5)
    long = decompose(prepare({"mslp": make_anom(n_time=2000)})[0], n_modes=5)
    assert (long.north_errors()[0] / long.eigenvalues[0]) < (
        short.north_errors()[0] / short.eigenvalues[0]
    )


def test_well_separated_leading_mode_passes_north():
    result = decompose(prepare({"mslp": make_anom(n_time=2000)})[0], n_modes=10)
    assert bool(result.north_separable()[0])


def test_degenerate_modes_fail_north():
    """A propagating wave produces two modes that are not separately meaningful.

    The pair is built to be exactly degenerate: a quadrature pattern pair of
    equal norm after weighting, driven by sine and cosine of the same period
    over a whole number of cycles, so both have identical variance by
    construction rather than by luck of a random draw. The two modes span the
    subspace the wave lives in; which rotation of that subspace the SVD
    returns is arbitrary, and interpreting either pattern on its own would be
    reading structure into an arbitrary choice.
    """
    lat = np.linspace(-10.0, -60.0, 9)
    lon = np.linspace(90.0, 180.0, 13)
    n_time, cycles = 800, 40
    time = pd.date_range("1979-01-01", periods=n_time, freq="1D")
    lon_g, lat_g = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))

    # Divide out the weighting the pipeline will apply, so the pair is an
    # exact quadrature pair in the space the decomposition actually sees.
    w = np.sqrt(np.cos(lat_g))
    p1, p2 = np.sin(2 * lon_g) / w, np.cos(2 * lon_g) / w

    # Equal norm as well as orthogonal: on a discrete grid a sine and a cosine
    # of the same wavenumber do not automatically have the same norm, and a
    # difference there would split the eigenvalues on its own.
    p1 = p1 / np.sqrt(np.sum((p1 * w) ** 2))
    p2 = p2 / np.sqrt(np.sum((p2 * w) ** 2))

    phase = 2 * np.pi * cycles * np.arange(n_time) / n_time
    data = (
        np.cos(phase)[:, None, None] * p1[None]
        + np.sin(phase)[:, None, None] * p2[None]
    )

    field = xr.DataArray(
        data, dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": lat, "longitude": lon}, name="mslp",
    )
    result = decompose(prepare({"mslp": field})[0], n_modes=4)
    assert result.eigenvalues[0] == pytest.approx(result.eigenvalues[1], rel=1e-6)
    assert not bool(result.north_separable()[0])


# --- inverting the preparation ------------------------------------------

def test_unstack_returns_a_map_per_variable():
    fields = {"mslp": make_anom(scale=700.0), "z": make_anom(scale=70.0, seed=1)}
    matrix, scales = prepare(fields)
    ds = unstack_pattern(decompose(matrix, n_modes=4).patterns, mode=1, scales=scales)
    assert set(ds.data_vars) == {"mslp", "z"}
    assert ds["mslp"].dims == ("latitude", "longitude")


def test_unstack_undoes_the_latitude_weighting():
    """A pattern uniform in the weighted space must not be uniform after."""
    matrix, scales = prepare({"mslp": make_anom()})
    result = decompose(matrix, n_modes=4)
    flat = xr.zeros_like(result.patterns) + 1.0
    ds = unstack_pattern(flat, mode=1)
    profile = ds["mslp"].mean("longitude")
    assert float(profile.isel(latitude=-1)) > float(profile.isel(latitude=0))


def test_unstack_restores_physical_scale():
    fields = {"mslp": make_anom(scale=700.0), "z": make_anom(scale=70.0, seed=1)}
    matrix, scales = prepare(fields)
    patterns = decompose(matrix, n_modes=4).patterns
    plain = unstack_pattern(patterns, mode=1)
    scaled = unstack_pattern(patterns, mode=1, scales=scales)
    ratio = float(np.abs(scaled["mslp"]).max() / np.abs(plain["mslp"]).max())
    assert ratio == pytest.approx(scales["mslp"], rel=1e-8)


# --- coarsening ----------------------------------------------------------

def test_coarsen_reduces_the_grid():
    out = coarsen(make_anom(n_lat=12, n_lon=16), factor=4)
    assert out.sizes["latitude"] == 3 and out.sizes["longitude"] == 4


def test_coarsen_of_one_is_a_no_op():
    field = make_anom()
    assert coarsen(field, factor=1) is field


def test_coarsen_averages_rather_than_subsamples():
    """A block mean must equal the mean of the block, not one of its members."""
    field = make_anom(n_lat=8, n_lon=8, n_time=5)
    out = coarsen(field, factor=2)
    block = field.isel(latitude=slice(0, 2), longitude=slice(0, 2)).mean(
        ("latitude", "longitude")
    )
    assert np.allclose(out.isel(latitude=0, longitude=0).values, block.values)


def test_coarsen_suppresses_small_scale_variance():
    """The point of averaging: sub-grid noise is filtered, not aliased.

    Subsampling would leave the noise variance untouched, since it simply
    keeps one point in four.
    """
    rng = np.random.default_rng(7)
    field = make_anom(n_lat=40, n_lon=40, n_time=200)
    noisy = field + rng.normal(0, 5.0, field.shape)
    smooth_var = float(coarsen(noisy, 4).var())
    subsampled_var = float(noisy.isel(
        latitude=slice(None, None, 4), longitude=slice(None, None, 4)
    ).var())
    assert smooth_var < subsampled_var


def test_coarsen_keeps_the_time_axis():
    field = make_anom(n_lat=12, n_lon=16)
    assert coarsen(field, 4).time.equals(field.time)


def test_prepare_accepts_a_coarsen_factor():
    fields = {"mslp": make_anom(n_lat=12, n_lon=16)}
    fine, _ = prepare(fields, coarsen_factor=1)
    coarse, _ = prepare(fields, coarsen_factor=4)
    assert coarse.sizes["cell"] * 16 == fine.sizes["cell"]


def test_coarsening_preserves_the_leading_mode():
    """The physical claim behind coarsening: the large-scale patterns survive.

    Correlating the leading PC before and after should be near perfect, since
    the planted patterns are large scale.
    """
    fields = {"mslp": make_anom(n_lat=40, n_lon=40, n_time=300)}
    fine = decompose(prepare(fields, coarsen_factor=1)[0], n_modes=3)
    coarse = decompose(prepare(fields, coarsen_factor=4)[0], n_modes=3)
    r = np.corrcoef(fine.pcs.isel(mode=0).values, coarse.pcs.isel(mode=0).values)[0, 1]
    assert abs(r) > 0.99

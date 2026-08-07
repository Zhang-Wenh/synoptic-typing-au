"""Tests for latitude weighting.

The distinction between cos(lat) and sqrt(cos(lat)) is invisible at run time:
using the wrong one produces plausible output with a systematic latitudinal
bias. These tests pin the property down.
"""

import numpy as np
import pytest
import xarray as xr

from src.preprocess.weights import (
    area_weights,
    eof_weights,
    apply_eof_weights,
    weighted_mean,
)


@pytest.fixture
def lat():
    return xr.DataArray(
        np.arange(-10.0, -60.25, -0.25), dims="latitude", name="latitude"
    ).assign_coords(latitude=np.arange(-10.0, -60.25, -0.25))


def test_eof_weights_are_sqrt_of_area_weights(lat):
    np.testing.assert_allclose(
        eof_weights(lat).values, np.sqrt(area_weights(lat).values), rtol=1e-12
    )


def test_weighted_variance_is_area_weighted(lat):
    """The point of sqrt(cos): variance of weighted data scales as cos(lat)."""
    rng = np.random.default_rng(0)
    field = xr.DataArray(
        rng.standard_normal((500, lat.size)),
        dims=("time", "latitude"),
        coords={"latitude": lat.latitude},
    )
    weighted = field * eof_weights(lat)
    ratio = weighted.var("time") / field.var("time")
    np.testing.assert_allclose(
        ratio.values, area_weights(lat).values, rtol=0.15
    )


def test_weights_decrease_toward_the_pole(lat):
    w = area_weights(lat).values
    assert np.all(np.diff(w) < 0)


def test_weights_are_hemisphere_symmetric():
    north = xr.DataArray([30.0, 60.0], dims="latitude")
    south = xr.DataArray([-30.0, -60.0], dims="latitude")
    np.testing.assert_allclose(
        area_weights(north).values, area_weights(south).values, rtol=1e-12
    )


def test_weights_are_non_negative_at_the_pole():
    pole = xr.DataArray([-90.0, 90.0], dims="latitude")
    assert np.all(area_weights(pole).values >= 0)
    assert np.all(np.isfinite(eof_weights(pole).values))


def test_weighted_mean_of_a_constant_field_is_that_constant(lat):
    field = xr.DataArray(
        np.full((10, lat.size, 20), 7.0),
        dims=("time", "latitude", "longitude"),
        coords={"latitude": lat.latitude},
    )
    np.testing.assert_allclose(weighted_mean(field).values, 7.0, rtol=1e-12)


def test_weighted_mean_differs_from_unweighted_on_a_gradient(lat):
    """A field varying with latitude must not give the same answer both ways."""
    field = (lat * 1.0).expand_dims(time=[0]).transpose("time", "latitude")
    assert not np.isclose(
        weighted_mean(field).values[0], field.mean("latitude").values[0]
    )


def test_apply_eof_weights_preserves_shape(lat):
    field = xr.DataArray(
        np.ones((5, lat.size, 8)),
        dims=("time", "latitude", "longitude"),
        coords={"latitude": lat.latitude},
    )
    assert apply_eof_weights(field).shape == field.shape

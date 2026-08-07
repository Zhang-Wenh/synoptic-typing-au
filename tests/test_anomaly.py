"""Tests for seasonal cycle removal and detrending."""

import numpy as np
import pandas as pd
import xarray as xr

from src.preprocess.anomaly import anomaly, detrend, harmonic_climatology


def make_field(n_years=10, amplitude=5.0, trend_per_year=0.0, noise=0.0, seed=0):
    time = pd.date_range("1979-01-01", periods=n_years * 365, freq="1D")
    doy = time.dayofyear.values
    # Continuous in time. Using whole-year offsets would make the trend a step
    # function, which a linear fit cannot remove.
    years = (time - time[0]).days.values / 365.25

    signal = amplitude * np.sin(2 * np.pi * doy / 365.25) + trend_per_year * years
    if noise:
        signal = signal + np.random.default_rng(seed).normal(0, noise, signal.size)

    data = np.repeat(signal[:, None, None], 3, axis=1).repeat(4, axis=2)
    return xr.DataArray(
        data,
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": [-20, -30, -40], "longitude": [140, 145, 150, 155]},
        name="test_field",
    )


def test_climatology_recovers_a_pure_annual_cycle():
    field = make_field(amplitude=5.0)
    fitted = harmonic_climatology(field, n_harmonics=3)
    assert float(np.abs(field - fitted).max()) < 0.1


def test_anomaly_of_a_pure_cycle_is_near_zero():
    assert float(np.abs(anomaly(make_field())).max()) < 0.1


def test_anomaly_preserves_shape_and_dims():
    field = make_field()
    out = anomaly(field)
    assert out.shape == field.shape
    assert out.dims == field.dims


def test_anomaly_leaves_the_trend_in_place():
    """Removing the seasonal cycle must not also remove the trend."""
    field = make_field(trend_per_year=0.5)
    resid = anomaly(field)
    first = float(resid.isel(time=slice(0, 365)).mean())
    last = float(resid.isel(time=slice(-365, None)).mean())
    assert last - first > 3.0


def test_detrend_removes_a_linear_trend():
    field = make_field(amplitude=0.0, trend_per_year=0.5)
    assert float(np.abs(detrend(field)).max()) < 0.1


def test_detrend_preserves_variance_of_a_trendless_field():
    field = make_field(amplitude=0.0, noise=1.0)
    before = float(field.var())
    after = float(detrend(field).var())
    assert abs(after - before) / before < 0.05

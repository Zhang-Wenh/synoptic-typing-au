"""Tests for daily aggregation, deseasonalising and detrending.

Two things are guarded here that would otherwise fail silently:

  - accumulation type. A float32 reduction over a long record is biased, not
    noisy, and produces output that looks entirely reasonable.
  - the order of operations. Deseasonalising before daily averaging, or
    detrending before deseasonalising, both run without error and both give
    subtly wrong climatologies.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.preprocess.anomaly import (
    anomaly,
    daily_mean,
    detrend,
    harmonic_climatology,
    harmonic_design,
    standardise,
)


def make_field(
    n_years=5,
    amplitude=5.0,
    trend_per_year=0.0,
    diurnal=0.0,
    offset=1013.0,
    noise=0.0,
    dtype="float32",
    seed=0,
):
    """Synthetic 6-hourly field with a controllable cycle, trend and diurnal signal."""
    n = int(n_years * 365.25 * 4)
    time = pd.date_range("1979-01-01", periods=n, freq="6h")
    doy = time.dayofyear.values
    years = (time - time[0]).days.values / 365.25
    hours = time.hour.values

    signal = (
        offset
        + amplitude * np.sin(2 * np.pi * doy / 365.25)
        + trend_per_year * years
        + diurnal * np.sin(2 * np.pi * hours / 24)
    )
    if noise:
        signal = signal + np.random.default_rng(seed).normal(0, noise, signal.size)

    data = np.repeat(signal[:, None, None], 3, axis=1).repeat(4, axis=2)
    return xr.DataArray(
        data.astype(dtype),
        dims=("time", "latitude", "longitude"),
        coords={
            "time": time,
            "latitude": [-20.0, -30.0, -40.0],
            "longitude": [140.0, 145.0, 150.0, 155.0],
        },
        name="mslp",
    )


# --- daily aggregation ---------------------------------------------------

def test_daily_mean_collapses_four_steps_into_one():
    field = make_field(n_years=1)
    assert daily_mean(field).sizes["time"] == pytest.approx(
        field.sizes["time"] / 4, rel=0.02
    )


def test_daily_mean_removes_the_diurnal_cycle():
    """A pure diurnal signal must average away, leaving the annual cycle."""
    with_diurnal = daily_mean(make_field(diurnal=3.0))
    without = daily_mean(make_field(diurnal=0.0))
    assert float(np.abs(with_diurnal - without).max()) < 0.05


def test_daily_mean_promotes_to_float64():
    assert daily_mean(make_field(dtype="float32")).dtype == np.float64


def test_daily_mean_casts_before_reducing_not_after():
    """Casting the result would preserve precision already lost in the sum."""
    out = daily_mean(make_field(n_years=2, amplitude=0.0, offset=1e5))
    assert float(np.abs(out - 1e5).max()) < 1e-6


def test_daily_mean_preserves_spatial_dims():
    out = daily_mean(make_field())
    assert out.sizes["latitude"] == 3 and out.sizes["longitude"] == 4


# --- harmonic design -----------------------------------------------------

def test_design_has_two_terms_per_harmonic_plus_a_constant():
    assert harmonic_design(make_field(n_years=1).time, n_harmonics=3).sizes["term"] == 7


def test_design_constant_term_is_one():
    design = harmonic_design(make_field(n_years=1).time, n_harmonics=2)
    assert np.allclose(design.sel(term="const").values, 1.0)


def test_design_is_float64():
    assert harmonic_design(make_field(n_years=1).time).dtype == np.float64


# --- climatology and anomaly --------------------------------------------

def test_climatology_recovers_a_pure_annual_cycle():
    field = daily_mean(make_field(amplitude=5.0))
    assert float(np.abs(field - harmonic_climatology(field, 3)).max()) < 0.05


def test_anomaly_of_a_pure_cycle_is_near_zero():
    assert float(np.abs(anomaly(daily_mean(make_field()))).max()) < 0.05


def test_anomaly_leaves_the_trend_in_place():
    """Removing the seasonal cycle must not remove the trend as well."""
    out = anomaly(daily_mean(make_field(n_years=5, trend_per_year=0.5)))
    first = float(out.isel(time=slice(0, 365)).mean())
    last = float(out.isel(time=slice(-365, None)).mean())
    assert last - first > 1.5


def test_anomaly_preserves_shape_and_dims():
    field = daily_mean(make_field())
    out = anomaly(field)
    assert out.shape == field.shape and out.dims == field.dims


def test_anomaly_is_float64():
    assert anomaly(daily_mean(make_field())).dtype == np.float64


def test_climatology_is_smooth_on_a_noisy_record():
    """The reason for fitting harmonics rather than a day-of-year mean."""
    field = daily_mean(make_field(n_years=5, amplitude=5.0, noise=3.0))
    clim = harmonic_climatology(field, 3)
    first_year = clim.isel(time=slice(0, 365), latitude=0, longitude=0).values
    assert np.abs(np.diff(first_year, n=2)).max() < 0.05


def test_climatology_removes_the_cycle_and_nothing_else():
    """Three harmonics must not absorb day-to-day variance.

    The construction makes the expected answer exact. Noise of standard
    deviation 3 at 6-hourly resolution has variance 9; averaging four steps to
    a daily value leaves 9/4. If the harmonic fit took any synoptic variance
    with it, the anomaly variance would come out below that.
    """
    noise, per_day = 3.0, 4
    field = daily_mean(make_field(n_years=5, amplitude=5.0, noise=noise))
    expected = noise**2 / per_day
    assert float(anomaly(field).var()) == pytest.approx(expected, rel=0.1)


def test_seasonal_cycle_is_the_dominant_removed_component():
    """The cycle accounts for the rest of the variance, to within the noise."""
    amplitude, noise, per_day = 5.0, 3.0, 4
    field = daily_mean(make_field(n_years=5, amplitude=amplitude, noise=noise))
    removed = float(field.var()) - float(anomaly(field).var())
    assert removed == pytest.approx(amplitude**2 / 2, rel=0.1)


# --- detrending ----------------------------------------------------------

def test_detrend_removes_a_linear_trend():
    out = detrend(daily_mean(make_field(amplitude=0.0, trend_per_year=0.5)))
    assert float(np.abs(out - out.mean()).max()) < 0.05


def test_detrend_preserves_variance_of_a_trendless_field():
    field = daily_mean(make_field(amplitude=0.0, noise=1.0))
    before = float(field.var())
    assert abs(float(detrend(field).var()) - before) / before < 0.05


def test_detrend_is_float64():
    assert detrend(daily_mean(make_field())).dtype == np.float64


# --- accumulation precision ---------------------------------------------

def test_float64_climatology_is_unbiased_on_a_float32_record():
    """The defect this module was rewritten for.

    A constant float32 field of order 1e5 over a long record must give a
    climatology equal to that constant. A float32 accumulation drifts low by
    a few parts in ten thousand, which on a 1013 hPa field is around 0.2 hPa:
    small against a 10 hPa synoptic anomaly, but systematic rather than noisy.
    """
    field = make_field(n_years=10, amplitude=0.0, offset=101300.0, dtype="float32")
    clim = harmonic_climatology(daily_mean(field), 3)
    assert float(np.abs(clim - 101300.0).max()) < 0.01


def test_anomaly_of_a_constant_float32_field_is_zero_not_biased():
    field = make_field(n_years=10, amplitude=0.0, offset=101300.0, dtype="float32")
    assert float(np.abs(anomaly(daily_mean(field))).max()) < 0.01


def test_standardise_gives_unit_variance():
    out = standardise(daily_mean(make_field(amplitude=0.0, noise=2.0)))
    assert float(out.std("time").mean()) == pytest.approx(1.0, rel=1e-6)

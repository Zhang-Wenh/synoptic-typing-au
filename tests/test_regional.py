"""Tests for reducing gridded rainfall to one regional daily series.

Each of the three steps here fails silently if wrong: an unmasked ocean cell
still produces a plausible number, an unweighted mean is off by a percent or
two, and a day-alignment error attributes rain to the wrong day's circulation
without changing anything about how the output looks.
"""

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.attribute.regional import (
    align_to,
    hot_day_indicator,
    regional_mean,
    shift_to_utc_day,
    subset,
)

TARGET = {"lat_north": -33.0, "lat_south": -40.0,
          "lon_west": 140.0, "lon_east": 150.0}


def make_grid(n_time=20, descending=True, lat_name="lat", ocean=None, value=1.0):
    lat = np.arange(-30.0, -44.0, -1.0) if descending else np.arange(-44.0, -30.0, 1.0)
    lon = np.arange(136.0, 155.0, 1.0)
    time = pd.date_range("1979-01-01", periods=n_time, freq="1D")

    data = np.full((n_time, lat.size, lon.size), value)
    da = xr.DataArray(
        data, dims=("time", lat_name, "lon"),
        coords={"time": time, lat_name: lat, "lon": lon}, name="daily_rain",
    )
    if ocean is not None:
        da = da.where(~ocean)
    return da


# --- subsetting ----------------------------------------------------------

def test_subset_cuts_to_the_target_region():
    out = subset(make_grid(), TARGET)
    assert float(out["lat"].max()) <= -33.0
    assert float(out["lat"].min()) >= -40.0
    assert float(out["lon"].min()) >= 140.0


def test_subset_handles_an_ascending_latitude_axis():
    """SILO and ERA5 do not agree on which way latitude runs."""
    a = subset(make_grid(descending=True), TARGET)
    b = subset(make_grid(descending=False), TARGET)
    assert a.sizes["lat"] == b.sizes["lat"]


def test_subset_accepts_either_coordinate_name():
    out = subset(make_grid(lat_name="latitude"), TARGET)
    assert out.sizes["latitude"] > 0


# --- masking and weighting ----------------------------------------------

def test_regional_mean_of_a_uniform_field_is_that_value():
    out = regional_mean(subset(make_grid(value=4.0), TARGET))
    assert out.dims == ("time",)
    assert np.allclose(out.values, 4.0)


def test_ocean_cells_are_excluded():
    """SILO extrapolates over water; those values are not observations."""
    grid = make_grid(value=1.0)
    ocean = xr.zeros_like(grid.isel(time=0), dtype=bool)
    ocean[{"lat": slice(0, 3)}] = True
    wet = make_grid(value=1.0).where(~ocean, 99.0)
    wet = wet.where(~ocean)

    out = regional_mean(subset(wet, TARGET))
    assert float(out.max()) == pytest.approx(1.0)


def test_high_latitudes_are_downweighted():
    """Cells shrink poleward, so an unweighted mean over-counts the south."""
    grid = subset(make_grid(value=1.0), TARGET)
    lat = grid["lat"]
    graded = grid * xr.where(lat < -36.5, 10.0, 0.0)

    weighted = float(regional_mean(graded).mean())
    plain = float(graded.mean())
    assert weighted < plain


def test_partial_missing_cells_are_reported(caplog):
    """A cell present on some days changes the effective area day to day."""
    grid = subset(make_grid(n_time=10, value=1.0), TARGET)
    patchy = grid.where(~((grid["time"].dt.day < 5) & (grid["lat"] < -38)))

    with caplog.at_level(logging.WARNING):
        regional_mean(patchy)
    assert any("missing on some days" in r.message for r in caplog.records)


# --- day alignment -------------------------------------------------------

def test_shift_moves_labels_back_by_the_given_hours():
    """SILO totals run to 9am local; ERA5 days are UTC."""
    series = regional_mean(subset(make_grid(n_time=5), TARGET))
    shifted = shift_to_utc_day(series, hours=-9)
    delta = shifted["time"].values[0] - series["time"].values[0]
    assert delta == np.timedelta64(-1, "D")


def test_shift_keeps_every_day():
    series = regional_mean(subset(make_grid(n_time=30), TARGET))
    assert shift_to_utc_day(series).sizes["time"] == 30


def test_shift_records_what_it_did():
    series = regional_mean(subset(make_grid(n_time=5), TARGET))
    assert "day_alignment" in shift_to_utc_day(series).attrs


def test_shift_of_zero_changes_nothing():
    series = regional_mean(subset(make_grid(n_time=5), TARGET))
    assert shift_to_utc_day(series, hours=0)["time"].equals(series["time"])


# --- aligning to the circulation record ---------------------------------

def test_align_keeps_only_shared_days():
    time = pd.date_range("1979-01-01", periods=100, freq="1D")
    series = xr.DataArray(np.ones(100), dims="time", coords={"time": time})
    reference = xr.DataArray(
        np.ones(50), dims="time", coords={"time": time[25:75]}
    )
    assert align_to(series, reference).sizes["time"] == 50


def test_align_is_an_inner_join_not_a_reindex():
    """Reindexing would fill gaps with NaN and quietly shrink the samples."""
    time = pd.date_range("1979-01-01", periods=50, freq="1D")
    series = xr.DataArray(np.ones(30), dims="time", coords={"time": time[:30]})
    reference = xr.DataArray(np.ones(50), dims="time", coords={"time": time})

    out = align_to(series, reference)
    assert out.sizes["time"] == 30
    assert not bool(out.isnull().any())


def test_align_raises_when_nothing_overlaps():
    a = pd.date_range("1979-01-01", periods=10, freq="1D")
    b = pd.date_range("1999-01-01", periods=10, freq="1D")
    with pytest.raises(ValueError, match="no overlapping days"):
        align_to(
            xr.DataArray(np.ones(10), dims="time", coords={"time": a}),
            xr.DataArray(np.ones(10), dims="time", coords={"time": b}),
        )


# --- hot-day indicator ---------------------------------------------------

def make_temp(n_years=10, amplitude=10.0, noise=3.0, trend_per_year=0.0, seed=0):
    """Daily maximum temperature with an annual cycle peaking in January."""
    rng = np.random.default_rng(seed)
    time = pd.date_range("1979-01-01", periods=int(n_years * 365.25), freq="1D")
    doy = time.dayofyear.values
    years = (time - time[0]).days.values / 365.25
    values = (
        25.0
        + amplitude * np.sin(2 * np.pi * (doy - 330) / 365.25)
        + trend_per_year * years
        + rng.normal(0, noise, time.size)
    )
    return xr.DataArray(values, dims="time", coords={"time": time}, name="max_temp")


def test_hot_days_are_the_requested_fraction():
    assert float(hot_day_indicator(make_temp(), percentile=90.0).mean()) == pytest.approx(
        0.10, abs=0.01
    )


def test_indicator_is_zero_or_one():
    out = hot_day_indicator(make_temp())
    assert set(np.unique(out.values)) <= {0.0, 1.0}


def test_season_aware_threshold_gives_both_seasons_hot_days():
    """A single annual threshold leaves the cool season with almost none.

    Southeast Australian summer maxima sit roughly seven degrees above winter
    ones, so an annual 90th percentile is exceeded almost only in summer and
    there is nothing left to decompose in the cool season.
    """
    temp = make_temp(n_years=15)
    month = temp["time"].dt.month
    cool = month.isin([4, 5, 6, 7, 8, 9, 10])

    aware = hot_day_indicator(temp, season_aware=True)
    plain = hot_day_indicator(temp, season_aware=False)

    assert float(aware.sel(time=cool).mean()) == pytest.approx(0.10, abs=0.02)
    assert float(plain.sel(time=cool).mean()) < 0.05


def test_season_aware_thresholds_differ_between_halves():
    out = hot_day_indicator(make_temp(n_years=15))
    assert out.attrs["warm_threshold"] > out.attrs["cool_threshold"]


def test_warming_raises_the_hot_day_count_in_later_years():
    """The indicator must respond to a trend, or there is nothing to attribute."""
    temp = make_temp(n_years=30, trend_per_year=0.05, noise=2.0)
    out = hot_day_indicator(temp)
    first = float(out.isel(time=slice(0, 3650)).mean())
    last = float(out.isel(time=slice(-3650, None)).mean())
    assert last > first


def test_indicator_records_its_definition():
    assert "percentile" in hot_day_indicator(make_temp()).attrs["definition"]


def test_a_higher_percentile_selects_fewer_days():
    temp = make_temp(n_years=15)
    assert float(hot_day_indicator(temp, percentile=99.0).mean()) < float(
        hot_day_indicator(temp, percentile=90.0).mean()
    )

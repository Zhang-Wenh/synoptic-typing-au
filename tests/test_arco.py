"""Tests for ARCO slicing helpers, run against a synthetic store.

The latitude-order case matters: ERA5 runs north to south, so a slice written
the intuitive way (south first) returns an empty array rather than an error.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.io.arco import find_variables, slice_year, subset_domain, subset_hours

DOMAIN = {
    "lat_north": -10.0,
    "lat_south": -60.0,
    "lon_west": 90.0,
    "lon_east": 180.0,
}


def make_store(descending_lat=True):
    lat = np.arange(90, -90.25, -0.25) if descending_lat else np.arange(-90, 90.25, 0.25)
    lon = np.arange(0, 360, 0.25)
    time = pd.date_range("1979-01-01", "1979-01-10", freq="1h")
    data = np.zeros((time.size, lat.size, lon.size), dtype="float32")
    return xr.Dataset(
        {"mean_sea_level_pressure": (("time", "latitude", "longitude"), data)},
        coords={"time": time, "latitude": lat, "longitude": lon},
    )


def test_domain_slice_is_not_empty_on_descending_latitude():
    da = subset_domain(make_store(True)["mean_sea_level_pressure"], DOMAIN)
    assert da.sizes["latitude"] > 0
    assert da.sizes["longitude"] > 0


def test_domain_slice_is_not_empty_on_ascending_latitude():
    da = subset_domain(make_store(False)["mean_sea_level_pressure"], DOMAIN)
    assert da.sizes["latitude"] > 0


def test_domain_slice_covers_the_requested_box():
    da = subset_domain(make_store()["mean_sea_level_pressure"], DOMAIN)
    assert da.latitude.min() >= DOMAIN["lat_south"]
    assert da.latitude.max() <= DOMAIN["lat_north"]
    assert da.longitude.min() >= DOMAIN["lon_west"]
    assert da.longitude.max() <= DOMAIN["lon_east"]


def test_domain_slice_size_matches_quarter_degree_spacing():
    da = subset_domain(make_store()["mean_sea_level_pressure"], DOMAIN)
    assert da.sizes["latitude"] == 201
    assert da.sizes["longitude"] == 361


def test_hour_selection_keeps_only_synoptic_hours():
    da = subset_hours(make_store()["mean_sea_level_pressure"], [0, 6, 12, 18])
    assert set(np.unique(da.time.dt.hour.values)) == {0, 6, 12, 18}


def test_hour_selection_reduces_count_by_six():
    full = make_store()["mean_sea_level_pressure"]
    assert subset_hours(full, [0, 6, 12, 18]).sizes["time"] == pytest.approx(
        full.sizes["time"] / 6, rel=0.05
    )


def test_slice_year_selects_by_date_not_index():
    ds = make_store()
    da = slice_year(ds, "mean_sea_level_pressure", 1979, DOMAIN, [0, 12])
    assert set(pd.to_datetime(da.time.values).year) == {1979}


def test_slice_year_outside_coverage_is_empty_not_an_error():
    ds = make_store()
    da = slice_year(ds, "mean_sea_level_pressure", 1999, DOMAIN, [0])
    assert da.sizes["time"] == 0


def test_find_variables_is_case_insensitive():
    ds = make_store()
    assert find_variables(ds, ["SEA_LEVEL"]) == ["mean_sea_level_pressure"]


def test_find_variables_returns_empty_for_no_match():
    assert find_variables(make_store(), ["nonexistent"]) == []

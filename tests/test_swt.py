"""Tests for reading the published Australian Synoptic Weather Types.

Two failures here would be silent and would corrupt everything downstream: a
day-boundary error attributes each day's rain to the wrong day's circulation,
and an ordering that depends on which labels happen to be present renumbers
the types whenever a subset excludes a rare one.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.io.swt import (
    REGIMES,
    encode,
    load,
    open_labels,
    to_daily_index,
    to_regimes,
    type_order,
)

ORDER = [
    "WH-A", "WH-B", "WH-C", "WH-D", "CH-A", "CH-B",
    "EH-A", "EH-B", "EH-C", "EH-D", "EH-E",
    "TH-A", "TH-B", "TH-C", "FH-A", "FH-B", "FH-C",
    "WCT-A", "WCT-B",
    "COL-A", "COL-B", "COL-C", "COL-D", "COL-E", "COL-F",
    "AM-A", "AM-B", "AM-C", "AM-D", "AM-E",
]


@pytest.fixture
def label_file(tmp_path):
    """A file shaped like the distributed one: 12 UTC stamps, string labels."""
    n = 400
    time = pd.date_range("1979-01-01 12:00", periods=n, freq="1D")
    rng = np.random.default_rng(0)
    values = rng.choice(ORDER, n)

    ds = xr.Dataset(
        {"assigned_SWT": ("time", values)},
        coords={"time": time, "SWTs": ORDER},
        attrs={"reference": "10.1029/2025JD043873", "data source": "ERA5"},
    )
    path = tmp_path / "SWT_climatology.nc"
    ds.to_netcdf(path)
    return path


# --- reading -------------------------------------------------------------

def test_open_returns_one_label_per_day(label_file):
    labels = open_labels(label_file)
    assert labels.dims == ("time",)
    assert labels.sizes["time"] == 400


def test_open_carries_the_reference(label_file):
    assert "2025JD043873" in open_labels(label_file).attrs["reference"]


def test_open_rejects_a_file_without_labels(tmp_path):
    path = tmp_path / "wrong.nc"
    xr.Dataset({"something_else": ("time", [1, 2])},
               coords={"time": pd.date_range("1979-01-01", periods=2)}).to_netcdf(path)
    with pytest.raises(KeyError, match="assigned_SWT"):
        open_labels(path)


def test_type_order_comes_from_the_file(label_file):
    """Not hardcoded, so a future version with different types still works."""
    assert type_order(label_file) == ORDER


# --- regimes -------------------------------------------------------------

def test_regimes_are_the_prefix_of_the_type_name(label_file):
    regimes = to_regimes(open_labels(label_file))
    assert set(np.unique(regimes.values)) <= set(REGIMES)


def test_every_type_maps_to_a_known_regime():
    for name in ORDER:
        assert name.split("-")[0] in REGIMES


def test_regimes_preserve_the_time_axis(label_file):
    labels = open_labels(label_file)
    assert to_regimes(labels).time.equals(labels.time)


# --- encoding ------------------------------------------------------------

def test_encoding_follows_the_given_order():
    labels = xr.DataArray(
        np.array(["EH", "WH", "AM"]), dims="time",
        coords={"time": pd.date_range("1979-01-01", periods=3)},
    )
    codes = encode(labels, REGIMES)
    assert list(codes.values) == [REGIMES.index(n) for n in ("EH", "WH", "AM")]


def test_encoding_is_stable_when_a_type_is_absent():
    """The defect this fixed order exists for.

    Deriving codes from the labels present would renumber everything whenever
    a seasonal subset excluded a rare type -- so the same integer would mean
    different things in the cool and warm season tables.
    """
    full = xr.DataArray(
        np.array(REGIMES), dims="time",
        coords={"time": pd.date_range("1979-01-01", periods=len(REGIMES))},
    )
    without_monsoon = full.isel(time=slice(0, len(REGIMES) - 1))

    a = encode(full, REGIMES)
    b = encode(without_monsoon, REGIMES)
    assert list(a.values[: b.sizes["time"]]) == list(b.values)


def test_encoding_records_what_the_codes_mean():
    labels = xr.DataArray(
        np.array(["WH", "AM"]), dims="time",
        coords={"time": pd.date_range("1979-01-01", periods=2)},
    )
    assert "0=WH" in encode(labels, REGIMES).attrs["codes"]


def test_encoding_rejects_an_unknown_label():
    labels = xr.DataArray(
        np.array(["WH", "NOPE"]), dims="time",
        coords={"time": pd.date_range("1979-01-01", periods=2)},
    )
    with pytest.raises(ValueError, match="not in the given order"):
        encode(labels, REGIMES)


# --- day boundaries ------------------------------------------------------

def test_time_of_day_is_dropped(label_file):
    """Downstream joins are on the calendar day; a 12:00 stamp never matches."""
    daily = to_daily_index(open_labels(label_file))
    assert set(pd.DatetimeIndex(daily.time.values).hour) == {0}


def test_shift_moves_labels_by_whole_days(label_file):
    labels = open_labels(label_file)
    shifted = to_daily_index(labels, shift_days=-1)
    plain = to_daily_index(labels)
    delta = shifted.time.values[0] - plain.time.values[0]
    assert delta == np.timedelta64(-1, "D")


def test_shift_does_not_reorder_the_labels(label_file):
    """Only the stamps move; each day keeps the type it was assigned."""
    labels = open_labels(label_file)
    assert list(to_daily_index(labels, -1).values) == list(labels.values)


def test_shift_records_itself(label_file):
    assert "day_shift" in to_daily_index(open_labels(label_file), -1).attrs


def test_zero_shift_leaves_the_dates_alone(label_file):
    labels = to_daily_index(open_labels(label_file))
    original = pd.DatetimeIndex(open_labels(label_file).time.values).normalize()
    assert np.array_equal(pd.DatetimeIndex(labels.time.values), original)


# --- the whole path ------------------------------------------------------

def test_load_regimes_gives_eight_codes(label_file):
    codes, names = load(label_file, grouping="regime")
    assert names == REGIMES
    assert set(np.unique(codes.values)) <= set(range(8))


def test_load_types_gives_thirty_codes(label_file):
    codes, names = load(label_file, grouping="type")
    assert len(names) == 30
    assert set(np.unique(codes.values)) <= set(range(30))


def test_load_restricts_to_the_requested_period(label_file):
    codes, _ = load(label_file, start="1979-06-01", end="1979-08-31")
    times = pd.DatetimeIndex(codes.time.values)
    assert times.min() >= pd.Timestamp("1979-06-01")
    assert times.max() <= pd.Timestamp("1979-08-31")


def test_load_raises_on_an_empty_period(label_file):
    with pytest.raises(ValueError, match="no labels"):
        load(label_file, start="1900-01-01", end="1900-12-31")


def test_load_rejects_an_unknown_grouping(label_file):
    with pytest.raises(ValueError, match="regime.*type"):
        load(label_file, grouping="cluster")


def test_regime_codes_are_coarser_than_type_codes(label_file):
    """The same days, grouped two ways: regimes must merge types, not split."""
    types, _ = load(label_file, grouping="type")
    regimes, _ = load(label_file, grouping="regime")
    assert len(np.unique(regimes.values)) < len(np.unique(types.values))
    assert regimes.time.equals(types.time)

"""Tests for the CDS fetch path.

Nothing here touches the network. The request builder and the normalisation
step are pure functions, and normalisation is where the real risk sits: CDS
NetCDF differs from ARCO Zarr in four ways at once, and getting any of them
wrong produces a file that opens fine and is silently inconsistent with the
years fetched from ARCO.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.io.cds import G0, build_request, normalise

DOMAIN = {
    "lat_north": -10.0,
    "lat_south": -60.0,
    "lon_west": 90.0,
    "lon_east": 180.0,
}
HOURS = [0, 6, 12, 18]


def make_cds_output(
    time_name="valid_time",
    with_level=True,
    with_expver=False,
    ascending_lat=False,
):
    """A synthetic file shaped like current CDS NetCDF output."""
    lat = np.arange(-60.0, -9.75, 0.25) if ascending_lat else np.arange(-10.0, -60.25, -0.25)
    lon = np.arange(90.0, 180.25, 0.25)
    time = pd.date_range("1979-01-01", periods=8, freq="6h")

    shape = [time.size, lat.size, lon.size]
    dims = [time_name, "latitude", "longitude"]
    coords = {time_name: time, "latitude": lat, "longitude": lon}

    if with_level:
        dims.insert(1, "pressure_level")
        shape.insert(1, 1)
        coords["pressure_level"] = [500]
    if with_expver:
        dims.insert(1, "expver")
        shape.insert(1, 2)
        coords["expver"] = ["0001", "0005"]

    # Around 5500 m at 500 hPa, expressed as geopotential in m2 s-2.
    data = np.full(shape, 5500.0 * G0, dtype="float32")
    ds = xr.Dataset({"z": (dims, data)}, coords=coords)
    ds["z"].attrs = {"units": "m**2 s**-2", "long_name": "Geopotential"}
    return ds


def test_request_puts_level_and_area_on_the_server():
    """The whole reason for using CDS over ARCO."""
    req = build_request("geopotential", 500, 1979, HOURS, DOMAIN)
    assert req["pressure_level"] == ["500"]
    assert req["area"] == [-10.0, 90.0, -60.0, 180.0]


def test_request_area_order_is_north_west_south_east():
    area = build_request("geopotential", 500, 1979, HOURS, DOMAIN)["area"]
    north, west, south, east = area
    assert north > south
    assert west < east


def test_request_times_match_the_configured_hours():
    req = build_request("geopotential", 500, 1979, [0, 12], DOMAIN)
    assert req["time"] == ["00:00", "12:00"]


def test_request_covers_a_whole_year():
    req = build_request("geopotential", 500, 1979, HOURS, DOMAIN)
    assert len(req["month"]) == 12
    assert len(req["day"]) == 31


def test_request_converts_longitudes_above_180():
    domain = dict(DOMAIN, lon_east=200.0)
    assert build_request("geopotential", 500, 1979, HOURS, domain)["area"][3] == -160.0


def test_normalise_renames_valid_time_to_time():
    ds = normalise(make_cds_output())
    assert "time" in ds.dims
    assert "valid_time" not in ds.dims


def test_normalise_drops_the_singleton_level_dimension():
    ds = normalise(make_cds_output())
    assert "pressure_level" not in ds.dims


def test_normalise_renames_z_to_geopotential():
    ds = normalise(make_cds_output())
    assert "geopotential" in ds.data_vars
    assert "z" not in ds.data_vars


def test_normalise_converts_geopotential_to_height():
    """5500 m expressed as geopotential must come back as 5500 m.

    Checked on a single value, not a mean. Summing half a million float32
    values of order 5e4 loses precision: the running total passes the point
    where float32 spacing exceeds the increment, and the mean comes back low
    by a fraction of a percent. That is a property of the reduction, not of
    the conversion under test.
    """
    ds = normalise(make_cds_output())
    assert float(ds["geopotential"].values.flat[0]) == pytest.approx(5500.0, rel=1e-5)
    assert ds["geopotential"].attrs["units"] == "m"


def test_normalise_can_leave_geopotential_unconverted():
    ds = normalise(make_cds_output(), to_height=False)
    value = float(ds["geopotential"].values.flat[0])
    assert value == pytest.approx(5500.0 * G0, rel=1e-5)


def test_float64_accumulation_is_exact_on_a_long_float32_record():
    """Guard the numerical trap this file exposed.

    Summing a long float32 record of order 5e4 loses precision once the
    running total passes the point where float32 spacing exceeds the
    increment. How much is lost depends on the summation algorithm, so the
    size of the error is not asserted here. What is asserted is the property
    the project relies on: an explicit float64 accumulation is exact.

    Every climatology and area mean over the 68,000-step record must pass
    dtype="float64" for this reason. MSLP anomalies are a few hPa; a relative
    error of a fraction of a percent on a 1013 hPa field is the same size as
    the signal.
    """
    ds = normalise(make_cds_output())
    assert float(ds["geopotential"].mean(dtype="float64")) == pytest.approx(
        5500.0, rel=1e-6
    )


def test_float32_accumulation_is_measurably_worse():
    """Show the loss directly, using a strictly sequential accumulation.

    numpy's own sum() uses pairwise summation and stays accurate, which is why
    the error only appears in some reductions. A cumulative sum accumulates in
    order and exposes it: over 200,000 values of order 5e4 the running total
    drifts by about 2 parts in 10,000.
    """
    n = 200_000
    values = np.full(n, 5500.0 * G0, dtype="float32")
    sequential = float(np.cumsum(values, dtype="float32")[-1])
    exact = 5500.0 * G0 * n
    assert abs(sequential - exact) / exact > 1e-4


def test_normalise_collapses_expver():
    ds = normalise(make_cds_output(with_expver=True))
    assert "expver" not in ds.dims


def test_normalise_makes_latitude_descending():
    """ARCO-derived files run north to south. Both paths must agree."""
    ds = normalise(make_cds_output(ascending_lat=True))
    assert ds.latitude.values[0] > ds.latitude.values[-1]


def test_normalise_leaves_descending_latitude_alone():
    ds = normalise(make_cds_output(ascending_lat=False))
    assert ds.latitude.values[0] == pytest.approx(-10.0)


def test_normalise_output_matches_the_arco_grid():
    ds = normalise(make_cds_output())
    assert ds.sizes["latitude"] == 201
    assert ds.sizes["longitude"] == 361


def test_normalise_records_its_source():
    assert "CDS" in normalise(make_cds_output()).attrs["source"]

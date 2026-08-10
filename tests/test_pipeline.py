"""End-to-end tests for the preprocessing pipeline.

Written against real Zarr files on a temporary directory rather than mocks:
the failure modes worth catching here are about how years are found, opened
and concatenated, and a mock would not exercise any of them.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.preprocess import pipeline


def write_year(root, key, year, varname="mslp", offset=101300.0, amplitude=500.0,
               scalar_coord=None, lat_dtype="float32"):
    """Write one year of synthetic 6-hourly data in the raw layout.

    `scalar_coord` and `lat_dtype` reproduce the inconsistencies that appear
    when the fetch code changes partway through a multi-year download.
    """
    time = pd.date_range(f"{year}-01-01", f"{year}-12-31 18:00", freq="6h")
    doy = time.dayofyear.values
    signal = offset + amplitude * np.sin(2 * np.pi * doy / 365.25)
    data = np.repeat(signal[:, None, None], 4, axis=1).repeat(5, axis=2)

    da = xr.DataArray(
        data.astype("float32"),
        dims=("time", "latitude", "longitude"),
        coords={
            "time": time,
            "latitude": np.array([-10.0, -20.0, -30.0, -40.0], dtype=lat_dtype),
            "longitude": [90.0, 110.0, 130.0, 150.0, 170.0],
        },
        name=varname,
    )
    if scalar_coord is not None:
        da = da.assign_coords(**scalar_coord)
    dest = root / key / f"{key}_{year}.zarr"
    dest.parent.mkdir(parents=True, exist_ok=True)
    da.to_dataset().to_zarr(dest, mode="w", consolidated=True)
    return dest


@pytest.fixture(scope="module")
def raw_root(tmp_path_factory):
    """Five years of raw input, written once for the whole module.

    Every test that uses this reads only; the tests that write go to their own
    tmp_path. Rebuilding it per test cost about a minute of suite time.
    """
    root = tmp_path_factory.mktemp("raw")
    for year in range(1979, 1984):
        write_year(root, "mslp", year)
    return root


# --- finding years -------------------------------------------------------

def test_year_paths_returns_every_year_in_order(raw_root):
    paths = pipeline.year_paths(raw_root, "mslp", 1979, 1983)
    assert len(paths) == 5
    assert [p.name for p in paths] == sorted(p.name for p in paths)


def test_year_paths_raises_on_a_gap(raw_root):
    """A missing year would otherwise leave a silent hole in the climatology."""
    with pytest.raises(FileNotFoundError, match="1984"):
        pipeline.year_paths(raw_root, "mslp", 1979, 1984)


def test_year_paths_names_the_missing_years(raw_root):
    with pytest.raises(FileNotFoundError) as exc:
        pipeline.year_paths(raw_root, "mslp", 1977, 1983)
    assert "1977" in str(exc.value) and "1978" in str(exc.value)


# --- opening and concatenating ------------------------------------------

def test_open_years_concatenates_along_time(raw_root):
    da = pipeline.open_years(pipeline.year_paths(raw_root, "mslp", 1979, 1983))
    assert da.sizes["time"] > 7000
    assert da.indexes["time"].is_monotonic_increasing


def test_open_years_spans_the_requested_period(raw_root):
    da = pipeline.open_years(pipeline.year_paths(raw_root, "mslp", 1979, 1983))
    assert da.indexes["time"][0].year == 1979
    assert da.indexes["time"][-1].year == 1983


def test_open_years_infers_the_variable_name(raw_root):
    da = pipeline.open_years(pipeline.year_paths(raw_root, "mslp", 1979, 1979))
    assert da.name == "mslp"


def test_open_years_preserves_the_grid(raw_root):
    da = pipeline.open_years(pipeline.year_paths(raw_root, "mslp", 1979, 1983))
    assert da.sizes["latitude"] == 4 and da.sizes["longitude"] == 5


# --- processing ----------------------------------------------------------

def test_process_produces_daily_output(raw_root):
    da = pipeline.open_years(pipeline.year_paths(raw_root, "mslp", 1979, 1983))
    out = pipeline.process(da)
    assert out.sizes["time"] == pytest.approx(da.sizes["time"] / 4, rel=0.02)


def test_process_removes_the_seasonal_cycle(raw_root):
    """Input is a pure annual cycle on a 101300 Pa base; output must be near zero."""
    da = pipeline.open_years(pipeline.year_paths(raw_root, "mslp", 1979, 1983))
    assert float(np.abs(pipeline.process(da)).max()) < 5.0


def test_process_preserves_the_variable_name(raw_root):
    """Arithmetic and xr.dot both drop it, and the write step needs it."""
    da = pipeline.open_years(pipeline.year_paths(raw_root, "mslp", 1979, 1983))
    assert pipeline.process(da).name == "mslp"


def test_process_output_is_float64(raw_root):
    da = pipeline.open_years(pipeline.year_paths(raw_root, "mslp", 1979, 1983))
    assert pipeline.process(da).dtype == np.float64


def test_process_order_matters_for_the_diurnal_cycle(raw_root):
    """Deseasonalising before daily averaging would fit harmonics to sub-daily
    structure. Averaging first is what the pipeline does; assert the daily step
    happens before the climatology by checking the output length."""
    da = pipeline.open_years(pipeline.year_paths(raw_root, "mslp", 1979, 1983))
    out = pipeline.process(da)
    assert out.sizes["time"] < da.sizes["time"]


# --- writing -------------------------------------------------------------

@pytest.mark.slow
def test_run_writes_a_readable_zarr(raw_root, tmp_path):
    work = tmp_path / "work"
    dest = pipeline.run(raw_root, work, "mslp", 1979, 1983)
    assert dest.exists()
    back = xr.open_zarr(dest, consolidated=True)["mslp"]
    assert back.sizes["time"] > 1800


@pytest.mark.slow
def test_run_records_its_provenance(raw_root, tmp_path):
    dest = pipeline.run(raw_root, tmp_path / "work", "mslp", 1979, 1983)
    attrs = xr.open_zarr(dest, consolidated=True)["mslp"].attrs
    assert "harmonics" in attrs["preprocessing"]
    assert attrs["period"] == "1979-1983"


@pytest.mark.slow
def test_run_leaves_no_temporary_directory(raw_root, tmp_path):
    work = tmp_path / "work"
    pipeline.run(raw_root, work, "mslp", 1979, 1983)
    assert not any(p.name.endswith(".tmp") for p in work.iterdir())


@pytest.mark.slow
def test_run_is_rerunnable(raw_root, tmp_path):
    """Re-running must replace the output cleanly, not fail or append."""
    work = tmp_path / "work"
    first = pipeline.run(raw_root, work, "mslp", 1979, 1983)
    n_first = xr.open_zarr(first, consolidated=True).sizes["time"]
    second = pipeline.run(raw_root, work, "mslp", 1979, 1983)
    assert xr.open_zarr(second, consolidated=True).sizes["time"] == n_first


@pytest.mark.slow
def test_run_without_detrending_keeps_the_trend(raw_root, tmp_path):
    da = pipeline.open_years(pipeline.year_paths(raw_root, "mslp", 1979, 1983))
    trended = da + xr.DataArray(
        np.linspace(0, 100, da.sizes["time"]), dims="time",
        coords={"time": da.time},
    )
    with_detrend = pipeline.process(trended, do_detrend=True)
    without = pipeline.process(trended, do_detrend=False)
    assert float(without.max()) - float(without.min()) > float(
        with_detrend.max()
    ) - float(with_detrend.min())


# --- inconsistency across years -----------------------------------------

@pytest.fixture(scope="module")
def mixed_root(tmp_path_factory):
    """Years written by two different versions of the fetch code.

    This is not hypothetical: a level coordinate was stripped from the CDS
    reader partway through a 47-year download, so one year lacked a scalar
    coordinate that the other forty-six carried.
    """
    root = tmp_path_factory.mktemp("mixed")
    for year in (1979, 1980):
        write_year(root, "z", year, scalar_coord={"level": 500}, lat_dtype="float64")
    for year in (1981, 1982, 1983):
        write_year(root, "z", year)
    return root


def test_open_years_survives_a_scalar_coord_on_some_years_only(mixed_root):
    da = pipeline.open_years(pipeline.year_paths(mixed_root, "z", 1979, 1983))
    assert "level" not in da.coords


def test_open_years_gives_one_coordinate_dtype_across_mixed_years(mixed_root):
    da = pipeline.open_years(pipeline.year_paths(mixed_root, "z", 1979, 1983))
    assert da.latitude.dtype == np.dtype("float64")


def test_open_years_loses_no_time_steps_to_alignment(mixed_root):
    """Aligning mismatched coords would drop or NaN-fill; neither is acceptable."""
    da = pipeline.open_years(pipeline.year_paths(mixed_root, "z", 1979, 1983))
    assert da.sizes["time"] > 7000
    assert int(da.isnull().sum()) == 0


def test_two_variables_from_mixed_sources_end_up_aligned(mixed_root, raw_root, tmp_path):
    """The property the whole stage depends on: MSLP and Z500 must share coords."""
    mslp = pipeline.process(
        pipeline.open_years(pipeline.year_paths(raw_root, "mslp", 1979, 1983))
    )
    z = pipeline.process(
        pipeline.open_years(pipeline.year_paths(mixed_root, "z", 1979, 1983))
    )
    assert mslp.time.equals(z.time)
    assert mslp.latitude.equals(z.latitude)
    assert mslp.longitude.equals(z.longitude)


def test_open_years_raises_on_a_genuine_grid_mismatch(tmp_path):
    """A real difference must be an error, not a silent NaN-filled union."""
    root = tmp_path / "bad"
    write_year(root, "mslp", 1979)
    time = pd.date_range("1980-01-01", "1980-12-31 18:00", freq="6h")
    xr.DataArray(
        np.ones((time.size, 4, 5), dtype="float32"),
        dims=("time", "latitude", "longitude"),
        coords={
            "time": time,
            "latitude": [-11.0, -21.0, -31.0, -41.0],
            "longitude": [90.0, 110.0, 130.0, 150.0, 170.0],
        },
        name="mslp",
    ).to_dataset().to_zarr(root / "mslp" / "mslp_1980.zarr", mode="w", consolidated=True)

    with pytest.raises(ValueError):
        pipeline.open_years(pipeline.year_paths(root, "mslp", 1979, 1980))


# --- variant outputs -----------------------------------------------------

@pytest.mark.slow
def test_tag_writes_a_separate_output(raw_root, tmp_path):
    """Detrended and non-detrended runs must be able to coexist.

    Both are wanted: detrending stops the classification grouping days by
    epoch, but it also removes the drift a real change in type frequency
    would produce, so frequency trends from detrended data are conservative.
    """
    work = tmp_path / "work"
    plain = pipeline.run(raw_root, work, "mslp", 1979, 1983)
    variant = pipeline.run(raw_root, work, "mslp", 1979, 1983,
                           do_detrend=False, tag="_nodetrend")
    assert plain != variant
    assert plain.exists() and variant.exists()


@pytest.mark.slow
def test_untagged_run_is_detrended_and_tagged_run_is_not(raw_root, tmp_path):
    work = tmp_path / "work"
    a = xr.open_zarr(pipeline.run(raw_root, work, "mslp", 1979, 1983),
                     consolidated=True)["mslp"]
    b = xr.open_zarr(pipeline.run(raw_root, work, "mslp", 1979, 1983,
                                  do_detrend=False, tag="_nd"),
                     consolidated=True)["mslp"]
    assert "detrended" in a.attrs
    assert "detrended" not in b.attrs

"""From raw yearly Zarr to analysis-ready daily anomalies.

    raw/era5/<key>/<key>_YYYY.zarr   6-hourly, one file per year
        -> daily mean
        -> seasonal cycle removed
        -> linear trend removed
    work/<key>_anom.zarr             daily, whole period, one file

Output is stored as float64. The anomalies themselves are small enough for
float32, but every downstream step reduces over 17,000 time steps, and
keeping the accumulation type in the data rather than in each call site is
one fewer thing to get wrong.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import xarray as xr

from .anomaly import anomaly, daily_mean, detrend

log = logging.getLogger(__name__)


def year_paths(raw_root: Path, key: str, start: int, end: int) -> list[Path]:
    """Yearly Zarr paths for one variable, in order. Fails on any gap.

    A missing year would otherwise pass silently through concatenation and
    leave a hole in the climatology that nothing downstream would flag.
    """
    paths, missing = [], []
    for year in range(start, end + 1):
        p = raw_root / key / f"{key}_{year}.zarr"
        (paths if p.exists() else missing).append(p if p.exists() else year)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} year(s) missing for '{key}': {missing}"
        )
    return paths


def open_years(paths: list[Path], varname: str | None = None) -> xr.DataArray:
    """Concatenate yearly Zarr files into one lazy array along time."""
    parts = [xr.open_zarr(p, consolidated=True) for p in paths]
    ds = xr.concat(parts, dim="time", combine_attrs="override")

    if varname is None:
        names = list(ds.data_vars)
        if len(names) != 1:
            raise ValueError(f"expected one variable, found {names}")
        varname = names[0]

    da = ds[varname]
    if not da.indexes["time"].is_monotonic_increasing:
        da = da.sortby("time")
    return da


def process(
    da: xr.DataArray,
    n_harmonics: int = 3,
    do_detrend: bool = True,
) -> xr.DataArray:
    """Daily mean, then deseasonalise, then detrend.

    Order matters. Deseasonalising before daily averaging would fit harmonics
    to the diurnal cycle as well; detrending before deseasonalising would let
    the seasonal cycle bias the trend estimate wherever the record does not
    start and end at the same point in the year.
    """
    name = da.name
    out = daily_mean(da)
    out = anomaly(out, n_harmonics=n_harmonics)
    if do_detrend:
        out = detrend(out)

    # Arithmetic and xr.dot both drop the name. Restoring it here rather than
    # at the write step keeps every consumer of process() consistent.
    out.name = name
    return out


def write(da: xr.DataArray, dest: Path, time_chunk: int = 2000) -> Path:
    """Write the analysis-ready array.

    Chunked along time with space whole, because the EOF that consumes this
    reduces over time across the full field.
    """
    da = da.chunk({"time": time_chunk})
    for dim in ("latitude", "longitude"):
        if dim in da.dims:
            da = da.chunk({dim: da.sizes[dim]})

    ds = da.to_dataset(name=da.name)
    for name in ds.variables:
        ds[name].encoding = {}

    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    dest.parent.mkdir(parents=True, exist_ok=True)

    ds.to_zarr(tmp, mode="w", consolidated=True)
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)
    return dest


def run(
    raw_root: Path,
    work_root: Path,
    key: str,
    start: int,
    end: int,
    n_harmonics: int = 3,
    do_detrend: bool = True,
) -> Path:
    """Full pipeline for one variable."""
    paths = year_paths(raw_root, key, start, end)
    log.info("%s: %d years", key, len(paths))

    da = open_years(paths)
    log.info("%s: %d sub-daily steps", key, da.sizes["time"])

    out = process(da, n_harmonics=n_harmonics, do_detrend=do_detrend)
    out.attrs["preprocessing"] = (
        f"daily mean; {n_harmonics} annual harmonics removed"
        + ("; linear trend removed" if do_detrend else "")
    )
    out.attrs["period"] = f"{start}-{end}"

    dest = write(out, work_root / f"{key}_anom.zarr")
    log.info("%s: wrote %d daily steps to %s", key, out.sizes["time"], dest.name)
    return dest

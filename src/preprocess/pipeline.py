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

import numpy as np
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


COORD_DTYPE = "float64"


def harmonise(ds: xr.Dataset) -> xr.Dataset:
    """Make one year's coordinates safe to concatenate with any other year's.

    Two differences appear across years of the same variable, both from
    changes to the fetch code partway through a download:

      - a scalar coordinate such as `level`, present on years fetched before
        `src/io/cds.py` learned to strip it
      - a spatial coordinate dtype that differs between years

    Neither raises. `xr.concat` reconciles them by aligning, which silently
    promotes dtypes and can produce an empty intersection. Normalising each
    part first means the pipeline does not depend on every year having been
    written by the same version of the fetch code.
    """
    extra = [c for c in ds.coords if c not in ds.dims]
    if extra:
        ds = ds.drop_vars(extra)

    for name in ("latitude", "longitude"):
        if name in ds.coords and ds[name].dtype != np.dtype(COORD_DTYPE):
            ds = ds.assign_coords({name: ds[name].astype(COORD_DTYPE)})

    return ds


def open_years(paths: list[Path], varname: str | None = None) -> xr.DataArray:
    """Concatenate yearly Zarr files into one lazy array along time.

    `join="exact"` makes a genuine grid mismatch an error. Without it, two
    years on different grids would be aligned to their union and the gaps
    filled with NaN, which nothing downstream would flag.
    """
    parts = [harmonise(xr.open_zarr(p, consolidated=True)) for p in paths]
    ds = xr.concat(parts, dim="time", join="exact", combine_attrs="override")

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
    tag: str = "",
) -> Path:
    """Full pipeline for one variable.

    `tag` suffixes the output name so that variants can sit side by side. The
    detrended and non-detrended versions are both wanted: detrending keeps the
    classification from grouping days by epoch, but it also removes the drift
    that a genuine change in type frequency would produce, so frequency trends
    measured on detrended data are conservative.
    """
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

    dest = write(out, work_root / f"{key}_anom{tag}.zarr")
    log.info("%s: wrote %d daily steps to %s", key, out.sizes["time"], dest.name)
    return dest

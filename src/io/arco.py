"""Slice ERA5 out of the ARCO Zarr store on Google Cloud.

Opening the store costs nothing: it reads consolidated metadata only. Bytes
cross the network when a slice is written to disk, and that is the one slow
step in this project, since the bucket is in us-central1.

Output is one Zarr per variable per year. Per-year files make the download
resumable: an interrupted run loses at most one year, and re-running skips
what is already there.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

log = logging.getLogger(__name__)

G0 = 9.80665  # standard gravity, for geopotential -> geopotential height


def open_store(store: str, storage_options: dict) -> xr.Dataset:
    """Open the ARCO store lazily. No data is transferred."""
    return xr.open_zarr(
        store,
        storage_options=storage_options,
        chunks=None,
        consolidated=True,
    )


def describe(ds: xr.Dataset) -> dict:
    """Facts worth confirming rather than assuming, before slicing anything."""
    return {
        "valid_time_start": ds.attrs.get("valid_time_start"),
        "valid_time_stop": ds.attrs.get("valid_time_stop"),
        "n_variables": len(ds.data_vars),
        "dims": dict(ds.sizes),
        "time_first": str(ds.time.values[0]),
        "time_last": str(ds.time.values[-1]),
        "lat_first": float(ds.latitude.values[0]),
        "lat_last": float(ds.latitude.values[-1]),
        "lon_first": float(ds.longitude.values[0]),
        "lon_last": float(ds.longitude.values[-1]),
    }


def find_variables(ds: xr.Dataset, patterns: list[str]) -> list[str]:
    """List variable names containing any of the given substrings.

    ARCO holds several hundred variables under CF-style long names. Use this
    to confirm a name before putting it in config, rather than guessing.
    """
    lowered = [p.lower() for p in patterns]
    return sorted(
        v for v in ds.data_vars if any(p in v.lower() for p in lowered)
    )


def subset_domain(da: xr.DataArray, domain: dict) -> xr.DataArray:
    """Cut to the configured lat/lon box.

    ERA5 latitude runs north to south, so the slice is given north first.
    Longitude runs 0 to 360, not -180 to 180; the configured box must use the
    same convention.
    """
    lat = da.latitude.values
    descending = lat[0] > lat[-1]
    lat_slice = (
        slice(domain["lat_north"], domain["lat_south"])
        if descending
        else slice(domain["lat_south"], domain["lat_north"])
    )
    return da.sel(latitude=lat_slice,
                  longitude=slice(domain["lon_west"], domain["lon_east"]))


def subset_hours(da: xr.DataArray, hours: list[int]) -> xr.DataArray:
    """Keep only the requested synoptic hours."""
    return da.sel(time=da.time.dt.hour.isin(hours))


def slice_year(
    ds: xr.Dataset,
    varname: str,
    year: int,
    domain: dict,
    hours: list[int],
    level: int | None = None,
    to_height: bool = False,
) -> xr.DataArray:
    """Build the lazy slice for one variable-year. Nothing is computed yet."""
    da = ds[varname].sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
    da = subset_hours(da, hours)
    da = subset_domain(da, domain)

    if level is not None:
        da = da.sel(level=level)

    if to_height:
        da = da / G0
        da.attrs["units"] = "m"
        da.attrs["long_name"] = "geopotential height"

    da.attrs.setdefault("source", "ERA5 via ARCO (gcp-public-data-arco-era5)")
    return da


def write_year(
    da: xr.DataArray,
    dest: Path,
    time_chunk: int = 200,
) -> Path:
    """Materialise one year to Zarr.

    Space is kept whole in each chunk and only time is chunked, because every
    downstream step reduces over time across the full field. Chunking space
    would force a rechunk before the first EOF.
    """
    if dest.exists():
        log.info("skip (exists): %s", dest.name)
        return dest

    da = da.chunk({"time": time_chunk})
    for dim in ("latitude", "longitude"):
        if dim in da.dims:
            da = da.chunk({dim: da.sizes[dim]})

    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        import shutil

        shutil.rmtree(tmp)

    da.to_dataset(name=da.name).to_zarr(tmp, mode="w", consolidated=True)
    tmp.rename(dest)
    log.info("wrote %s", dest.name)
    return dest


def estimate_size(da: xr.DataArray) -> float:
    """Approximate transfer volume in GB for a lazy slice."""
    return float(np.prod(list(da.sizes.values())) * 4 / 1e9)

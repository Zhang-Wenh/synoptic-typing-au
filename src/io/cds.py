"""Fetch ERA5 pressure-level data from the Copernicus Climate Data Store.

Why this exists alongside src/io/arco.py:

ARCO stores pressure-level variables with all 37 levels in a single chunk, so
selecting one level transfers all 37. Measured cost was about 2 hours per year
for 500 hPa geopotential, against 3.3 minutes per year for mean sea level
pressure, a ratio matching the level count.

CDS applies `pressure_level` and `area` on the server, so only the requested
level and region are transferred. Measured cost is about 5.5 minutes per year,
comparable to MSLP.

The trade-off is that CDS queues requests and returns NetCDF, so this module
adds a conversion step. Output is written in the same layout as arco.py so
that downstream code does not need to know which source a year came from.
"""

from __future__ import annotations

import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import xarray as xr

log = logging.getLogger(__name__)

G0 = 9.80665  # standard gravity, geopotential -> geopotential height

DATASET = "reanalysis-era5-pressure-levels"
ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]


def build_request(
    variable: str,
    level: int,
    year: int,
    hours: list[int],
    domain: dict,
) -> dict:
    """Build one year-sized CDS request.

    `pressure_level` and `area` are the point of using CDS: both are applied
    before anything is transferred.

    CDS expects area as [north, west, south, east]. Longitudes must be in the
    -180 to 180 convention, so an eastern bound of 180 stays 180 but anything
    above it would need converting.
    """
    return {
        "product_type": ["reanalysis"],
        "variable": [variable],
        "pressure_level": [str(level)],
        "year": [str(year)],
        "month": ALL_MONTHS,
        "day": ALL_DAYS,
        "time": [f"{h:02d}:00" for h in hours],
        "area": [
            domain["lat_north"],
            domain["lon_west"] if domain["lon_west"] <= 180 else domain["lon_west"] - 360,
            domain["lat_south"],
            domain["lon_east"] if domain["lon_east"] <= 180 else domain["lon_east"] - 360,
        ],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def normalise(ds: xr.Dataset, to_height: bool = True) -> xr.Dataset:
    """Bring a CDS NetCDF into the same shape as the ARCO-derived files.

    Four differences have to be reconciled:
      - the time coordinate is named valid_time in current CDS output
      - a length-one pressure_level dimension is left over from the selection
      - an expver dimension appears when a request spans ERA5 and ERA5T
      - geopotential is in m2 s-2, while the ARCO path stores height in metres
    """
    if "valid_time" in ds.coords or "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})

    # expver marks final (0001) versus preliminary (0005) data. Combining them
    # leaves NaNs where each is absent, so merge rather than select.
    if "expver" in ds.dims:
        ds = ds.reduce(lambda a, axis: a.max(axis=axis), dim="expver", keep_attrs=True)
    elif "expver" in ds.coords:
        ds = ds.drop_vars("expver")

    for dim in ("pressure_level", "level", "isobaricInhPa"):
        if dim in ds.dims and ds.sizes[dim] == 1:
            ds = ds.squeeze(dim, drop=True)

    if "z" in ds.data_vars:
        ds = ds.rename({"z": "geopotential"})

    if to_height and "geopotential" in ds.data_vars:
        attrs = dict(ds["geopotential"].attrs)
        ds["geopotential"] = ds["geopotential"] / G0
        attrs.update({"units": "m", "long_name": "geopotential height"})
        ds["geopotential"].attrs = attrs

    if "latitude" in ds.coords and ds.latitude.size > 1:
        if ds.latitude.values[0] < ds.latitude.values[-1]:
            ds = ds.isel(latitude=slice(None, None, -1))

    # CDS keeps a scalar coordinate for the selected level. It carries real
    # information, but a coordinate present on one source and absent on the
    # other makes DataArray.equals() disagree between otherwise identical
    # fields, and makes xarray align them to an empty intersection later.
    # Keep the value as an attribute; drop the coordinate.
    extra = [c for c in ds.coords if c not in ds.dims]
    for name in extra:
        ds.attrs[name] = str(ds[name].values)
    if extra:
        ds = ds.drop_vars(extra)

    ds.attrs["source"] = f"ERA5 via CDS ({DATASET})"
    return ds


def to_zarr(ds: xr.Dataset, dest: Path, time_chunk: int = 200) -> Path:
    """Write in the same chunking convention as the ARCO path.

    Encoding is cleared for the same reason as in arco.py: inherited codecs
    from another format are rejected by zarr-python v3.
    """
    ds = ds.chunk({"time": time_chunk})
    for dim in ("latitude", "longitude"):
        if dim in ds.dims:
            ds = ds.chunk({dim: ds.sizes[dim]})

    ds = ds.copy()
    for name in ds.variables:
        ds[name].encoding = {}

    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)

    ds.to_zarr(tmp, mode="w", consolidated=True)
    tmp.rename(dest)
    return dest


def fetch_year(
    client,
    year: int,
    variable: str,
    level: int,
    hours: list[int],
    domain: dict,
    dest_root: Path,
    scratch: Path,
    key: str = "z",
) -> Path:
    """Retrieve one year, convert it, and write Zarr. Skips completed years."""
    dest = dest_root / key / f"{key}_{year}.zarr"
    if dest.exists():
        log.info("skip (exists): %s", dest.name)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    nc_path = scratch / f"{key}_{year}.nc"

    if not nc_path.exists():
        log.info("requesting %d", year)
        client.retrieve(DATASET, build_request(variable, level, year, hours, domain)
                        ).download(str(nc_path))

    with xr.open_dataset(nc_path) as raw:
        ds = normalise(raw.load())

    to_zarr(ds, dest)
    nc_path.unlink(missing_ok=True)
    log.info("wrote %s", dest.name)
    return dest


def fetch(
    client,
    years: range,
    variable: str,
    level: int,
    hours: list[int],
    domain: dict,
    dest_root: Path,
    scratch: Path,
    workers: int = 3,
    key: str = "z",
) -> list[Path]:
    """Fetch several years concurrently.

    CDS limits how many requests one account may have active, and a queue is
    shared with every other user. Three workers is deliberately modest: it
    overlaps queue waiting without monopolising the service.
    """
    written: list[Path] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                fetch_year, client, y, variable, level, hours, domain,
                dest_root, scratch, key,
            ): y
            for y in years
        }
        for future in as_completed(futures):
            year = futures[future]
            try:
                written.append(future.result())
            except Exception as exc:  # noqa: BLE001
                log.error("year %d failed: %s", year, exc)
    return written

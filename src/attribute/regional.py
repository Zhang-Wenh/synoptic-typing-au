"""Reduce gridded rainfall to one daily series for the target region.

Three things have to be right here, and each fails quietly:

  Land masking. SILO is interpolated from station observations, so grid cells
  over water carry values that are extrapolations from distant stations rather
  than measurements. Including them biases the regional mean toward whatever
  the interpolation does offshore. SILO marks these as missing, so masking is
  a matter of not filling them.

  Area weighting. Cells shrink poleward, so an unweighted mean over-counts the
  southern part of the region. The effect is small over seven degrees of
  latitude, but it costs nothing to be right.

  Day alignment. SILO days are Australian local time to 9am, ERA5 days are
  UTC. A day labelled 15 January in SILO covers roughly 14 January 23:00 UTC
  to 15 January 23:00 UTC. Rainfall is therefore attributed to circulation
  roughly a day earlier than the label suggests, which matters when the point
  of the analysis is which circulation produced which rain.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from ..preprocess.weights import area_weights

log = logging.getLogger(__name__)

ACCUM = "float64"


def open_years(paths: list[Path], varname: str = "daily_rain") -> xr.DataArray:
    """Open a list of yearly SILO NetCDF files as one lazy array."""
    ds = xr.open_mfdataset(
        [str(p) for p in paths],
        combine="by_coords",
        chunks={"time": 365},
        parallel=False,
    )
    if varname not in ds.data_vars:
        raise KeyError(f"{varname!r} not in {list(ds.data_vars)}")
    return ds[varname]


def subset(da: xr.DataArray, target: dict) -> xr.DataArray:
    """Cut to the target region, whichever way the latitude axis runs."""
    lat = da["lat"] if "lat" in da.coords else da["latitude"]
    lat_name = lat.name
    lon_name = "lon" if "lon" in da.coords else "longitude"

    descending = bool(lat.values[0] > lat.values[-1])
    lat_slice = (
        slice(target["lat_north"], target["lat_south"])
        if descending
        else slice(target["lat_south"], target["lat_north"])
    )
    return da.sel(
        {lat_name: lat_slice,
         lon_name: slice(target["lon_west"], target["lon_east"])}
    )


def regional_mean(da: xr.DataArray, lat_name: str | None = None) -> xr.DataArray:
    """Area-weighted mean over land cells only.

    Cells that are missing everywhere are outside the land mask and drop out.
    Cells missing on some days only would silently change the effective area
    day to day, so those are checked rather than assumed away.
    """
    if lat_name is None:
        lat_name = "lat" if "lat" in da.coords else "latitude"
    lon_name = "lon" if "lon" in da.coords else "longitude"

    weights = area_weights(da[lat_name])
    valid = da.notnull()

    always = valid.all("time")
    never = (~valid).all("time")
    sometimes = int((~always & ~never).sum())
    if sometimes:
        log.warning(
            "%d cells are missing on some days but not all; the effective "
            "area varies day to day", sometimes
        )

    return (
        da.astype(ACCUM)
        .weighted(weights.where(always, 0.0))
        .mean(dim=[lat_name, lon_name])
    )


def shift_to_utc_day(da: xr.DataArray, hours: int = -9) -> xr.DataArray:
    """Relabel SILO days to the UTC day the rain mostly fell on.

    SILO totals run to 9am local time, so a day labelled D covers roughly
    D-1 23:00 UTC to D 23:00 UTC for southeast Australia. Shifting the label
    back by nine hours puts each total on the UTC day whose circulation
    produced most of it.

    Nine hours is an approximation. Eastern Australia moves between UTC+10 and
    UTC+11, and rain within a 24-hour window is not uniform, so this aligns the
    bulk rather than every event. The alternative -- leaving the labels alone
    -- misattributes a systematic fraction of rain to the following day's
    circulation, which is worse.
    """
    shifted = da.assign_coords(
        time=da["time"] + np.timedelta64(hours, "h")
    )
    shifted = shifted.assign_coords(time=shifted["time"].dt.floor("1D"))
    shifted.attrs = dict(da.attrs)
    shifted.attrs["day_alignment"] = (
        f"labels shifted {hours} h so SILO 9am-to-9am totals sit on the UTC "
        "day whose circulation produced them"
    )
    return shifted


def align_to(series: xr.DataArray, reference: xr.DataArray) -> xr.DataArray:
    """Restrict a series to the days a reference series covers.

    An inner join, deliberately. Reindexing to the reference would fill
    non-overlapping days with NaN, and those would then propagate into the
    per-type means as silently reduced sample sizes.
    """
    common = np.intersect1d(series["time"].values, reference["time"].values)
    if common.size == 0:
        raise ValueError("no overlapping days between the two series")
    if common.size < reference.sizes["time"]:
        log.info(
            "%d of %d reference days have no impact data",
            reference.sizes["time"] - common.size, reference.sizes["time"],
        )
    return series.sel(time=common)


def hot_day_indicator(
    series: xr.DataArray, percentile: float = 90.0, season_aware: bool = True
) -> xr.DataArray:
    """Turn a daily maximum temperature series into a hot-day indicator.

    Mean temperature and hot-day frequency answer different questions, and the
    difference matters for what the decomposition can show.

    A decomposition of mean temperature is dominated by warming appearing as a
    within-type intensity change: every type gets hotter because the whole
    distribution shifts. That is real but it is also the least surprising
    possible result.

    Hot days are where circulation earns its keep. Whether a given day exceeds
    a threshold depends on whether the synoptic situation delivers heat, so a
    change in how often each type occurs translates directly into a change in
    how often the threshold is crossed. This is the variable where the
    frequency term has a chance to matter.

    The threshold is a percentile of the series itself, computed separately
    for the cool and warm halves of the year when `season_aware` is set.
    A single annual threshold would put almost every hot day in summer and
    leave the cool season with no exceedances to analyse.
    """
    if not season_aware:
        threshold = float(series.quantile(percentile / 100.0))
        out = (series > threshold).astype(ACCUM)
        out.attrs["threshold"] = threshold
        out.attrs["definition"] = f"daily maximum above the {percentile:g}th percentile"
        return out.rename("hot_day")

    month = series["time"].dt.month
    is_cool = month.isin([4, 5, 6, 7, 8, 9, 10])

    cool_threshold = float(series.sel(time=is_cool).quantile(percentile / 100.0))
    warm_threshold = float(series.sel(time=~is_cool).quantile(percentile / 100.0))

    threshold = xr.where(is_cool, cool_threshold, warm_threshold)
    out = (series > threshold).astype(ACCUM)
    out.attrs["cool_threshold"] = cool_threshold
    out.attrs["warm_threshold"] = warm_threshold
    out.attrs["definition"] = (
        f"daily maximum above the {percentile:g}th percentile of its own half "
        "of the year, so that both seasons have exceedances to analyse"
    )
    return out.rename("hot_day")


def build(
    paths: list[Path],
    target: dict,
    varname: str = "daily_rain",
    shift_hours: int = -9,
) -> xr.DataArray:
    """Full path from yearly files to one daily regional series."""
    da = open_years(paths, varname)
    da = subset(da, target)
    log.info("target grid: %s", dict(da.sizes))

    series = regional_mean(da)
    series = shift_to_utc_day(series, shift_hours)
    series.name = varname
    series.attrs["region"] = (
        f"{target['lat_south']} to {target['lat_north']} lat, "
        f"{target['lon_west']} to {target['lon_east']} lon"
    )
    return series


INDEX_UNITS = {
    "rain": "mm/day",
    "tmax": "degC",
    "hot": "fraction of days",
}


def load_index(
    raw_root, target: dict, start: int, end: int, index: str
) -> xr.DataArray:
    """Build the regional daily series for one impact index.

    Three are available and they answer different questions.

      rain  mean daily rainfall
      tmax  mean daily maximum temperature
      hot   fraction of days above the 90th percentile of the same half-year

    The distinction between `tmax` and `hot` decides what a decomposition can
    show. Decomposing mean temperature mostly recovers warming as a
    within-type intensity change, because the whole distribution shifts and
    every type moves with it. Hot-day frequency is where circulation matters:
    whether a day crosses the threshold depends on whether the synoptic
    situation delivers heat.

    Lives here rather than in a script so that more than one entry point can
    use it without importing from another script.
    """
    variable = "daily_rain" if index == "rain" else "max_temp"
    folder = Path(raw_root) / "silo" / variable

    files = sorted(folder.glob("*.nc")) if folder.exists() else []
    files = [f for f in files if start <= int(f.name[:4]) <= end]
    if not files:
        raise FileNotFoundError(
            f"no SILO files for {start}-{end} in {folder}. "
            f"Run: python scripts/fetch_silo.py --variables {variable}"
        )
    log.info("SILO %s: %d yearly files", variable, len(files))

    series = build(files, target, varname=variable)
    if index == "hot":
        series = hot_day_indicator(series.compute())
        log.info("%s", series.attrs["definition"])
    return series


BANDS = {
    "tropics": {
        "lat_north": -10.0, "lat_south": -20.0,
        "lon_west": 112.0, "lon_east": 154.0,
        "season": "warm",
        "note": "monsoonal north; nearly all rain falls November to March",
    },
    "subtropics": {
        "lat_north": -20.0, "lat_south": -30.0,
        "lon_west": 112.0, "lon_east": 154.0,
        "season": None,
        "note": "arid interior and subtropical east; no single wet season",
    },
    "midlatitudes": {
        "lat_north": -30.0, "lat_south": -40.0,
        "lon_west": 112.0, "lon_east": 154.0,
        "season": "cool",
        "note": "frontal rainfall, mostly April to October",
    },
    "southeast": {
        "lat_north": -33.0, "lat_south": -40.0,
        "lon_west": 140.0, "lon_east": 150.0,
        "season": "cool",
        "note": "the target region of the earlier analysis, for comparison",
    },
}
"""Latitude bands for comparing the circulation-intensity balance by region.

Each band is analysed in its own wet season. The tropics receive almost all
their rain between November and March and the midlatitudes between April and
October, so a single fixed season would compare a real signal in one band
against near-zero in another.

The three zonal bands meet exactly and do not overlap. The southeast band sits
inside the midlatitude band on purpose, linking back to the earlier analysis.
"""

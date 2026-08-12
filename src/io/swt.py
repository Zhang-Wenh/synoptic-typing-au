"""Read the published Australian Synoptic Weather Types.

Barnes et al. (2025) define 30 synoptic weather types over Australia by
k-means clustering the 850 hPa wind field in ERA5, grouped into 8 weather
regimes. The labels are distributed as a daily series, so nothing needs to be
clustered here: the file is read and converted to the integer codes the
decomposition expects.

    Barnes, Liqui Lung, Jakob, Gunn and Reeder (2025),
    Australian Synoptic Weather Types, J. Geophys. Res. Atmos.,
    doi:10.1029/2025JD043873

Two things this module exists to get right.

**Day boundaries.** These labels are the circulation at 12 UTC, which is
around 10pm in southeastern Australia. SILO daily totals run to 9am local
time, so a SILO day labelled D covers roughly 23 UTC on D-1 to 23 UTC on D.
The 12 UTC snapshot that falls inside that window is the one labelled D-1 by
the SWT file. Rain on SILO day D therefore belongs with the SWT label from
day D-1, not day D.

This is a one-day shift in the opposite direction from the one used when
aligning SILO with daily-mean ERA5 anomalies, and getting it backwards would
attribute each day's rain to the following day's circulation -- which is
exactly the kind of error that produces a plausible but wrong answer.

**Regimes against types.** Both groupings are useful and they answer different
questions. The 30 types resolve more, but leave 300 to 800 days each over a
47-year record, which is thin for estimating a per-type mean and its trend.
The 8 regimes have 840 to 3300 days each and carry names with established
synoptic meaning. Regimes are the default here; types are the sensitivity
check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Ordered as in the published file, so that integer codes are stable and
# comparable with anything else built from the same source.
REGIMES = ["WH", "CH", "EH", "TH", "FH", "WCT", "COL", "AM"]

REGIME_NAMES = {
    "WH": "western high",
    "CH": "central high",
    "EH": "eastern high",
    "TH": "Tasman high",
    "FH": "flanking high",
    "WCT": "west coast trough",
    "COL": "cut-off low",
    "AM": "active monsoon",
}

SILO_TO_SWT_DAYS = -1


def open_labels(path: str | Path) -> xr.DataArray:
    """Open the distributed label series as strings, one value per day."""
    ds = xr.open_dataset(path)
    if "assigned_SWT" not in ds:
        raise KeyError(
            f"{path} has no assigned_SWT variable; found {list(ds.data_vars)}"
        )
    labels = ds["assigned_SWT"].load()
    labels.attrs.setdefault("reference", ds.attrs.get("reference", ""))
    labels.attrs.setdefault("source", "Barnes et al. (2025), Australian SWTs")
    return labels


def type_order(path: str | Path) -> list[str]:
    """The canonical type ordering, taken from the file rather than assumed."""
    ds = xr.open_dataset(path)
    if "SWTs" not in ds.coords:
        raise KeyError(f"{path} has no SWTs coordinate")
    return [str(name) for name in ds["SWTs"].values]


def to_regimes(labels: xr.DataArray) -> xr.DataArray:
    """Collapse the 30 types to their 8 regimes, keeping the strings."""
    values = np.array([str(name).split("-")[0] for name in labels.values])
    out = xr.DataArray(
        values, dims="time", coords={"time": labels["time"]}, name="regime"
    )
    out.attrs = dict(labels.attrs)
    out.attrs["grouping"] = "8 weather regimes"
    return out


def encode(labels: xr.DataArray, order: list[str] | None = None) -> xr.DataArray:
    """Map string labels to integer codes in a fixed order.

    The order is fixed rather than derived from the data so that codes mean
    the same thing across subsets. Deriving it from `np.unique` would silently
    renumber everything whenever a rare type happened to be absent from a
    seasonal subset.
    """
    names = np.array([str(name) for name in labels.values])
    if order is None:
        order = REGIMES if set(names) <= set(REGIMES) else sorted(set(names))

    lookup = {name: i for i, name in enumerate(order)}
    unknown = set(names) - set(lookup)
    if unknown:
        raise ValueError(f"labels contain names not in the given order: {sorted(unknown)}")

    out = xr.DataArray(
        np.array([lookup[name] for name in names]),
        dims="time",
        coords={"time": labels["time"]},
        name="type",
    )
    out.attrs = dict(labels.attrs)
    out.attrs["codes"] = ", ".join(f"{i}={name}" for i, name in enumerate(order))
    return out


def to_daily_index(labels: xr.DataArray, shift_days: int = 0) -> xr.DataArray:
    """Strip the time of day, optionally shifting to a different day convention.

    The published labels are stamped at 12 UTC. Downstream code joins on the
    calendar day, so the hour is dropped here rather than being carried around
    and silently failing to match.
    """
    time = pd.DatetimeIndex(labels["time"].values).normalize()
    if shift_days:
        time = time + pd.Timedelta(days=shift_days)

    out = labels.copy()
    out = out.assign_coords(time=time)
    if shift_days:
        out.attrs["day_shift"] = (
            f"labels moved {shift_days:+d} day(s) to match the impact series"
        )
    return out


def load(
    path: str | Path,
    grouping: str = "regime",
    start: str | None = None,
    end: str | None = None,
    shift_days: int = 0,
) -> tuple[xr.DataArray, list[str]]:
    """Read the labels as integer codes, with the names they stand for.

    Returns the coded series and the ordered names, so that any table or plot
    downstream can be labelled without hardcoding the ordering again.
    """
    labels = open_labels(path)

    if grouping == "regime":
        labels = to_regimes(labels)
        order = REGIMES
    elif grouping == "type":
        order = type_order(path)
    else:
        raise ValueError(f"grouping must be 'regime' or 'type', got {grouping!r}")

    labels = to_daily_index(labels, shift_days)
    if start or end:
        labels = labels.sel(time=slice(start, end))
    if labels.sizes["time"] == 0:
        raise ValueError(f"no labels within {start} to {end}")

    return encode(labels, order), order

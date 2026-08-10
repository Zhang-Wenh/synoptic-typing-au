"""Daily aggregation, seasonal cycle removal, and detrending.

Design decisions here are choices to defend in the write-up, not mechanical
preprocessing.

Daily means. Classification runs on daily fields, not on the 6-hourly record.
Four times a day quadruples the sample count but not the information: the
four steps within a day are strongly correlated, so the effective sample size
barely changes while clustering cost rises fourfold. Daily resolution is also
the convention in Australian synoptic typing, which keeps the results
comparable with published classifications.

Seasonal cycle. A day-of-year climatology estimates each calendar day from as
many samples as there are years, which is noisy. Fitting a small number of
annual harmonics gives a smooth cycle and does not absorb synoptic variance.

The fit solves the normal equations rather than calling lstsq on the stacked
array. Stacking 47 years of daily fields would materialise about 10 GB;
X'X is a 7x7 matrix and X'Y reduces over time, so both stay lazy and small.

Detrending. Leaving a long-term trend in the field lets the clustering define
types by epoch rather than by circulation, which makes a later analysis of
type frequency change circular. Types are built on detrended anomalies, and
the trend is examined afterwards in the frequencies.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

ACCUM = "float64"
DAYS_PER_YEAR = 365.25


def daily_mean(da: xr.DataArray, time_name: str = "time") -> xr.DataArray:
    """Average sub-daily steps to daily values.

    The cast to float64 happens before the reduction, not after: casting the
    result would preserve whatever precision was already lost.
    """
    out = da.astype(ACCUM).resample({time_name: "1D"}).mean()
    out.attrs = dict(da.attrs)
    out.attrs["resampled_from"] = f"{da.sizes[time_name]} sub-daily steps"
    return out


def harmonic_design(time: xr.DataArray, n_harmonics: int = 3) -> xr.DataArray:
    """Design matrix of a constant plus n annual harmonics.

    Dimensions are (time, term) with 2 * n_harmonics + 1 terms.
    """
    phase = 2 * np.pi * time.dt.dayofyear.astype(ACCUM) / DAYS_PER_YEAR

    terms = [xr.ones_like(phase)]
    labels = ["const"]
    for k in range(1, n_harmonics + 1):
        terms += [np.cos(k * phase), np.sin(k * phase)]
        labels += [f"cos{k}", f"sin{k}"]

    design = xr.concat(terms, dim="term").assign_coords(term=labels)
    return design.transpose(time.name, "term")


def harmonic_climatology(
    da: xr.DataArray, n_harmonics: int = 3, time_name: str = "time"
) -> xr.DataArray:
    """Smooth seasonal cycle, evaluated at every input time.

    Solved as beta = (X'X)^-1 X'Y. X'X is (n_terms, n_terms) and X'Y reduces
    over time, so nothing larger than the field itself is ever held in memory.
    """
    da = da.astype(ACCUM)
    design = harmonic_design(da[time_name], n_harmonics)

    xtx = xr.dot(design, design.rename(term="term2"), dim=time_name)
    xty = xr.dot(design, da, dim=time_name)

    beta_values = np.linalg.solve(
        xtx.transpose("term", "term2").values,
        xty.transpose("term", ...).values.reshape(xty.sizes["term"], -1),
    )
    beta = xr.DataArray(
        beta_values.reshape(xty.transpose("term", ...).shape),
        dims=xty.transpose("term", ...).dims,
        coords={d: xty[d] for d in xty.dims if d in xty.coords},
    )

    clim = xr.dot(design, beta, dim="term")
    clim.attrs["long_name"] = f"seasonal cycle, {n_harmonics} harmonics"
    return clim.transpose(*da.dims)


def anomaly(
    da: xr.DataArray, n_harmonics: int = 3, time_name: str = "time"
) -> xr.DataArray:
    """Field minus its smooth seasonal cycle."""
    out = da.astype(ACCUM) - harmonic_climatology(da, n_harmonics, time_name)
    out.attrs = dict(da.attrs)
    out.attrs["long_name"] = f"{da.attrs.get('long_name', da.name)} anomaly"
    out.attrs["deseasonalised"] = f"{n_harmonics} annual harmonics removed"
    return out


def detrend(da: xr.DataArray, time_name: str = "time", degree: int = 1) -> xr.DataArray:
    """Remove a polynomial trend in time at every grid point."""
    da = da.astype(ACCUM)
    fit = da.polyfit(dim=time_name, deg=degree)
    trend = xr.polyval(da[time_name], fit.polyfit_coefficients)
    out = da - trend
    out.attrs = dict(da.attrs)
    out.attrs["detrended"] = f"degree {degree} polynomial removed"
    return out


def standardise(da: xr.DataArray, time_name: str = "time") -> xr.DataArray:
    """Divide by the temporal standard deviation at each point.

    Used when clustering on more than one variable, so that a field with a
    larger natural range does not dominate the distance metric.
    """
    da = da.astype(ACCUM)
    return da / da.std(dim=time_name)

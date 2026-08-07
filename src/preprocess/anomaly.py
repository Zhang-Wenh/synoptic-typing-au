"""Seasonal cycle removal and detrending.

Both steps are design decisions that need defending in the write-up, not
mechanical preprocessing.

Seasonal cycle: a day-of-year climatology is noisy, because each calendar day
is estimated from only as many samples as there are years. Fitting a small
number of harmonics gives a smooth cycle and does not absorb synoptic-scale
variance.

Detrending: if the long-term trend is left in the field, the clustering can
partly define types by epoch rather than by circulation, and a later analysis
of type frequency change becomes circular. Types are therefore built on
detrended anomalies, and the trend is examined afterwards in the frequencies.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def harmonic_climatology(
    da: xr.DataArray, n_harmonics: int = 3, time_name: str = "time"
) -> xr.DataArray:
    """Smooth seasonal cycle from the first n annual harmonics.

    Returns the fitted cycle evaluated at every input time, so subtracting it
    from da gives the anomaly directly.
    """
    time = da[time_name]
    doy = time.dt.dayofyear
    year_length = 365.25
    phase = 2 * np.pi * doy / year_length

    terms = [xr.ones_like(phase, dtype=float)]
    for k in range(1, n_harmonics + 1):
        terms.append(np.cos(k * phase))
        terms.append(np.sin(k * phase))

    design = xr.concat(terms, dim="term").transpose(time_name, "term")

    stacked = da.stack(space=[d for d in da.dims if d != time_name])
    coeffs, *_ = np.linalg.lstsq(
        design.values, stacked.values, rcond=None
    )
    fitted = design.values @ coeffs

    out = xr.DataArray(
        fitted, coords=stacked.coords, dims=stacked.dims
    ).unstack("space")
    out.attrs["long_name"] = f"seasonal cycle, {n_harmonics} harmonics"
    return out.transpose(*da.dims)


def anomaly(
    da: xr.DataArray, n_harmonics: int = 3, time_name: str = "time"
) -> xr.DataArray:
    """Field minus its smooth seasonal cycle."""
    out = da - harmonic_climatology(da, n_harmonics, time_name)
    out.attrs = dict(da.attrs)
    out.attrs["long_name"] = f"{da.attrs.get('long_name', da.name)} anomaly"
    return out


def detrend(da: xr.DataArray, time_name: str = "time", degree: int = 1) -> xr.DataArray:
    """Remove a polynomial trend in time at every grid point."""
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
    return da / da.std(dim=time_name)

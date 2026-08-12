"""A Southern Annular Mode proxy, and how much of the type frequency change
follows from it.

The classification has produced a frequency trend with a clear sign and a
clear physical effect. The next question is whether that trend is the local
expression of something already documented, or something new. The Southern
Annular Mode is the obvious candidate: its positive trend since the 1970s is
well established and is the standard explanation for the poleward shift of
the Southern Hemisphere westerlies.

The index here is a proxy, not the standard one, and the difference matters
when comparing with published values.

  Gong and Wang (1999) define the index as the difference in normalised
  zonally averaged mean sea level pressure between 40S and 65S, taken around
  the whole hemisphere. Marshall (2003) computes the same quantity from twelve
  station records, which avoids the reanalysis's weaker constraint over the
  Southern Ocean before the satellite era.

  The domain here reaches 60S, not 65S, and spans 90E to 180E rather than the
  full circle. So this is a regional, partial version of the index. It should
  correlate strongly with the standard one -- the annular mode is by
  definition close to zonally symmetric -- but the amplitudes are not
  comparable and any quantitative claim should be checked against Marshall.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ACCUM = "float64"


def sam_proxy(
    mslp: xr.DataArray,
    north: float = -40.0,
    south: float = -60.0,
    half_width: float = 2.5,
) -> xr.DataArray:
    """Normalised zonal-mean pressure difference between two latitude bands.

    Each band is averaged zonally and then standardised over the whole record
    before differencing, as in the standard definition. Standardising first
    matters: the southern band has much larger variance, and differencing raw
    pressures would make the index almost entirely a southern-band index.
    """
    lat = mslp["latitude"]
    zonal = mslp.astype(ACCUM).mean("longitude")

    def band(centre: float) -> xr.DataArray:
        sel = zonal.sel(latitude=slice(centre + half_width, centre - half_width))
        if sel.sizes["latitude"] == 0:
            sel = zonal.sel(latitude=slice(centre - half_width, centre + half_width))
        if sel.sizes["latitude"] == 0:
            raise ValueError(f"no grid points within {half_width} deg of {centre}")
        mean = sel.mean("latitude")
        return (mean - mean.mean()) / mean.std()

    index = band(north) - band(south)
    index.name = "sam_proxy"
    index.attrs["definition"] = (
        f"normalised zonal-mean MSLP at {abs(north):g}S minus {abs(south):g}S, "
        f"averaged over the analysis domain only"
    )
    index.attrs["caution"] = (
        "regional and truncated at 60S; the standard index uses 65S and the "
        "full latitude circle, so amplitudes are not comparable"
    )
    return index


def seasonal_mean(index: xr.DataArray, season: str | None = None) -> xr.DataArray:
    """Yearly mean of a daily index, restricted to one season.

    The warm season straddles the calendar year, so November and December are
    counted with the January that follows, matching the convention used in the
    decomposition.
    """
    month = index["time"].dt.month
    year = index["time"].dt.year

    if season == "cool":
        keep = month.isin([4, 5, 6, 7, 8, 9, 10])
        season_year = year
    elif season == "warm":
        keep = month.isin([11, 12, 1, 2, 3])
        season_year = xr.where(month >= 11, year + 1, year)
    elif season is None:
        keep = xr.ones_like(month, dtype=bool)
        season_year = year
    else:
        raise ValueError(f"season must be 'cool', 'warm' or None, got {season!r}")

    sub = index.sel(time=keep)
    sub = sub.assign_coords(season_year=season_year.sel(time=keep))
    return sub.groupby("season_year").mean().rename(season_year="year")


def regress_on_index(
    frequency: xr.DataArray, index: xr.DataArray
) -> tuple[np.ndarray, np.ndarray]:
    """Sensitivity of each type's frequency to the index, and the correlation.

    Returns the regression slope in frequency units per index unit, and the
    correlation coefficient. The slope says how much a unit change in the
    index moves the frequency; the correlation says how much of the
    year-to-year variation it accounts for.
    """
    common = np.intersect1d(frequency["year"].values, index["year"].values)
    f = frequency.sel(year=common).values
    x = index.sel(year=common).values.astype(ACCUM)

    slopes = np.full(f.shape[1], np.nan)
    correlations = np.full(f.shape[1], np.nan)

    x_centred = x - x.mean()
    x_var = float((x_centred**2).sum())

    for i in range(f.shape[1]):
        y = f[:, i]
        ok = np.isfinite(y)
        if ok.sum() < 3 or x_var == 0:
            continue
        xi = x[ok] - x[ok].mean()
        yi = y[ok] - y[ok].mean()
        slopes[i] = float((xi * yi).sum() / (xi**2).sum())
        # A type whose frequency never varies has no correlation to report;
        # the slope is a well-defined zero but the correlation is 0/0.
        y_var = float((yi**2).sum())
        if y_var > 0:
            correlations[i] = float(
                (xi * yi).sum() / np.sqrt((xi**2).sum() * y_var)
            )

    return slopes, correlations


def ridge_index(
    mslp: xr.DataArray, centre: float = -37.5, half_width: float = 5.0
) -> xr.DataArray:
    """Strength of the subtropical ridge across the analysis longitudes.

    A second candidate, distinct from the annular mode. The subtropical ridge
    intensifies and shifts poleward as the Hadley cell widens, and that is not
    the same process as the annular mode even though the two are correlated:
    the ridge index responds to a change in one band, the annular index to a
    contrast between two.

    Not standardised against a second band, deliberately. The quantity of
    interest is the absolute strength of the ridge over southern Australia,
    which is what a widening Hadley cell would change.
    """
    zonal = mslp.astype(ACCUM).mean("longitude")
    band = zonal.sel(latitude=slice(centre + half_width, centre - half_width))
    if band.sizes["latitude"] == 0:
        band = zonal.sel(latitude=slice(centre - half_width, centre + half_width))
    if band.sizes["latitude"] == 0:
        raise ValueError(f"no grid points within {half_width} deg of {centre}")

    index = band.mean("latitude")
    index = (index - index.mean()) / index.std()
    index.name = "ridge_index"
    index.attrs["definition"] = (
        f"normalised zonal-mean MSLP averaged over "
        f"{abs(centre) - half_width:g}S to {abs(centre) + half_width:g}S"
    )
    return index


def read_nino34(path: str | Path) -> xr.DataArray:
    """Read the NOAA PSL Nino 3.4 anomaly file into a monthly series.

    The format is a header line giving the first and last year, then one line
    per year with twelve monthly values, then trailing metadata lines. Missing
    months are flagged with a large negative sentinel rather than a gap, so
    they have to be recognised rather than parsed as data.

    Why this matters for the analysis. Tropical Australian rainfall is
    dominated by ENSO at interannual scale, and a 47-year record holds only
    about fifteen ENSO cycles. A trend measured over that record can be
    produced by where the record happens to start and end in the cycle. A
    block bootstrap does not catch this: it resamples residuals around the
    fitted line, so it measures sampling error, not whether the line itself is
    an artefact of a few strong events.
    """
    lines = Path(path).read_text().splitlines()
    first, last = (int(v) for v in lines[0].split()[:2])

    years, values = [], []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != 13:
            break
        year = int(parts[0])
        if not first <= year <= last:
            break
        months = [float(v) for v in parts[1:]]
        years.append(year)
        values.append(months)

    if not years:
        raise ValueError(f"no data rows parsed from {path}")

    data = np.array(values, dtype=ACCUM)
    data[data < -90] = np.nan  # sentinel for missing months

    time = pd.date_range(f"{years[0]}-01-01", periods=data.size, freq="MS")
    out = xr.DataArray(
        data.ravel(), dims="time", coords={"time": time}, name="nino34"
    )
    out.attrs["source"] = "NOAA PSL, Nino 3.4 SST anomaly"
    out.attrs["definition"] = "SST anomaly averaged over 5N-5S, 170W-120W"
    return out.dropna("time")


def monthly_to_season(index: xr.DataArray, season: str | None = None) -> xr.DataArray:
    """Yearly mean of a monthly index, restricted to one season.

    Same season conventions as the decomposition, including counting November
    and December with the following January so that the austral summer is not
    split across two calendar years.
    """
    month = index["time"].dt.month
    year = index["time"].dt.year

    if season == "cool":
        keep = month.isin([4, 5, 6, 7, 8, 9, 10])
        season_year = year
    elif season == "warm":
        keep = month.isin([11, 12, 1, 2, 3])
        season_year = xr.where(month >= 11, year + 1, year)
    elif season is None:
        keep = xr.ones_like(month, dtype=bool)
        season_year = year
    else:
        raise ValueError(f"season must be 'cool', 'warm' or None, got {season!r}")

    sub = index.sel(time=keep)
    sub = sub.assign_coords(season_year=season_year.sel(time=keep))
    return sub.groupby("season_year").mean().rename(season_year="year")


def residual_trend(
    frequency: xr.DataArray, index: xr.DataArray
) -> tuple[np.ndarray, np.ndarray]:
    """Frequency trend before and after removing a linear response to an index.

    The per-type ratio of predicted to observed trend is unstable when a type
    barely changes: the denominator approaches zero and the ratio explodes
    without meaning anything. Regressing the index out first and then fitting
    a trend to what remains avoids that division entirely, and answers the
    question directly -- how much of the change is left once the index is
    accounted for.
    """
    common = np.intersect1d(frequency["year"].values, index["year"].values)
    f = frequency.sel(year=common).values
    x = index.sel(year=common).values.astype(ACCUM)
    years = common.astype(ACCUM)

    raw = np.full(f.shape[1], np.nan)
    residual = np.full(f.shape[1], np.nan)
    x_centred = x - x.mean()

    for i in range(f.shape[1]):
        y = f[:, i]
        ok = np.isfinite(y)
        if ok.sum() < 3:
            continue
        raw[i] = np.polyfit(years[ok], y[ok], 1)[0]

        xi = x_centred[ok]
        if float((xi**2).sum()) == 0:
            residual[i] = raw[i]
            continue
        slope = float((xi * (y[ok] - y[ok].mean())).sum() / (xi**2).sum())
        left = y[ok] - slope * xi
        residual[i] = np.polyfit(years[ok], left, 1)[0]

    return raw, residual


def attributable_fraction(
    frequency: xr.DataArray,
    index: xr.DataArray,
    freq_trend: np.ndarray,
) -> dict[str, np.ndarray]:
    """How much of each frequency trend follows from the trend in the index.

    The predicted trend is the regression slope times the index trend. Its
    ratio to the observed frequency trend is the fraction of the change that a
    linear response to the index would account for.

    A ratio near one means the frequency change is the local expression of the
    index; near zero means it is something else. Ratios above one or below
    zero are possible and simply mean the linear model does not describe the
    relationship, which is itself informative.

    This is a statement about covariance, not causation. The index and the
    type frequencies are both functions of the same circulation, so a high
    ratio shows they move together, not that one drives the other.
    """
    slopes, correlations = regress_on_index(frequency, index)

    years = index["year"].values.astype(ACCUM)
    values = index.values.astype(ACCUM)
    ok = np.isfinite(values)
    index_trend = float(np.polyfit(years[ok], values[ok], 1)[0]) if ok.sum() >= 3 else np.nan

    predicted = slopes * index_trend

    # A ratio is only meaningful where the observed trend is large enough to
    # be a sensible denominator. Below that, the ratio explodes on a near-zero
    # divisor and reports a spurious attribution for a type that barely
    # changed. One tenth of the largest observed trend is the cut.
    scale = float(np.nanmax(np.abs(freq_trend))) if freq_trend.size else 0.0
    usable = np.abs(freq_trend) > 0.1 * scale
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(usable, predicted / freq_trend, np.nan)

    return {
        "slope": slopes,
        "correlation": correlations,
        "index_trend": index_trend,
        "predicted_trend": predicted,
        "observed_trend": freq_trend,
        "ratio": ratio,
    }

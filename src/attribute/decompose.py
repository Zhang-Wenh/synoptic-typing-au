"""Decompose a change in a regional impact into circulation and intensity parts.

The regional mean of an impact y on days stratified into types k is

    ybar = sum_k f_k * ybar_k

where f_k is the fraction of days of type k and ybar_k the mean impact on
those days. Differentiating in time,

    dybar/dt = sum_k (df_k/dt) * ybar_k     "frequency" or dynamic term
             + sum_k f_k * (dybar_k/dt)     "intensity" or thermodynamic term
             + sum_k (df_k/dt) * (dybar_k/dt)   cross term

The first term is the change that would follow from circulation alone if each
type delivered the same rain it always did. The second is the change that
would follow if the circulation never changed but each type became wetter or
drier. The separation is standard in attribution work because the two have
different sources: the frequency term reflects dynamical change, which models
disagree about, while the intensity term reflects thermodynamic change, which
follows more directly from warming and which models handle better.

Two cautions on interpretation.

The split is not a statement about causation. A type becoming wetter may be
thermodynamic, or it may be that the days assigned to that type shifted
systematically within the type's own range while the label stayed the same.
The decomposition cannot distinguish those.

Types are not independent. Frequencies sum to one, so sum_k df_k/dt = 0 and
one type cannot become more common without others becoming less so. Individual
df_k terms should be read as a redistribution, not as separate changes, and
types that are the positive and negative phases of one pattern are especially
tied together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

ACCUM = "float64"


def type_means(
    impact: xr.DataArray, labels: xr.DataArray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mean impact and day count for each type."""
    lab = labels.sel(time=impact["time"]).values
    values = impact.astype(ACCUM).values

    means = np.full(k, np.nan)
    counts = np.zeros(k, dtype=np.int64)
    for i in range(k):
        sel = values[lab == i]
        counts[i] = sel.size
        if sel.size:
            means[i] = sel.mean(dtype=ACCUM)
    return means, counts


def frequencies(labels: xr.DataArray, k: int) -> np.ndarray:
    return np.bincount(labels.values, minlength=k) / labels.sizes["time"]


def yearly_table(
    impact: xr.DataArray, labels: xr.DataArray, k: int, season: str | None = None
) -> xr.Dataset:
    """Per-year frequency and mean impact for each type.

    The trend form of the decomposition needs a time series of f_k and ybar_k,
    not two period averages. Building the yearly table first also makes the
    seasonal restriction explicit rather than hidden inside a groupby.

    Seasons are named by their months rather than by hemisphere convention,
    because "winter" is ambiguous in a bilingual project and because the split
    that matters here is cool season against warm season.
    """
    lab = labels.sel(time=impact["time"])
    month = lab["time"].dt.month
    calendar_year = lab["time"].dt.year

    if season == "cool":
        keep = month.isin([4, 5, 6, 7, 8, 9, 10])
        season_year = calendar_year
    elif season == "warm":
        keep = month.isin([11, 12, 1, 2, 3])
        # The austral warm season straddles the calendar year. Grouping by
        # calendar year would split each summer in two and treat the halves as
        # separate years, which adds a sawtooth to every yearly series and
        # shows up as a large residual in the trend fit. November and December
        # are therefore counted with the January that follows them.
        season_year = xr.where(month >= 11, calendar_year + 1, calendar_year)
    elif season is None:
        keep = xr.ones_like(month, dtype=bool)
        season_year = calendar_year
    else:
        raise ValueError(f"season must be 'cool', 'warm' or None, got {season!r}")

    impact = impact.sel(time=keep)
    lab = lab.sel(time=keep)
    season_year = season_year.sel(time=keep)

    # A straddling season leaves a partial year at each end of the record.
    if season == "warm":
        counts = season_year.groupby(season_year).count()
        full = counts.where(counts > 100, drop=True)[season_year.name].values
        in_full = season_year.isin(full)
        impact = impact.sel(time=in_full)
        lab = lab.sel(time=in_full)
        season_year = season_year.sel(time=in_full)

    years = np.unique(season_year.values)
    freq = np.zeros((years.size, k))
    means = np.full((years.size, k), np.nan)
    counts = np.zeros((years.size, k), dtype=np.int64)

    year_of = season_year.values
    lab_values = lab.values
    impact_values = impact.astype(ACCUM).values

    for j, year in enumerate(years):
        in_year = year_of == year
        n = int(in_year.sum())
        if n == 0:
            continue
        for i in range(k):
            sel = in_year & (lab_values == i)
            counts[j, i] = int(sel.sum())
            freq[j, i] = counts[j, i] / n
            if counts[j, i]:
                means[j, i] = impact_values[sel].mean(dtype=ACCUM)

    return xr.Dataset(
        {
            "frequency": (("year", "type_index"), freq),
            "type_mean": (("year", "type_index"), means),
            "count": (("year", "type_index"), counts),
        },
        coords={"year": years, "type_index": np.arange(k)},
        attrs={"season": season or "all"},
    )


def linear_trend(values: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Least-squares slope per year, ignoring missing values.

    Returns NaN for a series with fewer than three valid points rather than
    fitting a line through two, which would report a slope with no meaning.
    """
    values = np.atleast_2d(values.T).T
    out = np.full(values.shape[1], np.nan)
    x = years.astype(ACCUM)

    for i in range(values.shape[1]):
        y = values[:, i]
        ok = np.isfinite(y)
        if ok.sum() < 3:
            continue
        out[i] = np.polyfit(x[ok], y[ok], 1)[0]
    return out


@dataclass
class Decomposition:
    """Trend in a regional impact, split into its parts."""

    total: float
    frequency_term: float
    intensity_term: float
    cross_term: float

    per_type_frequency: np.ndarray
    per_type_intensity: np.ndarray

    freq_trend: np.ndarray
    mean_trend: np.ndarray
    mean_frequency: np.ndarray
    mean_type_mean: np.ndarray
    season: str

    @property
    def explained(self) -> float:
        return self.frequency_term + self.intensity_term + self.cross_term

    @property
    def residual(self) -> float:
        """Difference between the fitted total and the sum of the parts.

        Should be small. A large residual means the yearly series are not
        well described by straight lines, and the trend form of the
        decomposition is then the wrong tool.
        """
        return self.total - self.explained

    def summary(self) -> str:
        lines = [
            f"season: {self.season}",
            f"  total trend      {self.total:+.4f} per year",
            f"  frequency term   {self.frequency_term:+.4f}",
            f"  intensity term   {self.intensity_term:+.4f}",
            f"  cross term       {self.cross_term:+.4f}",
            f"  residual         {self.residual:+.4f}",
        ]
        return "\n".join(lines)


def decompose(table: xr.Dataset) -> Decomposition:
    """Split the trend in the regional mean into frequency and intensity parts.

    Terms are evaluated at the period means of f_k and ybar_k, which makes the
    two main terms symmetric and puts the interaction in the cross term rather
    than silently in one of them.
    """
    years = table["year"].values
    freq = table["frequency"].values
    means = table["type_mean"].values

    freq_trend = linear_trend(freq, years)
    mean_trend = linear_trend(means, years)

    f_bar = np.nanmean(freq, axis=0)
    y_bar = np.nanmean(means, axis=0)

    per_type_frequency = freq_trend * y_bar
    per_type_intensity = f_bar * mean_trend

    regional = np.nansum(freq * np.nan_to_num(means), axis=1)
    total = float(linear_trend(regional[:, None], years)[0])

    return Decomposition(
        total=total,
        frequency_term=float(np.nansum(per_type_frequency)),
        intensity_term=float(np.nansum(per_type_intensity)),
        cross_term=float(np.nansum(freq_trend * mean_trend)),
        per_type_frequency=per_type_frequency,
        per_type_intensity=per_type_intensity,
        freq_trend=freq_trend,
        mean_trend=mean_trend,
        mean_frequency=f_bar,
        mean_type_mean=y_bar,
        season=str(table.attrs.get("season", "all")),
    )


def _fit_and_residuals(
    values: np.ndarray, years: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Straight-line fit of each column and what it leaves behind."""
    x = years.astype(ACCUM)
    fitted = np.full_like(values, np.nan, dtype=ACCUM)

    for i in range(values.shape[1]):
        y = values[:, i]
        ok = np.isfinite(y)
        if ok.sum() < 3:
            continue
        slope, intercept = np.polyfit(x[ok], y[ok], 1)
        fitted[:, i] = slope * x + intercept

    return fitted, values - fitted


def block_bootstrap(
    table: xr.Dataset, n: int = 1000, block: int = 3, seed: int = 0
) -> dict[str, np.ndarray]:
    """Interval on each term, from a moving-block bootstrap of the residuals.

    Residuals rather than the years themselves. Resampling years directly and
    then regressing against a renumbered axis scrambles the ordering that the
    trend is defined by, and the resulting distribution collapses toward zero
    no matter how strong the real trend is. Here the fitted lines are kept,
    the residuals around them are resampled in blocks, and the decomposition
    is refitted on line plus resampled residual.

    Blocks rather than single years because the impact series is autocorrelated
    at interannual scale through ENSO and the IOD. Resampling single years
    would treat consecutive years as independent and give intervals that are
    too narrow.

    This captures sampling uncertainty only. It says nothing about the
    uncertainty introduced by the classification itself, which needs the
    partition refitted under different seeds -- see `across_partitions`.
    """
    rng = np.random.default_rng(seed)
    years = table["year"].values
    n_years = years.size

    freq = table["frequency"].values
    means = table["type_mean"].values

    freq_fit, freq_res = _fit_and_residuals(freq, years)
    mean_fit, mean_res = _fit_and_residuals(means, years)

    n_blocks = int(np.ceil(n_years / block))
    out = {"total": [], "frequency": [], "intensity": [], "cross": []}

    for _ in range(n):
        starts = rng.integers(0, max(n_years - block + 1, 1), n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])
        idx = idx[idx < n_years][:n_years]

        sample_freq = freq_fit + freq_res[idx]
        sample_mean = mean_fit + mean_res[idx]

        # Frequencies must remain a valid set of shares after resampling.
        sample_freq = np.clip(sample_freq, 0.0, None)
        totals = sample_freq.sum(axis=1, keepdims=True)
        sample_freq = np.divide(
            sample_freq, totals, out=np.zeros_like(sample_freq), where=totals > 0
        )

        sample = xr.Dataset(
            {
                "frequency": (("year", "type_index"), sample_freq),
                "type_mean": (("year", "type_index"), sample_mean),
            },
            coords={"year": years, "type_index": table["type_index"]},
            attrs=dict(table.attrs),
        )
        d = decompose(sample)

        out["total"].append(d.total)
        out["frequency"].append(d.frequency_term)
        out["intensity"].append(d.intensity_term)
        out["cross"].append(d.cross_term)

    return {key: np.array(values) for key, values in out.items()}


def across_partitions(decompositions: list[Decomposition]) -> dict[str, dict]:
    """Spread of each term across classifications fitted with different seeds.

    The stratification is not a fixed property of the data. A three per cent
    perturbation to the anomaly field reassigns about a third of days, so the
    partition itself is a source of uncertainty in every term computed from it.

    A term whose sign is the same across every partition can be reported. A
    term whose sign flips cannot, however tight its bootstrap interval is --
    that interval only describes sampling within one arbitrary partition.
    """
    out = {}
    for name in ("total", "frequency_term", "intensity_term", "cross_term"):
        values = np.array([getattr(d, name) for d in decompositions])
        out[name] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
            "sign_stable": bool(np.all(values > 0) or np.all(values < 0)),
        }
    return out

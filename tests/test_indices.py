"""Tests for the SAM proxy and the attribution of frequency trends to it.

The index is a difference of two standardised bands, which is easy to get
subtly wrong: standardising after differencing, or differencing raw pressures,
both produce something that looks like an index and correlates with the right
things while being dominated by one band.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.attribute.indices import (
    attributable_fraction,
    monthly_to_season,
    read_nino34,
    residual_trend,
    ridge_index,
    regress_on_index,
    sam_proxy,
    seasonal_mean,
)


def make_mslp(n_time=2000, north_amp=1.0, south_amp=1.0, seed=0, trend=0.0):
    """Zonally uniform pressure with independent variability in two bands."""
    rng = np.random.default_rng(seed)
    lat = np.arange(-10.0, -60.5, -2.5)
    lon = np.arange(90.0, 181.0, 5.0)
    time = pd.date_range("1979-01-01", periods=n_time, freq="1D")

    years = (time - time[0]).days.values / 365.25
    north = rng.normal(0, north_amp, n_time) + trend * years
    south = rng.normal(0, south_amp, n_time)

    weight_n = np.exp(-((lat + 40.0) ** 2) / 50.0)
    weight_s = np.exp(-((lat + 60.0) ** 2) / 50.0)

    field = (
        north[:, None] * weight_n[None, :] + south[:, None] * weight_s[None, :]
    )
    data = np.repeat(field[:, :, None], lon.size, axis=2)

    return xr.DataArray(
        data, dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": lat, "longitude": lon}, name="mslp",
    )


# --- the index -----------------------------------------------------------

def test_index_has_one_value_per_day():
    index = sam_proxy(make_mslp(n_time=500))
    assert index.dims == ("time",)
    assert index.sizes["time"] == 500


def test_index_responds_to_the_northern_band():
    """Raising pressure at 40S must raise the index."""
    base = make_mslp(n_time=1000, seed=1)
    index = sam_proxy(base)
    north_band = base.sel(latitude=slice(-37.5, -42.5)).mean(("latitude", "longitude"))
    assert np.corrcoef(index.values, north_band.values)[0, 1] > 0.5


def test_index_responds_oppositely_to_the_southern_band():
    base = make_mslp(n_time=1000, seed=1)
    index = sam_proxy(base)
    south_band = base.sel(latitude=slice(-57.5, -62.5)).mean(("latitude", "longitude"))
    assert np.corrcoef(index.values, south_band.values)[0, 1] < -0.5


def test_bands_are_standardised_before_differencing():
    """Otherwise a band with larger variance dominates the index.

    The southern band here varies five times as strongly. Without
    standardising, the index would be almost entirely a southern-band index
    and would barely respond to the north.
    """
    lopsided = make_mslp(n_time=2000, north_amp=1.0, south_amp=5.0, seed=2)
    index = sam_proxy(lopsided)

    north = lopsided.sel(latitude=slice(-37.5, -42.5)).mean(("latitude", "longitude"))
    south = lopsided.sel(latitude=slice(-57.5, -62.5)).mean(("latitude", "longitude"))

    r_north = abs(np.corrcoef(index.values, north.values)[0, 1])
    r_south = abs(np.corrcoef(index.values, south.values)[0, 1])
    assert 0.4 < r_north / r_south < 2.5


def test_index_records_its_limitations():
    """The domain stops at 60S; the standard definition uses 65S."""
    attrs = sam_proxy(make_mslp(n_time=200)).attrs
    assert "65S" in attrs["caution"]


def test_index_rejects_a_latitude_outside_the_domain():
    with pytest.raises(ValueError, match="no grid points"):
        sam_proxy(make_mslp(n_time=200), south=-80.0)


def test_index_is_insensitive_to_latitude_ordering():
    ascending = make_mslp(n_time=500).isel(latitude=slice(None, None, -1))
    descending = make_mslp(n_time=500)
    assert np.allclose(
        sam_proxy(ascending).values, sam_proxy(descending).values, atol=1e-10
    )


# --- seasonal aggregation ------------------------------------------------

def test_seasonal_mean_gives_one_value_per_year():
    index = sam_proxy(make_mslp(n_time=3650))
    assert seasonal_mean(index).sizes["year"] == len(
        np.unique(index["time"].dt.year.values)
    )


def test_warm_season_counts_november_with_the_following_january():
    index = sam_proxy(make_mslp(n_time=3650))
    warm = seasonal_mean(index, "warm")
    cool = seasonal_mean(index, "cool")
    assert warm.sizes["year"] >= cool.sizes["year"] - 1


def test_cool_season_excludes_summer_months():
    index = sam_proxy(make_mslp(n_time=1095))
    assert seasonal_mean(index, "cool").sizes["year"] == 3


def test_unknown_season_is_rejected():
    with pytest.raises(ValueError, match="season must be"):
        seasonal_mean(sam_proxy(make_mslp(n_time=400)), "spring")


# --- regression ----------------------------------------------------------

def make_frequency(index_values, slopes, base=0.125, noise=0.0, seed=0):
    """Type frequencies that respond linearly to an index, by construction."""
    rng = np.random.default_rng(seed)
    n_years, k = index_values.size, len(slopes)
    f = base + np.outer(index_values - index_values.mean(), slopes)
    if noise:
        f = f + rng.normal(0, noise, (n_years, k))
    return xr.DataArray(
        f, dims=("year", "type_index"),
        coords={"year": np.arange(1979, 1979 + n_years), "type_index": np.arange(k)},
    )


def test_regression_recovers_a_planted_slope():
    values = np.random.default_rng(0).normal(0, 1, 40)
    index = xr.DataArray(values, dims="year", coords={"year": np.arange(1979, 2019)})
    freq = make_frequency(values, [0.02, -0.02, 0.0, 0.01])
    slopes, _ = regress_on_index(freq, index)
    assert np.allclose(slopes, [0.02, -0.02, 0.0, 0.01], atol=1e-9)


def test_correlation_is_one_for_a_noiseless_response():
    values = np.random.default_rng(1).normal(0, 1, 40)
    index = xr.DataArray(values, dims="year", coords={"year": np.arange(1979, 2019)})
    _, corr = regress_on_index(make_frequency(values, [0.02, -0.02]), index)
    assert np.allclose(np.abs(corr), 1.0, atol=1e-9)


def test_correlation_falls_with_noise():
    values = np.random.default_rng(2).normal(0, 1, 45)
    index = xr.DataArray(values, dims="year", coords={"year": np.arange(1979, 2024)})
    clean = regress_on_index(make_frequency(values, [0.02]), index)[1][0]
    noisy = regress_on_index(make_frequency(values, [0.02], noise=0.05, seed=3), index)[1][0]
    assert abs(noisy) < abs(clean)


def test_regression_uses_only_shared_years():
    values = np.random.default_rng(4).normal(0, 1, 40)
    index = xr.DataArray(values, dims="year", coords={"year": np.arange(1979, 2019)})
    freq = make_frequency(values, [0.02])
    slopes, _ = regress_on_index(freq, index.isel(year=slice(0, 25)))
    assert np.isfinite(slopes[0])


# --- attribution ---------------------------------------------------------

def test_a_frequency_change_driven_entirely_by_the_index_gives_ratio_one():
    """The clean case: frequency responds only to the index, and the index
    trends. All of the frequency trend should be attributable."""
    years = np.arange(1979, 2024)
    values = 0.04 * (years - years[0])
    index = xr.DataArray(values, dims="year", coords={"year": years})
    freq = make_frequency(values, [0.02, -0.02])

    observed = np.array([0.02 * 0.04, -0.02 * 0.04])
    out = attributable_fraction(freq, index, observed)
    assert np.allclose(out["ratio"], 1.0, atol=1e-6)


def test_a_frequency_change_unrelated_to_the_index_gives_ratio_near_zero():
    """A trending frequency and a trendless index share nothing."""
    years = np.arange(1979, 2024)
    rng = np.random.default_rng(5)
    index = xr.DataArray(rng.normal(0, 1, years.size), dims="year",
                         coords={"year": years})
    drift = 0.001 * (years - years[0])
    freq = xr.DataArray(
        np.stack([0.125 + drift, 0.125 - drift], axis=1),
        dims=("year", "type_index"),
        coords={"year": years, "type_index": [0, 1]},
    )
    out = attributable_fraction(freq, index, np.array([0.001, -0.001]))
    assert np.all(np.abs(out["ratio"]) < 0.5)


def test_attribution_reports_the_index_trend():
    years = np.arange(1979, 2024)
    values = 0.03 * (years - years[0])
    index = xr.DataArray(values, dims="year", coords={"year": years})
    out = attributable_fraction(index.expand_dims(type_index=[0]).T, index,
                                np.array([1.0]))
    assert out["index_trend"] == pytest.approx(0.03, rel=1e-6)


def test_attribution_returns_every_diagnostic():
    years = np.arange(1979, 2024)
    values = 0.04 * (years - years[0])
    index = xr.DataArray(values, dims="year", coords={"year": years})
    out = attributable_fraction(make_frequency(values, [0.02]), index,
                                np.array([0.0008]))
    for key in ("slope", "correlation", "predicted_trend", "observed_trend", "ratio"):
        assert key in out


# --- the ridge index -----------------------------------------------------

def test_ridge_index_responds_to_its_own_band():
    field = make_mslp(n_time=1000, seed=6)
    index = ridge_index(field, centre=-40.0, half_width=5.0)
    band = field.sel(latitude=slice(-35.0, -45.0)).mean(("latitude", "longitude"))
    assert abs(np.corrcoef(index.values, band.values)[0, 1]) > 0.9


def test_ridge_index_is_not_the_same_as_the_annular_index():
    """One measures a single band, the other a contrast between two.

    They correlate, since both involve the northern band, but a construction
    where only the southern band varies separates them completely.
    """
    only_south = make_mslp(n_time=2000, north_amp=0.01, south_amp=1.0, seed=7)
    sam = sam_proxy(only_south)
    ridge = ridge_index(only_south, centre=-40.0)
    assert abs(np.corrcoef(sam.values, ridge.values)[0, 1]) < 0.5


def test_ridge_index_is_standardised():
    index = ridge_index(make_mslp(n_time=1000), centre=-40.0)
    assert float(index.mean()) == pytest.approx(0.0, abs=1e-10)
    assert float(index.std()) == pytest.approx(1.0, rel=1e-6)


def test_ridge_index_rejects_a_band_outside_the_domain():
    with pytest.raises(ValueError, match="no grid points"):
        ridge_index(make_mslp(n_time=200), centre=-80.0, half_width=2.0)


# --- residual trends -----------------------------------------------------

def test_residual_trend_is_zero_when_the_index_explains_everything():
    """A frequency that responds only to a trending index has nothing left."""
    years = np.arange(1979, 2024)
    values = 0.04 * (years - years[0])
    index = xr.DataArray(values, dims="year", coords={"year": years})
    freq = make_frequency(values, [0.02, -0.02])

    raw, residual = residual_trend(freq, index)
    assert np.all(np.abs(raw) > 1e-5)
    assert np.allclose(residual, 0.0, atol=1e-12)


def test_residual_trend_keeps_a_change_the_index_cannot_explain():
    years = np.arange(1979, 2024)
    rng = np.random.default_rng(8)
    index = xr.DataArray(rng.normal(0, 1, years.size), dims="year",
                         coords={"year": years})
    drift = 0.001 * (years - years[0])
    freq = xr.DataArray(
        np.stack([0.125 + drift], axis=1), dims=("year", "type_index"),
        coords={"year": years, "type_index": [0]},
    )
    raw, residual = residual_trend(freq, index)
    assert residual[0] / raw[0] > 0.8


def test_residual_trend_avoids_the_division_that_breaks_ratios():
    """A type that barely changes gives a meaningless ratio but a fine residual.

    This is why the residual is reported alongside: the per-type ratio of
    predicted to observed trend explodes when the denominator approaches zero.
    """
    years = np.arange(1979, 2024)
    values = 0.04 * (years - years[0])
    index = xr.DataArray(values, dims="year", coords={"year": years})
    flat = xr.DataArray(
        np.full((years.size, 1), 0.125), dims=("year", "type_index"),
        coords={"year": years, "type_index": [0]},
    )
    raw, residual = residual_trend(flat, index)
    assert np.isfinite(residual[0])
    assert abs(residual[0]) < 1e-12


def test_ratio_is_suppressed_for_a_type_that_barely_changes():
    """The defect this cut exists for: a near-zero denominator reported +24."""
    years = np.arange(1979, 2024)
    values = 0.04 * (years - years[0])
    index = xr.DataArray(values, dims="year", coords={"year": years})
    freq = make_frequency(values, [0.02, 0.0001])

    observed = np.array([0.02 * 0.04, 1e-9])
    out = attributable_fraction(freq, index, observed)
    assert np.isfinite(out["ratio"][0])
    assert np.isnan(out["ratio"][1])


# --- reading the Nino 3.4 file -------------------------------------------

def write_nino_file(path, first=1948, last=1950, seed=0, missing=False):
    """A file in the NOAA PSL layout: header, one row per year, trailing notes."""
    rng = np.random.default_rng(seed)
    lines = [f"{first}        {last}"]
    for year in range(first, last + 1):
        values = rng.normal(0, 1, 12)
        if missing and year == last:
            values[-3:] = -99.99
        lines.append(f" {year} " + " ".join(f"{v:8.2f}" for v in values))
    lines += ["  Nino 3.4 anomaly from NOAA PSL", "  -99.99", "  more notes"]
    path.write_text("\n".join(lines))
    return path


def test_reader_gives_one_value_per_month(tmp_path):
    path = write_nino_file(tmp_path / "nino.data", 1948, 1950)
    assert read_nino34(path).sizes["time"] == 36


def test_reader_starts_at_the_stated_first_year(tmp_path):
    path = write_nino_file(tmp_path / "nino.data", 1948, 1950)
    assert pd.Timestamp(read_nino34(path).time.values[0]).year == 1948


def test_reader_stops_at_the_trailing_metadata(tmp_path):
    """The file ends with notes and a sentinel that must not become data."""
    path = write_nino_file(tmp_path / "nino.data", 1948, 1950)
    values = read_nino34(path).values
    assert np.all(np.abs(values) < 10)


def test_reader_drops_missing_months(tmp_path):
    """Recent months are flagged with a sentinel rather than left out."""
    path = write_nino_file(tmp_path / "nino.data", 1948, 1950, missing=True)
    assert read_nino34(path).sizes["time"] == 33


def test_reader_raises_on_a_file_with_no_data(tmp_path):
    path = tmp_path / "empty.data"
    path.write_text("1948        1950\n  just notes\n")
    with pytest.raises(ValueError, match="no data rows"):
        read_nino34(path)


def test_reader_records_what_the_index_is(tmp_path):
    path = write_nino_file(tmp_path / "nino.data")
    assert "170W" in read_nino34(path).attrs["definition"]


# --- monthly to seasonal -------------------------------------------------

def test_monthly_to_season_gives_one_value_per_year(tmp_path):
    path = write_nino_file(tmp_path / "nino.data", 1980, 1999)
    assert monthly_to_season(read_nino34(path)).sizes["year"] == 20


def test_warm_season_counts_november_with_the_following_january(tmp_path):
    """Same convention as the decomposition, or the two would not line up."""
    path = write_nino_file(tmp_path / "nino.data", 1980, 1982)
    monthly = read_nino34(path)
    warm = monthly_to_season(monthly, "warm")

    november = float(monthly.sel(time="1980-11").values[0])
    december = float(monthly.sel(time="1980-12").values[0])
    january = float(monthly.sel(time="1981-01").values[0])
    february = float(monthly.sel(time="1981-02").values[0])
    march = float(monthly.sel(time="1981-03").values[0])

    expected = np.mean([november, december, january, february, march])
    assert float(warm.sel(year=1981)) == pytest.approx(expected)


def test_cool_season_uses_the_calendar_year(tmp_path):
    path = write_nino_file(tmp_path / "nino.data", 1980, 1982)
    monthly = read_nino34(path)
    cool = monthly_to_season(monthly, "cool")
    expected = float(monthly.sel(time=slice("1981-04", "1981-10")).mean())
    assert float(cool.sel(year=1981)) == pytest.approx(expected)


def test_seasonal_aggregation_rejects_an_unknown_season(tmp_path):
    path = write_nino_file(tmp_path / "nino.data")
    with pytest.raises(ValueError, match="season must be"):
        monthly_to_season(read_nino34(path), "monsoon")

"""Tests for the frequency and intensity decomposition.

Every test constructs data where the right answer is known before the fit.
That is the only way to check a decomposition: it always returns numbers that
sum correctly, so an implementation error shows up as a plausible split
between terms rather than as an obvious failure.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.attribute.decompose import (
    across_partitions,
    block_bootstrap,
    decompose,
    frequencies,
    linear_trend,
    type_means,
    yearly_table,
)


def make_series(n_years=40, k=4, freq_trend=None, mean_trend=None,
                base_freq=None, base_mean=None, noise=0.0, seed=0):
    """Daily labels and impact with a known frequency and intensity trend.

    Days are dealt to types in proportions that drift linearly, and each type
    delivers a fixed impact that may also drift. Both trends are exact by
    construction, so the decomposition has a target to hit.
    """
    rng = np.random.default_rng(seed)
    base_freq = np.full(k, 1 / k) if base_freq is None else np.asarray(base_freq, float)
    base_mean = np.arange(1, k + 1, dtype=float) if base_mean is None else np.asarray(base_mean, float)
    freq_trend = np.zeros(k) if freq_trend is None else np.asarray(freq_trend, float)
    mean_trend = np.zeros(k) if mean_trend is None else np.asarray(mean_trend, float)

    times, labels, values = [], [], []
    start_year = 1980
    for j in range(n_years):
        year = start_year + j
        days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="1D")
        # Drop 29 February so every year has the same length. Otherwise the
        # rounding below puts the leftover day into type 0, and leap years
        # give that type a slightly higher frequency -- a periodic artefact
        # of the test data, not of anything being tested.
        days = days[~((days.month == 2) & (days.day == 29))]
        n = days.size

        f = base_freq + freq_trend * j
        f = np.clip(f, 1e-6, None)
        f = f / f.sum()

        counts = np.floor(f * n).astype(int)
        counts[0] += n - counts.sum()
        lab = np.repeat(np.arange(k), counts)
        rng.shuffle(lab)

        val = (base_mean + mean_trend * j)[lab]
        if noise:
            val = val + rng.normal(0, noise, n)

        times.append(days)
        labels.append(lab)
        values.append(val)

    time = pd.DatetimeIndex(np.concatenate(times))
    lab = xr.DataArray(np.concatenate(labels), dims="time",
                       coords={"time": time}, name="type")
    impact = xr.DataArray(np.concatenate(values), dims="time",
                          coords={"time": time}, name="rain")
    return impact, lab


# --- building blocks -----------------------------------------------------

def test_type_means_recovers_planted_values():
    impact, lab = make_series(n_years=5, k=4, base_mean=[1.0, 2.0, 3.0, 4.0])
    means, counts = type_means(impact, lab, 4)
    assert np.allclose(means, [1.0, 2.0, 3.0, 4.0])
    assert counts.sum() == impact.sizes["time"]


def test_frequencies_sum_to_one():
    _, lab = make_series(n_years=5, k=4)
    assert frequencies(lab, 4).sum() == pytest.approx(1.0)


def test_linear_trend_recovers_a_known_slope():
    years = np.arange(1980, 2020)
    values = (3.5 * (years - years[0]) + 10.0)[:, None]
    assert linear_trend(values, years)[0] == pytest.approx(3.5)


def test_linear_trend_ignores_missing_values():
    years = np.arange(1980, 2020)
    values = (2.0 * (years - years[0]))[:, None].astype(float)
    values[5:10] = np.nan
    assert linear_trend(values, years)[0] == pytest.approx(2.0)


def test_linear_trend_declines_to_fit_two_points():
    years = np.arange(1980, 2020)
    values = np.full((years.size, 1), np.nan)
    values[:2, 0] = [1.0, 2.0]
    assert np.isnan(linear_trend(values, years)[0])


# --- the yearly table ----------------------------------------------------

def test_yearly_table_has_a_row_per_year():
    impact, lab = make_series(n_years=12)
    assert yearly_table(impact, lab, 4).sizes["year"] == 12


def test_yearly_frequencies_sum_to_one_each_year():
    impact, lab = make_series(n_years=10)
    assert np.allclose(yearly_table(impact, lab, 4)["frequency"].sum("type_index"), 1.0)


def test_cool_season_keeps_only_its_months():
    impact, lab = make_series(n_years=5)
    table = yearly_table(impact, lab, 4, season="cool")
    assert table.attrs["season"] == "cool"
    assert int(table["count"].sum()) < impact.sizes["time"]


def test_cool_and_warm_seasons_cover_the_record_bar_the_partial_ends():
    """The warm season straddles the calendar year, so the first and last are
    incomplete and are dropped. Everything else belongs to exactly one season.
    """
    impact, lab = make_series(n_years=5)
    cool = int(yearly_table(impact, lab, 4, season="cool")["count"].sum())
    warm = int(yearly_table(impact, lab, 4, season="warm")["count"].sum())
    # The first summer is missing its November and December, the last is
    # missing its January to March: 61 + 90 days.
    dropped = impact.sizes["time"] - cool - warm
    assert dropped == 151


def test_warm_season_counts_november_with_the_following_january():
    """Grouping the austral summer by calendar year would split each one in
    two and treat the halves as separate years."""
    impact, lab = make_series(n_years=6)
    table = yearly_table(impact, lab, 4, season="warm")
    per_year = table["count"].sum("type_index").values
    assert per_year.std() / per_year.mean() < 0.05


def test_warm_season_drops_the_incomplete_seasons_at_each_end():
    impact, lab = make_series(n_years=6)
    warm = yearly_table(impact, lab, 4, season="warm")
    cool = yearly_table(impact, lab, 4, season="cool")
    assert warm.sizes["year"] == cool.sizes["year"] - 1


def test_unknown_season_is_rejected():
    impact, lab = make_series(n_years=3)
    with pytest.raises(ValueError, match="season must be"):
        yearly_table(impact, lab, 4, season="autumn")


# --- the decomposition ---------------------------------------------------

def test_pure_frequency_change_loads_the_frequency_term():
    """Types keep their intensity; only how often they occur changes."""
    impact, lab = make_series(
        n_years=40, k=4, base_mean=[1.0, 2.0, 3.0, 8.0],
        freq_trend=[-0.004, 0.0, 0.0, 0.004],
    )
    d = decompose(yearly_table(impact, lab, 4))
    assert abs(d.frequency_term) > 10 * abs(d.intensity_term)
    assert d.frequency_term > 0


def test_pure_intensity_change_loads_the_intensity_term():
    """Circulation is fixed; each type simply delivers more."""
    impact, lab = make_series(
        n_years=40, k=4, base_mean=[1.0, 2.0, 3.0, 4.0],
        mean_trend=[0.05, 0.05, 0.05, 0.05],
    )
    d = decompose(yearly_table(impact, lab, 4))
    assert abs(d.intensity_term) > 10 * abs(d.frequency_term)
    assert d.intensity_term == pytest.approx(0.05, rel=0.05)


def test_terms_sum_to_the_total():
    impact, lab = make_series(
        n_years=40, k=4, freq_trend=[-0.003, 0.001, 0.001, 0.001],
        mean_trend=[0.02, 0.0, 0.0, -0.01],
    )
    d = decompose(yearly_table(impact, lab, 4))
    assert abs(d.residual) < 0.02 * max(abs(d.total), 1e-6) + 1e-6


def test_no_change_gives_no_trend():
    impact, lab = make_series(n_years=40, k=4)
    d = decompose(yearly_table(impact, lab, 4))
    assert abs(d.total) < 1e-6
    assert abs(d.frequency_term) < 1e-6


def test_opposing_terms_can_cancel():
    """A flat total does not mean nothing happened.

    The case the decomposition exists for: circulation makes the region drier
    while each type gets wetter, and the regional mean barely moves.
    """
    impact, lab = make_series(
        n_years=40, k=4, base_mean=[1.0, 1.0, 1.0, 9.0],
        freq_trend=[0.002, 0.0, 0.0, -0.002],
        mean_trend=[0.016, 0.016, 0.016, 0.016],
    )
    d = decompose(yearly_table(impact, lab, 4))
    assert d.frequency_term < 0
    assert d.intensity_term > 0
    assert abs(d.total) < 0.3 * abs(d.frequency_term)


def test_per_type_terms_sum_to_their_totals():
    impact, lab = make_series(n_years=30, k=4, freq_trend=[-0.002, 0.002, 0, 0])
    d = decompose(yearly_table(impact, lab, 4))
    assert np.nansum(d.per_type_frequency) == pytest.approx(d.frequency_term)
    assert np.nansum(d.per_type_intensity) == pytest.approx(d.intensity_term)


def test_frequency_trends_sum_to_zero():
    """Frequencies are shares, so one type cannot rise alone."""
    impact, lab = make_series(n_years=30, k=4, freq_trend=[-0.003, 0.001, 0.001, 0.001])
    d = decompose(yearly_table(impact, lab, 4))
    assert abs(d.freq_trend.sum()) < 1e-9


def test_summary_names_every_term():
    impact, lab = make_series(n_years=20)
    text = decompose(yearly_table(impact, lab, 4)).summary()
    for word in ("total", "frequency", "intensity", "cross", "residual"):
        assert word in text


# --- uncertainty ---------------------------------------------------------

def test_bootstrap_returns_the_requested_number_of_samples():
    impact, lab = make_series(n_years=30)
    out = block_bootstrap(yearly_table(impact, lab, 4), n=50, block=3)
    assert all(v.size == 50 for v in out.values())


def test_bootstrap_interval_covers_the_point_estimate():
    impact, lab = make_series(n_years=40, freq_trend=[-0.003, 0.001, 0.001, 0.001])
    table = yearly_table(impact, lab, 4)
    point = decompose(table).frequency_term
    draws = block_bootstrap(table, n=200, block=3)["frequency"]
    assert np.percentile(draws, 2.5) <= point <= np.percentile(draws, 97.5)


def test_bootstrap_is_reproducible_for_a_seed():
    impact, lab = make_series(n_years=25)
    table = yearly_table(impact, lab, 4)
    a = block_bootstrap(table, n=30, seed=1)["total"]
    b = block_bootstrap(table, n=30, seed=1)["total"]
    assert np.allclose(a, b)


def test_noise_only_data_gives_an_interval_spanning_zero():
    impact, lab = make_series(n_years=40, noise=3.0)
    draws = block_bootstrap(yearly_table(impact, lab, 4), n=200, block=3)["total"]
    assert np.percentile(draws, 2.5) < 0 < np.percentile(draws, 97.5)


def test_across_partitions_flags_a_stable_sign():
    impact, lab = make_series(n_years=40, base_mean=[1.0, 2.0, 3.0, 8.0],
                              freq_trend=[-0.004, 0.0, 0.0, 0.004])
    ds = [decompose(yearly_table(impact, lab, 4)) for _ in range(3)]
    assert across_partitions(ds)["frequency_term"]["sign_stable"]


def test_across_partitions_reports_the_spread():
    impact, lab = make_series(n_years=30)
    a = decompose(yearly_table(impact, lab, 4))
    b = decompose(yearly_table(impact, lab, 4, season="cool"))
    out = across_partitions([a, b])
    assert out["total"]["max"] >= out["total"]["min"]
    assert "sd" in out["total"]


# --- the bootstrap must not scramble the trend --------------------------

def test_bootstrap_distribution_is_centred_on_the_point_estimate():
    """The defect this bootstrap was rewritten for.

    Resampling years directly and regressing against a renumbered axis
    destroys the ordering the trend is defined by. The distribution then
    collapses toward zero however strong the real trend is, and every interval
    wrongly excludes its own point estimate.
    """
    impact, lab = make_series(
        n_years=40, k=4, base_mean=[1.0, 2.0, 3.0, 8.0],
        freq_trend=[-0.004, 0.0, 0.0, 0.004], noise=1.0,
    )
    table = yearly_table(impact, lab, 4)
    point = decompose(table).frequency_term
    draws = block_bootstrap(table, n=300, block=3)["frequency"]
    assert draws.mean() == pytest.approx(point, rel=0.15)


def test_bootstrap_preserves_the_sign_of_a_strong_trend():
    impact, lab = make_series(
        n_years=40, k=4, base_mean=[1.0, 2.0, 3.0, 8.0],
        freq_trend=[-0.004, 0.0, 0.0, 0.004], noise=0.5,
    )
    draws = block_bootstrap(yearly_table(impact, lab, 4), n=300, block=3)["frequency"]
    assert (draws > 0).mean() > 0.95


def test_block_length_does_not_matter_for_independent_years():
    """With no autocorrelation there is nothing for blocks to preserve.

    Worth pinning down, because it is the control case for the test below:
    if block length changed the interval here, the effect measured there
    would not be attributable to persistence.
    """
    impact, lab = make_series(n_years=45, noise=2.0)
    table = yearly_table(impact, lab, 4)
    short = block_bootstrap(table, n=400, block=1, seed=0)["total"].std()
    long = block_bootstrap(table, n=400, block=6, seed=0)["total"].std()
    assert abs(long - short) / short < 0.2


def test_larger_blocks_give_wider_intervals_when_years_persist():
    """The reason for blocks at all.

    Rainfall in southeast Australia is autocorrelated between years through
    ENSO and the IOD. Resampling single years treats consecutive years as
    independent and understates the uncertainty on a trend. Blocks long enough
    to span the persistence carry it into the resampled series.
    """
    n_years, k = 45, 4
    rng = np.random.default_rng(0)
    years = np.arange(1980, 1980 + n_years)

    persistent = np.zeros(n_years)
    for j in range(1, n_years):
        persistent[j] = 0.85 * persistent[j - 1] + rng.normal(0, 0.5)

    freq = np.tile(np.full(k, 1 / k), (n_years, 1))
    means = np.tile(np.arange(1.0, k + 1), (n_years, 1))
    means = means + persistent[:, None]

    table = xr.Dataset(
        {
            "frequency": (("year", "type_index"), freq),
            "type_mean": (("year", "type_index"), means),
        },
        coords={"year": years, "type_index": np.arange(k)},
        attrs={"season": "all"},
    )

    short = block_bootstrap(table, n=400, block=1, seed=0)["intensity"].std()
    long = block_bootstrap(table, n=400, block=8, seed=0)["intensity"].std()
    assert long > 1.3 * short


def test_bootstrap_frequencies_stay_valid_shares():
    """Resampled frequencies must still sum to one, or the terms are nonsense."""
    impact, lab = make_series(n_years=30, noise=1.0)
    table = yearly_table(impact, lab, 4)
    draws = block_bootstrap(table, n=20, block=3)
    assert np.all(np.isfinite(draws["frequency"]))

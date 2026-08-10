"""Tests for composites and the sequence diagnostics.

The composite tests are straightforward. The sequence tests are the ones that
matter: they check that the cyclic asymmetry measure actually separates a
propagating wave sliced into sectors from a set of independent regimes, since
that distinction is what the diagnostic exists to make.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.cluster.composite import (
    composite,
    cyclic_asymmetry,
    cyclic_order,
    persistence,
    run_lengths,
    seasonal_distribution,
    sequence_report,
    transition_matrix,
)


def make_field(n_time=200, n_lat=4, n_lon=5, seed=0):
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.normal(0, 1, (n_time, n_lat, n_lon)).astype("float32"),
        dims=("time", "latitude", "longitude"),
        coords={
            "time": pd.date_range("1979-01-01", periods=n_time, freq="1D"),
            "latitude": np.linspace(-10, -60, n_lat),
            "longitude": np.linspace(90, 180, n_lon),
        },
        name="mslp",
    )


def make_labels(values, start="1979-01-01"):
    values = np.asarray(values)
    return xr.DataArray(
        values, dims="time",
        coords={"time": pd.date_range(start, periods=values.size, freq="1D")},
        name="type",
    )


# --- composites ----------------------------------------------------------

def test_composite_returns_one_map_per_type():
    field = make_field()
    labels = make_labels(np.tile(np.arange(4), 50))
    out = composite(field, labels, k=4)
    assert out.sizes["type_index"] == 4
    assert out.sizes["latitude"] == 4 and out.sizes["longitude"] == 5


def test_composite_equals_the_mean_of_the_assigned_days():
    field = make_field()
    labels = make_labels(np.tile(np.arange(4), 50))
    out = composite(field, labels, k=4)
    expected = field.isel(time=np.arange(0, 200, 4)).mean("time")
    assert np.allclose(out.sel(type_index=0).values, expected.values)


def test_composite_accumulates_in_float64():
    field = make_field()
    labels = make_labels(np.tile(np.arange(4), 50))
    assert composite(field, labels, k=4).dtype == np.float64


def test_composite_recovers_planted_patterns():
    """Two types with different mean fields must come back separated."""
    n = 200
    base = make_field(n_time=n)
    labels_values = np.tile([0, 1], n // 2)
    shifted = base + xr.DataArray(
        np.where(labels_values == 1, 5.0, -5.0), dims="time",
        coords={"time": base.time},
    )
    out = composite(shifted, make_labels(labels_values), k=2)
    assert float(out.sel(type_index=1).mean()) - float(out.sel(type_index=0).mean()) == pytest.approx(10.0, abs=0.5)


def test_composite_rejects_labels_that_do_not_cover_the_field():
    field = make_field(n_time=200)
    with pytest.raises(ValueError, match="do not cover"):
        composite(field, make_labels(np.zeros(50, dtype=int)), k=2)


# --- seasonality ---------------------------------------------------------

def test_seasonal_counts_sum_to_the_record_length():
    labels = make_labels(np.tile(np.arange(4), 200))
    assert int(seasonal_distribution(labels, k=4).sum()) == 800


def test_seasonal_distribution_detects_a_summer_only_type():
    time = pd.date_range("1979-01-01", "1983-12-31", freq="1D")
    is_summer = np.isin(time.month, [12, 1, 2])
    labels = xr.DataArray(
        np.where(is_summer, 1, 0), dims="time", coords={"time": time}, name="type"
    )
    counts = seasonal_distribution(labels, k=2)
    assert int(counts.sel(type_index=1, month=7)) == 0
    assert int(counts.sel(type_index=1, month=1)) > 0


# --- runs and persistence -----------------------------------------------

def test_run_lengths_split_consecutive_blocks():
    runs = run_lengths(np.array([0, 0, 0, 1, 1, 0, 2]), k=3)
    assert sorted(runs[0].tolist()) == [1, 3]
    assert runs[1].tolist() == [2]


def test_persistence_reports_mean_run_per_type():
    labels = np.repeat([0, 1, 0, 1], [5, 2, 5, 2])
    out = persistence(labels, k=2)
    assert float(out["mean_run"].sel(type_index=0)) == pytest.approx(5.0)
    assert float(out["mean_run"].sel(type_index=1)) == pytest.approx(2.0)


def test_persistence_counts_runs():
    labels = np.repeat([0, 1, 0], [3, 3, 3])
    assert int(persistence(labels, k=2)["n_runs"].sel(type_index=0)) == 2


# --- transitions ---------------------------------------------------------

def test_transition_rows_are_probabilities():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, 500)
    rows = transition_matrix(labels, k=4).sum(axis=1)
    assert np.allclose(rows, 1.0)


def test_transition_counts_are_integers_when_unnormalised():
    labels = np.array([0, 1, 1, 2, 0])
    counts = transition_matrix(labels, k=3, normalise=False)
    assert counts.sum() == 4
    assert counts[1, 1] == 1


def test_persistent_labels_load_the_diagonal():
    labels = np.repeat(np.arange(4), 50)
    assert np.diag(transition_matrix(labels, k=4)).mean() > 0.9


# --- the diagnostic that matters ----------------------------------------

def test_a_strict_cycle_gives_asymmetry_near_one():
    """Sectors of a propagating wave: the sequence always steps forward."""
    labels = np.tile(np.arange(8).repeat(3), 100)
    t = transition_matrix(labels, k=8)
    assert cyclic_asymmetry(t, cyclic_order(t)) > 0.95


def test_a_reversed_cycle_is_also_detected():
    """Direction of travel is a property of the wave, not of the measure."""
    labels = np.tile(np.arange(7, -1, -1).repeat(3), 100)
    t = transition_matrix(labels, k=8)
    assert abs(cyclic_asymmetry(t, cyclic_order(t))) > 0.95


def test_independent_regimes_give_asymmetry_near_zero():
    """Distinct states with no ordering must not look cyclic."""
    rng = np.random.default_rng(1)
    labels = np.repeat(rng.integers(0, 8, 2000), 4)
    t = transition_matrix(labels, k=8)
    assert abs(cyclic_asymmetry(t, cyclic_order(t))) < 0.35


def test_cyclic_order_recovers_a_shuffled_cycle():
    """Cluster indices are arbitrary, so the cycle can appear in any labelling."""
    rng = np.random.default_rng(2)
    relabel = rng.permutation(6)
    labels = relabel[np.tile(np.arange(6).repeat(3), 200)]
    order = cyclic_order(transition_matrix(labels, k=6))

    positions = np.empty(6, dtype=int)
    positions[order] = np.arange(6)
    steps = {(positions[relabel[(i + 1) % 6]] - positions[relabel[i]]) % 6 for i in range(6)}
    assert steps == {1}


def test_cyclic_order_returns_every_type_once():
    rng = np.random.default_rng(3)
    order = cyclic_order(transition_matrix(rng.integers(0, 5, 400), k=5))
    assert sorted(order.tolist()) == list(range(5))


def test_sequence_report_separates_a_cycle_from_regimes():
    """The comparison the report exists to support."""
    cyclic = np.tile(np.arange(8).repeat(3), 100)
    rng = np.random.default_rng(4)
    regimes = np.repeat(rng.integers(0, 8, 2000), 3)

    assert sequence_report(cyclic, 8)["cyclic_asymmetry"] > 0.9
    assert abs(sequence_report(regimes, 8)["cyclic_asymmetry"]) < 0.35


def test_sequence_report_includes_persistence():
    labels = np.tile(np.arange(4).repeat(5), 50)
    report = sequence_report(labels, 4)
    assert report["mean_persistence_days"] == pytest.approx(5.0, rel=0.1)
    assert "transitions" in report

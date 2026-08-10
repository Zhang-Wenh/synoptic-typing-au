"""Tests for the classification and its diagnostics.

The tests that matter here are the ones on the diagnostics, not on k-means
itself. k-means is scikit-learn's and does not need retesting. What needs
testing is whether the classifiability index and the surrogate comparison
actually distinguish structured data from structureless data -- because if
they do not, they would pass silently and give false reassurance about a
classification that means nothing.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.cluster.kmeans import (
    best_partition,
    classifiability,
    fit_once,
    frequencies,
    label_dataarray,
    partition_agreement,
    red_noise_surrogate,
    compare_with_surrogates,
)


def clustered(n=600, k=4, spread=0.35, n_modes=6, seed=0):
    """Well separated blobs: the case a classification should detect."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 4.0, (k, n_modes))
    which = rng.integers(0, k, n)
    return centres[which] + rng.normal(0, spread, (n, n_modes))


def gaussian_cloud(n=600, n_modes=6, seed=0):
    """No cluster structure at all."""
    return np.random.default_rng(seed).normal(0, 1, (n, n_modes))


def annulus(n=600, n_modes=6, seed=0):
    """A propagating wave in PC space: a filled ring, not separated blobs.

    This is the shape the real leading PCs are expected to have, given that
    EOFs 1 to 3 are quadrature pairs of an eastward wave.
    """
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, n)
    radius = 4.0 + rng.normal(0, 0.4, n)
    out = rng.normal(0, 0.5, (n, n_modes))
    out[:, 0] = radius * np.cos(phase)
    out[:, 1] = radius * np.sin(phase)
    return out


# --- partition agreement -------------------------------------------------

def test_identical_partitions_agree_completely():
    labels = np.array([0, 0, 1, 1, 2, 2])
    assert partition_agreement(labels, labels, 3) == pytest.approx(1.0)


def test_agreement_ignores_cluster_relabelling():
    """Cluster indices are arbitrary; only the grouping is meaningful."""
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([2, 2, 0, 0, 1, 1])
    assert partition_agreement(a, b, 3) == pytest.approx(1.0)


def test_agreement_falls_when_grouping_differs():
    a = np.array([0, 0, 0, 1, 1, 1])
    b = np.array([0, 0, 1, 1, 0, 1])
    assert partition_agreement(a, b, 2) < 0.9


def test_agreement_uses_optimal_not_greedy_matching():
    """A greedy match can pick a locally large overlap and lose overall."""
    a = np.array([0] * 10 + [1] * 10 + [2] * 10)
    b = np.array([1] * 10 + [2] * 10 + [0] * 10)
    assert partition_agreement(a, b, 3) == pytest.approx(1.0)


# --- classifiability -----------------------------------------------------

def test_well_separated_clusters_are_highly_reproducible():
    score, _ = classifiability(clustered(k=4), k=4, n_seeds=8)
    assert score > 0.95


def test_classifiability_returns_one_labelling_per_seed():
    _, labels = classifiability(clustered(), k=4, n_seeds=6)
    assert labels.shape == (6, 600)


def test_structureless_data_is_less_reproducible_than_clusters():
    """The comparison the whole diagnostic rests on."""
    real, _ = classifiability(clustered(k=5), k=5, n_seeds=8)
    noise, _ = classifiability(gaussian_cloud(), k=5, n_seeds=8)
    assert real > noise


def test_classifiability_is_bounded():
    score, _ = classifiability(gaussian_cloud(), k=4, n_seeds=6)
    assert 0.0 <= score <= 1.0


# --- surrogates ----------------------------------------------------------

def test_surrogate_matches_variance_of_each_pc():
    pcs = clustered(n=800, n_modes=4)
    surrogate = red_noise_surrogate(pcs, np.random.default_rng(0))
    assert np.allclose(surrogate.std(axis=0), pcs.std(axis=0), rtol=0.25)


def test_surrogate_matches_lag_one_autocorrelation():
    """Matching persistence is what makes the surrogate a fair comparison."""
    rng = np.random.default_rng(1)
    n = 3000
    base = np.zeros((n, 2))
    for j in range(2):
        for t in range(1, n):
            base[t, j] = 0.8 * base[t - 1, j] + rng.normal(0, 0.6)

    surrogate = red_noise_surrogate(base, rng)
    for j in range(2):
        r_base = np.corrcoef(base[:-1, j], base[1:, j])[0, 1]
        r_surr = np.corrcoef(surrogate[:-1, j], surrogate[1:, j])[0, 1]
        assert abs(r_base - r_surr) < 0.1


def test_surrogate_destroys_cluster_structure():
    """Same second-order statistics, no blobs left."""
    pcs = clustered(n=1500, k=4, spread=0.3)
    surrogate = red_noise_surrogate(pcs, np.random.default_rng(2))
    real, _ = classifiability(pcs, 4, n_seeds=6)
    fake, _ = classifiability(surrogate, 4, n_seeds=6)
    assert real > fake


def test_surrogate_preserves_shape():
    pcs = clustered(n=200, n_modes=5)
    assert red_noise_surrogate(pcs, np.random.default_rng(0)).shape == pcs.shape


# --- the full test -------------------------------------------------------

def test_real_clusters_beat_their_surrogates():
    result = compare_with_surrogates(
        clustered(n=800, k=4, spread=0.3), k=4, n_seeds=6, n_surrogates=6
    )
    assert result.exceeds_surrogate
    assert result.margin > 0


def test_gaussian_cloud_does_not_beat_its_surrogates():
    """The diagnostic must not certify structure that is not there.

    A cloud with no clusters is statistically indistinguishable from its own
    surrogate, so the test should decline to call it structured.
    """
    result = compare_with_surrogates(
        gaussian_cloud(n=800), k=5, n_seeds=6, n_surrogates=6
    )
    assert not result.exceeds_surrogate


def test_result_reports_the_surrogate_distribution():
    result = compare_with_surrogates(clustered(n=400), k=4, n_seeds=4, n_surrogates=5)
    assert result.surrogate_scores.size == 5
    assert result.surrogate_p95 >= result.surrogate_mean


# --- the partition actually used ----------------------------------------

def test_best_partition_returns_labels_and_centroids():
    labels, centres = best_partition(clustered(k=4), k=4, n_init=5)
    assert labels.shape == (600,)
    assert centres.shape == (4, 6)


def test_types_are_ordered_by_frequency():
    """Type 0 is the most common, so the numbering means something."""
    rng = np.random.default_rng(0)
    which = np.concatenate([np.zeros(400), np.ones(120), np.full(40, 2)]).astype(int)
    centres = np.array([[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]])
    pcs = centres[which] + rng.normal(0, 0.3, (which.size, 2))

    labels, _ = best_partition(pcs, k=3, n_init=10)
    counts = np.bincount(labels, minlength=3)
    assert list(counts) == sorted(counts, reverse=True)


def test_frequencies_sum_to_one():
    labels, _ = best_partition(clustered(k=4), k=4, n_init=5)
    assert frequencies(labels, 4).sum() == pytest.approx(1.0)


def test_label_dataarray_carries_time():
    labels, _ = best_partition(clustered(n=100, k=3), k=3, n_init=5)
    time = xr.DataArray(pd.date_range("1979-01-01", periods=100), dims="time")
    out = label_dataarray(labels, time)
    assert out.sizes["time"] == 100
    assert out.dims == ("time",)


def test_fit_once_is_deterministic_for_a_seed():
    pcs = clustered()
    assert np.array_equal(fit_once(pcs, 4, seed=3), fit_once(pcs, 4, seed=3))


# --- the case this project is actually in --------------------------------

def test_an_annulus_is_reproducibly_sliced_without_being_clustered():
    """The result to expect here, and the reason the surrogate test exists.

    A propagating wave fills a ring in PC space. k-means cuts it into sectors,
    and those cuts are reproducible across seeds -- the classifiability index
    alone would look reassuring. The surrogate comparison is what reveals that
    a structureless cloud with the same statistics scores similarly.
    """
    ring = annulus(n=800)
    score, _ = classifiability(ring, k=4, n_seeds=8)
    assert score > 0.7

    result = compare_with_surrogates(ring, k=4, n_seeds=6, n_surrogates=6)
    assert result.margin < 0.3

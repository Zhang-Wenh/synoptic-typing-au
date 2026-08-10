"""k-means classification of daily circulation, with the tests that decide
whether the result means anything.

Running k-means is the easy part. It always returns k clusters, always
produces composite maps that look like weather, and never signals that the
data had no cluster structure to begin with. Two questions have to be answered
separately from the fit:

  1. Is the partition reproducible? Different random starts must converge on
     the same partition, or the classification is an artefact of one seed.
     Measured by the classifiability index of Michelangeli et al. (1995).

  2. Is the partition better than one imposed on structureless data? A
     Gaussian cloud cut into k pieces also gives reproducible clusters if k is
     small. The comparison is against surrogate data with the same variance
     spectrum and the same autocorrelation as the real PCs, so the only thing
     the real data has that the surrogate lacks is genuine cluster structure.

The second test matters here more than usual. The leading EOFs of this field
are quadrature pairs of a propagating wave, so the data in PC space is closer
to a filled annulus than to separated blobs. k-means will slice that annulus
into k sectors and the composites will be perfectly interpretable, but they
are a discretisation of a continuum rather than distinct regimes.

That is not fatal. The frequency and intensity decomposition needs a stratifi-
cation that is stable and reproducible; it does not need the types to be
separated by gaps. But the distinction has to be stated rather than assumed.

Reference:
  Michelangeli, Vautard and Legras (1995), J. Atmos. Sci. 52, 1237-1256.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr
from sklearn.cluster import KMeans


def fit_once(pcs: np.ndarray, k: int, seed: int, n_init: int = 1) -> np.ndarray:
    """One k-means partition. Returns labels of shape (n_samples,)."""
    model = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    return model.fit_predict(pcs)


def partition_agreement(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Agreement between two partitions, invariant to cluster relabelling.

    The anomaly correlation of Michelangeli et al. is defined between cluster
    centroids. This uses the simpler and equivalent-in-spirit quantity: the
    best achievable fraction of samples on which the two partitions agree,
    maximised over all matchings of one labelling onto the other.

    The matching is solved exactly with the Hungarian algorithm rather than
    greedily; a greedy match can be badly wrong when two clusters are similar.
    """
    from scipy.optimize import linear_sum_assignment

    table = np.zeros((k, k), dtype=np.int64)
    np.add.at(table, (a, b), 1)
    rows, cols = linear_sum_assignment(-table)
    return float(table[rows, cols].sum() / a.size)


def classifiability(
    pcs: np.ndarray, k: int, n_seeds: int = 20, base_seed: int = 0
) -> tuple[float, np.ndarray]:
    """Mean pairwise agreement across independent random starts.

    A value near 1 means every start finds the same partition. Near 1/k means
    the partitions are unrelated.

    Each start uses n_init=1 deliberately. scikit-learn's default runs several
    starts internally and keeps the best, which would hide exactly the
    instability being measured.
    """
    labels = np.stack(
        [fit_once(pcs, k, seed=base_seed + i, n_init=1) for i in range(n_seeds)]
    )
    scores = [
        partition_agreement(labels[i], labels[j], k)
        for i in range(n_seeds)
        for j in range(i + 1, n_seeds)
    ]
    return float(np.mean(scores)), labels


def red_noise_surrogate(pcs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Surrogate PCs with no cluster structure but matched second-order stats.

    Each PC is replaced by a first-order autoregressive series with the same
    variance and the same lag-1 autocorrelation. The result is a structureless
    Gaussian cloud that is nevertheless as smooth in time and as elongated in
    PC space as the original.

    Matching autocorrelation matters. White noise of the same variance would
    be an easier target to beat, because the real data's day-to-day
    persistence alone raises apparent reproducibility.
    """
    n_time, n_modes = pcs.shape
    out = np.empty_like(pcs)

    for j in range(n_modes):
        series = pcs[:, j]
        sd = series.std(ddof=1)
        r1 = float(np.corrcoef(series[:-1], series[1:])[0, 1])
        r1 = np.clip(r1, -0.99, 0.99)

        noise = rng.normal(0.0, sd * np.sqrt(1 - r1**2), n_time)
        s = np.empty(n_time)
        s[0] = rng.normal(0.0, sd)
        for t in range(1, n_time):
            s[t] = r1 * s[t - 1] + noise[t]
        out[:, j] = s

    return out


@dataclass
class ClassifiabilityResult:
    """Classifiability of the real data against surrogates, for one k."""

    k: int
    observed: float
    surrogate_mean: float
    surrogate_p95: float
    surrogate_scores: np.ndarray = field(repr=False)

    @property
    def exceeds_surrogate(self) -> bool:
        """Whether the real partition is more reproducible than chance.

        Failing this does not invalidate the classification for use as a
        stratification. It means the types should not be described as distinct
        circulation regimes.
        """
        return self.observed > self.surrogate_p95

    @property
    def margin(self) -> float:
        return self.observed - self.surrogate_mean


def compare_with_surrogates(
    pcs: np.ndarray,
    k: int,
    n_seeds: int = 20,
    n_surrogates: int = 20,
    base_seed: int = 0,
) -> ClassifiabilityResult:
    """Compare the real classifiability index with a surrogate distribution."""
    observed, _ = classifiability(pcs, k, n_seeds=n_seeds, base_seed=base_seed)

    rng = np.random.default_rng(base_seed)
    scores = np.array(
        [
            classifiability(
                red_noise_surrogate(pcs, rng), k, n_seeds=n_seeds, base_seed=base_seed
            )[0]
            for _ in range(n_surrogates)
        ]
    )

    return ClassifiabilityResult(
        k=k,
        observed=observed,
        surrogate_mean=float(scores.mean()),
        surrogate_p95=float(np.percentile(scores, 95)),
        surrogate_scores=scores,
    )


def best_partition(pcs: np.ndarray, k: int, seed: int = 0, n_init: int = 50):
    """The partition to actually use, once k is settled.

    Many restarts here, unlike in the classifiability test: the point is no
    longer to measure variability between starts but to take the best fit.
    """
    model = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    labels = model.fit_predict(pcs)

    order = np.argsort(-np.bincount(labels, minlength=k))
    remap = np.empty(k, dtype=np.int64)
    remap[order] = np.arange(k)
    return remap[labels], model.cluster_centers_[order]


def frequencies(labels: np.ndarray, k: int) -> np.ndarray:
    """Fraction of days in each type."""
    return np.bincount(labels, minlength=k) / labels.size


def label_dataarray(labels: np.ndarray, time: xr.DataArray) -> xr.DataArray:
    """Wrap labels with their time coordinate."""
    return xr.DataArray(
        labels, dims="time", coords={"time": time}, name="type",
        attrs={"long_name": "weather type index"},
    )

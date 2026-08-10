"""EOF reduction of the joint MSLP and Z500 anomaly field.

The classification cannot run on the raw field: 201 x 361 grid points for two
variables is 145,161 dimensions against 17,167 days. Distances in a space of
that dimension are nearly all equal, so k-means on the raw field would find
almost nothing. EOF reduces this to a few dozen coordinates that carry most of
the variance.

Three preparation steps happen before the decomposition, each correcting a
different thing. Confusing them is easy and consequential.

  1. Area weighting, sqrt(cos(lat)). A grid cell at 60S covers about half the
     area of one at 10S, so without weighting high latitudes are counted more
     often than the area they represent. This is bookkeeping about the
     coordinate system, not a judgement about the physics.

  2. Variable standardisation, one scalar per variable. MSLP anomalies are
     stored in Pa with a standard deviation of order 700; Z500 anomalies are
     in metres with a standard deviation of order 70. Without rescaling, the
     variance ratio of roughly 100 to 1 means MSLP alone determines the
     result -- and storing MSLP in hPa instead would reverse that. Which
     variable dominates would be decided by a choice of unit.

     Dividing each variable by a single scalar equalises the two totals while
     leaving the spatial variance structure inside each variable untouched.
     The alternative -- dividing each grid point by its own standard deviation
     -- would additionally flatten that structure, inflating tropical MSLP
     variability of 1 to 2 hPa to the same weight as 10 hPa Southern Ocean
     systems. For a study of which circulation delivers rain and heat to
     southeast Australia, that removes the signal rather than a bias.

     The cost is an assumption that the two variables matter equally. That is
     a judgement, not a fact, and it should be tested by repeating the
     classification on MSLP alone.

  3. Joint rather than separate decomposition. The two levels are not
     independent: a developing system tilts westward with height, so the
     relationship between the MSLP and Z500 patterns is itself information.
     Decomposing them together keeps that phase relationship in the leading
     modes; decomposing separately and combining afterwards discards it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from ..preprocess.weights import eof_weights

ACCUM = "float64"


def coarsen(da: xr.DataArray, factor: int, lat_name: str = "latitude") -> xr.DataArray:
    """Block-average the grid before decomposition.

    Two reasons, and the physical one comes first. Weather types are synoptic
    to planetary scale structures, thousands of kilometres across. Defining
    them on a 0.25 degree grid resolves detail far below the scale of the
    patterns being classified, and published synoptic classifications work at
    1 to 2.5 degrees. Nothing about the types is lost by coarsening.

    The practical reason is that the full-resolution joint matrix is about
    20 GB in float64, and an SVD needs several times that in working memory.
    At 1 degree it is under 2 GB.

    Averaging rather than subsampling: a block mean is a low-pass filter,
    while taking every fourth point aliases small-scale variance into the
    large scales.
    """
    if factor <= 1:
        return da
    out = da.coarsen(
        {lat_name: factor, "longitude": factor}, boundary="trim"
    ).mean()
    out.attrs = dict(da.attrs)
    out.attrs["coarsened"] = f"block mean, factor {factor}"
    return out


def standardise_variable(da: xr.DataArray) -> tuple[xr.DataArray, float]:
    """Divide by a single scalar standard deviation over all times and points.

    Returns the scaled field and the scalar, which is needed to map patterns
    back to physical units.
    """
    scale = float(da.astype(ACCUM).std(dtype=ACCUM))
    if scale == 0.0:
        raise ValueError(f"{da.name!r} has zero variance; nothing to decompose")
    return da.astype(ACCUM) / scale, scale


def prepare(
    fields: dict[str, xr.DataArray],
    lat_name: str = "latitude",
    coarsen_factor: int = 1,
) -> tuple[xr.DataArray, dict[str, float]]:
    """Coarsen, weight, standardise and stack several fields into one matrix.

    Output has dims (time, cell), where `cell` runs over variable, latitude
    and longitude together. Scales are returned so that patterns recovered in
    this space can be converted back.

    Coarsening happens before standardisation so that the scale reflects the
    grid the decomposition actually sees.
    """
    weighted, scales = [], {}
    for name, da in fields.items():
        da = coarsen(da, coarsen_factor, lat_name)
        scaled, scale = standardise_variable(da)
        scales[name] = scale
        w = scaled * eof_weights(da[lat_name])
        w = w.rename(name)
        weighted.append(w.expand_dims(variable=[name]))

    joint = xr.concat(weighted, dim="variable", join="exact")
    matrix = joint.stack(cell=("variable", lat_name, "longitude"))
    return matrix.transpose("time", "cell"), scales


@dataclass
class EOFResult:
    """Output of a decomposition, with everything needed to interpret it."""

    pcs: xr.DataArray               # (time, mode) principal components
    patterns: xr.DataArray          # (mode, cell) spatial patterns
    variance_fraction: np.ndarray   # (mode,) fraction of total variance
    eigenvalues: np.ndarray         # (mode,)
    n_samples: int
    scales: dict[str, float]

    def cumulative(self) -> np.ndarray:
        return np.cumsum(self.variance_fraction)

    def n_modes_for(self, fraction: float) -> int:
        """Smallest number of modes reaching a given cumulative variance."""
        return int(np.searchsorted(self.cumulative(), fraction) + 1)

    def north_errors(self) -> np.ndarray:
        """Sampling error on each eigenvalue, after North et al. (1982).

        The rule of thumb is that two modes are only distinguishable if their
        eigenvalues differ by more than this. Modes closer together than their
        error bars are effectively degenerate: their individual patterns are
        arbitrary rotations within a subspace and should not be interpreted
        separately.

        The estimate assumes independent samples. Daily circulation is
        autocorrelated over several days, so the effective sample size is
        smaller than the day count and these errors are optimistic.
        """
        return self.eigenvalues * np.sqrt(2.0 / self.n_samples)

    def north_separable(self) -> np.ndarray:
        """Whether each mode is separable from the next one down."""
        errors = self.north_errors()
        gaps = -np.diff(self.eigenvalues)
        return np.append(gaps > errors[:-1], False)


def decompose(matrix: xr.DataArray, n_modes: int = 50) -> EOFResult:
    """Singular value decomposition of a (time, cell) matrix.

    SVD of the centred data matrix, not eigendecomposition of a covariance
    matrix: forming a 145,161 square covariance matrix is neither necessary
    nor numerically preferable.
    """
    values = np.asarray(matrix.astype(ACCUM).values)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {values.shape}")

    n_samples = values.shape[0]
    n_modes = min(n_modes, *values.shape)

    # The anomalies are already deseasonalised, but centring again costs
    # nothing and guards against a field arriving with a residual mean.
    values = values - values.mean(axis=0, keepdims=True, dtype=ACCUM)

    u, s, vt = np.linalg.svd(values, full_matrices=False)
    u, s, vt = u[:, :n_modes], s[:n_modes], vt[:n_modes]

    eigenvalues = (s**2) / (n_samples - 1)
    total = float((values**2).sum(dtype=ACCUM) / (n_samples - 1))

    modes = np.arange(1, n_modes + 1)
    pcs = xr.DataArray(
        u * s,
        dims=("time", "mode"),
        coords={"time": matrix["time"], "mode": modes},
        name="pcs",
    )
    patterns = xr.DataArray(
        vt,
        dims=("mode", "cell"),
        coords={"mode": modes, "cell": matrix["cell"]},
        name="patterns",
    )

    return EOFResult(
        pcs=pcs,
        patterns=patterns,
        variance_fraction=eigenvalues / total,
        eigenvalues=eigenvalues,
        n_samples=n_samples,
        scales={},
    )


def unstack_pattern(
    patterns: xr.DataArray, mode: int, scales: dict[str, float] | None = None
) -> xr.Dataset:
    """Return one spatial pattern as a map per variable, in physical units.

    Undoes the two preparation steps in reverse: divide out the sqrt(cos)
    weighting, then multiply by each variable's scale.
    """
    da = patterns.sel(mode=mode).unstack("cell")
    ds = da.to_dataset(dim="variable")

    for name in ds.data_vars:
        field = ds[name] / eof_weights(ds["latitude"])
        if scales:
            field = field * scales[name]
        ds[name] = field
    return ds

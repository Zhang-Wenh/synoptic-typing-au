"""Assign model days to the types defined from observations.

The comparison only means something if the model and the observations are
being described in the same terms. Two ways to arrange that:

  Let each model define its own types, then match them to the observed ones by
  pattern similarity. Closer to how the model organises its own circulation,
  but the matching is a further modelling choice, and models with genuinely
  different circulation produce types that match badly and ambiguously.

  Project the model field onto the observed EOF modes and assign each model
  day to the nearest observed centroid. The types are then the same objects by
  construction, and a model whose circulation differs shows that as differences
  in how often each type occurs -- which is the quantity being compared.

The second is used here. It is the simpler claim to defend, but it carries an
assumption worth stating: that the observed modes span the model's circulation
variability well enough for the projection to be meaningful. The residual
variance not captured by the observed modes is reported for each model so that
the assumption can be checked rather than asserted.

Regridding, weighting and standardisation must all be done exactly as they
were for the observations, or the projection lands in a different space than
the one the centroids live in. That is the main thing this module exists to
get right.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from ..preprocess.anomaly import anomaly, daily_mean, detrend
from ..preprocess.weights import eof_weights

ACCUM = "float64"


def regrid_to(model: xr.DataArray, reference: xr.DataArray) -> xr.DataArray:
    """Interpolate a model field onto the grid the EOF modes were built on.

    Bilinear interpolation, which is adequate here because the target grid is
    coarser than most model native grids and the fields are smooth at the
    scales that matter. Conservative regridding would be better for fluxes;
    for pressure it makes no visible difference and costs a dependency.

    Longitudes are normalised to the 0-360 convention first. Models disagree
    on this and a mismatch produces an all-NaN result rather than an error.
    """
    lon_name = "lon" if "lon" in model.coords else "longitude"
    lat_name = "lat" if "lat" in model.coords else "latitude"

    if float(model[lon_name].min()) < 0:
        model = model.assign_coords(
            {lon_name: (model[lon_name] % 360)}
        ).sortby(lon_name)

    model = model.rename({lon_name: "longitude", lat_name: "latitude"})
    if model.latitude.values[0] < model.latitude.values[-1]:
        model = model.isel(latitude=slice(None, None, -1))

    out = model.interp(
        latitude=reference["latitude"],
        longitude=reference["longitude"],
        method="linear",
    )
    if bool(out.isnull().all()):
        raise ValueError(
            "regridding produced only NaN; check that the model domain covers "
            "the reference domain and that longitude conventions agree"
        )
    return out


def prepare_like_observations(
    model: xr.DataArray,
    reference: xr.DataArray,
    scale: float,
    n_harmonics: int = 3,
    do_detrend: bool = True,
    coarsen_factor: int = 1,
) -> xr.DataArray:
    """Put a model field through the same steps the observations went through.

    The scale is the one measured from the observations, not recomputed from
    the model. Recomputing it would rescale each model to unit variance and
    hide exactly the bias being looked for: a model with too little
    circulation variability should project onto weaker PCs, not be normalised
    back to the observed amplitude.
    """
    from ..cluster.eof import coarsen

    out = regrid_to(model, reference)
    out = daily_mean(out)
    out = anomaly(out, n_harmonics=n_harmonics)
    if do_detrend:
        out = detrend(out)
    out = coarsen(out, coarsen_factor)
    return out / scale


def project(
    fields: dict[str, xr.DataArray], patterns: xr.DataArray, lat_name: str = "latitude"
) -> tuple[xr.DataArray, float]:
    """Project prepared model fields onto the observed modes.

    Returns the principal components and the fraction of the model's variance
    that the observed modes fail to capture. A large residual means the model's
    circulation lives partly outside the space the types are defined in, and
    the type frequencies for that model should be treated with suspicion.
    """
    weighted = []
    for name, da in fields.items():
        w = da.astype(ACCUM) * eof_weights(da[lat_name])
        weighted.append(w.rename(name).expand_dims(variable=[name]))

    joint = xr.concat(weighted, dim="variable", join="exact")
    matrix = joint.stack(cell=("variable", lat_name, "longitude")).transpose(
        "time", "cell"
    )

    values = np.asarray(matrix.values, dtype=ACCUM)
    values = values - values.mean(axis=0, keepdims=True, dtype=ACCUM)

    basis = np.asarray(patterns.values, dtype=ACCUM)
    if basis.shape[1] != values.shape[1]:
        raise ValueError(
            f"pattern has {basis.shape[1]} cells but the model field has "
            f"{values.shape[1]}; the two were not prepared the same way"
        )

    pcs = values @ basis.T
    total = float((values**2).sum(dtype=ACCUM))
    captured = float((pcs**2).sum(dtype=ACCUM))
    residual = 1.0 - captured / total if total > 0 else np.nan

    return (
        xr.DataArray(
            pcs,
            dims=("time", "mode"),
            coords={"time": matrix["time"], "mode": patterns["mode"]},
            name="pcs",
        ),
        residual,
    )


def assign(pcs: xr.DataArray, centroids: np.ndarray, pc_scale: float) -> xr.DataArray:
    """Assign each day to the nearest observed centroid.

    `pc_scale` is the standard deviation of the observed leading PC, the same
    constant the observed PCs were divided by before clustering. Applying it
    here rather than rescaling by the model's own leading PC keeps a model with
    weak variability near the centre of the space, where its days will be
    distributed differently -- which is a real difference, not one to normalise
    away.
    """
    values = np.asarray(pcs.values, dtype=ACCUM)[:, : centroids.shape[1]] / pc_scale
    distances = ((values[:, None, :] - centroids[None]) ** 2).sum(axis=2)
    labels = distances.argmin(axis=1)

    return xr.DataArray(
        labels,
        dims="time",
        coords={"time": pcs["time"]},
        name="type",
        attrs={"assignment": "nearest observed centroid in observed PC space"},
    )


@dataclass
class ModelResult:
    """Everything needed to compare one model with the observations."""

    source_id: str
    experiment: str
    labels: xr.DataArray
    frequencies: np.ndarray
    unexplained_variance: float
    n_days: int

    def frequency_bias(self, observed: np.ndarray) -> np.ndarray:
        return self.frequencies - observed

    def total_absolute_bias(self, observed: np.ndarray) -> float:
        """One number for how differently a model distributes its days.

        Half the sum of absolute differences, so the result runs from zero
        (identical) to one (no overlap at all) and is comparable between
        models and between values of k.
        """
        return float(np.abs(self.frequencies - observed).sum() / 2)


def frequencies(labels: xr.DataArray, k: int) -> np.ndarray:
    return np.bincount(labels.values, minlength=k) / labels.sizes["time"]

"""Latitude weighting.

On a regular lat/lon grid, cells shrink toward the poles in proportion to
cos(lat). Two different weights follow from this, and using the wrong one is a
common and silent error:

  area_weights   = cos(lat)        for area-weighted means
  eof_weights    = sqrt(cos(lat))  for EOF / PCA

EOF decomposes variance. Multiplying the data by sqrt(cos) makes the variance
of the weighted data proportional to cos, which is what area weighting the
variance means. Applying cos(lat) directly to the data would weight the
variance by cos squared and over-represent low latitudes.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def area_weights(lat: xr.DataArray) -> xr.DataArray:
    """cos(lat), for area-weighted means and sums."""
    w = np.cos(np.deg2rad(lat))
    w = w.where(w > 0, 0.0)
    w.name = "area_weights"
    w.attrs["long_name"] = "cosine latitude area weights"
    return w


def eof_weights(lat: xr.DataArray) -> xr.DataArray:
    """sqrt(cos(lat)), applied to the data before EOF decomposition."""
    w = np.sqrt(np.cos(np.deg2rad(lat)).clip(min=0.0))
    w.name = "eof_weights"
    w.attrs["long_name"] = "square root cosine latitude weights for EOF"
    return w


def apply_eof_weights(da: xr.DataArray, lat_name: str = "latitude") -> xr.DataArray:
    """Weight a field for EOF. Inverse-weight the resulting patterns to map back."""
    return da * eof_weights(da[lat_name])


def weighted_mean(da: xr.DataArray, lat_name: str = "latitude") -> xr.DataArray:
    """Area-weighted spatial mean over latitude and longitude."""
    dims = [d for d in (lat_name, "longitude") if d in da.dims]
    return da.weighted(area_weights(da[lat_name])).mean(dim=dims)

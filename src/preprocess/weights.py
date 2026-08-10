"""Latitude weighting.

On a regular lat/lon grid, cells shrink toward the poles in proportion to
cos(lat). Two different weights follow, and using the wrong one is a common
and silent error:

  area_weights = cos(lat)        for area-weighted means
  eof_weights  = sqrt(cos(lat))  for EOF / PCA

EOF decomposes variance. Multiplying the data by sqrt(cos) makes the variance
of the weighted data proportional to cos, which is what area weighting the
variance means. Applying cos(lat) directly would weight variance by cos
squared and over-represent low latitudes.

Every reduction here accumulates in float64. ERA5 is stored as float32 and the
record is about 68,000 steps long; a sequential float32 sum of values of order
1e5 drifts by roughly two parts in ten thousand, which on a 1013 hPa field is
0.2 hPa. Small against a 10 hPa synoptic anomaly, but systematic, and it
varies with latitude and season, so it biases the climatology rather than
adding noise to it.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

ACCUM = "float64"


def area_weights(lat: xr.DataArray) -> xr.DataArray:
    """cos(lat), for area-weighted means and sums."""
    w = np.cos(np.deg2rad(lat.astype(ACCUM)))
    w = w.where(w > 0, 0.0)
    w.name = "area_weights"
    w.attrs["long_name"] = "cosine latitude area weights"
    return w


def eof_weights(lat: xr.DataArray) -> xr.DataArray:
    """sqrt(cos(lat)), applied to the data before EOF decomposition."""
    w = np.sqrt(np.cos(np.deg2rad(lat.astype(ACCUM))).clip(min=0.0))
    w.name = "eof_weights"
    w.attrs["long_name"] = "square root cosine latitude weights for EOF"
    return w


def apply_eof_weights(da: xr.DataArray, lat_name: str = "latitude") -> xr.DataArray:
    """Weight a field for EOF.

    Patterns recovered from the weighted data must be divided by the same
    weights to map back to physical units.
    """
    return da.astype(ACCUM) * eof_weights(da[lat_name])


def weighted_mean(da: xr.DataArray, lat_name: str = "latitude") -> xr.DataArray:
    """Area-weighted spatial mean over latitude and longitude."""
    dims = [d for d in (lat_name, "longitude") if d in da.dims]
    return da.astype(ACCUM).weighted(area_weights(da[lat_name])).mean(dim=dims)

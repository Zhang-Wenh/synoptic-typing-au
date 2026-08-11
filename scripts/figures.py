#!/usr/bin/env python
"""Regenerate every figure in docs/results.md.

    python scripts/figures.py
    python scripts/figures.py --tag _nd

Writes PNGs to outputs/. Kept as a script rather than a notebook so the
figures in the repository can be traced to the code that made them.

Nothing here computes anything new: every figure reads a Zarr store written by
an earlier stage. If a store is missing the figure is skipped with a note
rather than failing the whole run, so a partial pipeline still produces what
it can.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.cluster.eof import unstack_pattern  # noqa: E402
from src.config import load_paths  # noqa: E402
from src.preprocess.weights import area_weights  # noqa: E402

EXTENT = [90, 180, -60, -10]
DPI = 150


def projection():
    """Cartopy if available, plain axes otherwise.

    The maps are more useful with coastlines, but cartopy is a heavy optional
    dependency and a missing coastline should not stop the figures being made.
    """
    try:
        import cartopy.crs as ccrs

        return ccrs.PlateCarree()
    except ImportError:
        return None


def make_map_axes(fig, nrows, ncols, proj):
    if proj is None:
        return fig.subplots(nrows, ncols)
    return fig.subplots(nrows, ncols, subplot_kw={"projection": proj})


def draw(ax, field, proj, **kwargs):
    if proj is None:
        field.plot(ax=ax, add_colorbar=False, **kwargs)
    else:
        field.plot(ax=ax, transform=proj, add_colorbar=False, **kwargs)
        ax.set_extent(EXTENT, crs=proj)
        ax.coastlines(linewidth=0.5)
    ax.set_xlabel("")
    ax.set_ylabel("")


def figure_eof_modes(paths, out, tag, proj, n=4):
    path = paths.work / f"eof{tag}.zarr"
    if not path.exists():
        return f"skipped eof modes: {path.name} not found"

    ds = xr.open_zarr(path, consolidated=True)
    patterns = ds["patterns"].set_index(cell=["variable", "latitude", "longitude"])
    fraction = ds["variance_fraction"].values

    fig = plt.figure(figsize=(13, 7))
    axes = make_map_axes(fig, 2, 2, proj)
    for i, ax in enumerate(np.ravel(axes)[:n]):
        field = unstack_pattern(patterns, mode=i + 1)["mslp"]
        lim = float(abs(field).max())
        draw(ax, field, proj, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(f"EOF {i + 1}   {fraction[i] * 100:.1f}%", fontsize=11)

    fig.suptitle("Leading modes of the joint anomaly field, MSLP component")
    fig.tight_layout()
    dest = out / f"eof_modes{tag}.png"
    fig.savefig(dest, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return f"wrote {dest.name}"


def figure_composites(paths, out, tag, proj):
    path = paths.work / f"composites{tag}.zarr"
    if not path.exists():
        return f"skipped composites: {path.name} not found"

    ds = xr.open_zarr(path, consolidated=True)
    mslp = ds["mslp"] / 100.0
    frequency = ds["frequency"].values
    k = mslp.sizes["type_index"]

    lim = float(abs(mslp).max())
    fig = plt.figure(figsize=(19, 8))
    axes = make_map_axes(fig, 2, (k + 1) // 2, proj)
    for ax, t in zip(np.ravel(axes), range(k)):
        draw(ax, mslp.sel(type_index=t), proj, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(f"type {t}   {frequency[t] * 100:.1f}%", fontsize=10)

    fig.suptitle("Mean sea level pressure anomaly composite for each type (hPa)")
    fig.tight_layout()
    dest = out / f"composites{tag}.png"
    fig.savefig(dest, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return f"wrote {dest.name}"


def figure_spectrum(paths, out, tag):
    path = paths.work / f"eof{tag}.zarr"
    if not path.exists():
        return f"skipped spectrum: {path.name} not found"

    fraction = xr.open_zarr(path, consolidated=True)["variance_fraction"].values[:30]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    modes = np.arange(1, fraction.size + 1)

    left.semilogy(modes, fraction * 100, "o-", markersize=4)
    left.set_xlabel("mode")
    left.set_ylabel("% of variance")
    left.grid(alpha=0.3)

    right.plot(modes, np.cumsum(fraction) * 100, "o-", markersize=4)
    for level in (70, 80, 90):
        right.axhline(level, linestyle=":", linewidth=0.8, color="grey")
    right.set_xlabel("mode")
    right.set_ylabel("cumulative % of variance")
    right.grid(alpha=0.3)

    fig.suptitle("Variance spectrum: no break to truncate at")
    fig.tight_layout()
    dest = out / f"spectrum{tag}.png"
    fig.savefig(dest, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return f"wrote {dest.name}"


def figure_advection(paths, out, tag):
    """Meridional wind against temperature and rainfall, across types."""
    path = paths.work / f"composites{tag}.zarr"
    if not path.exists():
        return f"skipped advection: {path.name} not found"

    ds = xr.open_zarr(path, consolidated=True)
    target = ds["mslp"].sel(latitude=slice(-33, -40), longitude=slice(140, 150))

    OMEGA, RHO, DEG = 7.2921e-5, 1.2, 111320.0
    lat = np.deg2rad(target["latitude"])
    f = 2 * OMEGA * np.sin(lat)
    v = (target.differentiate("longitude") / (DEG * np.cos(lat)) / (f * RHO)).mean(
        ("latitude", "longitude")
    ).values

    tmax = per_type_impact(paths, tag, "tmax")
    rain = per_type_impact(paths, tag, "rain")
    if tmax is None:
        return "skipped advection: SILO max_temp not available"

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, values, label in (
        (left, tmax, "mean daily maximum temperature (degC)"),
        (right, rain, "mean rainfall (mm/day)"),
    ):
        if values is None:
            continue
        ax.scatter(v, values)
        for i, (x, y) in enumerate(zip(v, values)):
            ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(5, 3))
        ax.set_xlabel("anomalous meridional geostrophic wind (m/s)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.set_title(f"r = {np.corrcoef(v, values)[0, 1]:+.2f}")

    fig.suptitle("Southerly flow cools; rainfall does not follow the wind direction")
    fig.tight_layout()
    dest = out / f"advection{tag}.png"
    fig.savefig(dest, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return f"wrote {dest.name}"


def per_type_impact(paths, tag, index):
    """Mean impact for each type, computed from the SILO files.

    Not stored by any earlier stage, so it is recomputed here. Returns None if
    the variable was never downloaded, since the temperature series is
    optional to the rest of the pipeline.
    """
    from src.attribute import decompose as dec
    from src.attribute import regional

    variable = "daily_rain" if index == "rain" else "max_temp"
    folder = paths.raw / "silo" / variable
    files = sorted(folder.glob("*.nc")) if folder.exists() else []
    if not files:
        return None

    types_path = paths.work / f"types{tag}.zarr"
    if not types_path.exists():
        return None
    types = xr.open_zarr(types_path, consolidated=True)
    labels = types["type"].load()
    k = int(types.attrs["k"])

    target = {"lat_north": -33.0, "lat_south": -40.0,
              "lon_west": 140.0, "lon_east": 150.0}
    series = regional.build(files, target, varname=variable).compute()
    series = regional.align_to(series, labels)

    means, _ = dec.type_means(series, labels, k)
    return means


def figure_frequency_trends(paths, out, tag):
    path = paths.work / f"composites{tag}.zarr"
    if not path.exists():
        return f"skipped frequency: {path.name} not found"

    ds = xr.open_zarr(path, consolidated=True)
    frequency = ds["frequency"].values
    run = ds["mean_run"].values
    k = frequency.size

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.bar(np.arange(k), frequency * 100)
    left.axhline(100 / k, linestyle=":", color="grey", linewidth=0.8)
    left.set_xlabel("type")
    left.set_ylabel("% of days")
    left.set_title("frequency (dotted line: even split)")

    right.bar(np.arange(k), run)
    right.set_xlabel("type")
    right.set_ylabel("mean run length (days)")
    right.set_title("persistence")

    fig.suptitle("Near-uniform frequency and persistence: the signature of a "
                 "partitioned continuum")
    fig.tight_layout()
    dest = out / f"frequency_persistence{tag}.png"
    fig.savefig(dest, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return f"wrote {dest.name}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    paths = load_paths()
    paths.check()
    out = Path("outputs")
    out.mkdir(exist_ok=True)

    proj = projection()
    if proj is None:
        print("cartopy not installed; maps will have no coastlines")

    for message in (
        figure_spectrum(paths, out, args.tag),
        figure_eof_modes(paths, out, args.tag, proj),
        figure_composites(paths, out, args.tag, proj),
        figure_frequency_trends(paths, out, args.tag),
        figure_advection(paths, out, args.tag),
    ):
        print(f"  {message}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Assign one CMIP6 model's days to the observed weather types.

    python scripts/evaluate.py --list
    python scripts/evaluate.py --model ACCESS-CM2 --dry-run
    python scripts/evaluate.py --model ACCESS-CM2

Deliberately one model at a time to begin with. The way this analysis fails is
silent: if the model field is prepared even slightly differently from the
observations, the projection lands elsewhere in the same-shaped space, nothing
raises, and every model appears biased. Running one model and checking the
diagnostics is how that gets caught. Running five at once is how it gets
mistaken for a real inter-model spread.

Two checks are printed and both should be read before believing anything else:

  unexplained variance   how much of the model's variability lives outside the
                         space the observed modes span. Large means the type
                         frequencies for this model mean little.
  amplitude ratio        model PC spread against observed. Far from one means
                         the model's circulation variability is biased, which
                         will show up as type frequencies pulled toward or away
                         from the centre regardless of pattern skill.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_domain, load_paths, load_sources  # noqa: E402
from src.evaluate import project as proj  # noqa: E402
from src.io import cmip6  # noqa: E402

G0 = 9.80665


def subset_domain(da: xr.DataArray, domain: dict) -> xr.DataArray:
    """Cut a model field to the typing domain, in the 0-360 convention."""
    lon = "lon" if "lon" in da.coords else "longitude"
    lat = "lat" if "lat" in da.coords else "latitude"

    if float(da[lon].min()) < 0:
        da = da.assign_coords({lon: da[lon] % 360}).sortby(lon)

    descending = bool(da[lat].values[0] > da[lat].values[-1])
    lat_slice = (
        slice(domain["lat_north"], domain["lat_south"])
        if descending
        else slice(domain["lat_south"], domain["lat_north"])
    )
    # A margin, so that interpolation onto the reference grid has neighbours
    # at the edges rather than extrapolating.
    margin = 5.0
    return da.sel(
        {
            lat: slice(lat_slice.start + margin, lat_slice.stop - margin)
            if descending
            else slice(lat_slice.start - margin, lat_slice.stop + margin),
            lon: slice(domain["lon_west"] - margin, domain["lon_east"] + margin),
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--experiments", nargs="*", default=["historical", "ssp245"])
    parser.add_argument("--tag", default="_nd")
    parser.add_argument(
        "--reference", default=None,
        help="anomaly store supplying the target grid; defaults to mslp_anom<tag>")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    paths = load_paths()
    paths.check()
    paths.mkdirs()

    domain_cfg = load_domain()
    domain = domain_cfg["typing"]
    start = int(domain_cfg["period"]["start"][:4])
    end = int(domain_cfg["period"]["end"][:4])

    src_cfg = load_sources()["cmip6"]
    catalog = cmip6.open_catalog(src_cfg["catalog"])
    found = cmip6.search(catalog, src_cfg, experiments=args.experiments)

    if args.list:
        complete = cmip6.models_with_all_experiments(found, args.experiments)
        print(f"{len(complete)} models carry every requested experiment:")
        for name in complete:
            print(f"  {name}")
        return 0

    if not args.model:
        parser.error("give --model, or --list to see what is available")

    eof = xr.open_zarr(paths.work / f"eof{args.tag}.zarr", consolidated=True)
    types = xr.open_zarr(paths.work / f"types{args.tag}.zarr", consolidated=True)

    patterns = eof["patterns"].set_index(cell=["variable", "latitude", "longitude"])
    scales = json.loads(eof.attrs["scales"])
    centroids = types["centroid"].values
    k = int(types.attrs["k"])

    reference = xr.open_zarr(
        paths.work / (args.reference or f"mslp_anom{args.tag}.zarr"),
        consolidated=True,
    )
    reference = reference[list(reference.data_vars)[0]]

    observed_pcs = eof["pcs"].isel(mode=slice(0, centroids.shape[1])).load().values
    pc_scale = float(observed_pcs[:, 0].std(ddof=1))
    observed_freq = types["frequency"].values

    print(f"model:       {args.model}")
    print(f"experiments: {', '.join(args.experiments)}")
    print(f"types:       k = {k}, from the observed classification")
    print(f"scales:      {', '.join(f'{a}={b:.2f}' for a, b in scales.items())}")

    if args.dry_run:
        rows = found.df[found.df["source_id"] == args.model]
        if rows.empty:
            print(f"\n{args.model} not found in the catalog")
            return 1
        for _, row in rows.iterrows():
            print(f"  {row['experiment_id']:<12} {row['zstore']}")
        return 0

    t0 = time.time()
    parts = []
    for experiment in args.experiments:
        logging.info("opening %s %s", args.model, experiment)
        ds = cmip6.load_model(found, args.model, experiment)
        field = subset_domain(ds["psl"], domain)
        parts.append(field)

    model = xr.concat(parts, dim="time").sortby("time")
    model = model.sel(time=slice(f"{start}-01-01", f"{end}-12-31"))
    logging.info("%d timesteps on the native grid %s",
                 model.sizes["time"], dict(model.sizes))

    prepared = proj.prepare_like_observations(
        model, reference, scale=scales["mslp"], coarsen_factor=4
    ).compute()

    # Only the MSLP half of the joint basis, since only psl is being used.
    mslp_patterns = patterns.sel(variable="mslp")
    pcs, unexplained = proj.project({"mslp": prepared}, mslp_patterns)

    amplitude = float(pcs.isel(mode=0).std()) / pc_scale
    print(f"\nunexplained variance: {unexplained * 100:.1f}%")
    print(f"amplitude ratio:      {amplitude:.2f}")
    if unexplained > 0.3:
        print("  Large. Much of this model's variability lies outside the space")
        print("  the observed types are defined in; treat its frequencies with care.")
    if not 0.7 < amplitude < 1.4:
        print("  The model's circulation variability is biased in amplitude, which")
        print("  shifts type frequencies regardless of how well patterns match.")

    labels = proj.assign(pcs, centroids, pc_scale)
    model_freq = proj.frequencies(labels, k)

    print(f"\n  type   observed   model     bias")
    for i in range(k):
        print(f"    {i}    {observed_freq[i] * 100:6.2f}%   "
              f"{model_freq[i] * 100:6.2f}%  {(model_freq[i] - observed_freq[i]) * 100:+6.2f}")

    total = float(np.abs(model_freq - observed_freq).sum() / 2)
    print(f"\n  total absolute frequency bias: {total * 100:.1f}%")
    print("  (zero if identical, one hundred if no overlap at all)")

    dest = paths.work / f"model_{args.model}{args.tag}.zarr"
    out = xr.Dataset(
        {"type": labels, "frequency": ("type_index", model_freq)},
        coords={"type_index": np.arange(k)},
    )
    out.attrs.update(
        source_id=args.model,
        experiments=", ".join(args.experiments),
        unexplained_variance=unexplained,
        amplitude_ratio=amplitude,
        assignment="nearest observed centroid; observed modes and scales throughout",
    )
    for name in out.variables:
        out[name].encoding = {}
    out.to_zarr(dest, mode="w", consolidated=True)

    print(f"\nwrote {dest} in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

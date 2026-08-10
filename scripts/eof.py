#!/usr/bin/env python
"""Reduce the joint MSLP and Z500 anomaly field to principal components.

    python scripts/eof.py --dry-run
    python scripts/eof.py
    python scripts/eof.py --vars mslp        # sensitivity check, MSLP alone

Writes the PCs, patterns and spectrum to work/eof.zarr. How many modes to
carry into the classification is decided after looking at the spectrum, not
here: the script reports cumulative variance and the North separability test
and leaves the choice open.
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

from src.cluster import eof  # noqa: E402
from src.config import load_paths  # noqa: E402


def load_fields(work: Path, names: list[str]) -> dict[str, xr.DataArray]:
    fields = {}
    for name in names:
        path = work / f"{name}_anom.zarr"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run scripts/preprocess.py first."
            )
        ds = xr.open_zarr(path, consolidated=True)
        fields[name] = ds[list(ds.data_vars)[0]].load()
    return fields


def report(result: eof.EOFResult, top: int = 15) -> None:
    separable = result.north_separable()
    cumulative = result.cumulative()

    print(f"\n{'mode':>5} {'variance':>10} {'cumulative':>11}  separable")
    for i in range(min(top, len(result.variance_fraction))):
        mark = "yes" if separable[i] else "no"
        print(
            f"{i + 1:>5} {result.variance_fraction[i] * 100:>9.2f}% "
            f"{cumulative[i] * 100:>10.2f}%  {mark}"
        )

    print()
    for target in (0.7, 0.8, 0.9, 0.95):
        print(f"  {target:.0%} of variance needs {result.n_modes_for(target)} modes")

    last = int(np.argmax(~separable)) if not separable.all() else len(separable)
    print(f"\n  first mode not separable from the next: {last + 1}")
    print("  North errors assume independent days; circulation is autocorrelated,")
    print("  so the true effective sample size is smaller and these are optimistic.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vars", nargs="*", default=["mslp", "z"])
    parser.add_argument(
        "--modes", type=int, default=50)
    parser.add_argument(
        "--coarsen", type=int, default=4,
        help="block-average factor. 4 gives 1 degree from ERA5's 0.25")
    parser.add_argument("--out", default="eof.zarr")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S"
    )

    paths = load_paths()
    paths.check()
    paths.mkdirs()

    fields = load_fields(paths.work, args.vars)
    n_time = next(iter(fields.values())).sizes["time"]
    c = max(args.coarsen, 1)
    n_cell = sum(
        (f.sizes["latitude"] // c) * (f.sizes["longitude"] // c)
        for f in fields.values()
    )
    gb = n_time * n_cell * 8 / 1e9

    print(f"variables: {', '.join(args.vars)}")
    print(f"grid:      {0.25 * c:.2f} degrees (coarsen factor {c})")
    print(f"matrix:    {n_time} days x {n_cell} cells, {gb:.2f} GB")

    if args.dry_run:
        print(f"SVD working memory: roughly {gb * 3:.1f} GB")
        return 0

    t0 = time.time()
    matrix, scales = eof.prepare(fields, coarsen_factor=c)
    logging.info("prepared; scales: %s", {k: round(v, 2) for k, v in scales.items()})

    result = eof.decompose(matrix, n_modes=args.modes)
    result.scales = scales
    logging.info("decomposed in %.1f min", (time.time() - t0) / 60)

    report(result)

    dest = paths.work / args.out
    ds = xr.Dataset(
        {
            "pcs": result.pcs,
            "patterns": result.patterns,
            "variance_fraction": ("mode", result.variance_fraction),
            "eigenvalues": ("mode", result.eigenvalues),
        }
    )
    ds.attrs["variables"] = ", ".join(args.vars)
    ds.attrs["scales"] = json.dumps(scales)
    ds.attrs["n_samples"] = result.n_samples
    ds.attrs["preparation"] = (
        f"block mean to {0.25 * c:.2f} degrees; sqrt(cos lat) area weighting; "
        "each variable divided by its own scalar standard deviation; "
        "joint decomposition of both variables"
    )

    for name in ds.variables:
        ds[name].encoding = {}
    # The cell MultiIndex carries a string level, which lands as object dtype
    # and forces xarray to load it to guess a storable type. Fix it explicitly.
    ds = ds.reset_index("cell")
    for name in ds.coords:
        if ds[name].dtype == object:
            ds[name] = ds[name].astype(str)
    ds = ds.chunk({"time": 2000})
    ds.to_zarr(dest, mode="w", consolidated=True)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

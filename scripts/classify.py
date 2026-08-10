#!/usr/bin/env python
"""Classify daily circulation into weather types, and test whether the
classification means anything.

    python scripts/classify.py --sweep              # k = 4..12, diagnostics only
    python scripts/classify.py --sweep --modes 8 12 16
    python scripts/classify.py --k 8                # fit and write

The sweep is the part that answers "how many types". It reports, for each k:

  classifiability  reproducibility across random starts, 0 to 1
  surrogate p95    the same quantity for structureless data with matched
                   variance and autocorrelation
  margin           how much the real data beats that

A large margin means the types are genuinely separated regions of circulation
space. A margin near zero means k-means is slicing a continuum -- still usable
as a stratification for the frequency and intensity decomposition, but not
describable as distinct regimes.

The surrogate loop is the expensive part: n_surrogates x n_seeds fits per k.
Defaults are modest; raise them once the shape of the answer is clear.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cluster import kmeans  # noqa: E402
from src.config import load_paths  # noqa: E402


def load_pcs(work: Path, name: str = "eof.zarr") -> tuple[xr.DataArray, dict]:
    path = work / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/eof.py first.")
    ds = xr.open_zarr(path, consolidated=True)
    return ds["pcs"].load(), dict(ds.attrs)


def normalise_pcs(pcs: np.ndarray) -> np.ndarray:
    """Scale each PC by the leading one, not to unit variance.

    Dividing every PC by its own standard deviation would give a mode holding
    1% of the variance the same weight in the distance metric as one holding
    20%, which discards the ordering the decomposition just established.
    Scaling all modes by a single constant preserves the relative weights.
    """
    return pcs / pcs[:, 0].std(ddof=1)


def sweep(pcs: np.ndarray, ks: list[int], n_seeds: int, n_surrogates: int) -> None:
    print(f"\n{'k':>3} {'observed':>10} {'surr mean':>10} {'surr p95':>10} "
          f"{'margin':>8}  beats p95")
    for k in ks:
        t0 = time.time()
        r = kmeans.compare_with_surrogates(
            pcs, k, n_seeds=n_seeds, n_surrogates=n_surrogates
        )
        mark = "yes" if r.exceeds_surrogate else "no"
        print(f"{k:>3} {r.observed:>10.3f} {r.surrogate_mean:>10.3f} "
              f"{r.surrogate_p95:>10.3f} {r.margin:>8.3f}  {mark:<9} "
              f"({time.time() - t0:.0f}s)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--k", type=int)
    parser.add_argument("--ks", nargs="*", type=int, default=list(range(4, 13)))
    parser.add_argument(
        "--modes", nargs="*", type=int, default=[8],
        help="how many PCs to classify on; several values run a sensitivity check",
    )
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--surrogates", type=int, default=20)
    parser.add_argument("--tag", default="", help="suffix matching eof --tag")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    paths = load_paths()
    paths.check()
    paths.mkdirs()

    pcs_da, attrs = load_pcs(paths.work, f"eof{args.tag}.zarr")
    print(f"PCs: {pcs_da.sizes['time']} days x {pcs_da.sizes['mode']} modes")
    print(f"EOF preparation: {attrs.get('preparation', 'unknown')}")

    if args.sweep:
        for n_modes in args.modes:
            data = normalise_pcs(pcs_da.isel(mode=slice(0, n_modes)).values)
            print(f"\n=== classifying on {n_modes} PCs ===")
            sweep(data, args.ks, args.seeds, args.surrogates)
        print("\nA margin near zero means k-means is partitioning a continuum.")
        print("That remains a valid stratification; it is not a set of regimes.")
        return 0

    if args.k is None:
        parser.error("give --k to fit, or --sweep to choose one")

    n_modes = args.modes[0]
    data = normalise_pcs(pcs_da.isel(mode=slice(0, n_modes)).values)

    labels, centres = kmeans.best_partition(data, args.k, n_init=50)
    freq = kmeans.frequencies(labels, args.k)

    print(f"\nk = {args.k} on {n_modes} PCs")
    for i, f in enumerate(freq):
        print(f"  type {i}: {f * 100:5.2f}%  ({int(f * labels.size):>5d} days)")

    ds = xr.Dataset(
        {
            "type": kmeans.label_dataarray(labels, pcs_da["time"]),
            "centroid": (("type_index", "mode"), centres),
            "frequency": ("type_index", freq),
        },
        coords={"type_index": np.arange(args.k), "mode": pcs_da["mode"][:n_modes]},
    )
    ds.attrs["k"] = args.k
    ds.attrs["n_modes"] = n_modes
    ds.attrs["eof_preparation"] = attrs.get("preparation", "")
    ds.attrs["scales"] = attrs.get("scales", "{}")

    for name in ds.variables:
        ds[name].encoding = {}
    dest = paths.work / (args.out or f"types{args.tag}.zarr")
    ds.to_zarr(dest, mode="w", consolidated=True)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

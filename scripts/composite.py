#!/usr/bin/env python
"""Composite maps and sequence diagnostics for the fitted weather types.

    python scripts/composite.py

Writes work/composites.zarr and prints the diagnostics that decide how the
types can honestly be described.

The transition matrix is the key output. Types that are sectors of a
propagating wave are traversed in a fixed cyclic order, so the sequence steps
one way around the ring far more often than the other. Independent regimes
have no such ordering. The cyclic asymmetry summarises this in one number
between minus one and one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cluster import composite as comp  # noqa: E402
from src.config import load_paths  # noqa: E402

MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


def print_transitions(t: np.ndarray, order: np.ndarray) -> None:
    k = t.shape[0]
    print("\ntransition probabilities, rows reordered along the fitted cycle")
    print("      " + "".join(f"{j:>7d}" for j in order))
    for i in order:
        row = "".join(f"{t[i, j]:>7.3f}" for j in order)
        print(f"{i:>4d}  {row}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="", help="suffix matching classify --tag")
    args = parser.parse_args()

    paths = load_paths()
    paths.check()

    types = xr.open_zarr(paths.work / f"types{args.tag}.zarr", consolidated=True)
    labels = types["type"].load()
    k = int(types.attrs["k"])
    print(f"k = {k}, {labels.sizes['time']} days")

    report = comp.sequence_report(labels.values, k)

    print(f"\nmean persistence:      {report['mean_persistence_days']:.2f} days")
    print(f"spread across types:   {report['persistence_spread']:.2f} days")
    print(f"mean self-transition:  {report['mean_self_transition']:.3f}")
    print(f"cyclic asymmetry:      {report['cyclic_asymmetry']:+.3f}")
    print(f"fitted cycle:          {' -> '.join(str(i) for i in report['cyclic_order'])}")

    a = abs(report["cyclic_asymmetry"])
    if a > 0.5:
        print("\n  Strongly ordered. The types are sectors of a propagating")
        print("  structure rather than independent circulation states.")
    elif a > 0.25:
        print("\n  Partially ordered. Some types follow a preferred sequence.")
    else:
        print("\n  No preferred ordering. Consistent with distinct states.")

    seasonal = comp.seasonal_distribution(labels, k)
    fraction = seasonal / seasonal.sum("month")
    print("\nseasonal distribution, fraction of each type's days per month")
    print("      " + "".join(f"{m:>5s}" for m in MONTHS))
    for i in range(k):
        row = "".join(f"{float(fraction.sel(type_index=i, month=m)) * 100:>5.1f}"
                      for m in range(1, 13))
        print(f"{i:>4d}  {row}")

    composites = {}
    for name in ("mslp", "z"):
        path = paths.work / f"{name}_anom{args.tag}.zarr"
        ds = xr.open_zarr(path, consolidated=True)
        field = ds[list(ds.data_vars)[0]]
        composites[name] = comp.composite(field, labels, k)
        print(f"composited {name}")

    runs = comp.persistence(labels.values, k)
    out = xr.Dataset(
        {
            "mslp": composites["mslp"],
            "z": composites["z"],
            "seasonal_counts": seasonal,
            "frequency": types["frequency"],
            "mean_run": runs["mean_run"],
            "transitions": (("type_index", "type_to"), report["transitions"]),
        },
        coords={"type_to": np.arange(k)},
    )
    out.attrs["k"] = k
    out.attrs["cyclic_asymmetry"] = report["cyclic_asymmetry"]
    out.attrs["cyclic_order"] = ", ".join(str(i) for i in report["cyclic_order"])

    for name in out.variables:
        out[name].encoding = {}
    dest = paths.work / f"composites{args.tag}.zarr"
    out.to_zarr(dest, mode="w", consolidated=True)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

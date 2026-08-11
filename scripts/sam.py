#!/usr/bin/env python
"""Relate the type frequency trends to a Southern Annular Mode proxy.

    python scripts/sam.py --tag _nd

Answers one question: is the change in how often each circulation type occurs
the local expression of the SAM trend, or something else?

Use the non-detrended classification. The detrended one has had the very
signal in question removed from its input, so it understates the frequency
trend by a factor of several.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attribute import decompose as dec  # noqa: E402
from src.attribute import indices  # noqa: E402
from src.config import load_paths  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="_nd")
    parser.add_argument("--seasons", nargs="*", default=["all", "cool", "warm"])
    parser.add_argument("--north", type=float, default=-40.0)
    parser.add_argument("--south", type=float, default=-60.0)
    parser.add_argument("--ridge", type=float, default=-37.5,
                        help="centre latitude of the subtropical ridge band")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    paths = load_paths()
    paths.check()

    anom = xr.open_zarr(paths.work / f"mslp_anom{args.tag}.zarr", consolidated=True)
    mslp = anom[list(anom.data_vars)[0]]

    sam = indices.sam_proxy(mslp, north=args.north, south=args.south).compute()
    ridge = indices.ridge_index(mslp, centre=args.ridge).compute()
    print("SAM proxy:  " + sam.attrs["definition"])
    print("            " + sam.attrs["caution"])
    print("ridge:      " + ridge.attrs["definition"])
    print(f"correlation between the two indices: "
          f"{float(xr.corr(sam, ridge)):+.2f}")

    types = xr.open_zarr(paths.work / f"types{args.tag}.zarr", consolidated=True)
    labels = types["type"].load()
    k = int(types.attrs["k"])

    # A dummy impact: the yearly table only needs labels to build frequencies.
    ones = xr.ones_like(labels, dtype="float64").rename("unit")

    for season in args.seasons:
        s = None if season == "all" else season
        table = dec.yearly_table(ones, labels, k, season=s)
        d = dec.decompose(table)

        print(f"\n=== {season} ===")
        results = {}
        for name, daily in (("SAM", sam), ("ridge", ridge)):
            seasonal_index = indices.seasonal_mean(daily, s)
            out = indices.attributable_fraction(
                table["frequency"], seasonal_index, d.freq_trend
            )
            raw, residual = indices.residual_trend(table["frequency"], seasonal_index)
            results[name] = (out, raw, residual)
            print(f"  {name} trend: {out['index_trend']:+.5f} per year "
                  f"({out['index_trend'] * 47:+.2f} over the record)")

        print(f"\n  type   df/dt      SAM r   left    |   ridge r  left")
        for i in range(k):
            sam_out, raw, sam_res = results["SAM"]
            ridge_out, _, ridge_res = results["ridge"]
            print(f"    {i}   {raw[i]:+.5f}   {sam_out['correlation'][i]:+.2f}  "
                  f"{sam_res[i]:+.5f}  |  {ridge_out['correlation'][i]:+.2f}  "
                  f"{ridge_res[i]:+.5f}")

        for name in ("SAM", "ridge"):
            out, raw, residual = results[name]
            den = float(np.nansum(np.abs(raw)))
            left = float(np.nansum(np.abs(residual)))
            strong = int((np.abs(out["correlation"]) > 0.5).sum())
            if den > 0:
                print(f"\n  {name}: removing it leaves {left / den * 100:.0f}% of the "
                      f"total absolute frequency change; "
                      f"{strong} of {k} types have |r| > 0.5")

    print("\n  Covariance, not causation: the index and the type frequencies are")
    print("  both functions of the same circulation field.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

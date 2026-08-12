#!/usr/bin/env python
"""Decompose impact trends across latitude bands, to see how the balance
between circulation and intensity varies with region.

    python scripts/bands.py --index rain
    python scripts/bands.py --index tmax --labels /path/to/SWT_climatology.nc

Southeast Australia gives a clear answer: almost all of the change in rainfall
and temperature comes from what each weather type delivers, not from how often
each type occurs. Whether that holds elsewhere is a different question, and
there is reason to expect it does not. Tropical rainfall depends heavily on
whether the monsoon is active, which is a question of type frequency; midlatitude
rainfall depends on moisture supply and lifting within frontal systems, which is
a question of intensity.

A gradient in that balance would say something about where circulation change
matters, and it needs a classification covering the whole continent to measure.

Bands are defined by latitude and the whole longitude range; SILO's land mask
removes the ocean, so a band is whatever land falls inside it. The bands are
deliberately broad, since narrow ones would mix the land of one climate zone
with none at all in places.

Each band is reported in its own wet season rather than a common one. The
tropics receive almost all their rain between November and March and the
southwest almost all of it between April and October, so comparing both in one
fixed season would compare a real signal against near-zero.
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
from src.attribute import regional  # noqa: E402
from src.attribute.regional import BANDS  # noqa: E402
from src.config import load_paths  # noqa: E402
from src.io import swt  # noqa: E402

def load_labels(paths, args) -> tuple[xr.DataArray, list[str]]:
    if args.labels:
        return swt.load(args.labels, grouping=args.grouping,
                        shift_days=args.shift_days)

    types = xr.open_zarr(paths.work / f"types{args.tag}.zarr", consolidated=True)
    labels = types["type"].load()
    k = int(types.attrs["k"])
    return labels, [str(i) for i in range(k)]


def run_band(paths, band, labels, k, index, period, boot, block):
    start, end = period
    series = regional.load_index(paths.raw, band, start, end, index).compute()
    series = regional.align_to(series, labels)
    aligned = labels.sel(time=series["time"])

    table = dec.yearly_table(series, aligned, k, season=band["season"])
    d = dec.decompose(table)
    draws = dec.block_bootstrap(table, n=boot, block=block)

    return d, draws, float(series.mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="rain", choices=["rain", "tmax", "hot"])
    parser.add_argument("--labels", default=None)
    parser.add_argument("--grouping", default="regime", choices=["regime", "type"])
    parser.add_argument("--shift-days", type=int, default=0)
    parser.add_argument("--tag", default="_nd")
    parser.add_argument("--bands", nargs="*", default=list(BANDS))
    parser.add_argument("--boot", type=int, default=500)
    parser.add_argument("--block", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    paths = load_paths()
    paths.check()

    from src.config import load_domain

    domain = load_domain()
    period = (int(domain["period"]["start"][:4]), int(domain["period"]["end"][:4]))

    labels, names = load_labels(paths, args)
    k = len(names)
    units = regional.INDEX_UNITS[args.index]

    print(f"index:  {args.index} ({units})")
    print(f"types:  {k} {'published ' + args.grouping + 's' if args.labels else 'fitted'}")
    print(f"\n{'band':<14} {'season':<7} {'mean':>8} {'total/yr':>10} "
          f"{'freq':>10} {'intensity':>10}   freq share")

    results = {}
    for name in args.bands:
        band = BANDS[name]
        try:
            d, draws, mean = run_band(paths, band, labels, k, args.index,
                                      period, args.boot, args.block)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<14} failed: {exc}")
            continue

        results[name] = (d, draws, mean)
        scale = abs(d.frequency_term) + abs(d.intensity_term)
        share = abs(d.frequency_term) / scale * 100 if scale > 0 else np.nan
        season = band["season"] or "all"

        print(f"{name:<14} {season:<7} {mean:>8.3f} {d.total:>+10.5f} "
              f"{d.frequency_term:>+10.5f} {d.intensity_term:>+10.5f}   {share:>5.1f}%")

    print("\n  freq share is the frequency term as a fraction of the two terms")
    print("  combined, so it measures the balance rather than the size of either.")

    print(f"\n{'band':<14} {'term':<11} {'value':>10}  95% interval")
    for name in results:
        d, draws, _ = results[name]
        for label, value, key in (
            ("total", d.total, "total"),
            ("frequency", d.frequency_term, "frequency"),
            ("intensity", d.intensity_term, "intensity"),
        ):
            lo, hi = np.percentile(draws[key], [2.5, 97.5])
            flag = "" if (lo > 0) == (hi > 0) else "  spans zero"
            print(f"{name:<14} {label:<11} {value:>+10.5f}  "
                  f"[{lo:+.5f}, {hi:+.5f}]{flag}")

    print("\n  Bands are reported in their own wet season, so the totals are not")
    print("  directly comparable between bands. The frequency share is.")
    for name in results:
        print(f"  {name}: {BANDS[name]['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

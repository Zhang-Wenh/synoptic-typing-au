#!/usr/bin/env python
"""Decompose the trend in southeast Australian rainfall into a circulation
term and an intensity term.

    python scripts/attribute.py --dry-run
    python scripts/attribute.py
    python scripts/attribute.py --tag _nd          # non-detrended variant
    python scripts/attribute.py --seeds 5          # partition uncertainty

Reports, for each season, the trend in the regional mean and its split into

    frequency term   what circulation change alone would have produced
    intensity term   what a change in how much each type delivers would
    cross term       the interaction

with two kinds of uncertainty. The bootstrap interval covers sampling. The
spread across partitions covers the classification itself, which is not a
fixed property of the data: a three per cent perturbation to the anomaly
field reassigns about a third of days. A term whose sign survives both can be
reported; one whose sign flips across partitions cannot, however tight its
bootstrap interval.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attribute import decompose as dec  # noqa: E402
from src.attribute import regional  # noqa: E402
from src.cluster.kmeans import best_partition, label_dataarray  # noqa: E402
from src.config import load_domain, load_paths  # noqa: E402

SEASONS = ["all", "cool", "warm"]


INDEX_UNITS = {
    "rain": "mm/day",
    "tmax": "degC",
    "hot": "fraction of days",
}


def load_impact(paths, target, start: int, end: int, index: str) -> xr.DataArray:
    """Build the regional daily series for one impact index.

    Three are available and they answer different questions.

      rain  mean daily rainfall
      tmax  mean daily maximum temperature
      hot   fraction of days above the 90th percentile of the same half-year

    The distinction between `tmax` and `hot` decides what the decomposition
    can show. Decomposing mean temperature mostly recovers warming as a
    within-type intensity change, because the whole distribution shifts and
    every type moves with it. Hot-day frequency is where circulation matters:
    whether a day crosses the threshold depends on whether the synoptic
    situation delivers heat, so a change in how often each type occurs
    translates directly into a change in exceedances.
    """
    variable = "daily_rain" if index == "rain" else "max_temp"
    folder = paths.raw / "silo" / variable

    files = sorted(folder.glob("*.nc"))
    files = [f for f in files if start <= int(f.name[:4]) <= end]
    if not files:
        raise FileNotFoundError(
            f"no SILO files for {start}-{end} in {folder}. "
            f"Run: python scripts/fetch_silo.py --variables {variable}"
        )
    logging.info("SILO %s: %d yearly files", variable, len(files))

    series = regional.build(files, target, varname=variable)
    if index == "hot":
        series = regional.hot_day_indicator(series.compute())
        logging.info("%s", series.attrs["definition"])
    return series


def report(d: dec.Decomposition, boot: dict, units: str) -> None:
    print(f"\n=== {d.season} season ===")
    print(f"  mean over period          {np.nansum(d.mean_frequency * d.mean_type_mean):.4f} {units}")

    for name, value, key in [
        ("total trend", d.total, "total"),
        ("frequency term", d.frequency_term, "frequency"),
        ("intensity term", d.intensity_term, "intensity"),
        ("cross term", d.cross_term, "cross"),
    ]:
        lo, hi = np.percentile(boot[key], [2.5, 97.5])
        crosses = "" if (lo > 0) == (hi > 0) else "   spans zero"
        print(f"  {name:<16} {value:+.5f}  [{lo:+.5f}, {hi:+.5f}]{crosses}")

    print(f"  residual         {d.residual:+.5f}")
    # Compared against the largest term, not the total. The total can be near
    # zero because the frequency and intensity terms cancel, and a ratio to it
    # would then flag every well-behaved decomposition.
    scale = max(abs(d.frequency_term), abs(d.intensity_term), abs(d.total), 1e-12)
    if abs(d.residual) > 0.2 * scale:
        print("    Large: the yearly series are not well described by straight")
        print("    lines, so the trend form of the decomposition is a poor fit.")

    print(f"\n  per type: frequency trend, mean impact, contribution")
    for i in range(d.freq_trend.size):
        print(f"    {i}  df/dt {d.freq_trend[i]:+.5f}/yr   "
              f"ybar {d.mean_type_mean[i]:6.3f}   "
              f"freq {d.per_type_frequency[i]:+.5f}   "
              f"int {d.per_type_intensity[i]:+.5f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--index", default="rain", choices=["rain", "tmax", "hot"],
        help="rain, tmax (mean daily maximum), or hot (90th-percentile days)")
    parser.add_argument("--seasons", nargs="*", default=SEASONS)
    parser.add_argument("--boot", type=int, default=1000)
    parser.add_argument("--block", type=int, default=3)
    parser.add_argument(
        "--seeds", type=int, default=1,
        help="refit the partition this many times to measure its contribution",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    paths = load_paths()
    paths.check()
    domain = load_domain()
    target = domain["target"]
    start = int(domain["period"]["start"][:4])
    end = int(domain["period"]["end"][:4])

    types = xr.open_zarr(paths.work / f"types{args.tag}.zarr", consolidated=True)
    labels = types["type"].load()
    k = int(types.attrs["k"])

    rain = load_impact(paths, target, start, end, args.index).compute()
    rain = regional.align_to(rain, labels)
    labels = labels.sel(time=rain["time"])

    units = INDEX_UNITS[args.index]
    print(f"k = {k}, {rain.sizes['time']} days with both circulation and impact")
    print(f"region: {rain.attrs['region']}")
    print(f"index:  {args.index}   mean {float(rain.mean()):.4f} {units}")

    if args.dry_run:
        for season in args.seasons:
            s = None if season == "all" else season
            table = dec.yearly_table(rain, labels, k, season=s)
            print(f"  {season}: {table.sizes['year']} years, "
                  f"{int(table['count'].sum())} days")
        return 0

    t0 = time.time()
    for season in args.seasons:
        s = None if season == "all" else season
        table = dec.yearly_table(rain, labels, k, season=s)
        d = dec.decompose(table)
        boot = dec.block_bootstrap(table, n=args.boot, block=args.block)
        report(d, boot, units)

    if args.seeds > 1:
        print(f"\n=== across {args.seeds} refitted partitions ===")
        eof = xr.open_zarr(paths.work / f"eof{args.tag}.zarr", consolidated=True)
        n_modes = int(types.attrs["n_modes"])
        pcs = eof["pcs"].isel(mode=slice(0, n_modes)).load().values
        pcs = pcs / pcs[:, 0].std(ddof=1)

        for season in args.seasons:
            s = None if season == "all" else season
            results = []
            for seed in range(args.seeds):
                lab, _ = best_partition(pcs, k, seed=seed, n_init=20)
                lab_da = label_dataarray(lab, eof["time"]).sel(time=rain["time"])
                results.append(dec.decompose(dec.yearly_table(rain, lab_da, k, season=s)))

            spread = dec.across_partitions(results)
            print(f"\n  {season}:")
            for name in ("total", "frequency_term", "intensity_term"):
                v = spread[name]
                flag = "stable" if v["sign_stable"] else "SIGN FLIPS"
                print(f"    {name:<16} {v['mean']:+.5f} +/- {v['sd']:.5f}   "
                      f"[{v['min']:+.5f}, {v['max']:+.5f}]   {flag}")

        print("\n  A term whose sign flips across partitions is not a result,")
        print("  however narrow its bootstrap interval.")

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

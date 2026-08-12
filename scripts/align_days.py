#!/usr/bin/env python
"""Find the day offset between the published SWT labels and the impact series.

    python scripts/align_days.py
    python scripts/align_days.py --index tmax

The two records use different day conventions and the offset between them
follows from reasoning that is easy to get backwards:

  The SWT labels are the circulation at 12 UTC, about 10pm in southeastern
  Australia. SILO daily totals run to 9am local time, so a SILO day D covers
  roughly 23 UTC on D-1 to 23 UTC on D, and the 12 UTC snapshot inside that
  window carries the SWT file's D-1 stamp.

That argues for shifting the labels forward by one day. Rather than trusting
it, this script measures it: the correct alignment is the one that maximises
the spread of the impact across types.

The logic is that a stratification only separates days if it is aligned with
them. Offset the labels by a day and each type receives a mixture of the
circulation it names and the ones next to it, which pulls every per-type mean
toward the overall mean. The true offset is a maximum, and a clear peak also
confirms the types carry information about the impact at all.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attribute import regional  # noqa: E402
from src.config import load_domain, load_paths  # noqa: E402
from src.io import swt  # noqa: E402


def spread(impact: xr.DataArray, labels: xr.DataArray, k: int) -> tuple[float, float]:
    """Weighted standard deviation of the per-type means, and the largest gap.

    Weighted by how many days each type holds, so that a rare type with a
    noisy mean cannot dominate. Both measures should peak at the same offset;
    if they disagree the signal is weak and no offset is well determined.
    """
    common = np.intersect1d(impact["time"].values, labels["time"].values)
    if common.size == 0:
        return np.nan, np.nan

    values = impact.sel(time=common).astype("float64").values
    codes = labels.sel(time=common).values

    means = np.full(k, np.nan)
    counts = np.zeros(k)
    for i in range(k):
        sel = values[codes == i]
        counts[i] = sel.size
        if sel.size:
            means[i] = sel.mean(dtype="float64")

    ok = np.isfinite(means) & (counts > 0)
    if ok.sum() < 2:
        return np.nan, np.nan

    weights = counts[ok] / counts[ok].sum()
    centre = float((means[ok] * weights).sum())
    sd = float(np.sqrt((weights * (means[ok] - centre) ** 2).sum()))
    return sd, float(means[ok].max() - means[ok].min())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--index", default="rain", choices=["rain", "tmax", "hot"])
    parser.add_argument("--grouping", default="regime", choices=["regime", "type"])
    parser.add_argument("--offsets", nargs="*", type=int,
                        default=[-3, -2, -1, 0, 1, 2, 3])
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    paths = load_paths()
    paths.check()

    label_path = args.labels or (paths.raw / "swt" / "SWT_climatology.nc")

    domain = load_domain()
    impact = regional.load_index(
        paths.raw, domain["target"],
        int(domain["period"]["start"][:4]), int(domain["period"]["end"][:4]),
        args.index,
    ).compute()

    print(f"index:  {args.index}, {impact.sizes['time']} days")
    print(f"labels: {label_path}")
    print(f"\n  offset   spread   max gap   days")

    best = None
    for offset in args.offsets:
        labels, names = swt.load(label_path, grouping=args.grouping,
                                 shift_days=offset)
        sd, gap = spread(impact, labels, len(names))
        overlap = np.intersect1d(impact["time"].values, labels["time"].values).size
        mark = ""
        if best is None or (np.isfinite(sd) and sd > best[1]):
            best = (offset, sd)
        print(f"  {offset:+3d}    {sd:7.4f}  {gap:7.4f}   {overlap}")

    print(f"\n  strongest separation at an offset of {best[0]:+d} day(s)")
    if best[0] != 0:
        print(f"  Pass --shift-days {best[0]} to scripts/attribute.py.")
    print("\n  A flat profile means the types barely separate this impact, and")
    print("  no offset is well determined. A peak one day wide is what a real")
    print("  alignment looks like, since circulation persists for a few days.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

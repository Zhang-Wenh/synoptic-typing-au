#!/usr/bin/env python
"""Test whether the band results survive removing ENSO.

    python scripts/enso.py --index rain --labels /path/to/SWT_climatology.nc

Tropical Australian rainfall is dominated by ENSO at interannual scale, and a
47-year record holds only about fifteen ENSO cycles. A trend over that record
can be produced by where the record happens to begin and end in the cycle
rather than by any secular change.

The block bootstrap does not test this. It resamples residuals around the
fitted line, so it measures how well the line is determined given that a line
is the right description -- not whether the line is an artefact of a few strong
events. The test here is different: regress each type's yearly frequency and
each type's yearly mean impact on the Nino 3.4 index, then refit the trends to
what is left. A term that keeps its sign and most of its size once ENSO is
removed is a secular change; one that collapses was ENSO.

The same caution applies as with the SAM analysis: this measures covariance,
not causation. ENSO and the type frequencies are both expressions of the same
circulation, so removing one necessarily removes part of the other. The result
is a lower bound on what is not ENSO, not a clean separation.
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
from src.attribute import indices, regional  # noqa: E402
from src.attribute.regional import BANDS  # noqa: E402
from src.config import load_domain, load_paths  # noqa: E402
from src.io import swt  # noqa: E402


def remove_index(table: xr.Dataset, index: xr.DataArray) -> xr.Dataset:
    """Regress the index out of every frequency and type-mean series.

    Returns a table with the same shape, holding what the index does not
    explain. Frequencies are renormalised afterwards, since regressing each
    type separately does not preserve the constraint that they sum to one.
    """
    common = np.intersect1d(table["year"].values, index["year"].values)
    if common.size < 10:
        raise ValueError(
            f"only {common.size} years shared between the table and the index"
        )

    table = table.sel(year=common)
    x = index.sel(year=common).values.astype("float64")
    x = x - x.mean()
    x_var = float((x**2).sum())

    def strip(values):
        out = values.copy()
        for i in range(values.shape[1]):
            y = values[:, i]
            ok = np.isfinite(y)
            if ok.sum() < 3 or x_var == 0:
                continue
            slope = float((x[ok] * (y[ok] - y[ok].mean())).sum() / (x[ok] ** 2).sum())
            out[ok, i] = y[ok] - slope * x[ok]
        return out

    freq = strip(table["frequency"].values)
    freq = np.clip(freq, 0.0, None)
    totals = freq.sum(axis=1, keepdims=True)
    freq = np.divide(freq, totals, out=np.zeros_like(freq), where=totals > 0)

    out = xr.Dataset(
        {
            "frequency": (("year", "type_index"), freq),
            "type_mean": (("year", "type_index"), strip(table["type_mean"].values)),
        },
        coords={"year": common, "type_index": table["type_index"]},
        attrs=dict(table.attrs),
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="rain", choices=["rain", "tmax", "hot"])
    parser.add_argument("--labels", default=None)
    parser.add_argument("--grouping", default="regime")
    parser.add_argument("--nino", default=None)
    parser.add_argument("--bands", nargs="*", default=list(BANDS))
    parser.add_argument("--tag", default="_nd")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    paths = load_paths()
    paths.check()

    nino_path = args.nino or (paths.raw / "nino34.txt")
    nino = indices.read_nino34(nino_path)
    print(f"Nino 3.4: {nino.sizes['time']} months, "
          f"{str(nino.time.values[0])[:7]} to {str(nino.time.values[-1])[:7]}")

    if args.labels:
        labels, names = swt.load(args.labels, grouping=args.grouping)
    else:
        types = xr.open_zarr(paths.work / f"types{args.tag}.zarr", consolidated=True)
        labels = types["type"].load()
        names = [str(i) for i in range(int(types.attrs["k"]))]
    k = len(names)

    domain = load_domain()
    period = (int(domain["period"]["start"][:4]), int(domain["period"]["end"][:4]))

    print(f"\nindex: {args.index}, {k} types")
    print(f"\n{'band':<14} {'term':<11} {'with ENSO':>11} {'without':>11}   kept")

    for name in args.bands:
        band = BANDS[name]
        try:
            series = regional.load_index(
                paths.raw, band, period[0], period[1], args.index
            ).compute()
            series = regional.align_to(series, labels)
            aligned = labels.sel(time=series["time"])

            table = dec.yearly_table(series, aligned, k, season=band["season"])
            seasonal_nino = indices.monthly_to_season(nino, band["season"])

            before = dec.decompose(table)
            after = dec.decompose(remove_index(table, seasonal_nino))
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<14} failed: {exc}")
            continue

        # Correlation between the regional series and ENSO, for context.
        yearly = (table["frequency"] * table["type_mean"].fillna(0)).sum("type_index")
        shared = np.intersect1d(yearly["year"].values, seasonal_nino["year"].values)
        r = float(np.corrcoef(
            yearly.sel(year=shared).values,
            seasonal_nino.sel(year=shared).values,
        )[0, 1])

        print(f"\n{name} (r with Nino 3.4 = {r:+.2f})")
        for label, a, b in (
            ("total", before.total, after.total),
            ("frequency", before.frequency_term, after.frequency_term),
            ("intensity", before.intensity_term, after.intensity_term),
        ):
            kept = b / a * 100 if abs(a) > 1e-12 else np.nan
            flip = "  SIGN FLIPS" if a * b < 0 else ""
            print(f"{'':<14} {label:<11} {a:>+11.5f} {b:>+11.5f}   "
                  f"{kept:>5.0f}%{flip}")

    print("\n  'kept' is the term after removing ENSO as a percentage of before.")
    print("  Near 100 means the change is not ENSO. Near zero, or a sign flip,")
    print("  means the trend was the record's sampling of the ENSO cycle.")
    print("  ENSO and the type frequencies are both expressions of the same")
    print("  circulation, so this is a lower bound on what is not ENSO.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

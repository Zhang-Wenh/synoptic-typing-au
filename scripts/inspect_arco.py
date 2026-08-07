#!/usr/bin/env python
"""Inspect the ARCO-ERA5 store: coverage, dimensions, variable names.

Run this first. It confirms the facts the rest of the pipeline assumes, and
costs nothing but a metadata read.

    python scripts/inspect_arco.py
    python scripts/inspect_arco.py --find pressure temperature
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_sources  # noqa: E402
from src.io import arco  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--find",
        nargs="*",
        default=["sea_level", "geopotential", "temperature"],
        help="substrings to search variable names for",
    )
    parser.add_argument(
        "--all", action="store_true", help="list every variable name"
    )
    args = parser.parse_args()

    cfg = load_sources()["era5"]
    print(f"opening {cfg['store']}")
    ds = arco.open_store(cfg["store"], cfg["storage_options"])

    print("\n--- store ---")
    for key, value in arco.describe(ds).items():
        print(f"{key:20s} {value}")

    if "level" in ds.coords:
        print(f"\nlevels: {ds.level.values}")

    if args.all:
        names = sorted(ds.data_vars)
        print(f"\n--- all {len(names)} variables ---")
        for name in names:
            print(f"  {name}")
    else:
        print(f"\n--- variables matching {args.find} ---")
        for name in arco.find_variables(ds, args.find):
            print(f"  {name:45s} {ds[name].dims}")

    print(
        "\nNote: select by date, never by time index. The time axis may be "
        "longer than the valid coverage window."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

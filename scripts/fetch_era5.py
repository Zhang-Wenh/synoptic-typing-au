#!/usr/bin/env python
"""Slice ERA5 out of ARCO, one Zarr per variable per year.

This is the slow step: the bucket is in us-central1 and the bytes have to
cross the Pacific. Run it once, then read locally.

    python scripts/fetch_era5.py --dry-run          # size estimate only
    python scripts/fetch_era5.py --vars mslp
    python scripts/fetch_era5.py --years 1979 1990
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_domain, load_paths, load_sources  # noqa: E402
from src.io import arco  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vars", nargs="*", default=["mslp", "z"])
    parser.add_argument("--years", nargs=2, type=int, metavar=("START", "END"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S"
    )

    paths = load_paths()
    paths.check()
    paths.mkdirs()

    domain_cfg = load_domain()
    src_cfg = load_sources()["era5"]

    domain = domain_cfg["typing"]
    hours = domain_cfg["sampling"]["hours"]
    start = args.years[0] if args.years else int(domain_cfg["period"]["start"][:4])
    end = args.years[1] if args.years else int(domain_cfg["period"]["end"][:4])

    ds = arco.open_store(src_cfg["store"], src_cfg["storage_options"])
    dest_root = paths.raw / "era5"

    total_gb = 0.0
    t0 = time.time()

    for key in args.vars:
        varname = src_cfg["variables"][key]
        if varname not in ds.data_vars:
            raise KeyError(
                f"'{varname}' not in the store. "
                f"Run scripts/inspect_arco.py --all to find the right name."
            )

        level = src_cfg.get("levels", {}).get(key)
        to_height = key == "z"

        for year in range(start, end + 1):
            da = arco.slice_year(
                ds, varname, year, domain, hours, level=level, to_height=to_height
            )
            gb = arco.estimate_size(da)
            total_gb += gb

            dest = dest_root / key / f"{key}_{year}.zarr"
            if args.dry_run:
                print(f"{dest.name:24s} {dict(da.sizes)}  ~{gb * 1000:.0f} MB")
                continue

            arco.write_year(da, dest)

    elapsed = time.time() - t0
    if args.dry_run:
        print(f"\ntotal estimated transfer: {total_gb:.1f} GB")
    else:
        print(f"\n{total_gb:.1f} GB in {elapsed / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

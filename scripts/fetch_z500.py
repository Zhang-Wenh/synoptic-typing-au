#!/usr/bin/env python
"""Fetch 500 hPa geopotential height from CDS.

Used instead of ARCO for pressure-level variables. ARCO chunks all 37 levels
together, so one level costs the same as all of them; CDS filters level and
area on the server. Measured: about 5.5 minutes per year, against about 2
hours per year via ARCO.

    python scripts/fetch_z500.py --years 1979 1979    # one year first
    python scripts/fetch_z500.py                      # all of it
    python scripts/fetch_z500.py --workers 4

Requires ~/.cdsapirc and acceptance of the dataset licence at
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_domain, load_paths, load_sources  # noqa: E402
from src.io import cds  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs=2, type=int, metavar=("START", "END"))
    parser.add_argument(
        "--workers", type=int, default=3,
        help="concurrent CDS requests; keep this modest, the queue is shared",
    )
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
    level = src_cfg["levels"]["z"]

    start = args.years[0] if args.years else int(domain_cfg["period"]["start"][:4])
    end = args.years[1] if args.years else int(domain_cfg["period"]["end"][:4])
    years = range(start, end + 1)

    if args.dry_run:
        req = cds.build_request("geopotential", level, start, hours, domain)
        print(f"dataset: {cds.DATASET}")
        for k in ("variable", "pressure_level", "time", "area", "data_format"):
            print(f"  {k}: {req[k]}")
        print(f"\n{len(years)} requests, ~5.5 min each, {args.workers} at a time")
        print(f"estimated wall clock: {len(years) * 5.5 / args.workers / 60:.1f} h")
        return 0

    try:
        import cdsapi
    except ImportError:
        print("cdsapi is not installed. conda install -c conda-forge cdsapi")
        return 1

    client = cdsapi.Client()
    t0 = time.time()

    written = cds.fetch(
        client=client,
        years=years,
        variable="geopotential",
        level=level,
        hours=hours,
        domain=domain,
        dest_root=paths.raw / "era5",
        scratch=paths.tmp / "cds",
        workers=args.workers,
        key="z",
    )

    print(f"\n{len(written)} of {len(years)} years in {(time.time() - t0) / 60:.1f} min")
    if len(written) < len(years):
        print("Some years failed. Re-run to retry: completed years are skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

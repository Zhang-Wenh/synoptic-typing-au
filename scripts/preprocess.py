#!/usr/bin/env python
"""Build analysis-ready daily anomalies from the raw yearly Zarr files.

    python scripts/preprocess.py --dry-run
    python scripts/preprocess.py
    python scripts/preprocess.py --vars mslp --no-detrend
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_domain, load_paths  # noqa: E402
from src.preprocess import pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vars", nargs="*", default=["mslp", "z"])
    parser.add_argument("--years", nargs=2, type=int, metavar=("START", "END"))
    parser.add_argument("--harmonics", type=int, default=3)
    parser.add_argument("--no-detrend", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S"
    )

    paths = load_paths()
    paths.check()
    paths.mkdirs()

    period = load_domain()["period"]
    start = args.years[0] if args.years else int(period["start"][:4])
    end = args.years[1] if args.years else int(period["end"][:4])

    raw_root = paths.raw / "era5"

    if args.dry_run:
        for key in args.vars:
            found = pipeline.year_paths(raw_root, key, start, end)
            da = pipeline.open_years(found)
            n_days = len(set(da.indexes["time"].normalize()))
            print(f"{key}: {len(found)} years, {da.sizes['time']} steps "
                  f"-> {n_days} daily, dtype {da.dtype}")
            print(f"     {da.indexes['time'][0]} to {da.indexes['time'][-1]}")
            print(f"     output ~{n_days * da.sizes['latitude'] * da.sizes['longitude'] * 8 / 1e9:.2f} GB")
        return 0

    t0 = time.time()
    for key in args.vars:
        pipeline.run(
            raw_root=raw_root,
            work_root=paths.work,
            key=key,
            start=start,
            end=end,
            n_harmonics=args.harmonics,
            do_detrend=not args.no_detrend,
        )
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

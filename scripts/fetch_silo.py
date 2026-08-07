#!/usr/bin/env python
"""Download SILO gridded climate surfaces.

Roughly 400 MB per variable-year. Safe to interrupt: partial files resume and
complete files are skipped.

    python scripts/fetch_silo.py
    python scripts/fetch_silo.py --years 1979 1985 --variables daily_rain
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_paths, load_sources  # noqa: E402
from src.io import silo  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs=2, type=int, metavar=("START", "END"))
    parser.add_argument("--variables", nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S"
    )

    paths = load_paths()
    paths.check()
    paths.mkdirs()

    cfg = load_sources()["silo"]
    variables = args.variables or cfg["variables"]
    start, end = args.years or (cfg["start_year"], cfg["end_year"])
    dest_root = paths.raw / "silo"

    n_files = len(variables) * (end - start + 1)
    print(f"{n_files} files -> {dest_root}")
    print(f"variables: {', '.join(variables)}   years: {start}-{end}")
    print(f"estimated: ~{n_files * 0.4:.0f} GB\n")

    if args.dry_run:
        for variable in variables:
            print(silo.file_url(cfg["base_url"], variable, start))
        return 0

    written = silo.fetch(dest_root, cfg["base_url"], variables, start, end)
    total = sum(p.stat().st_size for p in written) / 1e9
    print(f"\n{len(written)} files, {total:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

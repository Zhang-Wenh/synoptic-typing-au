"""Download SILO gridded climate surfaces.

SILO is the Queensland Government's gridded Australian climate dataset,
interpolated from Bureau of Meteorology station observations. Licensed CC BY
4.0 and hosted on AWS in ap-southeast-2, so no account and no cross-Pacific
transfer.

Files are one NetCDF per variable per year, roughly 400 MB each for daily
variables. Downloads resume from where they stopped, so an interrupted run
costs only the partial file, not the whole year.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

CHUNK_BYTES = 1 << 22  # 4 MB
MAX_RETRIES = 5
BACKOFF_SECONDS = 3


def file_url(base_url: str, variable: str, year: int) -> str:
    """Build the SILO URL for one variable-year."""
    return f"{base_url}/{variable}/{year}.{variable}.nc"


def remote_size(url: str, timeout: int = 30) -> int | None:
    """Content-Length of the remote file, or None if the server withholds it."""
    try:
        resp = requests.head(url, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        size = resp.headers.get("Content-Length")
        return int(size) if size is not None else None
    except requests.RequestException as exc:
        log.warning("HEAD failed for %s: %s", url, exc)
        return None


def download_file(url: str, dest: Path, timeout: int = 60) -> Path:
    """Download one file, resuming if a partial copy is already on disk.

    A .part suffix marks an incomplete download. The file is only moved to its
    final name once the byte count matches Content-Length, so a file without
    the suffix is always complete.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")

    expected = remote_size(url)

    if dest.exists():
        if expected is None or dest.stat().st_size == expected:
            log.info("skip (already complete): %s", dest.name)
            return dest
        log.warning(
            "size mismatch for %s (%d on disk, %d remote), re-downloading",
            dest.name, dest.stat().st_size, expected,
        )
        dest.unlink()

    for attempt in range(1, MAX_RETRIES + 1):
        have = partial.stat().st_size if partial.exists() else 0

        if expected is not None and have == expected:
            partial.rename(dest)
            return dest

        headers = {"Range": f"bytes={have}-"} if have else {}
        mode = "ab" if have else "wb"

        try:
            with requests.get(
                url, headers=headers, stream=True, timeout=timeout
            ) as resp:
                # 416 means the range is past the end: the file is already whole.
                if resp.status_code == 416 and partial.exists():
                    partial.rename(dest)
                    return dest
                resp.raise_for_status()

                # A server that ignores Range replies 200 and restarts at 0.
                if have and resp.status_code == 200:
                    log.warning("server ignored Range, restarting %s", dest.name)
                    mode, have = "wb", 0

                with open(partial, mode) as fh:
                    for chunk in resp.iter_content(CHUNK_BYTES):
                        fh.write(chunk)

        except requests.RequestException as exc:
            wait = BACKOFF_SECONDS * attempt
            log.warning(
                "attempt %d/%d failed for %s (%s), retrying in %ds",
                attempt, MAX_RETRIES, dest.name, exc, wait,
            )
            time.sleep(wait)
            continue

        got = partial.stat().st_size
        if expected is None or got == expected:
            partial.rename(dest)
            log.info("done: %s (%.1f MB)", dest.name, got / 1e6)
            return dest

        log.warning(
            "incomplete: %s (%d of %d bytes), retrying", dest.name, got, expected
        )

    raise RuntimeError(f"Failed to download {url} after {MAX_RETRIES} attempts")


def fetch(
    dest_root: Path,
    base_url: str,
    variables: list[str],
    start_year: int,
    end_year: int,
) -> list[Path]:
    """Download every variable-year into dest_root/<variable>/.

    Returns the list of local paths. Re-running skips files already complete,
    so this is safe to interrupt and restart.
    """
    written: list[Path] = []
    for variable in variables:
        for year in range(start_year, end_year + 1):
            url = file_url(base_url, variable, year)
            dest = dest_root / variable / f"{year}.{variable}.nc"
            written.append(download_file(url, dest))
    return written

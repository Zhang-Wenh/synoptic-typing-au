"""Configuration loading.

All machine-specific paths live in config/paths.yaml. Nothing else in the
package hardcodes a filesystem location, so moving to a different machine or
onto an HPC filesystem means editing one file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"


def _expand(value: str) -> Path:
    """Expand ~ and environment variables, return an absolute Path."""
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def load_yaml(name: str) -> dict:
    """Load a YAML file from config/ by stem name."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No config file at {path}")
    with open(path) as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class Paths:
    raw: Path
    work: Path
    out: Path
    tmp: Path

    def mkdirs(self) -> None:
        """Create every configured directory. Safe to call repeatedly."""
        for p in (self.raw, self.work, self.out, self.tmp):
            p.mkdir(parents=True, exist_ok=True)

    def check(self) -> None:
        """Fail early if the data volume is not mounted.

        Without this, a script silently writes to a newly created directory on
        the internal disk when the external drive is unplugged, and the mistake
        is only noticed much later.
        """
        for name, p in [("raw", self.raw), ("work", self.work)]:
            parent = p.parent
            if not parent.exists():
                raise RuntimeError(
                    f"Parent of {name}_root does not exist: {parent}\n"
                    f"Is the data volume mounted?"
                )


def load_paths() -> Paths:
    cfg = load_yaml("paths")
    return Paths(
        raw=_expand(cfg["raw_root"]),
        work=_expand(cfg["work_root"]),
        out=_expand(cfg["out_root"]),
        tmp=_expand(cfg["tmp_root"]),
    )


def load_domain() -> dict:
    return load_yaml("domain")


def load_sources() -> dict:
    return load_yaml("sources")

"""Query the Pangeo CMIP6 Zarr catalog on Google Cloud.

Used in the model evaluation stage only. Two things bite here and are handled
explicitly rather than left to defaults:

1. Calendars differ between models (360_day, noleap, proleptic_gregorian).
   use_cftime=True is mandatory, and any day-of-year arithmetic downstream has
   to tolerate a 360-day year.
2. Native grids differ between models. Anything comparing models to each other
   or to ERA5 has to regrid first.
"""

from __future__ import annotations

import logging

import intake
import xarray as xr

log = logging.getLogger(__name__)


def open_catalog(url: str):
    """Open the intake-esm catalog. Use the QC'd one, not the -noQC variant."""
    return intake.open_esm_datastore(url)


def search(catalog, cfg: dict, experiments: list[str] | None = None):
    """Filter the catalog down to the configured variable and experiments."""
    return catalog.search(
        variable_id=cfg["variable_id"],
        table_id=cfg["table_id"],
        grid_label=cfg["grid_label"],
        member_id=cfg["member_id"],
        experiment_id=experiments or cfg["experiments"],
    )


def available_models(subset) -> list[str]:
    """Source IDs present in a search result."""
    return sorted(subset.df["source_id"].unique())


def models_with_all_experiments(subset, experiments: list[str]) -> list[str]:
    """Models that carry every requested experiment.

    Dropping models that are missing an experiment keeps the ensemble balanced.
    An unbalanced ensemble makes a multi-model mean hard to interpret, because
    the composition changes between periods.
    """
    df = subset.df
    counts = df.groupby("source_id")["experiment_id"].nunique()
    return sorted(counts[counts == len(experiments)].index)


def to_datasets(subset) -> dict[str, xr.Dataset]:
    """Load the search result into a dict of lazy datasets."""
    return subset.to_dataset_dict(
        xarray_open_kwargs={"consolidated": True, "use_cftime": True},
        storage_options={"token": "anon"},
    )


def calendar_of(ds: xr.Dataset) -> str:
    """Calendar name for a dataset, for logging and sanity checks."""
    return getattr(ds.time.values[0], "calendar", "unknown")


def load_model(subset, source_id: str, experiment: str) -> xr.Dataset:
    """Open one model and experiment as a lazy dataset.

    intake-esm keys are built from the catalog's grouping columns and their
    exact form varies with the catalog version, so the key is found by
    matching on the two fields that matter rather than being constructed.
    """
    rows = subset.df[
        (subset.df["source_id"] == source_id)
        & (subset.df["experiment_id"] == experiment)
    ]
    if rows.empty:
        available = sorted(subset.df["source_id"].unique())
        raise KeyError(
            f"{source_id} has no {experiment} in this search; "
            f"{len(available)} models available"
        )

    narrowed = subset.search(source_id=source_id, experiment_id=experiment)
    datasets = to_datasets(narrowed)
    if len(datasets) != 1:
        raise ValueError(
            f"expected one dataset for {source_id} {experiment}, "
            f"got {len(datasets)}: {list(datasets)}"
        )

    ds = next(iter(datasets.values()))
    if "member_id" in ds.dims:
        ds = ds.isel(member_id=0, drop=True)
    if "dcpp_init_year" in ds.dims:
        ds = ds.isel(dcpp_init_year=0, drop=True)

    log.info("%s %s: %s calendar, %d steps",
             source_id, experiment, calendar_of(ds), ds.sizes.get("time", 0))
    return ds

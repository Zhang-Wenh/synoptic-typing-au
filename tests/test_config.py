"""Tests for config loading and path resolution."""

from pathlib import Path

import pytest

from src.config import load_domain, load_paths, load_sources, load_yaml


def test_paths_are_absolute():
    paths = load_paths()
    for p in (paths.raw, paths.work, paths.out, paths.tmp):
        assert p.is_absolute()


def test_paths_expand_home():
    assert "~" not in str(load_paths().raw)


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_yaml("does_not_exist")


def test_domain_period_starts_in_the_satellite_era():
    """1979 is a deliberate choice, not a default. Guard it."""
    assert load_domain()["period"]["start"].startswith("1979")


def test_cmip6_evaluation_window_ends_with_historical():
    assert load_domain()["period"]["eval_end"].startswith("2014")


def test_typing_domain_encloses_the_target_region():
    d = load_domain()
    typing, target = d["typing"], d["target"]
    assert typing["lat_north"] > target["lat_north"]
    assert typing["lat_south"] < target["lat_south"]
    assert typing["lon_west"] < target["lon_west"]
    assert typing["lon_east"] > target["lon_east"]


def test_longitudes_use_zero_to_360_convention():
    """ERA5 longitude runs 0-360. A negative bound would slice nothing."""
    d = load_domain()
    for box in (d["typing"], d["target"]):
        assert 0 <= box["lon_west"] <= 360
        assert 0 <= box["lon_east"] <= 360


def test_sampling_hours_are_valid():
    hours = load_domain()["sampling"]["hours"]
    assert all(0 <= h < 24 for h in hours)
    assert len(set(hours)) == len(hours)


def test_sources_declare_anonymous_access():
    assert load_sources()["era5"]["storage_options"]["token"] == "anon"

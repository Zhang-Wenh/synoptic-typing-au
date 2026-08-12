"""Tests for the latitude-band definitions and their seasons.

Small surface, but two things here would corrupt the comparison silently: a
band whose season is wrong compares a real signal against near-zero, and
overlapping bands double-count the land between them.
"""

import pytest

from src.attribute.regional import BANDS


def test_every_band_has_a_complete_box():
    for name, band in BANDS.items():
        for key in ("lat_north", "lat_south", "lon_west", "lon_east"):
            assert key in band, f"{name} is missing {key}"


def test_latitudes_run_north_to_south():
    for name, band in BANDS.items():
        assert band["lat_north"] > band["lat_south"], name


def test_longitudes_run_west_to_east():
    for name, band in BANDS.items():
        assert band["lon_west"] < band["lon_east"], name


def test_the_three_zonal_bands_do_not_overlap():
    """Overlapping bands would count the same land twice.

    The southeast band is excluded: it sits inside the midlatitude band on
    purpose, as a link back to the earlier analysis.
    """
    zonal = ["tropics", "subtropics", "midlatitudes"]
    edges = [(BANDS[n]["lat_north"], BANDS[n]["lat_south"]) for n in zonal]
    edges.sort(reverse=True)
    for (_, south), (north, _) in zip(edges, edges[1:]):
        assert south == north, "bands must meet exactly, not overlap or leave gaps"


def test_bands_span_the_continent_in_longitude():
    for name in ("tropics", "subtropics", "midlatitudes"):
        assert BANDS[name]["lon_west"] <= 113.0
        assert BANDS[name]["lon_east"] >= 153.0


def test_each_band_is_analysed_in_its_own_wet_season():
    """Comparing tropical rainfall in the cool season would compare a real
    signal against near-zero."""
    assert BANDS["tropics"]["season"] == "warm"
    assert BANDS["midlatitudes"]["season"] == "cool"
    assert BANDS["southeast"]["season"] == "cool"


def test_the_arid_band_has_no_single_wet_season():
    assert BANDS["subtropics"]["season"] is None


def test_the_southeast_band_sits_inside_the_midlatitude_band():
    inner, outer = BANDS["southeast"], BANDS["midlatitudes"]
    assert inner["lat_north"] <= outer["lat_north"]
    assert inner["lat_south"] >= outer["lat_south"]
    assert inner["lon_west"] >= outer["lon_west"]
    assert inner["lon_east"] <= outer["lon_east"]


def test_every_band_explains_itself():
    for name, band in BANDS.items():
        assert band.get("note"), f"{name} has no note saying what it is"

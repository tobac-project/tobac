"""Tests for latitude/longitude tracking"""

from __future__ import annotations

import pytest
import numpy as np
from pandas.testing import assert_frame_equal

import tobac.testing
import tobac.tracking


def meters_to_latlon(base_lat, base_lon, dx_meters, dy_meters):
    """
    Convert offsets in meters (e.g., hdim_1, hdim_2) to latitude/longitude.

    Parameters:
    -----------
    base_lat : float
        Base latitude in degrees
    base_lon : float
        Base longitude in degrees
    dx_meters : float or array-like
        Offset in meters in the x (east-west) direction
    dy_meters : float or array-like
        Offset in meters in the y (north-south) direction

    Returns:
    --------
    tuple : (latitude, longitude) in degrees
    """
    # Earth's radius in meters (WGS84)
    planet_radius = 6378137.0

    base_lat_rad = np.radians(base_lat)
    new_lat = base_lat + (dy_meters / planet_radius) * (180 / np.pi)

    new_lon = base_lon + (dx_meters / planet_radius) * (180 / np.pi) / np.cos(
        base_lat_rad
    )

    return new_lat, new_lon


@pytest.mark.parametrize(
    "base_latitude, base_longitude, dxy, dt, d_max, spd_h1, spd_h2",
    [
        [40, 40, 1000, 60, 2000, 1, 1],
        [-40, 40, 1000, 60, 2000, 1, 1],
        [-40, 320, 1000, 60, 10000, 1, 1],
        [-40, -150, 1000, 60, 10000, 1, 1],
        [40, -90, 2000, 60, 2000, 1, 1],
        [40, -90, 2000, 60, 2000, 5, 5],
        [40, -90, 2000, 60, 2000, 10, 10],
        [40, -90, 2000, 60, 2000, 20, 20],
    ],
)
def test_lat_lon_tracking_vs_base_simple(
    base_latitude: float,
    base_longitude: float,
    dxy: float,
    dt: float,
    d_max: float,
    spd_h1: float,
    spd_h2: float,
):
    """Tests the lat/lon tracking functions"""

    start_h1 = 0
    start_h2 = 0

    test_feature = tobac.testing.generate_single_feature(
        start_h1=start_h1,
        start_h2=start_h2,
        min_h1=0,
        max_h1=100,
        min_h2=0,
        max_h2=100,
        frame_start=0,
        num_frames=5,
        spd_h1=1,
        spd_h2=1,
        PBC_flag="none",
    )

    # calculate latitude/longitude from test_feature

    lats, lons = meters_to_latlon(
        base_latitude,
        base_longitude,
        test_feature["hdim_2"].values * dxy,
        test_feature["hdim_1"].values * dxy,
    )
    test_feature["latitude"] = lats
    test_feature["longitude"] = lons

    # standard tracking

    standard_linking = tobac.tracking.linking_trackpy(
        test_feature, None, dt, dxy, method_linking="predict", d_max=d_max
    )

    lat_lon_linking = tobac.tracking.linking_trackpy_latlon(
        test_feature, dt, method_linking="predict", d_max=d_max
    )
    assert_frame_equal(standard_linking, lat_lon_linking)


def test_lat_lon_tracking_3d():
    """Tests that 3D input raises an error"""

    test_feature = tobac.testing.generate_single_feature(
        1,
        1,
        min_h1=0,
        max_h1=100,
        min_h2=0,
        max_h2=100,
        start_v=0,
        frame_start=0,
        num_frames=5,
        spd_h1=1,
        spd_h2=1,
        PBC_flag="none",
    )
    assert "vdim" in test_feature
    with pytest.raises(ValueError):
        test_feature = tobac.tracking.linking_trackpy_latlon(test_feature, dt=30)

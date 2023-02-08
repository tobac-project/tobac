import tobac.utils.internal as internal_utils
import numpy as np
import xarray as xr
import pytest


@pytest.mark.parametrize(
    "lat_name, lon_name, lat_name_test, lon_name_test, expected_result",
    [
        ("lat", "lon", "auto", "auto", ("lat", "lon")),
        ("lat", "long", "auto", "auto", ("lat", "long")),
        ("lat", "altitude", "auto", "auto", ("lat", None)),
        ("lat", "longitude", "lat", "longitude", ("lat", "longitude")),
    ],
)
def test_detect_latlon_coord_name(
    lat_name, lon_name, lat_name_test, lon_name_test, expected_result
):
    """Tests tobac.utils.internal.detect_latlon_coord_name"""

    in_arr = np.empty((50, 50))
    lat_vals = np.empty(50)
    lon_vals = np.empty(50)

    in_xr = xr.Dataset(
        {"data": ((lat_name, lon_name), in_arr)},
        coords={lat_name: lat_vals, lon_name: lon_vals},
    )
    out_lat_name, out_lon_name = internal_utils.detect_latlon_coord_name(
        in_xr["data"].to_iris(), lat_name_test, lon_name_test
    )
    assert out_lat_name == expected_result[0]
    assert out_lon_name == expected_result[1]

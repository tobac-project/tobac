import tobac.utils.internal as internal_utils
import tobac.testing as tbtest

import pytest
import numpy as np
import xarray as xr


@pytest.mark.parametrize(
    "dset_type, time_axis, vertical_axis, expected_out",
    [
        ("iris", 0, 1, (2, 3)),
        ("iris", -1, 0, (1, 2)),
        ("iris", 0, -1, (1, 2)),
        ("iris", 0, 2, (1, 3)),
        ("iris", 3, 0, (1, 2)),
        ("iris", 0, 3, (1, 2)),
        ("iris", 1, 2, (0, 3)),
    ],
)
def test_find_hdim_axes_3D(dset_type, time_axis, vertical_axis, expected_out):
    """Tests tobac.utils.internal.file_hdim_axes_3D

    Parameters
    ----------
    dset_type: str{"xarray" or "iris"}
        type of the dataset to generate
    time_axis: int
        axis number of the time coordinate (or -1 to not have one)
    vertical_axis: int
        axis number of the vertical coordinate (or -1 to not have one)
    expected_out: tuple (int, int)
        expected output
    """
    ndims = 2 + (1 if time_axis >= 0 else 0) + (1 if vertical_axis >= 0 else 0)
    test_dset_size = [2] * ndims

    test_data = np.zeros(test_dset_size)

    dset_opts = {
        "in_arr": test_data,
        "data_type": dset_type,
    }
    if time_axis >= 0:
        dset_opts["time_dim_num"] = time_axis
    if vertical_axis >= 0:
        dset_opts["z_dim_num"] = vertical_axis
        dset_opts["z_dim_name"] = "altitude"

    y_set = False
    for dim_number in range(ndims):
        if time_axis != dim_number and vertical_axis != dim_number:
            if not y_set:
                dset_opts["y_dim_num"] = dim_number
                y_set = True
            else:
                dset_opts["x_dim_num"] = dim_number

    cube_test = tbtest.make_dataset_from_arr(**dset_opts)

    out_coords = internal_utils.find_hdim_axes_3D(cube_test)

    assert out_coords == expected_out


@pytest.mark.parametrize(
    "lat_name, lon_name, lat_name_test, lon_name_test, expected_result",
    [
        ("lat", "lon", None, None, ("lat", "lon")),
        ("lat", "long", None, None, ("lat", "long")),
        ("lat", "altitude", None, None, ("lat", None)),
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

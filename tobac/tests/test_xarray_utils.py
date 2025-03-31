"""Tests for tobac.utils.internal_utils.xarray_utils"""

from __future__ import annotations

from typing import Union

import pytest
import numpy as np
import xarray as xr

import tobac.utils.internal.xarray_utils as xr_utils
import tobac.testing as tbtest
import datetime


@pytest.mark.parametrize(
    "dim_names, coord_dim_map, coord_looking_for, expected_out, expected_raise",
    [
        (
            ("time", "altitude", "x", "y"),  # dim_names
            {  # coord_dim_map
                "time": ("time",),
                "latitude": ("x", "y"),
                "longitude": ("x", "y"),
                "altmsl": ("altitude", "x", "y"),
            },
            "time",  # coord_looking_for
            0,
            False,
        ),
        (
            ("time", "time", "time", "time", "time"),  # dim_names
            {  # coord_dim_map
                "time": ("time",),
            },
            "time",  # coord_looking_for
            0,
            True,
        ),
        (
            ("time", "altitude", "x", "y"),  # dim_names
            {  # coord_dim_map
                "time": ("time",),
                "latitude": ("x", "y"),
                "longitude": ("x", "y"),
                "altmsl": ("altitude", "x", "y"),
            },
            "altitude",  # coord_looking_for
            1,
            False,
        ),
        (
            ("time", "altitude", "x", "y"),  # dim_names
            {  # coord_dim_map
                "time": ("time",),
                "latitude": ("x", "y"),
                "longitude": ("x", "y"),
                "altmsl": ("altitude", "x", "y"),
            },
            "latitude",  # coord_looking_for
            None,
            True,
        ),
        (
            ("time", "altitude", "x", "y"),  # dim_names
            {  # coord_dim_map
                "time": ("time",),
                "latitude": ("x", "y"),
                "longitude": ("x", "y"),
                "altmsl": ("altitude", "x", "y"),
            },
            "x",  # coord_looking_for
            2,
            False,
        ),
        (
            ("time", "altitude", "x", "y"),  # dim_names
            {  # coord_dim_map
                "time": ("time",),
                "latitude": ("x", "y"),
                "longitude": ("x", "y"),
                "altmsl": ("altitude", "x", "y"),
            },
            "z",  # coord_looking_for
            2,
            True,
        ),
        (
            ("time", "altitude", "x", "y"),  # dim_names
            {  # coord_dim_map
                "t": ("time",),
                "latitude": ("x", "y"),
                "longitude": ("x", "y"),
                "altmsl": ("altitude", "x", "y"),
            },
            "t",  # coord_looking_for
            0,
            False,
        ),
    ],
)
def test_find_axis_from_dim_coord(
    dim_names: tuple[str],
    coord_dim_map: dict,
    coord_looking_for: str,
    expected_out: Union[int, None],
    expected_raise: bool,
):
    """Tests tobac.utils.internal.file_hdim_axes_3D

    Parameters
    ----------
    dim_names: tuple[str]
        Names of the dimensions to have
    coord_dim_map: dict[str : tuple[str],]
        Mapping of coordinates (keys) to dimensions (values)
    coord_looking_for: str
        what coordinate/dimension to look for
    expected_out: Union[int, None]
        What the expected output is
    expected_raise: bool
        Whether or not we expect a raise
    """

    # size of the array per dimension
    arr_sz = 4
    arr_da = np.empty((arr_sz,) * len(dim_names))
    coord_vals = {}
    for coord_nm in coord_dim_map:
        coord_vals[coord_nm] = (
            coord_dim_map[coord_nm],
            np.empty((arr_sz,) * len(coord_dim_map[coord_nm])),
        )

    xr_da = xr.DataArray(arr_da, dims=dim_names, coords=coord_vals)
    if expected_raise:
        with pytest.raises(ValueError):
            _ = xr_utils.find_axis_from_dim_coord(xr_da, coord_looking_for)
    else:
        out_val = xr_utils.find_axis_from_dim_coord(xr_da, coord_looking_for)
        if expected_out is not None:
            assert out_val == expected_out
        else:
            assert out_val is None


@pytest.mark.parametrize(
    "dim_names, coord_dim_map, feature_pos, expected_vals",
    [
        (
            ["time", "x", "y"],
            {
                "test_coord1": (tuple(), 1),
                "test_coord_time": ("time", [5, 6, 7, 8, 9, 10]),
            },
            (1, 1),
            {"test_coord1": (1, 1, 1), "test_coord_time": (5, 6, 7)},
        ),
    ],
)
def test_add_coordinates_to_features_interpolate_along_other_dims(
    dim_names: tuple[str],
    coord_dim_map: dict,
    feature_pos: tuple[int],
    expected_vals: dict[str, tuple],
):
    time_len: int = 6
    if len(feature_pos) == 2:
        all_feats = tbtest.generate_single_feature(
            feature_pos[0],
            feature_pos[1],
            feature_num=1,
            num_frames=3,
            max_h1=100,
            max_h2=100,
        )
        arr_size = (time_len, 5, 5)

    elif len(feature_pos) == 3:
        all_feats = tbtest.generate_single_feature(
            feature_pos[1],
            feature_pos[2],
            start_v=feature_pos[0],
            feature_num=1,
            num_frames=3,
            max_h1=100,
            max_h2=100,
        )
        arr_size = (time_len, 1, 5, 5)
    else:
        raise ValueError("too many dimensions")
    coord_dim_map["time"] = (
        ("time",),
        [
            datetime.datetime(2000, 1, 1, 0) + datetime.timedelta(hours=x)
            for x in range(time_len)
        ],
    )

    test_xr_arr = xr.DataArray(np.empty(arr_size), dims=dim_names, coords=coord_dim_map)

    resulting_df = xr_utils.add_coordinates_to_features(all_feats, test_xr_arr)
    for coord in coord_dim_map:
        assert coord in resulting_df
        if coord != "time":
            assert np.all(resulting_df[coord].values == expected_vals[coord])

"""Tests for tobac.utils.internal_utils.xarray_utils
"""

from typing import Union

import pytest
import numpy as np
import xarray as xr

import tobac.utils.internal.xarray_utils as xr_utils


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
    coord_dim_map: dict[str : tuple[str],],
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

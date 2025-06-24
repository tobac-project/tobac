"""Tests to confirm that xarray and iris pathways work the same and produce the same data
for the same input datasets.
"""

from __future__ import annotations

import copy
import datetime

import iris.cube
import numpy as np
import pandas as pd
import xarray as xr
import pytest


import tobac.testing as tbtest
import tobac.utils.internal.iris_utils as iris_utils
import tobac.utils.internal.xarray_utils as xr_utils
import tobac.utils.datetime as datetime_utils
from tobac.utils.decorators import convert_cube_to_dataarray


@pytest.mark.parametrize(
    "feature_positions, coordinates, expected_val",
    [
        (
            ((0, 0, 0), (9, 9, 9)),
            {"x": ("x", np.linspace(0, 10, 10)), "z": ("z", np.linspace(0, 10, 10))},
            {"x": (0, 10)},
        ),
        (
            ((0, 0), (9, 9)),
            {"x": ("x", np.linspace(0, 10, 10))},
            {"x": (0, 10)},
        ),
        (
            ((0, 0), (9, 9), (5, 7)),
            {
                "longitude": ("x", np.linspace(-30, 60, 10)),
                "latitude": ("y", np.linspace(-70, 20, 10)),
            },
            {"latitude": (-70, 20, 0), "longitude": (-30, 60, 20)},
        ),
        (
            ((0, 0), (9, 9), (5, 7), (3.6, 7.9)),
            {
                "longitude": (
                    ("x", "y"),
                    np.arange(-180, -80).reshape(10, -1),
                ),
                "latitude": (("x", "y"), np.arange(-50, 50).reshape(10, -1)),
            },
            {
                "latitude": (-50, 49, 7, -6.1),
                "longitude": (-180, -81, -123, -136.1),
            },
        ),
    ],
)
def test_add_coordinates_xarray_base(
    feature_positions: tuple[tuple[float]],
    coordinates: dict[str : tuple[str, np.ndarray]],
    expected_val: dict[str : tuple[float]],
):
    """
    Test that adding coordinates for xarray and iris are equal, using an
    xarray generated dataset as the base.

    Parameters
    ----------
    feature_positions: tuple of tuple of floats
        Locations of the features to test in (hdim_1, hdim_2, zdim [optional]) coordinates
    coordinates: dict, key: str; value: tuple of str, numpy array
        Coordinates to use, in xarray coordinate style. Dims will be ('x', 'y', 'z') for 3D
        data (determined by feature_positions) and ('x', 'y') for 2D data. All axes will have
        size 10.
    expected_val: dict, key: str; value: tuple of floats
        Expected interpolated coordinates

    """

    all_indiv_feats = []
    if len(feature_positions[0]) == 2:
        is_3D = False
    elif len(feature_positions[0]) == 3:
        is_3D = True
    else:
        raise ValueError("Feature positions should be 2 or 3D")
    for i, single_feat_position in enumerate(feature_positions):
        if not is_3D and len(single_feat_position) == 2:
            all_indiv_feats.append(
                tbtest.generate_single_feature(
                    single_feat_position[0],
                    single_feat_position[1],
                    feature_num=i,
                    max_h1=10,
                    max_h2=10,
                )
            )
        elif is_3D and len(single_feat_position) == 3:
            all_indiv_feats.append(
                tbtest.generate_single_feature(
                    single_feat_position[0],
                    single_feat_position[1],
                    start_v=single_feat_position[2],
                    feature_num=i,
                    max_h1=10,
                    max_h2=10,
                )
            )

        else:
            raise ValueError("Feature positions should be 2 or 3D")

    all_feats = pd.concat(all_indiv_feats)

    da_size = (1, 10, 10, 10) if is_3D else (1, 10, 10)
    dims = ("time", "x", "y", "z") if is_3D else ("time", "x", "y")
    coordinates["time"] = np.array((datetime.datetime(2000, 1, 1, 0),))
    da_with_coords = xr.DataArray(data=np.empty(da_size), dims=dims, coords=coordinates)
    if is_3D:
        iris_coord_interp = iris_utils.add_coordinates_3D(
            all_feats, da_with_coords.to_iris()
        )
        xr_coord_interp = xr_utils.add_coordinates_to_features(
            all_feats, da_with_coords
        )

    else:
        iris_coord_interp = iris_utils.add_coordinates(
            all_feats, da_with_coords.to_iris()
        )
        xr_coord_interp = xr_utils.add_coordinates_to_features(
            all_feats, da_with_coords
        )
    for val_name in expected_val:
        np.testing.assert_almost_equal(
            iris_coord_interp[val_name], expected_val[val_name]
        )
        np.testing.assert_almost_equal(
            xr_coord_interp[val_name], expected_val[val_name]
        )

        # assert (iris_coord_interp[val_name] == expected_val[val_name]).all()
        # assert (xr_coord_interp[val_name] == expected_val[val_name]).all()

    # Convert datetimes to ensure that they are the same type:
    xr_coord_interp["time"] = datetime_utils.match_datetime_format(
        xr_coord_interp.time, iris_coord_interp.time
    )

    pd.testing.assert_frame_equal(iris_coord_interp, xr_coord_interp)


@pytest.mark.parametrize(
    "coordinate_names, coordinate_standard_names",
    [(("lat",), ("latitude",))],
)
def test_add_coordinates_xarray_std_names(
    coordinate_names: tuple[str],
    coordinate_standard_names: tuple[str],
):
    """
    Test that adding coordinates for xarray and iris result in the same coordinate names
    when standard_names are added to the xarray coordinates

    Parameters
    ----------
    coordinate_names: tuple of str
        names of coordinates to give
    coordinate_standard_name: tuple of str
        standard_names of coordinates to give

    """

    all_feats = tbtest.generate_single_feature(
        0,
        0,
        feature_num=1,
        max_h1=10,
        max_h2=10,
    )

    da_size = (1, 10, 10)
    dims = ("time", "x", "y")
    coordinates = dict()
    coordinates["time"] = np.array((datetime.datetime(2000, 1, 1, 0),))

    for coord_name, coord_standard_name in zip(
        coordinate_names, coordinate_standard_names
    ):
        coordinates[coord_name] = xr.DataArray(data=np.arange(10), dims="x")
        coordinates[coord_name].attrs["standard_name"] = coord_standard_name

    da_with_coords = xr.DataArray(data=np.empty(da_size), dims=dims, coords=coordinates)

    iris_coord_interp = iris_utils.add_coordinates(
        copy.deepcopy(all_feats), da_with_coords.to_iris()
    )
    xr_coord_interp = xr_utils.add_coordinates_to_features(
        copy.deepcopy(all_feats), da_with_coords
    )
    xr_coord_interp["time"] = datetime_utils.match_datetime_format(
        xr_coord_interp.time, iris_coord_interp.time
    )
    pd.testing.assert_frame_equal(iris_coord_interp, xr_coord_interp)


def test_preserve_iris_datetime_types():
    """
    Test that xarray.add_coordinates_to_features correctly returns the same time types as
    iris when preserve_iris_datetime_types = True.
    """

    all_feats = tbtest.generate_single_feature(
        0,
        0,
        feature_num=1,
        max_h1=10,
        max_h2=10,
    )
    var_array: iris.cube.Cube = tbtest.make_simple_sample_data_2D(data_type="iris")

    xarray_output = xr_utils.add_coordinates_to_features(
        all_feats,
        convert_cube_to_dataarray(var_array, preserve_iris_datetime_types=True),
    )
    iris_output = iris_utils.add_coordinates(all_feats, var_array)

    pd.testing.assert_frame_equal(xarray_output, iris_output)
    assert xarray_output["time"].values[0] == iris_output["time"].values[0]
    assert isinstance(
        xarray_output["time"].values[0], type(iris_output["time"].values[0])
    )

    xarray_output_datetime_preserve_off = xr_utils.add_coordinates_to_features(
        all_feats,
        convert_cube_to_dataarray(var_array, preserve_iris_datetime_types=False),
    )

    assert isinstance(
        xarray_output_datetime_preserve_off["time"].values[0], np.datetime64
    )

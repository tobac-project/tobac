"""Tests to confirm that xarray and iris pathways work the same and produce the same data
 for the same input datasets.
"""
import datetime

import numpy as np
import pandas as pd
import xarray as xr
import pytest


import tobac.testing as tbtest
import tobac.utils.internal.iris_utils as iris_utils
import tobac.utils.internal.xarray_utils as xr_utils


@pytest.mark.parametrize(
    "feature_positions, coordinates, expected_val",
    [(((0, 0, 0), (9, 9, 9)), {"x": ("x", np.linspace(0, 9, 10))}, {"x": (0, 9)})],
)
def test_add_coordinates_2D(
    feature_positions: tuple[tuple[float]],
    coordinates: dict[str : tuple[str, np.ndarray]],
    expected_val: dict[str : tuple[float]],
):
    """
    Test that add_coordinates_2D for xarray and iris are equal.

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

    all_indiv_feats = list()
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
        da_with_coords = xr.DataArray(
            data=np.empty(da_size), dims=dims, coords=coordinates
        )
        if is_3D:
            iris_coord_interp = iris_utils.add_coordinates_3D(
                all_feats, da_with_coords.to_iris()
            )
        else:
            iris_coord_interp = iris_utils.add_coordinates(
                all_feats, da_with_coords.to_iris()
            )

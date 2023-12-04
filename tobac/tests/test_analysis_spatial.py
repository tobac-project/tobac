"""
Test spatial analysis functions
"""

from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr

from tobac.analysis.spatial import calculate_area


def test_calculate_area():
    """
    Test the calculate_area function for 2D and 3D masks
    """

    test_labels = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 2, 0],
                [0, 1, 0, 2, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ],
        dtype=int,
    )

    test_labels = xr.DataArray(
        test_labels,
        dims=("time", "projection_y_coordinate", "projection_x_coordinate"),
        coords={
            "time": [datetime(2000, 1, 1)],
            "projection_y_coordinate": np.arange(5),
            "projection_x_coordinate": np.arange(5),
        },
    )

    # We need to do this to avoid round trip bug with xarray to iris conversion
    test_cube = test_labels.to_iris()
    test_cube = test_cube.copy(test_cube.core_data().filled())

    test_features = pd.DataFrame(
        {
            "feature": [1, 2],
            "frame": [0, 0],
            "time": [
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
            ],
        }
    )

    expected_areas = np.array([3, 2])

    area = calculate_area(test_features, test_cube)

    assert np.all(area["area"] == expected_areas)

    test_labels = np.array(
        [
            [
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 2, 0],
                    [0, 1, 0, 2, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 1, 0, 3, 0],
                    [0, 1, 0, 3, 0],
                    [0, 0, 0, 0, 0],
                ],
            ],
        ],
        dtype=int,
    )

    test_labels = xr.DataArray(
        test_labels,
        dims=(
            "time",
            "model_level_number",
            "projection_y_coordinate",
            "projection_x_coordinate",
        ),
        coords={
            "time": [datetime(2000, 1, 1)],
            "model_level_number": np.arange(2),
            "projection_y_coordinate": np.arange(5),
            "projection_x_coordinate": np.arange(5),
        },
    )

    # We need to do this to avoid round trip bug with xarray to iris conversion
    test_cube = test_labels.to_iris()
    test_cube = test_cube.copy(test_cube.core_data().filled())

    test_features = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "frame": [0, 0, 0],
            "time": [
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
            ],
        }
    )

    expected_areas = np.array([3, 2, 2])

    area = calculate_area(test_features, test_cube)

    assert np.all(area["area"] == expected_areas)

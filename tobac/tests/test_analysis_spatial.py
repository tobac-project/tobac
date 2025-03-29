"""
Test spatial analysis functions
"""

from datetime import datetime
import pytest
import numpy as np
import pandas as pd
import xarray as xr
from iris.analysis.cartography import area_weights

from tobac.analysis.spatial import (
    calculate_distance,
    calculate_velocity_individual,
    calculate_velocity,
    calculate_nearestneighbordistance,
    calculate_area,
    calculate_areas_2Dlatlon,
)
from tobac.utils.datetime import to_cftime, to_datetime64


def test_calculate_distance():
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

    with pytest.raises(ValueError):
        calculate_distance(test_features.iloc[0], test_features.iloc[1])

    test_features = pd.DataFrame(
        {
            "feature": [1, 2],
            "frame": [0, 0],
            "time": [
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
            ],
            "projection_x_coordinate": [0, 1000],
            "projection_y_coordinate": [0, 0],
        }
    )

    assert calculate_distance(test_features.iloc[0], test_features.iloc[1]) == 1000

    test_features = pd.DataFrame(
        {
            "feature": [1, 2],
            "frame": [0, 0],
            "time": [
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
            ],
            "longitude": [0, 1],
            "latitude": [0, 0],
        }
    )

    assert calculate_distance(
        test_features.iloc[0], test_features.iloc[1]
    ) == pytest.approx(1.11e5, rel=1e4)

    with pytest.raises(ValueError):
        calculate_distance(
            test_features.iloc[0],
            test_features.iloc[1],
            method_distance="invalid_method",
        )


def test_calculate_velocity_individual_xy():
    test_features = pd.DataFrame(
        {
            "feature": [1, 2],
            "frame": [0, 1],
            "time": [
                datetime(2000, 1, 1, 0, 0),
                datetime(2000, 1, 1, 0, 10),
            ],
            "projection_x_coordinate": [0, 6000],
            "projection_y_coordinate": [0, 0],
        }
    )

    assert (
        calculate_velocity_individual(test_features.iloc[0], test_features.iloc[1])
        == 10
    )

    test_features = pd.DataFrame(
        {
            "feature": [1, 2],
            "frame": [0, 1],
            "time": [
                datetime(2000, 1, 1, 0, 0),
                datetime(2000, 1, 1, 0, 10),
            ],
            "x": [0, 6000],
            "y": [0, 0],
        }
    )

    assert (
        calculate_velocity_individual(test_features.iloc[0], test_features.iloc[1])
        == 10
    )


@pytest.mark.parametrize(
    "time_format", ("datetime", "datetime64", "proleptic_gregorian", "360_day")
)
def test_calculate_velocity(time_format):
    test_features = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "frame": [0, 0, 1, 1],
            "time": [
                datetime(2000, 1, 1, 0, 0),
                datetime(2000, 1, 1, 0, 0),
                datetime(2000, 1, 1, 0, 10),
                datetime(2000, 1, 1, 0, 10),
            ],
            "x": [0, 0, 6000, 0],
            "y": [0, 0, 0, 9000],
            "cell": [1, 2, 1, 2],
        }
    )

    if time_format == "datetime":
        pass
    elif time_format == "datetime64":
        test_features["time"] = to_datetime64(test_features.time)
    else:
        test_features["time"] = to_cftime(test_features.time, calendar=time_format)

    assert calculate_velocity(test_features).at[0, "v"] == 10
    assert calculate_velocity(test_features).at[1, "v"] == 15


def test_calculate_nearestneighbordistance():
    test_features = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "frame": [0, 0, 1, 1],
            "time": [
                datetime(2000, 1, 1, 0, 0),
                datetime(2000, 1, 1, 0, 0),
                datetime(2000, 1, 1, 0, 10),
                datetime(2000, 1, 1, 0, 10),
            ],
            "projection_x_coordinate": [0, 1000, 0, 2000],
            "projection_y_coordinate": [0, 0, 0, 0],
            "cell": [1, 2, 1, 2],
        }
    )

    assert calculate_nearestneighbordistance(test_features)[
        "min_distance"
    ].to_list() == [1000, 1000, 2000, 2000]

    test_features = pd.DataFrame(
        {
            "feature": [1, 2],
            "frame": [0, 1],
            "time": [
                datetime(2000, 1, 1, 0, 0),
                datetime(2000, 1, 1, 0, 10),
            ],
            "projection_x_coordinate": [0, 6000],
            "projection_y_coordinate": [0, 0],
            "cell": [1, 1],
        }
    )

    assert np.all(
        np.isnan(calculate_nearestneighbordistance(test_features)["min_distance"])
    )


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

    test_labels = xr.DataArray(
        test_labels,
        dims=(
            "time",
            "model_level_number",
            "hdim_0",
            "hdim_1",
        ),
        coords={
            "time": [datetime(2000, 1, 1)],
            "model_level_number": np.arange(2),
        },
    )

    # Test failure to find valid coordinates
    with pytest.raises(ValueError):
        calculate_area(test_features, test_labels)

    # Test failure for invalid method
    with pytest.raises(ValueError):
        calculate_area(test_features, test_labels, method_area="invalid_method")


def test_calculate_area_latlon():
    # Test with latitude/longitude
    test_labels = np.array(
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
                [0, 4, 0, 0, 0],
                [0, 4, 0, 3, 0],
                [0, 4, 0, 3, 0],
                [0, 0, 0, 0, 0],
            ],
        ],
        dtype=int,
    )

    test_labels = xr.DataArray(
        test_labels,
        dims=(
            "time",
            "latitude",
            "longitude",
        ),
        coords={
            "time": [datetime(2000, 1, 1), datetime(2000, 1, 1, 1)],
            "latitude": xr.DataArray(
                np.arange(5), dims="latitude", attrs={"units": "degrees"}
            ),
            "longitude": xr.DataArray(
                np.arange(5), dims="longitude", attrs={"units": "degrees"}
            ),
        },
    )

    test_features = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "frame": [0, 0, 1, 1],
            "time": [
                datetime(2000, 1, 1, 0),
                datetime(2000, 1, 1, 0),
                datetime(2000, 1, 1, 1),
                datetime(2000, 1, 1, 1),
            ],
        }
    )

    area = calculate_area(test_features, test_labels)

    expected_areas = np.array([3, 2, 2, 3]) * 1.11e5**2

    assert np.all(np.isclose(area["area"], expected_areas, atol=1e8))

    # Test invalid lat/lon dimensions

    # Test 1D lat but 2D lon
    test_labels = xr.DataArray(
        test_labels.values,
        dims=(
            "time",
            "y_dim",
            "x_dim",
        ),
        coords={
            "time": [datetime(2000, 1, 1), datetime(2000, 1, 1, 1)],
            "latitude": xr.DataArray(
                np.arange(5), dims="y_dim", attrs={"units": "degrees"}  # 1D lat
            ),
            "longitude": xr.DataArray(
                np.tile(np.arange(5), (5, 1)),
                dims=("y_dim", "x_dim"),  # 2D lon
                attrs={"units": "degrees"},
            ),
        },
    )

    with pytest.raises(ValueError):
        calculate_area(test_features, test_labels, method_area="latlon")

    # Test 3D lat/lon
    test_labels = xr.DataArray(
        np.tile(test_labels.values[:, np.newaxis, ...], (1, 2, 1, 1)),
        dims=(
            "time",
            "z_dim",
            "y_dim",
            "x_dim",
        ),
        coords={
            "time": [datetime(2000, 1, 1), datetime(2000, 1, 1, 1)],
            "latitude": xr.DataArray(
                np.tile(np.arange(5)[:, np.newaxis], (2, 1, 5)),
                dims=("z_dim", "y_dim", "x_dim"),
                attrs={"units": "degrees"},
            ),
            "longitude": xr.DataArray(
                np.tile(np.arange(5), (2, 5, 1)),
                dims=("z_dim", "y_dim", "x_dim"),
                attrs={"units": "degrees"},
            ),
        },
    )

    with pytest.raises(ValueError):
        calculate_area(test_features, test_labels, method_area="latlon")


def test_calculate_area_1D_latlon():
    """
    Test area calculation using 1D lat/lon coords
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
        dims=("time", "latitude", "longitude"),
        coords={
            "time": [datetime(2000, 1, 1)],
            "latitude": xr.DataArray(
                np.arange(5), dims=("latitude",), attrs={"units": "degrees"}
            ),
            "longitude": xr.DataArray(
                np.arange(5), dims=("longitude",), attrs={"units": "degrees"}
            ),
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

    # Calculate expected areas
    copy_of_test_cube = test_cube.copy()
    copy_of_test_cube.coord("latitude").guess_bounds()
    copy_of_test_cube.coord("longitude").guess_bounds()
    area_array = area_weights(copy_of_test_cube, normalize=False)

    expected_areas = np.array(
        [np.sum(area_array[test_labels.data == i]) for i in [1, 2]]
    )

    area = calculate_area(test_features, test_cube)

    assert np.all(area["area"] == expected_areas)


def test_calculate_areas_2Dlatlon():
    """
    Test calculation of area array from 2D lat/lon coords
    Note, in future this needs to be updated to account for non-orthogonal lat/lon arrays
    """

    test_labels = np.ones([1, 5, 5], dtype=int)

    test_labels = xr.DataArray(
        test_labels,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": [datetime(2000, 1, 1)],
            "latitude": xr.DataArray(
                np.arange(5), dims=("latitude",), attrs={"units": "degrees"}
            ),
            "longitude": xr.DataArray(
                np.arange(5), dims=("longitude",), attrs={"units": "degrees"}
            ),
        },
    )

    test_cube = test_labels.to_iris()
    test_cube = test_cube.copy(test_cube.core_data().filled())
    copy_of_test_cube = test_cube.copy()
    copy_of_test_cube.coord("latitude").guess_bounds()
    copy_of_test_cube.coord("longitude").guess_bounds()
    area_array = area_weights(copy_of_test_cube, normalize=False)

    lat_2d = xr.DataArray(
        np.stack([np.arange(5)] * 5, axis=1),
        dims=("y", "x"),
        attrs={"units": "degrees"},
    )

    lon_2d = xr.DataArray(
        np.stack([np.arange(5)] * 5, axis=0),
        dims=("y", "x"),
        attrs={"units": "degrees"},
    )

    test_labels = xr.DataArray(
        test_labels,
        dims=("time", "y", "x"),
        coords={
            "time": [datetime(2000, 1, 1)],
            "latitude": lat_2d,
            "longitude": lon_2d,
        },
    )

    test_cube = test_labels.to_iris()
    test_cube = test_cube.copy(test_cube.core_data().filled())

    assert np.allclose(
        calculate_areas_2Dlatlon(
            test_cube.coord("latitude"), test_cube.coord("longitude")
        ),
        area_array,
        rtol=0.01,
    )


def test_calculate_area_2D_latlon():
    """
    Test area calculation using 2D lat/lon coords
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

    lat_2d = xr.DataArray(
        np.stack([np.arange(5)] * 5, axis=1),
        dims=("y", "x"),
        attrs={"units": "degrees"},
    )

    lon_2d = xr.DataArray(
        np.stack([np.arange(5)] * 5, axis=0),
        dims=("y", "x"),
        attrs={"units": "degrees"},
    )

    test_labels = xr.DataArray(
        test_labels,
        dims=("time", "y", "x"),
        coords={
            "time": [datetime(2000, 1, 1)],
            "latitude": lat_2d,
            "longitude": lon_2d,
        },
    )

    test_cube = test_labels.to_iris()
    test_cube = test_cube.copy(test_cube.core_data().filled())

    area_array = calculate_areas_2Dlatlon(
        test_cube.coord("latitude"), test_cube.coord("longitude")
    )

    expected_areas = np.array(
        [np.sum(area_array[test_labels[0].data == i]) for i in [1, 2]]
    )

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

    area = calculate_area(test_features, test_cube)

    assert np.all(area["area"] == expected_areas)

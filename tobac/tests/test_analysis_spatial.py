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


def test_calculate_distance_xy():
    """
    Test for tobac.analysis.spatial.calculate_distance with cartesian coordinates
    """
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


def test_calculate_distance_latlon():
    """
    Test for tobac.analysis.spatial.calculate_distance with latitude/longitude
    coordinates
    """

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


def test_calculate_distance_latlon_wrong_order():
    """
    Test for tobac.analysis.spatial.calculate_distance with latitude/longitude
    coordinates provided in the wrong order. When lat/lon are provided with
    standard naming the function should detect this and switch their order to
    ensure that haversine distances are calculated correctly.
    """

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
    # Test that if latitude and longitude coord names are given in the wrong order, then they are swapped:
    # (expectation is hdim1=y=latitude, hdim2=x=longitude, doesn't matter for x/y but does matter for lat/lon)
    assert calculate_distance(
        test_features.iloc[0],
        test_features.iloc[1],
        hdim1_coord="longitude",
        hdim2_coord="latitude",
        method_distance="latlon",
    ) == pytest.approx(1.11e5, rel=1e4)


def test_calculate_distance_error_invalid_method():
    """Test invalid method_distance"""
    with pytest.raises(ValueError, match="method_distance invalid*"):
        calculate_distance(
            pd.DataFrame(), pd.DataFrame(), method_distance="invalid_method_distance"
        )


def test_calculate_distance_error_no_coords():
    """Test no horizontal coordinates in input dataframe"""
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


def test_calculate_distance_error_mismatched_coords():
    """Test dataframes with mismatching coordinates"""
    with pytest.raises(ValueError, match="Discovered coordinates*"):
        calculate_distance(
            pd.DataFrame(
                {
                    "feature": [1],
                    "frame": [0],
                    "time": [datetime(2000, 1, 1)],
                    "projection_x_coordinate": [0],
                    "projection_y_coordinate": [0],
                }
            ),
            pd.DataFrame(
                {
                    "feature": [1],
                    "frame": [0],
                    "time": [datetime(2000, 1, 1)],
                    "longitude": [0],
                    "latitude": [0],
                }
            ),
        )


def test_calculate_distance_error_no_method():
    """Test hdim1_coord/hdim2_coord specified but no method_distance"""
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

    with pytest.raises(ValueError, match="method_distance parameter must*"):
        calculate_distance(
            test_features.iloc[0],
            test_features.iloc[1],
            hdim1_coord="projection_y_coordinate",
        )

    with pytest.raises(ValueError, match="method_distance parameter must*"):
        calculate_distance(
            test_features.iloc[0],
            test_features.iloc[1],
            hdim2_coord="projection_x_coordinate",
        )


@pytest.mark.parametrize(
    "x_coord, y_coord",
    [("x", "y"), ("projection_x_coordinate", "projection_y_coordinate")],
)
def test_calculate_velocity_individual_xy(x_coord, y_coord):
    """
    Test calculate_velocity_individual gives the correct result for a single
    track woth different x/y coordinate names
    """
    test_features = pd.DataFrame(
        {
            "feature": [1, 2],
            "frame": [0, 1],
            "time": [
                datetime(2000, 1, 1, 0, 0),
                datetime(2000, 1, 1, 0, 10),
            ],
            x_coord: [0, 6000],
            y_coord: [0, 0],
        }
    )

    assert (
        calculate_velocity_individual(test_features.iloc[0], test_features.iloc[1])
        == 10
    )


@pytest.mark.parametrize(
    "lat_coord, lon_coord", [("lat", "lon"), ("latitude", "longitude")]
)
def test_calculate_velocity_individual_latlon(lat_coord, lon_coord):
    """
    Test calculate_velocity_individual gives the correct result for a single
    track woth different lat/lon coordinate names
    """
    test_features = pd.DataFrame(
        {
            "feature": [1, 2],
            "frame": [0, 0],
            "time": [
                datetime(2000, 1, 1, 0, 0),
                datetime(2000, 1, 1, 0, 10),
            ],
            lon_coord: [0, 1],
            lat_coord: [0, 0],
        }
    )

    assert calculate_velocity_individual(
        test_features.iloc[0], test_features.iloc[1]
    ) == pytest.approx(1.11e5 / 600, rel=1e2)


@pytest.mark.parametrize(
    "time_format", ("datetime", "datetime64", "proleptic_gregorian", "360_day")
)
def test_calculate_velocity(time_format):
    """
    Test velocity calculation using different time formats
    """
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


def test_calculate_distance_xy_3d():
    """
    3D distance for xy with use_3d flag and vertical coord
    """
    test_features = pd.DataFrame(
        {
            "feature": [1, 2],
            "frame": [0, 0],
            "time": [datetime(2000, 1, 1), datetime(2000, 1, 1)],
            "projection_x_coordinate": [0, 1000],
            "projection_y_coordinate": [0, 0],
            "height": [0, 600],
        }
    )
    d3d = calculate_distance(
        test_features.iloc[0],
        test_features.iloc[1],
        method_distance="xy",
        vertical_coord="height",
        use_3d=True,
    )
    assert d3d == pytest.approx(np.sqrt(1000**2 + 600**2), rel=1e-9)

    res = calculate_distance(
        test_features.iloc[0],
        test_features.iloc[1],
        method_distance="xy",
        vertical_coord="height",
        use_3d=True,
        return_components=True,
    )
    assert set(res.keys()) == {"distance_3d", "dx", "dy", "dz"}
    assert res["distance_3d"] == pytest.approx(np.sqrt(1000**2 + 600**2), rel=1e-9)


def test_calculate_distance_latlon_3d():
    """
    3D distance for lat/lon with use_3d flag and vertical coord
    """
    test_features = pd.DataFrame(
        {
            "feature": [1, 2],
            "frame": [0, 0],
            "time": [datetime(2000, 1, 1), datetime(2000, 1, 1)],
            "longitude": [0, 1],
            "latitude": [0, 0],
            "height": [0, 1000],
        }
    )
    d2d = calculate_distance(
        test_features.iloc[0], test_features.iloc[1], method_distance="latlon"
    )
    d3d = calculate_distance(
        test_features.iloc[0],
        test_features.iloc[1],
        method_distance="latlon",
        vertical_coord="height",
        use_3d=True,
    )
    assert d3d == pytest.approx(np.sqrt(d2d**2 + 1000**2), rel=1e-9)

    res = calculate_distance(
        test_features.iloc[0],
        test_features.iloc[1],
        method_distance="latlon",
        vertical_coord="height",
        use_3d=True,
        return_components=True,
    )
    assert "distance_3d" in res and "dx" in res and "dy" in res and "dz" in res


def test_calculate_velocity_individual_xy_3d():
    """
    3D velocity for xy with vertical coord and use_3d=True/False
    """
    test_features = pd.DataFrame(
        {
            "feature": [1, 2],
            "frame": [0, 1],
            "time": [datetime(2000, 1, 1, 0, 0), datetime(2000, 1, 1, 0, 10)],
            "projection_x_coordinate": [0, 6000],
            "projection_y_coordinate": [0, 300],
            "height": [0, 800],
        }
    )
    v3d = calculate_velocity_individual(
        test_features.iloc[0],
        test_features.iloc[1],
        method_distance="xy",
        vertical_coord="height",
        use_3d=True,
    )
    assert v3d == pytest.approx(np.sqrt(6000**2 + 300**2 + 800**2) / 600, rel=1e-9)

    res = calculate_velocity_individual(
        test_features.iloc[0],
        test_features.iloc[1],
        method_distance="xy",
        vertical_coord="height",
        use_3d=True,
        return_components=True,
    )
    assert set(res.keys()) >= {"v_3d", "vx", "vy", "vz"}
    assert res["v_3d"] == pytest.approx(
        np.sqrt(6000**2 + 300**2 + 800**2) / 600, rel=1e-9
    )

    res2d = calculate_velocity_individual(
        test_features.iloc[0],
        test_features.iloc[1],
        method_distance="xy",
        vertical_coord="height",
        use_3d=False,
        return_components=True,
    )
    assert set(res2d.keys()) >= {"v", "vx", "vy"}
    assert res2d["v"] == pytest.approx(np.sqrt(6000**2 + 300**2) / 600, rel=1e-9)


def test_calculate_velocity_individual_latlon_3d():
    """
    3D velocity for lat/lon with vertical coord and use_3d=True
    """
    test_features = pd.DataFrame(
        {
            "feature": [1, 2],
            "frame": [0, 0],
            "time": [datetime(2000, 1, 1, 0, 0), datetime(2000, 1, 1, 0, 10)],
            "longitude": [0, 1],
            "latitude": [0, 0],
            "height": [0, 1000],
        }
    )
    d2d = calculate_distance(
        test_features.iloc[0], test_features.iloc[1], method_distance="latlon"
    )
    v3d = calculate_velocity_individual(
        test_features.iloc[0],
        test_features.iloc[1],
        method_distance="latlon",
        vertical_coord="height",
        use_3d=True,
    )
    assert v3d == pytest.approx(np.sqrt(d2d**2 + 1000**2) / 600, rel=1e-9)

    res = calculate_velocity_individual(
        test_features.iloc[0],
        test_features.iloc[1],
        method_distance="latlon",
        vertical_coord="height",
        use_3d=True,
        return_components=True,
    )
    assert "v_3d" in res and "vx" in res and "vy" in res and "vz" in res


def test_calculate_velocity_3d_track():
    """
    Track with Z: use_3d=True -> 'v_3d' gets set
    """
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
            "projection_x_coordinate": [0, 0, 6000, 0],
            "projection_y_coordinate": [0, 0, 0, 9000],
            "height": [0, 0, 800, 1200],
            "cell": [1, 2, 1, 2],
        }
    )
    out = calculate_velocity(test_features, method_distance="xy", use_3d=True)
    assert out.at[0, "v_3d"] == pytest.approx(np.sqrt(6000**2 + 800**2) / 600, rel=1e-9)
    assert out.at[1, "v_3d"] == pytest.approx(
        np.sqrt(9000**2 + 1200**2) / 600, rel=1e-9
    )

    out_w_components = calculate_velocity(
        test_features,
        method_distance="xy",
        use_3d=True,
        return_components=True,
    )

    dt = 600.0
    # Expected values for cell 1
    dx1 = 6000.0
    dy1 = 0.0
    dz1 = 800.0
    v1_3d = np.sqrt(dx1**2 + dy1**2 + dz1**2) / dt
    vx1 = dx1 / dt
    vy1 = dy1 / dt
    vz1 = dz1 / dt

    # Expected values for cell 2
    dx2 = 0.0
    dy2 = 9000.0
    dz2 = 1200.0
    v2_3d = np.sqrt(dx2**2 + dy2**2 + dz2**2) / dt
    vx2 = dx2 / dt
    vy2 = dy2 / dt
    vz2 = dz2 / dt

    for col in ("v_3d", "vx", "vy", "vz"):
        assert col in out_w_components.columns

    # Cell 1:
    assert out_w_components.at[0, "v_3d"] == pytest.approx(v1_3d, rel=1e-12)
    assert out_w_components.at[0, "vx"] == pytest.approx(vx1, rel=1e-12)
    assert out_w_components.at[0, "vy"] == pytest.approx(vy1, rel=1e-12)
    assert out_w_components.at[0, "vz"] == pytest.approx(vz1, rel=1e-12)

    # Cell 2:
    assert out_w_components.at[1, "v_3d"] == pytest.approx(v2_3d, rel=1e-12)
    assert out_w_components.at[1, "vx"] == pytest.approx(vx2, rel=1e-12)
    assert out_w_components.at[1, "vy"] == pytest.approx(vy2, rel=1e-12)
    assert out_w_components.at[1, "vz"] == pytest.approx(vz2, rel=1e-12)

    for idx in (2, 3):
        assert np.isnan(out_w_components.at[idx, "v_3d"])
        assert np.isnan(out_w_components.at[idx, "vx"])
        assert np.isnan(out_w_components.at[idx, "vy"])
        assert np.isnan(out_w_components.at[idx, "vz"])


def test_latlon_3d_no_degree_components():
    """
    For lat/lon + use_3d + return_components: components dict must contain all components
    v_3d plus optional vx, vy, vz are expected.
    """
    test_features = pd.DataFrame(
        {
            "feature": [1, 2],
            "frame": [0, 0],
            "time": [datetime(2000, 1, 1), datetime(2000, 1, 1)],
            "longitude": [0.0, 1.0],
            "latitude": [0.0, 0.0],
            "height": [100.0, 500.0],
        }
    )

    res = calculate_velocity_individual(
        test_features.iloc[0],
        test_features.iloc[1],
        method_distance="latlon",
        vertical_coord="height",
        use_3d=True,
        return_components=True,
    )

    assert "v_3d" in res and "vx" in res and "vy" in res and "vz" in res


def test_latlon_3d_dt_zero_returns_nan_scalar():
    """
    Δt = 0 should return NaN (not crash) for scalar output.
    """
    t = datetime(2000, 1, 1, 0, 0, 0)
    test_features = pd.DataFrame(
        {
            "feature": [1, 2],
            "frame": [0, 0],
            "time": [t, t],
            "longitude": [0.0, 1.0],
            "latitude": [0.0, 0.0],
            "height": [0.0, 1000.0],
        }
    )

    v = calculate_velocity_individual(
        test_features.iloc[0],
        test_features.iloc[1],
        method_distance="latlon",
        vertical_coord="height",
        use_3d=True,
        return_components=False,
    )
    assert np.isnan(v)


def test_latlon_3d_dt_zero_returns_nan_components():
    """
    Δt = 0 should return NaN values in the components dict (not raise).
    """
    t = datetime(2000, 1, 1, 0, 0, 0)
    test_features = pd.DataFrame(
        {
            "feature": [1, 2],
            "frame": [0, 0],
            "time": [t, t],
            "longitude": [0.0, 1.0],
            "latitude": [0.0, 0.0],
            "height": [0.0, 1000.0],
        }
    )

    res = calculate_velocity_individual(
        test_features.iloc[0],
        test_features.iloc[1],
        method_distance="latlon",
        vertical_coord="height",
        use_3d=True,
        return_components=True,
    )
    # Keys present, values NaN
    assert "v_3d" in res and "vx" in res and "vy" in res and "vz" in res
    assert np.isnan(res["v_3d"])
    assert "v" not in res

    res_2d = calculate_velocity_individual(
        test_features.iloc[0],
        test_features.iloc[1],
        method_distance="latlon",
        vertical_coord="height",
        use_3d=False,
        return_components=True,
    )
    assert np.isnan(res_2d["v"])

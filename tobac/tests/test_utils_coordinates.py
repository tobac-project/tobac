"""Unit tests for tobac.utils.internal.general_internal.py"""

import pytest
import pandas as pd

import tobac.utils.internal.coordinates as coord_utils


def test_find_coord_in_dataframe_errors():
    """Test that find_coord_in_dataframe raises errors correctly"""
    defaults = ["x", "projection_x_coordinate", "__other_name"]

    # Test no options raises ValueError:
    with pytest.raises(ValueError, match="One of coord or defaults parameter*"):
        coord_utils.find_coord_in_dataframe(pd.DataFrame(columns=["time", "x"]))

    # Test coordinate specified not in dataframe raise ValueError:
    with pytest.raises(ValueError, match="Coordinate*"):
        coord_utils.find_coord_in_dataframe(
            pd.DataFrame(columns=["time", "x"]), coord="projection_x_coordinate"
        )

    # Test no coordinates matching defaults:
    with pytest.raises(ValueError, match="No coordinate found matching defaults*"):
        coord_utils.find_coord_in_dataframe(
            pd.DataFrame(columns=["time", "y"]), defaults=defaults
        )

    # Test multiple matches with defaults:
    with pytest.raises(ValueError, match="Multiple matching*"):
        coord_utils.find_coord_in_dataframe(
            pd.DataFrame(columns=["time", "x", "projection_x_coordinate"]),
            defaults=defaults,
        )

    # Test that giving an object that is not a dataframe or series returns an error
    with pytest.raises(ValueError, match="Input variable_dataframe is neither*"):
        coord_utils.find_coord_in_dataframe("test_str", defaults=defaults)


def test_find_coord_in_dataframe():
    """Test that find_coord_in_dataframe returns correct results for both
    default and specific coordinates
    """
    defaults = ["x", "projection_x_coordinate", "__other_name"]

    # Now test correct returns:
    assert (
        coord_utils.find_coord_in_dataframe(
            pd.DataFrame(columns=["time", "x", "projection_x_coordinate"]), coord="x"
        )
        == "x"
    )

    assert (
        coord_utils.find_coord_in_dataframe(
            pd.DataFrame(columns=["time", "x", "projection_x_coordinate"]),
            coord="projection_x_coordinate",
        )
        == "projection_x_coordinate"
    )

    assert (
        coord_utils.find_coord_in_dataframe(
            pd.DataFrame(columns=["time", "x", "y"]), defaults=defaults
        )
        == "x"
    )

    assert (
        coord_utils.find_coord_in_dataframe(
            pd.DataFrame(
                columns=["time", "projection_x_coordinate", "projection_y_coordinate"]
            ),
            defaults=defaults,
        )
        == "projection_x_coordinate"
    )

    assert (
        coord_utils.find_coord_in_dataframe(
            pd.DataFrame(columns=["time", "x", "projection_x_coordinate"]),
            coord="x",
            defaults=defaults,
        )
        == "x"
    )

    # Test pd.Series input:
    assert (
        coord_utils.find_coord_in_dataframe(
            pd.Series(index=["time", "x", "projection_x_coordinate"]), coord="x"
        )
        == "x"
    )


def test_find_dataframe_vertical_coord_warning():
    """Test the warning for coord="auto" in find_dataframe_vertical_coord"""
    with pytest.warns(DeprecationWarning):
        coord_utils.find_dataframe_vertical_coord(
            pd.DataFrame(columns=["z"]), vertical_coord="auto"
        )


def test_find_dataframe_vertical_coord_error():
    """Test find_dataframe_vertical_coord raises errors correctly"""
    # Test the error for invalid coord input:
    with pytest.raises(ValueError):
        coord_utils.find_dataframe_vertical_coord(
            pd.DataFrame(columns=["z"]), vertical_coord="__bad_coord_name"
        )

    # Test the error for no default coord found:
    with pytest.raises(ValueError):
        coord_utils.find_dataframe_vertical_coord(pd.DataFrame(columns=["x"]))

    # Test the error for multiple default coords found:
    with pytest.raises(ValueError):
        coord_utils.find_dataframe_vertical_coord(
            pd.DataFrame(columns=["z", "geopotential_height"])
        )


def test_find_dataframe_vertical_coord():
    """Test find_dataframe_vertical_coord provides correct results"""
    # Test default coords
    assert coord_utils.find_dataframe_vertical_coord(pd.DataFrame(columns=["z"])) == "z"
    assert (
        coord_utils.find_dataframe_vertical_coord(
            pd.DataFrame(columns=["geopotential_height"])
        )
        == "geopotential_height"
    )

    # Test coord input
    assert (
        coord_utils.find_dataframe_vertical_coord(
            pd.DataFrame(columns=["p"]), vertical_coord="p"
        )
        == "p"
    )

    # Test coord input when multiple default coords
    assert (
        coord_utils.find_dataframe_vertical_coord(
            pd.DataFrame(columns=["z", "geopotential_height"]), vertical_coord="z"
        )
        == "z"
    )


def test_find_dataframe_horizontal_coords_error():
    """Test find_dataframe_horizontal_coords raises errors correctly"""
    # Test no matching coords
    with pytest.raises(ValueError):
        coord_utils.find_dataframe_horizontal_coords(
            pd.DataFrame(columns=["time", "z"])
        )

    # Test hdim_1_coord or hdim_2_coord set but not coord_type
    with pytest.raises(ValueError):
        coord_utils.find_dataframe_horizontal_coords(
            pd.DataFrame(columns=["time", "x", "y"]), hdim1_coord="y"
        )

    with pytest.raises(ValueError):
        coord_utils.find_dataframe_horizontal_coords(
            pd.DataFrame(columns=["time", "x", "y"]), hdim2_coord="x"
        )

    # Test one exists but not both:
    with pytest.raises(ValueError):
        coord_utils.find_dataframe_horizontal_coords(
            pd.DataFrame(columns=["time", "x"])
        )

    with pytest.raises(ValueError):
        coord_utils.find_dataframe_horizontal_coords(
            pd.DataFrame(columns=["time", "y"])
        )

    # Test one of each exists
    with pytest.raises(ValueError):
        coord_utils.find_dataframe_horizontal_coords(
            pd.DataFrame(columns=["time", "x", "lat"])
        )

    # Test failure to detect coords when hdim1_coord or hdim2_coord is specified:
    with pytest.raises(ValueError):
        coord_utils.find_dataframe_horizontal_coords(
            pd.DataFrame(columns=["time", "x", "lat"]), hdim1_coord="y", coord_type="xy"
        )

    with pytest.raises(ValueError):
        coord_utils.find_dataframe_horizontal_coords(
            pd.DataFrame(columns=["time", "y", "lon"]), hdim2_coord="x", coord_type="xy"
        )

    with pytest.raises(ValueError):
        coord_utils.find_dataframe_horizontal_coords(
            pd.DataFrame(columns=["time", "x", "lon"]),
            hdim1_coord="lat",
            coord_type="latlon",
        )

    with pytest.raises(ValueError):
        coord_utils.find_dataframe_horizontal_coords(
            pd.DataFrame(columns=["time", "x", "lat"]),
            hdim1_coord="lon",
            coord_type="latlon",
        )


def test_find_dataframe_horizontal_coords_error_coord_type():
    """Test that find_dataframe_horizontal_coords raises errors correctly when
    the specified coord_type does not match the coords present
    """
    # Check that if coord_type is specified that an error is raised even if the other type of coords are present
    with pytest.raises(ValueError):
        coord_utils.find_dataframe_horizontal_coords(
            pd.DataFrame(columns=["time", "x", "y"]), coord_type="latlon"
        )

    with pytest.raises(ValueError):
        coord_utils.find_dataframe_horizontal_coords(
            pd.DataFrame(columns=["time", "lat", "lon"]), coord_type="xy"
        )


def test_find_dataframe_horizontal_coords_defaults_xy():
    """Test find_dataframe_horizontal_coords for xy coords"""
    # Test defaults xy:
    assert coord_utils.find_dataframe_horizontal_coords(
        pd.DataFrame(columns=["time", "x", "y"])
    ) == ("y", "x", "xy")

    assert coord_utils.find_dataframe_horizontal_coords(
        pd.DataFrame(
            columns=["time", "projection_x_coordinate", "projection_y_coordinate"]
        )
    ) == ("projection_y_coordinate", "projection_x_coordinate", "xy")

    # Test that xy take priority over latlon
    assert coord_utils.find_dataframe_horizontal_coords(
        pd.DataFrame(columns=["time", "x", "y", "lat", "lon"])
    ) == ("y", "x", "xy")


def test_find_dataframe_horizontal_coords_defaults_latlon():
    """Test find_dataframe_horizontal_coords for lat/lon coords"""
    # Test defaults latlon
    assert coord_utils.find_dataframe_horizontal_coords(
        pd.DataFrame(columns=["time", "lon", "lat"])
    ) == ("lat", "lon", "latlon")

    assert coord_utils.find_dataframe_horizontal_coords(
        pd.DataFrame(columns=["time", "Longitude", "Latitude"])
    ) == ("Latitude", "Longitude", "latlon")

    # Test that if only one of xy take latlon instead
    assert coord_utils.find_dataframe_horizontal_coords(
        pd.DataFrame(columns=["time", "x", "lat", "lon"])
    ) == ("lat", "lon", "latlon")

    # Test that setting coord_type to latlon ignores xy coords
    assert coord_utils.find_dataframe_horizontal_coords(
        pd.DataFrame(columns=["time", "x", "y", "lat", "lon"]), coord_type="latlon"
    ) == ("lat", "lon", "latlon")


def test_find_dataframe_horizontal_coords_specific():
    """Test find_dataframe_horizontal_coords when the coordinate name is
    specified
    """
    assert coord_utils.find_dataframe_horizontal_coords(
        pd.DataFrame(
            columns=[
                "time",
                "x",
                "y",
                "projection_x_coordinate",
                "projection_y_coordinate",
            ]
        ),
        hdim1_coord="y",
        hdim2_coord="x",
        coord_type="xy",
    ) == ("y", "x", "xy")

    assert coord_utils.find_dataframe_horizontal_coords(
        pd.DataFrame(
            columns=[
                "time",
                "x",
                "y",
                "projection_x_coordinate",
                "projection_y_coordinate",
            ]
        ),
        hdim1_coord="projection_y_coordinate",
        hdim2_coord="projection_x_coordinate",
        coord_type="xy",
    ) == ("projection_y_coordinate", "projection_x_coordinate", "xy")

    # Check that order does not matter
    assert coord_utils.find_dataframe_horizontal_coords(
        pd.DataFrame(
            columns=[
                "time",
                "x",
                "y",
                "projection_x_coordinate",
                "projection_y_coordinate",
            ]
        ),
        hdim1_coord="x",
        hdim2_coord="y",
        coord_type="xy",
    ) == ("x", "y", "xy")

    # Check that coord_type can be set wrong
    assert coord_utils.find_dataframe_horizontal_coords(
        pd.DataFrame(columns=["time", "x", "y", "lat", "lon"]),
        hdim1_coord="lat",
        hdim2_coord="lon",
        coord_type="xy",
    ) == ("lat", "lon", "xy")

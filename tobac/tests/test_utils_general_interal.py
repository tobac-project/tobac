"""Unit tests for tobac.utils.internal.general_internal.py"""

import pytest
import pandas as pd

import tobac.utils.internal.general_internal as gi_utils

def test_find_coord_in_dataframe():
    defaults = ["x", "projection_x_coordinate", "__other_name"]

    # Test no options raises ValueError:
    with pytest.raises(ValueError):
        gi_utils.find_coord_in_dataframe(
            pd.DataFrame(columns=["time", "x"])
        )

    # Test coordinate specified not in dataframe raise ValueError:
    with pytest.raises(ValueError):
        gi_utils.find_coord_in_dataframe(
            pd.DataFrame(columns=["time", "x"]), coord="projection_x_coordinate"
        )
    
    # Test no coordinates matching defaults:
    with pytest.raises(ValueError):
        gi_utils.find_coord_in_dataframe(
            pd.DataFrame(columns=["time", "y"]), defaults=defaults
        )

    # Test multiple matches with defaults:
    with pytest.raises(ValueError):
        gi_utils.find_coord_in_dataframe(
            pd.DataFrame(columns=["time", "x", "projection_x_coordinate"]), defaults=defaults
        )

    # Now test correct returns:
    assert gi_utils.find_coord_in_dataframe(
        pd.DataFrame(columns=["time", "x", "projection_x_coordinate"]), coord="x"
    ) == "x"

    assert gi_utils.find_coord_in_dataframe(
        pd.DataFrame(columns=["time", "x", "projection_x_coordinate"]), coord="projection_x_coordinate"
    ) == "projection_x_coordinate"

    assert gi_utils.find_coord_in_dataframe(
        pd.DataFrame(columns=["time", "x", "y"]), defaults=defaults
    ) == "x"

    assert gi_utils.find_coord_in_dataframe(
        pd.DataFrame(columns=["time", "projection_x_coordinate", "projection_y_coordinate"]), defaults=defaults
    ) == "projection_x_coordinate"

    assert gi_utils.find_coord_in_dataframe(
        pd.DataFrame(columns=["time", "x", "projection_x_coordinate"]), coord="x", defaults=defaults
    ) == "x"

def test_find_dataframe_vertical_coord_warning():
    # Test the warning for coord="auto":
    with pytest.warns(DeprecationWarning):
        gi_utils.find_dataframe_vertical_coord(pd.DataFrame(columns=["z"]), vertical_coord="auto")

def test_find_dataframe_vertical_coord_error():
    # Test the error for invalid coord input:
    with pytest.raises(ValueError):
        gi_utils.find_dataframe_vertical_coord(pd.DataFrame(columns=["z"]), vertical_coord="__bad_coord_name")

    # Test the error for no default coord found:
    with pytest.raises(ValueError):
        gi_utils.find_dataframe_vertical_coord(pd.DataFrame(columns=["x"]))

    # Test the error for multiple default coords found:
    with pytest.raises(ValueError):
        gi_utils.find_dataframe_vertical_coord(pd.DataFrame(columns=["z", "geopotential_height"]))

def test_find_dataframe_vertical_coord():
    # Test default coords
    assert gi_utils.find_dataframe_vertical_coord(pd.DataFrame(columns=["z"])) == "z"
    assert gi_utils.find_dataframe_vertical_coord(pd.DataFrame(columns=["geopotential_height"])) == "geopotential_height"

    # Test coord input
    assert gi_utils.find_dataframe_vertical_coord(pd.DataFrame(columns=["p"]), vertical_coord="p") == "p"

    # Test coord input when multiple default coords
    assert gi_utils.find_dataframe_vertical_coord(pd.DataFrame(columns=["z", "geopotential_height"]), vertical_coord="z") == "z"

def test_find_dataframe_horizontal_coords_error():
    # Test no matching coords
    with pytest.raises(ValueError):
        gi_utils.find_dataframe_horizontal_coords(pd.DataFrame(columns=["time", "z"]))

    # Test hdim_1_coord or hdim_2_coord set but not coord_type
    with pytest.raises(ValueError):
        gi_utils.find_dataframe_horizontal_coords(pd.DataFrame(columns=["time", "x", "y"]), hdim1_coord="y")

    with pytest.raises(ValueError):
        gi_utils.find_dataframe_horizontal_coords(pd.DataFrame(columns=["time", "x", "y"]), hdim2_coord="x")

    # Test one exists but not both:
    with pytest.raises(ValueError):
        gi_utils.find_dataframe_horizontal_coords(pd.DataFrame(columns=["time", "x"]))

    with pytest.raises(ValueError):
        gi_utils.find_dataframe_horizontal_coords(pd.DataFrame(columns=["time", "y"]))

    # Test one of each exists
    with pytest.raises(ValueError):
        gi_utils.find_dataframe_horizontal_coords(pd.DataFrame(columns=["time", "x", "lat"]))
    
def test_find_dataframe_horizontal_coords_error_coord_type():
    # Check that if coord_type is specified that an error is raised even if the other type of coords are present
    with pytest.raises(ValueError):
        gi_utils.find_dataframe_horizontal_coords(pd.DataFrame(columns=["time", "x", "y"]), coord_type="latlon")
    
    with pytest.raises(ValueError):
        gi_utils.find_dataframe_horizontal_coords(pd.DataFrame(columns=["time", "lat", "lon"]), coord_type="xy")

def test_find_dataframe_horizontal_coords_defaults_xy():
    # Test defaults xy:
    assert gi_utils.find_dataframe_horizontal_coords(pd.DataFrame(columns=["time", "x", "y"])) == ("y", "x", "xy")

    assert gi_utils.find_dataframe_horizontal_coords(pd.DataFrame(columns=["time", "projection_x_coordinate", "projection_y_coordinate"])) == ("projection_y_coordinate", "projection_x_coordinate", "xy")

    # Test that xy take priority over latlon
    assert gi_utils.find_dataframe_horizontal_coords(pd.DataFrame(columns=["time", "x", "y", "lat", "lon"])) == ("y", "x", "xy")

def test_find_dataframe_horizontal_coords_defaults_latlon():
    # Test defaults latlon
    assert gi_utils.find_dataframe_horizontal_coords(pd.DataFrame(columns=["time", "lon", "lat"])) == ("lat", "lon", "latlon")

    assert gi_utils.find_dataframe_horizontal_coords(pd.DataFrame(columns=["time", "Longitude", "Latitude"])) == ("Latitude", "Longitude", "latlon")

    # Test that if only one of xy take latlon instead
    assert gi_utils.find_dataframe_horizontal_coords(pd.DataFrame(columns=["time", "x", "lat", "lon"])) == ("lat", "lon", "latlon")

    # Test that setting coord_type to latlon ignores xy coords
    assert gi_utils.find_dataframe_horizontal_coords(pd.DataFrame(columns=["time", "x", "y", "lat", "lon"]), coord_type="latlon") == ("lat", "lon", "latlon")

def test_find_dataframe_horizontal_coords_specific():
    assert gi_utils.find_dataframe_horizontal_coords(
        pd.DataFrame(columns=["time", "x", "y", "projection_x_coordinate", "projection_y_coordinate"]), hdim1_coord="y", hdim2_coord="x", coord_type="xy"
    ) == ("y", "x", "xy")

    assert gi_utils.find_dataframe_horizontal_coords(
        pd.DataFrame(columns=["time", "x", "y", "projection_x_coordinate", "projection_y_coordinate"]), hdim1_coord="projection_y_coordinate", hdim2_coord="projection_x_coordinate", coord_type="xy"
    ) == ("projection_y_coordinate", "projection_x_coordinate", "xy")

    # Check that order does not matter
    assert gi_utils.find_dataframe_horizontal_coords(
        pd.DataFrame(columns=["time", "x", "y", "projection_x_coordinate", "projection_y_coordinate"]), hdim1_coord="x", hdim2_coord="y", coord_type="xy"
    ) == ("x", "y", "xy")

    # Check that coord_type can be set wrong
    assert gi_utils.find_dataframe_horizontal_coords(pd.DataFrame(columns=["time", "x", "y", "lat", "lon"]), hdim1_coord="lat", hdim2_coord="lon", coord_type="xy") == ("lat", "lon", "xy")


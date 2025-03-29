"""Internal tobac utilities"""

from __future__ import annotations
from typing import List, Literal, Optional, Tuple, Union, Callable

import numpy as np
import skimage.measure
import xarray as xr
import iris
import iris.cube
import pandas as pd
import warnings

from tobac.utils.decorators import irispandas_to_xarray, njit_if_available
from . import iris_utils
from . import xarray_utils as xr_utils

# list of common vertical coordinates to search for in various functions
COMMON_VERT_COORDS: list[str] = [
    "z",
    "model_level_number",
    "altitude",
    "geopotential_height",
]

COMMON_X_COORDS: list[str] = [
    "x",
    "projection_x_coordinate",
]

COMMON_Y_COORDS: list[str] = [
    "y",
    "projection_y_coordinate",
]

COMMON_LAT_COORDS: list[str] = [
    "latitude",
    "lat",
]

COMMON_LON_COORDS: list[str] = [
    "longitude",
    "lon",
    "long",
]


def _warn_auto_coordinate():
    """
    Internal function to warn on the use of `auto` as a default coordinate.
    """
    warnings.warn(
        '"auto" as a coordinate is deprecated. Use None instead.',
        DeprecationWarning,
    )


def get_label_props_in_dict(labels: np.array) -> dict:
    """Function to get the label properties into a dictionary format.

    Parameters
    ----------
    labels : 2D array-like
        Output of the `skimage.measure.label` function.

    Returns
    -------
    region_properties_dict: dict
        Output from skimage.measure.regionprops in dictionary
        format, where they key is the label number.
    """

    region_properties_raw = skimage.measure.regionprops(labels)
    region_properties_dict = {
        region_prop.label: region_prop for region_prop in region_properties_raw
    }

    return region_properties_dict


def get_indices_of_labels_from_reg_prop_dict(region_property_dict: dict) -> tuple[dict]:
    """Function to get the x, y, and z indices (as well as point count) of all labeled regions.
    Parameters
    ----------
    region_property_dict : dict of region_property objects
        This dict should come from the get_label_props_in_dict function.
    Returns
    -------
    curr_loc_indices : dict
        The number of points in the label number (key: label number).
    z_indices : dict
        The z indices in the label number. If a 2D property dict is passed, this value is not returned.
    y_indices : dict
        The y indices in the label number (key: label number).
    x_indices : dict
        The x indices in the label number (key: label number).
    Raises
    ------
    ValueError
        A ValueError is raised if there are no regions in the region
        property dict.
    """

    if len(region_property_dict) == 0:
        raise ValueError("No regions!")

    z_indices = dict()
    y_indices = dict()
    x_indices = dict()
    curr_loc_indices = dict()
    is_3D = False

    # loop through all skimage identified regions
    for region_prop_key in region_property_dict:
        region_prop = region_property_dict[region_prop_key]
        index = region_prop.label
        if len(region_prop.coords[0]) >= 3:
            is_3D = True
            curr_z_ixs, curr_y_ixs, curr_x_ixs = np.transpose(region_prop.coords)
            z_indices[index] = curr_z_ixs
        else:
            curr_y_ixs, curr_x_ixs = np.transpose(region_prop.coords)
            z_indices[index] = -1

        y_indices[index] = curr_y_ixs
        x_indices[index] = curr_x_ixs
        curr_loc_indices[index] = len(curr_y_ixs)
    # print("indices found")
    if is_3D:
        return [curr_loc_indices, z_indices, y_indices, x_indices]
    else:
        return [curr_loc_indices, y_indices, x_indices]


def find_vertical_axis_from_coord(
    variable_cube: Union[iris.cube.Cube, xr.DataArray],
    vertical_coord: Union[str, None] = None,
) -> str:
    """Function to find the vertical coordinate in the iris cube

    TODO: this function should be renamed

    Parameters
    ----------
    variable_cube: iris.cube.Cube or xarray.DataArray
        Input variable cube, containing a vertical coordinate.
    vertical_coord: str
        Vertical coordinate name. If None, this function tries to auto-detect.

    Returns
    -------
    str
        the vertical coordinate name

    Raises
    ------
    ValueError
        Raised if the vertical coordinate isn't found in the cube.
    """

    if vertical_coord == "auto":
        _warn_auto_coordinate()

    if isinstance(variable_cube, iris.cube.Cube):
        return iris_utils.find_vertical_axis_from_coord(variable_cube, vertical_coord)
    if isinstance(variable_cube, xr.DataArray):
        return xr_utils.find_vertical_coord_name(variable_cube, vertical_coord)

    raise ValueError("variable_cube must be xr.DataArray or iris.cube.Cube")


def find_coord_in_dataframe(
    variable_dataframe: Union[pd.DataFrame, pd.Series],
    coord: Optional[str] = None,
    defaults: Optional[List[str]] = None,
) -> str:
    """Find a coord in the columns of a dataframe matching either a specific coordinate name or a list
    of default values

    Parameters
    ----------
    variable_dataframe : pd.DataFrame
        Input dataframe
    coord : Optional[str], optional
        Coordinate name to search for, by default None
    defaults : Optional[List[str]], optional
        Default list of coordinates to search for if no coordinate name is provided by the coord
        parameter, by default None

    Returns
    -------
    str
        The coordinate name in the columns of the input dataframe

    Raises
    ------
    ValueError
        If the coordinate specified by the coord parameter is not present in the columns of the input dataframe
    ValueError
        If multiple coordinates in the default parameter are present in the columns of the input dataframe
    ValueError
        If no coordinates in the default parameter are present in the columns of the input dataframe
    ValueError
        If neither the coord or defaults parameters are set
    """
    if isinstance(variable_dataframe, pd.DataFrame):
        columns = variable_dataframe.columns
    elif isinstance(variable_dataframe, pd.Series):
        columns = variable_dataframe.index
    else:
        raise ValueError("Input variable_dataframe is neither a dataframe or a series")

    if coord is not None:
        if coord in columns:
            return coord
        else:
            raise ValueError(f"Coordinate {coord} is note present in the dataframe")

    if defaults is not None:
        intersect_id = np.intersect1d(
            columns.str.lower(), defaults, return_indices=True
        )[1]
        if len(intersect_id) == 1:
            return columns[intersect_id[0]]
        elif len(intersect_id) > 1:
            raise ValueError(
                "Multiple matching coord names found, please specify coordinate using coord parameter"
            )
        raise ValueError(
            f"No coordinate found matching defaults {defaults}, please specify coordinate using coord parameter"
        )

    raise ValueError("One of coord or defaults parameter must be set")


def find_dataframe_vertical_coord(
    variable_dataframe: pd.DataFrame, vertical_coord: Union[str, None] = None
) -> str:
    """Function to find the vertical coordinate in the iris cube

    Parameters
    ----------
    variable_dataframe: pandas.DataFrame
        Input dataframe, containing a vertical coordinate.
    vertical_coord: str
        Vertical coordinate name. If None, this function tries to auto-detect.

    Returns
    -------
    str
        the vertical coordinate name

    Raises
    ------
    ValueError
        Raised if the vertical coordinate isn't found in the cube.
    """

    if vertical_coord == "auto":
        _warn_auto_coordinate()
        vertical_coord = None

    return find_coord_in_dataframe(
        variable_dataframe, coord=vertical_coord, defaults=COMMON_VERT_COORDS
    )


def find_dataframe_horizontal_coords(
    variable_dataframe: pd.DataFrame,
    hdim1_coord: Optional[str] = None,
    hdim2_coord: Optional[str] = None,
    coord_type: Optional[Literal["xy", "latlon"]] = None,
) -> Tuple[str, str, str]:
    """Function to find the coordinates for the horizontal dimensions in a dataframe,
    either in Cartesian (xy) or Lat/Lon space. If both Cartesian and lat/lon coordinates
    exist, the cartesian coords will take priority

    Parameters
    ----------
    variable_dataframe : pd.DataFrame
        Input dataframe
    hdim1_coord : Optional[str], optional
        First horzontal coordinate name, by default None
    hdim2_coord : Optional[str], optional
        Second horizontal coordinate name, by default None
    coord_type : Optional[Literal[xy, latlon]], optional
        The coordinate type to search for, either 'xy' or 'latlon', must be set if
        providing either hdim1_coord or hdim2_coord parameters, by default None

    Returns
    -------
    Tuple[str, str, str]
        First horzontal coordinate name, second horizontal coordinate name, and the coordinate type

    Raises
    ------
    ValueError
        If coord_type is not set when either hdim1_coord or hdim2_coord are
    ValueError
        If no coordinates are found using the defaults for either xy or latlon
    """
    hdim_1_auto = hdim1_coord is None
    hdim_2_auto = hdim2_coord is None

    if coord_type is None and (not hdim_1_auto or not hdim_2_auto):
        raise ValueError(
            "Coord type parameter must be set if either hdim1_coord or hdim2_coord parameters are specified"
        )

    if coord_type in ["xy", None]:
        try:
            hdim1_coord_out = find_coord_in_dataframe(
                variable_dataframe, coord=hdim1_coord, defaults=COMMON_Y_COORDS
            )
        except ValueError as e:
            if not hdim_1_auto:
                raise e
            hdim1_coord_out = None

        try:
            hdim2_coord_out = find_coord_in_dataframe(
                variable_dataframe, coord=hdim2_coord, defaults=COMMON_X_COORDS
            )
        except ValueError as e:
            if not hdim_2_auto:
                raise e
            hdim2_coord_out = None

        if hdim1_coord_out is not None and hdim2_coord_out is not None:
            return hdim1_coord_out, hdim2_coord_out, "xy"
        else:
            # Reset output coords to None to ensure we don't match an xy coord in one dimension with latlon in another
            hdim1_coord_out = None
            hdim2_coord_out = None

    if coord_type in ["latlon", None]:
        try:
            hdim1_coord_out = find_coord_in_dataframe(
                variable_dataframe, coord=hdim1_coord, defaults=COMMON_LAT_COORDS
            )
            coord_type = "latlon"
        except ValueError as e:
            if not hdim_1_auto:
                raise e
            hdim1_coord_out = None

        try:
            hdim2_coord_out = find_coord_in_dataframe(
                variable_dataframe, coord=hdim2_coord, defaults=COMMON_LON_COORDS
            )
            coord_type = "latlon"
        except ValueError as e:
            if not hdim_2_auto:
                raise e
            hdim2_coord_out = None

        if hdim1_coord_out is not None and hdim2_coord_out is not None:
            return hdim1_coord_out, hdim2_coord_out, "latlon"

    raise ValueError(
        "No coordinates found matching defaults, please specify coordinate using hdim1_coord and hdim2_coord parameters"
    )


@njit_if_available
def calc_distance_coords(coords_1: np.array, coords_2: np.array) -> float:
    """Function to calculate the distance between cartesian
    coordinate set 1 and coordinate set 2.
    Parameters
    ----------
    coords_1: 2D or 3D array-like
        Set of coordinates passed in from trackpy of either (vdim, hdim_1, hdim_2)
        coordinates or (hdim_1, hdim_2) coordinates.
    coords_2: 2D or 3D array-like
        Similar to coords_1, but for the second pair of coordinates
    Returns
    -------
    float
        Distance between coords_1 and coords_2 in cartesian space.
    """

    is_3D = len(coords_1) == 3

    if not is_3D:
        # Let's make the accounting easier.
        coords_1 = np.array((0, coords_1[0], coords_1[1]))
        coords_2 = np.array((0, coords_2[0], coords_2[1]))

    deltas = coords_1 - coords_2
    return np.sqrt(np.sum(deltas**2))


def find_hdim_axes_3D(
    field_in: Union[iris.cube.Cube, xr.DataArray],
    vertical_coord: Union[str, None] = None,
    vertical_axis: Union[int, None] = None,
) -> tuple[int]:
    """Finds what the hdim axes are given a 3D (including z) or
    4D (including z and time) dataset.

    Parameters
    ----------
    field_in: iris cube or xarray dataarray
        Input field, can be 3D or 4D
    vertical_coord: str
        The name of the vertical coord, or None, which will attempt to find
        the vertical coordinate name
    vertical_axis: int or None
        The axis number of the vertical coordinate, or None. Note
        that only one of vertical_axis or vertical_coord can be set.

    Returns
    -------
    (hdim_1_axis, hdim_2_axis): (int, int)
        The axes for hdim_1 and hdim_2

    """

    if vertical_coord == "auto":
        _warn_auto_coordinate()

    if vertical_coord is not None and vertical_axis is not None:
        if vertical_coord != "auto":
            raise ValueError("Cannot set both vertical_coord and vertical_axis.")

    if type(field_in) is iris.cube.Cube:
        return iris_utils.find_hdim_axes_3d(field_in, vertical_coord, vertical_axis)
    elif type(field_in) is xr.DataArray:
        return xr_utils.find_hdim_axes_3d(field_in, vertical_coord, vertical_axis)
    else:
        raise ValueError("Unknown data type: " + type(field_in).__name__)


def find_axis_from_coord(
    variable_arr: Union[iris.cube.Cube, xr.DataArray], coord_name: str
) -> Union[int, None]:
    """Finds the axis number in an xarray or iris cube given a coordinate or dimension name.

    Parameters
    ----------
    variable_arr: iris.cube.Cube or xarray.DataArray
        Input variable cube
    coord_name: str
        coordinate or dimension to look for

    Returns
    -------
    axis_number: int
        the number of the axis of the given coordinate, or None if the coordinate
        is not found in the variable or not a dimensional coordinate
    """

    if isinstance(variable_arr, iris.cube.Cube):
        return iris_utils.find_axis_from_coord(variable_arr, coord_name)
    elif isinstance(variable_arr, xr.DataArray):
        return xr_utils.find_axis_from_dim_coord(variable_arr, coord_name)
    else:
        raise ValueError("variable_arr must be Iris Cube or Xarray DataArray")


@irispandas_to_xarray
def detect_latlon_coord_name(
    in_dataset: Union[xr.DataArray, iris.cube.Cube],
    latitude_name: Union[str, None] = None,
    longitude_name: Union[str, None] = None,
) -> tuple[str]:
    """Function to detect the name of latitude/longitude coordinates

    Parameters
    ----------
    in_dataset: iris.cube.Cube or xarray.DataArray
        Input dataset to detect names from
    latitude_name: str
        The name of the latitude coordinate. If None, tries to auto-detect.
    longitude_name: str
        The name of the longitude coordinate. If None, tries to auto-detect.

    Returns
    -------
    lat_name, lon_name: tuple(str)
        the detected names of the latitude and longitude coordinates. If
        coordinate is not detected, returns None for that coordinate.

    """

    if latitude_name == "auto" or longitude_name == "auto":
        _warn_auto_coordinate()

    out_lat = None
    out_lon = None
    test_lat_names = ["lat", "latitude"]
    test_lon_names = ["lon", "long", "longitude"]
    if latitude_name is not None and latitude_name != "auto":
        if latitude_name in in_dataset.coords:
            out_lat = latitude_name
    else:
        for test_lat_name in test_lat_names:
            if test_lat_name in in_dataset.coords:
                out_lat = test_lat_name
                break
    if longitude_name is not None and longitude_name != "auto":
        if longitude_name in in_dataset.coords:
            out_lon = longitude_name
    else:
        for test_lon_name in test_lon_names:
            if test_lon_name in in_dataset.coords:
                out_lon = test_lon_name
                break
    return out_lat, out_lon

"""Internal tobac utilities"""

from __future__ import annotations
import numpy as np
import skimage.measure
import xarray as xr
import iris
from tobac.utils.decorators import irispandas_to_xarray, njit_if_available
import iris.cube
import pandas as pd
import warnings
from . import iris_utils
from . import xarray_utils as xr_utils
from typing import Union, Callable

# list of common vertical coordinates to search for in various functions
COMMON_VERT_COORDS: list[str] = [
    "z",
    "model_level_number",
    "altitude",
    "geopotential_height",
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


def find_vertical_coord_name(
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


def find_dataframe_vertical_coord(
    variable_dataframe: pd.DataFrame, vertical_coord: Union[str, None] = None
) -> str:
    """Function to find the vertical coordinate in the iris cube

    Parameters
    ----------
    variable_dataframe: pandas.DataFrame
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

    if vertical_coord is None or vertical_coord == "auto":
        all_vertical_axes = list(
            set(variable_dataframe.columns) & set(COMMON_VERT_COORDS)
        )
        if len(all_vertical_axes) == 1:
            return all_vertical_axes[0]
        else:
            raise ValueError("Please specify vertical coordinate")

    else:
        if vertical_coord in variable_dataframe.columns:
            return vertical_coord
        else:
            raise ValueError("Please specify vertical coordinate")


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
) -> tuple[int, int]:
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

    if isinstance(field_in, iris.cube.Cube):
        return iris_utils.find_hdim_axes_3d(field_in, vertical_coord, vertical_axis)
    elif isinstance(field_in, xr.DataArray):
        return xr_utils.find_hdim_axes_3d(field_in, vertical_coord, vertical_axis)
    else:
        raise ValueError("Unknown data type: " + type(field_in).__name__)


def find_axis_from_coord(
    variable_arr: Union[iris.cube.Cube, xr.DataArray], coord_name: str
) -> int:
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


@irispandas_to_xarray()
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

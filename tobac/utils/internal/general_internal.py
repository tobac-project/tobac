"""Internal tobac utilities
"""
from __future__ import annotations

import numpy as np
import skimage.measure
import xarray as xr
import iris
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


def iris_to_xarray(func: Callable) -> Callable:
    """Decorator that converts all input of a function that is in the form of
    Iris cubes into xarray DataArrays and converts all outputs with type
    xarray DataArrays back into Iris cubes.

    Parameters
    ----------
    func : function
        Function to be decorated

    Returns
    -------
    wrapper : function
        Function including decorator
    """

    import iris
    import xarray

    def wrapper(*args, **kwargs):
        # print(kwargs)
        if any([type(arg) == iris.cube.Cube for arg in args]) or any(
            [type(arg) == iris.cube.Cube for arg in kwargs.values()]
        ):
            # print("converting iris to xarray and back")
            args = tuple(
                [
                    xarray.DataArray.from_iris(arg)
                    if type(arg) == iris.cube.Cube
                    else arg
                    for arg in args
                ]
            )
            kwargs_new = dict(
                zip(
                    kwargs.keys(),
                    [
                        xarray.DataArray.from_iris(arg)
                        if type(arg) == iris.cube.Cube
                        else arg
                        for arg in kwargs.values()
                    ],
                )
            )
            # print(args)
            # print(kwargs)
            output = func(*args, **kwargs_new)
            if type(output) == tuple:
                output = tuple(
                    [
                        xarray.DataArray.to_iris(output_item)
                        if type(output_item) == xarray.DataArray
                        else output_item
                        for output_item in output
                    ]
                )
            elif type(output) == xarray.DataArray:
                output = xarray.DataArray.to_iris(output)
            # if output is neither tuple nor an xr.DataArray
            else:
                output = func(*args, **kwargs)

        else:
            output = func(*args, **kwargs)

        return output

    return wrapper


def xarray_to_iris(func: Callable) -> Callable:
    """Decorator that converts all input of a function that is in the form of
    xarray DataArrays into Iris cubes and converts all outputs with type
    Iris cubes back into xarray DataArrays.

    Parameters
    ----------
    func : function
        Function to be decorated.

    Returns
    -------
    wrapper : function
        Function including decorator.

    Examples
    --------
    >>> segmentation_xarray = xarray_to_iris(segmentation)

    This line creates a new function that can process xarray fields and
    also outputs fields in xarray format, but otherwise works just like
    the original function:

    >>> mask_xarray, features = segmentation_xarray(
        features, data_xarray, dxy, threshold
        )
    """

    import iris
    import xarray

    def wrapper(*args, **kwargs):
        # print(args)
        # print(kwargs)
        if any([type(arg) == xarray.DataArray for arg in args]) or any(
            [type(arg) == xarray.DataArray for arg in kwargs.values()]
        ):
            # print("converting xarray to iris and back")
            args = tuple(
                [
                    xarray.DataArray.to_iris(arg)
                    if type(arg) == xarray.DataArray
                    else arg
                    for arg in args
                ]
            )
            if kwargs:
                kwargs_new = dict(
                    zip(
                        kwargs.keys(),
                        [
                            xarray.DataArray.to_iris(arg)
                            if type(arg) == xarray.DataArray
                            else arg
                            for arg in kwargs.values()
                        ],
                    )
                )
            else:
                kwargs_new = kwargs
            # print(args)
            # print(kwargs)
            output = func(*args, **kwargs_new)
            if type(output) == tuple:
                output = tuple(
                    [
                        xarray.DataArray.from_iris(output_item)
                        if type(output_item) == iris.cube.Cube
                        else output_item
                        for output_item in output
                    ]
                )
            else:
                if type(output) == iris.cube.Cube:
                    output = xarray.DataArray.from_iris(output)

        else:
            output = func(*args, **kwargs)
        # print(output)
        return output

    return wrapper


def irispandas_to_xarray(func: Callable) -> Callable:
    """Decorator that converts all input of a function that is in the form of
    Iris cubes/pandas Dataframes into xarray DataArrays/xarray Datasets and
    converts all outputs with the type xarray DataArray/xarray Dataset
    back into Iris cubes/pandas Dataframes.

    Parameters
    ----------
    func : function
        Function to be decorated.

    Returns
    -------
    wrapper : function
        Function including decorator.
    """
    import iris
    import xarray
    import pandas as pd

    def wrapper(*args, **kwargs):
        # print(kwargs)
        if any(
            [(type(arg) == iris.cube.Cube or type(arg) == pd.DataFrame) for arg in args]
        ) or any(
            [
                (type(arg) == iris.cube.Cube or type(arg) == pd.DataFrame)
                for arg in kwargs.values()
            ]
        ):
            # print("converting iris to xarray and back")
            args = tuple(
                [
                    xarray.DataArray.from_iris(arg)
                    if type(arg) == iris.cube.Cube
                    else arg.to_xarray()
                    if type(arg) == pd.DataFrame
                    else arg
                    for arg in args
                ]
            )
            kwargs = dict(
                zip(
                    kwargs.keys(),
                    [
                        xarray.DataArray.from_iris(arg)
                        if type(arg) == iris.cube.Cube
                        else arg.to_xarray()
                        if type(arg) == pd.DataFrame
                        else arg
                        for arg in kwargs.values()
                    ],
                )
            )

            output = func(*args, **kwargs)
            if type(output) == tuple:
                output = tuple(
                    [
                        xarray.DataArray.to_iris(output_item)
                        if type(output_item) == xarray.DataArray
                        else output_item.to_dataframe()
                        if type(output_item) == xarray.Dataset
                        else output_item
                        for output_item in output
                    ]
                )
            else:
                if type(output) == xarray.DataArray:
                    output = xarray.DataArray.to_iris(output)
                elif type(output) == xarray.Dataset:
                    output = output.to_dataframe()

        else:
            output = func(*args, **kwargs)
        return output

    return wrapper


def xarray_to_irispandas(func: Callable) -> Callable:
    """Decorator that converts all input of a function that is in the form of
    DataArrays/xarray Datasets into xarray Iris cubes/pandas Dataframes and
    converts all outputs with the type Iris cubes/pandas Dataframes back into
    xarray DataArray/xarray Dataset.

    Parameters
    ----------
    func : function
        Function to be decorated.

    Returns
    -------
    wrapper : function
        Function including decorator.

    Examples
    --------
    >>> linking_trackpy_xarray = xarray_to_irispandas(
        linking_trackpy
        )

    This line creates a new function that can process xarray inputs and
    also outputs in xarray formats, but otherwise works just like the
    original function:

    >>> track_xarray = linking_trackpy_xarray(
        features_xarray, field_xarray, dt, dx
        )
    """
    import iris
    import xarray
    import pandas as pd

    def wrapper(*args, **kwargs):
        # print(args)
        # print(kwargs)
        if any(
            [
                (type(arg) == xarray.DataArray or type(arg) == xarray.Dataset)
                for arg in args
            ]
        ) or any(
            [
                (type(arg) == xarray.DataArray or type(arg) == xarray.Dataset)
                for arg in kwargs.values()
            ]
        ):
            # print("converting xarray to iris and back")
            args = tuple(
                [
                    xarray.DataArray.to_iris(arg)
                    if type(arg) == xarray.DataArray
                    else arg.to_dataframe()
                    if type(arg) == xarray.Dataset
                    else arg
                    for arg in args
                ]
            )
            if kwargs:
                kwargs_new = dict(
                    zip(
                        kwargs.keys(),
                        [
                            xarray.DataArray.to_iris(arg)
                            if type(arg) == xarray.DataArray
                            else arg.to_dataframe()
                            if type(arg) == xarray.Dataset
                            else arg
                            for arg in kwargs.values()
                        ],
                    )
                )
            else:
                kwargs_new = kwargs
            # print(args)
            # print(kwargs)
            output = func(*args, **kwargs_new)
            if type(output) == tuple:
                output = tuple(
                    [
                        xarray.DataArray.from_iris(output_item)
                        if type(output_item) == iris.cube.Cube
                        else output_item.to_xarray()
                        if type(output_item) == pd.DataFrame
                        else output_item
                        for output_item in output
                    ]
                )
            else:
                if type(output) == iris.cube.Cube:
                    output = xarray.DataArray.from_iris(output)
                elif type(output) == pd.DataFrame:
                    output = output.to_xarray()

        else:
            output = func(*args, **kwargs)
        # print(output)
        return output

    return wrapper


def njit_if_available(func: Callable, **kwargs) -> Callable:
    """Decorator to wrap a function with numba.njit if available.
    If numba isn't available, it just returns the function.

    Parameters
    ----------
    func: function object
        Function to wrap with njit
    kwargs:
        Keyword arguments to pass to numba njit
    """
    try:
        from numba import njit

        return njit(func, kwargs)
    except KeyboardInterrupt as kie:
        raise
    except Exception as exc:
        warnings.warn(
            "Numba not able to be imported; periodic boundary calculations will be slower."
            "Exception raised: " + repr(exc)
        )
        return func


def find_vertical_axis_from_coord(
    variable_cube: Union[iris.cube.Cube, xr.DataArray],
    vertical_coord: Union[str, None] = None,
) -> str:
    """Function to find the vertical coordinate in the iris cube

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
    if isinstance(variable_cube, xr.Dataset) or isinstance(variable_cube, xr.DataArray):
        return xr_utils.find_vertical_axis_from_coord(variable_cube, vertical_coord)

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
        raise NotImplementedError("Xarray find_hdim_axes_3D not implemented")
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
        raise NotImplementedError(
            "xarray version of find_axis_from_coord not implemented."
        )
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

"""Internal tobac utilities for xarray datasets/dataarrays
"""
from __future__ import annotations


from typing import Union
import numpy as np
import xarray as xr
from . import general_internal as tb_utils_gi


def find_axis_from_dim_coord(
    in_da: xr.DataArray, dim_coord_name: str
) -> Union[int, None]:
    """Finds the axis number in an xarray dataarray given a coordinate or
     dimension name.

    Parameters
    ----------
    in_da: xarray.DataArray
        Input variable array
    dim_coord_name: str
        coordinate or dimension to look for

    Returns
    -------
    axis_number: int
        the number of the axis of the given coordinate, or None if the coordinate
        is not found in the cube or not a dimensional coordinate

    Raises
    ------
    ValueError
        Returns ValueError if there are more than one matching dimension name or
        if the dimension/coordinate isn't found.
    """

    dim_axis = find_axis_from_dim(in_da, dim_coord_name)

    try:
        coord_axes = find_axis_from_coord(in_da, dim_coord_name)
    except ValueError:
        coord_axes = tuple()

    if dim_axis is None and len(coord_axes) == 0:
        raise ValueError("Coordinate/Dimension " + dim_coord_name + " not found.")

    # if we find a dimension with an axis and/or the coordinates, return that axis number
    if len(coord_axes) == 1 and dim_axis == coord_axes[0]:
        return dim_axis
    if len(coord_axes) == 0 and dim_axis is not None:
        return dim_axis
    if dim_axis is None and len(coord_axes) == 1:
        return coord_axes[0]

    return None


def find_axis_from_dim(in_da: xr.DataArray, dim_name: str) -> Union[int, None]:
    """
    Finds the axis number from a DataArray dimension name

    Parameters
    ----------
    in_da: xarray.DataArray
        Input DataArray to find the axis number from
    dim_name: str
        Name of the dimension

    Returns
    -------
    int or None
        axis number or None if axis isn't found

    Raises
    ------
    ValueError
        raises ValueError if dim_name matches multiple dimensions
    """
    list_dims = in_da.dims
    all_matching_dims = [
        dim
        for dim in list_dims
        if dim
        in [
            dim_name,
        ]
    ]
    if len(all_matching_dims) == 1:
        return list_dims.index(all_matching_dims[0])
    if len(all_matching_dims) > 1:
        raise ValueError(
            "More than one matching dimension. Need to specify which axis number or rename "
            "your dimensions."
        )
    return None


def find_axis_from_coord(in_da: xr.DataArray, coord_name: str) -> tuple[int]:
    """
    Finds the axis number from a DataArray coordinate name

    Parameters
    ----------
    in_da: xarray.DataArray
        Input DataArray to find the axis number from
    coord_name: str
        Name of the coordinate

    Returns
    -------
    tuple of int
        axis number(s)

    Raises
    ------
    ValueError
        raises ValueError if the coordinate has more than 1 axis or 0 axes; or if
        there are >1 matching coordinate of that name
    """
    list_coords = in_da.coords
    all_matching_coords = list(set(list_coords) & {coord_name})
    if len(all_matching_coords) == 1:
        curr_coord = list_coords[all_matching_coords[0]]
        return tuple(
            (
                find_axis_from_dim(in_da, x)
                for x in curr_coord.dims
                if find_axis_from_dim(in_da, x) is not None
            )
        )

    if len(all_matching_coords) > 1:
        raise ValueError("Too many matching coords")
    return tuple()


def find_vertical_coord_name(
    variable_cube: xr.DataArray,
    vertical_coord: Union[str, None] = None,
) -> str:
    """Function to find the vertical coordinate in the iris cube

    Parameters
    ----------
    variable_cube: xarray.DataArray
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

    list_coord_names = variable_cube.coords

    if vertical_coord is None or vertical_coord == "auto":
        # find the intersection
        all_vertical_axes = list(
            set(list_coord_names) & set(tb_utils_gi.COMMON_VERT_COORDS)
        )
        if len(all_vertical_axes) >= 1:
            return all_vertical_axes[0]
        coord_names_err = str(tuple(tb_utils_gi.COMMON_VERT_COORDS))
        raise ValueError(
            "Cube lacks suitable automatic vertical coordinate " + coord_names_err
        )
    if vertical_coord in list_coord_names:
        return vertical_coord
    raise ValueError("Please specify vertical coordinate found in cube")


def find_hdim_axes_3d(
    field_in: xr.DataArray,
    vertical_coord: Union[str, None] = None,
    vertical_axis: Union[int, None] = None,
    time_dim_coord_name: str = "time",
) -> tuple[int]:
    """Finds what the hdim axes are given a 3D (including z) or
    4D (including z and time) dataset.

    Parameters
    ----------
    field_in: xarray.DataArray
        Input field, can be 3D or 4D
    vertical_coord: str or None
        The name of the vertical coord, or None, which will attempt to find
        the vertical coordinate name
    vertical_axis: int or None
        The axis number of the vertical coordinate, or None. Note
        that only one of vertical_axis or vertical_coord can be set.
    time_dim_coord_name: str
        Name of the time dimension/coordinate

    Returns
    -------
    (hdim_1_axis, hdim_2_axis): (int, int)
        The axes for hdim_1 and hdim_2
    """

    if vertical_coord is not None and vertical_axis is not None:
        if vertical_coord != "auto":
            raise ValueError("Cannot set both vertical_coord and vertical_axis.")

    time_axis = find_axis_from_dim_coord(field_in, time_dim_coord_name)
    # we have already specified the axis.
    if vertical_axis is not None:
        vertical_coord_axis = vertical_axis
        vert_coord_found = True
    else:
        try:
            vertical_coord_name = find_vertical_coord_name(
                field_in, vertical_coord=vertical_coord
            )
            vert_coord_found = True
            ndim_vertical = find_axis_from_dim_coord(field_in, vertical_coord_name)
            if ndim_vertical is None:
                raise ValueError(
                    "please specify 1 dimensional vertical coordinate."
                    f" Current vertical coordinates: {ndim_vertical}"
                )
            vertical_coord_axis = ndim_vertical

        except ValueError:
            vert_coord_found = False

    if not vert_coord_found:
        # if we don't have a vertical coordinate, and we are 3D or lower
        # that is okay.
        if (field_in.ndim == 3 and time_axis is not None) or field_in.ndim < 3:
            vertical_coord_axis = None
        else:
            raise ValueError("No suitable vertical coordinate found")
    # Once we know the vertical coordinate, we can resolve the
    # horizontal coordinates

    all_axes = np.arange(0, field_in.ndim)
    output_vals = tuple(
        all_axes[np.logical_not(np.isin(all_axes, [time_axis, vertical_coord_axis]))]
    )
    return output_vals

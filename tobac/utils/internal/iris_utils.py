"""Internal tobac utilities for iris cubes
The goal will be to, ultimately, remove these when we sunset iris
"""
from __future__ import annotations

from typing import Union

import iris
import iris.cube
import numpy as np

from . import basic as tb_utils_gi


def find_axis_from_coord(
    variable_cube: iris.cube.Cube, coord_name: str
) -> Union[int, None]:
    """Finds the axis number in an iris cube given a coordinate name.

    Parameters
    ----------
    variable_cube: iris.cube
        Input variable cube
    coord_name: str
        coordinate to look for

    Returns
    -------
    axis_number: int
        the number of the axis of the given coordinate, or None if the coordinate
        is not found in the cube or not a dimensional coordinate
    """

    list_coord_names = [coord.name() for coord in variable_cube.coords()]
    all_matching_axes = list(set(list_coord_names) & {coord_name})
    if (
        len(all_matching_axes) == 1
        and len(variable_cube.coord_dims(all_matching_axes[0])) > 0
    ):
        return variable_cube.coord_dims(all_matching_axes[0])[0]
    if len(all_matching_axes) > 1:
        raise ValueError("Too many axes matched.")

    return None


def find_vertical_axis_from_coord(
    variable_cube: iris.cube.Cube, vertical_coord: Union[str, None] = None
) -> str:
    """Function to find the vertical coordinate in the iris cube

    Parameters
    ----------
    variable_cube: iris.cube
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

    list_coord_names = [coord.name() for coord in variable_cube.coords()]

    if vertical_coord is None or vertical_coord == "auto":
        # find the intersection
        all_vertical_axes = list(
            set(list_coord_names) & set(tb_utils_gi.COMMON_VERT_COORDS)
        )
        if len(all_vertical_axes) >= 1:
            return all_vertical_axes[0]
        raise ValueError(
            "Cube lacks suitable automatic vertical coordinate (z, model_level_number, altitude, "
            "or geopotential_height)"
        )
    if vertical_coord in list_coord_names:
        return vertical_coord
    raise ValueError("Please specify vertical coordinate found in cube")


def find_hdim_axes_3d(
    field_in: iris.cube.Cube,
    vertical_coord: Union[str, None] = None,
    vertical_axis: Union[int, None] = None,
) -> tuple[int]:
    """Finds what the hdim axes are given a 3D (including z) or
    4D (including z and time) dataset.

    Parameters
    ----------
    field_in: iris cube
        Input field, can be 3D or 4D
    vertical_coord: str or None
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

    if vertical_coord is not None and vertical_axis is not None:
        if vertical_coord != "auto":
            raise ValueError("Cannot set both vertical_coord and vertical_axis.")

    time_axis = find_axis_from_coord(field_in, "time")
    if vertical_axis is not None:
        vertical_coord_axis = vertical_axis
        vert_coord_found = True
    else:
        try:
            vertical_axis = find_vertical_axis_from_coord(
                field_in, vertical_coord=vertical_coord
            )
        except ValueError:
            vert_coord_found = False
        else:
            vert_coord_found = True
            ndim_vertical = field_in.coord_dims(vertical_axis)
            if len(ndim_vertical) > 1:
                raise ValueError(
                    "please specify 1 dimensional vertical coordinate."
                    f" Current vertical coordinates: {ndim_vertical}"
                )
            if len(ndim_vertical) != 0:
                vertical_coord_axis = ndim_vertical[0]
            else:
                # this means the vertical coordinate is an auxiliary coordinate of some kind.
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

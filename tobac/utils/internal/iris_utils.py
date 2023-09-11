"""Internal tobac utilities for iris cubes
The goal will be to, ultimately, remove these when we sunset iris
"""

from typing import Union
import iris
import iris.cube


def find_axis_from_coord(variable_cube: iris.cube.Cube, coord_name: str):
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
):
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
    list_vertical = [
        "z",
        "model_level_number",
        "altitude",
        "geopotential_height",
    ]

    list_coord_names = [coord.name() for coord in variable_cube.coords()]

    if vertical_coord is None or vertical_coord == "auto":
        # find the intersection
        all_vertical_axes = list(set(list_coord_names) & set(list_vertical))
        if len(all_vertical_axes) >= 1:
            return all_vertical_axes[0]
        raise ValueError(
            "Cube lacks suitable automatic vertical coordinate (z, model_level_number, altitude, "
            "or geopotential_height)"
        )
    if vertical_coord in list_coord_names:
        return vertical_coord
    raise ValueError("Please specify vertical coordinate found in cube")

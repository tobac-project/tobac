"""Internal tobac utilities for xarray datasets/dataarrays
"""

from typing import Union
import xarray as xr
from . import general_internal as tb_utils_gi


def find_vertical_axis_from_coord(
    variable_cube: xr.DataArray,
    vertical_coord: Union[str, None] = None,
):
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

    list_coord_names = variable_cube.coords

    if vertical_coord is None or vertical_coord == "auto":
        # find the intersection
        all_vertical_axes = list(
            set(list_coord_names) & set(tb_utils_gi.COMMON_VERT_COORDS)
        )
        if len(all_vertical_axes) >= 1:
            return all_vertical_axes[0]
        raise ValueError(
            "Cube lacks suitable automatic vertical coordinate (z, model_level_number, "
            "altitude, or geopotential_height)"
        )
    if vertical_coord in list_coord_names:
        return vertical_coord
    raise ValueError("Please specify vertical coordinate found in cube")

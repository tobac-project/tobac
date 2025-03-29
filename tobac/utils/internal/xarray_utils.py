"""Internal tobac utilities for xarray datasets/dataarrays"""

from __future__ import annotations

import copy
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr
from . import coordinates as tb_utils_gi


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
        the number of the axis of the given coordinate

    Raises
    ------
    ValueError
        Returns ValueError if there are more than one matching dimension name or
        if the dimension/coordinate isn't found.
    """
    try:
        dim_axis = find_axis_from_dim(in_da, dim_coord_name)
    except ValueError:
        dim_axis = None

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
    raise ValueError("Coordinate/Dimension " + dim_coord_name + " not found.")
    # return None


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
    raise ValueError("Dimension not found. ")


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
    raise ValueError("No matching coords")


def find_vertical_coord_name(
    variable_da: xr.DataArray,
    vertical_coord: Union[str, None] = None,
) -> str:
    """Function to find the vertical coordinate in the iris cube

    Parameters
    ----------
    variable_da: xarray.DataArray
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

    list_coord_names = variable_da.coords

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
    return_vertical_axis: bool = False,
) -> Union[tuple[int, int], tuple[int, int, int]]:
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
    return_vertical_axis: bool
        True if you want to also return the vertical axis number as the last value

    Returns
    -------
    (hdim_1_axis, hdim_2_axis): (int, int)
        The axes for hdim_1 and hdim_2
    """

    if vertical_coord is not None and vertical_axis is not None:
        if vertical_coord != "auto":
            raise ValueError("Cannot set both vertical_coord and vertical_axis.")

    try:
        time_axis = find_axis_from_dim_coord(field_in, time_dim_coord_name)
    except ValueError:
        # time axis not found
        time_axis = None
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
    if return_vertical_axis:
        output_vals = (*output_vals, vertical_coord_axis)
    return output_vals


def add_coordinates_to_features(
    feature_df: pd.DataFrame,
    variable_da: xr.DataArray,
    vertical_coord: Union[str, None] = None,
    vertical_axis: Union[int, None] = None,
    use_standard_names: bool = True,
    interp_dims_without_coords: bool = False,
) -> pd.DataFrame:
    """Function to populate the interpolated coordinates to feature

    Parameters
    ----------
    feature_df: pandas DataFrame
        Feature dataframe
    variable_da: xarray.DataArray
        DataArray (usually the one you are tracking on) at least conaining the dimension of 'time'.
        Typically, 'longitude','latitude','x_projection_coordinate','y_projection_coordinate',
        and 'altitude' (if 3D) are the coordinates that we expect, although this function
        will happily interpolate along any coordinates you give.
    vertical_coord: str
        Name of the vertical coordinate. If None, tries to auto-detect.
        If it is a string, it looks for the coordinate or the dimension name corresponding
        to the string. If it is an int, it assumes that it is the vertical axis.
        Note that if you only have a 2D or 3D coordinate for altitude, you must
        pass in an int.
    vertical_axis: int or None
        Axis number of the vertical.
    use_standard_names: bool
        If true, when interpolating a coordinate, it looks for a standard_name
        and uses that to name the output coordinate, to mimic iris functionality.
        If false, uses the actual name of the coordinate to output.
    interp_dims_without_coords: bool
        If True, interpolates dimensions without coordinates
        If False, skips dimensions without coordinates

    Returns
    -------
    Dataframe with coordinates added

    """
    # make a copy to avoid editing in place.
    return_feat_df = copy.deepcopy(feature_df)

    time_dim_name: str = "time"
    # first, we must find the names of the dimensions corresponding to the numbered
    # dimensions.

    ndims: int = variable_da.ndim

    time_dim_number = find_axis_from_dim(variable_da, time_dim_name)

    is_3d = (time_dim_number is not None and ndims == 4) or (
        time_dim_number is None and ndims == 3
    )
    vdim_coord = None
    if is_3d:
        hdim1_axis, hdim2_axis, vertical_axis = find_hdim_axes_3d(
            variable_da,
            vertical_coord,
            vertical_axis,
            time_dim_coord_name=time_dim_name,
            return_vertical_axis=True,
        )
        if vertical_axis is None:
            vdim_coord = find_vertical_coord_name(variable_da, vertical_coord)
            vertical_axis = find_axis_from_dim_coord(variable_da, vdim_coord)
        else:
            vdim_coord = variable_da.dims[vertical_axis]
    else:  # 2D
        if ndims == 2:
            hdim1_axis = 0
            hdim2_axis = 1
        elif ndims == 3 and time_dim_number is not None:
            possible_dims = [0, 1, 2]
            possible_dims.pop(time_dim_number)
            hdim1_axis, hdim2_axis = possible_dims
        else:
            raise ValueError("DataArray has too many or too few dimensions")

    # If the dimensions share a name with the coordinates and those coordinates do not match
    # with the i, j, k-style indices, you cannot `interp` along those i, j, k indices.
    # so, instead, we rename the dimensions to random strings so that we can
    # run interpolation.

    hdim1_name_original = variable_da.dims[hdim1_axis]
    hdim2_name_original = variable_da.dims[hdim2_axis]

    # generate random names for the new coordinates that are based on i, j, k values
    hdim1_name_new = "__temp_hdim1_name"
    hdim2_name_new = "__temp_hdim2_name"
    vdim_name_new = "__temp_vdim_name"
    time_name_new = "__temp_time_name"

    if (
        hdim1_name_new in variable_da.dims
        or hdim2_name_new in variable_da.dims
        or vdim_name_new in variable_da.dims
    ):
        raise ValueError(
            "Cannot have dimensions named {0}, {1}, or {2}".format(
                hdim1_name_new, hdim2_name_new, vdim_name_new
            )
        )

    dim_new_names = {
        hdim1_name_original: hdim1_name_new,
        hdim2_name_original: hdim2_name_new,
        time_dim_name: time_name_new,
    }
    dim_interp_coords = {
        hdim1_name_new: xr.DataArray(
            return_feat_df["hdim_1"].values,
            dims="features",
            coords={"features": return_feat_df.index},
        ),
        hdim2_name_new: xr.DataArray(
            return_feat_df["hdim_2"].values,
            dims="features",
            coords={"features": return_feat_df.index},
        ),
        time_name_new: xr.DataArray(
            return_feat_df["frame"].values,
            dims="features",
            coords={"features": return_feat_df.index},
        ),
    }

    if is_3d:
        vdim_name_original = variable_da.dims[vertical_axis]
        dim_interp_coords[vdim_name_new] = xr.DataArray(
            return_feat_df["vdim"].values, dims="features"
        )

        dim_new_names[vdim_name_original] = vdim_name_new

    # you can only rename dims alone when operating on datasets, so add our dataarray to a
    # dataset
    renamed_dim_da = variable_da.swap_dims(dim_new_names)
    # interpolated_df = renamed_dim_da.interp(coords=dim_interp_coords)
    # interpolated_df = interpolated_df.drop_vars(
    #     [hdim1_name_new, hdim2_name_new, vdim_name_new], errors="ignore"
    # )
    return_feat_df[time_dim_name] = variable_da[time_dim_name].values[
        return_feat_df["frame"]
    ]
    return_feat_df[time_dim_name + "str"] = [
        pd.to_datetime(str(x)).strftime("%Y-%m-%d %H:%M:%S")
        for x in variable_da[time_dim_name].values[return_feat_df["frame"]]
    ]

    for interp_coord in renamed_dim_da.coords:
        # skip time coordinate because we dealt with that already
        if interp_coord == time_dim_name:
            continue

        if interp_coord not in variable_da.coords and not interp_dims_without_coords:
            continue

        interp_coord_name = interp_coord
        # if we have standard names and are using them, rename our coordinates.
        if use_standard_names:
            try:
                interp_coord_name = renamed_dim_da[interp_coord].attrs["standard_name"]
            except KeyError:
                pass

        return_feat_df[interp_coord_name] = renamed_dim_da[interp_coord].interp(
            coords={
                dim: dim_interp_coords[dim] for dim in renamed_dim_da[interp_coord].dims
            }
        )
    return return_feat_df

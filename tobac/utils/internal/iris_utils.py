"""Internal tobac utilities for iris cubes
The goal will be to, ultimately, remove these when we sunset iris
"""
from __future__ import annotations

from typing import Union
import logging

import iris
import iris.cube
import numpy as np
import pandas as pd

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
) -> tuple[int, int]:
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


def add_coordinates(t: pd.DataFrame, variable_cube: iris.cube.Cube) -> pd.DataFrame:
    """Add coordinates from the input cube of the feature detection
    to the trajectories/features.

    Parameters
    ----------
    t : pandas.DataFrame
        Trajectories/features from feature detection or linking step.

    variable_cube : iris.cube.Cube
        Input data used for the tracking with coordinate information
        to transfer to the resulting DataFrame. Needs to contain the
        coordinate 'time'.

    Returns
    -------
    t : pandas.DataFrame
        Trajectories with added coordinates.

    """

    from scipy.interpolate import interp1d, interpn

    logging.debug("start adding coordinates from cube")

    # pull time as datetime object and timestr from input data and add it to DataFrame:
    t["time"] = None
    t["timestr"] = None

    logging.debug("adding time coordinate")

    time_in = variable_cube.coord("time")
    time_in_datetime = time_in.units.num2date(time_in.points)

    t["time"] = time_in_datetime[t["frame"]]
    t["timestr"] = [
        x.strftime("%Y-%m-%d %H:%M:%S") for x in time_in_datetime[t["frame"]]
    ]

    # Get list of all coordinates in input cube except for time (already treated):
    coord_names = [coord.name() for coord in variable_cube.coords()]
    coord_names.remove("time")

    logging.debug("time coordinate added")

    # chose right dimension for horizontal axis based on time dimension:
    ndim_time = variable_cube.coord_dims("time")[0]
    if ndim_time == 0:
        hdim_1 = 1
        hdim_2 = 2
    elif ndim_time == 1:
        hdim_1 = 0
        hdim_2 = 2
    elif ndim_time == 2:
        hdim_1 = 0
        hdim_2 = 1

    # create vectors to use to interpolate from pixels to coordinates
    dimvec_1 = np.arange(variable_cube.shape[hdim_1])
    dimvec_2 = np.arange(variable_cube.shape[hdim_2])

    # loop over coordinates in input data:
    for coord in coord_names:
        logging.debug("adding coord: %s", coord)
        # interpolate 2D coordinates:
        if variable_cube.coord(coord).ndim == 1:
            if variable_cube.coord_dims(coord) == (hdim_1,):
                f = interp1d(
                    dimvec_1,
                    variable_cube.coord(coord).points,
                    fill_value="extrapolate",
                )
                coordinate_points = f(t["hdim_1"])

            if variable_cube.coord_dims(coord) == (hdim_2,):
                f = interp1d(
                    dimvec_2,
                    variable_cube.coord(coord).points,
                    fill_value="extrapolate",
                )
                coordinate_points = f(t["hdim_2"])

        # interpolate 2D coordinates:
        elif variable_cube.coord(coord).ndim == 2:
            if variable_cube.coord_dims(coord) == (hdim_1, hdim_2):
                points = (dimvec_1, dimvec_2)
                values = variable_cube.coord(coord).points
                xi = np.column_stack((t["hdim_1"], t["hdim_2"]))
                coordinate_points = interpn(points, values, xi)

            if variable_cube.coord_dims(coord) == (hdim_2, hdim_1):
                points = (dimvec_2, dimvec_1)
                values = variable_cube.coord(coord).points
                xi = np.column_stack((t["hdim_2"], t["hdim_1"]))
                coordinate_points = interpn(points, values, xi)

        # interpolate 3D coordinates:
        # mainly workaround for wrf latitude and longitude (to be fixed in future)

        elif variable_cube.coord(coord).ndim == 3:
            if variable_cube.coord_dims(coord) == (ndim_time, hdim_1, hdim_2):
                points = (dimvec_1, dimvec_2)
                values = variable_cube[0, :, :].coord(coord).points
                xi = np.column_stack((t["hdim_1"], t["hdim_2"]))
                coordinate_points = interpn(points, values, xi)

            if variable_cube.coord_dims(coord) == (ndim_time, hdim_2, hdim_1):
                points = (dimvec_2, dimvec_1)
                values = variable_cube[0, :, :].coord(coord).points
                xi = np.column_stack((t["hdim_2"], t["hdim_1"]))
                coordinate_points = interpn(points, values, xi)

            if variable_cube.coord_dims(coord) == (hdim_1, ndim_time, hdim_2):
                points = (dimvec_1, dimvec_2)
                values = variable_cube[:, 0, :].coord(coord).points
                xi = np.column_stack((t["hdim_1"], t["hdim_2"]))
                coordinate_points = interpn(points, values, xi)

            if variable_cube.coord_dims(coord) == (hdim_1, hdim_2, ndim_time):
                points = (dimvec_1, dimvec_2)
                values = variable_cube[:, :, 0].coord(coord).points
                xi = np.column_stack((t["hdim_1"], t["hdim_2"]))
                coordinate_points = interpn(points, values, xi)

            if variable_cube.coord_dims(coord) == (hdim_2, ndim_time, hdim_1):
                points = (dimvec_2, dimvec_1)
                values = variable_cube[:, 0, :].coord(coord).points
                xi = np.column_stack((t["hdim_2"], t["hdim_1"]))
                coordinate_points = interpn(points, values, xi)

            if variable_cube.coord_dims(coord) == (hdim_2, hdim_1, ndim_time):
                points = (dimvec_2, dimvec_1)
                values = variable_cube[:, :, 0].coord(coord).points
                xi = np.column_stack((t["hdim_2"], t["hdim_1"]))
                coordinate_points = interpn(points, values, xi)

        # write resulting array or list into DataFrame:
        t[coord] = coordinate_points

        logging.debug("added coord: " + coord)
    return t


def add_coordinates_3D(
    t: pd.DataFrame,
    variable_cube: iris.cube.Cube,
    vertical_coord: Union[str, int] = None,
    vertical_axis: Union[int, None] = None,
    assume_coords_fixed_in_time=True,
):
    """Function adding coordinates from the tracking cube to the trajectories
        for the 3D case: time, longitude&latitude, x&y dimensions, and altitude

    Parameters
    ----------
    t:             pandas DataFrame
                   trajectories/features
    variable_cube: iris.cube.Cube
        Cube (usually the one you are tracking on) at least conaining the dimension of 'time'.
        Typically, 'longitude','latitude','x_projection_coordinate','y_projection_coordinate',
        and 'altitude' (if 3D) are the coordinates that we expect, although this function
        will happily interpolate along any dimension coordinates you give.
    vertical_coord: str or int
        Name or axis number of the vertical coordinate. If None, tries to auto-detect.
        If it is a string, it looks for the coordinate or the dimension name corresponding
        to the string. If it is an int, it assumes that it is the vertical axis.
        Note that if you only have a 2D or 3D coordinate for altitude, you must
        pass in an int.
    vertical_axis: int or None
        Axis number of the vertical.
    assume_coords_fixed_in_time: bool
        If true, it assumes that the coordinates are fixed in time, even if the
        coordinates say they vary in time. This is, by default, True, to preserve
        legacy functionality. If False, it assumes that if a coordinate says
        it varies in time, it takes the coordinate at its word.

    Returns
    -------
    pandas DataFrame
                   trajectories with added coordinates
    """
    from scipy.interpolate import interp2d, interp1d, interpn

    logging.debug("start adding coordinates from cube")

    # pull time as datetime object and timestr from input data and add it to DataFrame:
    t["time"] = None
    t["timestr"] = None

    logging.debug("adding time coordinate")

    time_in = variable_cube.coord("time")
    time_in_datetime = time_in.units.num2date(time_in.points)

    t["time"] = time_in_datetime[t["frame"]]
    t["timestr"] = [
        x.strftime("%Y-%m-%d %H:%M:%S") for x in time_in_datetime[t["frame"]]
    ]

    # Get list of all coordinates in input cube except for time (already treated):
    coord_names = [coord.name() for coord in variable_cube.coords()]
    coord_names.remove("time")

    logging.debug("time coordinate added")

    # chose right dimension for horizontal and vertical axes based on time dimension:
    ndim_time = variable_cube.coord_dims("time")[0]

    if type(vertical_coord) is int:
        ndim_vertical = vertical_coord
        vertical_axis = None
    else:
        vertical_axis = tb_utils_gi.find_vertical_coord_name(
            variable_cube, vertical_coord=vertical_coord
        )

    if vertical_axis is not None:
        ndim_vertical = tb_utils_gi.find_axis_from_coord(variable_cube, vertical_axis)
        if ndim_vertical is None:
            raise ValueError("Vertical Coordinate not found")

    # We need to figure out the axis number of hdim_1 and hdim_2.
    ndim_hdim_1, ndim_hdim_2 = tb_utils_gi.find_hdim_axes_3D(
        variable_cube, vertical_axis=ndim_vertical
    )

    if ndim_hdim_1 is None or ndim_hdim_2 is None:
        raise ValueError("Could not find hdim coordinates.")

    # create vectors to use to interpolate from pixels to coordinates
    dimvec_1 = np.arange(variable_cube.shape[ndim_vertical])
    dimvec_2 = np.arange(variable_cube.shape[ndim_hdim_1])
    dimvec_3 = np.arange(variable_cube.shape[ndim_hdim_2])
    dimvec_time = np.arange(variable_cube.shape[ndim_time])

    coord_to_ax = {
        ndim_vertical: (dimvec_1, "vdim"),
        ndim_time: (dimvec_time, "time"),
        ndim_hdim_1: (dimvec_2, "hdim_1"),
        ndim_hdim_2: (dimvec_3, "hdim_2"),
    }

    # loop over coordinates in input data:
    for coord in coord_names:
        logging.debug("adding coord: " + coord)
        # interpolate 1D coordinates:
        var_coord = variable_cube.coord(coord)
        if var_coord.ndim == 1:
            curr_dim = coord_to_ax[variable_cube.coord_dims(coord)[0]]
            f = interp1d(curr_dim[0], var_coord.points, fill_value="extrapolate")
            coordinate_points = f(t[curr_dim[1]])

        # interpolate 2D coordinates
        elif var_coord.ndim == 2:
            first_dim = coord_to_ax[variable_cube.coord_dims(coord)[1]]
            second_dim = coord_to_ax[variable_cube.coord_dims(coord)[0]]
            points = (second_dim[0], first_dim[0])
            values = var_coord.points
            xi = np.column_stack((t[second_dim[1]], t[first_dim[1]]))
            coordinate_points = interpn(points, values, xi)

        # Deal with the special case where the coordinate is 3D but
        # one of the dimensions is time and we assume the coordinates
        # don't vary in time.
        elif (
            var_coord.ndim == 3
            and ndim_time in variable_cube.coord_dims(coord)
            and assume_coords_fixed_in_time
        ):
            time_pos = variable_cube.coord_dims(coord).index(ndim_time)
            hdim1_pos = 0 if time_pos != 0 else 1
            hdim2_pos = 1 if time_pos == 2 else 2
            first_dim = coord_to_ax[variable_cube.coord_dims(coord)[hdim2_pos]]
            second_dim = coord_to_ax[variable_cube.coord_dims(coord)[hdim1_pos]]
            points = (second_dim[0], first_dim[0])
            values = var_coord.points
            xi = np.column_stack((t[second_dim[1]], t[first_dim[1]]))
            coordinate_points = interpn(points, values, xi)

        # interpolate 3D coordinates:
        elif var_coord.ndim == 3:
            first_dim = coord_to_ax[variable_cube.coord_dims(coord)[0]]
            second_dim = coord_to_ax[variable_cube.coord_dims(coord)[1]]
            third_dim = coord_to_ax[variable_cube.coord_dims(coord)[2]]
            coordinate_points = interpn(
                [first_dim[0], second_dim[0], third_dim[0]],
                var_coord.points,
                [
                    [a, b, c]
                    for a, b, c in zip(
                        t[first_dim[1]], t[second_dim[1]], t[third_dim[1]]
                    )
                ],
            )
            # coordinate_points=[f(a,b) for a,b in zip(t[first_dim[1]],t[second_dim[1]])]

        # write resulting array or list into DataFrame:
        t[coord] = coordinate_points

        logging.debug("added coord: " + coord)
    return t

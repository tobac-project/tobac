"""Utilities for handling indexing and distance calculation with periodic boundaries"""

from __future__ import annotations
import functools

import numpy as np

from tobac.utils.decorators import njit_if_available


def adjust_pbc_point(in_dim: int, dim_min: int, dim_max: int) -> int:
    """Function to adjust a point to the other boundary for PBCs

    Parameters
    ----------
    in_dim : int
        Input coordinate to adjust
    dim_min : int
        Minimum point for the dimension
    dim_max : int
        Maximum point for the dimension (inclusive)

    Returns
    -------
    int
        The adjusted point on the opposite boundary

    Raises
    ------
    ValueError
        If in_dim isn't on one of the boundary points
    """
    if in_dim == dim_min:
        return dim_max
    elif in_dim == dim_max:
        return dim_min
    else:
        raise ValueError("In adjust_pbc_point, in_dim isn't on a boundary.")


def get_pbc_coordinates(
    h1_min: int,
    h1_max: int,
    h2_min: int,
    h2_max: int,
    h1_start_coord: int,
    h1_end_coord: int,
    h2_start_coord: int,
    h2_end_coord: int,
    PBC_flag: str = "none",
) -> list[tuple[int, int, int, int]]:
    """Function to get the real (i.e., shifted away from periodic boundaries) coordinate
    boxes of interest given a set of coordinates that may cross periodic boundaries. This computes,
    for example, multiple bounding boxes to encompass the real coordinates when given periodic
    coordinates that loop around to the other boundary.

    For example, if you pass in [as h1_start_coord, h1_end_coord, h2_start_coord, h2_end_coord]
    (-3, 5, 2,6) with PBC_flag of 'both' or 'hdim_1', h1_max of 10, and h1_min of 0
    this function will return: [(0,5,2,6), (7,10,2,6)].

    If you pass in something outside the bounds of the array, this will truncate your
    requested box. For example, if you pass in [as h1_start_coord, h1_end_coord, h2_start_coord, h2_end_coord]
    (-3, 5, 2,6) with PBC_flag of 'none' or 'hdim_2', this function will return:
    [(0,5,2,6)], assuming h1_min is 0.

    Parameters
    ----------
    h1_min: int
        Minimum array value in hdim_1, typically 0.
    h1_max: int
        Maximum array value in hdim_1 (exclusive). h1_max - h1_min should be the size in h1.
    h2_min: int
        Minimum array value in hdim_2, typically 0.
    h2_max: int
        Maximum array value in hdim_2 (exclusive). h2_max - h2_min should be the size in h2.
    h1_start_coord: int
        Start coordinate in hdim_1. Can be < h1_min if dealing with PBCs.
    h1_end_coord: int
        End coordinate in hdim_1. Can be >= h1_max if dealing with PBCs.
    h2_start_coord: int
        Start coordinate in hdim_2. Can be < h2_min if dealing with PBCs.
    h2_end_coord: int
        End coordinate in hdim_2. Can be >= h2_max if dealing with PBCs.
    PBC_flag : str('none', 'hdim_1', 'hdim_2', 'both')
        Sets whether to use periodic boundaries, and if so in which directions.
        'none' means that we do not have periodic boundaries
        'hdim_1' means that we are periodic along hdim1
        'hdim_2' means that we are periodic along hdim2
        'both' means that we are periodic along both horizontal dimensions

    Returns
    -------
    list of tuples
        A list of tuples containing (h1_start, h1_end, h2_start, h2_end) of each of the
        boxes needed to encompass the coordinates.
    """

    if PBC_flag not in ["none", "hdim_1", "hdim_2", "both"]:
        raise ValueError("PBC_flag must be 'none', 'hdim_1', 'hdim_2', or 'both'")

    h1_start_coords = list()
    h1_end_coords = list()
    h2_start_coords = list()
    h2_end_coords = list()

    # In both of these cases, we just need to truncate the hdim_1 points.
    if PBC_flag in ["none", "hdim_2"]:
        h1_start_coords.append(max(h1_min, h1_start_coord))
        h1_end_coords.append(min(h1_max, h1_end_coord))

    # In both of these cases, we only need to truncate the hdim_2 points.
    if PBC_flag in ["none", "hdim_1"]:
        h2_start_coords.append(max(h2_min, h2_start_coord))
        h2_end_coords.append(min(h2_max, h2_end_coord))

    # If the PBC flag is none, we can just return.
    if PBC_flag == "none":
        return [
            (h1_start_coords[0], h1_end_coords[0], h2_start_coords[0], h2_end_coords[0])
        ]

    # We have at least one periodic boundary.

    # hdim_1 boundary is periodic.
    if PBC_flag in ["hdim_1", "both"]:
        if (h1_end_coord - h1_start_coord) >= (h1_max - h1_min):
            # In this case, we have selected the full h1 length of the domain,
            # so we set the start and end coords to just that.
            h1_start_coords.append(h1_min)
            h1_end_coords.append(h1_max)

        # We know we only have either h1_end_coord > h1_max or h1_start_coord < h1_min
        # and not both. If both are true, the previous if statement should trigger.
        elif h1_start_coord < h1_min:
            # First set of h1 start coordinates
            h1_start_coords.append(h1_min)
            h1_end_coords.append(h1_end_coord)
            # Second set of h1 start coordinates
            pts_from_begin = h1_min - h1_start_coord
            h1_start_coords.append(h1_max - pts_from_begin)
            h1_end_coords.append(h1_max)

        elif h1_end_coord > h1_max:
            h1_start_coords.append(h1_start_coord)
            h1_end_coords.append(h1_max)
            pts_from_end = h1_end_coord - h1_max
            h1_start_coords.append(h1_min)
            h1_end_coords.append(h1_min + pts_from_end)

        # We have no PBC-related issues, actually
        else:
            h1_start_coords.append(h1_start_coord)
            h1_end_coords.append(h1_end_coord)

    if PBC_flag in ["hdim_2", "both"]:
        if (h2_end_coord - h2_start_coord) >= (h2_max - h2_min):
            # In this case, we have selected the full h2 length of the domain,
            # so we set the start and end coords to just that.
            h2_start_coords.append(h2_min)
            h2_end_coords.append(h2_max)

        # We know we only have either h1_end_coord > h1_max or h1_start_coord < h1_min
        # and not both. If both are true, the previous if statement should trigger.
        elif h2_start_coord < h2_min:
            # First set of h1 start coordinates
            h2_start_coords.append(h2_min)
            h2_end_coords.append(h2_end_coord)
            # Second set of h1 start coordinates
            pts_from_begin = h2_min - h2_start_coord
            h2_start_coords.append(h2_max - pts_from_begin)
            h2_end_coords.append(h2_max)

        elif h2_end_coord > h2_max:
            h2_start_coords.append(h2_start_coord)
            h2_end_coords.append(h2_max)
            pts_from_end = h2_end_coord - h2_max
            h2_start_coords.append(h2_min)
            h2_end_coords.append(h2_min + pts_from_end)

        # We have no PBC-related issues, actually
        else:
            h2_start_coords.append(h2_start_coord)
            h2_end_coords.append(h2_end_coord)

    out_coords = list()
    for h1_start_coord_single, h1_end_coord_single in zip(
        h1_start_coords, h1_end_coords
    ):
        for h2_start_coord_single, h2_end_coord_single in zip(
            h2_start_coords, h2_end_coords
        ):
            out_coords.append(
                (
                    h1_start_coord_single,
                    h1_end_coord_single,
                    h2_start_coord_single,
                    h2_end_coord_single,
                )
            )
    return out_coords


@njit_if_available
def calc_distance_coords_pbc(
    coords_1: np.ndarray[float], coords_2: np.ndarray[float], max_dims: np.ndarray[int]
) -> float:
    """Function to calculate the distance between 2D cartesian
    coordinate set 1 and coordinate set 2. Note that we assume both
    coordinates are within their min/max already.

    Parameters
    ----------
    coords_1: 2D or 3D array-like
        Set of coordinates passed in from trackpy of either (vdim, hdim_1, hdim_2)
        coordinates or (hdim_1, hdim_2) coordinates.
    coords_2: 2D or 3D array-like
        Similar to coords_1, but for the second pair of coordinates
    max_dims: array-like
        Array of same length as dimensionality of coords. Each item in max_dims
        corresponds to a dimension of coords ([(vdim), hdim_1, hdim_2]) with
        value equal to the size of that dimension if periodic, or 0 if not

    Returns
    -------
    float
        Distance between coords_1 and coords_2 in cartesian space.

    """
    deltas = np.abs(coords_1 - coords_2)
    deltas = np.where(deltas > 0.5 * max_dims, deltas - max_dims, deltas)
    return np.sqrt(np.sum(deltas**2))


def build_distance_function(min_h1, max_h1, min_h2, max_h2, PBC_flag, is_3D):
    """Function to build a partial ```calc_distance_coords_pbc``` function
    suitable for use with trackpy

    Parameters
    ----------
    min_h1: int
        Minimum point in hdim_1
    max_h1: int
        Maximum point in hdim_1
    min_h2: int
        Minimum point in hdim_2
    max_h2: int
        Maximum point in hdim_2
    PBC_flag : str('none', 'hdim_1', 'hdim_2', 'both')
        Sets whether to use periodic boundaries, and if so in which directions.
        'none' means that we do not have periodic boundaries
        'hdim_1' means that we are periodic along hdim1
        'hdim_2' means that we are periodic along hdim2
        'both' means that we are periodic along both horizontal dimensions
    is_3D : bool
        True if coordinates are to be provided in 3D, False if 2D

    Returns
    -------
    function object
        A version of calc_distance_coords_pbc suitable to be called by
        just f(coords_1, coords_2)

    """
    h1_size, h2_size = validate_pbc_dims(min_h1, max_h1, min_h2, max_h2, PBC_flag)

    if is_3D:
        max_dims = np.array([0, h1_size, h2_size])
    else:
        max_dims = np.array([h1_size, h2_size])
    return functools.partial(
        calc_distance_coords_pbc,
        max_dims=max_dims,
    )


def validate_pbc_dims(
    min_h1: int, max_h1: int, min_h2: int, max_h2: int, PBC_flag: str
) -> tuple[int, int]:
    """Validate the input parameters for build_distance_function and return size of each axis

    Parameters
    ----------
    min_h1: int
        Minimum point in hdim_1
    max_h1: int
        Maximum point in hdim_1
    min_h2: int
        Minimum point in hdim_2
    max_h2: int
        Maximum point in hdim_2
    PBC_flag : str('none', 'hdim_1', 'hdim_2', 'both')
        Sets whether to use periodic boundaries, and if so in which directions.
        'none' means that we do not have periodic boundaries
        'hdim_1' means that we are periodic along hdim1
        'hdim_2' means that we are periodic along hdim2
        'both' means that we are periodic along both horizontal dimensions

    Returns
    -------
    tuple[int, int]
        size of domain in hdim1 and hdim2
    """
    if PBC_flag == "none":
        return (0, 0)
    if PBC_flag == "both":
        invalid_dim_limits = invalid_limit_names(
            min_h1=min_h1, max_h1=max_h1, min_h2=min_h2, max_h2=max_h2
        )
        if invalid_dim_limits:
            raise PBCLimitError(invalid_dim_limits, PBC_flag)
        return (max_h1 - min_h1, max_h2 - min_h2)
    if PBC_flag == "hdim_1":
        invalid_dim_limits = invalid_limit_names(min_h1=min_h1, max_h1=max_h1)
        if invalid_dim_limits:
            raise PBCLimitError(invalid_dim_limits, PBC_flag)
        return (max_h1 - min_h1, 0)
    if PBC_flag == "hdim_2":
        invalid_dim_limits = invalid_limit_names(min_h2=min_h2, max_h2=max_h2)
        if invalid_dim_limits:
            raise PBCLimitError(invalid_dim_limits, PBC_flag)
        return (0, max_h2 - min_h2)
    # if PBC_flag not in ('none', 'hdim_1', 'hdim_2', 'both'):
    raise PBCflagError()


def invalid_limit_names(**limits) -> list[str]:
    """Return the names of keywords if their value is None

    Returns
    -------
    list[str]
        List of provided keywords with value None
    """
    return [k for k, v in limits.items() if v is None]


class PBCflagError(ValueError):
    def __init__(self):
        super().__init__(
            "PBC_flag keyword is not valid, must be one of ['none', 'hdim_1', 'hdim_2', 'both']"
        )


class PBCLimitError(ValueError):
    def __init__(self, invalid_limits, PBC_flag):
        self.message = f"Keyword parameters {invalid_limits} must be provided for PBC_flag {PBC_flag}"
        super().__init__(self.message)


def weighted_circmean(
    values: np.ndarray,
    weights: np.ndarray,
    high: float = 2 * np.pi,
    low: float = 0,
    axis: int | None = None,
) -> np.ndarray:
    """
    Calculate the weighted circular mean over a set of values. If all the
    weights are equal, this function is equivalent to scipy.stats.circmean

    Parameters
    ----------
    values: array-like
        Array of values to calculate the mean over
    weights: array-like
        Array of weights corresponding to each value
    high: float, optional
        Upper bound of the range of values. Defaults to 2*pi
    low: float, optional
        Lower bound of the range of values. Defaults to 0
    axis: int | None, optional
        Axis over which to take the average. If None, the average will be taken
        over the entire array. Defaults to None

    Returns
    -------
    rescaled_average: numpy.ndarray
        The weighted, circular mean over the given values

    """
    scaling_factor = (high - low) / (2 * np.pi)
    scaled_values = (np.asarray(values) - low) / scaling_factor
    sin_average = np.average(np.sin(scaled_values), axis=axis, weights=weights)
    cos_average = np.average(np.cos(scaled_values), axis=axis, weights=weights)
    # If the values are evenly spaced throughout the range rounding errors have a big impact. Default to np.pi (half way between low and high) if this is the case
    if np.isclose(sin_average, 0) and np.isclose(cos_average, 0):
        angle_average = np.pi
    else:
        angle_average = np.arctan2(sin_average, cos_average) % (2 * np.pi)
    rescaled_average = (angle_average * scaling_factor) + low
    # Round return value to try and supress rounding errors
    rescaled_average = np.round(rescaled_average, 12)
    if rescaled_average == high:
        rescaled_average = low
    return rescaled_average


def transfm_pbc_point(in_dim, dim_min, dim_max):
    """Function to transform a PBC-feature point for contiguity

    Parameters
    ----------
    in_dim : int
        Input coordinate to adjust
    dim_min : int
        Minimum point for the dimension
    dim_max : int
        Maximum point for the dimension (inclusive)

    Returns
    -------
    int
        The transformed point

    """
    if in_dim < ((dim_min + dim_max) / 2):
        return in_dim + dim_max + 1
    else:
        return in_dim

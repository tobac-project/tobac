"""
Calulcate spatial properties (distances, velocities, areas, volumes) of tracked objects
"""

import logging
from itertools import combinations

import numpy as np
import pandas as pd
import xarray as xr
from iris.analysis.cartography import area_weights

from tobac.utils.bulk_statistics import get_statistics_from_mask
from tobac.utils.internal.basic import find_vertical_axis_from_coord
from tobac.utils import decorators

__all__ = (
    "haversine",
    "calculate_distance",
    "calculate_velocity",
    "calculate_velocity_individual",
    "calculate_areas_2Dlatlon",
    "calculate_area",
)


def haversine(lat1, lon1, lat2, lon2):
    """Computes the Haversine distance in kilometers.

    Calculates the Haversine distance between two points
    (based on implementation CIS https://github.com/cedadev/cis).

    Parameters
    ----------
    lat1, lon1 : array of latitude, longitude
        First point or points as array in degrees.

    lat2, lon2 : array of latitude, longitude
        Second point or points as array in degrees.

    Returns
    -------
    arclen * RADIUS_EARTH : array
        Array of Distance(s) between the two points(-arrays) in
        kilometers.

    """

    RADIUS_EARTH = 6378.0
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    # print(lat1,lat2,lon1,lon2)
    arclen = 2 * np.arcsin(
        np.sqrt(
            (np.sin((lat2 - lat1) / 2)) ** 2
            + np.cos(lat1) * np.cos(lat2) * (np.sin((lon2 - lon1) / 2)) ** 2
        )
    )
    return arclen * RADIUS_EARTH


def calculate_distance(feature_1, feature_2, method_distance=None):
    """Compute the distance between two features. It is based on
    either lat/lon coordinates or x/y coordinates.

    Parameters
    ----------
    feature_1, feature_2 : pandas.DataFrame or pandas.Series
        Dataframes containing multiple features or pandas.Series
        of one feature. Need to contain either projection_x_coordinate
        and projection_y_coordinate or latitude and longitude
        coordinates.

    method_distance : {None, 'xy', 'latlon'}, optional
        Method of distance calculation. 'xy' uses the length of the
        vector between the two features, 'latlon' uses the haversine
        distance. None checks wether the required coordinates are
        present and starts with 'xy'. Default is None.

    Returns
    -------
    distance : float or pandas.Series
        Float with the distance between the two features in meters if
        the input are two pandas.Series containing one feature,
        pandas.Series of the distances if one of the inputs contains
        multiple features.

    """
    if method_distance is None:
        if (
            ("projection_x_coordinate" in feature_1)
            and ("projection_y_coordinate" in feature_1)
            and ("projection_x_coordinate" in feature_2)
            and ("projection_y_coordinate" in feature_2)
        ):
            method_distance = "xy"
        elif (
            ("latitude" in feature_1)
            and ("longitude" in feature_1)
            and ("latitude" in feature_2)
            and ("longitude" in feature_2)
        ):
            method_distance = "latlon"
        else:
            raise ValueError(
                "either latitude/longitude or projection_x_coordinate/projection_y_coordinate have to be present to calculate distances"
            )

    if method_distance == "xy":
        distance = np.sqrt(
            (
                feature_1["projection_x_coordinate"]
                - feature_2["projection_x_coordinate"]
            )
            ** 2
            + (
                feature_1["projection_y_coordinate"]
                - feature_2["projection_y_coordinate"]
            )
            ** 2
        )
    elif method_distance == "latlon":
        distance = 1000 * haversine(
            feature_1["latitude"],
            feature_1["longitude"],
            feature_2["latitude"],
            feature_2["longitude"],
        )
    else:
        raise ValueError("method undefined")
    return distance


def calculate_velocity_individual(feature_old, feature_new, method_distance=None):
    """Calculate the mean velocity of a feature between two timeframes.

    Parameters
    ----------
    feature_old : pandas.Series
        pandas.Series of a feature at a certain timeframe. Needs to
        contain a 'time' column and either projection_x_coordinate
        and projection_y_coordinate or latitude and longitude coordinates.

    feature_new : pandas.Series
        pandas.Series of the same feature at a later timeframe. Needs
        to contain a 'time' column and either projection_x_coordinate
        and projection_y_coordinate or latitude and longitude coordinates.

    method_distance : {None, 'xy', 'latlon'}, optional
        Method of distance calculation, used to calculate the velocity.
        'xy' uses the length of the vector between the two features,
        'latlon' uses the haversine distance. None checks wether the
        required coordinates are present and starts with 'xy'.
        Default is None.

    Returns
    -------
    velocity : float
        Value of the approximate velocity.

    """

    distance = calculate_distance(
        feature_old, feature_new, method_distance=method_distance
    )
    diff_time = (feature_new["time"] - feature_old["time"]).total_seconds()
    velocity = distance / diff_time
    return velocity


def calculate_velocity(track, method_distance=None):
    """Calculate the velocities of a set of linked features.

    Parameters
    ----------
    track : pandas.DataFrame
        Dataframe of linked features, containing the columns 'cell',
         'time' and either 'projection_x_coordinate' and
         'projection_y_coordinate' or 'latitude' and 'longitude'.

    method_distance : {None, 'xy', 'latlon'}, optional
        Method of distance calculation, used to calculate the
        velocity. 'xy' uses the length of the vector between the
        two features, 'latlon' uses the haversine distance. None
        checks wether the required coordinates are present and
        starts with 'xy'. Default is None.

    Returns
    -------
    track  : pandas.DataFrame
        DataFrame from the input, with an additional column 'v',
        contain the value of the velocity for every feature at
        every possible timestep
    """

    for cell_i, track_i in track.groupby("cell"):
        index = track_i.index.values
        for i, index_i in enumerate(index[:-1]):
            velocity = calculate_velocity_individual(
                track_i.loc[index[i]],
                track_i.loc[index[i + 1]],
                method_distance=method_distance,
            )
            track.at[index_i, "v"] = velocity
    return track


def calculate_nearestneighbordistance(features, method_distance=None):
    """Calculate the distance between a feature and the nearest other
    feature in the same timeframe.

    Parameters
    ----------
    features : pandas.DataFrame
        DataFrame of the features whose nearest neighbor distance is to
        be calculated. Needs to contain either projection_x_coordinate
        and projection_y_coordinate or latitude and longitude coordinates.

    method_distance : {None, 'xy', 'latlon'}, optional
        Method of distance calculation. 'xy' uses the length of the vector
        between the two features, 'latlon' uses the haversine distance.
        None checks wether the required coordinates are present and starts
        with 'xy'. Default is None.

    Returns
    -------
    features : pandas.DataFrame
        DataFrame of the features with a new column 'min_distance',
        containing the calculated minimal distance to other features.

    """

    features["min_distance"] = np.nan
    for time_i, features_i in features.groupby("time"):
        logging.debug(str(time_i))
        indeces = combinations(features_i.index.values, 2)
        # Loop over combinations to remove features that are closer together than min_distance and keep larger one (either higher threshold or larger area)
        distances = []
        for index_1, index_2 in indeces:
            if index_1 is not index_2:
                distance = calculate_distance(
                    features_i.loc[index_1],
                    features_i.loc[index_2],
                    method_distance=method_distance,
                )
                distances.append(
                    pd.DataFrame(
                        {"index_1": index_1, "index_2": index_2, "distance": distance},
                        index=[0],
                    )
                )
        if any([x is not None for x in distances]):
            distances = pd.concat(distances, ignore_index=True)
            for i in features_i.index:
                min_distance = distances.loc[
                    (distances["index_1"] == i) | (distances["index_2"] == i),
                    "distance",
                ].min()
                features.at[i, "min_distance"] = min_distance
    return features


def calculate_areas_2Dlatlon(_2Dlat_coord, _2Dlon_coord):
    """Calculate an array of cell areas when given two 2D arrays
    of latitude and longitude values

    NOTE: This currently assuems that the lat/lon grid is orthogonal,
    which is not strictly true! It's close enough for most cases, but
    should be updated in future to use the cross product of the
    distances to the neighbouring cells. This will require the use
    of a more advanced calculation. I would advise using pyproj
    at some point in the future to solve this issue and replace
    haversine distance.

    Parameters
    ----------
    _2Dlat_coord : AuxCoord
        Iris auxilliary coordinate containing a 2d grid of latitudes
        for each point.

    _2Dlon_coord : AuxCoord
        Iris auxilliary coordinate containing a 2d grid of longitudes
        for each point.

    Returns
    -------
    area : ndarray
        A numpy array approximating the area of each cell.

    """

    hdist1 = (
        haversine(
            _2Dlat_coord.points[:-1],
            _2Dlon_coord.points[:-1],
            _2Dlat_coord.points[1:],
            _2Dlon_coord.points[1:],
        )
        * 1000
    )

    dists1 = np.zeros(_2Dlat_coord.points.shape)
    dists1[0] = hdist1[0]
    dists1[-1] = hdist1[-1]
    dists1[1:-1] = (hdist1[0:-1] + hdist1[1:]) * 0.5

    hdist2 = (
        haversine(
            _2Dlat_coord.points[:, :-1],
            _2Dlon_coord.points[:, :-1],
            _2Dlat_coord.points[:, 1:],
            _2Dlon_coord.points[:, 1:],
        )
        * 1000
    )

    dists2 = np.zeros(_2Dlat_coord.points.shape)
    dists2[:, 0] = hdist2[:, 0]
    dists2[:, -1] = hdist2[:, -1]
    dists2[:, 1:-1] = (hdist2[:, 0:-1] + hdist2[:, 1:]) * 0.5

    area = dists1 * dists2

    return area


@decorators.xarray_to_iris
def calculate_area(features, mask, method_area=None, vertical_coord=None):
    """Calculate the area of the segments for each feature.

    Parameters
    ----------
    features : pandas.DataFrame
        DataFrame of the features whose area is to be calculated.

    mask : iris.cube.Cube
        Cube containing mask (int for tracked volumes 0 everywhere
        else). Needs to contain either projection_x_coordinate and
        projection_y_coordinate or latitude and longitude
        coordinates.

    method_area : {None, 'xy', 'latlon'}, optional
        Flag determining how the area is calculated. 'xy' uses the
        areas of the individual pixels, 'latlon' uses the
        area_weights method of iris.analysis.cartography, None
        checks wether the required coordinates are present and
        starts with 'xy'. Default is None.

    vertical_coord: None | str, optional (default: None)
        Name of the vertical coordinate. If None, tries to auto-detect.
        It looks for the coordinate or the dimension name corresponding
        to the string.

    Returns
    -------
    features : pandas.DataFrame
        DataFrame of the features with a new column 'area',
        containing the calculated areas.

    Raises
    ------
    ValueError
        If neither latitude/longitude nor
        projection_x_coordinate/projection_y_coordinate are
        present in mask_coords.

        If latitude/longitude coordinates are 2D.

        If latitude/longitude shapes are not supported.

        If method is undefined, i.e. method is neither None,
        'xy' nor 'latlon'.

    """

    features["area"] = np.nan

    # Get the first time step of mask to remove time dimension of calculated areas
    mask_slice = next(mask.slices_over("time"))
    is_3d = len(mask_slice.core_data().shape) == 3
    if is_3d:
        vertical_coord_name = find_vertical_axis_from_coord(mask_slice, vertical_coord)
        # Need to get var_name as xarray uses this to label dims
        collapse_dim = mask_slice.coords(vertical_coord_name)[0].var_name
    else:
        collapse_dim = None

    mask_coords = [coord.name() for coord in mask_slice.coords()]
    if method_area is None:
        if ("projection_x_coordinate" in mask_coords) and (
            "projection_y_coordinate" in mask_coords
        ):
            method_area = "xy"
        elif ("latitude" in mask_coords) and ("longitude" in mask_coords):
            method_area = "latlon"
        else:
            raise ValueError(
                "either latitude/longitude or projection_x_coordinate/projection_y_coordinate have to be present to calculate distances"
            )
    # logging.debug("calculating area using method " + method_area)
    if method_area == "xy":
        if not (
            mask_slice.coord("projection_x_coordinate").has_bounds()
            and mask_slice.coord("projection_y_coordinate").has_bounds()
        ):
            mask_slice.coord("projection_x_coordinate").guess_bounds()
            mask_slice.coord("projection_y_coordinate").guess_bounds()
        area = np.outer(
            np.diff(mask_slice.coord("projection_y_coordinate").bounds, axis=1),
            np.diff(mask_slice.coord("projection_x_coordinate").bounds, axis=1),
        )
    elif method_area == "latlon":
        if (mask_slice.coord("latitude").ndim == 1) and (
            mask_slice.coord("latitude").ndim == 1
        ):
            if not (
                mask_slice.coord("latitude").has_bounds()
                and mask_slice.coord("longitude").has_bounds()
            ):
                mask_slice.coord("latitude").guess_bounds()
                mask_slice.coord("longitude").guess_bounds()
            area = area_weights(mask_slice, normalize=False)
        elif (
            mask_slice.coord("latitude").ndim == 2
            and mask_slice.coord("longitude").ndim == 2
        ):
            area = calculate_areas_2Dlatlon(
                mask_slice.coord("latitude"), mask_slice.coord("longitude")
            )
        else:
            raise ValueError("latitude/longitude coordinate shape not supported")
    else:
        raise ValueError("method undefined")

    # Area needs to be a dataarray for get_statistics from mask, but otherwise dims/coords don't actually matter
    area = xr.DataArray(area, dims=("a", "b"))

    features = get_statistics_from_mask(
        features,
        mask,
        area,
        statistic={"area": np.sum},
        default=np.nan,
        collapse_dim=collapse_dim,
    )

    return features

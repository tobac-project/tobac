"""
Perform analysis on the properties of detected features
"""

import logging
import numpy as np

from tobac.analysis.spatial import (
    calculate_nearestneighbordistance,
    calculate_area,
)

__all__ = (
    "nearestneighbordistance_histogram",
    "area_histogram",
    "histogram_featurewise",
)


def nearestneighbordistance_histogram(
    features,
    bin_edges=np.arange(0, 30000, 500),
    density=False,
    method_distance=None,
    return_values=False,
):
    """Create an nearest neighbor distance histogram of the features.
    If the DataFrame does not contain a 'min_distance' column, the
    distances are calculated.

    ----------
    features

    bin_edges : int or ndarray, optional
        If bin_edges is an int, it defines the number of equal-width
        bins in the given range. If bins is a ndarray, it defines a
        monotonically increasing array of bin edges, including the
        rightmost edge. Default is np.arange(0, 30000, 500).

    density : bool, optional
        If False, the result will contain the number of samples in
        each bin. If True, the result is the value of the probability
        density function at the bin, normalized such that the integral
        over the range is 1. Default is False.

    method_distance : {None, 'xy', 'latlon'}, optional
        Method of distance calculation. 'xy' uses the length of the
        vector between the two features, 'latlon' uses the haversine
        distance. None checks wether the required coordinates are
        present and starts with 'xy'. Default is None.

    return_values : bool, optional
        Bool determining wether the nearest neighbor distance of the
        features are returned from this function. Default is False.

    Returns
    -------
    hist : ndarray
        The values of the histogram.

    bin_edges : ndarray
        The edges of the histogram.

    distances, optional : ndarray
        A numpy array with the nearest neighbor distances of each
        feature.

    """

    if "min_distance" not in features.columns:
        logging.debug("calculate nearest neighbor distances")
        features = calculate_nearestneighbordistance(
            features, method_distance=method_distance
        )
    distances = features["min_distance"].values
    hist, bin_edges = np.histogram(
        distances[~np.isnan(distances)], bin_edges, density=density
    )
    if return_values:
        return hist, bin_edges, distances
    else:
        return hist, bin_edges


def area_histogram(
    features,
    mask,
    bin_edges=np.arange(0, 30000, 500),
    density=False,
    method_area=None,
    return_values=False,
    representative_area=False,
):
    """Create an area histogram of the features. If the DataFrame
    does not contain an area column, the areas are calculated.

    Parameters
    ----------
    features : pandas.DataFrame
        DataFrame of the features.

    mask : iris.cube.Cube
        Cube containing mask (int for tracked volumes 0
        everywhere else). Needs to contain either
        projection_x_coordinate and projection_y_coordinate or
        latitude and longitude coordinates. The output of a
        segmentation should be used here.

    bin_edges : int or ndarray, optional
        If bin_edges is an int, it defines the number of
        equal-width bins in the given range. If bins is a ndarray,
        it defines a monotonically increasing array of bin edges,
        including the rightmost edge.
        Default is np.arange(0, 30000, 500).

    density : bool, optional
        If False, the result will contain the number of samples
        in each bin. If True, the result is the value of the
        probability density function at the bin, normalized such
        that the integral over the range is 1. Default is False.

    return_values : bool, optional
        Bool determining wether the areas of the features are
        returned from this function. Default is False.

    representive_area: bool, optional
        If False, no weights will associated to the values.
        If True, the weights for each area will be the areas
        itself, i.e. each bin count will have the value of
        the sum of all areas within the edges of the bin.
        Default is False.

    Returns
    -------
    hist : ndarray
        The values of the histogram.

    bin_edges : ndarray
        The edges of the histogram.

    bin_centers : ndarray
        The centers of the histogram intervalls.

    areas : ndarray, optional
        A numpy array approximating the area of each feature.

    """

    if "area" not in features.columns:
        logging.info("calculate area")
        features = calculate_area(features, mask, method_area)
    areas = features["area"].values
    # restrict to non NaN values:
    areas = areas[~np.isnan(areas)]
    if representative_area:
        weights = areas
    else:
        weights = None
    hist, bin_edges = np.histogram(areas, bin_edges, density=density, weights=weights)
    bin_centers = bin_edges[:-1] + 0.5 * np.diff(bin_edges)

    if return_values:
        return hist, bin_edges, bin_centers, areas
    else:
        return hist, bin_edges, bin_centers


def histogram_featurewise(Track, variable=None, bin_edges=None, density=False):
    """Create a histogram of a variable from the features
    (detected objects at a single time step) of a track.
    Essentially a wrapper of the numpy.histogram() method.

    Parameters
    ----------
    Track : pandas.DataFrame
        The track containing the variable to create the
        histogram from.

    variable : string, optional
        Column of the DataFrame with the variable on which the
        histogram is to be based on. Default is None.

    bin_edges : int or ndarray, optional
        If bin_edges is an int, it defines the number of
        equal-width bins in the given range. If bins is
        a sequence, it defines a monotonically increasing
        array of bin edges, including the rightmost edge.

    density : bool, optional
        If False, the result will contain the number of
        samples in each bin. If True, the result is the
        value of the probability density function at the
        bin, normalized such that the integral over the
        range is 1. Default is False.

    Returns
    -------
    hist : ndarray
        The values of the histogram

    bin_edges : ndarray
        The edges of the histogram

    bin_centers : ndarray
        The centers of the histogram intervalls

    """

    hist, bin_edges = np.histogram(Track[variable].values, bin_edges, density=density)
    bin_centers = bin_edges[:-1] + 0.5 * np.diff(bin_edges)

    return hist, bin_edges, bin_centers

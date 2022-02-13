"""Provide feature detection.

This module can work with any two-dimensional field either present or
derived from the input data. To identify the features, contiguous
regions above or below a threshold are determined and labelled
individually. To describe the specific location of the feature at a
specific point in time, different spatial properties are used to
describe the identified region. [2]_

References
----------
.. [2] Heikenfeld, M., Marinescu, P. J., Christensen, M., Watson-Parris,
   D., Senf, F., van den Heever, S. C., and Stier, P.: tobac v1.0:
   towards a flexible framework for tracking and analysis of clouds in
   diverse datasets, Geosci. Model Dev. Discuss.,
   https://doi.org/10.5194/gmd-2019-105 , in review, 2019, 6f.
"""

import logging
import numpy as np
import pandas as pd
from tobac.utils import xarray_to_iris
from tobac.utils.general import spectral_filtering 


def feature_detection_multithreshold(
    field_in,
    dxy,
    threshold=None,
    min_num=0,
    target="maximum",
    position_threshold="center",
    sigma_threshold=0.5,
    n_erosion_threshold=0,
    n_min_threshold=0,
    min_distance=0,
    feature_number_start=1,
    wavelength_filtering=None,
):
    """Perform feature detection based on contiguous regions.

    The regions are above/below a threshold.

    Parameters
    ----------
    field_in : xarray.DataArrray
        2D field to perform the tracking on (needs to have coordinate
        'time' along one of its dimensions),

    dxy : float
        Grid spacing of the input data (in meter).

    thresholds : list of floats, optional
        Threshold values used to select target regions to track.
        Default is None.

    target : {'maximum', 'minimum'}, optional
        Flag to determine if tracking is targetting minima or maxima in
        the data. Default is 'maximum'.

    position_threshold : {'center', 'extreme', 'weighted_diff',
                          'weighted_abs'}, optional
        Flag choosing method used for the position of the tracked
        feature. Default is 'center'.

    sigma_threshold: float, optional
        Standard deviation for intial filtering step. Default is 0.5.

    n_erosion_threshold: int, optional
        Number of pixel by which to erode the identified features.
        Default is 0.

    n_min_threshold : int, optional
        Minimum number of identified features. Default is 0.

    min_distance : float, optional
        Minimum distance between detected features. Default is 0.

    feature_number_start : int, optional
        Feature id to start with. Default is 1.


    wavelength_filtering: tuple, optional
        minimum and maximum wavelengths in km, if spectral filtering of input field is desired
        note that the wavelengths need to be smaller than the distances along x and y


    Returns
    -------
    features : xarray.Dataset
        Detected features.
    """
    from tobac.utils import add_coordinates

    # convert from xarray to Iris:
    field_iris = field_in.to_iris()

    logging.debug("start feature detection based on thresholds")

    # create empty list to store features for all timesteps
    list_features_timesteps = []

    # loop over timesteps for feature identification:
    data_time = field_iris.slices_over("time")

    # if single threshold is put in as a single value, turn it into a list
    if type(threshold) in [int, float]:
        threshold = [threshold]

    for i_time, data_i in enumerate(data_time):
        time_i = data_i.coord("time").units.num2date(data_i.coord("time").points[0])
        features_thresholds = feature_detection_multithreshold_timestep(dxy,
            data_i,
            i_time,
            threshold=threshold,
            sigma_threshold=sigma_threshold,
            min_num=min_num,
            target=target,
            position_threshold=position_threshold,
            n_erosion_threshold=n_erosion_threshold,
            n_min_threshold=n_min_threshold,
            min_distance=min_distance,
            feature_number_start=feature_number_start,
            wavelength_filtering=wavelength_filtering,
        )
        # check if list of features is not empty, then merge features from different threshold values
        # into one DataFrame and append to list for individual timesteps:
        if not features_thresholds.empty:
            # Loop over DataFrame to remove features that are closer than distance_min to each other:
            if min_distance > 0:
                features_thresholds = filter_min_distance(
                    features_thresholds.reset_index(), dxy, min_distance
                )
        list_features_timesteps.append(features_thresholds)

        logging.debug(
            "Finished feature detection for " + time_i.strftime("%Y-%m-%d_%H:%M:%S")
        )

    logging.debug("feature detection: merging DataFrames")
    # Check if features are detected and then concatenate features from different timesteps into one pandas DataFrame
    # If no features are detected raise error
    if any([not x.empty for x in list_features_timesteps]):
        features = pd.concat(list_features_timesteps, ignore_index=True)
        features["feature"] = features.index + feature_number_start
        #    features_filtered = features.drop(features[features['num'] < min_num].index)
        #    features_filtered.drop(columns=['idx','num','threshold_value'],inplace=True)
        features = add_coordinates(features, field_iris)
        # convert pandas DataFrame to xarray
        list_coords = [key for key in field_in.coords.keys()]
        print(features)
        print(list_coords)
        # features=features.set_index(list_coords)
        features = features.to_xarray()
    else:
        features = None
        logging.info("No features detected")
    logging.debug("feature detection completed")
    return features


def feature_detection_multithreshold_timestep(
    dxy =-1, 
    data_i,
    i_time,
    threshold=None,
    min_num=0,
    target="maximum",
    position_threshold="center",
    sigma_threshold=0.5,
    n_erosion_threshold=0,
    n_min_threshold=0,
    min_distance=0,
    feature_number_start=1,
    wavelength_filtering=None
):
    """Find features in each timestep.

    Based on iteratively finding regions above/below a set of
    thresholds. Smoothing the input data with the Gaussian filter makes
    output more reliable. [2]_

    Parameters
    ----------


    dxy : float
        Grid spacing in meter, that has to be specified if  wavelength_filtering is not None.

    data_i : iris.cube.Cube
        2D field to perform the feature detection (single timestep) on.

    threshold : float, optional
        Threshold value used to select target regions to track. Default
        is None.

    min_num : int, optional
        Default is 0.

    target : {'maximum', 'minimum'}, optinal
        Flag to determine if tracking is targetting minima or maxima
        in the data. Default is 'maximum'.

    position_threshold : {'center', 'extreme', 'weighted_diff',
                          'weighted_abs'}, optional
        Flag choosing method used for the position of the tracked
        feature. Default is 'center'.

    sigma_threshold: float, optional
        Standard deviation for intial filtering step. Default is 0.5.

    n_erosion_threshold: int, optional
        Number of pixel by which to erode the identified features.
        Default is 0.

    n_min_threshold : int, optional
        Minimum number of identified features. Default is 0.

    min_distance : float, optional
        Minimum distance between detected features. Default is 0.

    feature_number_start : int, optional
        Feature id to start with. Default is 1.


    wavelength_filtering: tuple, optional
        minimum and maximum wavelengths in km, if spectral filtering of input field is desired
        common fields for spectral filtering are e.g.specific humidity or relative vorticity

    dxy : float
        If wavelength_filtering is not None, grid spacing of the input data needs to be specified in meter.


    Returns
    -------
    features_threshold : pandas.DataFrame
        Detected features for individual timestep.
    """

    from scipy.ndimage.filters import gaussian_filter

    track_data = data_i.core_data()

    track_data = gaussian_filter(
        track_data, sigma=sigma_threshold
    )  # smooth data slightly to create rounded, continuous field

    # spectrally filtering of data, if desired
    # note that grid spacing needs to be provided in km, so that it matches wavelengths
    if wavelength_filtering is not None:
        track_data = spectral_filtering(
            dxy, track_data, wavelength_filtering[0], wavelength_filtering[1]
        )

    # create empty lists to store regions and features for individual timestep
    features_thresholds = pd.DataFrame()
    regions_old = {}
    for i_threshold, threshold_i in enumerate(threshold):
        if i_threshold > 0 and not features_thresholds.empty:
            idx_start = features_thresholds["idx"].max() + 1
        else:
            idx_start = 0
        features_threshold_i, regions_i = feature_detection_threshold(
            track_data,
            i_time,
            threshold=threshold_i,
            sigma_threshold=sigma_threshold,
            min_num=min_num,
            target=target,
            position_threshold=position_threshold,
            n_erosion_threshold=n_erosion_threshold,
            n_min_threshold=n_min_threshold,
            min_distance=min_distance,
            idx_start=idx_start,
        )
        if any([x is not None for x in features_threshold_i]):
            features_thresholds = features_thresholds.append(features_threshold_i)

        # For multiple threshold, and features found both in the current and previous step, remove "parent" features from Dataframe
        if i_threshold > 0 and not features_thresholds.empty and regions_old:
            # for each threshold value: check if newly found features are surrounded by feature based on less restrictive threshold
            features_thresholds = remove_parents(
                features_thresholds, regions_i, regions_old
            )
        regions_old = regions_i

        logging.debug(
            "Finished feature detection for threshold "
            + str(i_threshold)
            + " : "
            + str(threshold_i)
        )
    return features_thresholds


def feature_detection_threshold(
    data_i,
    i_time,
    threshold=None,
    min_num=0,
    target="maximum",
    position_threshold="center",
    sigma_threshold=0.5,
    n_erosion_threshold=0,
    n_min_threshold=0,
    min_distance=0,
    idx_start=0,
):
    """Find features based on individual threshold value.

    Parameters
    ----------
    data_i : iris.cube.Cube
        2D field to perform the feature detection (single timestep) on.

    i_time : int
        Number of the current timestep.

    threshold : float, optional
        Threshold value used to select target regions to track. Default
                is None.

    target : {'maximum', 'minimum'}, optional
        Flag to determine if tracking is targetting minima or maxima
        in the data. Default is 'maximum'.

    position_threshold : {'center', 'extreme', 'weighted_diff',
                          'weighted_abs'}, optional
        Flag choosing method used for the position of the tracked
        feature. Default is 'center'.

    sigma_threshold: float, optional
        Standard deviation for intial filtering step. Default is 0.5.

    n_erosion_threshold: int, optional
        Number of pixel by which to erode the identified features.
        Default is 0.

    n_min_threshold : int, optional
        Minimum number of identified features. Default is 0.

    min_distance : float, optional
        Minimum distance between detected features. Default is 0.

    idx_start : int, optional
        Feature id to start with. Default is 0.

    Returns
    -------
    features_threshold : pandas DataFrame
        Detected features for individual threshold.

    regions : dict
        Dictionary containing the regions above/below threshold used
        for each feature (feature ids as keys).
    """

    from skimage.measure import label
    from skimage.morphology import binary_erosion

    # if looking for minima, set values above threshold to 0 and scale by data minimum:
    if target == "maximum":
        mask = 1 * (data_i >= threshold)
        # if looking for minima, set values above threshold to 0 and scale by data minimum:
    elif target == "minimum":
        mask = 1 * (data_i <= threshold)
    # only include values greater than threshold
    # erode selected regions by n pixels
    if n_erosion_threshold > 0:
        selem = np.ones((n_erosion_threshold, n_erosion_threshold))
        mask = binary_erosion(mask, selem).astype(np.int64)
        # detect individual regions, label  and count the number of pixels included:
    labels = label(mask, background=0)
    values, count = np.unique(labels[:, :].ravel(), return_counts=True)
    values_counts = dict(zip(values, count))
    # Filter out regions that have less pixels than n_min_threshold
    values_counts = {k: v for k, v in values_counts.items() if v > n_min_threshold}
    # check if not entire domain filled as one feature
    if 0 in values_counts:
        # Remove background counts:
        values_counts.pop(0)
        # create empty list to store individual features for this threshold
        list_features_threshold = []
        # create empty dict to store regions for individual features for this threshold
        regions = dict()
        # create emptry list of features to remove from parent threshold value
        # loop over individual regions:
        # Find label regions first using binning as this is more efficient
        bins = np.cumsum(np.bincount(labels.ravel()))
        args = np.argsort(labels.ravel())

        for cur_idx, count in values_counts.items():
            if bins[cur_idx] > bins[cur_idx - 1]:
                # region=labels[:,:] == cur_idx
                # [hdim1_indices,hdim2_indices] = np.nonzero(region)
                hdim1_indices, hdim2_indices = np.unravel_index(
                    args[bins[cur_idx - 1] : bins[cur_idx]], labels.shape
                )
                # write region for individual threshold and feature to dict
                region_i = list(zip(hdim1_indices, hdim2_indices))
                regions[cur_idx + idx_start] = region_i
                # Determine feature position for region by one of the following methods:
                hdim1_index, hdim2_index = feature_position(
                    hdim1_indices,
                    hdim2_indices,
                    data_i,
                    threshold,
                    position_threshold,
                    target,
                )
                # create individual DataFrame row in tracky format for identified feature
                list_features_threshold.append(
                    {
                        "frame": int(i_time),
                        "idx": cur_idx + idx_start,
                        "hdim_1": hdim1_index,
                        "hdim_2": hdim2_index,
                        "num": count,
                        "threshold_value": threshold,
                    }
                )
        features_threshold = pd.DataFrame(list_features_threshold)
    else:
        features_threshold = pd.DataFrame()
        regions = dict()

    return features_threshold, regions


def feature_position(
    hdim1_indices, hdim2_indices, track_data, threshold_i, position_threshold, target
):
    """Determine feature position with regard to the horizontal dimensions in pixels from the identified region above threshold values
    Parameters
    ----------
    hdim1_indices: list
        indices of pixels in region along first horizontal dimension

    hdim2_indices : list
            indices of pixels in region along second horizontal dimension

    track_data : numpy.ndarray
        2D numpy array containing the data.

    threshold_i : float

    position_threshold : {center,extreme,weighted_diff,weighted_abs}
        Method to determine the position from the region

    target : {'maximum', 'minimum'}
        Flag to determine if tracking is targetting minima or maxima in
        the data.


    Returns
    -------
    hdim1_index, hdim2_index : float
        Feature position along 1st and 2nd horizontal dimension in pixels
    """

    if position_threshold == "center":
        # get position as geometrical centre of identified region:
        hdim1_index = np.mean(hdim1_indices)
        hdim2_index = np.mean(hdim2_indices)

    elif position_threshold == "extreme":
        # get position as max/min position inside the identified region:
        if target == "maximum":
            index = np.argmax(track_data[(hdim1_indices, hdim2_indices)])
            hdim1_index = hdim1_indices[index]
            hdim2_index = hdim2_indices[index]

        if target == "minimum":
            index = np.argmin(track_data[(hdim1_indices, hdim2_indices)])
            hdim1_index = hdim1_indices[index]
            hdim2_index = hdim2_indices[index]

    elif position_threshold == "weighted_diff":
        # get position as centre of identified region, weighted by difference from the threshold:
        weights = np.abs(track_data[(hdim1_indices, hdim2_indices)] - threshold_i)
        sum_weights = np.sum(weights)
        if sum_weights == 0:
            weights[:] = 1
            sum_weights = weights.size
        hdim1_index = np.sum(hdim1_indices * weights) / sum_weights
        hdim2_index = np.sum(hdim2_indices * weights) / sum_weights

    elif position_threshold == "weighted_abs":
        # get position as centre of identified region, weighted by absolute values if the field:
        weights = np.abs(track_data[(hdim1_indices, hdim2_indices)])
        sum_weights = np.sum(weights)
        if sum_weights == 0:
            weights[:] = 1
            sum_weights = weights.size
        hdim1_index = np.sum(hdim1_indices * weights) / sum_weights
        hdim2_index = np.sum(hdim2_indices * weights) / sum_weights

    else:
        raise ValueError(
            "position_threshold must be center,extreme,weighted_diff or weighted_abs"
        )
    return hdim1_index, hdim2_index


def remove_parents(features_thresholds, regions_i, regions_old):
    """Remove parents of newly detected feature regions.

    Remove features where its regions surround newly detected feature
    regions.

    Parameters
    ----------
    features_thresholds : pandas.DataFrame
        Dataframe containing detected features.

    regions_i : dict
        Dictionary containing the regions above/below threshold for the
        newly detected feature (feature ids as keys).

    regions_old : dict
        Dictionary containing the regions above/below threshold from
        previous threshold (feature ids as keys).

    Returns
    -------
    features_thresholds : pandas.DataFrame
        Dataframe containing detected features excluding those that are
        superseded by newly detected ones.
    """

    # create a set out of all the tuples in newly detected feature
    sets_i = set().union(*regions_i.values())
    # save the overlapped indexes to one list
    list_remove = [
        idx_old
        for idx_old, sets_old in regions_old.items()
        if not sets_i.isdisjoint(sets_old)
    ]

    # remove parent regions:
    if features_thresholds is not None:
        features_thresholds = features_thresholds[
            ~features_thresholds["idx"].isin(list_remove)
        ]

    return features_thresholds


def filter_min_distance(features, dxy, min_distance):
    """Perform feature detection based on contiguous regions.

    Regions are above/below a threshold.

    Parameters
    ----------
    features : pandas.DataFrame

    dxy : float
        Grid spacing of the input data.

    min_distance : float, optional
        Minimum distance between detected features.

    Returns
    -------
    features : pandas.DataFrame
        Detected features.
    """

    from itertools import combinations

    remove_list_distance = []
    # create list of tuples with all combinations of features at the timestep:
    indices = combinations(features.index.values, 2)
    # Loop over combinations to remove features that are closer together than min_distance and keep larger one (either higher threshold or larger area)
    for index_1, index_2 in indices:
        if index_1 is not index_2:
            features.loc[index_1, "hdim_1"]
            distance = dxy * np.sqrt(
                (features.loc[index_1, "hdim_1"] - features.loc[index_2, "hdim_1"]) ** 2
                + (features.loc[index_1, "hdim_2"] - features.loc[index_2, "hdim_2"])
                ** 2
            )
            if distance <= min_distance:
                #                        logging.debug('distance<= min_distance: ' + str(distance))
                if (
                    features.loc[index_1, "threshold_value"]
                    > features.loc[index_2, "threshold_value"]
                ):
                    remove_list_distance.append(index_2)
                elif (
                    features.loc[index_1, "threshold_value"]
                    < features.loc[index_2, "threshold_value"]
                ):
                    remove_list_distance.append(index_1)
                elif (
                    features.loc[index_1, "threshold_value"]
                    == features.loc[index_2, "threshold_value"]
                ):
                    if features.loc[index_1, "num"] > features.loc[index_2, "num"]:
                        remove_list_distance.append(index_2)
                    elif features.loc[index_1, "num"] < features.loc[index_2, "num"]:
                        remove_list_distance.append(index_1)
                    elif features.loc[index_1, "num"] == features.loc[index_2, "num"]:
                        remove_list_distance.append(index_2)
    features = features[~features.index.isin(remove_list_distance)]
    return features


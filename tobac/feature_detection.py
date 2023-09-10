"""Provide feature detection.

This module can work with any two-dimensional field.
To identify the features, contiguous regions above or 
below a threshold are determined and labelled individually.
To describe the specific location of the feature at a 
specific point in time, different spatial properties 
are used to describe the identified region. [2]_

References
----------
.. Heikenfeld, M., Marinescu, P. J., Christensen, M.,
   Watson-Parris, D., Senf, F., van den Heever, S. C.
   & Stier, P. (2019). tobac 1.2: towards a flexible 
   framework for tracking and analysis of clouds in 
   diverse datasets. Geoscientific Model Development,
   12(11), 4551-4570.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree
from .tracking import build_distance_function
from .utils import internal as internal_utils
from .utils import periodic_boundaries as pbc_utils
from .utils.general import spectral_filtering
import warnings
from typing import Union
from typing_extensions import Literal
import iris
import xarray as xr


def feature_position(
    hdim1_indices: list[int],
    hdim2_indices: list[int],
    vdim_indices: Union[list[int], None] = None,
    region_small: np.ndarray = None,
    region_bbox: Union[list[int], tuple[int]] = None,
    track_data: np.ndarray = None,
    threshold_i: float = None,
    position_threshold: Literal[
        "center", "extreme", "weighted_diff", "weighted abs"
    ] = "center",
    target: Literal["maximum", "minimum"] = None,
    PBC_flag: Literal["none", "hdim_1", "hdim_2", "both"] = "none",
    hdim1_min: int = 0,
    hdim1_max: int = 0,
    hdim2_min: int = 0,
    hdim2_max: int = 0,
) -> tuple[float]:
    """Determine feature position with regard to the horizontal
    dimensions in pixels from the identified region above
    threshold values

    Parameters
    ----------
    hdim1_indices : list
        indices of pixels in region along first horizontal
        dimension

    hdim2_indices : list
        indices of pixels in region along second horizontal
        dimension

    vdim_indices : list, optional
        List of indices of feature along optional vdim (typically ```z```)

    region_small : 2D or 3D array-like
        A true/false array containing True where the threshold
        is met and false where the threshold isn't met. This
        array should be the the size specified by region_bbox,
        and can be a subset of the overall input array
        (i.e., ```track_data```).

    region_bbox : list or tuple with length of 4 or 6
        The coordinates that region_small occupies within the total track_data
        array. This is in the order that the coordinates come from the
        ```get_label_props_in_dict``` function. For 2D data, this should be:
        (hdim1 start, hdim 2 start, hdim 1 end, hdim 2 end). For 3D data, this
        is: (vdim start, hdim1 start, hdim 2 start, vdim end, hdim 1 end, hdim 2 end).

    track_data : 2D or 3D array-like
        2D or 3D array containing the data

    threshold_i : float
        The threshold value that we are testing against

    position_threshold : {'center', 'extreme', 'weighted_diff', '
                          weighted abs'}
        How to select the single point position from our data.
        'center' picks the geometrical centre of the region,
        and is typically not recommended. 'extreme' picks the
        maximum or minimum value inside the region (max/min set by
         ```target```) 'weighted_diff' picks the centre of the
         region weighted by the distance from the threshold value
        'weighted_abs' picks the centre of the region weighted by
        the absolute values of the field

    target : {'maximum', 'minimum'}
        Used only when position_threshold is set to 'extreme',
        this sets whether it is looking for maxima or minima.

    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}
        Sets whether to use periodic boundaries, and if so in which directions.
        'none' means that we do not have periodic boundaries
        'hdim_1' means that we are periodic along hdim1
        'hdim_2' means that we are periodic along hdim2
        'both' means that we are periodic along both horizontal dimensions

    hdim1_min : int
        Minimum real array index of the first horizontal dimension (for PBCs)

    hdim1_max: int
        Maximum real array index of the first horizontal dimension (for PBCs)
        Note that this coordinate is INCLUSIVE, meaning that this is
        the maximum coordinate value, and it is not a length.

    hdim2_min : int
        Minimum real array index of the first horizontal dimension (for PBCs)

    hdim2_max : int
        Maximum real array index of the first horizontal dimension (for PBCs)
        Note that this coordinate is INCLUSIVE, meaning that this is
        the maximum coordinate value, and it is not a length.

    Returns
    -------
    2-element or 3-element tuple of floats
        If input data is 2D, this will be a 2-element tuple of floats,
        where the first element is the feature position along the first
        horizontal dimension and the second element is the feature position
        along the second horizontal dimension.
        If input data is 3D, this will be a 3-element tuple of floats, where
        the first element is the feature position along the vertical dimension
        and the second two elements are the feature position on the first and
        second horizontal dimensions.
        Note for PBCs: this point *can* be >hdim1_max or hdim2_max if the
        point is between hdim1_max and hdim1_min. For example, if a feature
        lies exactly between hdim1_max and hdim1_min, the output could be
        between hdim1_max and hdim1_max+1. While a value between hdim1_min-1
        and hdim1_min would also be valid, we choose to overflow on the max side of things.
    """

    # First, if necessary, run PBC processing.
    # processing of PBC indices
    # checks to see if minimum and maximum values are present in dimensional array
    # then if true, adds max value to any indices past the halfway point of their
    # respective dimension. this, in essence, shifts the set of points to the high side.
    pbc_options = ["hdim_1", "hdim_2", "both"]

    if len(region_bbox) == 4:
        # 2D case
        is_3D = False
        track_data_region = track_data[
            region_bbox[0] : region_bbox[2], region_bbox[1] : region_bbox[3]
        ]
    elif len(region_bbox) == 6:
        # 3D case
        is_3D = True
        track_data_region = track_data[
            region_bbox[0] : region_bbox[3],
            region_bbox[1] : region_bbox[4],
            region_bbox[2] : region_bbox[5],
        ]
    else:
        raise ValueError("region_bbox must have 4 or 6 elements.")
    # whether or not to run the means at the end
    run_mean = False
    if position_threshold == "center":
        # get position as geometrical centre of identified region:

        hdim1_weights = np.ones(np.size(hdim1_indices))
        hdim2_weights = np.ones(np.size(hdim2_indices))
        if is_3D:
            vdim_weights = np.ones(np.size(hdim2_indices))

        run_mean = True

    elif position_threshold == "extreme":
        # get position as max/min position inside the identified region:
        if target == "maximum":
            index = np.argmax(track_data_region[region_small])
        if target == "minimum":
            index = np.argmin(track_data_region[region_small])
        hdim1_index = hdim1_indices[index]
        hdim2_index = hdim2_indices[index]
        if is_3D:
            vdim_index = vdim_indices[index]

    elif position_threshold == "weighted_diff":
        # get position as centre of identified region, weighted by difference from the threshold:
        weights = abs(track_data_region[region_small] - threshold_i)
        if sum(weights) == 0:
            weights = None
        hdim1_weights = weights
        hdim2_weights = weights
        if is_3D:
            vdim_weights = weights

        run_mean = True

    elif position_threshold == "weighted_abs":
        # get position as centre of identified region, weighted by absolute values if the field:
        weights = abs(track_data_region[region_small])
        if sum(weights) == 0:
            weights = None
        hdim1_weights = weights
        hdim2_weights = weights
        if is_3D:
            vdim_weights = weights
        run_mean = True

    else:
        raise ValueError(
            "position_threshold must be center,extreme,weighted_diff or weighted_abs"
        )

    if run_mean:
        if PBC_flag in ("hdim_1", "both"):
            hdim1_index = pbc_utils.weighted_circmean(
                hdim1_indices, weights=hdim1_weights, high=hdim1_max + 1, low=hdim1_min
            )
        else:
            hdim1_index = np.average(hdim1_indices, weights=hdim1_weights)
        if PBC_flag in ("hdim_2", "both"):
            hdim2_index = pbc_utils.weighted_circmean(
                hdim2_indices, weights=hdim2_weights, high=hdim2_max + 1, low=hdim2_min
            )
        else:
            hdim2_index = np.average(hdim2_indices, weights=hdim2_weights)
        if is_3D:
            vdim_index = np.average(vdim_indices, weights=vdim_weights)

    if is_3D:
        return vdim_index, hdim1_index, hdim2_index
    else:
        return hdim1_index, hdim2_index


def test_overlap(
    region_inner: list[tuple[int]], region_outer: list[tuple[int]]
) -> bool:
    """Test for overlap between two regions

    Parameters
    ----------
    region_1 : list
        list of 2-element tuples defining the indices of
        all cell in the region

    region_2 : list
        list of 2-element tuples defining the indices of
        all cell in the region

    Returns
    ----------
    overlap : bool
        True if there are any shared points between the two
        regions
    """

    overlap = frozenset(region_outer).isdisjoint(region_inner)
    return not overlap


def remove_parents(
    features_thresholds: pd.DataFrame, regions_i: dict, regions_old: dict
) -> pd.DataFrame:
    """Remove parents of newly detected feature regions.

    Remove features where its regions surround newly
    detected feature regions.

    Parameters
    ----------
    features_thresholds : pandas.DataFrame
        Dataframe containing detected features.

    regions_i : dict
        Dictionary containing the regions above/below
        threshold for the newly detected feature
        (feature ids as keys).

    regions_old : dict
        Dictionary containing the regions above/below
        threshold from previous threshold
        (feature ids as keys).

    Returns
    -------
    features_thresholds : pandas.DataFrame
        Dataframe containing detected features excluding those
        that are superseded by newly detected ones.
    """

    try:
        all_curr_pts = np.concatenate([vals for idx, vals in regions_i.items()])
        all_old_pts = np.concatenate([vals for idx, vals in regions_old.items()])
    except ValueError:
        # the case where there are no regions
        return features_thresholds
    old_feat_arr = np.empty((len(all_old_pts)))
    curr_loc = 0
    for idx_old in regions_old:
        old_feat_arr[curr_loc : curr_loc + len(regions_old[idx_old])] = idx_old
        curr_loc += len(regions_old[idx_old])

    _, _, common_ix_old = np.intersect1d(all_curr_pts, all_old_pts, return_indices=True)
    list_remove = np.unique(old_feat_arr[common_ix_old])

    # remove parent regions:
    if features_thresholds is not None:
        features_thresholds = features_thresholds[
            ~features_thresholds["idx"].isin(list_remove)
        ]

    return features_thresholds


def feature_detection_threshold(
    data_i: iris.cube.Cube,
    i_time: int,
    threshold: float = None,
    min_num: int = 0,
    target: Literal["maximum", "minimum"] = "maximum",
    position_threshold: Literal[
        "center", "extreme", "weighted_diff", "weighted_abs"
    ] = "center",
    sigma_threshold: float = 0.5,
    n_erosion_threshold: int = 0,
    n_min_threshold: int = 0,
    min_distance: float = 0,
    idx_start: int = 0,
    PBC_flag: Literal["none", "hdim_1", "hdim_2", "both"] = "none",
    vertical_axis: int = 0,
) -> tuple[pd.DataFrame, dict]:
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
        Minimum distance between detected features (in meter). Default is 0.

    idx_start : int, optional
        Feature id to start with. Default is 0.

    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}
         Sets whether to use periodic boundaries, and if so in which directions.
         'none' means that we do not have periodic boundaries
         'hdim_1' means that we are periodic along hdim1
         'hdim_2' means that we are periodic along hdim2
         'both' means that we are periodic along both horizontal dimensions
    vertical_axis: int
        The vertical axis number of the data.


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
    from copy import deepcopy

    if min_num != 0:
        warnings.warn(
            "min_num parameter has no effect and will be deprecated in a future version of tobac. "
            "Please use n_min_threshold instead",
            FutureWarning,
        )

    # If we are given a 3D data array, we should do 3D feature detection.
    is_3D = len(data_i.shape) == 3

    # We need to transpose the input data
    if is_3D:
        if vertical_axis == 1:
            data_i = np.transpose(data_i, axes=(1, 0, 2))
        elif vertical_axis == 2:
            data_i = np.transpose(data_i, axes=(2, 0, 1))

    # if looking for minima, set values above threshold to 0 and scale by data minimum:
    if target == "maximum":
        mask = data_i >= threshold
        # if looking for minima, set values above threshold to 0 and scale by data minimum:
    elif target == "minimum":
        mask = data_i <= threshold
    # only include values greater than threshold
    # erode selected regions by n pixels
    if n_erosion_threshold > 0:
        if is_3D:
            selem = np.ones(
                (n_erosion_threshold, n_erosion_threshold, n_erosion_threshold)
            )
        else:
            selem = np.ones((n_erosion_threshold, n_erosion_threshold))
        mask = binary_erosion(mask, selem)
        # detect individual regions, label  and count the number of pixels included:
    labels, num_labels = label(mask, background=0, return_num=True)
    if not is_3D:
        # let's transpose labels to a 1,y,x array to make calculations etc easier.
        labels = labels[np.newaxis, :, :]
    # these are [min, max], meaning that the max value is inclusive and a valid
    # value.
    z_min = 0
    z_max = labels.shape[0] - 1
    y_min = 0
    y_max = labels.shape[1] - 1
    x_min = 0
    x_max = labels.shape[2] - 1

    # deal with PBCs
    # all options that involve dealing with periodic boundaries
    pbc_options = ["hdim_1", "hdim_2", "both"]
    if PBC_flag not in pbc_options and PBC_flag != "none":
        raise ValueError(
            "Options for periodic are currently: none, " + ", ".join(pbc_options)
        )

    # we need to deal with PBCs in some way.
    if PBC_flag in pbc_options and num_labels > 0:
        #
        # create our copy of `labels` to edit
        labels_2 = deepcopy(labels)
        # points we've already edited
        skip_list = np.array([])
        # labels that touch the PBC walls
        wall_labels = np.array([], dtype=np.int32)

        all_label_props = internal_utils.get_label_props_in_dict(labels)
        [
            all_labels_max_size,
            all_label_locs_v,
            all_label_locs_h1,
            all_label_locs_h2,
        ] = internal_utils.get_indices_of_labels_from_reg_prop_dict(all_label_props)

        # find the points along the boundaries

        # along hdim_1 or both horizontal boundaries
        if PBC_flag == "hdim_1" or PBC_flag == "both":
            # north and south wall
            ns_wall = np.unique(labels[:, (y_min, y_max), :])
            wall_labels = np.append(wall_labels, ns_wall)

        # along hdim_2 or both horizontal boundaries
        if PBC_flag == "hdim_2" or PBC_flag == "both":
            # east/west wall
            ew_wall = np.unique(labels[:, :, (x_min, x_max)])
            wall_labels = np.append(wall_labels, ew_wall)

        wall_labels = np.unique(wall_labels)

        for label_ind in wall_labels:
            new_label_ind = label_ind
            # 0 isn't a real index
            if label_ind == 0:
                continue
            # skip this label if we have already dealt with it.
            if np.any(label_ind == skip_list):
                continue

            # create list for skip labels for this wall label only
            skip_list_thisind = list()

            # get all locations of this label.
            # TODO: harmonize x/y/z vs hdim1/hdim2/vdim.
            label_locs_v = all_label_locs_v[label_ind]
            label_locs_h1 = all_label_locs_h1[label_ind]
            label_locs_h2 = all_label_locs_h2[label_ind]

            # loop through every point in the label
            for label_z, label_y, label_x in zip(
                label_locs_v, label_locs_h1, label_locs_h2
            ):
                # check if this is the special case of being a corner point.
                # if it's doubly periodic AND on both x and y boundaries, it's a corner point
                # and we have to look at the other corner.
                # here, we will only look at the corner point and let the below deal with x/y only.
                if PBC_flag == "both" and (
                    np.any(label_y == [y_min, y_max])
                    and np.any(label_x == [x_min, x_max])
                ):
                    # adjust x and y points to the other side
                    y_val_alt = pbc_utils.adjust_pbc_point(label_y, y_min, y_max)
                    x_val_alt = pbc_utils.adjust_pbc_point(label_x, x_min, x_max)

                    label_on_corner = labels[label_z, y_val_alt, x_val_alt]

                    if (label_on_corner != 0) and (
                        ~np.any(label_on_corner == skip_list)
                    ):
                        # alt_inds = np.where(labels==alt_label_3)
                        # get a list of indices where the label on the corner is so we can switch
                        # them in the new list.

                        labels_2[
                            all_label_locs_v[label_on_corner],
                            all_label_locs_h1[label_on_corner],
                            all_label_locs_h2[label_on_corner],
                        ] = label_ind
                        skip_list = np.append(skip_list, label_on_corner)
                        skip_list_thisind = np.append(
                            skip_list_thisind, label_on_corner
                        )

                    # if it's labeled and has already been dealt with for this label
                    elif (
                        (label_on_corner != 0)
                        and (np.any(label_on_corner == skip_list))
                        and (np.any(label_on_corner == skip_list_thisind))
                    ):
                        # print("skip_list_thisind label - has already been treated this index")
                        continue

                    # if it's labeled and has already been dealt with via a previous label
                    elif (
                        (label_on_corner != 0)
                        and (np.any(label_on_corner == skip_list))
                        and (~np.any(label_on_corner == skip_list_thisind))
                    ):
                        # find the updated label, and overwrite all of label_ind indices with
                        # updated label
                        labels_2_alt = labels_2[label_z, y_val_alt, x_val_alt]
                        labels_2[
                            label_locs_v, label_locs_h1, label_locs_h2
                        ] = labels_2_alt
                        skip_list = np.append(skip_list, label_ind)
                        break

                # on the hdim1 boundary and periodic on hdim1
                if (PBC_flag == "hdim_1" or PBC_flag == "both") and np.any(
                    label_y == [y_min, y_max]
                ):
                    y_val_alt = pbc_utils.adjust_pbc_point(label_y, y_min, y_max)

                    # get the label value on the opposite side
                    label_alt = labels[label_z, y_val_alt, label_x]

                    # if it's labeled and not already been dealt with
                    if (label_alt != 0) and (~np.any(label_alt == skip_list)):
                        # find the indices where it has the label value on opposite side and change
                        # their value to original side
                        # print(all_label_locs_v[label_alt], alt_inds[0])
                        labels_2[
                            all_label_locs_v[label_alt],
                            all_label_locs_h1[label_alt],
                            all_label_locs_h2[label_alt],
                        ] = new_label_ind
                        # we have already dealt with this label.
                        skip_list = np.append(skip_list, label_alt)
                        skip_list_thisind = np.append(skip_list_thisind, label_alt)

                    # if it's labeled and has already been dealt with for this label
                    elif (
                        (label_alt != 0)
                        and (np.any(label_alt == skip_list))
                        and (np.any(label_alt == skip_list_thisind))
                    ):
                        continue

                    # if it's labeled and has already been dealt with
                    elif (
                        (label_alt != 0)
                        and (np.any(label_alt == skip_list))
                        and (~np.any(label_alt == skip_list_thisind))
                    ):
                        # find the updated label, and overwrite all of label_ind indices with
                        # updated label
                        labels_2_alt = labels_2[label_z, y_val_alt, label_x]
                        labels_2[
                            label_locs_v, label_locs_h1, label_locs_h2
                        ] = labels_2_alt
                        new_label_ind = labels_2_alt
                        skip_list = np.append(skip_list, label_ind)

                if (PBC_flag == "hdim_2" or PBC_flag == "both") and np.any(
                    label_x == [x_min, x_max]
                ):
                    x_val_alt = pbc_utils.adjust_pbc_point(label_x, x_min, x_max)

                    # get the label value on the opposite side
                    label_alt = labels[label_z, label_y, x_val_alt]

                    # if it's labeled and not already been dealt with
                    if (label_alt != 0) and (~np.any(label_alt == skip_list)):
                        # find the indices where it has the label value on opposite side and change
                        # their value to original side
                        labels_2[
                            all_label_locs_v[label_alt],
                            all_label_locs_h1[label_alt],
                            all_label_locs_h2[label_alt],
                        ] = new_label_ind
                        # we have already dealt with this label.
                        skip_list = np.append(skip_list, label_alt)
                        skip_list_thisind = np.append(skip_list_thisind, label_alt)

                    # if it's labeled and has already been dealt with for this label
                    elif (
                        (label_alt != 0)
                        and (np.any(label_alt == skip_list))
                        and (np.any(label_alt == skip_list_thisind))
                    ):
                        continue

                    # if it's labeled and has already been dealt with
                    elif (
                        (label_alt != 0)
                        and (np.any(label_alt == skip_list))
                        and (~np.any(label_alt == skip_list_thisind))
                    ):
                        # find the updated label, and overwrite all of label_ind indices with
                        # updated label
                        labels_2_alt = labels_2[label_z, label_y, x_val_alt]
                        labels_2[
                            label_locs_v, label_locs_h1, label_locs_h2
                        ] = labels_2_alt
                        new_label_ind = labels_2_alt
                        skip_list = np.append(skip_list, label_ind)

        # copy over new labels after we have adjusted everything
        labels = labels_2

    # END PBC treatment
    # we need to get label properties again after we handle PBCs.

    label_props = internal_utils.get_label_props_in_dict(labels)
    if len(label_props) > 0:
        (
            total_indices_all,
            vdim_indices_all,
            hdim1_indices_all,
            hdim2_indices_all,
        ) = internal_utils.get_indices_of_labels_from_reg_prop_dict(label_props)

    # values, count = np.unique(labels[:,:].ravel(), return_counts=True)
    # values_counts=dict(zip(values, count))
    # Filter out regions that have less pixels than n_min_threshold
    # values_counts={k:v for k, v in values_counts.items() if v>n_min_threshold}

    # check if not entire domain filled as one feature
    if num_labels > 0:
        # create empty list to store individual features for this threshold
        list_features_threshold = list()
        # create empty dict to store regions for individual features for this threshold
        regions = dict()
        # create empty list of features to remove from parent threshold value

        region = np.empty(mask.shape, dtype=bool)
        # loop over individual regions:
        for cur_idx in total_indices_all:
            # skip this if there aren't enough points to be considered a real feature
            # as defined above by n_min_threshold
            curr_count = total_indices_all[cur_idx]
            if curr_count <= n_min_threshold:
                continue
            if is_3D:
                vdim_indices = vdim_indices_all[cur_idx]
            else:
                vdim_indices = None
            hdim1_indices = hdim1_indices_all[cur_idx]
            hdim2_indices = hdim2_indices_all[cur_idx]

            label_bbox = label_props[cur_idx].bbox
            (
                bbox_zstart,
                bbox_ystart,
                bbox_xstart,
                bbox_zend,
                bbox_yend,
                bbox_xend,
            ) = label_bbox
            bbox_zsize = bbox_zend - bbox_zstart
            bbox_xsize = bbox_xend - bbox_xstart
            bbox_ysize = bbox_yend - bbox_ystart
            # build small region box
            if is_3D:
                region_small = np.full((bbox_zsize, bbox_ysize, bbox_xsize), False)
                region_small[
                    vdim_indices - bbox_zstart,
                    hdim1_indices - bbox_ystart,
                    hdim2_indices - bbox_xstart,
                ] = True

            else:
                region_small = np.full((bbox_ysize, bbox_xsize), False)
                region_small[
                    hdim1_indices - bbox_ystart, hdim2_indices - bbox_xstart
                ] = True
                # we are 2D and need to remove the dummy 3D coordinate.
                label_bbox = (
                    label_bbox[1],
                    label_bbox[2],
                    label_bbox[4],
                    label_bbox[5],
                )

            # [hdim1_indices,hdim2_indices]= np.nonzero(region)
            # write region for individual threshold and feature to dict

            """
            This block of code creates 1D coordinates from the input 
            2D or 3D coordinates. Dealing with 1D coordinates is substantially
            faster than having to carry around (x, y, z) or (x, y) as 
            separate arrays. This also makes comparisons in remove_parents
            substantially faster. 
            """
            if is_3D:
                region_i = np.ravel_multi_index(
                    (hdim1_indices, hdim2_indices, vdim_indices),
                    (y_max + 1, x_max + 1, z_max + 1),
                )
            else:
                region_i = np.ravel_multi_index(
                    (hdim1_indices, hdim2_indices), (y_max + 1, x_max + 1)
                )

            regions[cur_idx + idx_start] = region_i
            # Determine feature position for region by one of the following methods:
            single_indices = feature_position(
                hdim1_indices,
                hdim2_indices,
                vdim_indices=vdim_indices,
                region_small=region_small,
                region_bbox=label_bbox,
                track_data=data_i,
                threshold_i=threshold,
                position_threshold=position_threshold,
                target=target,
                PBC_flag=PBC_flag,
                hdim2_min=x_min,
                hdim2_max=x_max,
                hdim1_min=y_min,
                hdim1_max=y_max,
            )
            if is_3D:
                vdim_index, hdim1_index, hdim2_index = single_indices
            else:
                hdim1_index, hdim2_index = single_indices
            # create individual DataFrame row in tracky format for identified feature
            appending_dict = {
                "frame": int(i_time),
                "idx": cur_idx + idx_start,
                "hdim_1": hdim1_index,
                "hdim_2": hdim2_index,
                "num": curr_count,
                "threshold_value": threshold,
            }
            column_names = [
                "frame",
                "idx",
                "hdim_1",
                "hdim_2",
                "num",
                "threshold_value",
            ]
            if is_3D:
                appending_dict["vdim"] = vdim_index
                column_names = [
                    "frame",
                    "idx",
                    "vdim",
                    "hdim_1",
                    "hdim_2",
                    "num",
                    "threshold_value",
                ]
            list_features_threshold.append(appending_dict)
        # after looping thru proto-features, check if any exceed num threshold
        # if they do not, provide a blank pandas df and regions dict
        if list_features_threshold == []:
            # print("no features above num value at threshold: ",threshold)
            features_threshold = pd.DataFrame()
            regions = dict()
        # if they do, provide a dataframe with features organized with 2D and 3D metadata
        else:
            # print("at least one feature above num value at threshold: ",threshold)
            # print("column_names, after cur_idx loop: ",column_names)
            features_threshold = pd.DataFrame(
                list_features_threshold, columns=column_names
            )

        # features_threshold=pd.DataFrame(list_features_threshold, columns = column_names)
    else:
        features_threshold = pd.DataFrame()
        regions = dict()

    return features_threshold, regions


def feature_detection_multithreshold_timestep(
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
    PBC_flag="none",
    vertical_axis=None,
    dxy=-1,
    wavelength_filtering=None,
    strict_thresholding=False,
):
    """Find features in each timestep.

    Based on iteratively finding regions above/below a set of
    thresholds. Smoothing the input data with the Gaussian filter makes
    output less sensitive to noisiness of input data.

    Parameters
    ----------

    data_i : iris.cube.Cube
        2D field to perform the feature detection (single timestep) on.

    threshold : float, optional
        Threshold value used to select target regions to track. Default
        is None.

    min_num : int, optional
        This parameter is not used in the function. Default is 0.

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
        Minimum distance between detected features (in meter). Default is 0.

    feature_number_start : int, optional
        Feature id to start with. Default is 1.

    PBC_flag : str('none', 'hdim_1', 'hdim_2', 'both')
        Sets whether to use periodic boundaries, and if so in which directions.
        'none' means that we do not have periodic boundaries
        'hdim_1' means that we are periodic along hdim1
        'hdim_2' means that we are periodic along hdim2
        'both' means that we are periodic along both horizontal dimensions

    vertical_axis: int
        The vertical axis number of the data.
    dxy : float
        Grid spacing in meter.

    wavelength_filtering: tuple, optional
       Minimum and maximum wavelength for spectral filtering in meter. Default is None.

    strict_thresholding: Bool, optional
        If True, a feature can only be detected if all previous thresholds have been met.
        Default is False.

    Returns
    -------
    features_threshold : pandas DataFrame
        Detected features for individual timestep.
    """
    # Handle scipy depreciation gracefully
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        from scipy.ndimage.filters import gaussian_filter

    if min_num != 0:
        warnings.warn(
            "min_num parameter has no effect and will be deprecated in a future version of tobac. "
            "Please use n_min_threshold instead",
            FutureWarning,
        )

    # get actual numpy array and make a copy so as not to change the data in the iris cube
    track_data = data_i.core_data().copy()

    track_data = gaussian_filter(
        track_data, sigma=sigma_threshold
    )  # smooth data slightly to create rounded, continuous field

    # spectrally filter the input data, if desired
    if wavelength_filtering is not None:
        track_data = spectral_filtering(
            dxy, track_data, wavelength_filtering[0], wavelength_filtering[1]
        )

    # sort thresholds from least extreme to most extreme
    threshold_sorted = sorted(threshold, reverse=target == "minimum")

    # check if each threshold has a n_min_threshold (minimum nr. of grid cells associated with
    # thresholds), if multiple n_min_threshold are given
    if isinstance(n_min_threshold, list) or isinstance(n_min_threshold, dict):
        if len(n_min_threshold) is not len(threshold):
            raise ValueError(
                "Number of elements in n_min_threshold needs to be the same as thresholds, if "
                "n_min_threshold is given as dict or list."
            )

        # check if thresholds in dict correspond to given thresholds
        if isinstance(n_min_threshold, dict):
            if threshold_sorted != sorted(
                n_min_threshold.keys(), reverse=(target == "minimum")
            ):
                raise ValueError(
                    "Ambiguous input for threshold values. If n_min_threshold is given as a dict,"
                    " the keys not to correspond to the values in threshold."
                )
            # sort dictionary by keys (threshold values) so that they match sorted thresholds and
            # get values for n_min_threshold
            n_min_threshold = [
                n_min_threshold[threshold] for threshold in threshold_sorted
            ]

        elif isinstance(n_min_threshold, list):
            # if n_min_threshold is a list, sort it such that it still matches with the sorted
            # threshold values
            n_min_threshold = [
                x
                for _, x in sorted(
                    zip(threshold, n_min_threshold), reverse=(target == "minimum")
                )
            ]
    elif (
        not isinstance(n_min_threshold, list)
        and not isinstance(n_min_threshold, dict)
        and not isinstance(n_min_threshold, int)
    ):
        raise ValueError(
            "N_min_threshold must be an integer. If multiple values for n_min_threshold are given,"
            " please provide a dictionary or list."
        )

    # create empty lists to store regions and features for individual timestep
    features_thresholds = pd.DataFrame()
    for i_threshold, threshold_i in enumerate(threshold_sorted):
        if i_threshold > 0 and not features_thresholds.empty:
            idx_start = features_thresholds["idx"].max() + feature_number_start
        else:
            idx_start = feature_number_start - 1

        # select n_min_threshold for respective threshold, if multiple values are given
        if isinstance(n_min_threshold, list):
            n_min_threshold_i = n_min_threshold[i_threshold]
        else:
            n_min_threshold_i = n_min_threshold

        features_threshold_i, regions_i = feature_detection_threshold(
            track_data,
            i_time,
            threshold=threshold_i,
            sigma_threshold=sigma_threshold,
            min_num=min_num,
            target=target,
            position_threshold=position_threshold,
            n_erosion_threshold=n_erosion_threshold,
            n_min_threshold=n_min_threshold_i,
            min_distance=min_distance,
            idx_start=idx_start,
            PBC_flag=PBC_flag,
            vertical_axis=vertical_axis,
        )
        if any([x is not None for x in features_threshold_i]):
            features_thresholds = pd.concat(
                [features_thresholds, features_threshold_i], ignore_index=True
            )

        # For multiple threshold, and features found both in the current and previous step, remove
        # "parent" features from Dataframe
        if i_threshold > 0 and not features_thresholds.empty and regions_old:
            # for each threshold value: check if newly found features are surrounded by feature
            # based on less restrictive threshold
            features_thresholds = remove_parents(
                features_thresholds, regions_i, regions_old
            )

        if strict_thresholding:
            if regions_i:
                # remove data in regions where no features were detected
                valid_regions: np.ndarray = np.zeros_like(track_data)
                region_indices: list[int] = list(regions_i.values())[
                    0
                ]  # linear indices
                valid_regions.ravel()[region_indices] = 1
                track_data: np.ndarray = np.multiply(valid_regions, track_data)
            else:
                # since regions_i is empty no further features can be detected
                logging.debug(
                    "Finished feature detection for threshold "
                    + str(i_threshold)
                    + " : "
                    + str(threshold_i)
                )
                return features_thresholds

        if i_threshold > 0 and not features_thresholds.empty and regions_old:
            # Work out which regions are still in feature_thresholds to keep
            # This is faster than calling "in" for every idx
            keep_old_keys = np.isin(
                list(regions_old.keys()), features_thresholds["idx"]
            )
            regions_old = {
                k: v for i, (k, v) in enumerate(regions_old.items()) if keep_old_keys[i]
            }
            regions_old.update(regions_i)
        else:
            regions_old = regions_i

        logging.debug(
            "Finished feature detection for threshold "
            + str(i_threshold)
            + " : "
            + str(threshold_i)
        )
    return features_thresholds


def feature_detection_multithreshold(
    field_in,
    dxy=None,
    threshold=None,
    min_num=0,
    target="maximum",
    position_threshold="center",
    sigma_threshold=0.5,
    n_erosion_threshold=0,
    n_min_threshold=0,
    min_distance=0,
    feature_number_start=1,
    PBC_flag="none",
    vertical_coord=None,
    vertical_axis=None,
    detect_subset=None,
    wavelength_filtering=None,
    dz=None,
    strict_thresholding=False,
):
    """Perform feature detection based on contiguous regions.

    The regions are above/below a threshold.

    Parameters
    ----------
    field_in : iris.cube.Cube
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

    coord_interp_kind : str, optional
        The kind of interpolation for coordinates. Default is 'linear'.
        For 1d interp, {'linear', 'nearest', 'nearest-up', 'zero',
                        'slinear', 'quadratic', 'cubic',
                        'previous', 'next'}.
        For 2d interp, {'linear', 'cubic', 'quintic'}.

    sigma_threshold: float, optional
        Standard deviation for intial filtering step. Default is 0.5.

    n_erosion_threshold: int, optional
        Number of pixel by which to erode the identified features.
        Default is 0.

    n_min_threshold : int, optional
        Minimum number of identified features. Default is 0.

    min_distance : float, optional
        Minimum distance between detected features (in meter). Default is 0.

    feature_number_start : int, optional
        Feature id to start with. Default is 1.

    PBC_flag : str('none', 'hdim_1', 'hdim_2', 'both')
        Sets whether to use periodic boundaries, and if so in which directions.
        'none' means that we do not have periodic boundaries
        'hdim_1' means that we are periodic along hdim1
        'hdim_2' means that we are periodic along hdim2
        'both' means that we are periodic along both horizontal dimensions
    vertical_coord: str
        Name of the vertical coordinate. If None, tries to auto-detect.
        It looks for the coordinate or the dimension name corresponding
        to the string.
    vertical_axis: int or None.
        The vertical axis number of the data. If None, uses vertical_coord
        to determine axis. This must be >=0.
    detect_subset: dict-like or None
        Whether to run feature detection on only a subset of the data.
        If this is not None, it will subset the grid that we run feature detection
        on to the range specified for each axis specified. The format of this dict is:
        {axis-number: (start, end)}, where axis-number is the number of the axis to subset,
        start is inclusive, and end is exclusive.
        For example, if your data are oriented as (time, z, y, x) and you want to
        only detect on values between z levels 10 and 29, you would set:
        {1: (10, 30)}.
    wavelength_filtering: tuple, optional
       Minimum and maximum wavelength for horizontal spectral filtering in meter.
       Default is None.

    dz : float
        Constant vertical grid spacing (m), optional. If not specified
        and the input is 3D, this function requires that `altitude` is available
        in the `features` input. If you specify a value here, this function assumes
        that it is the constant z spacing between points, even if ```z_coordinate_name```
        is specified.

    strict_thresholding: Bool, optional
        If True, a feature can only be detected if all previous thresholds have been met.
        Default is False.

    Returns
    -------
    features : pandas.DataFrame
        Detected features. The structure of this dataframe is explained
        `here <https://tobac.readthedocs.io/en/latest/data_input.html>`__
    """
    from .utils import add_coordinates, add_coordinates_3D

    logging.debug("start feature detection based on thresholds")

    if "time" not in [coord.name() for coord in field_in.coords()]:
        raise ValueError(
            "input to feature detection step must include a dimension named 'time'"
        )

    # Check whether we need to run 2D or 3D feature detection
    if field_in.ndim == 3:
        logging.debug("Running 2D feature detection")
        is_3D = False
    elif field_in.ndim == 4:
        logging.debug("Running 3D feature detection")
        is_3D = True
    else:
        raise ValueError("Feature detection only works with 2D or 3D data")

    ndim_time = field_in.coord_dims("time")[0]

    if detect_subset is not None:
        raise NotImplementedError("Subsetting feature detection not yet supported.")

    if detect_subset is not None and ndim_time in detect_subset:
        raise NotImplementedError("Cannot subset on time")

    if is_3D:
        # We need to determine the time axis so that we can determine the
        # vertical axis in each timestep if vertical_axis is not none.

        if vertical_axis is not None and vertical_coord is not None:
            raise ValueError(
                "Only one of vertical_axis or vertical_coord should be set."
            )

        if vertical_axis is None:
            # We need to determine vertical axis.
            # first, find the name of the vertical axis
            vertical_axis_name = internal_utils.find_vertical_axis_from_coord(
                field_in, vertical_coord=vertical_coord
            )
            # then find our axis number.
            vertical_axis = internal_utils.find_axis_from_coord(
                field_in, vertical_axis_name
            )

            if vertical_axis is None:
                raise ValueError("Cannot find vertical coordinate.")

        if vertical_axis < 0:
            raise ValueError("vertical_axis must be >=0.")
        # adjust vertical axis number down based on time
        if ndim_time < vertical_axis:
            # We only need to adjust the axis number if the time axis
            # is a lower axis number than the specified vertical coordinate.

            vertical_axis = vertical_axis - 1

    # create empty list to store features for all timesteps
    list_features_timesteps = []

    # loop over timesteps for feature identification:
    data_time = field_in.slices_over("time")

    # if single threshold is put in as a single value, turn it into a list
    if type(threshold) in [int, float]:
        threshold = [threshold]

    # if wavelength_filtering is given, check that value cannot be larger than distances along
    # x and y, that the value cannot be smaller or equal to the grid spacing
    # and throw a warning if dxy and wavelengths have about the same order of magnitude
    if wavelength_filtering is not None:
        if is_3D:
            raise ValueError("Wavelength filtering is not supported for 3D input data.")
        else:
            distance_x = field_in.shape[1] * (dxy)
            distance_y = field_in.shape[2] * (dxy)
        distance = min(distance_x, distance_y)

        # make sure the smaller value is taken as the minimum and the larger as the maximum
        lambda_min = min(wavelength_filtering)
        lambda_max = max(wavelength_filtering)

        if lambda_min > distance or lambda_max > distance:
            raise ValueError(
                "The given wavelengths cannot be larger than the total distance in m along the axes"
                " of the domain."
            )

        elif lambda_min <= dxy:
            raise ValueError(
                "The given minimum wavelength cannot be smaller than gridspacing dxy. Please note "
                "that both dxy and the values for wavelength_filtering should be given in meter."
            )

        elif np.floor(np.log10(lambda_min)) - np.floor(np.log10(dxy)) > 1:
            warnings.warn(
                "Warning: The values for dxy and the minimum wavelength are close in order of "
                "magnitude. Please note that both dxy and for wavelength_filtering should be "
                "given in meter."
            )

    for i_time, data_i in enumerate(data_time):
        time_i = data_i.coord("time").units.num2date(data_i.coord("time").points[0])

        features_thresholds = feature_detection_multithreshold_timestep(
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
            PBC_flag=PBC_flag,
            vertical_axis=vertical_axis,
            dxy=dxy,
            wavelength_filtering=wavelength_filtering,
            strict_thresholding=strict_thresholding,
        )
        # check if list of features is not empty, then merge features from different threshold
        # values into one DataFrame and append to list for individual timesteps:
        if not features_thresholds.empty:
            hdim1_ax, hdim2_ax = internal_utils.find_hdim_axes_3D(
                field_in, vertical_coord=vertical_coord
            )
            hdim1_max = field_in.shape[hdim1_ax] - 1
            hdim2_max = field_in.shape[hdim2_ax] - 1
            # Loop over DataFrame to remove features that are closer than distance_min to each
            # other:
            if min_distance > 0:
                features_thresholds = filter_min_distance(
                    features_thresholds,
                    dxy=dxy,
                    dz=dz,
                    min_distance=min_distance,
                    z_coordinate_name=vertical_coord,
                    target=target,
                    PBC_flag=PBC_flag,
                    min_h1=0,
                    max_h1=hdim1_max,
                    min_h2=0,
                    max_h2=hdim2_max,
                )

        list_features_timesteps.append(features_thresholds)

        logging.debug(
            "Finished feature detection for " + time_i.strftime("%Y-%m-%d_%H:%M:%S")
        )

    logging.debug("feature detection: merging DataFrames")
    # Check if features are detected and then concatenate features from different timesteps into
    # one pandas DataFrame
    # If no features are detected raise error
    if any([not x.empty for x in list_features_timesteps]):
        features = pd.concat(list_features_timesteps, ignore_index=True)
        features["feature"] = features.index + feature_number_start
        #    features_filtered = features.drop(features[features['num'] < min_num].index)
        #    features_filtered.drop(columns=['idx','num','threshold_value'],inplace=True)
        if "vdim" in features:
            features = add_coordinates_3D(
                features, field_in, vertical_coord=vertical_coord
            )
        else:
            features = add_coordinates(features, field_in)
    else:
        features = None
        logging.debug("No features detected")
    logging.debug("feature detection completed")
    return features


def filter_min_distance(
    features,
    dxy=None,
    dz=None,
    min_distance=None,
    x_coordinate_name=None,
    y_coordinate_name=None,
    z_coordinate_name=None,
    target="maximum",
    PBC_flag="none",
    min_h1=0,
    max_h1=0,
    min_h2=0,
    max_h2=0,
):
    """Function to remove features that are too close together.
    If two features are closer than `min_distance`, it keeps the
    larger feature.

    Parameters
    ----------
    features:      pandas DataFrame
                   features
    dxy:           float
        Constant horzontal grid spacing (m).
    dz: float
        Constant vertical grid spacing (m), optional. If not specified
        and the input is 3D, this function requires that `z_coordinate_name` is available
        in the `features` input. If you specify a value here, this function assumes
        that it is the constant z spacing between points, even if ```z_coordinate_name```
        is specified.
    min_distance:  float
        minimum distance between detected features (m)
    x_coordinate_name: str
        The name of the x coordinate to calculate distance based on in meters.
        This is typically `projection_x_coordinate`. Currently unused.
    y_coordinate_name: str
        The name of the y coordinate to calculate distance based on in meters.
        This is typically `projection_y_coordinate`. Currently unused.
    z_coordinate_name: str or None
        The name of the z coordinate to calculate distance based on in meters.
        This is typically `altitude`. If None, tries to auto-detect.
    target: {'maximum', 'minimum'}, optional
        Flag to determine if tracking is targetting minima or maxima in
        the data. Default is 'maximum'.
    PBC_flag : str('none', 'hdim_1', 'hdim_2', 'both')
        Sets whether to use periodic boundaries, and if so in which directions.
        'none' means that we do not have periodic boundaries
        'hdim_1' means that we are periodic along hdim1
        'hdim_2' means that we are periodic along hdim2
        'both' means that we are periodic along both horizontal dimensions
    min_h1: int, optional
        Minimum real point in hdim_1, for use with periodic boundaries.
    max_h1: int, optional
        Maximum point in hdim_1, exclusive. max_h1-min_h1 should be the size.
    min_h2: int, optional
        Minimum real point in hdim_2, for use with periodic boundaries.
    max_h2: int, optional
        Maximum point in hdim_2, exclusive. max_h2-min_h2 should be the size.


    Returns
    -------
    pandas DataFrame
        features after filtering
    """
    if dxy is None:
        raise NotImplementedError("dxy currently must be set.")

    # if PBC_flag != "none":
    #    raise NotImplementedError("We haven't yet implemented PBCs into this.")

    # if we are 3D, the vertical dimension is in features. if we are 2D, there
    # is no vertical dimension in features.
    is_3D = "vdim" in features

    if is_3D and dz is None:
        z_coordinate_name = internal_utils.find_dataframe_vertical_coord(
            features, z_coordinate_name
        )

    # Check if both dxy and their coordinate names are specified.
    # If they are, warn that we will use dxy.
    if dxy is not None and (
        x_coordinate_name in features and y_coordinate_name in features
    ):
        warnings.warn(
            "Both " + x_coordinate_name + "/" + y_coordinate_name + " and dxy "
            "set. Using constant dxy. Set dxy to None if you want to use the "
            "interpolated coordinates, or set `x_coordinate_name` and "
            "`y_coordinate_name` to None to use a constant dxy."
        )

    # Check and if both dz is specified and altitude is available, warn that we will use dz.
    if is_3D and (dz is not None and z_coordinate_name in features):
        warnings.warn(
            "Both "
            + z_coordinate_name
            + " and dz available to filter_min_distance; using constant dz. "
            "Set dz to none if you want to use altitude or set `z_coordinate_name` to None to use "
            "constant dz."
        )

    # As optional coordinate names are not yet implemented, set to defaults here:
    z_coordinate_name = "vdim"
    y_coordinate_name = "hdim_1"
    x_coordinate_name = "hdim_2"

    if target not in ["minimum", "maximum"]:
        raise ValueError(
            "target parameter must be set to either 'minimum' or 'maximum'"
        )

    # Calculate feature locations in cartesian coordinates
    if is_3D:
        feature_locations = features[
            [z_coordinate_name, y_coordinate_name, x_coordinate_name]
        ].to_numpy()
        feature_locations[0] *= dz
        feature_locations[1:] *= dxy
    else:
        feature_locations = (
            features[[y_coordinate_name, x_coordinate_name]].to_numpy() * dxy
        )

    # Create array of flags for features to remove
    removal_flag = np.zeros(len(features), dtype=bool)

    # Create Tree of feature locations in cartesian coordinates
    # Check if we have PBCs.
    if PBC_flag in ["hdim_1", "hdim_2", "both"]:
        # Note that we multiply by dxy to get the distances in spatial coordinates
        dist_func = build_distance_function(
            min_h1 * dxy, max_h1 * dxy, min_h2 * dxy, max_h2 * dxy, PBC_flag
        )
        features_tree = BallTree(feature_locations, metric="pyfunc", func=dist_func)
        neighbours = features_tree.query_radius(feature_locations, r=min_distance)

    else:
        features_tree = KDTree(feature_locations)
        # Find neighbours for each point
        neighbours = features_tree.query_ball_tree(features_tree, r=min_distance)

    # Iterate over list of neighbours to find which features to remove
    for i, neighbour_list in enumerate(neighbours):
        if len(neighbour_list) > 1:
            # Remove the feature we're interested in as it's always included
            neighbour_list = list(neighbour_list)
            neighbour_list.remove(i)
            # If maximum target check if any neighbours have a larger threshold value
            if target == "maximum" and np.any(
                features["threshold_value"].iloc[neighbour_list]
                > features["threshold_value"].iloc[i]
            ):
                removal_flag[i] = True
            # If minimum target check if any neighbours have a smaller threshold value
            elif target == "minimum" and np.any(
                features["threshold_value"].iloc[neighbour_list]
                < features["threshold_value"].iloc[i]
            ):
                removal_flag[i] = True
            # Else check if any neighbours have an equal threshold value
            else:
                wh_equal_threshold = (
                    features["threshold_value"].iloc[neighbour_list]
                    == features["threshold_value"].iloc[i]
                )
                if np.any(wh_equal_threshold):
                    # Check if any have a larger number of points
                    if np.any(
                        features["num"].iloc[neighbour_list][wh_equal_threshold]
                        > features["num"].iloc[i]
                    ):
                        removal_flag[i] = True
                    # Check if any have the same number of points and a lower index value
                    else:
                        wh_equal_area = (
                            features["num"].iloc[neighbour_list][wh_equal_threshold]
                            == features["num"].iloc[i]
                        )
                        if np.any(wh_equal_area):
                            if np.any(wh_equal_area.index[wh_equal_area] < i):
                                removal_flag[i] = True

    # Return the features that are not flagged for removal
    return features.iloc[~removal_flag]

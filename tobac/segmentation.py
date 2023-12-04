"""Provide segmentation techniques.

Segmentation techniques are used to associate areas or volumes to each
identified feature. The segmentation is implemented using watershedding
techniques from the field of image processing with a fixed threshold
value. This value has to be set specifically for every type of input
data and application. The segmentation can be performed for both
two-dimensional and three-dimensional data. At each timestep, a marker
is set at the position (weighted mean center) of each feature identified
in the detection step in an array otherwise filled with zeros. In case
of the three-dimentional watershedding, all cells in the column above
the weighted mean center position of the identified features fulfilling
the threshold condition are set to the respective marker. The algorithm
then fills the area (2D) or volume (3D) based on the input field
starting from these markers until reaching the threshold. If two or more
features are directly connected, the border runs along the
watershed line between the two regions. This procedure creates a mask 
that has the same form as the input data, with the corresponding integer 
number at all grid points that belong to a feature, else with zero. This 
mask can be conveniently and efficiently used to select the volume of each
feature at a specific time step for further analysis or visialization. 

References
----------
.. Heikenfeld, M., Marinescu, P. J., Christensen, M.,
   Watson-Parris, D., Senf, F., van den Heever, S. C.
   & Stier, P. (2019). tobac 1.2: towards a flexible 
   framework for tracking and analysis of clouds in 
   diverse datasets. Geoscientific Model Development,
   12(11), 4551-4570.
"""
import copy
import logging

import iris.cube
import xarray as xr
import numpy as np
import pandas as pd
from typing_extensions import Literal
from typing import Union, Callable

import skimage
import numpy as np
import pandas as pd

from . import utils as tb_utils
from .utils import periodic_boundaries as pbc_utils
from .utils import internal as internal_utils
from .utils import get_statistics


def add_markers(
    features: pd.DataFrame,
    marker_arr: np.array,
    seed_3D_flag: Literal["column", "box"],
    seed_3D_size: Union[int, tuple[int]] = 5,
    level: Union[None, slice] = None,
    PBC_flag: Literal["none", "hdim_1", "hdim_2", "both"] = "none",
) -> np.array:
    """Adds markers for watershedding using the `features` dataframe
    to the marker_arr.

    Parameters
    ----------
    features: pandas.DataFrame
        Features for one point in time to add as markers.
    marker_arr: 2D or 3D array-like
        Array to add the markers to. Assumes a (z, y, x) configuration.
    seed_3D_flag: str('column', 'box')
        Seed 3D field at feature positions with either the full column
         or a box of user-set size
    seed_3D_size: int or tuple (dimensions equal to dimensions of `field`)
        This sets the size of the seed box when `seed_3D_flag` is 'box'. If it's an
        integer (units of number of pixels), the seed box is identical in all dimensions.
        If it's a tuple, it specifies the seed area for each dimension separately, in units of pixels.
        Note: we strongly recommend the use of odd numbers for this. If you give
        an even number, your seed box will be biased and not centered
        around the feature.
        Note: if two seed boxes overlap, the feature that is seeded will be the
        closer feature.
    level: slice or None
        If `seed_3D_flag` is 'column', the levels at which to seed the
        cells for the watershedding algorithm. If None, seeds all levels.
    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}
        Sets whether to use periodic boundaries, and if so in which directions.
        'none' means that we do not have periodic boundaries
        'hdim_1' means that we are periodic along hdim1
        'hdim_2' means that we are periodic along hdim2
        'both' means that we are periodic along both horizontal dimensions

    Returns
    -------
    2D or 3D array like (same type as `marker_arr`)
        The marker array
    """
    if seed_3D_flag not in ["column", "box"]:
        raise ValueError('seed_3D_flag must be either "column" or "box"')

    # What marker number is the background? Assumed 0.
    bg_marker = 0

    if level is None:
        level = slice(None)

    if len(marker_arr.shape) == 3:
        is_3D = True
        z_len = marker_arr.shape[0]
        h1_len = marker_arr.shape[1]
        h2_len = marker_arr.shape[2]

    else:
        is_3D = False
        z_len = 0
        h1_len = marker_arr.shape[0]
        h2_len = marker_arr.shape[1]
        # transpose to 3D array to make things easier.
        marker_arr = marker_arr[np.newaxis, :, :]

    if seed_3D_flag == "column":
        for _, row in features.iterrows():
            # Offset marker locations by 0.5 to find nearest pixel
            marker_arr[
                level,
                int(row["hdim_1"] + 0.5) % h1_len,
                int(row["hdim_2"] + 0.5) % h2_len,
            ] = row["feature"]

    elif seed_3D_flag == "box":
        # Get the size of the seed box from the input parameter
        try:
            if is_3D:
                seed_z = seed_3D_size[0]
                start_num = 1
            else:
                start_num = 0
            seed_h1 = seed_3D_size[start_num]
            seed_h2 = seed_3D_size[start_num + 1]
        except TypeError:
            # Not iterable, assume int.
            seed_z = seed_3D_size
            seed_h1 = seed_3D_size
            seed_h2 = seed_3D_size

        for _, row in features.iterrows():
            if is_3D:
                # If we have a 3D input and we need to do box seeding
                # we need to have 3D features.
                try:
                    row["vdim"]
                except KeyError:
                    raise ValueError(
                        "For Box seeding on 3D segmentation,"
                        " you must have a 3D input source."
                    )

            # Because we don't support PBCs on the vertical axis,
            # this is simple- just go in the seed_z/2 points around the
            # vdim of the feature, up to the limits of the array.
            if is_3D:
                z_seed_start = int(np.max([0, np.ceil(row["vdim"] - seed_z / 2)]))
                z_seed_end = int(np.min([z_len, np.ceil(row["vdim"] + seed_z / 2)]))
            else:
                z_seed_start = 0
                z_seed_end = 1
            # For the horizontal dimensions, it's more complicated if we have
            # PBCs.
            hdim_1_min = int(np.ceil(row["hdim_1"] - seed_h1 / 2))
            hdim_1_max = int(np.ceil(row["hdim_1"] + seed_h1 / 2))
            hdim_2_min = int(np.ceil(row["hdim_2"] - seed_h2 / 2))
            hdim_2_max = int(np.ceil(row["hdim_2"] + seed_h2 / 2))

            all_seed_boxes = pbc_utils.get_pbc_coordinates(
                h1_min=0,
                h1_max=h1_len,
                h2_min=0,
                h2_max=h2_len,
                h1_start_coord=hdim_1_min,
                h1_end_coord=hdim_1_max,
                h2_start_coord=hdim_2_min,
                h2_end_coord=hdim_2_max,
                PBC_flag=PBC_flag,
            )
            for seed_box in all_seed_boxes:
                # Need to see if there are any other points seeded
                # in this seed box first.
                curr_box_markers = marker_arr[
                    z_seed_start:z_seed_end,
                    seed_box[0] : seed_box[1],
                    seed_box[2] : seed_box[3],
                ]
                all_feats_in_box = np.unique(curr_box_markers)
                if np.any(curr_box_markers != bg_marker):
                    # If we have non-background points already seeded,
                    # we need to find the best way to seed them.
                    # Currently seeding with the closest point.
                    # Loop through all points in the box
                    with np.nditer(curr_box_markers, flags=["multi_index"]) as it:
                        for curr_box_pt in it:
                            # Get its global index so that we can calculate
                            # distance and set the array.
                            local_index = it.multi_index
                            global_index = (
                                local_index[0] + z_seed_start,
                                local_index[1] + seed_box[0],
                                local_index[2] + seed_box[2],
                            )
                            # If it's a background marker, we can just set it
                            # with the feature we're working on.
                            if curr_box_pt == bg_marker:
                                marker_arr[global_index] = row["feature"]
                                continue
                            # it has another feature in it. Calculate the distance
                            # from its current set feature and the new feature.
                            if is_3D:
                                curr_coord = (row["vdim"], row["hdim_1"], row["hdim_2"])
                            else:
                                curr_coord = (0, row["hdim_1"], row["hdim_2"])

                            dist_from_curr_pt = pbc_utils.calc_distance_coords_pbc(
                                np.array(global_index),
                                np.array(curr_coord),
                                min_h1=0,
                                max_h1=h1_len,
                                min_h2=0,
                                max_h2=h2_len,
                                PBC_flag=PBC_flag,
                            )

                            # This is technically an O(N^2) operation, but
                            # hopefully performance isn't too bad as this should
                            # be rare.
                            orig_row = features[
                                features["feature"] == curr_box_pt
                            ].iloc[0]
                            if is_3D:
                                orig_coord = (
                                    orig_row["vdim"],
                                    orig_row["hdim_1"],
                                    orig_row["hdim_2"],
                                )
                            else:
                                orig_coord = (0, orig_row["hdim_1"], orig_row["hdim_2"])
                            dist_from_orig_pt = pbc_utils.calc_distance_coords_pbc(
                                np.array(global_index),
                                np.array(orig_coord),
                                min_h1=0,
                                max_h1=h1_len,
                                min_h2=0,
                                max_h2=h2_len,
                                PBC_flag=PBC_flag,
                            )
                            # The current point center is further away
                            # than the original point center, so do nothing
                            if dist_from_curr_pt > dist_from_orig_pt:
                                continue
                            else:
                                # the current point center is closer.
                                marker_arr[global_index] = row["feature"]
                # completely unseeded region so far.
                else:
                    marker_arr[
                        z_seed_start:z_seed_end,
                        seed_box[0] : seed_box[1],
                        seed_box[2] : seed_box[3],
                    ] = row["feature"]

    # If we aren't 3D, transpose back.
    if not is_3D:
        marker_arr = marker_arr[0, :, :]

    return marker_arr


def segmentation_3D(
    features,
    field,
    dxy,
    threshold=3e-3,
    target="maximum",
    level=None,
    method="watershed",
    max_distance=None,
    PBC_flag="none",
    seed_3D_flag="column",
    statistic=None,
):
    """Wrapper for the segmentation()-function."""

    return segmentation(
        features,
        field,
        dxy,
        threshold=threshold,
        target=target,
        level=level,
        method=method,
        max_distance=max_distance,
        PBC_flag=PBC_flag,
        seed_3D_flag=seed_3D_flag,
        statistic=statistic,
    )


def segmentation_2D(
    features,
    field,
    dxy,
    threshold=3e-3,
    target="maximum",
    level=None,
    method="watershed",
    max_distance=None,
    PBC_flag="none",
    seed_3D_flag="column",
    statistic=None,
):
    """Wrapper for the segmentation()-function."""
    return segmentation(
        features,
        field,
        dxy,
        threshold=threshold,
        target=target,
        level=level,
        method=method,
        max_distance=max_distance,
        PBC_flag=PBC_flag,
        seed_3D_flag=seed_3D_flag,
        statistic=statistic,
    )


@internal_utils.iris_to_xarray
def segmentation_timestep(
    field_in: xr.DataArray,
    features_in: pd.DataFrame,
    dxy: float,
    threshold: float = 3e-3,
    target: Literal["maximum", "minimum"] = "maximum",
    level: Union[None, slice] = None,
    method: Literal["watershed"] = "watershed",
    max_distance: Union[None, float] = None,
    vertical_coord: Union[str, None] = None,
    PBC_flag: Literal["none", "hdim_1", "hdim_2", "both"] = "none",
    seed_3D_flag: Literal["column", "box"] = "column",
    seed_3D_size: Union[int, tuple[int]] = 5,
    segment_number_below_threshold: int = 0,
    segment_number_unassigned: int = 0,
    statistic: Union[dict[str, Union[Callable, tuple[Callable, dict]]], None] = None,
) -> tuple[iris.cube.Cube, pd.DataFrame]:
    """Perform watershedding for an individual time step of the data. Works
    for both 2D and 3D data

    Parameters
    ----------
    field_in : iris.cube.Cube
        Input field to perform the watershedding on (2D or 3D for one
        specific point in time).

    features_in : pandas.DataFrame
        Features for one specific point in time.

    dxy : float
        Grid spacing of the input data in metres

    threshold : float, optional
        Threshold for the watershedding field to be used for the mask. The watershedding is exclusive of the threshold value, i.e. values greater (less) than the threshold are included in the target region, while values equal to the threshold value are excluded.
        Default is 3e-3.

    target : {'maximum', 'minimum'}, optional
        Flag to determine if tracking is targeting minima or maxima in
        the data to determine from which direction to approach the threshold
        value. Default is 'maximum'.

    level : slice of iris.cube.Cube, optional
        Levels at which to seed the cells for the watershedding
        algorithm. Default is None.

    method : {'watershed'}, optional
        Flag determining the algorithm to use (currently watershedding
        implemented).

    max_distance : float, optional
        Maximum distance from a marker allowed to be classified as
        belonging to that cell in meters. Default is None.

    vertical_coord : str, optional
        Vertical coordinate in 3D input data. If None, input is checked for
        one of {'z', 'model_level_number', 'altitude','geopotential_height'}
        as a likely coordinate name

    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}
        Sets whether to use periodic boundaries, and if so in which directions.
        'none' means that we do not have periodic boundaries
        'hdim_1' means that we are periodic along hdim1
        'hdim_2' means that we are periodic along hdim2
        'both' means that we are periodic along both horizontal dimensions
    seed_3D_flag: str('column', 'box')
        Seed 3D field at feature positions with either the full column (default)
         or a box of user-set size
    seed_3D_size: int or tuple (dimensions equal to dimensions of `field`)
        This sets the size of the seed box when `seed_3D_flag` is 'box'. If it's an
        integer (units of number of pixels), the seed box is identical in all dimensions.
        If it's a tuple, it specifies the seed area for each dimension separately, in units of pixels.
        Note: we strongly recommend the use of odd numbers for this. If you give
        an even number, your seed box will be biased and not centered
        around the feature.
        Note: if two seed boxes overlap, the feature that is seeded will be the
        closer feature.
    segment_number_below_threshold: int
        the marker to use to indicate a segmentation point is below the threshold.
    segment_number_unassigned: int
        the marker to use to indicate a segmentation point is above the threshold but unsegmented.
        This can be the same as `segment_number_below_threshold`, but can also be set separately.
    statistics: boolean, optional
        Default is None. If True, bulk statistics for the data points assigned to each feature are saved in output.


    Returns
    -------
    segmentation_out : iris.cube.Cube
        Mask, 0 outside and integer numbers according to track
        inside the ojects.

    features_out : pandas.DataFrame
        Feature dataframe including the number of cells (2D or 3D) in
        the segmented area/volume of the feature at the timestep.

    Raises
    ------
    ValueError
        If target is neither 'maximum' nor 'minimum'.

        If vertical_coord is not in {'auto', 'z', 'model_level_number',
                                     'altitude', geopotential_height'}.

        If there is more than one coordinate name.

        If the spatial dimension is neither 2 nor 3.

        If method is not 'watershed'.

    """

    # The location of watershed within skimage submodules changes with v0.19, but I've kept both for backward compatibility for now
    try:
        from skimage.segmentation import watershed
    except ImportError:
        from skimage.morphology import watershed
    # from skimage.segmentation import random_walker
    from scipy.ndimage import distance_transform_edt
    from copy import deepcopy

    if max_distance is not None and PBC_flag in ["hdim_1", "hdim_2", "both"]:
        raise NotImplementedError("max_distance not yet implemented for PBCs")

    # How many dimensions are we using?
    if field_in.ndim == 2:
        hdim_1_axis = 0
        hdim_2_axis = 1
        vertical_coord_axis = None
    elif field_in.ndim == 3:
        vertical_axis = internal_utils.find_vertical_axis_from_coord(
            field_in, vertical_coord=vertical_coord
        )
        ndim_vertical = internal_utils.find_axis_from_coord(vertical_axis)
        if len(ndim_vertical) > 1:
            raise ValueError("please specify 1 dimensional vertical coordinate")
        vertical_coord_axis = ndim_vertical[0]
        # Once we know the vertical coordinate, we can resolve the
        # horizontal coordinates
        # To make things easier, we will transpose the axes
        # so that they are consistent.

        hdim_1_axis, hdim_2_axis = internal_utils.find_hdim_axes_3D(
            field_in, vertical_axis=vertical_coord_axis
        )
    else:
        raise ValueError(
            "Segmentation routine only possible with 2 or 3 spatial dimensions"
        )

    if segment_number_below_threshold > 0 or segment_number_unassigned > 0:
        raise ValueError("Below/above threshold markers must be <=0")

    # copy feature dataframe for output
    features_out = deepcopy(features_in)
    # Create cube of the same dimensions and coordinates as input data to store mask:
    segmentation_out = field_in.copy(deep=True)
    segmentation_out = segmentation_out.rename("segmentation_mask")

    # Get raw array from input data:
    data = field_in.values
    is_3D_seg = len(data.shape) == 3
    # To make things easier, we will transpose the axes
    # so that they are consistent: z, hdim_1, hdim_2
    # We only need to do this for 3D.
    transposed_data = False
    if is_3D_seg:
        if vertical_coord_axis == 1:
            data = np.transpose(data, axes=(1, 0, 2))
            transposed_data = True
        elif vertical_coord_axis == 2:
            data = np.transpose(data, axes=(2, 0, 1))
            transposed_data = True

    # Set level at which to create "Seed" for each feature in the case of 3D watershedding:
    # If none, use all levels (later reduced to the ones fulfilling the theshold conditions)
    if level is None:
        level = slice(None)

    # transform max_distance in metres to distance in pixels:
    if max_distance is not None:
        max_distance_pixel = np.ceil(max_distance / dxy)

    # mask data outside region above/below threshold and invert data if tracking maxima:
    if target == "maximum":
        unmasked = data > threshold
        data_segmentation = -1 * data
    elif target == "minimum":
        unmasked = data < threshold
        data_segmentation = data
    else:
        raise ValueError("unknown type of target")

    # set markers at the positions of the features:
    markers = np.zeros(unmasked.shape).astype(np.int32)
    markers = add_markers(
        features_in, markers, seed_3D_flag, seed_3D_size, level, PBC_flag
    )
    # set markers in cells not fulfilling threshold condition to zero:
    markers[~unmasked] = 0
    # marker_vals = np.unique(markers)

    # Turn into np arrays (not necessary for markers) as dask arrays don't yet seem to work for watershedding algorithm
    data_segmentation = np.array(data_segmentation)
    unmasked = np.array(unmasked)

    # perform segmentation:
    if method == "watershed":
        segmentation_mask = watershed(
            np.array(data_segmentation), markers.astype(np.int32), mask=unmasked
        )
    else:
        raise ValueError("unknown method, must be watershed")

    # remove everything from the individual masks that is more than max_distance_pixel away from the markers
    if max_distance is not None:
        D = distance_transform_edt((markers == 0))
        segmentation_mask[
            np.bitwise_and(segmentation_mask > 0, D > max_distance_pixel)
        ] = 0

    # mask all segmentation_mask points below threshold as -1
    # to differentiate from those unmasked points NOT filled by watershedding
    # TODO: allow user to specify
    points_below_threshold_val = -1
    segmentation_mask[~unmasked] = points_below_threshold_val

    hdim1_min = 0
    hdim1_max = segmentation_mask.shape[hdim_1_axis] - 1
    hdim2_min = 0
    hdim2_max = segmentation_mask.shape[hdim_2_axis] - 1

    # all options that involve dealing with periodic boundaries
    pbc_options = ["hdim_1", "hdim_2", "both"]
    # Only run this if we need to deal with PBCs
    if PBC_flag in pbc_options:
        if not is_3D_seg:
            # let's transpose segmentation_mask to a 1,y,x array to make calculations etc easier.
            segmentation_mask = segmentation_mask[np.newaxis, :, :]
            unmasked = unmasked[np.newaxis, :, :]
            data_segmentation = data_segmentation[np.newaxis, :, :]
            vertical_coord_axis = 0
            hdim_1_axis = 1
            hdim_2_axis = 2

        seg_mask_unseeded = np.zeros(segmentation_mask.shape)

        # Return all indices where segmentation field == 0
        # meaning unfilled but above threshold
        # TODO: is there a way to do this without np.where?
        vdim_unf, hdim1_unf, hdim2_unf = np.where(segmentation_mask == 0)
        seg_mask_unseeded[vdim_unf, hdim1_unf, hdim2_unf] = 1

        # create labeled field of unfilled, unseeded features
        labels_unseeded, label_num = skimage.measure.label(
            seg_mask_unseeded, return_num=True
        )

        markers_2 = np.zeros(data_segmentation.shape, dtype=np.int32)

        # PBC marker seeding approach
        # loop thru LB points, then check if fillable region (labels_unseeded > 0) and seed
        # then check if point on other side of boundary is > 0 in segmentation_mask and
        # adjust where needed
        """
        "First pass" at seeding features across the boundaries. This first pass will bring in
        eligible (meaning values that are higher than threshold) but not previously watershedded 
        points across the boundary by seeding them with the appropriate feature across the boundary.

        Later, we will run the second pass or "buddy box" approach that handles cases where points across the boundary
        have been watershedded already. 
        """
        if PBC_flag == "hdim_1" or PBC_flag == "both":
            check_add_unseeded_across_bdrys(
                "hdim_1",
                segmentation_mask,
                labels_unseeded,
                hdim1_min,
                hdim1_max,
                markers_2,
            )
        if PBC_flag == "hdim_2" or PBC_flag == "both":
            check_add_unseeded_across_bdrys(
                "hdim_2",
                segmentation_mask,
                labels_unseeded,
                hdim2_min,
                hdim2_max,
                markers_2,
            )

        # Deal with the opposite corner only
        if PBC_flag == "both":
            # TODO: This seems quite slow, is there scope for further speedup?
            for vdim_ind in range(0, segmentation_mask.shape[0]):
                for hdim1_ind in [hdim1_min, hdim1_max]:
                    for hdim2_ind in [hdim2_min, hdim2_max]:
                        # If this point is unseeded and unlabeled
                        if labels_unseeded[vdim_ind, hdim1_ind, hdim2_ind] == 0:
                            continue

                        # Find the opposite point in hdim1 space
                        hdim1_opposite_corner = (
                            hdim1_min if hdim1_ind == hdim1_max else hdim1_max
                        )
                        hdim2_opposite_corner = (
                            hdim2_min if hdim2_ind == hdim2_max else hdim2_max
                        )
                        if (
                            segmentation_mask[
                                vdim_ind, hdim1_opposite_corner, hdim2_opposite_corner
                            ]
                            <= 0
                        ):
                            continue

                        markers_2[vdim_ind, hdim1_ind, hdim2_ind] = segmentation_mask[
                            vdim_ind, hdim1_opposite_corner, hdim2_opposite_corner
                        ]

        markers_2[~unmasked] = 0

        if method == "watershed":
            segmentation_mask_2 = watershed(
                data_segmentation, markers_2.astype(np.int32), mask=unmasked
            )
        else:
            raise ValueError("unknown method, must be watershed")

        # Sum up original mask and secondary PBC-mask for full PBC segmentation
        segmentation_mask_3 = segmentation_mask + segmentation_mask_2

        # Secondary seeding complete, now blending periodic boundaries
        # keep segmentation mask fields for now so we can save these all later
        # for demos of changes, otherwise, could add deletion for memory efficiency, e.g.

        # del segmentation_mask
        # del segmentation_mask_2
        # gc.collect()

        # update mask coord regions

        """
        Now, start the second round of watershedding- the "buddy box" approach.
        'buddies' array contains features of interest and any neighbors that are across the boundary or 
        otherwise have lateral and/or diagonal physical contact with that label.
        The "buddy box" is also used for multiple crossings of the boundaries with segmented features.
        """

        # TODO: this is a very inelegant way of handling this problem. We should wrap up the pure
        # segmentation routines and simply call them again here with the same parameters.
        reg_props_dict = internal_utils.get_label_props_in_dict(segmentation_mask_3)

        if len(reg_props_dict) != 0:
            (
                curr_reg_inds,
                z_reg_inds,
                y_reg_inds,
                x_reg_inds,
            ) = internal_utils.get_indices_of_labels_from_reg_prop_dict(reg_props_dict)

        wall_labels = np.array([])

        w_wall = np.unique(segmentation_mask_3[:, :, 0])
        wall_labels = np.append(wall_labels, w_wall)

        s_wall = np.unique(segmentation_mask_3[:, 0, :])
        wall_labels = np.append(wall_labels, s_wall)

        wall_labels = np.unique(wall_labels)
        wall_labels = wall_labels[(wall_labels) > 0].astype(int)

        # Loop through all segmentation mask labels on the wall
        for cur_idx in wall_labels:
            vdim_indices = z_reg_inds[cur_idx]
            hdim1_indices = y_reg_inds[cur_idx]
            hdim2_indices = x_reg_inds[cur_idx]

            # start buddies array with feature of interest
            buddies = np.array([cur_idx], dtype=int)
            # Loop through all points in the segmentation mask that we're intertested in
            for label_z, label_y, label_x in zip(
                vdim_indices, hdim1_indices, hdim2_indices
            ):
                # check if this is the special case of being a corner point.
                # if it's doubly periodic AND on both x and y boundaries, it's a corner point
                # and we have to look at the other corner.
                # here, we will only look at the corner point and let the below deal with x/y only.
                if PBC_flag == "both" and (
                    np.any(label_y == [hdim1_min, hdim1_max])
                    and np.any(label_x == [hdim2_min, hdim2_max])
                ):
                    # adjust x and y points to the other side
                    y_val_alt = pbc_utils.adjust_pbc_point(
                        label_y, hdim1_min, hdim1_max
                    )
                    x_val_alt = pbc_utils.adjust_pbc_point(
                        label_x, hdim2_min, hdim2_max
                    )
                    label_on_corner = segmentation_mask_3[label_z, y_val_alt, x_val_alt]

                    if label_on_corner >= 0:
                        # add opposite-corner buddy if it exists
                        buddies = np.append(buddies, label_on_corner)

                # on the hdim1 boundary and periodic on hdim1
                if (PBC_flag == "hdim_1" or PBC_flag == "both") and np.any(
                    label_y == [hdim1_min, hdim1_max]
                ):
                    y_val_alt = pbc_utils.adjust_pbc_point(
                        label_y, hdim1_min, hdim1_max
                    )

                    # get the label value on the opposite side
                    label_alt = segmentation_mask_3[label_z, y_val_alt, label_x]

                    # if it's labeled and not already been dealt with
                    if label_alt >= 0:
                        # add above/below buddy if it exists
                        buddies = np.append(buddies, label_alt)

                if (PBC_flag == "hdim_2" or PBC_flag == "both") and np.any(
                    label_x == [hdim2_min, hdim2_max]
                ):
                    x_val_alt = pbc_utils.adjust_pbc_point(
                        label_x, hdim2_min, hdim2_max
                    )

                    # get the seg value on the opposite side
                    label_alt = segmentation_mask_3[label_z, label_y, x_val_alt]

                    # if it's labeled and not already been dealt with
                    if label_alt >= 0:
                        # add left/right buddy if it exists
                        buddies = np.append(buddies, label_alt)

            buddies = np.unique(buddies)

            if np.all(buddies == cur_idx):
                continue
            else:
                inter_buddies, feat_inds, buddy_inds = np.intersect1d(
                    features_in.feature.values[:], buddies, return_indices=True
                )

            # Get features that are needed for the buddy box
            buddy_features = deepcopy(features_in.iloc[feat_inds])

            # create arrays to contain points of all buddies
            # and their transpositions/transformations
            # for use in Buddy Box space

            # z,y,x points in the grid domain with no transformations
            # NOTE: when I think about it, not sure if these are really needed
            # as we use the y_a1/x_a1 points for the data transposition
            # to the buddy box rather than these and their z2/y2/x2 counterparts
            buddy_z = np.array([], dtype=int)
            buddy_y = np.array([], dtype=int)
            buddy_x = np.array([], dtype=int)

            # z,y,x points from the grid domain WHICH MAY OR MAY NOT BE TRANSFORMED
            # so as to be continuous/contiguous across a grid boundary for that dimension
            # (e.g., instead of [1496,1497,0,1,2,3] it would be [1496,1497,1498,1499,1500,1501])
            buddy_z2 = np.array([], dtype=int)
            buddy_y2 = np.array([], dtype=int)
            buddy_x2 = np.array([], dtype=int)

            # These are just for feature positions and are in z2/y2/x2 space
            # (may or may not be within real grid domain)
            # so that when the buddy box is constructed, seeding is done properly
            # in the buddy box space

            # NOTE: We may not need this, as we already do this editing the buddy_features df
            # and an iterrows call through this is what's used to actually seed the buddy box

            buddy_looper = 0

            # loop thru buddies
            for buddy in buddies:
                if buddy == 0:
                    continue
                # isolate feature from set of buddies
                buddy_feat = features_in[features_in["feature"] == buddy].iloc[0]

                # transform buddy feature position if needed for positioning in z2/y2/x2 space
                # MAY be redundant with what is done just below here
                yf2 = pbc_utils.transfm_pbc_point(
                    int(buddy_feat.hdim_1), hdim1_min, hdim1_max
                )
                xf2 = pbc_utils.transfm_pbc_point(
                    int(buddy_feat.hdim_2), hdim2_min, hdim2_max
                )

                # edit value in buddy_features dataframe
                buddy_features.hdim_1.values[
                    buddy_looper
                ] = pbc_utils.transfm_pbc_point(
                    float(buddy_feat.hdim_1), hdim1_min, hdim1_max
                )
                buddy_features.hdim_2.values[
                    buddy_looper
                ] = pbc_utils.transfm_pbc_point(
                    float(buddy_feat.hdim_2), hdim2_min, hdim2_max
                )

                buddy_looper = buddy_looper + 1
                # Create 1:1 map through actual domain points and continuous/contiguous points
                # used to identify buddy box dimension lengths for its construction
                for z, y, x in zip(
                    z_reg_inds[buddy], y_reg_inds[buddy], x_reg_inds[buddy]
                ):
                    buddy_z = np.append(buddy_z, z)
                    buddy_y = np.append(buddy_y, y)
                    buddy_x = np.append(buddy_x, x)

                    y2 = pbc_utils.transfm_pbc_point(y, hdim1_min, hdim1_max)
                    x2 = pbc_utils.transfm_pbc_point(x, hdim2_min, hdim2_max)

                    buddy_z2 = np.append(buddy_z2, z)
                    buddy_y2 = np.append(buddy_y2, y2)
                    buddy_x2 = np.append(buddy_x2, x2)

            # Buddy Box!
            # Identify mins and maxes of Buddy Box continuous points range
            # so that box of correct size can be constructed
            bbox_zstart = int(np.min(buddy_z2))
            bbox_ystart = int(np.min(buddy_y2))
            bbox_xstart = int(np.min(buddy_x2))
            bbox_zend = int(np.max(buddy_z2) + 1)
            bbox_yend = int(np.max(buddy_y2) + 1)
            bbox_xend = int(np.max(buddy_x2) + 1)

            bbox_zsize = bbox_zend - bbox_zstart
            bbox_ysize = bbox_yend - bbox_ystart
            bbox_xsize = bbox_xend - bbox_xstart

            # Creation of actual Buddy Box space for transposition
            # of data in domain and re-seeding with Buddy feature markers
            buddy_rgn = np.zeros((bbox_zsize, bbox_ysize, bbox_xsize))

            # need to loop thru ALL z,y,x inds in buddy box
            # not just the ones that have nonzero seg mask values

            # "_a1" points are re-transformations from the continuous buddy box points
            # back to original grid/domain space to ensure that the correct data are
            # copied to the proper Buddy Box locations
            for z in range(bbox_zstart, bbox_zend):
                for y in range(bbox_ystart, bbox_yend):
                    for x in range(bbox_xstart, bbox_xend):
                        z_a1 = z
                        if y > hdim1_max:
                            y_a1 = y - (hdim1_max + 1)
                        else:
                            y_a1 = y

                        if x > hdim2_max:
                            x_a1 = x - (hdim2_max + 1)
                        else:
                            x_a1 = x
                        if is_3D_seg:
                            buddy_rgn[
                                z - bbox_zstart, y - bbox_ystart, x - bbox_xstart
                            ] = field_in.data[z_a1, y_a1, x_a1]
                        else:
                            buddy_rgn[
                                z - bbox_zstart, y - bbox_ystart, x - bbox_xstart
                            ] = field_in.data[y_a1, x_a1]

            # Update buddy_features feature positions to correspond to buddy box space
            # rather than domain space or continuous/contiguous point space
            if "vdim" not in buddy_features:
                buddy_features["vdim"] = np.zeros(len(buddy_features), dtype=int)
            for buddy_looper in range(0, len(buddy_features)):
                buddy_features.vdim.values[buddy_looper] = (
                    buddy_features.vdim.values[buddy_looper] - bbox_zstart
                )

                buddy_features.hdim_1.values[buddy_looper] = (
                    buddy_features.hdim_1.values[buddy_looper] - bbox_ystart
                )
                buddy_features.hdim_2.values[buddy_looper] = (
                    buddy_features.hdim_2.values[buddy_looper] - bbox_xstart
                )

            # Create dask array from input data:
            buddy_data = buddy_rgn

            # All of the below is the same overarching segmentation procedure as in the original
            # segmentation approach until the line which states
            # "#transform segmentation_mask_4 data back to original mask after PBC first-pass ("segmentation_mask_3")"
            # It's just performed on the buddy box and its data rather than our full domain

            # mask data outside region above/below threshold and invert data if tracking maxima:
            if target == "maximum":
                unmasked_buddies = buddy_data > threshold
                buddy_segmentation = -1 * buddy_data
            elif target == "minimum":
                unmasked_buddies = buddy_data < threshold
                buddy_segmentation = buddy_data
            else:
                raise ValueError("unknown type of target")

            # set markers at the positions of the features:
            buddy_markers = np.zeros(unmasked_buddies.shape).astype(np.int32)
            # Buddy boxes are always without PBCs
            buddy_markers = add_markers(
                buddy_features,
                buddy_markers,
                seed_3D_flag,
                seed_3D_size,
                level,
                PBC_flag="none",
            )

            # set markers in cells not fulfilling threshold condition to zero:
            buddy_markers[~unmasked_buddies] = 0

            marker_vals = np.unique(buddy_markers)

            # Turn into np arrays (not necessary for markers) as dask arrays don't yet seem to work for watershedding algorithm
            buddy_segmentation = np.array(buddy_segmentation)
            unmasked_buddies = np.array(unmasked_buddies)

            # perform segmentation:
            if method == "watershed":
                segmentation_mask_4 = watershed(
                    np.array(buddy_segmentation),
                    buddy_markers.astype(np.int32),
                    mask=unmasked_buddies,
                )

            else:
                raise ValueError("unknown method, must be watershed")

            # remove everything from the individual masks that is more than max_distance_pixel away from the markers

            # mask all segmentation_mask points below threshold as -1
            # to differentiate from those unmasked points NOT filled by watershedding
            segmentation_mask_4[~unmasked_buddies] = -1

            # transform segmentation_mask_4 data back to mask created after PBC first-pass ("segmentation_mask_3")

            # loop through buddy box inds and analogous seg mask inds
            for z_val in range(bbox_zstart, bbox_zend):
                z_seg = z_val - bbox_zstart
                z_val_o = z_val
                for y_val in range(bbox_ystart, bbox_yend):
                    y_seg = y_val - bbox_ystart
                    # y_val_o = y_val
                    if y_val > hdim1_max:
                        y_val_o = y_val - (hdim1_max + 1)
                    else:
                        y_val_o = y_val
                    for x_val in range(bbox_xstart, bbox_xend):
                        x_seg = x_val - bbox_xstart
                        # x_val_o = x_val
                        if x_val > hdim2_max:
                            x_val_o = x_val - (hdim2_max + 1)
                        else:
                            x_val_o = x_val

                        # fix to
                        # overwrite IF:
                        # 1) feature of interest
                        # 2) changing to/from feature of interest or adjacent segmented feature

                        # We don't want to overwrite other features that may be in the
                        # buddy box if not contacting the intersected seg field

                        if np.any(
                            segmentation_mask_3[z_val_o, y_val_o, x_val_o] == buddies
                        ) and np.any(
                            segmentation_mask_4.data[z_seg, y_seg, x_seg] == buddies
                        ):
                            # only do updating procedure if old and new values both in buddy set
                            # and values are different
                            if (
                                segmentation_mask_3[z_val_o, y_val_o, x_val_o]
                                != segmentation_mask_4.data[z_seg, y_seg, x_seg]
                            ):
                                segmentation_mask_3[
                                    z_val_o, y_val_o, x_val_o
                                ] = segmentation_mask_4.data[z_seg, y_seg, x_seg]
        if not is_3D_seg:
            segmentation_mask_3 = segmentation_mask_3[0]

        segmentation_mask = segmentation_mask_3

    if transposed_data:
        if vertical_coord_axis == 1:
            segmentation_mask = np.transpose(segmentation_mask, axes=(1, 0, 2))
        elif vertical_coord_axis == 2:
            segmentation_mask = np.transpose(segmentation_mask, axes=(1, 2, 0))

    # Finished PBC checks and new PBC updated segmentation now in segmentation_mask.
    # Write resulting mask into cube for output
    wh_below_threshold = segmentation_mask == -1
    wh_unsegmented = segmentation_mask == 0
    segmentation_mask[wh_unsegmented] = segment_number_unassigned
    segmentation_mask[wh_below_threshold] = segment_number_below_threshold
    segmentation_out.data = segmentation_mask

    # add ncells to feature dataframe with new statistic method
    features_out = get_statistics(
        features_out,
        np.array(segmentation_out.data.copy()),
        np.array(field_in.data.copy()),
        statistic={"ncells": np.count_nonzero},
        default=0,
    )

    # compute additional statistics, if requested
    if statistic:
        features_out = get_statistics(
            features_out,
            segmentation_out.data.copy(),
            field_in.data.copy(),
            statistic=statistic,
        )

    return segmentation_out, features_out


def check_add_unseeded_across_bdrys(
    dim_to_run: str,
    segmentation_mask: np.array,
    unseeded_labels: np.array,
    border_min: int,
    border_max: int,
    markers_arr: np.array,
    inplace: bool = True,
) -> np.array:
    """Add new markers to unseeded but eligible regions when they are bordering
    an appropriate boundary.

    Parameters
    ----------
    dim_to_run:  {'hdim_1', 'hdim_2'}
        what dimension to run
    segmentation_mask: np.array
        the incomming segmentation mask
    unseeded_labels: np.array
        The list of labels that are unseeded
    border_min: int
        minimum real point in the dimension we are running on
    border_max: int
        maximum real point in the dimension we are running on (inclusive)
    markers_arr: np.array
        The array of markers to re-run segmentation with
    inplace: bool
        whether or not to modify markers_arr in place

    Returns
    -------
    markers_arr with new markers added

    """

    # if we are okay modifying the marker array inplace, do that
    if inplace:
        markers_out = markers_arr
    else:
        # If we can't modify the marker array inplace, make a deep copy.
        markers_out = copy.deepcopy(markers_arr)

    # identify border points and the loop points depending on what we want to run
    if dim_to_run == "hdim_1":
        border_axnum = 1
    elif dim_to_run == "hdim_2":
        border_axnum = 2
    # loop through vertical levels
    for border_ind, border_opposite in [
        (border_min, border_max),
        (border_max, border_min),
    ]:
        label_border_pts = np.take(unseeded_labels, border_ind, axis=border_axnum)
        seg_opp_pts = np.take(segmentation_mask, border_opposite, axis=border_axnum)
        if dim_to_run == "hdim_1":
            cond_to_check = np.logical_and(label_border_pts != 0, seg_opp_pts > 0)
            markers_out[:, border_ind, :][cond_to_check] = seg_opp_pts[cond_to_check]

        elif dim_to_run == "hdim_2":
            cond_to_check = np.logical_and(label_border_pts != 0, seg_opp_pts > 0)
            markers_out[:, :, border_ind][cond_to_check] = seg_opp_pts[cond_to_check]
    return markers_out


def segmentation(
    features: pd.DataFrame,
    field: iris.cube.Cube,
    dxy: float,
    threshold: float = 3e-3,
    target: Literal["maximum", "minimum"] = "maximum",
    level: Union[None, slice] = None,
    method: Literal["watershed"] = "watershed",
    max_distance: Union[None, float] = None,
    vertical_coord: Union[str, None] = None,
    PBC_flag: Literal["none", "hdim_1", "hdim_2", "both"] = "none",
    seed_3D_flag: Literal["column", "box"] = "column",
    seed_3D_size: Union[int, tuple[int]] = 5,
    segment_number_below_threshold: int = 0,
    segment_number_unassigned: int = 0,
    statistic: Union[dict[str, Union[Callable, tuple[Callable, dict]]], None] = None,
) -> tuple[iris.cube.Cube, pd.DataFrame]:
    """Use watershedding to determine region above a threshold
    value around initial seeding position for all time steps of
    the input data. Works both in 2D (based on single seeding
    point) and 3D and returns a mask with zeros everywhere around
    the identified regions and the feature id inside the regions.

    Calls segmentation_timestep at each individal timestep of the
    input data.

    Parameters
    ----------
    features : pandas.DataFrame
        Output from trackpy/maketrack.

    field : iris.cube.Cube
        Containing the field to perform the watershedding on.

    dxy : float
        Grid spacing of the input data in meters.

    threshold : float, optional
        Threshold for the watershedding field to be used for the mask.
        Default is 3e-3.

    target : {'maximum', 'minimum'}, optional
        Flag to determine if tracking is targetting minima or maxima in
        the data. Default is 'maximum'.

    level : slice of iris.cube.Cube, optional
        Levels at which to seed the cells for the watershedding
        algorithm. Default is None.

    method : {'watershed'}, optional
        Flag determining the algorithm to use (currently watershedding
        implemented). 'random_walk' could be uncommented.

    max_distance : float, optional
        Maximum distance from a marker allowed to be classified as
        belonging to that cell in meters. Default is None.

    vertical_coord : {'auto', 'z', 'model_level_number', 'altitude',
                      'geopotential_height'}, optional
        Name of the vertical coordinate for use in 3D segmentation case

    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}
        Sets whether to use periodic boundaries, and if so in which directions.
        'none' means that we do not have periodic boundaries
        'hdim_1' means that we are periodic along hdim1
        'hdim_2' means that we are periodic along hdim2
        'both' means that we are periodic along both horizontal dimensions

    seed_3D_flag: str('column', 'box')
        Seed 3D field at feature positions with either the full column (default)
         or a box of user-set size

    seed_3D_size: int or tuple (dimensions equal to dimensions of `field`)
        This sets the size of the seed box when `seed_3D_flag` is 'box'. If it's an
        integer (units of number of pixels), the seed box is identical in all dimensions.
        If it's a tuple, it specifies the seed area for each dimension separately, in units of pixels.
        Note: we strongly recommend the use of odd numbers for this. If you give
        an even number, your seed box will be biased and not centered
        around the feature.
        Note: if two seed boxes overlap, the feature that is seeded will be the
        closer feature.
    segment_number_below_threshold: int
        the marker to use to indicate a segmentation point is below the threshold.
    segment_number_unassigned: int
        the marker to use to indicate a segmentation point is above the threshold but unsegmented.
    statistic : dict, optional
        Default is None. Optional parameter to calculate bulk statistics within feature detection.
        Dictionary with callable function(s) to apply over the region of each detected feature and the name of the statistics to appear in the feature output dataframe. The functions should be the values and the names of the metric the keys (e.g. {'mean': np.mean})


    Returns
    -------
    segmentation_out : iris.cube.Cube
        Mask, 0 outside and integer numbers according to track
        inside the area/volume of the feature.

    features_out : pandas.DataFrame
        Feature dataframe including the number of cells (2D or 3D) in
        the segmented area/volume of the feature at the timestep.

    Raises
    ------
    ValueError
        If field_in.ndim is neither 3 nor 4 and 'time' is not included
        in coords.
    """
    import pandas as pd
    from iris.cube import CubeList

    logging.info("Start watershedding 3D")

    # check input for right dimensions:
    if not (field.ndim == 3 or field.ndim == 4):
        raise ValueError(
            "input to segmentation step must be 3D or 4D including a time dimension"
        )
    if "time" not in [coord.name() for coord in field.coords()]:
        raise ValueError(
            "input to segmentation step must include a dimension named 'time'"
        )

    # CubeList and list to store individual segmentation masks and feature DataFrames with information about segmentation
    segmentation_out_list = CubeList()
    features_out_list = []

    # loop over individual input timesteps for segmentation:
    # OR do segmentation on single timestep
    field_time = field.slices_over("time")

    for i, field_i in enumerate(field_time):
        time_i = field_i.coord("time").units.num2date(field_i.coord("time").points[0])
        features_i = features.loc[features["time"] == np.datetime64(time_i)]
        segmentation_out_i, features_out_i = segmentation_timestep(
            field_i,
            features_i,
            dxy,
            threshold=threshold,
            target=target,
            level=level,
            method=method,
            max_distance=max_distance,
            vertical_coord=vertical_coord,
            PBC_flag=PBC_flag,
            seed_3D_flag=seed_3D_flag,
            seed_3D_size=seed_3D_size,
            segment_number_unassigned=segment_number_unassigned,
            segment_number_below_threshold=segment_number_below_threshold,
            statistic=statistic,
        )
        segmentation_out_list.append(segmentation_out_i)
        features_out_list.append(features_out_i)
        logging.debug(
            "Finished segmentation for " + time_i.strftime("%Y-%m-%d_%H:%M:%S")
        )

    # Merge output from individual timesteps:
    segmentation_out = segmentation_out_list.merge_cube()
    features_out = pd.concat(features_out_list)

    logging.debug("Finished segmentation")
    return segmentation_out, features_out


def watershedding_3D(track, field_in, **kwargs):
    """Wrapper for the segmentation()-function."""
    kwargs.pop("method", None)
    return segmentation_3D(track, field_in, method="watershed", **kwargs)


def watershedding_2D(track, field_in, **kwargs):
    """Wrapper for the segmentation()-function."""
    kwargs.pop("method", None)
    return segmentation_2D(track, field_in, method="watershed", **kwargs)

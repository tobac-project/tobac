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

from __future__ import annotations
import copy
import logging
import datetime
import warnings

import iris.cube
import xarray as xr
from typing_extensions import Literal
from typing import Union, Callable, Optional

import skimage
import numpy as np
import pandas as pd

from tobac.utils import periodic_boundaries as pbc_utils
from tobac.utils import internal as internal_utils
from tobac.utils import get_statistics
from tobac.utils import decorators
from tobac.utils.generators import field_and_features_over_time
from tobac.utils.mask import convert_feature_mask_to_cells


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

    :hidden:

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

    if seed_3D_flag == "column":
        marker_arr[
            level if is_3D else None,
            features.hdim_1.to_numpy().round().astype(int) % h1_len,
            features.hdim_2.to_numpy().round().astype(int) % h2_len,
        ] = features.feature

    elif seed_3D_flag == "box":
        # Get the size of the seed box from the input parameter
        try:
            seed_z = seed_3D_size[0] if is_3D else 1
            seed_h1 = seed_3D_size[-2]
            seed_h2 = seed_3D_size[-1]
        except TypeError:
            # Not iterable, assume int.
            seed_z = seed_3D_size
            seed_h1 = seed_3D_size
            seed_h2 = seed_3D_size

        # Now we create a big list of points to assign seed boxes
        if is_3D:
            seed_box_offsets = np.meshgrid(
                np.arange(-seed_z / 2, seed_z / 2),
                np.arange(-seed_h1 / 2, seed_h1 / 2),
                np.arange(-seed_h2 / 2, seed_h2 / 2),
                indexing="ij",
            )
            seed_box_offset_z, seed_box_offset_h1, seed_box_offset_h2 = (
                sbo.ravel()[:, np.newaxis] for sbo in seed_box_offsets
            )
            seed_box_locs_z = np.ceil(
                (seed_box_offset_z + features.vdim.to_numpy())
            ).astype(int)
        else:
            seed_box_offsets = np.meshgrid(
                np.arange(-seed_h1 / 2, seed_h1 / 2),
                np.arange(-seed_h2 / 2, seed_h2 / 2),
                indexing="ij",
            )
            seed_box_offset_h1, seed_box_offset_h2 = (
                sbo.ravel()[:, np.newaxis] for sbo in seed_box_offsets
            )

        seed_box_locs_h1 = np.ceil(
            (seed_box_offset_h1 + features.hdim_1.to_numpy())
        ).astype(int)
        seed_box_locs_h2 = np.ceil(
            (seed_box_offset_h2 + features.hdim_2.to_numpy())
        ).astype(int)

        # Reorder to furthest first, closest last - this assigns preferring the closest features
        ordering = np.argsort(
            (
                (seed_box_locs_z - features.vdim.to_numpy()) ** 2
                if is_3D
                else 0
                + (seed_box_locs_h1 - features.hdim_1.to_numpy()) ** 2
                + (seed_box_locs_h2 - features.hdim_2.to_numpy()) ** 2
            ).ravel()
        )[::-1]

        if is_3D:
            seed_box_locs_z = seed_box_locs_z.flatten()[ordering]
            seed_box_locs_z = np.clip(seed_box_locs_z, 0, z_len - 1)
        seed_box_locs_h1 = seed_box_locs_h1.flatten()[ordering]
        seed_box_locs_h2 = seed_box_locs_h2.flatten()[ordering]
        marker_values = np.tile(
            features.feature.to_numpy()[np.newaxis, :], (seed_box_offset_h1.shape[0], 1)
        ).flatten()[ordering]

        if PBC_flag in ["hdim_1", "both"]:
            seed_box_locs_h1 = seed_box_locs_h1 % h1_len
        else:
            seed_box_locs_h1 = np.clip(seed_box_locs_h1, 0, h1_len - 1)

        if PBC_flag in ["hdim_2", "both"]:
            seed_box_locs_h2 = seed_box_locs_h2 % h2_len
        else:
            seed_box_locs_h2 = np.clip(seed_box_locs_h2, 0, h2_len - 1)

        # Now assign to the marker array:
        marker_arr[
            seed_box_locs_z if is_3D else None,
            seed_box_locs_h1,
            seed_box_locs_h2,
        ] = marker_values

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
    suppress_warnings=False,
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
        suppress_warnings=suppress_warnings,
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
    suppress_warnings=False,
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
        suppress_warnings=suppress_warnings,
    )


@decorators.iris_to_xarray()
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
    for both 2D and 3D data.

    :hidden:

    Parameters
    ----------
    field_in : xr.DataArray
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
    statistic : dict, optional
        Default is None. Optional parameter to calculate bulk statistics within segmentation.
        Dictionary with callable function(s) to apply over the region of each segmented feature and
        the name of the statistics to appear in the feature output dataframe.
        The functions should be the values and the names of the metric the keys (e.g. {'mean': np.mean})

    Returns
    -------
    segmentation_out : xarray.DataArray
        Mask, 0 outside and integer numbers according to track
        inside the objects.

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
        vertical_axis = internal_utils.find_vertical_coord_name(
            field_in, vertical_coord=vertical_coord
        )
        vertical_coord_axis = internal_utils.find_axis_from_coord(
            field_in, vertical_axis
        )
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
    segmentation_out = xr.zeros_like(field_in, dtype=int)
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
    # If none, use all levels (later reduced to the ones fulfilling the threshold conditions)
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
            markers = markers[np.newaxis, :, :]
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

            # z,y,x points in the grid domain with no transformations
            buddy_z = np.concatenate(
                [z_reg_inds[buddy] for buddy in buddies if buddy != 0]
            )
            buddy_y = np.concatenate(
                [y_reg_inds[buddy] for buddy in buddies if buddy != 0]
            )
            buddy_x = np.concatenate(
                [x_reg_inds[buddy] for buddy in buddies if buddy != 0]
            )

            # z,y,x points from the grid domain WHICH MAY OR MAY NOT BE TRANSFORMED
            # so as to be continuous/contiguous across a grid boundary for that dimension
            # (e.g., instead of [1496,1497,0,1,2,3] it would be [1496,1497,1498,1499,1500,1501])
            buddy_z2 = buddy_z
            # Vectorised logic from transfm_pbc_point
            buddy_y2 = np.where(
                buddy_y < ((hdim1_min + hdim1_max) / 2),
                buddy_y + hdim1_max + 1,
                buddy_y,
            )
            buddy_x2 = np.where(
                buddy_x < ((hdim2_min + hdim2_max) / 2),
                buddy_x + hdim2_max + 1,
                buddy_x,
            )
            # These are just for feature positions and are in z2/y2/x2 space
            # (may or may not be within real grid domain)
            # so that when the buddy box is constructed, seeding is done properly
            # in the buddy box space

            # NOTE: We may not need this, as we already do this editing the buddy_features df
            # and an iterrows call through this is what's used to actually seed the buddy box

            # Buddy Box!
            # Identify mins and maxes of Buddy Box continuous points range
            # so that box of correct size can be constructed
            bbox_zstart = int(np.min(buddy_z2))
            bbox_ystart = int(np.min(buddy_y2))
            bbox_xstart = int(np.min(buddy_x2))
            bbox_zend = int(np.max(buddy_z2) + 1)
            bbox_yend = int(np.max(buddy_y2) + 1)
            bbox_xend = int(np.max(buddy_x2) + 1)

            # Creation of actual Buddy Box space for transposition
            # of data in domain and re-seeding with Buddy feature markers
            buddy_locs = np.meshgrid(
                np.arange(bbox_zstart, bbox_zend, dtype=int),
                np.arange(bbox_ystart, bbox_yend, dtype=int) % (hdim1_max + 1),
                np.arange(bbox_xstart, bbox_xend, dtype=int) % (hdim2_max + 1),
                indexing="ij",
            )

            buddy_segmentation = data_segmentation[
                buddy_locs[0], buddy_locs[1], buddy_locs[2]
            ]

            unmasked_buddies = unmasked[buddy_locs[0], buddy_locs[1], buddy_locs[2]]

            buddy_markers = markers[buddy_locs[0], buddy_locs[1], buddy_locs[2]]
            buddy_markers = np.where(np.isin(buddy_markers, buddies), buddy_markers, 0)

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

            # overwrite IF:
            # 1) feature of interest
            # 2) changing to/from feature of interest or adjacent segmented feature
            # We don't want to overwrite other features that may be in the
            # buddy box if not contacting the intersected seg field
            seg_mask_3_subset = segmentation_mask_3[
                buddy_locs[0], buddy_locs[1], buddy_locs[2]
            ]
            segmentation_mask_3[buddy_locs[0], buddy_locs[1], buddy_locs[2]] = np.where(
                np.logical_and(
                    np.isin(seg_mask_3_subset, buddies),
                    np.isin(segmentation_mask_4, buddies),
                ),
                segmentation_mask_4,
                seg_mask_3_subset,
            )

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

    # add ncells to feature dataframe
    if len(features_out):
        features_out["ncells"] = np.bincount(
            np.maximum(segmentation_out.values.ravel(), 0),
            minlength=int(features_out.feature.max() + 1),
        )[features_out.feature.astype(int)]

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
    :hidden:
    Parameters
    ----------
    dim_to_run:  {'hdim_1', 'hdim_2'}
        what dimension to run
    segmentation_mask: np.array
        the incoming segmentation mask
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


@decorators.iris_to_xarray()
def segmentation(
    features: pd.DataFrame,
    field: xr.DataArray,
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
    time_padding: Optional[datetime.timedelta] = datetime.timedelta(seconds=0.5),
    suppress_warnings: Optional[bool] = False,
    return_cells: bool = False,
    stubs: Optional[int] = None,
) -> tuple[xr.DataArray, pd.DataFrame]:
    """Use watershedding to determine region above a threshold
    value around initial seeding position for all time steps of
    the input data. Works both in 2D (based on single seeding
    point) and 3D and returns a mask with zeros everywhere around
    the identified regions and the feature id inside the regions.

    Calls segmentation_timestep at each individual timestep of the
    input data.

    Parameters
    ----------
    features : pandas.DataFrame
        Output from trackpy/maketrack.

    field : iris.cube.Cube or xarray.DataArray
        Containing the field to perform the watershedding on.

    dxy : float
        Grid spacing of the input data in meters.

    threshold : float, optional
        Threshold for the watershedding field to be used for the mask.
        Default is 3e-3.

    target : {'maximum', 'minimum'}, optional
        Flag to determine if tracking is targeting minima or maxima in
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
    time_padding: timedelta, optional
        If set, allows for segmentation to be associated with a feature input
        timestep that is time_padding off of the feature. Extremely useful when
        converting between micro- and nanoseconds, as is common when using Pandas
        dataframes.
    suppress_warnings: bool, optional
        If True, suppresses warnings. Default is False.
    return_cells: bool, optional (default: False)
        If True, the segmentation mask returned will use the cell values of the
        input dataframe, rather than the feature values. This requires the
        features input to be the output from tobac.linking_trackpy
    stubs: int, optional (default: None)
        The stub values used for unlinked cells in tobac.linking_trackpy, used
        when return_cells=True If None, the stub cells with be relabelled with
        the stub cell value in the feature dataframe. If a value is provided,
        the masked regions corresponding to stub cells with be removed from the
        output. Warning: the presence of stub cells may make it impossible to
        perfectly reconstruct the feature mask afterwards as any stub features
        will be removed.

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

    time_var_name: str = "time"
    seg_out_type: str = "int64"

    logging.info("Start watershedding")

    # check input for right dimensions:
    if not (field.ndim == 3 or field.ndim == 4):
        raise ValueError(
            "input to segmentation step must be 3D or 4D including a time dimension"
        )
    try:
        ndim_time = internal_utils.find_axis_from_coord(field, time_var_name)
    except ValueError as exc:
        raise ValueError(
            "input to segmentation step must include a dimension named '{0}'".format(
                time_var_name
            )
        ) from exc

    # Check features has cell column if return_cells is True:
    if return_cells and "cell" not in features.columns:
        raise ValueError(
            "`cell` column not found in features input, please perform tracking on this data before performing segmentation with `return_cells=True`"
        )

    # create our output dataarray
    segmentation_out_data = xr.DataArray(
        np.zeros(field.shape, dtype=int),
        coords=field.coords,
        dims=field.dims,
        name="segmentation_mask",
    ).assign_attrs(threshold=threshold)

    if features is None or features.empty:
        logging.debug(
            "No features provided; returning empty segmentation mask and empty features DataFrame"
        )
        return segmentation_out_data, features

    features_out_list = []

    if not suppress_warnings and len(field.coords[time_var_name]) == 1:
        warnings.warn(
            "As of v1.6.0, segmentation with time length 1 will return time as a coordinate"
            " instead of dropping it (i.e., output will now be 1xMxN instead of MxN). ",
            UserWarning,
        )

    for (
        time_iteration_number,
        time_iteration_value,
        field_at_time,
        features_i,
    ) in field_and_features_over_time(
        field, features, time_var_name=time_var_name, time_padding=time_padding
    ):
        segmentation_out_i, features_out_i = segmentation_timestep(
            field_at_time,
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
        segmentation_out_data.loc[{time_var_name: time_iteration_value}] = (
            segmentation_out_i
        )
        features_out_list.append(features_out_i)
        logging.debug(f"Finished segmentation for {time_iteration_value.values}")

    # Merge output from individual timesteps:
    features_out = pd.concat(features_out_list)

    # Convert feature mask to cells if return_cells is True:
    if return_cells:
        segmentation_out_data = convert_feature_mask_to_cells(
            features_out,
            segmentation_out_data,
            stubs=stubs,
            inplace=True,
        )

    logging.debug("Finished segmentation")
    return segmentation_out_data, features_out


def watershedding_3D(track, field_in, **kwargs):
    """Wrapper for the segmentation()-function."""
    kwargs.pop("method", None)
    return segmentation_3D(track, field_in, method="watershed", **kwargs)


def watershedding_2D(track, field_in, **kwargs):
    """Wrapper for the segmentation()-function."""
    kwargs.pop("method", None)
    return segmentation_2D(track, field_in, method="watershed", **kwargs)

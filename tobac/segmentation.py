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

import logging
import numpy as np

import skimage
import numpy as np

from . import utils as tb_utils
from .utils import internal as internal_utils


def add_markers(features, marker_arr, seed_3D_flag, seed_3D_size=5, level=None):
    """Adds markers for watershedding using the `features` dataframe
    to the marker_arr.

    Parameters
    ----------
    features: pandas.DataFrame
        Features for one point in time to add as markers.
    marker_arr: 2D or 3D array-like
        Array to add the markers to. Assumes a (z, h1, h2) or (h1, h2) configuration.
    seed_3D_flag: str('column', 'box')
        Seed 3D field at feature positions with either the full column
         or a box of user-set size
    seed_3D_size: int or tuple (dimensions equal to dimensions of `field`)
        This sets the size of the seed box when `seed_3D_flag` is 'box'. If it's an
        integer, the seed box is identical in all dimensions. If it's a tuple, it specifies the
        seed area for each dimension separately.
        Note: we recommend the use of odd numbers for this. If you give
        an even number, your seed box will be biased and not centered
        around the feature.
        Note: if two seed boxes overlap, the feature that is seeded will be the
        closer feature.
    level: slice or None
        If `seed_3D_flag` is 'column', the levels at which to seed the
        cells for the watershedding algorithm. If None, seeds all levels.

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

    else:
        is_3D = False
        z_len = 0
        # transpose to 3D array to make things easier.
        marker_arr = marker_arr[np.newaxis, :, :]

    if seed_3D_flag == "column":
        for index, row in features.iterrows():
            marker_arr[level, int(row["hdim_1"]), int(row["hdim_2"])] = row["feature"]

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

        for index, row in features.iterrows():
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

            # this is simple- just go in the seed_z/2 points around the
            # vdim of the feature, up to the limits of the array.
            if is_3D:
                z_seed_start = int(np.max([0, np.ceil(row["vdim"] - seed_z / 2)]))
                z_seed_end = int(np.min([z_len, np.ceil(row["vdim"] + seed_z / 2)]))

            hdim_1_min = int(np.ceil(row["hdim_1"] - seed_h1 / 2))
            hdim_1_max = int(np.ceil(row["hdim_1"] + seed_h1 / 2))
            hdim_2_min = int(np.ceil(row["hdim_2"] - seed_h2 / 2))
            hdim_2_max = int(np.ceil(row["hdim_2"] + seed_h2 / 2))
            seed_box = [hdim_1_min, hdim_1_max, hdim_2_min, hdim_2_max]
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

                        dist_from_curr_pt = internal_utils.calc_distance_coords(
                            np.array(global_index),
                            np.array(curr_coord),
                        )

                        # This is technically an O(N^2) operation, but
                        # hopefully performance isn't too bad as this should
                        # be rare.
                        orig_row = features[features["feature"] == curr_box_pt].iloc[0]
                        if is_3D:
                            orig_coord = (
                                orig_row["vdim"],
                                orig_row["hdim_1"],
                                orig_row["hdim_2"],
                            )
                        else:
                            orig_coord = (0, orig_row["hdim_1"], orig_row["hdim_2"])
                        dist_from_orig_pt = internal_utils.calc_distance_coords(
                            np.array(global_index),
                            np.array(orig_coord),
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
    seed_3D_flag="column",
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
        seed_3D_flag=seed_3D_flag,
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
    seed_3D_flag="column",
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
        seed_3D_flag=seed_3D_flag,
    )


def segmentation_timestep(
    field_in,
    features_in,
    dxy,
    threshold=3e-3,
    target="maximum",
    level=None,
    method="watershed",
    max_distance=None,
    vertical_coord="auto",
    seed_3D_flag="column",
    seed_3D_size=5,
):
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
        Threshold for the watershedding field to be used for the mask.
        Default is 3e-3.

    target : {'maximum', 'minimum'}, optional
        Flag to determine if tracking is targetting minima or maxima in
        the data to determine from which direction to approach the threshold
        value. Default is 'maximum'.

    level : slice of iris.cube.Cube, optional
        Levels at which to seed the cells for the watershedding
        algorithm. Default is None.

    method : {'watershed'}, optional
        Flag determining the algorithm to use (currently watershedding
        implemented). 'random_walk' could be uncommented.

    max_distance : float, optional
        Maximum distance from a marker allowed to be classified as
        belonging to that cell. Default is None.

    vertical_coord : str, optional
        Vertical coordinate in 3D input data. If 'auto', input is checked for
        one of {'z', 'model_level_number', 'altitude','geopotential_height'}
        as a likely coordinate name
    seed_3D_flag: str('column', 'box')
        Seed 3D field at feature positions with either the full column (default)
         or a box of user-set size
    seed_3D_size: int or tuple (dimensions equal to dimensions of `field`)
        This sets the size of the seed box when `seed_3D_flag` is 'box'. If it's an
        integer, the seed box is identical in all dimensions. If it's a tuple, it specifies the
        seed area for each dimension separately. Note: we recommend the use
        of odd numbers for this. If you give an even number, your seed box will be
        biased and not centered around the feature.


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

    # How many dimensions are we using?
    if field_in.ndim == 2:
        hdim_1_axis = 0
        hdim_2_axis = 1
    elif field_in.ndim == 3:
        vertical_axis = internal_utils.find_vertical_axis_from_coord(
            field_in, vertical_coord=vertical_coord
        )
        ndim_vertical = field_in.coord_dims(vertical_axis)
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

    # copy feature dataframe for output
    features_out = deepcopy(features_in)
    # Create cube of the same dimensions and coordinates as input data to store mask:
    segmentation_out = 1 * field_in
    segmentation_out.rename("segmentation_mask")
    segmentation_out.units = 1

    # Get raw array from input data:
    data = field_in.core_data()
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
    if level == None:
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
    markers = add_markers(features_in, markers, seed_3D_flag, seed_3D_size, level)
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
        D = distance_transform_edt((markers == 0).astype(int))
        segmentation_mask[
            np.bitwise_and(segmentation_mask > 0, D > max_distance_pixel)
        ] = 0

    # mask all segmentation_mask points below threshold as -1
    # to differentiate from those unmasked points NOT filled by watershedding
    # TODO: allow user to specify
    segmentation_mask[~unmasked] = -1

    if transposed_data:
        if vertical_coord_axis == 1:
            segmentation_mask = np.transpose(segmentation_mask, axes=(1, 0, 2))
        elif vertical_coord_axis == 2:
            segmentation_mask = np.transpose(segmentation_mask, axes=(1, 2, 0))

    # Write resulting mask into cube for output
    segmentation_out.data = segmentation_mask

    # count number of grid cells associated to each tracked cell and write that into DataFrame:
    values, count = np.unique(segmentation_mask, return_counts=True)
    counts = dict(zip(values, count))
    ncells = np.zeros(len(features_out))
    for i, (index, row) in enumerate(features_out.iterrows()):
        if row["feature"] in counts.keys():
            # assign a value for ncells for the respective feature in data frame
            features_out.loc[features_out.feature == row["feature"], "ncells"] = counts[
                row["feature"]
            ]

    return segmentation_out, features_out


def segmentation(
    features,
    field,
    dxy,
    threshold=3e-3,
    target="maximum",
    level=None,
    method="watershed",
    max_distance=None,
    vertical_coord="auto",
    seed_3D_flag="column",
    seed_3D_size=5,
):
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
        Grid spacing of the input data.

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
        belonging to that cell. Default is None.

    vertical_coord : {'auto', 'z', 'model_level_number', 'altitude',
                      'geopotential_height'}, optional
        Name of the vertical coordinate for use in 3D segmentation case

    seed_3D_flag: str('column', 'box')
        Seed 3D field at feature positions with either the full column (default)
         or a box of user-set size

    seed_3D_size: int or tuple (dimensions equal to dimensions of `field`)
        This sets the size of the seed box when `seed_3D_flag` is 'box'. If it's an
        integer, the seed box is identical in all dimensions. If it's a tuple, it specifies the
        seed area for each dimension separately. Note: we recommend the use
        of odd numbers for this. If you give an even number, your seed box will be
        biased and not centered around the feature.


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
            seed_3D_flag=seed_3D_flag,
            seed_3D_size=seed_3D_size,
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

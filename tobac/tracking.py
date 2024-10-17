"""Provide tracking methods.

The individual features and associated area/volumes identified in
each timestep have to be linked into trajectories to analyse
the time evolution of their properties for a better understanding of
the underlying physical processes.
The implementations are structured in a way that allows for the future
addition of more complex tracking methods recording a more complex
network of relationships between features at different points in
time.

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
from typing import Optional, Literal, Union

import logging
from operator import is_
import numpy as np
import pandas as pd
import warnings
import math
from . import utils as tb_utils
from .utils import periodic_boundaries as pbc_utils
from .utils import internal as internal_utils

from packaging import version as pkgvsn
import trackpy as tp
import trackpy.linking
from copy import deepcopy


def linking_trackpy(
    features: pd.DataFrame,
    field_in: None,
    dt: float,
    dxy: float,
    dz: Optional[float] = None,
    v_max: Optional[float] = None,
    d_max: Optional[float] = None,
    d_min: Optional[float] = None,
    subnetwork_size: Optional[int] = None,
    memory: int = 0,
    stubs: int = 1,
    time_cell_min: Optional[float] = None,
    order: int = 1,
    extrapolate: int = 0,
    method_linking: Literal["random", "predict"] = "random",
    adaptive_step: Optional[float] = None,
    adaptive_stop: Optional[float] = None,
    cell_number_start: int = 1,
    cell_number_unassigned: int = -1,
    vertical_coord: str = "auto",
    min_h1: Optional[int] = None,
    max_h1: Optional[int] = None,
    min_h2: Optional[int] = None,
    max_h2: Optional[int] = None,
    PBC_flag: Literal["none", "hdim_1", "hdim_2", "both"] = "none",
    features_append: Union[None, pd.DataFrame] = None,
) -> pd.DataFrame:
    """Perform Linking of features in trajectories.

    The linking determines which of the features detected in a specific
    timestep is most likely identical to an existing feature in the
    previous timestep. For each existing feature, the movement within
    a time step is extrapolated based on the velocities in a number
    previous time steps. The algorithm then breaks the search process
    down to a few candidate features by restricting the search to a
    circular search region centered around the predicted position of
    the feature in the next time step. For newly initialized trajectories,
    where no velocity from previous time steps is available, the
    algorithm resorts to the average velocity of the nearest tracked
    objects. v_max and d_min are given as physical quantities and then
    converted into pixel-based values used in trackpy. This allows for
    tracking that is controlled by physically-based parameters that are
    independent of the temporal and spatial resolution of the input
    data. The algorithm creates a continuous track for the feature
    that is the most probable based on the previous cell path.

    Parameters
    ----------
    features : pandas.DataFrame
        Detected features to be linked.

    features_append : pandas.DataFrame
        New features to be tracked. This dataframe can overlap with the features dataframe
        features, but only times where there is no cell information (either `cell` column for that
        time are all equal to cell_number_unassigned or there is no cell column) for the
        *whole time* are considered.
        If this is not None, features must have a cell column.

    dt : float
        Time resolution of tracked features in seconds.

    dxy : float
        Horizontal grid spacing of the input data in meters.

    dz : float
        Constant vertical grid spacing (meters), optional. If not specified
        and the input is 3D, this function requires that `vertical_coord` is available
        in the `features` input. If you specify a value here, this function assumes
        that it is the constant z spacing between points, even if ```vertical_coord```
        is specified.

    v_max : float, optional
        Speed at which features are allowed to move in meters per second.
        Only one of `d_max`, `d_min`, or `v_max` can be set.
        Default is None.

    d_max : float, optional
        Maximum search range in meters. Only one of `d_max`, `d_min`, or `v_max` can be set.
        Default is None.

    d_min : float, optional
        Deprecated. Only one of `d_max`, `d_min`, or `v_max` can be set.
        Default is None.

    subnetwork_size : int, optional
        Maximum size of subnetwork for linking. This parameter should be
        adjusted when using adaptive search. Usually a lower value is desired
        in that case. For a more in depth explanation have look
        `here <https://soft-matter.github.io/trackpy/v0.5.0/tutorial/adaptive-search.html>`_
        If None, 30 is used for regular search and 15 for adaptive search.
        Default is None.


    memory : int, optional
        Number of output timesteps features allowed to vanish for to
        be still considered tracked. Default is 0.
        .. warning :: This parameter should be used with caution, as it
                     can lead to erroneous trajectory linking,
                     espacially for data with low time resolution.

    stubs : int, optional
        Minimum number of timesteps of a tracked cell to be reported
        Default is 1

    time_cell_min : float, optional
        Minimum length in time that a cell must be tracked for to be considered a
        valid cell in seconds.
        Default is None.

    order : int, optional
        Order of polynomial used to extrapolate trajectory into gaps and
        ond start and end point.
        Default is 1.

    extrapolate : int, optional
        Number or timesteps to extrapolate trajectories. Currently unused.
        Default is 0.

    method_linking : {'random', 'predict'}, optional
        Flag choosing method used for trajectory linking.
        Default is 'random', although we typically encourage users to use 'predict'.

    adaptive_step : float, optional
        Reduce search range by multiplying it by this factor. Needs to be
        used in combination with adaptive_stop. Default is None.

    adaptive_stop : float, optional
        If not None, when encountering an oversize subnet, retry by progressively
        reducing search_range by multiplying with adaptive_step until the subnet
        is solvable. If search_range becomes <= adaptive_stop, give up and raise
        a SubnetOversizeException. Needs to be used in combination with
        adaptive_step. Default is None.

    cell_number_start : int, optional
        Cell number for first tracked cell.
        Default is 1

    cell_number_unassigned: int
        Number to set the unassigned/non-tracked cells to. Note that if you set this
        to `np.nan`, the data type of 'cell' will change to float.
        Default is -1

    vertical_coord: str
        Name of the vertical coordinate. The vertical coordinate used
        must be meters. If None, tries to auto-detect.
        It looks for the coordinate or the dimension name corresponding
        to the string. To use `dz`, set this to `None`.

    min_h1: int
        Minimum hdim_1 value, required when PBC_flag is 'hdim_1' or 'both'

    max_h1: int
        Maximum hdim_1 value, required when PBC_flag is 'hdim_1' or 'both'

    min_h2: int
        Minimum hdim_2 value, required when PBC_flag is 'hdim_2' or 'both'

    max_h2: int
        Maximum hdim_2 value, required when PBC_flag is 'hdim_2' or 'both'

    PBC_flag : str('none', 'hdim_1', 'hdim_2', 'both')
        Sets whether to use periodic boundaries, and if so in which directions.
        'none' means that we do not have periodic boundaries
        'hdim_1' means that we are periodic along hdim1
        'hdim_2' means that we are periodic along hdim2
        'both' means that we are periodic along both horizontal dimensions
    field_in : None
        Input field. Not currently used; can be set to `None`.

    Returns
    -------
    trajectories_final : pandas.DataFrame
        Dataframe of the linked features, containing the variable 'cell',
        with integers indicating the affiliation of a feature to a specific
        track, and the variable 'time_cell' with the time the cell has
        already existed.

    Raises
    ------
    ValueError
        If method_linking is neither 'random' nor 'predict'.
    """

    if extrapolate != 0:
        raise NotImplementedError(
            "Extrapolation is not yet implemented. Set this parameter to 0 to continue."
        )

    search_range = _calc_search_range(dt, dxy, v_max=v_max, d_max=d_max, d_min=d_min)
    # Check if we are 3D.
    is_3D, found_vertical_coord = _get_vertical_coord(
        features, dz=dz, vertical_coord=vertical_coord
    )

    _ = _check_pbc_coords(PBC_flag, min_h1, max_h1, min_h2, max_h2)
    size_cache = _check_set_adaptive_params(
        adaptive_stop=adaptive_stop,
        adaptive_step=adaptive_step,
        subnetwork_size=subnetwork_size,
    )

    stubs = _calc_frames_for_stubs(dt, stubs=stubs, time_cell_min=time_cell_min)

    logging.debug("stubs: " + str(stubs))

    logging.debug("start linking features into trajectories")

    # deep copy to preserve features field:
    features_linking = deepcopy(features)

    # check if we are 3D or not
    if is_3D:
        # If we are 3D, we need to convert the vertical
        # coordinates so that 1 unit is equal to dxy.

        if dz is not None:
            features_linking["vdim_adj"] = features_linking["vdim"] * dz / dxy
        else:
            features_linking["vdim_adj"] = features_linking[found_vertical_coord] / dxy

        pos_columns_tp = ["vdim_adj", "hdim_1", "hdim_2"]

    else:
        pos_columns_tp = ["hdim_1", "hdim_2"]

    # Check if we have PBCs.
    if PBC_flag in ["hdim_1", "hdim_2", "both"]:
        # Per the trackpy docs, to specify a custom distance function
        # which we need for PBCs, neighbor_strategy must be 'BTree'.
        # I think this shouldn't change results, but it will degrade performance.
        neighbor_strategy = "BTree"
        dist_func = pbc_utils.build_distance_function(
            min_h1, max_h1, min_h2, max_h2, PBC_flag, is_3D
        )

    else:
        neighbor_strategy = "KDTree"
        dist_func = None

    if method_linking == "random":
        #     link features into trajectories:
        trajectories_unfiltered = tp.link(
            features_linking,
            search_range=search_range,
            memory=memory,
            t_column="frame",
            pos_columns=pos_columns_tp,
            adaptive_step=adaptive_step,
            adaptive_stop=adaptive_stop,
            neighbor_strategy=neighbor_strategy,
            link_strategy="auto",
            dist_func=dist_func,
        )
    elif method_linking == "predict":
        if is_3D and pkgvsn.parse(tp.__version__) < pkgvsn.parse("0.6.0"):
            raise ValueError(
                "3D Predictive Tracking Only Supported with trackpy versions newer than 0.6.0."
            )

        # avoid setting pos_columns by renaming to default values to avoid trackpy bug
        features_linking.rename(
            columns={
                "y": "__temp_y_coord",
                "x": "__temp_x_coord",
                "z": "__temp_z_coord",
            },
            inplace=True,
        )

        features_linking.rename(
            columns={"hdim_1": "y", "hdim_2": "x", "vdim_adj": "z"}, inplace=True
        )

        # generate list of features as input for df_link_iter to avoid bug in df_link
        features_linking_list = [
            frame for i, frame in features_linking.groupby("frame", sort=True)
        ]

        pred = tp.predict.NearestVelocityPredict(span=1)
        trajectories_unfiltered = pred.link_df_iter(
            features_linking_list,
            search_range=search_range,
            memory=memory,
            # pos_columns=["hdim_1", "hdim_2"], # not working atm
            t_column="frame",
            neighbor_strategy=neighbor_strategy,
            link_strategy="auto",
            adaptive_step=adaptive_step,
            adaptive_stop=adaptive_stop,
            dist_func=dist_func,
            #                                 copy_features=False, diagnostics=False,
            #                                 hash_size=None, box_size=None, verify_integrity=True,
            #                                 retain_index=False
        )
        # recreate a single dataframe from the list

        trajectories_unfiltered = pd.concat(trajectories_unfiltered)

        # change to column names back
        trajectories_unfiltered.rename(
            columns={"y": "hdim_1", "x": "hdim_2", "z": "vdim_adj"}, inplace=True
        )
        trajectories_unfiltered.rename(
            columns={
                "__temp_y_coord": "y",
                "__temp_x_coord": "x",
                "__temp_z_coord": "z",
            },
            inplace=True,
        )

    else:
        raise ValueError("method_linking unknown")

    # Reset trackpy parameters to previously set values
    if subnetwork_size is not None:
        if adaptive_step is None and adaptive_stop is None:
            tp.linking.Linker.MAX_SUB_NET_SIZE = size_cache
        else:
            tp.linking.Linker.MAX_SUB_NET_SIZE_ADAPTIVE = size_cache

    # Filter trajectories to exclude short trajectories that are likely to be spurious
    #    trajectories_filtered = filter_stubs(trajectories_unfiltered,threshold=stubs)
    #    trajectories_filtered=trajectories_filtered.reset_index(drop=True)

    # clean up our temporary filters
    if is_3D:
        trajectories_unfiltered = trajectories_unfiltered.drop("vdim_adj", axis=1)

    # Reset particle numbers from the arbitray numbers at the end of the feature detection and linking to consecutive cell numbers
    # keep 'particle' for reference to the feature detection step.
    trajectories_unfiltered["cell"] = None
    particle_num_to_cell_num = dict()
    for i_particle, particle in enumerate(
        pd.Series.unique(trajectories_unfiltered["particle"])
    ):
        cell = int(i_particle + cell_number_start)
        particle_num_to_cell_num[particle] = int(cell)
    remap_particle_to_cell_vec = np.vectorize(remap_particle_to_cell_nv)
    trajectories_unfiltered["cell"] = remap_particle_to_cell_vec(
        particle_num_to_cell_num, trajectories_unfiltered["particle"]
    )
    trajectories_unfiltered["cell"] = trajectories_unfiltered["cell"].astype(int)
    trajectories_unfiltered.drop(columns=["particle"], inplace=True)

    trajectories_bycell = trajectories_unfiltered.groupby("cell")
    stub_cell_nums = list()
    for cell, trajectories_cell in trajectories_bycell:
        # logging.debug("cell: "+str(cell))
        # logging.debug("feature: "+str(trajectories_cell['feature'].values))
        # logging.debug("trajectories_cell.shape[0]: "+ str(trajectories_cell.shape[0]))

        if trajectories_cell.shape[0] < stubs:
            logging.debug(
                "cell"
                + str(cell)
                + "  is a stub ("
                + str(trajectories_cell.shape[0])
                + "), setting cell number to "
                + str(cell_number_unassigned)
            )
            stub_cell_nums.append(cell)

    trajectories_unfiltered.loc[
        trajectories_unfiltered["cell"].isin(stub_cell_nums), "cell"
    ] = cell_number_unassigned

    trajectories_filtered = trajectories_unfiltered

    trajectories_filtered_filled = deepcopy(trajectories_filtered)

    trajectories_final = add_cell_time(
        trajectories_filtered_filled, cell_number_unassigned=cell_number_unassigned
    )
    # Add metadata
    trajectories_final.attrs["cell_number_unassigned"] = cell_number_unassigned

    # add coordinate to raw features identified:
    logging.debug("start adding coordinates to detected features")
    logging.debug("feature linking completed")
    return trajectories_final


def append_tracks_trackpy(
    tracks_orig: pd.DataFrame,
    new_features: pd.DataFrame,
    dt: float,
    dxy: float,
    dz: Optional[float] = None,
    v_max: Optional[float] = None,
    d_max: Optional[float] = None,
    d_min: Optional[float] = None,
    subnetwork_size: Optional[int] = None,
    memory: int = 0,
    stubs: int = 1,
    time_cell_min: Optional[float] = None,
    order: int = 1,
    extrapolate: int = 0,
    method_linking: Literal["random", "predict"] = "random",
    adaptive_step: Optional[float] = None,
    adaptive_stop: Optional[float] = None,
    cell_number_start: int = 1,
    cell_number_unassigned: int = -1,
    vertical_coord: str = "auto",
    min_h1: Optional[int] = None,
    max_h1: Optional[int] = None,
    min_h2: Optional[int] = None,
    max_h2: Optional[int] = None,
    PBC_flag: Literal["none", "hdim_1", "hdim_2", "both"] = "none",
) -> pd.DataFrame:
    """Append a new feature dataframe onto an existing tracked dataframe using the same logic as
    tracking.linking_trackpy.

    Parameters
    ----------
    tracks_orig: pd.DataFrame
        Original tracked file. Must contain a 'cell' column.
    new_features: pd.DataFrame
        New features to be tracked. This dataframe can overlap with the tracks_orig dataframe
        features, but only times where there is no cell information (either `cell` column for that
        time are all equal to cell_number_unassigned or there is no cell column) for the
        *whole time* are considered.
    dt : float
        Time resolution of tracked features in seconds.

    dxy : float
        Horizontal grid spacing of the input data in meters.

    dz : float
        Constant vertical grid spacing (meters), optional. If not specified
        and the input is 3D, this function requires that `vertical_coord` is available
        in the `features` input. If you specify a value here, this function assumes
        that it is the constant z spacing between points, even if ```vertical_coord```
        is specified.

    v_max : float, optional
        Speed at which features are allowed to move in meters per second.
        Only one of `d_max`, `d_min`, or `v_max` can be set.
        Default is None.

    d_max : float, optional
        Maximum search range in meters. Only one of `d_max`, `d_min`, or `v_max` can be set.
        Default is None.

    d_min : float, optional
        Deprecated. Only one of `d_max`, `d_min`, or `v_max` can be set.
        Default is None.

    subnetwork_size : int, optional
        Maximum size of subnetwork for linking. This parameter should be
        adjusted when using adaptive search. Usually a lower value is desired
        in that case. For a more in depth explanation have look
        `here <https://soft-matter.github.io/trackpy/v0.5.0/tutorial/adaptive-search.html>`_
        If None, 30 is used for regular search and 15 for adaptive search.
        Default is None.


    memory : int, optional
        Number of output timesteps features allowed to vanish for to
        be still considered tracked. Default is 0.
        .. warning :: This parameter should be used with caution, as it
                     can lead to erroneous trajectory linking,
                     espacially for data with low time resolution.

    stubs : int, optional
        Minimum number of timesteps of a tracked cell to be reported
        Default is 1

    time_cell_min : float, optional
        Minimum length in time that a cell must be tracked for to be considered a
        valid cell in seconds.
        Default is None.

    order : int, optional
        Order of polynomial used to extrapolate trajectory into gaps and
        ond start and end point.
        Default is 1.

    extrapolate : int, optional
        Number or timesteps to extrapolate trajectories. Currently unused.
        Default is 0.

    method_linking : {'random', 'predict'}, optional
        Flag choosing method used for trajectory linking.
        Default is 'random', although we typically encourage users to use 'predict'.

    adaptive_step : float, optional
        Reduce search range by multiplying it by this factor. Needs to be
        used in combination with adaptive_stop. Default is None.

    adaptive_stop : float, optional
        If not None, when encountering an oversize subnet, retry by progressively
        reducing search_range by multiplying with adaptive_step until the subnet
        is solvable. If search_range becomes <= adaptive_stop, give up and raise
        a SubnetOversizeException. Needs to be used in combination with
        adaptive_step. Default is None.

    cell_number_start : int, optional
        Cell number for first tracked cell.
        Default is 1

    cell_number_unassigned: int
        Number to set the unassigned/non-tracked cells to. Note that if you set this
        to `np.nan`, the data type of 'cell' will change to float.
        Default is -1

    vertical_coord: str
        Name of the vertical coordinate. The vertical coordinate used
        must be meters. If None, tries to auto-detect.
        It looks for the coordinate or the dimension name corresponding
        to the string. To use `dz`, set this to `None`.

    min_h1: int
        Minimum hdim_1 value, required when PBC_flag is 'hdim_1' or 'both'

    max_h1: int
        Maximum hdim_1 value, required when PBC_flag is 'hdim_1' or 'both'

    min_h2: int
        Minimum hdim_2 value, required when PBC_flag is 'hdim_2' or 'both'

    max_h2: int
        Maximum hdim_2 value, required when PBC_flag is 'hdim_2' or 'both'

    PBC_flag : str('none', 'hdim_1', 'hdim_2', 'both')
        Sets whether to use periodic boundaries, and if so in which directions.
        'none' means that we do not have periodic boundaries
        'hdim_1' means that we are periodic along hdim1
        'hdim_2' means that we are periodic along hdim2
        'both' means that we are periodic along both horizontal dimensions

    Returns
    -------
    trajectories_final : pandas.DataFrame
        Dataframe of the linked features, containing the variable 'cell',
        with integers indicating the affiliation of a feature to a specific
        track, and the variable 'time_cell' with the time the cell has
        already existed.

    Raises
    ------
    ValueError
        If method_linking is neither 'random' nor 'predict'.

    """
    if "cell" not in tracks_orig:
        raise ValueError("Need to have existing tracks.")

    if memory > 0:
        pass
        # raise NotImplementedError("Append tracks with memory not yet implemented.")

    search_range = _calc_search_range(dt, dxy, v_max=v_max, d_max=d_max, d_min=d_min)

    if ("vdim" in tracks_orig) is not ("vdim" in new_features):
        raise ValueError(
            "One track is 3D, new track is 2D. Need to both have the same dimensions."
        )

    is_3D, found_vertical_coord = _get_vertical_coord(
        tracks_orig, dz=dz, vertical_coord=vertical_coord
    )

    _ = _check_pbc_coords(PBC_flag, min_h1, max_h1, min_h2, max_h2)
    size_cache = _check_set_adaptive_params(
        adaptive_stop=adaptive_stop,
        adaptive_step=adaptive_step,
        subnetwork_size=subnetwork_size,
    )
    stubs = _calc_frames_for_stubs(dt, stubs=stubs, time_cell_min=time_cell_min)

    logging.debug("stubs: " + str(stubs))

    logging.debug("start linking features into trajectories")

    tracks_orig_cleaned, tracks_cut, new_features_cleaned = _clean_track_dfs_for_append(
        tracks_orig, new_features, memory, dt
    )
    # drop time_cell if it's there.

    tracks_orig_cleaned.drop("time_cell", axis=1, inplace=True)
    tracks_cut.drop("time_cell", axis=1, inplace=True)
    if is_3D:
        pos_columns_tp = ["vdim_adj", "hdim_1", "hdim_2"]
    else:
        pos_columns_tp = ["hdim_1", "hdim_2"]

    # check if we are 3D or not
    if is_3D:
        # If we are 3D, we need to convert the vertical
        # coordinates so that 1 unit is equal to dxy.

        if dz is not None:
            tracks_orig_cleaned["vdim_adj"] = tracks_orig_cleaned["vdim"] * dz / dxy
            new_features_cleaned["vdim_adj"] = new_features_cleaned["vdim"] * dz / dxy

        else:
            tracks_orig_cleaned["vdim_adj"] = (
                tracks_orig_cleaned[found_vertical_coord] / dxy
            )
            new_features_cleaned["vdim_adj"] = (
                new_features_cleaned[found_vertical_coord] / dxy
            )

    # Check if we have PBCs.
    if PBC_flag in ["hdim_1", "hdim_2", "both"]:
        # Per the trackpy docs, to specify a custom distance function
        # which we need for PBCs, neighbor_strategy must be 'BTree'.
        # I think this shouldn't change results, but it will degrade performance.
        neighbor_strategy = "BTree"
        dist_func = build_distance_function(min_h1, max_h1, min_h2, max_h2, PBC_flag)

    else:
        neighbor_strategy = "KDTree"
        dist_func = None

    if method_linking == "predict":
        if is_3D and pkgvsn.parse(tp.__version__) < pkgvsn.parse("0.6.0"):
            raise ValueError(
                "3D Predictive Tracking Only Supported with trackpy versions newer than 0.6.0."
            )
        # avoid setting pos_columns by renaming to default values to avoid trackpy bug
        tracks_orig_cleaned.rename(
            columns={
                "y": "__temp_y_coord",
                "x": "__temp_x_coord",
                "z": "__temp_z_coord",
            },
            inplace=True,
        )
        new_features_cleaned.rename(
            columns={
                "y": "__temp_y_coord",
                "x": "__temp_x_coord",
                "z": "__temp_z_coord",
            },
            inplace=True,
        )

        tracks_orig_cleaned.rename(
            columns={"hdim_1": "y", "hdim_2": "x", "vdim_adj": "z"}, inplace=True
        )
        new_features_cleaned.rename(
            columns={"hdim_1": "y", "hdim_2": "x", "vdim_adj": "z"}, inplace=True
        )

    # generate list of features as input for df_link_iter
    features_linking_list_orig = [
        frame for i, frame in tracks_orig_cleaned.groupby("frame", sort=True)
    ]
    # generate list of features as input for df_link_iter
    features_linking_list_new = [
        frame for i, frame in new_features_cleaned.groupby("frame", sort=True)
    ]
    comb_features_linking = features_linking_list_orig + features_linking_list_new

    if method_linking == "random":
        #     link features into trajectories:
        trajectories_unfiltered = tp.link_df_iter(
            comb_features_linking,
            search_range=search_range,
            memory=memory,
            t_column="frame",
            pos_columns=pos_columns_tp,
            adaptive_step=adaptive_step,
            adaptive_stop=adaptive_stop,
            neighbor_strategy=neighbor_strategy,
            link_strategy="auto",
            dist_func=dist_func,
        )
        trajectories_unfiltered = pd.concat(trajectories_unfiltered)

    elif method_linking == "predict":
        # if span is updated, need to make sure we have enough timesteps.
        span = 1

        guess_speed_pos_df = _calc_velocity_to_most_recent_point(
            tracks_orig_cleaned, span=span
        )
        if is_3D:
            speed_cols = ["vdim_spd_mean", "hdim_1_spd_mean", "hdim_2_spd_mean"]
            pos_cols = ["z", "y", "x"]
        else:
            speed_cols = ["hdim_1_spd_mean", "hdim_2_spd_mean"]
            pos_cols = ["y", "x"]
        # I'm not sure our predictions here are working for position.
        pred = tp.predict.NearestVelocityPredict(
            initial_guess_positions=guess_speed_pos_df[pos_cols].values,
            initial_guess_vels=guess_speed_pos_df[speed_cols].values,
            pos_columns=pos_cols,
            span=span,
        )
        concat_df_linking = pd.concat(comb_features_linking)
        trajectories_unfiltered = pred.link_df(
            concat_df_linking,
            search_range=search_range,
            memory=memory,
            # pos_columns=["hdim_1", "hdim_2"], # not working atm
            t_column="frame",
            neighbor_strategy=neighbor_strategy,
            link_strategy="auto",
            adaptive_step=adaptive_step,
            adaptive_stop=adaptive_stop,
            dist_func=dist_func,
            #                                 copy_features=False, diagnostics=False,
            #                                 hash_size=None, box_size=None, verify_integrity=True,
            #                                 retain_index=False
        )
        # recreate a single dataframe from the list

        # trajectories_unfiltered = pd.concat(trajectories_unfiltered)

        # change to column names back
        trajectories_unfiltered.rename(
            columns={"y": "hdim_1", "x": "hdim_2", "z": "vdim_adj"}, inplace=True
        )
        trajectories_unfiltered.rename(
            columns={
                "__temp_y_coord": "y",
                "__temp_x_coord": "x",
                "__temp_z_coord": "z",
            },
            inplace=True,
        )

    else:
        raise ValueError("method_linking unknown")

    # Reset trackpy parameters to previously set values
    if subnetwork_size is not None:
        if adaptive_step is None and adaptive_stop is None:
            tp.linking.Linker.MAX_SUB_NET_SIZE = size_cache
        else:
            tp.linking.Linker.MAX_SUB_NET_SIZE_ADAPTIVE = size_cache

    # Filter trajectories to exclude short trajectories that are likely to be spurious
    #    trajectories_filtered = filter_stubs(trajectories_unfiltered,threshold=stubs)
    #    trajectories_filtered=trajectories_filtered.reset_index(drop=True)

    # clean up our temporary filters
    if is_3D:
        trajectories_unfiltered = trajectories_unfiltered.drop("vdim_adj", axis=1)

    cell_particle_pairs = (
        trajectories_unfiltered[["cell", "particle", "hdim_1"]]
        .groupby(["cell", "particle"])
        .nunique()
        .index
    )
    # check to make sure that there are no double matches, which would be bad.
    # checking for 1:1 matches.
    cell_matches = (
        trajectories_unfiltered[["cell", "particle", "hdim_1"]]
        .drop_duplicates(["cell", "particle"])
        .groupby("cell")["particle"]
        .count()
        .max()
    )
    particle_matches = (
        trajectories_unfiltered[["cell", "particle", "hdim_1"]]
        .drop_duplicates(["cell", "particle"])
        .groupby("particle")["cell"]
        .count()
        .max()
    )

    if cell_matches + particle_matches != 2:
        raise ValueError(
            "Error in appending tracks. Multiple pairs of cell:particle found. Please report this bug."
            " Number of cell matches: {0}, Number of particle matches: {1}".format(
                cell_matches, particle_matches
            )
        )
    # dictionary of particle:cell pairs
    particle_num_to_cell_num = {b: a for a, b in cell_particle_pairs}
    # update cells to link between pre-cut and post-cut.
    cell_offset = max(trajectories_unfiltered["cell"].max(), tracks_cut["cell"].max())
    i_additional_particle = 1
    for i_particle, particle in enumerate(
        pd.Series.unique(trajectories_unfiltered["particle"])
    ):
        if particle not in particle_num_to_cell_num:
            cell = int(i_additional_particle + cell_offset)
            i_additional_particle += 1
            particle_num_to_cell_num[particle] = int(cell)

    remap_particle_to_cell_vec = np.vectorize(remap_particle_to_cell_nv)
    trajectories_unfiltered["cell"] = remap_particle_to_cell_vec(
        particle_num_to_cell_num, trajectories_unfiltered["particle"]
    )
    trajectories_unfiltered["cell"] = trajectories_unfiltered["cell"].astype(int)
    trajectories_unfiltered.drop(columns=["particle"], inplace=True)

    trajectories_unfiltered = pd.concat([tracks_cut, trajectories_unfiltered])

    trajectories_bycell = trajectories_unfiltered.groupby("cell")
    stub_cell_nums = list()
    for cell, trajectories_cell in trajectories_bycell:
        if trajectories_cell.shape[0] < stubs:
            logging.debug(
                "cell"
                + str(cell)
                + "  is a stub ("
                + str(trajectories_cell.shape[0])
                + "), setting cell number to "
                + str(cell_number_unassigned)
            )
            stub_cell_nums.append(cell)

    trajectories_unfiltered.loc[
        trajectories_unfiltered["cell"].isin(stub_cell_nums), "cell"
    ] = cell_number_unassigned

    trajectories_filtered = trajectories_unfiltered

    # Interpolate to fill the gaps in the trajectories (left from allowing memory in the linking)
    trajectories_filtered_unfilled = deepcopy(trajectories_filtered)

    #    trajectories_filtered_filled=fill_gaps(trajectories_filtered_unfilled,order=order,
    #                                extrapolate=extrapolate,frame_max=field_in.shape[0]-1,
    #                                hdim_1_max=field_in.shape[1],hdim_2_max=field_in.shape[2])
    #     add coorinates from input fields to output trajectories (time,dimensions)
    #    logging.debug('start adding coordinates to trajectories')
    #    trajectories_filtered_filled=add_coordinates(trajectories_filtered_filled,field_in)
    #     add time coordinate relative to cell initiation:
    #    logging.debug('start adding cell time to trajectories')
    trajectories_filtered_filled = trajectories_filtered_unfilled
    trajectories_final = add_cell_time(
        trajectories_filtered_filled, cell_number_unassigned=cell_number_unassigned
    )
    # Add metadata
    trajectories_final.attrs["cell_number_unassigned"] = cell_number_unassigned

    # add coordinate to raw features identified:
    logging.debug("start adding coordinates to detected features")
    logging.debug("feature linking completed")
    # trajectories_final.reset_index(inplace=True, drop=True)

    return trajectories_final


def _clean_track_dfs_for_append(
    tracks_orig: pd.DataFrame, new_features: pd.DataFrame, memory: int, dt: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Get two cleaned up

    Parameters
    ----------
    tracks_orig
    new_features
    memory

    Returns
    -------

    """

    # need to cut down the existing track array to just the parts we are interested in
    min_frame_of_interest = max(max(tracks_orig["frame"]) - memory - 1, 0)
    max_frame_of_interest = max(tracks_orig["frame"])
    frames_of_interest = np.arange(min_frame_of_interest, max_frame_of_interest + 1, 1)
    tracks_orig_cut = copy.deepcopy(
        tracks_orig[tracks_orig["frame"].isin(frames_of_interest)]
    )
    remainder_tracks = copy.deepcopy(
        tracks_orig[~tracks_orig["frame"].isin(frames_of_interest)]
    )

    # now we need to get the parts of new_features that we need.
    if min(new_features["frame"]) == max_frame_of_interest + 1:
        # we have exactly the next dataframe we need. we are good to go.
        new_feats_cut = copy.deepcopy(new_features)
        return tracks_orig_cut, remainder_tracks, new_feats_cut
    # we need to cut or otherwise combine the new features.
    # check to see if the dataframes have been combined
    # if we have the last frame of the old one, we will assume the features are just combined.
    vars_of_interest = ["hdim_1", "hdim_2", "frame", "idx"]
    if tracks_orig[tracks_orig["frame"] == max_frame_of_interest][
        vars_of_interest
    ].equals(
        new_features[new_features["frame"] == max_frame_of_interest][vars_of_interest]
    ):
        new_feats_cut = copy.deepcopy(
            new_features[new_features["frame"] > max_frame_of_interest]
        )
        return tracks_orig_cut, remainder_tracks, new_feats_cut

    # now, let's assume that the new features are entirely separate, but start at the next timestep
    new_feats_cut = copy.deepcopy(new_features)
    # adjust frame number to be one larger than the tracks_orig max frame number
    new_feats_cut["frame"] += max_frame_of_interest - min(new_feats_cut["frame"]) + 1
    return tracks_orig_cut, remainder_tracks, new_feats_cut


def fill_gaps(
    t, order=1, extrapolate=0, frame_max=None, hdim_1_max=None, hdim_2_max=None
):
    """Add cell time as time since the initiation of each cell.

    Parameters
    ----------
    t : pandas.DataFrame
        Trajectories from trackpy.

    order : int, optional
        Order of polynomial used to extrapolate trajectory into
        gaps and beyond start and end point. Default is 1.

    extrapolate : int, optional
        Number or timesteps to extrapolate trajectories. Default is 0.

    frame_max : int, optional
        Size of input data along time axis. Default is None.

    hdim_1_max, hdim2_max : int, optional
        Size of input data along first and second horizontal axis.
        Default is None.

    Returns
    -------
    t : pandas.DataFrame
        Trajectories from trackpy with with filled gaps and potentially
        extrapolated.
    """

    from scipy.interpolate import InterpolatedUnivariateSpline

    logging.debug("start filling gaps")

    t_list = []  # empty list to store interpolated DataFrames

    # group by cell number and perform process for each cell individually:
    t_grouped = t.groupby("cell")
    for cell, track in t_grouped:
        # Setup interpolator from existing points (of order given as keyword)
        frame_in = track["frame"].values
        hdim_1_in = track["hdim_1"].values
        hdim_2_in = track["hdim_2"].values
        s_x = InterpolatedUnivariateSpline(frame_in, hdim_1_in, k=order)
        s_y = InterpolatedUnivariateSpline(frame_in, hdim_2_in, k=order)

        # Create new index filling in gaps and possibly extrapolating:
        index_min = min(frame_in) - extrapolate
        index_min = max(index_min, 0)
        index_max = max(frame_in) + extrapolate
        index_max = min(index_max, frame_max)
        new_index = range(index_min, index_max + 1)  # +1 here to include last value
        track = track.reindex(new_index)

        # Interpolate to extended index:
        frame_out = new_index
        hdim_1_out = s_x(frame_out)
        hdim_2_out = s_y(frame_out)

        # Replace fields in data frame with
        track["frame"] = new_index
        track["hdim_1"] = hdim_1_out
        track["hdim_2"] = hdim_2_out
        track["cell"] = cell

        # Append DataFrame to list of DataFrames
        t_list.append(track)
    # Concatenate interpolated trajectories into one DataFrame:
    t_out = pd.concat(t_list)
    # Restrict output trajectories to input data in time and space:
    t_out = t_out.loc[
        (t_out["hdim_1"] < hdim_1_max)
        & (t_out["hdim_2"] < hdim_2_max)
        & (t_out["hdim_1"] > 0)
        & (t_out["hdim_2"] > 0)
    ]
    t_out = t_out.reset_index(drop=True)
    return t_out


def _calc_search_range(
    dt: float,
    dxy: float,
    v_max: Optional[float] = None,
    d_max: Optional[float] = None,
    d_min: Optional[float] = None,
) -> float:
    """Internal function to calculate the trackpy search_range given the various radius parameters.

    Parameters
    ----------
    dt: float
        Time resolution (in seconds)
    dxy: float
        Space resolution (in meters)
    v_max: float, optional
        Speed at which features are allowed to move in meters per second.
        Only one of `d_max`, `d_min`, or `v_max` can be set.
        Default is None.
    d_max: float, optional
        Maximum search range in meters. Only one of `d_max`, `d_min`, or `v_max` can be set.
        Default is None.
    d_min: float, optional
        Deprecated. Only one of `d_max`, `d_min`, or `v_max` can be set.
        Default is None.

    Returns
    -------
    search_range: float
        search_range in hdim_1/hdim_2 coordinates from physical coordinates

    Raises
    ------
    ValueError:
        Raises ValueError if there are multiple parameter inputs.

    """
    if (v_max is None) and (d_min is None) and (d_max is None):
        raise ValueError(
            "Neither d_max nor v_max has been provided. "
            "Either one of these arguments must be specified."
        )

    # calculate search range based on timestep and grid spacing
    if v_max is not None:
        return dt * v_max / dxy

    # calculate search range based on timestep and grid spacing
    if d_max is not None:
        if v_max is not None:
            raise ValueError(
                "Multiple parameter inputs for v_max, d_max or d_min have been provided. "
                "Only use one of these parameters as they supercede each other leading to "
                "unexpected behaviour."
            )
        return d_max / dxy

    # calculate search range based on timestep and grid spacing
    if d_min is not None:
        if (v_max is not None) or (d_max is not None):
            raise ValueError(
                "Multiple parameter inputs for v_max, d_max or d_min have been provided. Only use one of these parameters as they supercede each other leading to unexpected behaviour"
            )
        warnings.warn(
            "d_min parameter will be deprecated in tobac v1.6. Please use d_max instead",
            FutureWarning,
        )

        return d_min / dxy
    raise ValueError("Invalid choice for _calculate_search_radius")


def _calc_velocity_to_most_recent_point(
    in_track: pd.DataFrame, span: int = 1
) -> pd.DataFrame:
    """ """
    in_track = in_track.reset_index()
    min_frame = in_track["frame"].min()
    speed_tracks = in_track[in_track["frame"] <= min_frame + span]

    if "vdim_adj" in speed_tracks:
        # 3D case
        pos_cols = ["y", "x", "z"]
        spd_cols = ["hdim_1_spd", "hdim_2_spd", "vdim_spd"]

        speed_rename = {
            "y": "hdim_1_spd",
            "x": "hdim_2_spd",
            "z": "vdim_spd",
        }
    else:
        # 2D case
        pos_cols = ["y", "x"]
        spd_cols = ["hdim_1_spd", "hdim_2_spd"]
        speed_rename = {
            "y": "hdim_1_spd",
            "x": "hdim_2_spd",
        }
    calculated_speeds = (
        speed_tracks.sort_values(["frame", "idx"])
        .groupby("cell")[pos_cols]
        .transform("diff")
        .rename(speed_rename, axis="columns")
    )
    all_speeds = speed_tracks[["feature", "frame", "idx", "cell"] + pos_cols].join(
        calculated_speeds,
    )
    mean_speed_df = all_speeds[all_speeds["frame"] == all_speeds["frame"].min()].join(
        all_speeds.groupby("cell")[["hdim_1_spd", "hdim_2_spd"]].mean(),
        on="cell",
        rsuffix="_mean",
    )

    mean_speed_df.drop(spd_cols, inplace=True, axis="columns")
    mean_speed_df.dropna(inplace=True, axis=0)
    return mean_speed_df


def add_cell_time(t: pd.DataFrame, cell_number_unassigned: int):
    """add cell time as time since the initiation of each cell

    Parameters
    ----------
    t : pandas.DataFrame
        trajectories with added coordinates
    cell_number_unassigned: int
        unassigned cell value

    Returns
    -------
    t : pandas.Dataframe
        trajectories with added cell time
    """

    # logging.debug('start adding time relative to cell initiation')
    t_grouped = t.groupby("cell")

    t["time_cell"] = t["time"] - t.groupby("cell")["time"].transform("min")
    t["time_cell"] = pd.to_timedelta(t["time_cell"])
    t.loc[t["cell"] == cell_number_unassigned, "time_cell"] = pd.Timedelta("nat")
    return t


def remap_particle_to_cell_nv(particle_cell_map, input_particle):
    """Remaps the particles to new cells given an input map and the current particle.
    Helper function that is designed to be vectorized with np.vectorize

    Parameters
    ----------
    particle_cell_map: dict-like
        The dictionary mapping particle number to cell number
    input_particle: key for particle_cell_map
        The particle number to remap

    """
    return particle_cell_map[input_particle]


def _get_vertical_coord(
    tracks: pd.DataFrame,
    dz: Optional[float] = None,
    vertical_coord: Optional[str] = None,
) -> tuple[bool, str]:
    """Internal function to get the name of the vertical coordinate if 3D and
    return false if the tracks are not 3D.

    Parameters
    ----------
    tracks: pd.DataFrame
        Input feature dataframe (could include tracking information or not).
    dz: float, optional
        Constant vertical grid spacing (meters), optional. If not specified
        and the input is 3D, this function requires that `vertical_coord` is available
        in the `features` input. If you specify a value here, this function assumes
        that it is the constant z spacing between points, even if ```vertical_coord```
        is specified.

    vertical_coord: str
        Name of the vertical coordinate. The vertical coordinate used
        must be meters. If None, tries to auto-detect.
        It looks for the coordinate or the dimension name corresponding
        to the string. To use `dz`, set this to `None`.

    Returns
    -------
    is_3d, vertical_coord: bool, str
        True if input is 3D, False otherwise. vertical_coord is the column name of the
        vertical coordinate to use in the feature dataframe.

    """
    found_vertical_coord = ""
    if "vdim" in tracks:
        is_3d = True
        if dz is not None and vertical_coord is not None:
            raise ValueError(
                "dz and vertical_coord both set, vertical"
                " spacing is ambiguous. Set one to None."
            )
        if dz is None and vertical_coord is None:
            raise ValueError(
                "Neither dz nor vertical_coord are set. One" " must be set."
            )
        if vertical_coord is not None:
            found_vertical_coord = internal_utils.find_dataframe_vertical_coord(
                variable_dataframe=tracks, vertical_coord=vertical_coord
            )
    else:
        is_3d = False

    return is_3d, found_vertical_coord


def _check_pbc_coords(
    pbc_flag: Literal["none", "hdim_1", "hdim_2", "both"],
    min_h1: Optional[int] = None,
    max_h1: Optional[int] = None,
    min_h2: Optional[int] = None,
    max_h2: Optional[int] = None,
) -> bool:
    """check that we have min/max h1 and h2 coords if we are on PBCs.

    Parameters
    ----------
    pbc_flag: str
        Input PBC flag
    min_h1: int or None
        minimum hdim_1 point
    max_h1: int or None
        maximum hdim_1 point
    min_h2: int or None
        minimum hdim_2 point
    max_h2: int or None
        maximum hdim_2 point

    Returns
    -------
    True if all parameters are OK

    Raises
    ------
    ValueError if min/max h1/h2 are not set when they should be
    """

    if pbc_flag in ["hdim_1", "both"] and (min_h1 is None or max_h1 is None):
        raise ValueError(
            "For PBCs, must set min (min_h1) and max (max_h1) coordinates."
        )

    if pbc_flag in ["hdim_2", "both"] and (min_h2 is None or max_h2 is None):
        raise ValueError(
            "For PBC tracking, must set min (min_h2) and max (max_h2) coordinates."
        )

    return True


def _check_set_adaptive_params(
    adaptive_stop: float, adaptive_step: float, subnetwork_size: Optional[int]
) -> int:
    """Internal function to check that all trackpy parameters relevant to adaptive search are OK,
    and sets the subnetwork_size internal to trackpy.

    Parameters
    ----------
    adaptive_stop: float
        adaptive_stop value for trackpy
    adaptive_step: float
        adaptive_step value for trackpy
    subnetwork_size: int

    Returns
    -------
    int
        returns the original trackpy subnetwork size.

    Raises
    ------
    ValueError if values are not set or acceptable.

    """

    if adaptive_stop is not None:
        if adaptive_step is None:
            raise ValueError(
                "Adaptive search requires values for adaptive_step and adaptive_stop. Please specify adaptive_step."
            )

    if adaptive_step is not None:
        if adaptive_stop is None:
            raise ValueError(
                "Adaptive search requires values for adaptive_step and adaptive_stop. Please specify adaptive_stop."
            )
    # If subnetwork size given, set maximum subnet size
    if subnetwork_size is not None:
        # Choose the right parameter depending on the use of adaptive search, save previously set values
        if adaptive_step is None and adaptive_stop is None:
            size_cache = tp.linking.Linker.MAX_SUB_NET_SIZE
            tp.linking.Linker.MAX_SUB_NET_SIZE = subnetwork_size
        else:
            size_cache = tp.linking.Linker.MAX_SUB_NET_SIZE_ADAPTIVE
            tp.linking.Linker.MAX_SUB_NET_SIZE_ADAPTIVE = subnetwork_size

        return size_cache
    else:
        return 0


def _calc_frames_for_stubs(
    dt: float, stubs: Optional[int] = None, time_cell_min: Optional[float] = None
) -> int:
    """Internal function to calculate the number of frames to track our minimum cells through.

    Parameters
    ----------

    stubs: int
        depreciated parameter setting number of frames that cells have to last for to be considered
        a contiguous cell
    time_cell_min: float
        minimum time for a cell to last (in the same units as dt)

    Returns
    -------
    number of frames that a cell has to exist for to be considered a full cell

    Raises
    ------
    raises a ValueError if stubs and time_cell_min are both not set.

    """
    if time_cell_min is None and stubs is None:
        raise ValueError(
            "time_cell_min and stubs both not set. One or the other must be set."
        )

    if time_cell_min is None:
        return stubs
    else:
        return np.floor(time_cell_min / dt) + 1

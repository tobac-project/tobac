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

import logging
import numpy as np
import pandas as pd
import warnings


def linking_trackpy(
    features,
    field_in,
    dt,
    dxy,
    v_max=None,
    d_max=None,
    d_min=None,
    subnetwork_size=None,
    memory=0,
    stubs=1,
    time_cell_min=None,
    order=1,
    extrapolate=0,
    method_linking="random",
    adaptive_step=None,
    adaptive_stop=None,
    cell_number_start=1,
    cell_number_unassigned=-1,
):

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

    field_in : xarray.DataArray
        Input field to perform the watershedding on (2D or 3D for one
        specific point in time).

    dt : float
        Time resolution of tracked features.

    dxy : float
        Grid spacing of the input data.

    d_max : float, optional
        Maximum search range
        Default is None.

    d_min : float, optional
        Variations in the shape of the regions used to determine the
        positions of the features can lead to quasi-instantaneous shifts
        of the position of the feature by one or two grid cells even for
        a very high temporal resolution of the input data, potentially
        jeopardising the tracking procedure. To prevent this, tobac uses
        an additional minimum radius of the search range.
        Default is None.

    subnetwork_size : int, optional
        Maximum size of subnetwork for linking. This parameter should be
        adjusted when using adaptive search. Usually a lower value is desired
        in that case. For a more in depth explanation have look
        `here <https://soft-matter.github.io/trackpy/v0.5.0/tutorial/adaptive-search.html>`_
        If None, 30 is used for regular search and 15 for adaptive search.
        Default is None.

    v_max : float, optional
        Speed at which features are allowed to move. Default is None.

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
        Minimum length in time of tracked cell to be reported in minutes
        Default is None.

    order : int, optional
        Order of polynomial used to extrapolate trajectory into gaps and
        ond start and end point.
        Default is 1.

    extrapolate : int, optional
        Number or timesteps to extrapolate trajectories.
        Default is 0.

    method_linking : {'random', 'predict'}, optional
        Flag choosing method used for trajectory linking.
        Default is 'random'.

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

    #    from trackpy import link_df
    import trackpy as tp
    from copy import deepcopy

    #    from trackpy import filter_stubs
    #    from .utils import add_coordinates

    if (v_max is None) and (d_min is None) and (d_max is None):
        raise ValueError(
            "Neither d_max nor v_max has been provided. Either one of these arguments must be specified."
        )

    # calculate search range based on timestep and grid spacing
    if v_max is not None:
        search_range = int(dt * v_max / dxy)

    # calculate search range based on timestep and grid spacing
    if d_max is not None:
        if v_max is not None:
            raise ValueError(
                "Multiple parameter inputs for v_max, d_max or d_min have been provided. Only use one of these parameters as they supercede each other leading to unexpected behaviour"
            )
        search_range = int(d_max / dxy)

    # calculate search range based on timestep and grid spacing
    if d_min is not None:
        if (v_max is not None) or (d_max is not None):
            raise ValueError(
                "Multiple parameter inputs for v_max, d_max or d_min have been provided. Only use one of these parameters as they supercede each other leading to unexpected behaviour"
            )
        search_range = int(d_min / dxy)
        warnings.warn(
            "d_min parameter will be deprecated in a future version of tobac. Please use d_max instead",
            FutureWarning,
        )

    # in case of adaptive search, check wether both parameters are specified
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

    if time_cell_min:
        stubs = np.floor(time_cell_min / dt) + 1

    logging.debug("stubs: " + str(stubs))

    logging.debug("start linking features into trajectories")

    # If subnetwork size given, set maximum subnet size
    if subnetwork_size is not None:
        # Choose the right parameter depending on the use of adaptive search, save previously set values
        if adaptive_step is None and adaptive_stop is None:
            size_cache = tp.linking.Linker.MAX_SUB_NET_SIZE
            tp.linking.Linker.MAX_SUB_NET_SIZE = subnetwork_size
        else:
            size_cache = tp.linking.Linker.MAX_SUB_NET_SIZE_ADAPTIVE
            tp.linking.Linker.MAX_SUB_NET_SIZE_ADAPTIVE = subnetwork_size

    # deep copy to preserve features field:
    features_linking = deepcopy(features)

    if method_linking == "random":
        #     link features into trajectories:
        trajectories_unfiltered = tp.link(
            features_linking,
            search_range=search_range,
            memory=memory,
            t_column="frame",
            pos_columns=["hdim_2", "hdim_1"],
            adaptive_step=adaptive_step,
            adaptive_stop=adaptive_stop,
            neighbor_strategy="KDTree",
            link_strategy="auto",
        )
    elif method_linking == "predict":

        # avoid setting pos_columns by renaimng to default values to avoid trackpy bug
        features.rename(columns={"hdim_1": "y", "hdim_2": "x"}, inplace=True)

        # generate list of features as input for df_link_iter to avoid bug in df_link
        features_linking_list = [
            frame for i, frame in features.groupby("frame", sort=True)
        ]

        pred = tp.predict.NearestVelocityPredict(span=1)
        trajectories_unfiltered = pred.link_df_iter(
            features_linking_list,
            search_range=search_range,
            memory=memory,
            # pos_columns=["hdim_1", "hdim_2"], # not working atm
            t_column="frame",
            neighbor_strategy="KDTree",
            link_strategy="auto",
            adaptive_step=adaptive_step,
            adaptive_stop=adaptive_stop
            #                                 copy_features=False, diagnostics=False,
            #                                 hash_size=None, box_size=None, verify_integrity=True,
            #                                 retain_index=False
        )
        # recreate a single dataframe from the list
        trajectories_unfiltered = pd.concat(trajectories_unfiltered)

        # change to column names back
        trajectories_unfiltered.rename(
            columns={"y": "hdim_1", "x": "hdim_2"}, inplace=True
        )
        features.rename(columns={"y": "hdim_1", "x": "hdim_2"}, inplace=True)
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

    # Reset particle numbers from the arbitray numbers at the end of the feature detection and linking to consecutive cell numbers
    # keep 'particle' for reference to the feature detection step.
    trajectories_unfiltered["cell"] = None
    particle_num_to_cell_num = dict()
    for i_particle, particle in enumerate(
        pd.Series.unique(trajectories_unfiltered["particle"])
    ):
        cell = int(i_particle + cell_number_start)
        particle_num_to_cell_num[particle] = int(cell)

    remap_particle_to_cell_vec = np.vectorize(
        lambda particle_cell_map, input_particle: particle_cell_map[input_particle]
    )
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
            trajectories_unfiltered.loc[
                trajectories_unfiltered["cell"] == cell, "cell"
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
    trajectories_final = add_cell_time(trajectories_filtered_filled)
    # Add metadata
    trajectories_final.attrs["cell_number_unassigned"] = cell_number_unassigned

    # add coordinate to raw features identified:
    logging.debug("start adding coordinates to detected features")
    logging.debug("feature linking completed")

    return trajectories_final


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


def add_cell_time(t):
    """add cell time as time since the initiation of each cell

    Parameters
    ----------
    t : pandas.DataFrame
        trajectories with added coordinates

    Returns
    -------
    t : pandas.Dataframe
        trajectories with added cell time
    """

    # logging.debug('start adding time relative to cell initiation')
    t_grouped = t.groupby("cell")

    t["time_cell"] = t["time"] - t.groupby("cell")["time"].transform("min")
    t["time_cell"] = pd.to_timedelta(t["time_cell"])
    return t

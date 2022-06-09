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
    """
    Function to perform the linking of features in trajectories

    Parameters:
    features:     pandas.DataFrame
                  Detected features to be linked
    v_max:        float
                  speed at which features are allowed to move
    dt:           float
                  time resolution of tracked features
    dxy:          float
                  grid spacing of input data
    memory        int
                  number of output timesteps features allowed to vanish for to be still considered tracked
    subnetwork_size int
                    maximim size of subnetwork for linking
    method_detection: str('trackpy' or 'threshold')
                      flag choosing method used for feature detection
    method_linking:   str('predict' or 'random')
                      flag choosing method used for trajectory linking
    cell_number_unassigned: int
        Number to set the unassigned/non-tracked cells to. By default, this is -1.
        Note that if you set this to `np.nan`, the data type of 'cell' will
        change to float.
    """
    #    from trackpy import link_df
    import trackpy as tp
    from copy import deepcopy

    #    from trackpy import filter_stubs
    #    from .utils import add_coordinates

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

    if time_cell_min:
        stubs = np.floor(time_cell_min / dt) + 1

    logging.debug("stubs: " + str(stubs))

    logging.debug("start linking features into trajectories")

    # If subnetwork size given, set maximum subnet size
    if subnetwork_size is not None:
        # Choose the right parameter depending on the use of adaptive search
        if adaptive_step is None and adaptive_stop is None:
            tp.linking.Linker.MAX_SUB_NET_SIZE = subnetwork_size
        else:
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

        pred = tp.predict.NearestVelocityPredict(span=1)
        trajectories_unfiltered = pred.link_df(
            features_linking,
            search_range=search_range,
            memory=memory,
            pos_columns=["hdim_1", "hdim_2"],
            t_column="frame",
            neighbor_strategy="KDTree",
            link_strategy="auto",
            adaptive_step=adaptive_step,
            adaptive_stop=adaptive_stop
            #                                 copy_features=False, diagnostics=False,
            #                                 hash_size=None, box_size=None, verify_integrity=True,
            #                                 retain_index=False
        )
    else:
        raise ValueError("method_linking unknown")

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
    """add cell time as time since the initiation of each cell
    Input:
    t:             pandas dataframe
                   trajectories from trackpy
    order:         int
                    Order of polynomial used to extrapolate trajectory into gaps and beyond start and end point
    extrapolate     int
                    number of timesteps to extrapolate trajectories by
    frame_max:      int
                    size of input data along time axis
    hdim_1_max:     int
                    size of input data along first horizontal axis
    hdim_2_max:     int
                    size of input data along second horizontal axis
    Output:
    t:             pandas dataframe
                   trajectories from trackpy with with filled gaps and potentially extrapolated
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
    t:             pandas  DataFrame
                   trajectories with added coordinates

    Returns
    -------
    t:             pandas dataframe
                   trajectories with added cell time
    """

    # logging.debug('start adding time relative to cell initiation')
    t_grouped = t.groupby("cell")

    t["time_cell"] = t["time"] - t.groupby("cell")["time"].transform("min")
    t["time_cell"] = pd.to_timedelta(t["time_cell"])
    return t

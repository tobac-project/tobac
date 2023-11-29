"""
    Tobac merge and split
    This submodule is a post processing step to address tracked cells which merge/split. 
    The first iteration of this module is to combine the cells which are merging but have received
    a new cell id (and are considered a new cell) once merged. In general this submodule will label merged/split cells
    with a TRACK number in addition to its CELL number.
    
"""

import logging

import numpy as np
import pandas as pd
from pandas.core.common import flatten
import xarray as xr
from sklearn.neighbors import BallTree

try:
    import networkx as nx
except ImportError:
    networkx = None

from tobac.tracking import build_distance_function


def merge_split_MEST(
    tracks: pd.DataFrame,
    dxy: float,
    dz: float = None,
    distance: float = None,
    frame_len: int = 5,
    PBC_flag: "str" = None,
    min_h1: int = None,
    max_h1: int = None,
    min_h2: int = None,
    max_h2: int = None,
) -> xr.Dataset:
    """
    function to  postprocess tobac track data for merge/split cells using a minimum euclidian spanning tree


    Parameters
    ----------
    TRACK : pandas.core.frame.DataFrame
        Pandas dataframe of tobac Track information

    dxy : float, mandatory
        The x/y grid spacing of the data.
        Should be in meters.

    dz : float, optional
        Constant vertical grid spacing (m)

    distance : float, optional
        Distance threshold determining how close two features must be in order to consider merge/splitting.
        Default is 25x the x/y grid spacing of the data, given in dxy.
        The distance should be in units of meters.

    frame_len : float, optional
        Threshold for the maximum number of frames that can separate the end of cell and the start of a related cell.
        Default is five (5) frames.

    PBC_flag : str('none', 'hdim_1', 'hdim_2', 'both'), optional
        Sets whether to use periodic boundaries, and if so in which directions.
        'none' means that we do not have periodic boundaries
        'hdim_1' means that we are periodic along hdim1
        'hdim_2' means that we are periodic along hdim2
        'both' means that we are periodic along both horizontal dimensions

    min_h1: int, optional
        Minimum real point in hdim_1, for use with periodic boundaries.

    max_h1: int, optional
        Maximum point in hdim_1, exclusive. max_h1-min_h1 should be the size of hdim_1.

    min_h2: int, optional
        Minimum real point in hdim_2, for use with periodic boundaries.

    max_h2: int, optional
        Maximum point in hdim_2, exclusive. max_h2-min_h2 should be the size of hdim_2.


    Returns
    -------

    d : xarray.core.dataset.Dataset
        xarray dataset of tobac merge/split cells with parent and child designations.

                Parent/child variables include:
        - cell_parent_track_id: The associated track id for each cell. All cells that have merged or split will have the same parent track id. If a cell never merges/splits, only one cell will have a particular track id.
        - feature_parent_cell_id: The associated parent cell id for each feature. All features in a given cell will have the same cell id. This is the original TRACK cell_id.
        - feature_parent_track_id: The associated parent track id for each feature. This is not the same as the cell id number.
        - track_child_cell_count: The total number of features belonging to all child cells of a given track id.
        - cell_child_feature_count: The total number of features for each cell.


    Example usage:
        d = merge_split_MEST(Track)
        ds = tobac.utils.standardize_track_dataset(Track, refl_mask)
        both_ds = xr.merge([ds, d],compat ='override')
        both_ds = tobac.utils.compress_all(both_ds)
        both_ds.to_netcdf(os.path.join(savedir,'Track_features_merges.nc'))

    """

    # Immediately convert pandas dataframe of track information to xarray:
    tracks = tracks.to_xarray()
    track_groups = tracks.groupby("cell")
    first = track_groups.first()
    last = track_groups.last()

    if distance is None:
        distance = dxy * 25.0

    # As optional coordinate names are not yet implemented, set to defaults here:
    z_coordinate_name = "vdim"
    y_coordinate_name = "hdim_1"
    x_coordinate_name = "hdim_2"

    is_3D = "vdim" in tracks

    if is_3D and dz is None:
        raise ValueError("dz must be specified for 3D data")

    # Calculate feature locations in cartesian coordinates
    if is_3D:
        cell_start_locations = np.stack(
            [
                first[var].values
                for var in [z_coordinate_name, y_coordinate_name, x_coordinate_name]
            ],
            axis=0,
        )
        cell_start_locations[0] *= dz
        cell_start_locations[1:] *= dxy
        cell_end_locations = np.stack(
            [
                last[var].values
                for var in [z_coordinate_name, y_coordinate_name, x_coordinate_name]
            ],
            axis=0,
        )
        cell_end_locations[0] *= dz
        cell_end_locations[1:] *= dxy
    else:
        cell_start_locations = np.stack(
            [first[var].values for var in [y_coordinate_name, x_coordinate_name]],
            axis=-1,
        )
        cell_end_locations = np.stack(
            [last[var].values for var in [y_coordinate_name, x_coordinate_name]],
            axis=-1,
        )

    if PBC_flag in ["hdim_1", "hdim_2", "both"]:
        # Note that we multiply by dxy to get the distances in spatial coordinates
        dist_func = build_distance_function(
            min_h1 * dxy, max_h1 * dxy, min_h2 * dxy, max_h2 * dxy, PBC_flag
        )
        cell_start_tree = BallTree(
            cell_start_locations, metric="pyfunc", func=dist_func
        )

    else:
        cell_start_tree = BallTree(cell_start_locations, metric="euclidean")

    neighbours, distances = cell_start_tree.query_radius(
        cell_end_locations, r=distance, return_distance=True
    )

    # Input data to the graph which will perform the spanning tree.
    g = nx.Graph()
    start_node_cells = first["cell"].values[
        np.repeat(np.arange(len(neighbours), dtype=int), [len(n) for n in neighbours])
    ]
    end_node_cells = last["cell"].values[np.concatenate(neighbours)]
    weights = np.concatenate(distances)
    g.add_weighted_edges_from(zip(start_node_cells, end_node_cells, weights))

    tree = nx.minimum_spanning_edges(g)
    tree_list = list(tree)

    new_tree = []

    # Pruning the tree for time limits.
    for i, j in enumerate(tree_list):
        frame_a = np.nanmax(track_groups[j[0]].frame.values)
        frame_b = np.nanmin(track_groups[j[1]].frame.values)
        if np.abs(frame_a - frame_b) <= frame_len:
            new_tree.append(tree_list[i][0:2])
    new_tree_arr = np.array(new_tree)

    tracks["cell_parent_track_id"] = np.zeros(len(tracks["cell"].values))
    cell_id = np.unique(
        tracks.cell.values.astype(int)[~np.isnan(tracks.cell.values.astype(int))]
    )
    track_id = dict()  # same size as number of total merged tracks

    # Cleaning up tracks, combining tracks which contain the same cells.
    arr = np.array([0])
    for p in cell_id:
        j = np.where(arr == int(p))
        if len(j[0]) > 0:
            continue
        else:
            k = np.where(new_tree_arr == p)
            if len(k[0]) == 0:
                track_id[p] = [p]
                arr = np.append(arr, p)
            else:
                temp1 = list(np.unique(new_tree_arr[k[0]]))
                temp = list(np.unique(new_tree_arr[k[0]]))

                for l in range(len(cell_id)):
                    for i in temp1:
                        k2 = np.where(new_tree_arr == i)
                        temp.append(list(np.unique(new_tree_arr[k2[0]]).squeeze()))
                        temp = list(flatten(temp))
                        temp = list(np.unique(temp))

                    if len(temp1) == len(temp):
                        break
                    temp1 = np.array(temp)

                for i in temp1:
                    k2 = np.where(new_tree_arr == i)
                    temp.append(list(np.unique(new_tree_arr[k2[0]]).squeeze()))

                temp = list(flatten(temp))
                temp = list(np.unique(temp))
                arr = np.append(arr, np.unique(temp))

                track_id[np.nanmax(np.unique(temp))] = list(np.unique(temp))

    cell_id = list(np.unique(tracks.cell.values.astype(int)))
    logging.debug("found cell ids")

    cell_parent_track_id = np.zeros(len(cell_id))
    cell_parent_track_id[:] = -1

    for i, id in enumerate(track_id, start=0):
        for j in track_id[int(id)]:
            cell_parent_track_id[cell_id.index(j)] = int(i)

    logging.debug("found cell parent track ids")

    track_ids = np.array(np.unique(cell_parent_track_id))
    logging.debug("found track ids")

    feature_parent_cell_id = list(tracks.cell.values.astype(int))
    logging.debug("found feature parent cell ids")

    #     # This version includes all the feature regardless of if they are used in cells or not.
    feature_id = list(tracks.feature.values.astype(int))
    logging.debug("found feature ids")

    feature_parent_track_id = []
    feature_parent_track_id = np.zeros(len(feature_id))
    for i, id in enumerate(feature_id):
        cellid = feature_parent_cell_id[i]
        if cellid < 0:
            feature_parent_track_id[i] = -1
        else:
            feature_parent_track_id[i] = cell_parent_track_id[cell_id.index(cellid)]

    track_child_cell_count = np.zeros(len(track_id))
    for i, id in enumerate(track_id):
        track_child_cell_count[i] = len(np.where(cell_parent_track_id == i)[0])
    logging.debug("found track child cell count")

    cell_child_feature_count = np.zeros(len(cell_id))
    for i, id in enumerate(cell_id):
        cell_child_feature_count[i] = len(track_groups[id].feature.values)
    logging.debug("found cell child feature count")

    track_dim = "track"
    cell_dim = "cell"
    feature_dim = "feature"

    d = xr.Dataset(
        {
            "track": (track_dim, track_ids),
            "cell": (cell_dim, cell_id),
            "cell_parent_track_id": (cell_dim, cell_parent_track_id),
            "feature": (feature_dim, feature_id),
            "feature_parent_cell_id": (feature_dim, feature_parent_cell_id),
            "feature_parent_track_id": (feature_dim, feature_parent_track_id),
            "track_child_cell_count": (track_dim, track_child_cell_count),
            "cell_child_feature_count": (cell_dim, cell_child_feature_count),
        }
    )

    d = d.set_coords(["feature", "cell", "track"])

    #     assert len(cell_id) == len(cell_parent_track_id)
    #     assert len(feature_id) == len(feature_parent_cell_id)
    #     assert sum(track_child_cell_count) == len(cell_id)
    #     assert sum(cell_child_feature_count) == len(feature_id)

    return d

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
import xarray as xr
import scipy.sparse
from sklearn.neighbors import BallTree

from tobac.tracking import build_distance_function


def merge_split_MEST(
    tracks: pd.DataFrame,
    dxy: float,
    dz: float = None,
    distance: float = None,
    frame_len: int = 5,
    cell_number_unassigned: int = -1,
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

    cell_number_unassigned: int, optional
        Value given tp unassigned/non-tracked cells by tracking. Default is -1

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

    track_groups = tracks[tracks["cell"] != cell_number_unassigned].groupby("cell")
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
            axis=-1,
        )
        cell_start_locations[:, 0] *= dz
        cell_start_locations[:, 1:] *= dxy
        cell_end_locations = np.stack(
            [
                last[var].values
                for var in [z_coordinate_name, y_coordinate_name, x_coordinate_name]
            ],
            axis=-1,
        )
        cell_end_locations[0] *= dz
        cell_end_locations[1:] *= dxy
    else:
        cell_start_locations = (
            np.stack(
                [first[var].values for var in [y_coordinate_name, x_coordinate_name]],
                axis=-1,
            )
            * dxy
        )
        cell_end_locations = (
            np.stack(
                [last[var].values for var in [y_coordinate_name, x_coordinate_name]],
                axis=-1,
            )
            * dxy
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
    nodes = np.repeat(
        np.arange(len(neighbours), dtype=int), [len(n) for n in neighbours]
    )
    neighbours = np.concatenate(neighbours)
    weights = np.concatenate(distances)

    # Remove edges where the frame gap is greater than frame_len, and also remove connections to the same cell
    wh_frame_len = (
        np.abs(first["frame"].values[nodes] - last["frame"].values[neighbours])
        <= frame_len
    )
    wh_valid_edge = np.logical_and(wh_frame_len, nodes != neighbours)
    start_node_cells = first.index.values[nodes[wh_valid_edge]].astype(np.int32)
    end_node_cells = last.index.values[neighbours[wh_valid_edge]].astype(np.int32)

    cell_id = np.unique(tracks.cell.values)
    cell_id = cell_id[cell_id != cell_number_unassigned].astype(int)
    max_cell = np.max(cell_id)

    if len(start_node_cells):
        # We need to add a small value to the dists to prevent 0-length edges
        cell_graph = scipy.sparse.coo_array(
            (weights[wh_valid_edge] + 0.01, (start_node_cells, end_node_cells)),
            shape=(max_cell + 1, max_cell + 1),
        )
        cell_graph = scipy.sparse.csgraph.minimum_spanning_tree(
            cell_graph, overwrite=True
        )
        # Find remaining start/end nodes after calculating minimum spanning tree
        start_node_cells, end_node_cells = cell_graph.nonzero()

        cell_parent_track_id = scipy.sparse.csgraph.connected_components(cell_graph)[1][
            cell_id
        ]
        cell_parent_track_id = (
            np.unique(cell_parent_track_id, return_inverse=True)[1] + 1
        )
    else:
        cell_parent_track_id = np.arange(cell_id.size, dtype=int) + 1

    track_dim = "track"
    cell_dim = "cell"
    feature_dim = "feature"

    cell_parent_track_id = xr.DataArray(
        cell_parent_track_id, dims=(cell_dim,), coords={cell_dim: cell_id}
    )
    logging.debug("found cell parent track ids")

    track_id = np.unique(cell_parent_track_id)
    logging.debug("found track ids")

    # This version includes all the feature regardless of if they are used in cells or not.
    feature_id = tracks.feature.values.astype(int)
    logging.debug("found feature ids")

    feature_parent_cell_id = tracks.cell.values.astype(int)
    feature_parent_cell_id = xr.DataArray(
        feature_parent_cell_id,
        dims=(feature_dim,),
        coords={feature_dim: feature_id},
    )
    logging.debug("found feature parent cell ids")

    wh_feature_in_cell = (feature_parent_cell_id != cell_number_unassigned).values
    feature_parent_track_id = np.full(wh_feature_in_cell.shape, cell_number_unassigned)
    feature_parent_track_id[wh_feature_in_cell] = cell_parent_track_id.loc[
        feature_parent_cell_id[wh_feature_in_cell]
    ].values
    feature_parent_track_id = xr.DataArray(
        feature_parent_track_id,
        dims=(feature_dim,),
        coords={feature_dim: feature_id},
    )

    track_child_cell_count = (
        cell_parent_track_id.groupby(cell_parent_track_id).reduce(np.size).values
    )
    track_child_cell_count = xr.DataArray(
        track_child_cell_count,
        dims=(track_dim,),
        coords={track_dim: track_id},
    )

    cell_child_feature_count = (
        feature_parent_cell_id[wh_feature_in_cell]
        .groupby(feature_parent_cell_id[wh_feature_in_cell])
        .reduce(np.size)
        .values
    )
    cell_child_feature_count = xr.DataArray(
        cell_child_feature_count, dims=(cell_dim), coords={cell_dim: cell_id}
    )

    cell_starts_with_split = np.isin(cell_id, start_node_cells)
    cell_starts_with_split = xr.DataArray(
        cell_starts_with_split, dims=(cell_dim), coords={cell_dim: cell_id}
    )

    cell_ends_with_merge = np.isin(cell_id, end_node_cells)
    cell_ends_with_merge = xr.DataArray(
        cell_ends_with_merge, dims=(cell_dim), coords={cell_dim: cell_id}
    )

    merge_split_ds = xr.Dataset(
        data_vars={
            "cell_parent_track_id": cell_parent_track_id,
            "feature_parent_cell_id": feature_parent_cell_id,
            "feature_parent_track_id": feature_parent_track_id,
            "track_child_cell_count": track_child_cell_count,
            "cell_child_feature_count": cell_child_feature_count,
            "cell_starts_with_split": cell_starts_with_split,
            "cell_ends_with_merge": cell_ends_with_merge,
        },
        coords={feature_dim: feature_id, cell_dim: cell_id, track_dim: track_id},
    )

    return merge_split_ds

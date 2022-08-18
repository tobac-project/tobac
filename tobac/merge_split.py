"""
    Tobac merge and split v0.1
    This submodule is a post processing step to address tracked cells which merge/split. 
    The first iteration of this module is to combine the cells which are merging but receive 
    a new cell id once merged. In general this submodule will label these three cell-ids using
    the largest cell number of those within the merged/split cell ids. 
    
    
"""


def merge_split(TRACK, distance=25000, frame_len=5, dxy=500):
    """
    function to  postprocess tobac track data for merge/split cells


    Parameters
    ----------
    TRACK : pandas.core.frame.DataFrame
        Pandas dataframe of tobac Track information

    distance : float, optional
        Distance threshold prior to adding a pair of points into the minimum spanning tree.
        Default is 25000 meters.

    frame_len : float, optional
        Threshold for the spanning length within which two points can be separated.
        Default is five (5) frames.

    dxy : float, optional
        The x/y/ grid spacing of the data.
        Default is 500m.

    Returns
    -------

    d : xarray.core.dataset.Dataset
        xarray dataset of tobac merge/split cells with parent and child designations.


    Example usage:
        d = merge_split(Track)
        ds = standardize_track_dataset(Track, refl_mask)
        both_ds = xr.merge([ds, d],compat ='override')
        both_ds = compress_all(both_ds)
        both_ds.to_netcdf(os.path.join(savedir,'Track_features_merges.nc'))

    """

    try:
        import geopy
    except ImportError:
        geopy = None

    if geopy:
        from geopy.distance import geodesic
    else:
        print(
            "Geopy not available, Merge/Split will proceed in tobac general coordinates."
        )

    try:
        import networkx as nx
    except ImportError:
        networkx = None

    #     if networkx:
    #         import networkx as nx
    #     else:
    #         print("Cannot Merge/Split. Please install networkx.")
    import logging
    import numpy as np
    from pandas.core.common import flatten
    import xarray as xr

    # Check if dxy is in meters of in kilometers. It should be in meters
    if dxy <= 5:
        dxy *= 1000

    # Immediately convert pandas dataframe of track information to xarray:
    TRACK = TRACK.to_xarray()
    track_groups = TRACK.groupby("cell")
    cell_ids = {cid: len(v) for cid, v in track_groups.groups.items()}
    id_data = np.fromiter(cell_ids.keys(), dtype=int)
    count_data = np.fromiter(cell_ids.values(), dtype=int)
    all_frames = np.sort(np.unique(TRACK.frame))
    a_points = list()
    b_points = list()
    a_names = list()
    b_names = list()
    dist = list()

    if hasattr(TRACK, "longitude"):
        print("is in lat/lon")
        for i in id_data:
            a_pointx = track_groups[i].grid_longitude[-1].values
            a_pointy = track_groups[i].grid_latitude[-1].values
            for j in id_data:
                b_pointx = track_groups[j].grid_longitude[0].values
                b_pointy = track_groups[j].grid_latitude[0].values
                d = geodesic((a_pointy, a_pointx), (b_pointy, b_pointx)).m
                if d <= distance:
                    a_points.append([a_pointx, a_pointy])
                    b_points.append([b_pointx, b_pointy])
                    dist.append(d)
                    a_names.append(i)
                    b_names.append(j)
    else:

        for i in id_data:
            # print(i)
            a_pointx = track_groups[i].hdim_2[-1].values * dxy
            a_pointy = track_groups[i].hdim_1[-1].values * dxy
            for j in id_data:
                b_pointx = track_groups[j].hdim_2[0].values * dxy
                b_pointy = track_groups[j].hdim_1[0].values * dxy
                d = np.linalg.norm(
                    np.array((a_pointx, a_pointy)) - np.array((b_pointx, b_pointy))
                )
                if d <= distance:
                    a_points.append([a_pointx, a_pointy])
                    b_points.append([b_pointx, b_pointy])
                    dist.append(d)
                    a_names.append(i)
                    b_names.append(j)

    id = []
    # This is removing any tracks that have matched to themselves - e.g.,
    # a beginning of a track has found its tail.
    for i in range(len(dist) - 1, -1, -1):
        if a_names[i] == b_names[i]:
            id.append(i)
            a_points.pop(i)
            b_points.pop(i)
            dist.pop(i)
            a_names.pop(i)
            b_names.pop(i)
        else:
            continue

    g = nx.Graph()
    for i in np.arange(len(dist)):
        g.add_edge(a_names[i], b_names[i], weight=dist[i])

    tree = nx.minimum_spanning_edges(g)
    tree_list = list(tree)

    new_tree = []
    for i, j in enumerate(tree_list):
        frame_a = np.nanmax(track_groups[j[0]].frame.values)
        frame_b = np.nanmin(track_groups[j[1]].frame.values)
        if np.abs(frame_a - frame_b) <= frame_len:
            new_tree.append(tree_list[i][0:2])
    new_tree_arr = np.array(new_tree)

    TRACK["cell_parent_track_id"] = np.zeros(len(TRACK["cell"].values))
    cell_id = np.unique(
        TRACK.cell.values.astype(int)[~np.isnan(TRACK.cell.values.astype(int))]
    )
    track_id = dict()  # same size as number of total merged tracks

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

                for l in range(15):
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

    track_ids = np.array(list(track_id.keys()))
    logging.debug("found track ids")

    cell_id = list(
        np.unique(
            TRACK.cell.values.astype(int)[~np.isnan(TRACK.cell.values.astype(int))]
        )
    )
    logging.debug("found cell ids")

    cell_parent_track_id = []

    for i, id in enumerate(track_id):

        if len(track_id[int(id)]) == 1:
            cell_parent_track_id.append(int(id))

        else:
            cell_parent_track_id.append(np.repeat(int(id), len(track_id[int(id)])))

    cell_parent_track_id = list(flatten(cell_parent_track_id))
    logging.debug("found cell parent track ids")

    feature_parent_cell_id = list(TRACK.cell.values.astype(int))
    logging.debug("found feature parent cell ids")

    # This version includes all the feature regardless of if they are used in cells or not.
    feature_id = list(TRACK.feature.values.astype(int))
    logging.debug("found feature ids")

    feature_parent_track_id = []
    feature_parent_track_id = np.zeros(len(feature_id))
    for i, id in enumerate(feature_id):
        cellid = feature_parent_cell_id[i]
        if np.isnan(cellid):
            feature_parent_track_id[i] = -1
        else:
            j = np.where(cell_id == cellid)
            j = np.squeeze(j)
            trackid = cell_parent_track_id[j]
            feature_parent_track_id[i] = trackid

    logging.debug("found feature parent track ids")

    track_child_cell_count = []
    for i, id in enumerate(track_id):
        track_child_cell_count.append(len(track_id[int(id)]))
    logging.debug("found track child cell count")

    cell_child_feature_count = []
    for i, id in enumerate(cell_id):
        cell_child_feature_count.append(len(track_groups[id].feature.values))
    logging.debug("found cell child feature count")

    track_dim = "tracks"
    cell_dim = "cells"
    feature_dim = "features"

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

    assert len(cell_id) == len(cell_parent_track_id)
    assert len(feature_id) == len(feature_parent_cell_id)
    assert sum(track_child_cell_count) == len(cell_id)
    assert (
        sum(
            [
                sum(cell_child_feature_count[1:]),
                (len(np.where(feature_parent_track_id < 0)[0])),
            ]
        )
        == len(feature_id)
    )

    return d

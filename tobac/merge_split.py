"""
    Tobac merge and split
    This submodule is a post processing step to address tracked cells which merge/split. 
    The first iteration of this module is to combine the cells which are merging but have received
    a new cell id (and are considered a new cell) once merged. In general this submodule will label merged/split cells
    with a TRACK number in addition to its CELL number.
    
"""


def merge_split_MEST(TRACK, dxy, distance=None, frame_len=5):
    """
    function to  postprocess tobac track data for merge/split cells using a minimum euclidian spanning tree


    Parameters
    ----------
    TRACK : pandas.core.frame.DataFrame
        Pandas dataframe of tobac Track information

    dxy : float, mandatory
        The x/y grid spacing of the data.
        Should be in meters.


    distance : float, optional
        Distance threshold determining how close two features must be in order to consider merge/splitting.
        Default is 25x the x/y grid spacing of the data, given in dxy.
        The distance should be in units of meters.

    frame_len : float, optional
        Threshold for the maximum number of frames that can separate the end of cell and the start of a related cell.
        Default is five (5) frames.

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
    try:
        import networkx as nx
    except ImportError:
        networkx = None

    import logging
    import numpy as np
    from pandas.core.common import flatten
    import xarray as xr
    from scipy.spatial.distance import cdist

    # Immediately convert pandas dataframe of track information to xarray:
    TRACK = TRACK.to_xarray()
    track_groups = TRACK.groupby("cell")
    first = track_groups.first()
    last = track_groups.last()

    if distance is None:
        distance = dxy * 25.0

    a_names = list()
    b_names = list()
    dist = list()

    # write all sets of points (a and b) as Nx2 arrays
    l = len(last["hdim_2"].values)
    cells = first["cell"].values
    a_xy = np.zeros((l, 2))
    a_xy[:, 0] = last["hdim_2"].values * dxy
    a_xy[:, 1] = last["hdim_1"].values * dxy
    b_xy = np.zeros((l, 2))
    b_xy[:, 0] = first["hdim_2"].values * dxy
    b_xy[:, 1] = first["hdim_1"].values * dxy
    # Use cdist to find distance matrix
    out = cdist(a_xy, b_xy)
    # Find all cells under the distance threshold
    j = np.where(out <= distance)

    # Compile cells meeting the criteria to an array of both the distance and cell ids
    a_names = cells[j[0]]
    b_names = cells[j[1]]
    dist = out[j]

    # This is inputing data to the object which will perform the spanning tree.
    g = nx.Graph()
    for i in np.arange(len(dist)):
        g.add_edge(a_names[i], b_names[i], weight=dist[i])

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

    TRACK["cell_parent_track_id"] = np.zeros(len(TRACK["cell"].values))
    cell_id = np.unique(
        TRACK.cell.values.astype(int)[~np.isnan(TRACK.cell.values.astype(int))]
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

    cell_id = list(np.unique(TRACK.cell.values.astype(int)))
    logging.debug("found cell ids")

    cell_parent_track_id = np.zeros(len(cell_id))
    cell_parent_track_id[:] = -1

    for i, id in enumerate(track_id, start=0):
        for j in track_id[int(id)]:
            cell_parent_track_id[cell_id.index(j)] = int(i)

    logging.debug("found cell parent track ids")

    track_ids = np.array(np.unique(cell_parent_track_id))
    logging.debug("found track ids")

    feature_parent_cell_id = list(TRACK.cell.values.astype(int))
    logging.debug("found feature parent cell ids")

    #     # This version includes all the feature regardless of if they are used in cells or not.
    feature_id = list(TRACK.feature.values.astype(int))
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

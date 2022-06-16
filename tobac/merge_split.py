# Tobac merge and split v0.1

from geopy.distance import geodesic
from networkx import *
import numpy as np
from pandas.core.common import flatten
import xarray as xr


def compress_all(nc_grids, min_dims=2):
    for var in nc_grids:
        if len(nc_grids[var].dims) >= min_dims:
            # print("Compressing ", var)
            nc_grids[var].encoding["zlib"] = True
            nc_grids[var].encoding["complevel"] = 4
            nc_grids[var].encoding["contiguous"] = False
    return nc_grids


def standardize_track_dataset(TrackedFeatures, Mask, Projection):
    """Combine a feature mask with the feature data table into a common dataset.
    Also renames th

    returned by tobac.themes.tobac_v1.segmentation
    with the TrackedFeatures dataset returned by tobac.themes.tobac_v1.linking_trackpy.

    Also rename the variables to be more desciptive and comply with cf-tree.

    Convert the default cell parent ID  to an integer table.

    Add a cell dimension to reflect

    Projection is an xarray DataArray

    TODO: Add metadata attributes to

    """
    feature_standard_names = {
        # new variable name, and long description for the NetCDF attribute
        "frame": (
            "feature_time_index",
            "positional index of the feature along the time dimension of the mask, from 0 to N-1",
        ),
        "hdim_1": (
            "feature_hdim1_coordinate",
            "position of the feature along the first horizontal dimension in grid point space; a north-south coordinate for dim order (time, y, x)."
            "The numbering is consistent with positional indexing of the coordinate, but can be"
            "fractional, to account for a centroid not aligned to the grid.",
        ),
        "hdim_2": (
            "feature_hdim2_coordinate",
            "position of the feature along the second horizontal dimension in grid point space; an east-west coordinate for dim order (time, y, x)"
            "The numbering is consistent with positional indexing of the coordinate, but can be"
            "fractional, to account for a centroid not aligned to the grid.",
        ),
        "idx": ("feature_id_this_frame",),
        "num": (
            "feature_grid_cell_count",
            "Number of grid points that are within the threshold of this feature",
        ),
        "threshold_value": (
            "feature_threshold_max",
            "Feature number within that frame; starts at 1, increments by 1 to the number of features for each frame, and resets to 1 when the frame increments",
        ),
        "feature": (
            "feature_id",
            "Unique number of the feature; starts from 1 and increments by 1 to the number of features",
        ),
        "time": (
            "feature_time",
            "time of the feature, consistent with feature_time_index",
        ),
        "timestr": (
            "feature_time_str",
            "String representation of the feature time, YYYY-MM-DD HH:MM:SS",
        ),
        "projection_y_coordinate": (
            "feature_projection_y_coordinate",
            "y position of the feature in the projection given by ProjectionCoordinateSystem",
        ),
        "projection_x_coordinate": (
            "feature_projection_x_coordinate",
            "x position of the feature in the projection given by ProjectionCoordinateSystem",
        ),
        "lat": ("feature_latitude", "latitude of the feature"),
        "lon": ("feature_longitude", "longitude of the feature"),
        "ncells": (
            "feature_ncells",
            "number of grid cells for this feature (meaning uncertain)",
        ),
        "areas": ("feature_area",),
        "isolated": ("feature_isolation_flag",),
        "num_objects": ("number_of_feature_neighbors",),
        "cell": ("feature_parent_cell_id",),
        "time_cell": ("feature_parent_cell_elapsed_time",),
        "segmentation_mask": ("2d segmentation mask",),
    }
    new_feature_var_names = {
        k: feature_standard_names[k][0]
        for k in feature_standard_names.keys()
        if k in TrackedFeatures.variables.keys()
    }

    TrackedFeatures = TrackedFeatures.drop(["cell_parent_track_id"])
    # Combine Track and Mask variables. Use the 'feature' variable as the coordinate variable instead of
    # the 'index' variable and call the dimension 'feature'
    ds = xr.merge(
        [
            TrackedFeatures.swap_dims({"index": "feature"})
            .drop("index")
            .rename_vars(new_feature_var_names),
            Mask,
        ]
    )

    # Add the projection data back in
    ds["ProjectionCoordinateSystem"] = Projection

    # Convert the cell ID variable from float to integer
    if "int" not in str(TrackedFeatures.cell.dtype):
        # The raw output from the tracking is actually an object array
        # array([nan, 2, 3], dtype=object)
        # (and is cast to a float array when saved as NetCDF, I think).
        # Cast to float.
        int_cell = xr.zeros_like(TrackedFeatures.cell, dtype="int64")

        cell_id_data = TrackedFeatures.cell.astype("float64")
        valid_cell = np.isfinite(cell_id_data)
        valid_cell_ids = cell_id_data[valid_cell]
        if not (np.unique(valid_cell_ids) > 0).all():
            raise AssertionError(
                "Lowest cell ID cell is less than one, conflicting with use of zero to indicate an untracked cell"
            )
        int_cell[valid_cell] = valid_cell_ids.astype("int64")
        # ds['feature_parent_cell_id'] = int_cell
    return ds


def merge_split(TRACK, distance=25000, frame_len=5):
    """
    function to  postprocess tobac track data for merge/split cells
    Input:
        TRACK:    xarray dataset of tobac Track information

        distance:    float, optional distance threshold prior to adding a pair of points
                            into the minimum spanning tree. Default is 25000 meters.

        frame_len: float, optional threshold for the spanning length within which two points
                          can be separated. Default is five (5) frames.


    Output:
        d:    xarray dataset of
                        feature position along 1st horizontal dimension
        hdim2_index:    float
                        feature position along 2nd horizontal dimension

    Example:
        d = merge_split(Track)
       ds = standardize_track_dataset(Track, refl_mask, data['ProjectionCoordinateSystem'])
        # both_ds = xarray.combine_by_coords((ds,d), compat='override')
        both_ds = xr.merge([ds, d],compat ='override')
        both_ds = compress_all(both_ds)
        both_ds.to_netcdf(os.path.join(savedir,'Track_features_merges.nc'))

    """
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

    for i in id_data:
        # print(i)
        a_pointx = track_groups[i].projection_x_coordinate[-1].values
        a_pointy = track_groups[i].projection_y_coordinate[-1].values
        for j in id_data:
            b_pointx = track_groups[j].projection_x_coordinate[0].values
            b_pointy = track_groups[j].projection_y_coordinate[0].values
            d = np.linalg.norm(
                np.array((a_pointx, a_pointy)) - np.array((b_pointx, b_pointy))
            )
            if d <= distance:
                a_points.append([a_pointx, a_pointy])
                b_points.append([b_pointx, b_pointy])
                dist.append(d)
                a_names.append(i)
                b_names.append(j)

    #     for i in id_data:
    #         a_pointx = track_groups[i].grid_longitude[-1].values
    #         a_pointy = track_groups[i].grid_latitude[-1].values
    #         for j in id_data:
    #             b_pointx = track_groups[j].grid_longitude[0].values
    #             b_pointy = track_groups[j].grid_latitude[0].values
    #             d = geodesic((a_pointy,a_pointx),(b_pointy,b_pointx)).km
    #             if d <= distance:
    #                 a_points.append([a_pointx,a_pointy])
    #                 b_points.append([b_pointx, b_pointy])
    #                 dist.append(d)
    #                 a_names.append(i)
    #                 b_names.append(j)

    id = []
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

    g = Graph()
    for i in np.arange(len(dist)):
        g.add_edge(a_names[i], b_names[i], weight=dist[i])

    tree = minimum_spanning_edges(g)
    tree_list = list(minimum_spanning_edges(g))

    new_tree = []
    for i, j in enumerate(tree_list):
        frame_a = np.nanmax(track_groups[j[0]].frame.values)
        frame_b = np.nanmin(track_groups[j[1]].frame.values)
        if np.abs(frame_a - frame_b) <= frame_len:
            new_tree.append(tree_list[i][0:2])
    new_tree_arr = np.array(new_tree)

    TRACK["cell_parent_track_id"] = np.zeros(len(TRACK["cell"].values))
    cell_id = np.unique(
        TRACK.cell.values.astype(float)[~np.isnan(TRACK.cell.values.astype(float))]
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

    storm_id = [0]  # default because we don't track larger storm systems *yet*
    print("found storm id")

    track_parent_storm_id = np.repeat(
        0, len(track_id)
    )  # This will always be zero when we don't track larger storm systems *yet*
    print("found track parent storm ids")

    track_ids = np.array(list(track_id.keys()))
    print("found track ids")

    cell_id = list(
        np.unique(
            TRACK.cell.values.astype(float)[~np.isnan(TRACK.cell.values.astype(float))]
        )
    )
    print("found cell ids")

    cell_parent_track_id = []

    for i, id in enumerate(track_id):

        if len(track_id[int(id)]) == 1:
            cell_parent_track_id.append(int(id))

        else:
            cell_parent_track_id.append(np.repeat(int(id), len(track_id[int(id)])))

    cell_parent_track_id = list(flatten(cell_parent_track_id))
    print("found cell parent track ids")

    feature_parent_cell_id = list(TRACK.cell.values.astype(float))

    print("found feature parent cell ids")

    # This version includes all the feature regardless of if they are used in cells or not.
    feature_id = list(TRACK.feature.values.astype(int))
    print("found feature ids")

    feature_parent_storm_id = np.repeat(0, len(feature_id))  # we don't do storms atm
    print("found feature parent storm ids")

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

    print("found feature parent track ids")

    storm_child_track_count = [len(track_id)]
    print("found storm child track count")

    track_child_cell_count = []
    for i, id in enumerate(track_id):
        track_child_cell_count.append(len(track_id[int(id)]))
    print("found track child cell count")

    cell_child_feature_count = []
    for i, id in enumerate(cell_id):
        cell_child_feature_count.append(len(track_groups[id].feature.values))
    print("found cell child feature count")

    storm_child_cell_count = [len(cell_id)]
    storm_child_feature_count = [len(feature_id)]

    storm_dim = "nstorms"
    track_dim = "ntracks"
    cell_dim = "ncells"
    feature_dim = "nfeatures"

    d = xr.Dataset(
        {
            "storm_id": (storm_dim, storm_id),
            "track_id": (track_dim, track_ids),
            "track_parent_storm_id": (track_dim, track_parent_storm_id),
            "cell_id": (cell_dim, cell_id),
            "cell_parent_track_id": (cell_dim, cell_parent_track_id),
            "feature_id": (feature_dim, feature_id),
            "feature_parent_cell_id": (feature_dim, feature_parent_cell_id),
            "feature_parent_track_id": (feature_dim, feature_parent_track_id),
            "feature_parent_storm_id": (feature_dim, feature_parent_storm_id),
            "storm_child_track_count": (storm_dim, storm_child_track_count),
            "storm_child_cell_count": (storm_dim, storm_child_cell_count),
            "storm_child_feature_count": (storm_dim, storm_child_feature_count),
            "track_child_cell_count": (track_dim, track_child_cell_count),
            "cell_child_feature_count": (cell_dim, cell_child_feature_count),
        }
    )
    d = d.set_coords(["feature_id", "cell_id", "track_id", "storm_id"])

    assert len(track_id) == len(track_parent_storm_id)
    assert len(cell_id) == len(cell_parent_track_id)
    assert len(feature_id) == len(feature_parent_cell_id)
    assert sum(storm_child_track_count) == len(track_id)
    assert sum(storm_child_cell_count) == len(cell_id)
    assert sum(storm_child_feature_count) == len(feature_id)
    assert sum(track_child_cell_count) == len(cell_id)
    assert sum(
        [sum(cell_child_feature_count), (len(np.where(feature_parent_track_id < 0)[0]))]
    ) == len(feature_id)

    return d

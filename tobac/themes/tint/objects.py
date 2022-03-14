import numpy as np
import pandas as pd

from scipy import ndimage
from .grid_utils import get_filtered_frame

# Tracking Parameter Defaults
default_params = {
    "FIELD_THRESH": 32,
    "ISO_THRESH": 8,
    "ISO_SMOOTH": 3,
    "MIN_SIZE": 8,
    "SEARCH_MARGIN": 4000,
    "FLOW_MARGIN": 10000,
    "MAX_DISPARITY": 999,
    "MAX_FLOW_MAG": 50,
    "MAX_SHIFT_DISP": 15,
    "GS_ALT": 1500,
}


def get_object_center(obj_id, labeled_image):
    """Returns index of center pixel of the given object id from labeled
    image. The center is calculated as the median pixel of the object extent;
    it is not a true centroid."""
    obj_index = np.argwhere(labeled_image == obj_id)
    center = np.median(obj_index, axis=0).astype("i")
    return center


def check_isolation(raw, filtered, grid_size, params):
    """Returns list of booleans indicating object isolation. Isolated objects
    are not connected to any other objects by pixels greater than ISO_THRESH,
    and have at most one peak."""
    nobj = np.max(filtered)
    min_size = params["MIN_SIZE"] / np.prod(grid_size[1:] / 1000)
    iso_filtered = get_filtered_frame(raw, min_size, params["ISO_THRESH"])
    nobj_iso = np.max(iso_filtered)
    iso = np.empty(nobj, dtype="bool")

    for iso_id in np.arange(nobj_iso) + 1:
        obj_ind = np.where(iso_filtered == iso_id)
        objects = np.unique(filtered[obj_ind])
        objects = objects[objects != 0]

        if len(objects) == 1 and single_max(obj_ind, raw, params):
            iso[objects - 1] = True
        else:
            iso[objects - 1] = False
    return iso


def get_obj_extent(labeled_image, obj_label):
    """Takes in labeled image and finds the radius, area, and center of the
    given object."""
    obj_index = np.argwhere(labeled_image == obj_label)

    xlength = np.max(obj_index[:, 0]) - np.min(obj_index[:, 0]) + 1
    ylength = np.max(obj_index[:, 1]) - np.min(obj_index[:, 1]) + 1
    obj_radius = np.max((xlength, ylength)) / 2
    obj_center = np.round(np.median(obj_index, axis=0), 0)
    obj_area = len(obj_index[:, 0])

    obj_extent = {
        "obj_center": obj_center,
        "obj_radius": obj_radius,
        "obj_area": obj_area,
        "obj_index": obj_index,
    }
    return obj_extent


def init_current_objects(first_frame, second_frame, pairs, counter):
    """Returns a dictionary for objects with unique ids and their
    corresponding ids in frame1 and frame1. This function is called when
    echoes are detected after a period of no echoes."""
    nobj = np.max(first_frame)

    id1 = np.arange(nobj) + 1
    uid = counter.next_uid(count=nobj)
    id2 = pairs
    obs_num = np.zeros(nobj, dtype="i")
    origin = np.array(["-1"] * nobj)

    current_objects = {
        "id1": id1,
        "uid": uid,
        "id2": id2,
        "obs_num": obs_num,
        "origin": origin,
    }
    current_objects = attach_last_heads(first_frame, second_frame, current_objects)
    return current_objects, counter


def update_current_objects(frame1, frame2, pairs, old_objects, counter):
    """Removes dead objects, updates living objects, and assigns new uids to
    new-born objects."""
    nobj = np.max(frame1)
    id1 = np.arange(nobj) + 1
    uid = np.array([], dtype="str")
    obs_num = np.array([], dtype="i")
    origin = np.array([], dtype="str")

    for obj in np.arange(nobj) + 1:
        if obj in old_objects["id2"]:
            obj_index = old_objects["id2"] == obj
            uid = np.append(uid, old_objects["uid"][obj_index])
            obs_num = np.append(obs_num, old_objects["obs_num"][obj_index] + 1)
            origin = np.append(origin, old_objects["origin"][obj_index])
        else:
            #  obj_orig = get_origin_uid(obj, frame1, old_objects)
            obj_orig = "-1"
            origin = np.append(origin, obj_orig)
            if obj_orig != "-1":
                uid = np.append(uid, counter.next_cid(obj_orig))
            else:
                uid = np.append(uid, counter.next_uid())
            obs_num = np.append(obs_num, 0)

    id2 = pairs
    current_objects = {
        "id1": id1,
        "uid": uid,
        "id2": id2,
        "obs_num": obs_num,
        "origin": origin,
    }
    current_objects = attach_last_heads(frame1, frame2, current_objects)
    return current_objects, counter


def single_max(obj_ind, raw, params):
    """Returns True if object has at most one peak."""
    if len(raw.shape) == 3:
        max_proj = np.max(raw, axis=0)
    else:
        max_proj = raw
    smooth = ndimage.filters.gaussian_filter(max_proj, params["ISO_SMOOTH"])
    padded = np.pad(smooth, 1, mode="constant")
    obj_ind = [axis + 1 for axis in obj_ind]  # adjust for padding
    maxima = 0
    for pixel in range(len(obj_ind[0])):
        ind_0 = obj_ind[0][pixel]
        ind_1 = obj_ind[1][pixel]
        neighborhood = padded[(ind_0 - 1) : (ind_0 + 2), (ind_1 - 1) : (ind_1 + 2)]
        max_ind = np.unravel_index(neighborhood.argmax(), neighborhood.shape)
        if max_ind == (1, 1):
            maxima += 1
            if maxima > 1:
                return False
    return True


def attach_last_heads(frame1, frame2, current_objects):
    """Attaches last heading information to current_objects dictionary."""
    nobj = len(current_objects["uid"])
    heads = np.ma.empty((nobj, 2))
    for obj in range(nobj):
        if (current_objects["id1"][obj] > 0) and (current_objects["id2"][obj] > 0):
            center1 = get_object_center(current_objects["id1"][obj], frame1)
            center2 = get_object_center(current_objects["id2"][obj], frame2)
            heads[obj, :] = center2 - center1
        else:
            heads[obj, :] = np.ma.array([-999, -999], mask=[True, True])

    current_objects["last_heads"] = heads
    return current_objects


def get_object_prop(image1, grid1, field, record, params):
    """Returns dictionary of object properties for all objects found in
    image1."""
    id1 = []
    center = []
    grid_x = []
    grid_y = []
    area = []
    longitude = []
    latitude = []
    field_max = []
    max_height = []
    volume = []
    nobj = np.max(image1)

    unit_dim = record.grid_size
    unit_alt = unit_dim[0] / 1000
    unit_area = (unit_dim[1] * unit_dim[2]) / (1000**2)
    unit_vol = (unit_dim[0] * unit_dim[1] * unit_dim[2]) / (1000**3)

    raw3D = grid1[field].values

    for obj in np.arange(nobj) + 1:
        obj_index = np.argwhere(image1 == obj)
        id1.append(obj)

        # 2D frame stats
        center.append(np.median(obj_index, axis=0))
        this_centroid = np.round(np.mean(obj_index, axis=0), 3)
        grid_x.append(this_centroid[1])
        grid_y.append(this_centroid[0])
        area.append(obj_index.shape[0] * unit_area)

        rounded = np.round(this_centroid).astype("i")
        cent_met = np.array([grid1.y.values[rounded[0]], grid1.x.values[rounded[1]]])
        if len(grid1.point_latitude.values.shape) == 3:
            lat = grid1.point_latitude.values[0, rounded[0], 0]
            lon = grid1.point_longitude.values[0, 0, rounded[1]]
        else:
            lat = grid1.point_latitude.values[rounded[0], 0]
            lon = grid1.point_longitude.values[0, rounded[1]]

        longitude.append(np.round(lon, 4))
        latitude.append(np.round(lat, 4))

        # raw 3D grid stats
        if len(raw3D.shape) == 3:
            obj_slices = [raw3D[:, ind[0], ind[1]] for ind in obj_index]
            filtered_slices = [
                obj_slice > params["FIELD_THRESH"] for obj_slice in obj_slices
            ]
            heights = [np.arange(raw3D.shape[0])[ind] for ind in filtered_slices]
            max_height.append(np.max(np.concatenate(heights)) * unit_alt)
            volume.append(np.sum(filtered_slices) * unit_vol)
            field_max.append(np.max(obj_slices))
        else:
            obj_slices = [raw3D[ind[0], ind[1]] for ind in obj_index]
            max_height.append(np.nan)
            volume.append(np.nan)
            field_max.append(np.max(obj_slices))
    # cell isolation
    isolation = check_isolation(raw3D, image1, record.grid_size, params)

    objprop = {
        "id1": id1,
        "center": center,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "area": area,
        "field_max": field_max,
        "max_height": max_height,
        "volume": volume,
        "lon": longitude,
        "lat": latitude,
        "isolated": isolation,
    }
    return objprop


def write_tracks(old_tracks, record, current_objects, obj_props):
    """Writes all cell information to tracks dataframe."""
    print("Writing tracks for scan", record.scan)

    nobj = len(obj_props["id1"])
    scan_num = [record.scan] * nobj
    uid = current_objects["uid"]

    new_tracks = pd.DataFrame(
        {
            "scan": scan_num,
            "cell": uid,
            "time": record.time,
            "grid_x": obj_props["grid_x"],
            "grid_y": obj_props["grid_y"],
            "longitude": obj_props["lon"],
            "latitude": obj_props["lat"],
            "area": obj_props["area"],
            "vol": obj_props["volume"],
            "max": obj_props["field_max"],
            "max_alt": obj_props["max_height"],
            "isolated": obj_props["isolated"],
        }
    )
    new_tracks.set_index(["scan"], inplace=True)
    tracks = old_tracks.append(new_tracks)
    return tracks

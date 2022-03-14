"""
tint.matching
=============

Functions for object matching between adjacent radar scans.

"""

import numpy as np

from scipy import optimize

from .phase_correlation import get_ambient_flow
from .objects import get_obj_extent


LARGE_NUM = 1000


def get_object_center(obj_id, labeled_image):
    """Returns index of center pixel of the given object id from labeled
    image. The center is calculated as the median pixel of the object extent;
    it is not a true centroid."""
    obj_index = np.argwhere(labeled_image == obj_id)
    center = np.median(obj_index, axis=0).astype("i")
    return center


def euclidean_dist(vec1, vec2):
    """Computes euclidean distance."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dist = np.sqrt(sum((vec1 - vec2) ** 2))
    return dist


def get_sizeChange(size1, size2):
    """Returns change in size of an echo as the ratio of the larger size to
    the smaller, minus 1."""
    if (size1 < 5) and (size2 < 5):
        return 0
    elif size1 >= size2:
        return size1 / size2 - 1
    else:
        return size2 / size1 - 1


def find_objects(search_box, image2):
    """Identifies objects found in the search region."""
    if not search_box["valid"]:
        obj_found = np.array(-1)
    else:
        search_area = image2[
            search_box["x1"] : search_box["x2"], search_box["y1"] : search_box["y2"]
        ]
        obj_found = np.unique(search_area)
    return obj_found


def shifts_disagree(shift1, shift2, record, params):
    """Returns True if shift disparity greater than MAX_SHIFT_DISP
    parameter."""
    shift1 = shift1 * record.grid_size[1:]
    shift2 = shift2 * record.grid_size[1:]
    shift_disparity = euclidean_dist(shift1, shift2)
    return shift_disparity / record.interval.seconds > params["MAX_SHIFT_DISP"]


def clip_shift(shift, record, params):
    """Clips shift according to MAX_FLOW_MAG parameter."""
    shift_meters = shift * record.grid_size[1:]
    shift_mag = np.linalg.norm(shift_meters)
    velocity = shift_mag / record.interval.seconds
    unit = shift_meters / shift_mag
    if velocity > params["MAX_FLOW_MAG"]:
        clipped = unit * params["MAX_FLOW_MAG"] * record.interval.seconds
        clipped_pix = clipped / record.grid_size[1:]
        return clipped_pix
    else:
        return shift


def correct_shift(local_shift, current_objects, obj_id1, global_shift, record, params):
    """Takes in flow vector based on local phase correlation (see
    get_std_flow) and compares it to the last headings of the object and
    the global_shift vector for that timestep. Corrects accordingly.
    Note: At the time of this function call, current_objects has not yet been
    updated for the current frame1 and frame2, so the id2s in current_objects
    correspond to the objects in the current frame1."""
    global_shift = clip_shift(global_shift, record, params)

    if current_objects is None:
        last_heads = None
    else:
        obj_index = current_objects["id2"] == obj_id1
        last_heads = current_objects["last_heads"][obj_index].flatten()
        last_heads = np.round(last_heads * record.interval_ratio, 2)
        if len(last_heads) == 0:
            last_heads = None

    if last_heads is None:
        if shifts_disagree(local_shift, global_shift, record, params):
            case = 0
            corrected_shift = global_shift
        else:
            case = 1
            corrected_shift = (local_shift + global_shift) / 2
    elif shifts_disagree(local_shift, last_heads, record, params):
        if shifts_disagree(local_shift, global_shift, record, params):
            case = 2
            corrected_shift = last_heads
        else:
            case = 3
            corrected_shift = local_shift
    else:
        case = 4
        corrected_shift = (local_shift + last_heads) / 2

    corrected_shift = np.round(corrected_shift, 2)

    record.count_case(case)
    record.record_shift(corrected_shift, global_shift, last_heads, local_shift, case)
    return corrected_shift


def predict_search_extent(obj1_extent, shift, params, grid_size):
    """Predicts search extent/region for the object in image2 given
    the image shift."""
    shifted_center = obj1_extent["obj_center"] + shift
    search_radius_r = params["SEARCH_MARGIN"] / grid_size[1]
    search_radius_c = params["SEARCH_MARGIN"] / grid_size[2]
    x1 = shifted_center[0] - search_radius_r
    x2 = shifted_center[0] + search_radius_r + 1
    y1 = shifted_center[1] - search_radius_c
    y2 = shifted_center[1] + search_radius_c + 1
    x1 = np.int(x1)
    x2 = np.int(x2)
    y1 = np.int(y1)
    y2 = np.int(y2)
    return {
        "x1": x1,
        "x2": x2,
        "y1": y1,
        "y2": y2,
        "center_pred": shifted_center,
        "valid": True,
    }


def check_search_box(search_box, img_dims):
    """Checks if search_box is within the boundaries of the frame. Clips to
    edges of frame if out of bounds. Marks as invalid if too small."""
    if search_box["x1"] < 0:
        search_box["x1"] = 0
    if search_box["y1"] < 0:
        search_box["y1"] = 0
    if search_box["x2"] > img_dims[0]:
        search_box["x2"] = img_dims[0]
    if search_box["y2"] > img_dims[1]:
        search_box["y2"] = img_dims[1]
    if (search_box["x2"] - search_box["x1"] < 5) or (
        search_box["y2"] - search_box["y1"] < 5
    ):
        search_box["valid"] = False
    return search_box


def get_disparity(obj_found, image2, search_box, obj1_extent):
    """Computes disparities for objects in obj_found."""
    dist_pred = np.empty(0)
    change = np.empty(0)
    for target_obj in obj_found:
        target_extent = get_obj_extent(image2, target_obj)
        euc_dist = euclidean_dist(
            target_extent["obj_center"], search_box["center_pred"]
        )
        dist_pred = np.append(dist_pred, euc_dist)
        size_changed = get_sizeChange(
            target_extent["obj_area"], obj1_extent["obj_area"]
        )
        change = np.append(change, size_changed)

    disparity = dist_pred + change
    return disparity


def get_disparity_all(obj_found, image2, search_box, obj1_extent):
    """Returns disparities of all objects found within the search box."""
    if np.max(obj_found) <= 0:
        disparity = np.array([LARGE_NUM])
    else:
        obj_found = obj_found[obj_found > 0]
        disparity = get_disparity(obj_found, image2, search_box, obj1_extent)
    return disparity


def save_obj_match(obj_id1, obj_found, disparity, obj_match, params):
    """Saves disparity values in obj_match matrix. If disparity is greater
    than MAX_DISPARITY, saves a large number."""
    disparity[disparity > params["MAX_DISPARITY"]] = LARGE_NUM
    if np.max(obj_found) > 0:
        obj_found = obj_found[obj_found > 0]
        obj_found = obj_found - 1
        obj_id1 = obj_id1 - 1
        obj_match[obj_id1, obj_found] = disparity
    return obj_match


def locate_all_objects(image1, image2, global_shift, current_objects, record, params):
    """Matches all the objects in image1 to objects in image2. This is the
    main function called on a pair of images."""
    nobj1 = np.max(image1)
    nobj2 = np.max(image2)

    if (nobj2 == 0) or (nobj1 == 0):
        print("No echoes to track!")
        return

    obj_match = np.full((nobj1, np.max((nobj1, nobj2))), LARGE_NUM, dtype="f")

    for obj_id1 in np.arange(nobj1) + 1:
        obj1_extent = get_obj_extent(image1, obj_id1)
        shift = get_ambient_flow(obj1_extent, image1, image2, params, record.grid_size)
        if shift is None:
            record.count_case(5)
            shift = global_shift

        shift = correct_shift(
            shift, current_objects, obj_id1, global_shift, record, params
        )

        search_box = predict_search_extent(obj1_extent, shift, params, record.grid_size)
        search_box = check_search_box(search_box, image2.shape)
        objs_found = find_objects(search_box, image2)
        disparity = get_disparity_all(objs_found, image2, search_box, obj1_extent)
        obj_match = save_obj_match(obj_id1, objs_found, disparity, obj_match, params)
    return obj_match


def match_pairs(obj_match, params):
    """Matches objects into pairs given a disparity matrix and removes
    bad matches. Bad matches have a disparity greater than the maximum
    threshold."""
    pairs = optimize.linear_sum_assignment(obj_match)

    for id1 in pairs[0]:
        if obj_match[id1, pairs[1][id1]] > params["MAX_DISPARITY"]:
            pairs[1][id1] = -1  # -1 indicates the object has died

    pairs = pairs[1] + 1  # ids in current_objects are 1-indexed
    return pairs


def get_pairs(image1, image2, global_shift, current_objects, record, params):
    """Given two images, this function identifies the matching objects and
    pairs them appropriately. See disparity function."""
    nobj1 = np.max(image1)
    nobj2 = np.max(image2)

    if nobj1 == 0:
        print("No echoes found in the first scan.")
        return
    elif nobj2 == 0:
        zero_pairs = np.zeros(nobj1)
        return zero_pairs

    obj_match = locate_all_objects(
        image1, image2, global_shift, current_objects, record, params
    )
    pairs = match_pairs(obj_match, params)
    return pairs

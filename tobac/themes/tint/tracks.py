import numpy as np
import pandas as pd

from .grid_utils import get_grid_size, extract_grid_data
from .phase_correlation import get_global_shift
from .matching import get_pairs
from .objects import init_current_objects, update_current_objects
from .objects import get_object_prop, write_tracks
from .objects import default_params
from .helpers import Record, Counter


def make_tracks(grid_ds, field, params=None):
    """
    Use TINT's phase correlation tracker to make tracks. Outputs will be in the form
    of pandas DataFrames using tobac's naming conventions.

    Args:
        grid_ds:
        params:

    Returns:

    """
    if params is None:
        params = default_params

    newRain = True

    # Go through each time period
    times = grid_ds.time.values
    grid_size = get_grid_size(grid_ds)
    record = Record(grid_ds)
    grid_obj2 = grid_ds.isel(time=0)
    raw2, frame2 = extract_grid_data(grid_obj2, field, grid_size, params)
    current_objects = None
    counter = Counter()
    tracks = pd.DataFrame()

    for i in range(1, len(times)):
        raw1 = raw2
        frame1 = frame2
        grid_obj1 = grid_obj2
        grid_obj2 = grid_ds.isel(time=i)
        if np.max(frame1) == 0:
            print("No cells found in scan %d" % i)
            current_objects = None
            newRain = True
            continue
        record.update_scan_and_time(grid_obj1, grid_obj2)

        raw2, frame2 = extract_grid_data(
            grid_obj2, field, grid_size, params)
        global_shift = get_global_shift(raw1, raw2, params)
        pairs = get_pairs(frame1, frame2, global_shift, current_objects, record, params)
        if newRain:
            # first nonempty scan after a period of empty scans
            current_objects, counter = init_current_objects(
                frame1, frame2, pairs, counter)
            newRain = False
        else:
            current_objects, counter = update_current_objects(
                frame1, frame2, pairs, current_objects, counter)

        obj_props = get_object_prop(frame1, grid_obj1, field,
                                    record, params)
        record.add_uids(current_objects)
        tracks = write_tracks(
            tracks, record, current_objects, obj_props)

    record.update_scan_and_time(grid_obj1)
    tracks = tracks.to_xarray()
    tracks.attrs["cf_tree_order"] = "0 1 2"
    tracks.attrs["tree_id"] = grid_ds.attrs["tree_id"]
    print(np.unique(tracks["cell"].values))
    tracks["cell_id"] = ('cell', np.unique(tracks["cell"].values).sort())
    tracks["cell_id"].attrs["parent"] = "storm_id"
    tracks["cell_id"].attrs["parent_id"] = "cell_parent_storm_id"
    tracks["cell_parent_storm_id"] = grid_ds["storm_id"]
    tracks["cell_mask"]


    # Add hierarchy

    return tracks


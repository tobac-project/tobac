import numpy as np
import pandas as pd
import xarray as xr

from .grid_utils import get_grid_size, extract_grid_data, extract_grid_data_2d
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
            The xarray dataset containing the grids.
        params:
            The input parameters.
    Returns:
        grid_ds
            The xarray dataset containing the cell information.
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
    cell_mask = xr.DataArray(
        np.zeros((len(times), frame2.shape[0], frame2.shape[1])),
        dims=('time', 'x', 'y'))
    for i in range(1, len(times)):
        raw1 = raw2
        frame1 = frame2
        grid_obj1 = grid_obj2
        grid_obj2 = grid_ds.isel(time=i)
        record.update_scan_and_time(grid_obj1, grid_obj2)        
        raw2, frame2 = extract_grid_data(
            grid_obj2, field, grid_size, params)
        if np.max(frame1) == 0:
            print("No cells found in scan %d" % i)
            current_objects = None
            newRain = True
            continue

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
        cell_mask[i] = frame1
         
    record.update_scan_and_time(grid_obj1)
    tracks = tracks.set_index(['cell'])
    tracks = tracks.to_xarray()
    tracks["cell_time"] = tracks["time"]
    tracks = tracks.drop("time")
    tracks["time"] = grid_ds.time.astype('datetime64[s]')
    tracks.attrs["cf_tree_order"] = "storm_id cell_id"
    tracks.attrs["tree_id"] = grid_ds.attrs["tree_id"]
    tracks["cell_id"] = tracks["cell"]
    tracks["cell_id"].attrs["parent"] = "storm_id"
    tracks["cell_id"].attrs["parent_id"] = "cell_parent_storm_id"
    tracks["cell_parent_storm_id"] = grid_ds["storm_id"]
    tracks["cell_mask"] = cell_mask.astype(int)
    tracks["cell_mask"].attrs["cf_role"] = grid_ds.attrs["tree_id"]
    tracks["cell_mask"].attrs["long_name"] = "cell ID for this grid cell"
    tracks["cell_mask"].attrs['coordinates'] = 'cell_id time latitude longitude'    
    # Add hierarchy

    return tracks


def make_tracks_2d_field(grid_ds, field, params=None):
    """
    Use TINT's phase correlation tracker to make tracks. Outputs will be in the form
    of pandas DataFrames using tobac's naming conventions. This is for 2D fields.

    Args:
        grid_ds:
            The xarray dataset containing the grids.
        params:
            The input parameters.
    Returns:
        grid_ds
            The xarray dataset containing the cell information.
    """
    if params is None:
        params = default_params

    newRain = True

    # Go through each time period
    times = grid_ds.Time.values
    grid_obj2 = grid_ds.isel(Time=0)
    raw2, frame2 = extract_grid_data_2d(grid_obj2, field, params)
    
    record = Record(grid_ds)
    record.grid_size = np.array([1,
        grid_obj2.attrs["DY"], 
        grid_obj2.attrs["DX"]])
    current_objects = None
    counter = Counter()
    tracks = pd.DataFrame()
    cell_mask = xr.DataArray(
        np.zeros((len(times), frame2.shape[0], frame2.shape[1])),
        dims=('Time', 'x', 'y'))
    
    for i in range(1, len(times)):
        raw1 = raw2
        frame1 = frame2
        grid_obj1 = grid_obj2
        grid_obj2 = grid_ds.isel(Time=i)
        record.update_scan_and_time(grid_obj1, grid_obj2)

        raw2, frame2 = extract_grid_data_2d(
            grid_obj2, field, params)
        if np.max(frame1) == 0:
            print("No cells found in scan %d" % i)
            current_objects = None
            newRain = True
            continue
        print(frame2.shape)
        print(frame1.shape)
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
        cell_mask[i] = frame1

    record.update_scan_and_time(grid_obj1)
    tracks = tracks.set_index(['cell'])
    tracks = tracks.to_xarray()
    tracks["cell_time"] = tracks["time"]
    tracks = tracks.drop("time")
    tracks["time"] = grid_ds.time.astype('datetime64[s]')
    tracks.attrs["cf_tree_order"] = "storm_id cell_id"
    tracks.attrs["tree_id"] = grid_ds.attrs["tree_id"]
    tracks["cell_id"] = tracks["cell"]
    tracks["cell_id"].attrs["parent"] = "storm_id"
    tracks["cell_id"].attrs["parent_id"] = "cell_parent_storm_id"
    tracks["cell_parent_storm_id"] = grid_ds["storm_id"]
    tracks["cell_mask"] = cell_mask.astype(int)
    tracks["cell_mask"].attrs["cf_role"] = grid_ds.attrs["tree_id"]
    tracks["cell_mask"].attrs["long_name"] = "cell ID for this grid cell"
    tracks["cell_mask"].attrs['coordinates'] = 'cell_id time latitude longitude'

    return tracks


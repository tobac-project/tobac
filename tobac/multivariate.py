"""Share the labels of one tracked field with another, where they overlap.

For example, we may want to track updrafts and condensate separately, but then link each updraft with the condensate it coincides with.

Mathilde Ritman, 2026 (mathilde.ritman@phyiscs.ox.ac.uk)

References
----------
Ritman, M., Jones, W., and Stier, P.: Convective controls on anvil cloud evolution in the ICON km-scale global climate model, Atmos. Chem. Phys., 26, 7105-7126, https://doi.org/10.5194/acp-26-7105-2026, 2026.

"""

import xarray as xr
import numpy as np
from scipy import ndimage as ndi
import dask
import joblib
import logging


def get_multivariate_label_maps(child_mask, child_tracks, parent_mask, parent_tracks):
    """Find label mappings from child to parent features where they overlap, then store them in the track tables.

    Parameters
    ----------
    child_mask : xarray.DataArray
        DataArray containing the mask to update (int > 0 where belonging to area/volume of feature, 0 else).

    child_tracks : pandas.DataFrame
        DataFrame containing the corresponding labels and tracking details of the tracked child features.

    parent_mask : xarray.DataArray
        DataArray containing the mask to use to update the child mask (int > 0 where belonging to area/volume of feature, 0 else).

    parent_tracks : pandas.DataFrame
        DataFrame containing the corresponding labels and tracking details of the tracked parent features.

    Returns
    -------
    child_tracks_out : pandas.DataFrame
        DataFrame containing the new label asasignments.

    parent_tracks_out : pandas.DataFrame
        DataFrame containing label asasignments for when a child had multiple parents but was only assigned to one.

    """

    child_name = child_mask.name  # expect "cell" / "updraft"
    parent_name = parent_mask.name

    child_tracks[child_name] = np.where(
        child_tracks[child_name] > 0, child_tracks[child_name], 0
    )
    parent_tracks[parent_name] = np.where(
        parent_tracks[parent_name] > 0, parent_tracks[parent_name], 0
    )

    # list all labels to update
    child_labels = np.unique(child_tracks[child_name].values)
    parent_labels = np.unique(parent_tracks[parent_name].values)
    NAN_VAL = (
        max(
            child_labels.max(initial=0),
            parent_labels.max(initial=0),
        )
        + 1000
    )

    logging.info("Start calculating the coincident label mappings")

    child_mask = child_mask.where(child_mask > 0).fillna(NAN_VAL).astype(int)
    parent_mask = parent_mask.where(parent_mask > 0).fillna(NAN_VAL).astype(int)
    ntimes = child_mask.time.size if "time" in child_mask.dims else 1

    def get_arrays_at_time_t(t, unique_child_values=None):
        child_arr = (
            child_mask.isel(time=t).values.reshape(-1)
            if ntimes > 1
            else child_mask.values.reshape(-1)
        )
        parent_arr = (
            parent_mask.isel(time=t).values.reshape(-1)
            if ntimes > 1
            else parent_mask.values.reshape(-1)
        )
        if unique_child_values is None:
            unique_child_values = [
                x for x in np.unique(child_arr) if x > 0 and x != NAN_VAL
            ]
        label_arrays = {
            "child": child_arr,
            "parent": parent_arr,
        }
        return label_arrays, unique_child_values

    all_child_mappings = []
    all_child_values = []

    for t in range(ntimes):
        label_arrays, unique_child_values = get_arrays_at_time_t(
            t, unique_child_values=None
        )
        label_arrays = {
            k: {
                **label_arrays,
                "index": k,
            }
            for k in unique_child_values
        }
        child_mappings = joblib.Parallel(n_jobs=-1, prefer="threads")(
            joblib.delayed(find_mappings_for_chunk)(label_arrays[k], NAN_VAL)
            for k in unique_child_values
        )

        all_child_values.extend(unique_child_values)  # results
        all_child_mappings.extend(child_mappings)

    # drop repeated mappings
    final_child_mappings, resulting_parent_mappings = drop_repeated_mappings(
        all_child_values, all_child_mappings, NAN_VAL
    )

    child_tracks["multivariate"] = (
        child_tracks[child_name].map(final_child_mappings).fillna(0).astype(int)
    )  # record mapping as new dataframe column

    # sometimes a child actually had multiple parents (e.g., it's parent at time 0 is different to at time 1), in these cases, we record only one parent for the child, and those parents that got left out are recorded here:

    parent_tracks["multivariate"] = (
        parent_tracks[parent_name].map(resulting_parent_mappings).fillna(0).astype(int)
    )

    logging.info("Finished calculating the coincident label mappings")

    return child_tracks, parent_tracks


def find_mappings_for_chunk(chunk, nan_val):
    """Use scipy.ndi labeled_comprehension to find the minimum parent label overlapping each child label in one chunk.

    Parameters
    ----------
    chunk : dict
        Containing nump.Arrays "parent", "child" and "index".

    nan_val : integer
        Must be larger that the maximum label value in either child or parent.

    Returns
    -------
    result : numpy.Array
        Minimum parent label that coincides with each child label in the chunk.

    """
    return ndi.labeled_comprehension(
        input=chunk["parent"],
        labels=chunk["child"],
        index=chunk["index"],
        func=np.min,
        out_dtype=np.int64,
        default=nan_val,
        pass_positions=False,
    )


def drop_repeated_mappings(list_A, list_B, nan_val):
    """Collapse repeated child-to-parent mappings to a single parent per child and record the discarded alternatives."""

    a_mappings = {}  # i: min(j); for mapping i->min(j)
    j_values = {}  # i: all js;  for mapping j->min(j)
    for i, j in zip(list_A, list_B):
        if i in a_mappings:
            a_mappings[i] = min(a_mappings[i], j)  # parent to min j
            j_values[i] += [j]  # Collect all j values for i
        else:
            a_mappings[i] = j  # First occurrence of i
            j_values[i] = [j]  # Start collecting j values for i
    # collect map values to format j: min(j)
    b_mappings = {}
    for i, js in j_values.items():
        min_j = a_mappings[i]
        for j in js:
            b_mappings[j] = min_j

    # ensure validity
    def valid_maps(di):
        keys = np.array(list(di.keys())).astype(int)
        vals = np.array(list(di.values()))
        vals = np.where(vals != nan_val, vals, 0).astype(int)  # only valid mappings
        return dict(zip(keys, vals))

    return valid_maps(a_mappings), valid_maps(b_mappings)


def apply_multivariate_label_maps(child_mask, child_tracks, parent_mask, parent_tracks):
    """Relabel child and parent masks using the mappings stored in the track tables.

    Parameters
    ----------
    child_mask : xarray.DataArray
        DataArray containing the mask to update (int > 0 where belonging to area/volume of feature, 0 else).

    child_tracks : pandas.DataFrame
        DataFrame containing mappings from child labels to parent labels.

    parent_mask : xarray.DataArray
        DataArray containing the mask to use to update the child mask (int > 0 where belonging to area/volume of feature, 0 else).

    parent_tracks : pandas.DataFrame
        DataFrame containing mappings from duplicate parent labels to final parent labels.

    Returns
    -------
    child_mask_out : xarray.DataArray
        DataArray containing the new mask labels.

    parent_mask_out : xarray.DataArray
        DataArray containing the new mask labels.

    """

    child_name = child_mask.name
    parent_name = parent_mask.name

    child_darray = dask.array.from_array(child_mask.fillna(0))
    parent_darray = dask.array.from_array(parent_mask.fillna(0))

    def _map_child(vals):
        map_as_dict = child_tracks.set_index(child_name)["multivariate"].to_dict()
        map_as_array = np.full(child_tracks[child_name].max() + 1, 0, dtype=np.int64)
        for k, v in map_as_dict.items():
            map_as_array[k] = v
        return map_as_array[vals.astype(int)]

    child_mapped = xr.DataArray(
        child_darray.map_blocks(_map_child, dtype=np.int64),
        dims=child_mask.dims,
        coords=child_mask.coords,
    ).rename("multivariate")

    def _map_parent(vals):
        map_as_dict = parent_tracks.set_index(parent_name)["multivariate"].to_dict()
        map_as_array = np.full(parent_tracks[parent_name].max() + 1, 0, dtype=np.int64)
        for k, v in map_as_dict.items():
            map_as_array[k] = v
        return map_as_array[vals.astype(int)]

    parent_mapped = xr.DataArray(
        parent_darray.map_blocks(_map_parent, dtype=np.int64),
        dims=parent_mask.dims,
        coords=parent_mask.coords,
    ).rename("multivariate")

    return child_mapped, parent_mapped

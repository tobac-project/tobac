""" Provide tracking method 'eroded-contiguity'.

This tracking method was developed to handle cases less suited to tobacs centroid-based approach, which works well for mobile objects and less well for objects with complex morphologies, such as large, 3D anvil clouds. It is essentially a modified overlap test. 

It first erodes objects by a portion of their maximum size, then tests for contiguity between timesteps. This mitigates cases where the next centroid is not found by tobac linking, but can be sensitive to feature merging, depending on the choice of erosion amount.

Mathilde Ritman, 2026 (mathilde.ritman@phyiscs.ox.ac.uk)

References
----------
Ritman, M., Jones, W., and Stier, P.: Convective controls on anvil cloud evolution in the ICON km-scale global climate model, Atmos. Chem. Phys., 26, 7105-7126, https://doi.org/10.5194/acp-26-7105-2026, 2026.

"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage as ndi
import logging
import warnings
import joblib
import itertools


def track_using_contiguity(mask, table, PBC_flag=None, vdim=None, dims_to_skip=()):
    """Perform tracking by testing for contiguity between features.

    Parameters
    ----------
    mask : xarray.DataArray
        Input data array containing integer labels for features or cells to be tracked.

    table : pandas.DataFrame
        DataFrame containing corresponding detection or tracking information.

    PBC_flag : None or str
        As passed to tobac detection.

    vdim : int or str
        If PBC_flag is not None, vdim must be specified. This is the dimension number (starting from 0) or name of the vertical dimension.

    dims_to_skip : tuple of ints
        Dimension numbers to treat as independent when testing for contiguity, i.e., labels are not shared between these dimensions.

    Returns
    -------
    table : pandas.DataFrame
        DataFrame containing the mappings between the original labels and the tracked labels.

    tracked_mask : xarray.DataArray
        DataArray containing integer labels for each tracked feature.

    """

    boolean_array = mask > 0

    # check if the data are chunked
    if isinstance(boolean_array, xr.DataArray) and boolean_array.chunks is not None:
        warnings.warn(
            "the input data are chunked, these chunks are being reset to allow the method to work"
        )
        boolean_array = boolean_array.chunk({dim: -1 for dim in boolean_array.dims})

    # check non-horizontal dimensions are known when applying period boundary conditions
    if PBC_flag is not None:
        if "time" not in boolean_array.dims:
            raise ValueError("time dimension not found in data array dimensions")
        else:
            tdim = boolean_array.dims.index("time")
        if isinstance(vdim, str):
            if not vdim in boolean_array.dims:
                raise ValueError(f"vdim '{vdim}' not found in data array dimensions")
            vdim = boolean_array.dims.index(vdim)

    # calculate conncected components using scipy.ndi
    logging.info("Start tracking using contiguity")
    tracked_mask = np.zeros(boolean_array.shape, dtype=int)
    slices = [slice(None)] * boolean_array.ndim
    ranges = [range(boolean_array.shape[i]) for i in dims_to_skip]
    prev_max = 0
    for idx in itertools.product(*ranges):
        for i, dim in enumerate(dims_to_skip):
            slices[dim] = idx[i]
        sliced_arr = boolean_array[tuple(slices)]
        tracked_mask[tuple(slices)] = ndi.label(sliced_arr)[0] + (prev_max * sliced_arr)
        prev_max = np.max(tracked_mask[tuple(slices)])

    # calculate_periodic_boundary
    if tracked_mask.ndim == 4:
        slices = [slice(None)] * tracked_mask.ndim
        for t in range(tracked_mask.shape[tdim]):
            for v in range(tracked_mask.shape[vdim]):
                slices[tdim] = t
                slices[vdim] = v
                tracked_mask[tuple(slices)] = apply_periodic_boundary(
                    tracked_mask[tuple(slices)]
                )

    elif tracked_mask.ndim == 3:
        slices = [slice(None)] * tracked_mask.ndim
        for t in range(tracked_mask.shape[tdim]):
            slices[tdim] = t
            tracked_mask[tuple(slices)] = apply_periodic_boundary(
                tracked_mask[tuple(slices)]
            )

    else:
        tracked_mask = apply_periodic_boundary(tracked_mask)

    logging.info("Completed tracking using contiguity")

    tracked_mask = xr.DataArray(
        data=tracked_mask, dims=mask.dims, coords=mask.coords
    ).fillna(0)

    # record a table of the mappings between the original labels and the tracked labels, to be used for recording tracks in the output table
    pairs = set()
    for t in mask["time"].values:
        m = mask.sel({"time": t})
        r = tracked_mask.sel({"time": t})
        df_t = pd.DataFrame(
            {
                "mask": m.values.ravel(),
                "contiguous": r.values.ravel(),
            }
        ).dropna()
        pairs.update(map(tuple, df_t.to_numpy()))

    df = pd.DataFrame(sorted(pairs), columns=["mask", "contiguous"])

    # add to tracking table
    table = table.merge(
        df,
        on=mask.name,
        how="left",
    )

    return tracked_mask, table


def apply_periodic_boundary(x, PBC_flag):
    """Applies periodic boundary conditions to the output of the connected components method for 4D and 3D data (time, space).

    Parameters
    ----------
    x : numpy.ndarray
        3D or 4D array containing integer labels for connected components.
    PBC_flag : str
        As passed for tobac detection.

    Returns
    ----------
    x : numpy.ndarray
        The input array with periodic boundary conditions applied along the specified dimension.

    """

    def _apply(x, PBC_flag):
        dim = int(PBC_flag.split("_")[-1]) - 1
        dims = list(range(x.ndim))
        dims.remove(dim)
        altdim = dims[0]

        slices = [slice(None)] * x.ndim
        for i in range(x.shape[altdim]):
            slices[altdim] = i
            first = slices.copy()
            first[dim] = 0
            last = slices.copy()
            last[dim] = -1
            first, last = tuple(first), tuple(last)

            if x[first] > 0 and x[last] > 0:
                x[x == x[last]] = x[first]
        return x

    if PBC_flag == "both":
        x = _apply(x, "hdim_1")
        x = _apply(x, "hdim_2")
    
    else:
        x = _apply(x, PBC_flag)

    return x


def track_using_eroded_contiguity(
    mask,
    table,
    fraction,
    vdim=None,
    PBC_flag=None,
    use_parallel=True,
):
    """Perform tracking by eroding the input mask by the given fraction before testing for contiguity between timesteps.

    Parameters
    ----------
    mask : xarray.DataArray
        Input data array containing integer labels for features or cells to be tracked.

    table : pandas.DataFrame
        DataFrame containing corresponding detection or tracking information.

    fraction : float
        Fraction of the maximum distance from the edge to erode features by before testing for contiguity. Must be between 0 and 1.

    vdim : str
        Name of the vertical dimension in the input data array, if applicable.

    use_parallel : bool
        Whether to use parallel processing.

    Returns
    ----------
    table : pandas.DataFrame
        DataFrame containing the mappings between the original labels and the tracked labels.

    """

    eroded_mask = erode_mask(mask, fraction, vdim, use_parallel)
    _, tracking_table = track_using_contiguity(
        eroded_mask > 0, table, PBC_flag=PBC_flag, vdim=vdim
    )

    return tracking_table


def erode_mask(mask, fraction, vdim, use_parallel):
    """Perform erosion of the input features by the given fraction.

    Parameters
    ----------
    mask : xarray.DataArray
        Input data array containing integer labels for features or cells to be tracked.

    fraction : float
        Fraction of the maximum distance from the edge to erode features by before testing for contiguity. Must be between 0 and 1.

    vdim : str
        Name of the vertical dimension in the input data array, if applicable.

    use_parallel : bool
        Whether to use parallel processing.

    Returns
    ----------
    eroded_mask : xarray.DataArray
        DataArray containing eroded feature mask.

    """

    logging.info("Start erosion by fraction: %s" % fraction)
    topography = calculate_mask_topography(mask > 0, vdim, use_parallel)
    eroded_mask = mask.where(topography > fraction).fillna(0)

    logging.info("Completed erosion by fraction: %s" % fraction)

    return eroded_mask


def calculate_object_topography(label_value, arr, PBC_flag=None, max_object_length=1):
    """Computes the normalised distances between each pixel in the feature with label label_value and the edge of that feature.

    Parameters
    ----------
    label_value : int
        Integer label value of the object to calculate topography for.

    arr : numpy.ndarray
        Array containing integer labels for features or cells to be tracked.

    PBC_flag : None or str
        As passed to tobac detection.

    max_object_length : int
        Maximum expected length of the object in pixels, used for padding when applying periodic boundary conditions.

    Returns
    ----------
    output : numpy.ndarray
        Array containing the normalised distance from the edge of the object for each pixel in the input array.

    """    
    binary_mask = arr == label_value

    if PBC_flag is None:
        pad_axes = (0,1)
        periodic_axes = ()

    elif PBC_flag == "both":
        pad_axes = ()
        periodic_axes = (0,1)

    else:
        hdim = int(PBC_flag.split("_")[-1]) - 1
        pad_axes = list((0,1))
        pad_axes.remove(hdim)
        pad_axes = tuple(pad_axes)
        periodic_axes = (hdim,)

    # pad non-periodic edges with 0s
    pad_width = [(0,0)] * binary_mask.ndim
    for ax in pad_axes:
        pad_width[ax] = (1,1)
    padded = np.pad(binary_mask, pad_width)

    # wrap periodic edges by max_object_length pixels
    pad_width = [(0,0)] * binary_mask.ndim
    for ax in periodic_axes:
        pad_width[ax] = (max_object_length, max_object_length)
    padded = np.pad(padded, pad_width, mode="wrap")

    # calculate topography
    distance = ndi.distance_transform_edt(padded)

    # drop padding
    slices = []
    for ax in range(binary_mask.ndim):
        if ax in periodic_axes:
            slices.append(slice(max_object_length,-max_object_length))
        else:
            slices.append(slice(1,-1))

    distance = distance[tuple(slices)]    
    return distance / distance.max()


def calculate_mask_topography(mask, vdim, PBC_flag=None, max_object_length=1, use_parallel=True):
    """Computes the normalised distances between the pixels in each mask feature and the edge of that feature.

    Parameters
    ----------
    mask : xarray.DataArray
        Input data array containing integer labels for features or cells.

    vdim : str or None
        Name of the vertical dimension in the input data array, if applicable.

    use_parallel : bool
        Whether to use parallel processing.

    Returns
    ----------
    topography : numpy.ndarray
        Array containing the normalised distance from the edge of each feature for each pixel in the input array.

    """

    topography = np.zeros_like(mask, dtype=float)

    ntimes = len(mask.time) if "time" in mask.dims else 1
    slices = [slice(None)] * len(mask.dims)

    itr_times = range(ntimes)

    for tidx in itr_times:
        slices[0] = tidx if ntimes > 1 else slice(None)
        mask_t = mask.isel(time=tidx) if ntimes > 1 else mask

        for level in range(len(mask_t[vdim])):
            slices[-3] = level
            arr = mask_t.isel({vdim: level}).values.astype(np.int16)
            unique_labels = np.unique(arr)
            unique_labels = unique_labels[unique_labels != 0]

            if use_parallel:
                results = joblib.Parallel(n_jobs=-1)(
                    joblib.delayed(calculate_object_topography)(label, arr, PBC_flag, max_object_length)
                    for label in unique_labels
                )
            else:
                results = [
                    calculate_object_topography(label, arr, PBC_flag, max_object_length) for label in unique_labels
                ]

            for label_topography in results:
                topography[tuple(slices)] += label_topography

    return topography

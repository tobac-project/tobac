"""Function for converting feature labels back into an artificial interest field."""

import xarray as xr
import numpy as np
from typing import Literal, Optional, Tuple, Union


def features_to_interest_field(
    features,
    template: xr.DataArray,  # field for correct geometrie
    *,
    position_mode: Literal["hdim"] = "hdim",
    position_cols: Optional[Tuple[str, ...]] = None,
    time_key: Literal["frame", "time"] = "frame",
    blob: Literal["gaussian", "cone", "tophat"] = "gaussian",
    mode: Literal["max", "add"] = "max",
    amp_from: Union[str, float] = "max",
    thresh_from: Optional[str] = "threshold_value",
    size_from: Optional[str] = "num",
    amp_factor: float = 2.0,
    default_amp: float = 2.0,
    default_size: float = 5.0,
    default_thresh: float = 1.0,
):
    """
    Internal function to reconstruct an artificial interest field from feature
    positions.

    Parameters
    ----------
    features : pandas.DataFrame
        Table containing feature positions, times, and optional amplitude and
        size information.
    template : xarray.DataArray
        Reference array defining the output shape, dimensions, and coordinates.
    position_mode : {"hdim"}, optional
        Convention used to interpret spatial positions.
    position_cols : sequence of str, optional
        Column names containing the spatial coordinates of each feature.
    time_key : {"frame", "time"}, optional
        Column used to assign features to time steps.
    blob : {"gaussian", "cone", "tophat"}, optional
        Blob shape used to reconstruct each feature.
    mode : {"max", "add"}, optional
        Method used to combine overlapping blobs.
    amp_from : str or float, optional
        Column name or constant value used to define blob amplitude.
    thresh_from : str, optional
        Column name used to estimate the threshold for blob amplitude.
    size_from : str, optional
        Column name used to estimate blob size.
    default_amp : float, optional
        Factor applied to amplitudes derived from `amp_from`.
    default_thresh : float, optional
        Default threshold if `thresh_from` is not provided or invalid.
    default_size : float, optional
        Default blob size if `size_from` is not provided or invalid.

    Returns
    -------
    xarray.DataArray
        Output interest field with the same structure as `template`.
    """

    if not isinstance(template, xr.DataArray):
        raise TypeError("template must be an xarray.DataArray")

    if template.ndim < 3:
        raise ValueError("template must be at least 3D (time + spatial dims)")

    tdim = template.dims[0]
    spatial_dims = template.dims[1:]
    n_spatial = len(spatial_dims)

    out = xr.zeros_like(template, dtype=float)

    sizes = [template.sizes[d] for d in spatial_dims]

    # default position columns
    if position_cols is None:
        if position_mode == "hdim":
            if n_spatial == 2:
                position_cols = ("hdim_1", "hdim_2")
            elif n_spatial == 3:
                position_cols = ("vdim", "hdim_1", "hdim_2")
        else:
            raise ValueError("position_mode must be 'hdim'")
    # coordinate grids
    if position_mode == "hdim":
        coords = [np.arange(n) for n in sizes]
    else:
        raise ValueError("position_mode must be 'hdim'")

    grids = np.meshgrid(*coords, indexing="ij")

    for _, row in features.iterrows():

        if time_key == "frame":
            tidx = int(row["frame"])
            selector = {tdim: out[tdim].values[tidx]}
        elif time_key == "time":
            selector = {tdim: np.datetime64(row["time"])}
        else:
            raise ValueError("time_key must be 'frame' or 'time'")

        pos = [float(row[c]) for c in position_cols]

        if thresh_from is not None:
            if (
                isinstance(thresh_from, str)
                and thresh_from in row.index
                and np.isfinite(row[thresh_from])
            ):
                fthresh = float(row[thresh_from])
            else:
                fthresh = default_thresh

        if amp_from is not None:
            if (
                isinstance(amp_from, str)
                and amp_from in row.index
                and np.isfinite(row[amp_from])
            ):
                fmax = float(row[amp_from])
            else:
                fmax = amp_factor * fthresh
        else:
            fmax = amp_factor * fthresh

        if (
            size_from is not None
            and size_from in row.index
            and np.isfinite(row[size_from])
        ):
            area_or_volume = float(row[size_from])
            if area_or_volume == 0:
                area_or_volume = default_size
        else:
            area_or_volume = default_size

        r2 = 0
        for g, p in zip(grids, pos):
            r2 += (g - p) ** 2

        if n_spatial == 2:
            area = area_or_volume
            r_scale = np.sqrt(area / np.pi)
        elif n_spatial == 3:
            volume = area_or_volume
            r_scale = np.cbrt(3 * volume / (4 * np.pi))

        # scale r-squared by the size of the blob
        r2 /= r_scale**2

        if blob == "gaussian":

            # set gauss parameters
            A = fmax
            B = np.log(fmax / fthresh)

            maxBr2 = np.log(1e10)  # avoid numerical issues with very small values
            Br2 = B * r2
            Br2 = np.minimum(Br2, maxBr2)
            blob_nd = A * np.exp(-Br2)

        elif blob == "cone":
            A = fthresh - fmax
            B = fmax
            blob_nd = A * np.sqrt(r2) + B
            blob_nd = np.maximum(blob_nd, 0)
        elif blob == "tophat":
            blob_nd = fmax * (r2 <= 1)
        else:
            raise ValueError("blob must be 'gaussian', 'cone', or 'tophat'")
        current = out.sel(selector).values

        if mode == "add":
            out.loc[selector] = current + blob_nd
        elif mode == "max":
            out.loc[selector] = np.maximum(current, blob_nd)
        else:
            raise ValueError("mode must be 'add' or 'max'")

    return out

"""
Support functions to compute bulk statistics of features, either as a postprocessing step
or within feature detection or segmentation.

"""

from __future__ import annotations
from datetime import timedelta
import logging
import warnings
from functools import partial
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from skimage.measure import regionprops_table

from tobac.utils import decorators
from tobac.utils.generators import field_and_features_over_time


def get_collapsed_coords(coords, uncollapsed_axes, shape):
    return np.stack(
        np.unravel_index(
            np.unique(
                np.ravel_multi_index(coords.T[uncollapsed_axes], shape),
            ),
            shape,
        ),
        -1,
    )


def get_statistics(
    features: pd.DataFrame,
    labels: xr.DataArray,
    *fields: tuple[xr.DataArray],
    statistic: dict[str, Union[Callable, tuple[Callable, dict]]] = {
        "ncells": np.count_nonzero
    },
    index: Optional[list[int]] = None,
    default: Optional[float] = None,
    id_column: str = "feature",
    collapse_dim: Optional[Union[str, list[str]]] = None,
) -> pd.DataFrame:
    """Get bulk statistics for objects (e.g. features or segmented features)
    given a labelled mask of the objects and any input field with the same
    dimensions or that can be broadcast with labels according to numpy-like
    broadcasting rules.

    The statistics are added as a new column to the existing feature dataframe.
    Users can specify which statistics are computed by providing a dictionary
    with the column name of the metric and the respective function.

    Parameters
    ----------
    features: pd.DataFrame
        Dataframe with features or segmented features (output from feature
        detection or segmentation), which can be for the specific timestep or
        for the whole dataset

    labels : xr.DataArray
        Mask with labels of each regions to apply function to (e.g. output of
        segmentation for a specific timestep)

    *fields : tuple[xr.DataArray]
        Fields to give as arguments to each function call. If the shape does not
        match that of labels, numpy-style broadcasting will be applied.

    statistic: dict[str, Callable], optional (default: {'ncells':np.count_nonzero})
        Dictionary with function(s) to apply over each region as values and the
        name of the respective statistics as keys. Default is to just count the
        number of cells associated with each feature and write it to the feature
        dataframe.

    index: None | list[int], optional (default: None)
        list of indices of regions in labels to apply function to. If None, will
        default to all integer feature labels in labels.

    default: None | float, optional (default: None)
        default value to return in a region that has no values.

    id_column: str, optional (default: "feature")
        Name of the column in feature dataframe that contains IDs that match with
        the labels in mask. The default is the column "feature".

    collapse_axis: None | int | list[int], optional (default: None):
        Index or indices of axes of labels to collapse. This will reduce the dimensionality of labels
        while allowing labelled features to overlap. This can be used, for example, to calculate the
        footprint area (2D) of 3D labels


    Returns
    -------
    features: pd.DataFrame
        Updated feature dataframe with bulk statistics for each feature saved
        in a new column.
    """

    if len(features):
        # Drop existing stats columns
        stats_columns = list(statistic.keys())
        features = features.drop(columns=stats_columns, errors="ignore")

        if index is None:
            index = features[id_column]

        coords = (
            pd.DataFrame(
                regionprops_table(
                    np.where(np.isin(labels.values, index), labels.values, 0),
                    properties=("label", "coords"),
                )
            )
            .set_index("label")
            .coords
        )

        if len(coords):

            dims = list(labels.dims)
            if collapse_dim is not None:
                if isinstance(collapse_dim, str):
                    if any((collapse_dim in f.dims for f in fields)):
                        warnings.warn(
                            f"collapse_dim {collapse_dim} found in one or more fields, this selection could cause unexpected results"
                        )
                    dims.remove(collapse_dim)
                else:
                    if any((c in f.dims for f in fields for c in collapse_dim)):
                        warnings.warn(
                            f"collapse_dim {collapse_dim} found in one or more fields, this selection could cause unexpected results"
                        )

                    dims = [d for d in dims if d not in collapse_dim]

                if not len(dims):
                    ValueError("Cannot collapse all axes of labels")

                uncollapsed_axes = [
                    i for i, dim in enumerate(labels.dims) if dim in dims
                ]
                shape = np.array([labels.sizes[d] for d in dims])
                coords = coords.apply(
                    partial(
                        get_collapsed_coords,
                        uncollapsed_axes=uncollapsed_axes,
                        shape=shape,
                    )
                )

            fields = xr.broadcast(*fields)

            data = coords.apply(
                lambda r: [
                    f.isel(
                        dict(zip(dims, xr.DataArray(r.T, dims=("dim_", "points")))),
                        missing_dims="ignore",
                    ).values
                    for f in fields
                ]
            )  # Get data corresponding to each feature

            features = features.reset_index().set_index(features[id_column])

            for stats_name in statistic:
                # if function is given as a tuple, take the input parameters provided
                if type(statistic[stats_name]) is tuple:
                    # assure that key word arguments are provided as dictionary
                    if not type(statistic[stats_name][1]) is dict:
                        raise TypeError(
                            "Tuple must contain dictionary with key word arguments for function."
                        )

                    statistic[stats_name] = partial(
                        statistic[stats_name][0], **statistic[stats_name][1]
                    )

                features.loc[data.index, stats_name] = data.apply(
                    lambda r: statistic[stats_name](*r)
                )

            if default is not None:
                features[stats_columns] = features[stats_columns].fillna(value=default)

            return features.set_index("index")

        else:
            features = features.copy()
            features[stats_columns] = default
            return features
    return features


@decorators.iris_to_xarray()
def get_statistics_from_mask(
    features: pd.DataFrame,
    segmentation_mask: xr.DataArray,
    *fields: tuple[xr.DataArray],
    statistic: dict[str, tuple[Callable]] = {"Mean": np.mean},
    index: Union[None, list[int]] = None,
    default: Union[None, float] = None,
    id_column: str = "feature",
    collapse_dim: Union[None, str, list[str]] = None,
    time_var_name: str = "time",
    time_padding: Optional[timedelta] = None,
) -> pd.DataFrame:
    """Derives bulk statistics for each object in the segmentation mask, and
    returns a features Dataframe with these properties for each feature.

    Parameters
    ----------
    features: pd.DataFrame
        Dataframe with segmented features (output from feature detection or
        segmentation). Timesteps must not be exactly the same as in segmentation
        mask but all labels in the mask need to be present in the feature
        dataframe.

    segmentation_mask : xr.DataArray
        Segmentation mask output

    *fields : tuple[xr.DataArray]
        Field(s) with input data. If field does not have a time dimension it
        will be considered time invariant, and the entire field will be passed
        for each time step in segmentation_mask. If the shape does not match
        that of labels, numpy-style broadcasting will be applied.

    statistic: dict[str, Callable], optional (default: {'ncells':np.count_nonzero})
        Dictionary with function(s) to apply over each region as values and the
        name of the respective statistics as keys. Default is to calculate the
        mean value of the field over each feature.

    index: None | list[int], optional (default: None)
        list of indexes of regions in labels to apply function to. If None, will
        default to all integers between 1 and the maximum value in labels

    default: None | float, optional (default: None)
        default value to return in a region that has no values

    id_column: str, optional (default: "feature")
        Name of the column in feature dataframe that contains IDs that match with the labels in mask. The default is the column "feature".

    collapse_dim: None | str | list[str], optional (default: None)
        Dimension names of labels to collapse, allowing, e.g. calulcation of statistics on 2D
        fields for the footprint of 3D objects

    time_var_name : str, optional (default: "time")
        The name of the time dimension in the input fields and the time column
        in features, by default "time"

    time_padding: timedelta, optional (default: None)
        If set, allows for statistics to be associated with a feature input
        timestep that is within time_padding off of the feature. Extremely useful when
        converting between micro- and nanoseconds, as is common when using Pandas
        dataframes.

     Returns
     -------
     features: pd.DataFrame
         Updated feature dataframe with bulk statistics for each feature saved in a new column
    """
    # warning when feature labels are not unique in dataframe
    if not features[id_column].is_unique:
        logging.warning(
            "Feature labels are not unique which may cause unexpected results for the computation of bulk statistics."
        )
    # extra warning when feature labels are not unique in timestep
    uniques = features.groupby("time")[id_column].value_counts().values
    if not uniques[uniques > 1].size == 0:
        logging.warning(
            "Note that non-unique feature labels occur also in the same timestep. This likely causes unexpected results for the computation of bulk statistics."
        )

    # check if any of the feature dataframe input values match with segmentation mask IDs
    if not np.any(np.isin(features[id_column], np.unique(segmentation_mask))):
        raise ValueError(
            "The labels of the segmentation mask and the feature dataframe do not seem to match. Please make sure you provide the correct input feature dataframe  to calculate the bulk statistics."
        )

    # get bulk statistics for each timestep
    step_statistics = []

    for _, tt, segmentation_mask_t, features_t in field_and_features_over_time(
        segmentation_mask,
        features,
        time_var_name=time_var_name,
        time_padding=time_padding,
    ):
        # select specific timestep
        fields_t = (
            (
                field.sel({time_var_name: tt}, method="nearest", tolerance=time_padding)
                if time_var_name in field.coords
                else field
            )
            for field in fields
        )

        # make sure that the labels in the segmentation mask exist in feature dataframe
        # continue loop because not all timesteps might have matching IDs
        if not np.any(np.isin(features_t[id_column], np.unique(segmentation_mask_t))):
            warnings.warn("Not all timesteps have matching features", UserWarning)
            step_statistics.append(features_t.copy())
            continue
        else:
            # make sure that features are not double-defined
            step_statistics.append(
                get_statistics(
                    features_t,
                    segmentation_mask_t,
                    *fields_t,
                    statistic=statistic,
                    default=default,
                    index=index,
                    id_column=id_column,
                    collapse_dim=collapse_dim,
                )
            )

    features = pd.concat(step_statistics)

    return features

"""
Support functions to compute bulk statistics of features, either as a postprocessing step
or within feature detection or segmentation. 

"""

import logging
import warnings
from functools import partial
from typing import Callable, Union

import numpy as np

# numpy renamed core to _core recently
try:
    from numpy._core import multiarray as mu
except ModuleNotFoundError:
    from numpy.core import multiarray as mu
import pandas as pd
import xarray as xr

from tobac.utils import decorators


def get_statistics(
    features: pd.DataFrame,
    labels: np.ndarray[int],
    *fields: tuple[np.ndarray],
    statistic: dict[str, Union[Callable, tuple[Callable, dict]]] = {
        "ncells": np.count_nonzero
    },
    index: Union[None, list[int]] = None,
    default: Union[None, float] = None,
    id_column: str = "feature",
    collapse_axis: Union[None, int, list[int]] = None,
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

    labels : np.ndarray[int]
        Mask with labels of each regions to apply function to (e.g. output of
        segmentation for a specific timestep)

    *fields : tuple[np.ndarray]
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

    # if mask and input data dimensions do not match we can broadcast using numpy broadcasting rules
    if collapse_axis is not None:
        # Test if iterable and if not make a list
        try:
            collapse_axis = list(iter(collapse_axis))
        except TypeError:
            collapse_axis = [collapse_axis]

        # Normalise axes to handle negative axis number conventions
        ndim = len(labels.shape)
        collapse_axis = [mu.normalize_axis_index(axis, ndim) for axis in collapse_axis]
        uncollapsed_axes = [
            i for i, _ in enumerate(labels.shape) if i not in collapse_axis
        ]
        if not len(uncollapsed_axes):
            raise ValueError("Cannot collapse all axes of labels")
        collapsed_shape = tuple(
            [s for i, s in enumerate(labels.shape) if i not in collapse_axis]
        )
        broadcast_flag = any([collapsed_shape != field.shape for field in fields])
        if broadcast_flag:
            raise ValueError("Broadcasting not supported with collapse_axis")

    else:
        broadcast_flag = any([labels.shape != field.shape for field in fields])
        if broadcast_flag:
            # Broadcast input labels and fields to ensure they work according to numpy broadcasting rules
            broadcast_fields = np.broadcast_arrays(labels, *fields)
            labels = broadcast_fields[0]
            fields = broadcast_fields[1:]

    # mask must contain positive values to calculate statistics
    if np.any(labels > 0):
        if index is None:
            index = features[id_column].to_numpy().astype(int)
        else:
            # get the statistics only for specified feature objects
            if np.max(index) > np.max(labels):
                raise ValueError("Index contains values that are not in labels!")

        # Find which labels exist in features for output:
        index_in_features = np.isin(index, features[id_column])

        # set negative markers to 0 as they are unsegmented
        bins = np.cumsum(np.bincount(np.maximum(labels.ravel(), 0)))
        argsorted = np.argsort(labels.ravel())

        # Create lambdas to get (ravelled) label locations using argsorted and bins
        if collapse_axis is None:
            label_locs = lambda i: argsorted[bins[i - 1] : bins[i]]
        else:
            # Collapse ravelled locations to the remaining axes
            label_locs = lambda i: np.unique(
                np.ravel_multi_index(
                    np.array(
                        np.unravel_index(argsorted[bins[i - 1] : bins[i]], labels.shape)
                    )[uncollapsed_axes],
                    collapsed_shape,
                )
            )

        # apply each function given per statistic parameter for the labeled regions sorted in ascending order
        for stats_name in statistic.keys():
            # if function is given as a tuple, take the input parameters provided
            if type(statistic[stats_name]) is tuple:
                # assure that key word arguments are provided as dictionary
                if not type(statistic[stats_name][1]) is dict:
                    raise TypeError(
                        "Tuple must contain dictionary with key word arguments for function."
                    )

                func = partial(statistic[stats_name][0], **statistic[stats_name][1])
            else:
                func = statistic[stats_name]

            # default needs to be sequence when function output is array-like
            output = func(*([np.random.rand(10)] * len(fields)))
            if hasattr(output, "__len__"):
                default = np.full(output.shape, default)

            stats = np.array(
                [
                    (
                        func(*(field.ravel()[label_locs(i)] for field in fields))
                        if i < bins.size and bins[i] > bins[i - 1]
                        else default
                    )
                    for i in index
                ]
            )

            # add results of computed statistics to feature dataframe with column name given per statistic
            # initiate new column in feature dataframe if it does not already exist
            if stats_name not in features.columns:
                if default is not None and not hasattr(default, "__len__"):
                    # If result is a scalar value we can create an empty column with the correct dtype
                    features[stats_name] = np.full(
                        [len(features)], default, type(default)
                    )
                else:
                    features[stats_name] = np.full([len(features)], None, object)

            for idx, label in enumerate(index):
                if index_in_features[idx]:
                    # test if values are scalars
                    if not hasattr(stats[idx], "__len__"):
                        # if yes, we can just assign the value to the new column and row of the respective feature
                        features.loc[ features[id_column] == label, stats_name] = stats[
                            idx
                        ]
                        # if stats output is array-like it has to be added in a different way
                    else:
                        df = pd.DataFrame({stats_name: [stats[idx]]})
                        # get row index rather than pd.Dataframe index value since we need to use .iloc indexing
                        row_idx = np.where(features[id_column] == label)[0]
                        features.iloc[
                            row_idx,
                            features.columns.get_loc(stats_name),
                        ] = df.apply(lambda r: tuple(r), axis=1)

    return features


@decorators.iris_to_xarray()
def get_statistics_from_mask(
    features: pd.DataFrame,
    segmentation_mask: xr.DataArray,
    *fields: xr.DataArray,
    statistic: dict[str, tuple[Callable]] = {"Mean": np.mean},
    index: Union[None, list[int]] = None,
    default: Union[None, float] = None,
    id_column: str = "feature",
    collapse_dim: Union[None, str, list[str]] = None,
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

    *fields : xr.DataArray[np.ndarray]
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
    collapse_dim: None | str | list[str], optional (defailt: None)
        Dimension names of labels to collapse, allowing, e.g. calulcation of statistics on 2D
        fields for the footprint of 3D objects

     Returns:
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
    if not np.unique([unique_features.size for unique_features in features.groupby('time').id.unique().values]).size == 1: 
        logging.warning('Note that non-unique feature labels occur also in the same timestep.')

    if collapse_dim is not None:
        if isinstance(collapse_dim, str):
            collapse_dim = [collapse_dim]
        non_time_dims = [dim for dim in segmentation_mask.dims if dim != "time"]
        collapse_axis = [
            i for i, dim in enumerate(non_time_dims) if dim in collapse_dim
        ]
        if len(collapse_dim) != len(collapse_axis):
            raise ValueError(
                "One or more of collapse_dim not found in dimensions of segmentation_mask"
            )
    else:
        collapse_axis = None

    # get bulk statistics for each timestep
    step_statistics = []

    for tt in pd.to_datetime(segmentation_mask.time):
        # select specific timestep
        segmentation_mask_t = segmentation_mask.sel(time=tt).data
        fields_t = (
            field.sel(time=tt).values if "time" in field.coords else field.values
            for field in fields
        )
        features_t = features.loc[features.time == tt].copy()

        # make sure that the labels in the segmentation mask exist in feature dataframe
        if (
            np.intersect1d(np.unique(segmentation_mask_t), features_t[id_column]).size
            > np.unique(segmentation_mask_t).size
        ):
            raise ValueError(
                "The labels of the segmentation mask and the feature dataframe do not seem to match. Please make sure you provide the correct input feature dataframe to calculate the bulk statistics. "
            )
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
                    collapse_axis=collapse_axis,
                )
            )

    features = pd.concat(step_statistics)

    return features

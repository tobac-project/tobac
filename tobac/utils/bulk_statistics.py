"""
Support functions to compute bulk statistics of features, either as a postprocessing step
or within feature detection or segmentation. 

"""

import logging
from . import internal as internal_utils
from typing import Callable, Union
from functools import partial
import numpy as np
import pandas as pd
import xarray as xr


def get_statistics(
    labels: np.ndarray[int],
    *fields: tuple[np.ndarray],
    features: pd.DataFrame,
    statistic: dict[str, Union[Callable, tuple[Callable, dict]]] = {
        "ncells": np.count_nonzero
    },
    index: Union[None, list[int]] = None,
    default: Union[None, float] = None,
    id_column: str = "feature",
) -> pd.DataFrame:
    """
    Get bulk statistics for objects (e.g. features or segmented features) given a labelled mask of the objects
    and any input field with the same dimensions.

    The statistics are added as a new column to the existing feature dataframe. Users can specify which statistics are computed by
    providing a dictionary with the column name of the metric and the respective function.

    Parameters
    ----------
    labels : np.ndarray[int]
        Mask with labels of each regions to apply function to (e.g. output of segmentation for a specific timestep)
    *fields : tuple[np.ndarray]
        Fields to give as arguments to each function call. Must have the same shape as labels.
    features: pd.DataFrame
        Dataframe with features or segmented features (output from feature detection or segmentation)
        can be for the specific timestep or for the whole dataset
    statistic: dict[str, Callable], optional (default: {'ncells':np.count_nonzero})
        Dictionary with function(s) to apply over each region as values and the name of the respective statistics as keys
        default is to just count the number of cells associated with each feature and write it to the feature dataframe
    index: None | list[int], optional (default: None)
        list of indexes of regions in labels to apply function to. If None, will
            default to all integers between the minimum and the maximum value in labels
    default: None | float, optional (default: None)
        default value to return in a region that has no values
    id_column: str, optional (default: "feature")
       Name of the column in feature dataframe that contains IDs that match with the labels in mask. The default is the column "feature".

     Returns:
     -------
     features: pd.DataFrame
         Updated feature dataframe with bulk statistics for each feature saved in a new column
    """
    # raise error if mask and input data dimensions do not match
    for field in fields:
        if labels.shape != field.shape:
            raise ValueError("Input labels and field do not have the same shape")

    # mask must contain positive values to calculate statistics
    if labels[labels > 0].size > 0:
        if index is None:
            index = features.feature.to_numpy()
        else:
            # get the statistics only for specified feature objects
            if np.max(index) > np.max(labels):
                raise ValueError("Index contains values that are not in labels!")

        # Find which labels exist in features for output:
        index_in_features = np.isin(index, features[id_column])

        # set negative markers to 0 as they are unsegmented
        bins = np.cumsum(np.bincount(np.maximum(labels.ravel(), 0)))
        argsorted = np.argsort(labels.ravel())

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
            output = func(np.random.rand(10))
            if hasattr(output, "__len__"):
                default = np.full(output.shape, default)

            stats = np.array(
                [
                    func(
                        *(
                            field.ravel()[argsorted[bins[i - 1] : bins[i]]]
                            for field in fields
                        )
                    )
                    if i < bins.size and bins[i] > bins[i - 1]
                    else default
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
                        features.loc[features[id_column] == label, stats_name] = stats[
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


@internal_utils.iris_to_xarray
def get_statistics_from_mask(
    segmentation_mask: xr.DataArray,
    *fields: xr.DataArray,
    features: pd.DataFrame,
    statistic: dict[str, tuple[Callable]] = {"Mean": np.mean},
    index: Union[None, list[int]] = None,
    default: Union[None, float] = None,
    id_column: str = "feature",
) -> pd.DataFrame:
    """
    Derives bulk statistics for each object in the segmentation mask.


    Parameters:
    -----------
    segmentation_mask : xr.DataArray
        Segmentation mask output
    *fields : xr.DataArray[np.ndarray]
        Field(s) with input data. Needs to have the same dimensions as the segmentation mask.
    features: pd.DataFrame
        Dataframe with segmented features (output from feature detection or segmentation).
        Timesteps must not be exactly the same as in segmentation mask but all labels in the mask need to be present in the feature dataframe.
    statistic: dict[str, Callable], optional (default: {'ncells':np.count_nonzero})
        Dictionary with function(s) to apply over each region as values and the name of the respective statistics as keys
        default is to just count the number of cells associated with each feature and write it to the feature dataframe
    index: None | list[int], optional (default: None)
        list of indexes of regions in labels to apply function to. If None, will
            default to all integers between 1 and the maximum value in labels
    default: None | float, optional (default: None)
        default value to return in a region that has no values
    id_column: str, optional (default: "feature")
       Name of the column in feature dataframe that contains IDs that match with the labels in mask. The default is the column "feature".


     Returns:
     -------
     features: pd.DataFrame
         Updated feature dataframe with bulk statistics for each feature saved in a new column
    """
    # check that mask and input data have the same dimensions
    for field in fields:
        if segmentation_mask.shape != field.shape:
            raise ValueError("Input labels and field do not have the same shape")

    # warning when feature labels are not unique in dataframe
    if not features.feature.is_unique:
        raise logging.warning(
            "Feature labels are not unique which may cause unexpected results for the computation of bulk statistics."
        )

    # get bulk statistics for each timestep
    features_t_list = []

    for tt in pd.to_datetime(segmentation_mask.time):
        # select specific timestep
        segmentation_mask_t = segmentation_mask.sel(time=tt).data
        field_t = field.sel(time=tt).data
        features_t = features.loc[features.time == tt]

        # make sure that the labels in the segmentation mask exist in feature dataframe
        if (
            np.intersect1d(np.unique(segmentation_mask_t), features.feature).size
            > np.unique(segmentation_mask_t).size
        ):
            raise ValueError(
                "The labels of the segmentation mask and the feature dataframe do not seem to match. Please make sure you provide the correct input feature dataframe to calculate the bulk statistics. "
            )
        else:
            # make sure that features are not double-defined
            features_t_list.append(
                get_statistics(
                    segmentation_mask_t,
                    field_t,
                    features=features_t,
                    statistic=statistic,
                    default=default,
                    index=index,
                    id_column=id_column,
                )
            )

    features = pd.concat(features_t_list)

    return features

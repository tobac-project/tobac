"""Custom generators used for iterators required by tobac"""

import datetime
from typing import Generator, Optional, Tuple, Union

import cftime
import numpy as np
import pandas as pd
import xarray as xr

import tobac.utils.datetime as datetime_utils


def field_and_features_over_time(
    field: xr.DataArray,
    features: pd.DataFrame,
    time_var_name: str = "time",
    time_padding: Optional[datetime.timedelta] = None,
) -> Generator[
    Tuple[
        int,
        Union[datetime.datetime, np.datetime64, cftime.datetime],
        xr.DataArray,
        pd.DataFrame,
    ],
    None,
    None,
]:
    """Generator that iterates over time through a paired field dataarray and a
    features dataframe. time_padding parameter allows a tolerance to be set for
    matching time stamps in the datarray and dataframe

    Parameters
    ----------
    field : xr.DataArray
        The field to iterate over
    features : pd.DataFrame
        The features dataframe to iterate through
    time_var_name : str, optional (default: "time")
        The name of the time dimension in field and the time column in features,
        by default "time"
    time_padding : datetime.timedelta, optional (default: None)
        The tolerance for matching features at the same time as each time step
        in the field dataframe, by default None

    Yields
    ------
    Generator[tuple[int, Union[datetime.datetime, np.datetime64, cftime.datetime], xr.DataArray, pd.DataFrame], None, None]
        A generator that returns the iteration index, the time, the slice of
        field at that time the slice of features with times within the time
        padding tolerance of the time step
    """
    if time_var_name not in field.coords:
        raise ValueError(f"{time_var_name} not present in input field coordinates")

    if time_var_name not in features.columns:
        raise ValueError(f"{time_var_name} not present in input feature columns")

    all_times = pd.Series(
        datetime_utils.match_datetime_format(
            features[time_var_name], field.coords[time_var_name]
        ),
        index=features.index,
    )
    for time_iteration_number, time_iteration_value in enumerate(
        field.coords[time_var_name]
    ):
        field_at_time = field.isel({time_var_name: time_iteration_number})
        if time_padding is not None:
            # padded_conv = pd.Timedelta(time_padding).to_timedelta64()
            if isinstance(time_iteration_value.values.item(), int):
                min_time = (
                    time_iteration_value.values
                    - pd.Timedelta(time_padding).to_timedelta64()
                )
                max_time = (
                    time_iteration_value.values
                    + pd.Timedelta(time_padding).to_timedelta64()
                )
            else:
                min_time = time_iteration_value.values - time_padding
                max_time = time_iteration_value.values + time_padding
            features_i = features.loc[all_times.between(min_time, max_time)]
        else:
            features_i = features.loc[all_times == time_iteration_value.values]

        yield time_iteration_number, time_iteration_value, field_at_time, features_i

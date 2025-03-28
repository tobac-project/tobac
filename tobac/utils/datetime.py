"""Functions for converting between and working with different datetime formats"""

from typing import Union
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import cftime


def to_cftime(
    dates: Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime],
    calendar: str,
    align_on: str = "date",
) -> cftime.datetime:
    dates_arr = np.atleast_1d(dates)
    if isinstance(dates_arr[0], cftime.datetime):
        cftime_dates = (
            xr.DataArray(dates_arr, {"time": dates_arr})
            .convert_calendar(calendar, use_cftime=True, align_on=align_on)
            .time.values
        )
    else:
        cftime_dates = (
            xr.DataArray(dates_arr, {"time": pd.to_datetime(dates_arr)})
            .convert_calendar(calendar, use_cftime=True, align_on=align_on)
            .time.values
        )
    if not hasattr(dates, "__iter__") or isinstance(dates, str) and len(cftime_dates):
        return cftime_dates[0]
    return cftime_dates


def to_timestamp(
    dates: Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime],
) -> pd.Timestamp:
    squeeze_output = False
    if not hasattr(dates, "__iter__") or isinstance(dates, str):
        dates = np.atleast_1d(dates)
        squeeze_output = True

    if isinstance(dates[0], cftime.datetime):
        pd_dates = xr.CFTimeIndex(dates).to_datetimeindex()
    else:
        pd_dates = pd.to_datetime(dates)

    if squeeze_output:
        return pd_dates[0]
    return pd_dates


def to_datetime(
    dates: Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime],
) -> datetime.datetime:
    return to_timestamp(dates).to_pydatetime()


def to_datetime64(
    dates: Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime],
) -> np.datetime64:
    return to_timestamp(dates).to_numpy()


def to_datestr(
    dates: Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime],
) -> str:
    dates = to_datetime64(dates)
    if hasattr(dates, "__iter__"):
        return dates.astype(str)
    return str(dates)


def match_datetime_format(
    dates: Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime],
    target: Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime],
):
    if isinstance(target, str):
        return to_datestr(dates)
    if isinstance(target, xr.DataArray):
        target = target.values
    if isinstance(target, pd.Series):
        target = target.to_numpy()
    if hasattr(target, "__iter__"):
        target = target[0]
    if isinstance(target, str):
        return to_datestr(dates)
    if isinstance(target, cftime.datetime):
        return to_cftime(dates, target.calendar)
    if isinstance(target, pd.Timestamp):
        return to_timestamp(dates)
    if isinstance(target, np.datetime64):
        return to_datetime64(dates)
    if isinstance(target, datetime.datetime):
        return to_datetime(dates)
    raise ValueError("Target is not a valid datetime format")

"""Functions for converting between and working with different datetime formats"""

from typing import Union, Optional, Literal
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import cftime
import re


def to_cftime(
    dates: Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime],
    calendar: str,
    align_on: str = "date",
) -> cftime.datetime:
    """Converts a provided datetime-like object to a cftime datetime with the
    given calendar

    Parameters
    ----------
    dates : Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime]
        A datetime-like object or array of datetime-like objects to be converted
    calendar : str
        The requested cftime calender
    align_on : str, optional
        The 'align-on' parameter required for 360-day, 365-day and 366-day
        cftime dates, by default "date"

    Returns
    -------
    cftime.datetime
        A cftime object or array of cftime objects in the requested calendar
    """
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
    precision: Optional[Literal["ns", "us", "s", "ms"]] = None,
) -> pd.Timestamp:
    """Converts a provided datetime-like object to a pandas timestamp

    Parameters
    ----------
    dates : Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime]
        A datetime-like object or array of datetime-like objects to be converted
    precision : Optional[Literal["ns", "us", "s", "ms"]]
        The precision of the timestamp. If None, the default precision is used.
        The default precision is ns for Pandas 2 and before; us for Pandas 3
        - "ns": nanoseconds
        - "us": microseconds
        - "ms": milliseconds
        - "s": seconds

    Returns
    -------
    pd.Timestamp
        A pandas timestamp or array of pandas timestamps
    """
    squeeze_output = False
    if not hasattr(dates, "__iter__") or isinstance(dates, str):
        dates = np.atleast_1d(dates)
        squeeze_output = True

    if isinstance(next(iter(dates)), cftime.datetime):
        pd_dates = xr.CFTimeIndex(dates).to_datetimeindex()
    else:
        pd_dates = pd.to_datetime(dates)

    if precision is not None:
        pd_dates = pd_dates.astype(f"datetime64[{precision}]")

    if squeeze_output:
        return next(iter(pd_dates))
    return pd_dates


def to_datetime(
    dates: Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime],
) -> datetime.datetime:
    """Converts a provided datetime-like object to python datetime objects

    Parameters
    ----------
    dates : Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime]
        A datetime-like object or array of datetime-like objects to be converted

    Returns
    -------
    datetime.datetime
        A python datetime or array of python datetimes
    """
    return to_timestamp(dates).to_pydatetime()


def to_datetime64(
    dates: Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime],
    precision: Optional[Literal["ns", "us", "s", "ms"]] = None,
) -> np.datetime64:
    """Converts a provided datetime-like object to numpy datetime64 objects

    Parameters
    ----------
    dates : Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime]
        A datetime-like object or array of datetime-like objects to be converted
    precision : Optional[Literal["ns", "us", "s", "ms"]]
        The precision of the timestamp. If None, the default precision is used.
        The default precision is ns for Pandas 2 and before; us for Pandas 3
        - "ns": nanoseconds
        - "us": microseconds
        - "ms": milliseconds
        - "s": seconds

    Returns
    -------
    np.datetime64
        A numpy datetime64 or array of numpy datetime64s
    """
    return to_timestamp(dates, precision).to_numpy()


def to_datestr(
    dates: Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime],
    precision: Optional[Literal["ns", "us", "s", "ms"]] = None,
) -> str:
    """Converts a provided datetime-like object to ISO format date strings

    Parameters
    ----------
    dates : Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime]
        A datetime-like object or array of datetime-like objects to be converted
    precision : Optional[Literal["ns", "us", "s", "ms"]]
        The precision of the timestamp. If None, the default precision is used.
        The default precision is ns for Pandas 2 and before; us for Pandas 3
        - "ns": nanoseconds
        - "us": microseconds
        - "ms": milliseconds
        - "s": seconds

    Returns
    -------
    str
        A string or array of strings in ISO date format
    """
    dates = to_datetime64(dates, precision)
    if hasattr(dates, "__iter__"):
        return dates.astype(str)
    return str(dates)


def detect_str_precision(datestr: str) -> Literal["s", "ms", "us", "ns"]:
    """Detects the precision of a datetime str by counting the number of digits after .
    Parameters
    ----------
    datestr : str
        Input string

    Returns
    -------
    Literal['s', 'ms', 'us', 'ns']
        The precision of the string based on the number of digits after .

    Raises
    ------
    ValueError
        Raises a ValueError if the input string is not a datetime string or if
        the number of digits after . is not evenly divisible by 3
    """

    digits_matching = re.search(r"\.(\d+)", datestr)
    if not digits_matching:
        return "s"
    n = len(digits_matching.group(1))
    if n <= 3:
        return "ms"
    elif n <= 6:
        return "us"
    elif n <= 9:
        return "ns"
    else:
        raise ValueError("Finer than ns precision.")


def match_datetime_format(
    dates: Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime],
    target: Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime],
) -> Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime]:
    """Converts the provided datetime-like objects to the same datetime format
    as the provided target, ensuring that the precisions match

    Parameters
    ----------
    dates : Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime]
        A datetime-like object or array of datetime-like objects to be converted
    target : Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime]
        A datetime-like object or array of datetime-like objects which the dates
        input will be converted to match

    Returns
    -------
    Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime]
        The datetime-like values of the date parameter, converted to a format
        which matches that of the target input

    Raises
    ------
    ValueError
        If the target parameter provided is not a datetime-time object or array
        of datetime-like objects
    """
    if isinstance(target, str):
        precision = detect_str_precision(target)
        return to_datestr(dates, precision)
    if isinstance(target, xr.DataArray):
        target = target.values
    if isinstance(target, pd.Series):
        target = target.to_numpy()
    if hasattr(target, "__iter__"):
        target = target[0]
    if isinstance(target, str):
        precision = detect_str_precision(target)
        return to_datestr(dates, precision)
    if isinstance(target, cftime.datetime):
        return to_cftime(dates, target.calendar)
    if isinstance(target, pd.Timestamp):
        precision = target.unit
        return to_timestamp(dates, precision=precision)
    if isinstance(target, np.datetime64):
        precision = np.datetime_data(target)[0]
        return to_datetime64(dates, precision=precision)
    if isinstance(target, datetime.datetime):
        return to_datetime(dates)
    raise ValueError("Target is not a valid datetime format")

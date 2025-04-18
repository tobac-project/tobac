from datetime import datetime

import numpy as np
import pandas as pd
import cftime
import pytest

import tobac.utils.datetime as datetime_utils


def test_to_cftime():
    """Test conversion of datetime types to cftime calendars
    """
    test_dates = [
        "2000-01-01",
        "2000-01-01 00:00:00",
        datetime(2000, 1, 1),
        np.datetime64("2000-01-01 00:00:00.000000000"),
        np.datetime64("2000-01-01 00:00:00"),
        pd.to_datetime("2000-01-01"),
        cftime.datetime(2000, 1, 1),
        cftime.DatetimeGregorian(2000, 1, 1),
        cftime.Datetime360Day(2000, 1, 1),
        cftime.DatetimeNoLeap(2000, 1, 1),
    ]

    for date in test_dates:
        assert datetime_utils.to_cftime(date, "standard") == cftime.datetime(2000, 1, 1)
        assert datetime_utils.to_cftime(date, "gregorian") == cftime.DatetimeGregorian(
            2000, 1, 1
        )
        assert datetime_utils.to_cftime(date, "360_day") == cftime.Datetime360Day(
            2000, 1, 1
        )
        assert datetime_utils.to_cftime(date, "365_day") == cftime.DatetimeNoLeap(
            2000, 1, 1
        )

    # Test array-like input
    for date in test_dates:
        assert datetime_utils.to_cftime([date], "standard")[0] == cftime.datetime(2000, 1, 1)
        assert datetime_utils.to_cftime([date], "gregorian")[0] == cftime.DatetimeGregorian(
            2000, 1, 1
        )
        assert datetime_utils.to_cftime([date], "360_day")[0] == cftime.Datetime360Day(
            2000, 1, 1
        )
        assert datetime_utils.to_cftime([date], "365_day")[0] == cftime.DatetimeNoLeap(
            2000, 1, 1
        )


def test_to_timestamp():
    """Test conversion of various datetime types to pandas timestamps
    """
    test_dates = [
        "2000-01-01",
        "2000-01-01 00:00:00",
        datetime(2000, 1, 1),
        np.datetime64("2000-01-01 00:00:00.000000000"),
        np.datetime64("2000-01-01 00:00:00"),
        pd.to_datetime("2000-01-01"),
        cftime.datetime(2000, 1, 1),
        cftime.DatetimeGregorian(2000, 1, 1),
        cftime.Datetime360Day(2000, 1, 1),
        cftime.DatetimeNoLeap(2000, 1, 1),
    ]

    for date in test_dates:
        assert datetime_utils.to_timestamp(date) == pd.to_datetime("2000-01-01")

    # Test array input
    for date in test_dates:
        assert datetime_utils.to_timestamp([date])[0] == pd.to_datetime("2000-01-01")


def test_to_datetime():
    """Test conversion of various datetime types to python datetime
    """
    test_dates = [
        "2000-01-01",
        "2000-01-01 00:00:00",
        datetime(2000, 1, 1),
        np.datetime64("2000-01-01 00:00:00.000000000"),
        np.datetime64("2000-01-01 00:00:00"),
        pd.to_datetime("2000-01-01"),
        cftime.datetime(2000, 1, 1),
        cftime.DatetimeGregorian(2000, 1, 1),
        cftime.Datetime360Day(2000, 1, 1),
        cftime.DatetimeNoLeap(2000, 1, 1),
    ]

    for date in test_dates:
        assert datetime_utils.to_datetime(date) == datetime(2000, 1, 1)

    # Test array input
    for date in test_dates:
        assert datetime_utils.to_datetime([date])[0] == datetime(2000, 1, 1)


def test_to_datetime64():
    """Test conversion of various datetime types to numpy datetime64
    """
    test_dates = [
        "2000-01-01",
        "2000-01-01 00:00:00",
        datetime(2000, 1, 1),
        np.datetime64("2000-01-01 00:00:00.000000000"),
        np.datetime64("2000-01-01 00:00:00"),
        pd.to_datetime("2000-01-01"),
        cftime.datetime(2000, 1, 1),
        cftime.DatetimeGregorian(2000, 1, 1),
        cftime.Datetime360Day(2000, 1, 1),
        cftime.DatetimeNoLeap(2000, 1, 1),
    ]

    for date in test_dates:
        assert datetime_utils.to_datetime64(date) == np.datetime64(
            "2000-01-01 00:00:00.000000000"
        )

    # Test array input
    for date in test_dates:
        assert datetime_utils.to_datetime64([date])[0] == np.datetime64(
            "2000-01-01 00:00:00.000000000"
        )


def test_to_datestr():
    """Test conversion of various datetime types to ISO format datestring
    """
    test_dates = [
        "2000-01-01",
        "2000-01-01 00:00:00",
        datetime(2000, 1, 1),
        np.datetime64("2000-01-01 00:00:00.000000000"),
        np.datetime64("2000-01-01 00:00:00"),
        pd.to_datetime("2000-01-01"),
        cftime.datetime(2000, 1, 1),
        cftime.DatetimeGregorian(2000, 1, 1),
        cftime.Datetime360Day(2000, 1, 1),
        cftime.DatetimeNoLeap(2000, 1, 1),
    ]

    for date in test_dates:
        assert (
            datetime_utils.to_datestr(date) == "2000-01-01T00:00:00.000000000"
            or datetime_utils.to_datestr(date) == "2000-01-01T00:00:00"
        )


def test_to_datestr_array():
    """Test conversion of arrays of various datetime types to ISO format 
    datestring
    """
    test_dates = [
        "2000-01-01",
        "2000-01-01 00:00:00",
        datetime(2000, 1, 1),
        np.datetime64("2000-01-01 00:00:00.000000000"),
        np.datetime64("2000-01-01 00:00:00"),
        pd.to_datetime("2000-01-01"),
        cftime.datetime(2000, 1, 1),
        cftime.DatetimeGregorian(2000, 1, 1),
        cftime.Datetime360Day(2000, 1, 1),
        cftime.DatetimeNoLeap(2000, 1, 1),
    ]
    for date in test_dates:
        assert datetime_utils.to_datestr([date]) == [
            "2000-01-01T00:00:00.000000000"
        ] or datetime_utils.to_datestr([date]) == ["2000-01-01T00:00:00"]


def test_match_datetime_format():
    """Test match_datetime_format for various datetime-like combinations
    """
    test_dates = [
        "2000-01-01T00:00:00.000000000",
        datetime(2000, 1, 1),
        np.datetime64("2000-01-01 00:00:00.000000000"),
        pd.to_datetime("2000-01-01"),
        cftime.datetime(2000, 1, 1),
        cftime.DatetimeGregorian(2000, 1, 1),
        cftime.Datetime360Day(2000, 1, 1),
        cftime.DatetimeNoLeap(2000, 1, 1),
    ]

    for target in test_dates:
        for date in test_dates:
            assert datetime_utils.match_datetime_format(date, target) == target


def test_match_datetime_format_array():
    """Test match_datetime_format for various datetime-like combinations with 
    array input
    """
    test_dates = [
        "2000-01-01T00:00:00.000000000",
        datetime(2000, 1, 1),
        np.datetime64("2000-01-01 00:00:00.000000000"),
        pd.to_datetime("2000-01-01"),
        cftime.datetime(2000, 1, 1),
        cftime.DatetimeGregorian(2000, 1, 1),
        cftime.Datetime360Day(2000, 1, 1),
        cftime.DatetimeNoLeap(2000, 1, 1),
    ]

    for target in test_dates:
        for date in test_dates:
            assert datetime_utils.match_datetime_format([date], [target]) == np.array(
                [target]
            )


def test_match_datetime_format_error():
    """ Test that if a non datetime-like object is provided as target to 
    match_datetime_format that a ValueError is raised:
    """
    with pytest.raises(ValueError, match="Target is not a valid datetime*"):
        datetime_utils.match_datetime_format(datetime(2000, 1, 1), 1.5)

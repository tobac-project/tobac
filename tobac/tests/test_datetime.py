from datetime import datetime

import numpy as np
import pandas as pd
import cftime

import tobac.utils.datetime as datetime_utils


def test_to_cftime():
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


def test_to_timestamp():
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


def test_to_datetime():
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


def test_to_datetime64():
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


def test_to_datestr():
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


def test_match_datetime_format():
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

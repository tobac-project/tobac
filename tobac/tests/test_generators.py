"""Unit tests for tobac.utils.generators module"""

from datetime import datetime, timedelta

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pandas.testing import assert_frame_equal

from tobac.utils import generators


def test_field_and_features_over_time():
    """Test iterating over field_and_features_over_time generator
    """
    test_data = xr.DataArray(
        np.zeros([2, 10, 10]),
        dims=("time", "y", "x"),
        coords={"time": [datetime(2000, 1, 1), datetime(2000, 1, 1, 1)]},
    )

    test_features = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "frame": [0, 0, 1],
            "time": [
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
                datetime(2000, 1, 1, 1),
            ],
        }
    )

    iterator = generators.field_and_features_over_time(test_data, test_features)

    iter_0 = next(iterator)

    assert iter_0[0] == 0
    assert iter_0[1] == np.datetime64("2000-01-01")
    assert np.all(iter_0[2] == test_data.isel(time=0))
    assert_frame_equal(
        iter_0[3], test_features[test_features.time == datetime(2000, 1, 1)]
    )

    iter_1 = next(iterator)

    assert iter_1[0] == 1
    assert iter_1[1] == np.datetime64("2000-01-01 01:00:00")
    assert np.all(iter_1[2] == test_data.isel(time=1))
    assert_frame_equal(
        iter_1[3], test_features[test_features.time == datetime(2000, 1, 1, 1)]
    )

    with pytest.raises(StopIteration):
        next(iterator)


def test_field_and_features_over_time_time_padding():
    """Test the time_padding functionality of field_and_features_over_time 
    generator
    """
    test_data = xr.DataArray(
        np.zeros([1, 10, 10]),
        dims=("time", "y", "x"),
        coords={"time": [datetime(2000, 1, 1)]},
    )

    test_features = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "frame": [0, 0, 0],
            "time": [
                datetime(2000, 1, 1),
                datetime(2000, 1, 1, 0, 0, 1),
                datetime(2000, 1, 1, 0, 0, 2),
            ],
        }
    )

    #  Test no time padding
    _, _, _, df_slice = next(
        generators.field_and_features_over_time(test_data, test_features)
    )

    assert len(df_slice) == 1
    assert_frame_equal(df_slice, test_features.loc[0:0])

    # Test time padding of 1 second
    _, _, _, df_slice = next(
        generators.field_and_features_over_time(
            test_data, test_features, time_padding=timedelta(seconds=1)
        )
    )

    assert len(df_slice) == 2
    assert_frame_equal(df_slice, test_features.loc[0:1])

    # Test time padding of 2 seconds
    _, _, _, df_slice = next(
        generators.field_and_features_over_time(
            test_data, test_features, time_padding=timedelta(seconds=2)
        )
    )

    assert len(df_slice) == 3
    assert_frame_equal(df_slice, test_features.loc[0:2])


def test_field_and_features_over_time_cftime():
    """Test field_and_features_over_time when given cftime datetime formats
    """
    test_data = xr.DataArray(
        np.zeros([2, 10, 10]),
        dims=("time", "y", "x"),
        coords={
            "time": [
                cftime.Datetime360Day(2000, 1, 1),
                cftime.Datetime360Day(2000, 1, 1, 1),
            ]
        },
    )

    test_features = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "frame": [0, 0, 1],
            "time": [
                cftime.Datetime360Day(2000, 1, 1),
                cftime.Datetime360Day(2000, 1, 1, 0, 0, 1),
                cftime.Datetime360Day(2000, 1, 1, 1),
            ],
        }
    )

    iterator = generators.field_and_features_over_time(
        test_data, test_features, time_padding=timedelta(seconds=1)
    )

    iter_0 = next(iterator)

    assert iter_0[0] == 0
    assert iter_0[1] == cftime.Datetime360Day(2000, 1, 1)
    assert np.all(iter_0[2] == test_data.isel(time=0))
    assert_frame_equal(iter_0[3], test_features.loc[0:1])

    iter_1 = next(iterator)

    assert iter_1[0] == 1
    assert iter_1[1] == cftime.Datetime360Day(2000, 1, 1, 1)
    assert np.all(iter_1[2] == test_data.isel(time=1))
    assert_frame_equal(
        iter_1[3],
        test_features[test_features.time == cftime.Datetime360Day(2000, 1, 1, 1)],
    )

    with pytest.raises(StopIteration):
        next(iterator)


def test_field_and_features_over_time_time_var_name():
    """Test field_and_features_over_time generator works correctly with a time 
    coordinate name other than "time"
    """
    # Test non-standard time coord name:
    test_data = xr.DataArray(
        np.zeros([2, 10, 10]),
        dims=("time_testing", "y", "x"),
        coords={"time_testing": [datetime(2000, 1, 1), datetime(2000, 1, 1, 1)]},
    )

    test_features = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "frame": [0, 0, 1],
            "time_testing": [
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
                datetime(2000, 1, 1, 1),
            ],
        }
    )

    _ = next(
        generators.field_and_features_over_time(
            test_data, test_features, time_var_name="time_testing"
        )
    )


def test_field_and_features_over_time_time_var_name_error():
    """Test that field_and_features_over_time generator raises the correct 
    error when the name of the time coordinates do not match between the given 
    data and dataframe
    """
    # Test if time_var_name not in dataarray:
    test_data = xr.DataArray(
        np.zeros([2, 10, 10]),
        dims=("time_testing", "y", "x"),
        coords={"time_testing": [datetime(2000, 1, 1), datetime(2000, 1, 1, 1)]},
    )

    test_features = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "frame": [0, 0, 1],
            "time": [
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
                datetime(2000, 1, 1, 1),
            ],
        }
    )

    with pytest.raises(ValueError, match="time not present in input field*"):
        next(generators.field_and_features_over_time(test_data, test_features))

    # Test if time var name not in dataframe:
    test_data = xr.DataArray(
        np.zeros([2, 10, 10]),
        dims=("time", "y", "x"),
        coords={"time": [datetime(2000, 1, 1), datetime(2000, 1, 1, 1)]},
    )

    test_features = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "frame": [0, 0, 1],
            "time_testing": [
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
                datetime(2000, 1, 1, 1),
            ],
        }
    )

    with pytest.raises(ValueError, match="time not present in input feature*"):
        next(generators.field_and_features_over_time(test_data, test_features))

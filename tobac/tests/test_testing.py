"""
Audit of the testing functions that produce our test data.
Who's watching the watchmen, basically.
"""
import pytest
from tobac.testing import generate_single_feature
import tobac.testing as tbtest
from collections import Counter
import pandas as pd
from pandas.testing import assert_frame_equal
import datetime


def lists_equal_without_order(a, b):
    """
    This will make sure the inner list contain the same,
    but doesn't account for duplicate groups.
    """
    for l1 in a:
        check_counter = Counter(l1)
        if not any(Counter(l2) == check_counter for l2 in b):
            return False
    return True


def test_make_feature_blob():
    """Tests ```tobac.testing.make_feature_blob```
    Currently runs the following tests:
    Creates a blob in the right location and cuts off without PBCs
    Blob extends off PBCs for all dimensions when appropriate
    """
    import numpy as np

    # Test without PBCs first, make sure that a blob is generated in the first place.
    # 2D test
    out_blob = tbtest.make_feature_blob(
        np.zeros((10, 10)),
        h1_loc=5,
        h2_loc=5,
        h1_size=2,
        h2_size=2,
        shape="rectangle",
        amplitude=1,
    )
    assert np.all(out_blob[4:6, 4:6] == 1)
    # There should be exactly 4 points of value 1.
    assert np.sum(out_blob) == 4 and np.min(out_blob) == 0

    # 3D test
    out_blob = tbtest.make_feature_blob(
        np.zeros((10, 10, 10)),
        h1_loc=5,
        h2_loc=5,
        v_loc=5,
        h1_size=2,
        h2_size=2,
        v_size=2,
        shape="rectangle",
        amplitude=1,
    )
    assert np.all(out_blob[4:6, 4:6, 4:6] == 1)
    # There should be exactly 8 points of value 1.
    assert np.sum(out_blob) == 8 and np.min(out_blob) == 0

    # Test that it cuts things off along a boundary.
    # 2D test
    out_blob = tbtest.make_feature_blob(
        np.zeros((10, 10)),
        h1_loc=5,
        h2_loc=9,
        h1_size=2,
        h2_size=4,
        shape="rectangle",
        amplitude=1,
    )
    assert np.all(out_blob[4:6, 7:10] == 1)
    assert np.all(out_blob[4:6, 0:1] == 0)
    # There should be exactly 4 points of value 1.
    assert np.sum(out_blob) == 6 and np.min(out_blob) == 0

    # 3D test
    out_blob = tbtest.make_feature_blob(
        np.zeros((10, 10, 10)),
        h1_loc=5,
        h2_loc=9,
        v_loc=5,
        h1_size=2,
        h2_size=4,
        v_size=2,
        shape="rectangle",
        amplitude=1,
    )
    assert np.all(out_blob[4:6, 4:6, 7:10] == 1)
    assert np.all(out_blob[4:6, 4:6, 0:1] == 0)
    # There should be exactly 4 points of value 1.
    assert np.sum(out_blob) == 12 and np.min(out_blob) == 0


def test_generate_single_feature():
    """Tests the `generate_single_feature` function.
    Currently runs the following tests:
    A single feature is generated

    """

    # Testing a simple 2D case
    expected_df = pd.DataFrame.from_dict(
        [
            {
                "hdim_1": 1,
                "hdim_2": 1,
                "frame": 0,
                "time": datetime.datetime(2022, 1, 1, 0, 0),
                "feature": 1,
            }
        ]
    )
    assert_frame_equal(
        generate_single_feature(1, 1, frame_start=0).sort_index(axis=1),
        expected_df.sort_index(axis=1),
    )

    # Testing a simple 2D case with movement
    expected_df = pd.DataFrame.from_dict(
        [
            {
                "hdim_1": 1,
                "hdim_2": 1,
                "frame": 0,
                "time": datetime.datetime(2022, 1, 1, 0, 0),
                "feature": 1,
            },
            {
                "hdim_1": 2,
                "hdim_2": 2,
                "frame": 1,
                "time": datetime.datetime(2022, 1, 1, 0, 5),
                "feature": 2,
            },
            {
                "hdim_1": 3,
                "hdim_2": 3,
                "frame": 2,
                "time": datetime.datetime(2022, 1, 1, 0, 10),
                "feature": 3,
            },
            {
                "hdim_1": 4,
                "hdim_2": 4,
                "frame": 3,
                "time": datetime.datetime(2022, 1, 1, 0, 15),
                "feature": 4,
            },
        ]
    )
    assert_frame_equal(
        generate_single_feature(
            1, 1, frame_start=0, num_frames=4, spd_h1=1, spd_h2=1
        ).sort_index(axis=1),
        expected_df.sort_index(axis=1),
    )

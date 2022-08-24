"""
Test for the trackpy tracking functions
"""
import pytest
import tobac.testing
import tobac.tracking
import copy
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np


def test_linking_trackpy():
    """Function to test ```tobac.tracking.linking_trackpy```
    Currently tests:
    2D tracking
    3D tracking
    3D tracking with PBCs
    """

    # Test 2D tracking of a simple moving feature
    test_feature = tobac.testing.generate_single_feature(
        1,
        1,
        min_h1=0,
        max_h1=100,
        min_h2=0,
        max_h2=100,
        frame_start=0,
        num_frames=5,
        spd_h1=1,
        spd_h2=1,
    )

    expected_out_feature = copy.deepcopy(test_feature)
    expected_out_feature["cell"] = 1.0

    actual_out_feature = tobac.tracking.linking_trackpy(
        test_feature, None, 5, 1000, v_max=20000, method_linking="predict"
    )
    print(actual_out_feature)
    # Just want to remove the time_cell column here.
    actual_out_feature = actual_out_feature[
        ["hdim_1", "hdim_2", "frame", "time", "cell", "feature"]
    ]
    actual_out_feature["cell"] = actual_out_feature["cell"].astype(float)
    assert_frame_equal(
        expected_out_feature.sort_index(axis=1), actual_out_feature.sort_index(axis=1)
    )

def test_trackpy_predict():
    """Function to test if linking_trackpy() with method='predict' correctly links two 
    features at constant speeds crossing each other.
    """

    cell_1 = tobac.testing.generate_single_feature(
        1,
        1,
        min_h1=0,
        max_h1=100,
        min_h2=0,
        max_h2=100,
        frame_start=0,
        num_frames=5,
        spd_h1=20,
        spd_h2=20,
    )

    cell_1_expected = copy.deepcopy(cell_1)
    cell_1_expected["cell"] = 1

    cell_2 = tobac.testing.generate_single_feature(
        1,
        100,
        min_h1=0,
        max_h1=100,
        min_h2=0,
        max_h2=100,
        frame_start=0,
        num_frames=5,
        spd_h1=20,
        spd_h2=-20,
    )

    cell_2_expected = copy.deepcopy(cell_2)
    cell_2_expected["cell"] = 2

    features = pd.concat([cell_1, cell_2])
    expected_output = pd.concat([cell_1_expected, cell_2_expected])

    output = tobac.linking_trackpy(features, 
                                None, 
                                1, 
                                1, 
                                d_max=100,
                                method_linking='predict')
    output = output[
        ["hdim_1", "hdim_2", "frame", "time",  "feature", "cell"]
    ]

    assert_frame_equal(
        expected_output.sort_index(), output.sort_index()
    )
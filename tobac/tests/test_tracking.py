"""
Test for the trackpy tracking functions
"""
import pytest
import tobac.testing
import tobac.tracking
import copy
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
import trackpy as tp


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


@pytest.mark.parametrize(
    "max_trackpy, max_tobac, adaptive_step, adaptive_stop",
    [(5, 10, None, None), (5, 10, 0.9, 0.1)],
)
def test_keep_trackpy_parameters(max_trackpy, max_tobac, adaptive_step, adaptive_stop):
    """
    Tests that tobac does not change the parameters of trackpy
    """

    tp.linking.Linker.MAX_SUB_NET_SIZE = max_trackpy
    tp.linking.Linker.MAX_SUB_NET_SIZE_ADAPTIVE = max_trackpy

    expected_value = tp.linking.Linker.MAX_SUB_NET_SIZE
    expected_value_adaptive = tp.linking.Linker.MAX_SUB_NET_SIZE_ADAPTIVE

    data = tobac.testing.make_simple_sample_data_2D()
    dxy, dt = tobac.utils.get_spacings(data)
    features = tobac.feature_detection.feature_detection_multithreshold(
        data, dxy, threshold=0.1
    )

    track = tobac.linking_trackpy(
        features,
        data,
        dt=dt,
        dxy=dxy,
        v_max=100,
        adaptive_step=adaptive_step,
        adaptive_stop=adaptive_stop,
        subnetwork_size=max_tobac,
    )

    assert expected_value == tp.linking.Linker.MAX_SUB_NET_SIZE
    assert expected_value_adaptive == tp.linking.Linker.MAX_SUB_NET_SIZE_ADAPTIVE


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

    output = tobac.linking_trackpy(
        features, None, 1, 1, d_max=100, method_linking="predict"
    )

    output_random = tobac.linking_trackpy(
        features, None, 1, 1, d_max=100, method_linking="random"
    )

    # check that the two methods of linking produce different results for this case
    assert not output_random.equals(output)

    # sorting and dropping indices for comparison with the expected output
    output = output[["hdim_1", "hdim_2", "frame", "time", "feature", "cell"]]

    assert_frame_equal(expected_output.sort_index(), output.sort_index())


def test_tracking_extrapolation():
    """Tests the extrapolation capabilities of tracking.
    Right now, this is not implemented, so it will raise an error.
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
    with pytest.raises(NotImplementedError):
        output = tobac.linking_trackpy(
            cell_1, None, 1, 1, d_max=100, method_linking="predict", extrapolate=1
        )

"""
Test for the trackpy tracking functions
"""

from __future__ import annotations
import datetime
import copy

import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import trackpy as tp

import tobac.testing
import tobac.tracking


def convert_cell_dtype_if_appropriate(output, expected_output):
    """Helper function to convert datatype of output if
    necessary. Fixes a bug in testing on some OS/Python versions that cause
    default int types to be different

    Parameters
    ----------
    output: pd.DataFrame
        the pandas dataframe to base cell datatype off of
    expected_output: pd.DataFrame
        the pandas dataframe to change the cell datatype

    Returns
    -------
    expected_output: pd.DataFrame
        an adjusted dataframe with a matching int dtype
    """

    # if they are already the same datatype, can return.
    if output["cell"].dtype == expected_output["cell"].dtype:
        return expected_output

    if output["cell"].dtype == np.int32:
        expected_output["cell"] = expected_output["cell"].astype(np.int32)

    if output["cell"].dtype == np.int64:
        expected_output["cell"] = expected_output["cell"].astype(np.int64)

    return expected_output


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
        PBC_flag="none",
    )

    expected_out_feature = copy.deepcopy(test_feature)
    expected_out_feature["cell"] = 1

    actual_out_feature = tobac.tracking.linking_trackpy(
        test_feature,
        None,
        5,
        1000,
        v_max=10000,
        method_linking="predict",
        PBC_flag="none",
    )
    # Just want to remove the time_cell column here.
    actual_out_feature = actual_out_feature[
        ["hdim_1", "hdim_2", "frame", "feature", "time", "cell", "idx"]
    ]

    expected_out_feature = convert_cell_dtype_if_appropriate(
        actual_out_feature, expected_out_feature
    )
    assert_frame_equal(
        expected_out_feature.sort_index(axis=1), actual_out_feature.sort_index(axis=1)
    )

    # Test 3D tracking of a simple moving feature
    test_feature = tobac.testing.generate_single_feature(
        1,
        1,
        start_v=1,
        min_h1=0,
        max_h1=100,
        min_h2=0,
        max_h2=100,
        frame_start=0,
        num_frames=5,
        spd_h1=1,
        spd_h2=1,
        spd_v=1,
        PBC_flag="none",
    )

    expected_out_feature = copy.deepcopy(test_feature)
    expected_out_feature["cell"] = 1

    actual_out_feature = tobac.tracking.linking_trackpy(
        test_feature,
        None,
        5,
        1000,
        dz=1000,
        v_max=10000,
        method_linking="random",
        PBC_flag="none",
        vertical_coord=None,
    )
    # Just want to remove the time_cell column here.
    actual_out_feature = actual_out_feature[
        ["hdim_1", "hdim_2", "vdim", "frame", "feature", "time", "cell", "idx"]
    ]

    expected_out_feature = convert_cell_dtype_if_appropriate(
        actual_out_feature, expected_out_feature
    )

    assert_frame_equal(
        expected_out_feature.sort_index(axis=1), actual_out_feature.sort_index(axis=1)
    )

    # Test 3D tracking of a simple moving feature with periodic boundaries on hdim_1
    test_feature = tobac.testing.generate_single_feature(
        1,
        1,
        start_v=1,
        min_h1=0,
        max_h1=10,
        min_h2=0,
        max_h2=10,
        frame_start=0,
        num_frames=8,
        spd_h1=3,
        spd_h2=1,
        spd_v=1,
        PBC_flag="hdim_1",
    )

    expected_out_feature = copy.deepcopy(test_feature)
    expected_out_feature["cell"] = 1

    actual_out_feature = tobac.tracking.linking_trackpy(
        test_feature,
        None,
        1,
        1,
        dz=1,
        min_h1=0,
        max_h1=10,
        min_h2=0,
        max_h2=10,
        v_max=4,
        method_linking="random",
        vertical_coord=None,
        PBC_flag="hdim_1",
    )
    # Just want to remove the time_cell column here.
    actual_out_feature = actual_out_feature[
        ["hdim_1", "hdim_2", "vdim", "frame", "feature", "time", "cell", "idx"]
    ]
    expected_out_feature = convert_cell_dtype_if_appropriate(
        actual_out_feature, expected_out_feature
    )

    assert_frame_equal(
        expected_out_feature.sort_index(axis=1), actual_out_feature.sort_index(axis=1)
    )

    # Test 3D tracking of a simple moving feature with periodic boundaries on hdim_2
    test_feature = tobac.testing.generate_single_feature(
        1,
        1,
        start_v=1,
        min_h1=0,
        max_h1=10,
        min_h2=0,
        max_h2=10,
        frame_start=0,
        num_frames=8,
        spd_h1=1,
        spd_h2=3,
        spd_v=1,
        PBC_flag="hdim_2",
    )

    expected_out_feature = copy.deepcopy(test_feature)
    expected_out_feature["cell"] = 1

    actual_out_feature = tobac.tracking.linking_trackpy(
        test_feature,
        None,
        1,
        1,
        dz=1,
        min_h1=0,
        max_h1=10,
        min_h2=0,
        max_h2=10,
        v_max=4,
        method_linking="random",
        vertical_coord=None,
        PBC_flag="hdim_2",
    )
    # Just want to remove the time_cell column here.
    actual_out_feature = actual_out_feature.drop("time_cell", axis=1)
    expected_out_feature = convert_cell_dtype_if_appropriate(
        actual_out_feature, expected_out_feature
    )

    assert_frame_equal(
        expected_out_feature.sort_index(axis=1), actual_out_feature.sort_index(axis=1)
    )

    # Test 3D tracking of a simple moving feature with periodic boundaries on both hdim_1 and hdim_2
    test_feature = tobac.testing.generate_single_feature(
        1,
        1,
        start_v=1,
        min_h1=0,
        max_h1=10,
        min_h2=0,
        max_h2=10,
        frame_start=0,
        num_frames=8,
        spd_h1=3,
        spd_h2=3,
        spd_v=0,
        PBC_flag="both",
    )

    expected_out_feature = copy.deepcopy(test_feature)
    expected_out_feature["cell"] = 1

    actual_out_feature = tobac.tracking.linking_trackpy(
        test_feature,
        None,
        1,
        1,
        dz=1,
        min_h1=0,
        max_h1=10,
        min_h2=0,
        max_h2=10,
        v_max=5,
        method_linking="random",
        vertical_coord=None,
        PBC_flag="both",
    )
    # Just want to remove the time_cell column here.
    actual_out_feature = actual_out_feature[
        ["hdim_1", "hdim_2", "vdim", "frame", "feature", "time", "cell", "idx"]
    ]
    expected_out_feature = convert_cell_dtype_if_appropriate(
        actual_out_feature, expected_out_feature
    )
    assert_frame_equal(
        expected_out_feature.sort_index(axis=1), actual_out_feature.sort_index(axis=1)
    )


@pytest.mark.parametrize(
    "point_init, speed, dxy, actual_dz, v_max, use_dz, features_connected",
    [
        ((0, 0, 0), (1, 0, 0), 1000, 100, 200, True, True),
        ((0, 0, 0), (1, 0, 0), 1000, 100, 200, False, True),
        ((0, 0, 0), (5, 0, 0), 1000, 100, 200, True, False),
        ((0, 0, 0), (5, 0, 0), 1000, 100, 200, False, False),
    ],
)
def test_3D_tracking_min_dist_z(
    point_init, speed, dxy, actual_dz, v_max, use_dz, features_connected
):
    """Tests ```tobac.tracking.linking_trackpy``` with
    points in z with varying distances between them.

    Parameters
    ----------
    point_init: 3D array-like
        Initial point (z, y, x)
    speed: 3D array-like
        Speed of the feature (z, y, x)
    dxy: float
        grid spacing for dx and dy
    actual_dz: float
        grid spacing for Z
    use_dz: bool
        True to use the passed in constant dz, False
        to use the calculated vertical coordinates
    features_connected: bool
        Do we expect the features to be connected?
    """

    test_feature = tobac.testing.generate_single_feature(
        start_h1=point_init[1],
        start_h2=point_init[2],
        start_v=point_init[0],
        min_h1=0,
        max_h1=100,
        min_h2=0,
        max_h2=100,
        frame_start=0,
        num_frames=2,
        spd_h1=speed[1],
        spd_h2=speed[2],
        spd_v=speed[0],
        PBC_flag="none",
    )
    if not use_dz:
        test_feature["z"] = test_feature["vdim"] * actual_dz

    expected_out_feature = copy.deepcopy(test_feature)

    if features_connected:
        expected_out_feature["cell"] = 1
    else:
        expected_out_feature["cell"] = -1

    common_params = {
        "features": test_feature,
        "field_in": None,
        "dt": 1,
        "time_cell_min": 1,
        "dxy": dxy,
        "v_max": v_max,
        "method_linking": "random",
        "cell_number_unassigned": -1,
    }
    if use_dz:
        common_params["dz"] = actual_dz
        common_params["vertical_coord"] = None
    else:
        common_params["vertical_coord"] = "z"

    actual_out_feature = tobac.tracking.linking_trackpy(**common_params)
    # Just want to remove the time_cell column here.
    actual_out_feature = actual_out_feature.drop("time_cell", axis=1)
    expected_out_feature = convert_cell_dtype_if_appropriate(
        actual_out_feature, expected_out_feature
    )
    assert_frame_equal(
        expected_out_feature.sort_index(axis=1), actual_out_feature.sort_index(axis=1)
    )

    # Check that we only add two columns, and all the other columns are the same as the input features
    assert len(actual_out_feature.columns.tolist()) == len(
        set(actual_out_feature.columns.tolist())
    )
    assert set(actual_out_feature.columns.tolist()) - set(
        test_feature.columns.tolist()
    ) == {"cell"}


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
        max_h1=101,
        min_h2=0,
        max_h2=101,
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
        max_h1=101,
        min_h2=0,
        max_h2=101,
        frame_start=0,
        num_frames=5,
        spd_h1=20,
        spd_h2=-20,
    )

    cell_2_expected = copy.deepcopy(cell_2)
    cell_2_expected["cell"] = np.int32(2)

    features = pd.concat([cell_1, cell_2], ignore_index=True, verify_integrity=True)
    expected_output = pd.concat(
        [cell_1_expected, cell_2_expected], ignore_index=True, verify_integrity=True
    )

    output = tobac.linking_trackpy(
        features, None, 1, 1, d_max=100, method_linking="predict"
    )

    output_random = tobac.linking_trackpy(
        features, None, 1, 1, d_max=100, method_linking="random"
    )

    # check that the two methods of linking produce different results for this case
    assert not output_random.equals(output)

    # sorting and dropping indices for comparison with the expected output
    output = output[["hdim_1", "hdim_2", "frame", "idx", "time", "feature", "cell"]]
    expected_output = convert_cell_dtype_if_appropriate(output, expected_output)

    assert_frame_equal(expected_output.sort_index(), output.sort_index())

    # Check that we only add two columns, and all the other columns are the same as the input features
    assert len(output.columns.tolist()) == len(set(output.columns.tolist()))
    assert set(output.columns.tolist()) - set(features.columns.tolist()) == {"cell"}


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


def test_argument_logic():
    """Tests whether missing arguments are handled correctly,
    i.e. whether a ValueError is raised if neither d_min, d_max nor v_max have
    been provided to tobac.linking_trackpy.
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
    with pytest.raises(ValueError):
        output = tobac.linking_trackpy(
            cell_1, None, 1, 1, d_min=None, d_max=None, v_max=None
        )


def test_untracked_nat():
    """
    Tests to make sure that the untracked cells don't have timedelta assigned.
    """
    features = tobac.testing.generate_single_feature(
        1,
        1,
        min_h1=0,
        max_h1=101,
        min_h2=0,
        max_h2=101,
        frame_start=0,
        num_frames=2,
        spd_h1=50,
        spd_h2=50,
    )

    output = tobac.linking_trackpy(
        features,
        None,
        1,
        1,
        d_max=40,
        method_linking="random",
        cell_number_unassigned=-1,
        time_cell_min=2,
    )

    assert np.all(output["cell"].values == np.array([-1, -1]))
    # NaT values cannot be compared, so instead we check for null values
    # and check for the data type
    assert np.all(pd.isnull(output["time_cell"]))
    # the exact data type depends on architecture, so
    # instead just check by name
    assert output["time_cell"].dtype.name == "timedelta64[ns]"


@pytest.mark.parametrize(
    "cell_time_lengths, min_time_length, expected_there",
    [
        (
            [0, 5, 10, 120, 300],
            0,
            [True, True, True, True, True],
        ),
        (
            [0, 5, 10, 120, 300],
            5,
            [False, False, True, True, True],
        ),
        (
            [0, 5, 10, 120, 300],
            6,
            [False, False, True, True, True],
        ),
        (
            [0, 5, 10, 120, 300],
            120,
            [False, False, False, False, True],
        ),
        (
            [0, 5, 10, 120, 300],
            900,
            [False, False, False, False, False],
        ),
    ],
)
def test_time_cell_min(
    cell_time_lengths: list[int],
    min_time_length: int,
    expected_there: list[bool],
):
    """
    Tests time_cell_min in particle-based tracking
    Parameters
    ----------
    cell_time_lengths: list[int]
        Length that each cell appears for
    min_time_length: int
        time_cell_min value
    expected_there: list[bool]
        whether to expect the cell there.

    """
    # delta time automatically set by smallest difference between cell lengths
    delta_time = 1
    # how far horizontally to separate test features
    sep_factor = 2
    all_feats = list()
    # generate dataframes
    for i, cell_time in enumerate(cell_time_lengths):
        curr_feat = tobac.testing.generate_single_feature(
            start_h1=i * sep_factor,
            start_h2=i * sep_factor,
            spd_h1=1,
            spd_h2=1,
            max_h1=1000,
            max_h2=1000,
            num_frames=cell_time // delta_time,
            dt=datetime.timedelta(seconds=delta_time),
        )
        curr_feat["orig_cell_num"] = i

        all_feats.append(curr_feat)
    all_feats_df = tobac.utils.combine_feature_dataframes(all_feats)

    all_feats_tracked = tobac.tracking.linking_trackpy(
        all_feats_df,
        field_in=None,
        dt=delta_time,
        dxy=1000,
        v_max=(1000 * 2) / delta_time,
        cell_number_unassigned=-1,
        time_cell_min=min_time_length,
    )
    all_feats_tracked_drop_no_cells = all_feats_tracked[all_feats_tracked["cell"] != -1]

    for i, cell_expected in enumerate(expected_there):
        if cell_expected:
            expected_val = cell_time_lengths[i] // delta_time
        else:
            expected_val = 0
        assert (
            np.sum(all_feats_tracked_drop_no_cells["orig_cell_num"] == i)
            == expected_val
        )


def test_trackpy_predict_PBC():
    """Test if predictive tracking with PBCs works correctly"""

    test_features = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5, 6, 7, 8],
            "hdim_1": [85, 15, 95, 5, 5, 95, 15, 85],
            "hdim_2": [50, 45, 50, 45, 50, 45, 50, 45],
            "frame": [0, 0, 1, 1, 2, 2, 3, 3],
            "time": [
                datetime.datetime(2000, 1, 1),
                datetime.datetime(2000, 1, 1),
                datetime.datetime(2000, 1, 1, 0, 5),
                datetime.datetime(2000, 1, 1, 0, 5),
                datetime.datetime(2000, 1, 1, 0, 10),
                datetime.datetime(2000, 1, 1, 0, 10),
                datetime.datetime(2000, 1, 1, 0, 15),
                datetime.datetime(2000, 1, 1, 0, 15),
            ],
        }
    )

    output_random_no_pbc = tobac.linking_trackpy(
        test_features, None, 1, 1, d_max=10, method_linking="random"
    )

    # Assert cell does not cross border
    assert output_random_no_pbc["cell"].tolist() == [1, 2, 1, 2, 2, 1, 2, 1]

    output_random_pbc = tobac.linking_trackpy(
        test_features,
        None,
        1,
        1,
        d_max=10,
        method_linking="random",
        PBC_flag="hdim_1",
        min_h1=0,
        max_h1=100,
        min_h2=0,
        max_h2=100,
    )

    # Assert cell does not cross border even with PBC because of random tracking
    assert output_random_pbc["cell"].tolist() == [1, 2, 1, 2, 2, 1, 2, 1]

    output_predict_no_pbc = tobac.linking_trackpy(
        test_features, None, 1, 1, d_max=10, method_linking="predict"
    )

    # Assert that without PBCs predictive tracking creates 4 cells because the d_max criteria is too small
    assert output_predict_no_pbc["cell"].tolist() == [1, 2, 1, 2, 3, 4, 3, 4]

    output_predict_pbc = tobac.linking_trackpy(
        test_features,
        None,
        1,
        1,
        d_max=10,
        method_linking="predict",
        PBC_flag="hdim_1",
        min_h1=0,
        max_h1=100,
        min_h2=0,
        max_h2=100,
    )

    # Assert with PBCs and prdictive tracking the cells should cross the border
    assert output_predict_pbc["cell"].tolist() == [1, 2, 1, 2, 1, 2, 1, 2]

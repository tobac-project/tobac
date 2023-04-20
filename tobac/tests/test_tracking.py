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
    expected_out_feature["cell"] = 1

    actual_out_feature = tobac.tracking.linking_trackpy(
        test_feature,
        None,
        5,
        1000,
        v_max=10000,
        method_linking="predict",
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

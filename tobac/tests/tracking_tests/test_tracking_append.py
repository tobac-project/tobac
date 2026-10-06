"""
Test for the trackpy tracking functions that append one track to another track
"""

import tobac.testing
import tobac.tracking
import pytest
import copy
import pandas as pd
import numpy as np
import datetime


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


@pytest.mark.parametrize(
    "features_points, dt, dxy, v_max, memory, time_cell_min",
    [
        (
            (
                (
                    (0, 1, 2, 3, 4, 5),
                    (1, 2, 3, 4, 5, 6),
                ),
                (
                    (10, 20, 30, 40, 50, 60),
                    (10, 20, 30, 40, 50, 60),
                ),
                (
                    (6, 8),
                    (7, 9),
                ),
            ),
            60,
            1000,
            30,
            0,
            0,
        )
    ],
)
def test_append_tracking_single_track(
    features_points: tuple[tuple[tuple[float]]],
    dt: float,
    dxy: float,
    v_max: float,
    memory: int,
    time_cell_min: float,
):
    """
    Function to test (with a single set of feature points) whether append_tracks_trackpy and
    link_trackpy produce the same result.

    Parameters
    ----------
    features_points
    dt
    dxy
    v_max

    Returns
    -------

    """

    all_features = []
    for feature_point_values in features_points:
        v_points = None
        if len(feature_point_values) > 2:
            v_points = feature_point_values[2]

        test_feature = tobac.testing.generate_single_feature(
            start_h1=feature_point_values[0],
            start_h2=feature_point_values[1],
            start_v=v_points,
            min_h1=0,
            max_h1=1000,
            min_h2=0,
            max_h2=1000,
            frame_start=0,
            PBC_flag="none",
        )
        all_features.append(test_feature)

    all_feats = tobac.testing.combine_single_features(all_features, 1)

    # shared tracking parameters
    tracking_params = {
        "dt": dt,
        "dxy": dxy,
        "v_max": v_max,
        "memory": memory,
        "time_cell_min": time_cell_min,
    }

    # Standard tracking
    # base tracking - original tracking function
    orig_tracking = tobac.tracking.linking_trackpy(all_feats, None, **tracking_params)

    # tracking with appends

    # let's extract the first two times
    first_two_times_df = all_feats[all_feats["frame"] < 2]
    initial_tracking_append = tobac.tracking.linking_trackpy(
        first_two_times_df, None, **tracking_params
    )

    append_all_tracking = tobac.tracking.append_tracks_trackpy(
        initial_tracking_append, all_feats, **tracking_params
    )

    assert tobac.testing.check_tracking_identical(orig_tracking, append_all_tracking)

    # let's try to append one by one, with the full dataframe
    curr_tracking_append = tobac.tracking.linking_trackpy(
        first_two_times_df, None, **tracking_params
    )
    for i in range(3, max(all_feats["frame"]) + 2):
        curr_times_df = all_feats[all_feats["frame"] < i]
        curr_tracking_append = tobac.tracking.append_tracks_trackpy(
            curr_tracking_append, curr_times_df, **tracking_params
        )
    assert tobac.testing.check_tracking_identical(curr_tracking_append, orig_tracking)


@pytest.mark.parametrize(
    "features_points, dt, dxy, v_max, memory, time_cell_min",
    [
        (
            (
                (
                    (0, 1, 2, 3, 4, 5),
                    (1, 2, 3, 4, 5, 6),
                ),
                (
                    (5, 4, 3, 2, 1, 0),
                    (6, 5, 4, 3, 2, 1),
                ),
            ),
            60,
            1000,
            30,
            0,
            0,
        )
    ],
)
def test_append_tracking_single_track_predict(
    features_points: tuple[tuple[tuple[float]]],
    dt: float,
    dxy: float,
    v_max: float,
    memory: int,
    time_cell_min: float,
):
    """
    Function to test (with a single set of feature points) whether append_tracks_trackpy and
    link_trackpy produce the same result.

    Parameters
    ----------
    features_points
    dt
    dxy
    v_max

    Returns
    -------

    """

    all_features = list()
    for feature_point_values in features_points:
        v_points = None
        if len(feature_point_values) > 2:
            v_points = feature_point_values[2]

        test_feature = tobac.testing.generate_single_feature(
            start_h1=feature_point_values[0],
            start_h2=feature_point_values[1],
            start_v=v_points,
            min_h1=0,
            max_h1=1000,
            min_h2=0,
            max_h2=1000,
            frame_start=0,
            PBC_flag="none",
        )
        all_features.append(test_feature)

    all_feats = tobac.testing.combine_single_features(all_features, 1)

    # shared tracking parameters
    tracking_params = {
        "dt": dt,
        "dxy": dxy,
        "v_max": v_max,
        "memory": memory,
        "time_cell_min": time_cell_min,
        "method_linking": "predict",
    }

    # Standard tracking
    # base tracking - original tracking function
    orig_tracking = tobac.tracking.linking_trackpy(all_feats, None, **tracking_params)

    # tracking with appends

    # let's extract the first two times
    first_two_times_df = all_feats[all_feats["frame"] < 2]
    initial_tracking_append = tobac.tracking.linking_trackpy(
        first_two_times_df, None, **tracking_params
    )

    append_all_tracking = tobac.tracking.append_tracks_trackpy(
        initial_tracking_append, all_feats, **tracking_params
    )

    assert tobac.testing.check_tracking_identical(orig_tracking, append_all_tracking)
    # let's try to append one by one, with the full dataframe
    curr_tracking_append = tobac.tracking.linking_trackpy(
        first_two_times_df, None, **tracking_params
    )
    for i in range(3, max(all_feats["frame"]) + 2):
        curr_times_df = all_feats[all_feats["frame"] < i]
        curr_tracking_append = tobac.tracking.append_tracks_trackpy(
            curr_tracking_append, curr_times_df, **tracking_params
        )
    assert tobac.testing.check_tracking_identical(curr_tracking_append, orig_tracking)
    # let's try to append one by one, with only individual times
    curr_tracking_append = tobac.tracking.linking_trackpy(
        first_two_times_df, None, **tracking_params
    )
    for i in range(2, max(all_feats["frame"]) + 1):
        curr_times_df = all_feats[all_feats["frame"] == i]
        curr_tracking_append = tobac.tracking.append_tracks_trackpy(
            curr_tracking_append, curr_times_df, **tracking_params
        )
    assert tobac.testing.check_tracking_identical(curr_tracking_append, orig_tracking)

    # let's try to append one by one, with only individual times
    curr_tracking_append = tobac.tracking.linking_trackpy(
        first_two_times_df, None, **tracking_params
    )
    for i in range(2, max(all_feats["frame"]) + 1):
        curr_times_df = all_feats[all_feats["frame"] == i]
        curr_tracking_append = tobac.tracking.append_tracks_trackpy(
            curr_tracking_append, curr_times_df, **tracking_params
        )
    assert tobac.testing.check_tracking_identical(curr_tracking_append, orig_tracking)


def test_trackpy_predict_append():
    """Function to test if append_tracks_trackpy() with method='predict' correctly links two
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
    cell_2["idx"] = 1

    cell_2_expected = copy.deepcopy(cell_2)
    cell_2_expected["cell"] = np.int32(2)

    features = pd.concat([cell_1, cell_2], ignore_index=True, verify_integrity=True)

    tracking_params = {"dt": 1, "dxy": 1, "d_max": 100, "method_linking": "predict"}

    output_correct = tobac.linking_trackpy(features, None, **tracking_params)

    # let's extract the first two times
    first_two_times_df = features[features["frame"] < 2]

    initial_tracking_append = tobac.tracking.linking_trackpy(
        first_two_times_df, None, **tracking_params
    )

    append_all_tracking = tobac.tracking.append_tracks_trackpy(
        initial_tracking_append, features, **tracking_params
    )

    assert tobac.testing.check_tracking_identical(output_correct, append_all_tracking)

    # let's try to append one by one, with the full dataframe
    curr_tracking_append = tobac.tracking.linking_trackpy(
        first_two_times_df, None, **tracking_params
    )
    for i in range(3, max(features["frame"]) + 2):
        curr_times_df = features[features["frame"] < i]
        curr_tracking_append = tobac.tracking.append_tracks_trackpy(
            curr_tracking_append, curr_times_df, **tracking_params
        )
    assert tobac.testing.check_tracking_identical(curr_tracking_append, output_correct)

    # let's try to append one by one, with only individual times
    curr_tracking_append = tobac.tracking.linking_trackpy(
        first_two_times_df, None, **tracking_params
    )
    for i in range(2, max(features["frame"]) + 1):
        curr_times_df = features[features["frame"] == i]
        curr_tracking_append = tobac.tracking.append_tracks_trackpy(
            curr_tracking_append, curr_times_df, **tracking_params
        )
    assert tobac.testing.check_tracking_identical(curr_tracking_append, output_correct)


@pytest.mark.parametrize(
    "features_points, dt, dxy, v_max, memory, time_cell_min",
    [
        (
            (
                ((0, 1, 2, 3, 4, 5), (1, 2, 3, 4, 5, 6)),
                ((5, 4, 3, -1, 1, 0), (6, 5, 4, -1, 2, 1)),
                ((-1, -1, 0, 1, 2), (-1, -1, 4, 5, 6)),
                ((-1, -1, 10, 11, 12), (-1, -1, 14, 15, 16)),
            ),
            60,
            1000,
            30,
            0,
            0,
        ),
        (
            (
                ((0, 1, 2, 3, 4, 5), (1, 2, 3, 4, 5, 6)),
                ((5, 4, 3, -1, 1, 0), (6, 5, 4, -1, 2, 1)),
                ((-1, -1, 0, 1, 2), (-1, -1, 4, 5, 6)),
                ((-1, -1, 10, 11, 12), (-1, -1, 14, 15, 16)),
            ),
            60,
            1000,
            30,
            0,
            60,
        ),
    ],
)
def test_append_tracking_single_track_predict_memory(
    features_points: tuple[tuple[tuple[float]]],
    dt: float,
    dxy: float,
    v_max: float,
    memory: int,
    time_cell_min: float,
):
    """
    Function to test (with a single set of feature points) whether append_tracks_trackpy and
    link_trackpy produce the same result.

    Parameters
    ----------
    features_points
    dt
    dxy
    v_max

    Returns
    -------

    """

    all_features = list()
    for feature_point_values in features_points:
        v_points = None
        if len(feature_point_values) > 2:
            v_points = feature_point_values[2]

        test_feature = tobac.testing.generate_single_feature(
            start_h1=feature_point_values[0],
            start_h2=feature_point_values[1],
            start_v=v_points,
            min_h1=0,
            max_h1=1000,
            min_h2=0,
            max_h2=1000,
            frame_start=0,
            PBC_flag="none",
        )
        all_features.append(test_feature)

    all_feats = tobac.testing.combine_single_features(all_features, 1)

    # shared tracking parameters
    tracking_params = {
        "dt": dt,
        "dxy": dxy,
        "v_max": v_max,
        "memory": memory,
        "time_cell_min": time_cell_min,
        "method_linking": "predict",
    }

    # Standard tracking
    # base tracking - original tracking function
    orig_tracking = tobac.tracking.linking_trackpy(all_feats, None, **tracking_params)

    # tracking with appends
    # let's extract the first two times
    first_two_times_df = all_feats[all_feats["frame"] < 2]
    initial_tracking_append = tobac.tracking.linking_trackpy(
        first_two_times_df, None, **tracking_params
    )

    append_all_tracking = tobac.tracking.append_tracks_trackpy(
        initial_tracking_append, all_feats, **tracking_params
    )

    assert tobac.testing.check_tracking_identical(orig_tracking, append_all_tracking)
    # let's try to append one by one, with the full dataframe
    curr_tracking_append = tobac.tracking.linking_trackpy(
        first_two_times_df, None, **tracking_params
    )
    for i in range(3, max(all_feats["frame"]) + 2):
        curr_times_df = all_feats[all_feats["frame"] < i]
        curr_tracking_append = tobac.tracking.append_tracks_trackpy(
            curr_tracking_append, curr_times_df, **tracking_params
        )
    assert tobac.testing.check_tracking_identical(curr_tracking_append, orig_tracking)
    # let's try to append one by one, with only individual times
    curr_tracking_append = tobac.tracking.linking_trackpy(
        first_two_times_df, None, **tracking_params
    )
    for i in range(2, max(all_feats["frame"]) + 1):
        curr_times_df = all_feats[all_feats["frame"] == i]
        curr_tracking_append = tobac.tracking.append_tracks_trackpy(
            curr_tracking_append, curr_times_df, **tracking_params
        )
    assert tobac.testing.check_tracking_identical(curr_tracking_append, orig_tracking)

    # let's try to append one by one, with only individual times
    curr_tracking_append = tobac.tracking.linking_trackpy(
        first_two_times_df, None, **tracking_params
    )
    for i in range(2, max(all_feats["frame"]) + 1):
        curr_times_df = all_feats[all_feats["frame"] == i]
        curr_tracking_append = tobac.tracking.append_tracks_trackpy(
            curr_tracking_append, curr_times_df, **tracking_params
        )
    assert tobac.testing.check_tracking_identical(curr_tracking_append, orig_tracking)


@pytest.mark.parametrize(
    "seed, hdim1_max, hdim2_max, n_features, n_times",
    [
        (2032, 200, 200, 3, 4),
        (2032, 200, 200, 4, 4),
        (201532, 100, 100, 20, 6),
        (10032, 1000, 1000, 20, 20),
    ],
)
@pytest.mark.parametrize(
    "method_linking, time_cell_min",
    [
        ("random", 1200),
        ("predict", 300),
    ],
)
def test_append_tracks_random(
    seed: int,
    hdim1_max: int,
    hdim2_max: int,
    n_features: int,
    n_times: int,
    method_linking: str,
    time_cell_min: float,
):
    """
    Function to test that append and regular tracking work with
    a set of randomly generated features

    Parameters
    ----------

    Returns
    -------

    """
    rng = np.random.default_rng(seed)
    curr_time = datetime.datetime(2026, 1, 1)
    delta_time = datetime.timedelta(seconds=300)
    all_features = list()

    feature_id = 1
    for time_number in range(n_times):
        hdim_1_vals = rng.integers(0, hdim1_max, size=n_features)
        hdim_2_vals = rng.integers(0, hdim2_max, size=n_features)
        idx = 0
        for i in range(n_features):
            all_features.append(
                {
                    "feature": feature_id,
                    "frame": time_number,
                    "idx": idx,
                    "time": np.datetime64(curr_time),
                    "hdim_1": hdim_1_vals[i],
                    "hdim_2": hdim_2_vals[i],
                    "num": 40,  # dummy: min connected pixels
                    "threshold_value": 50.0,  # dummy: detection threshold
                }
            )
            feature_id += 1
            idx += 1
        curr_time = curr_time + delta_time

    features = pd.DataFrame(all_features)

    # shared tracking parameters
    tracking_params = {
        "dt": 300,
        "dxy": 500,
        "v_max": 30,
        "memory": 0,
        "time_cell_min": time_cell_min,
        "method_linking": method_linking,
        "subnetwork_size": 15,
    }

    # Standard tracking
    # base tracking - original tracking function
    orig_tracking = tobac.tracking.linking_trackpy(features, None, **tracking_params)

    # tracking with appends
    # let's extract the first two times
    first_two_times_df = features[features["frame"] < 2]
    initial_tracking_append = tobac.tracking.linking_trackpy(
        first_two_times_df, None, **tracking_params
    )

    append_all_tracking = tobac.tracking.append_tracks_trackpy(
        initial_tracking_append, features, **tracking_params
    )

    assert tobac.testing.check_tracking_identical(orig_tracking, append_all_tracking)
    # let's try to append one by one, with the full dataframe
    curr_tracking_append = tobac.tracking.linking_trackpy(
        first_two_times_df, None, **tracking_params
    )
    for i in range(3, max(features["frame"]) + 2):
        curr_times_df = features[features["frame"] < i]
        curr_tracking_append = tobac.tracking.append_tracks_trackpy(
            curr_tracking_append, curr_times_df, **tracking_params
        )
    assert tobac.testing.check_tracking_identical(curr_tracking_append, orig_tracking)
    # let's try to append one by one, with only individual times
    curr_tracking_append = tobac.tracking.linking_trackpy(
        first_two_times_df, None, **tracking_params
    )
    for i in range(2, max(features["frame"]) + 1):
        curr_times_df = features[features["frame"] == i]
        curr_tracking_append = tobac.tracking.append_tracks_trackpy(
            curr_tracking_append, curr_times_df, **tracking_params
        )
    assert tobac.testing.check_tracking_identical(curr_tracking_append, orig_tracking)

    # let's try to append one by one, with only individual times
    curr_tracking_append = tobac.tracking.linking_trackpy(
        first_two_times_df, None, **tracking_params
    )
    for i in range(2, max(features["frame"]) + 1):
        curr_times_df = features[features["frame"] == i]
        curr_tracking_append = tobac.tracking.append_tracks_trackpy(
            curr_tracking_append, curr_times_df, **tracking_params
        )
    assert tobac.testing.check_tracking_identical(curr_tracking_append, orig_tracking)


@pytest.mark.parametrize(
    "time_cell_min, should_raise",
    [
        (300, False),
        (600, True),
    ],
)
def test_append_tracks_predict_stubs_error(time_cell_min: float, should_raise: bool):
    """
    Test that appending with predict linking raises an error when stubs
    is greater than 2 frames.
    """
    features = tobac.testing.generate_single_feature(
        start_h1=1,
        start_h2=1,
        min_h1=0,
        max_h1=100,
        min_h2=0,
        max_h2=100,
        frame_start=0,
        num_frames=4,
        spd_h1=1,
        spd_h2=1,
        PBC_flag="none",
    )

    tracking_params = {
        "dt": 300,
        "dxy": 500,
        "v_max": 30,
        "memory": 0,
        "time_cell_min": time_cell_min,
        "method_linking": "predict",
    }

    initial_tracking = tobac.tracking.linking_trackpy(
        features[features["frame"] < 2], None, **tracking_params
    )
    if should_raise:
        with pytest.raises(ValueError):
            tobac.tracking.append_tracks_trackpy(
                initial_tracking, features, **tracking_params
            )
    else:
        tobac.tracking.append_tracks_trackpy(
            initial_tracking, features, **tracking_params
        )

    # with the unfiltered cell numbers saved, appending should always work
    initial_tracking = tobac.tracking.linking_trackpy(
        features[features["frame"] < 2],
        None,
        save_unfiltered_cell=True,
        **tracking_params,
    )
    tobac.tracking.append_tracks_trackpy(initial_tracking, features, **tracking_params)


def _generate_random_features(
    seed: int, hdim1_max: int, hdim2_max: int, n_features: int, n_times: int
) -> pd.DataFrame:
    """Generate a dataframe of randomly placed features for testing."""
    rng = np.random.default_rng(seed)
    curr_time = datetime.datetime(2026, 1, 1)
    delta_time = datetime.timedelta(seconds=300)
    all_features = list()

    feature_id = 1
    for time_number in range(n_times):
        hdim_1_vals = rng.integers(0, hdim1_max, size=n_features)
        hdim_2_vals = rng.integers(0, hdim2_max, size=n_features)
        for i in range(n_features):
            all_features.append(
                {
                    "feature": feature_id,
                    "frame": time_number,
                    "idx": i,
                    "time": np.datetime64(curr_time),
                    "hdim_1": hdim_1_vals[i],
                    "hdim_2": hdim_2_vals[i],
                    "num": 40,
                    "threshold_value": 50.0,
                }
            )
            feature_id += 1
        curr_time = curr_time + delta_time

    return pd.DataFrame(all_features)


@pytest.mark.parametrize("save_unfiltered_cell", [True, False])
def test_linking_trackpy_save_unfiltered_cell(save_unfiltered_cell: bool):
    """
    Test that linking_trackpy saves the unfiltered cell numbers only when requested
    and that they match the filtered cell numbers for non-stub cells.
    """
    features = _generate_random_features(201532, 100, 100, 20, 10)
    tracking_params = {
        "dt": 300,
        "dxy": 500,
        "v_max": 30,
        "time_cell_min": 900,
        "method_linking": "predict",
        "subnetwork_size": 15,
    }
    tracks = tobac.tracking.linking_trackpy(
        features, None, save_unfiltered_cell=save_unfiltered_cell, **tracking_params
    )
    if not save_unfiltered_cell:
        assert "cell_unfiltered" not in tracks
        return

    assert "cell_unfiltered" in tracks
    assert (tracks["cell_unfiltered"] != -1).all()
    kept = tracks["cell"] != -1
    # there should be both stubs and kept cells for this test to be meaningful
    assert kept.any() and (~kept).any()
    assert (tracks.loc[kept, "cell"] == tracks.loc[kept, "cell_unfiltered"]).all()
    # stubs should all be shorter than 4 frames
    assert (tracks.loc[~kept].groupby("cell_unfiltered").size() < 4).all()
    # filtered tracking should be the same as without the unfiltered column
    tracks_no_unfilt = tobac.tracking.linking_trackpy(features, None, **tracking_params)
    assert tobac.testing.check_tracking_identical(tracks, tracks_no_unfilt)


@pytest.mark.parametrize(
    "seed, hdim1_max, hdim2_max, n_features, n_times",
    [
        (2032, 200, 200, 3, 4),
        (201532, 100, 100, 20, 6),
        (201532, 100, 100, 20, 10),
        (10032, 1000, 1000, 20, 20),
    ],
)
@pytest.mark.parametrize(
    "method_linking, time_cell_min",
    [
        ("random", 1200),
        ("predict", 300),
        ("predict", 600),
        ("predict", 1200),
    ],
)
def test_append_tracks_unfiltered_cell(
    seed: int,
    hdim1_max: int,
    hdim2_max: int,
    n_features: int,
    n_times: int,
    method_linking: str,
    time_cell_min: float,
):
    """
    Test that append tracking with unfiltered cell numbers reproduces regular tracking,
    including predictive tracking with stubs > 2, and that the output keeps
    the unfiltered cell numbers.
    """
    features = _generate_random_features(
        seed, hdim1_max, hdim2_max, n_features, n_times
    )

    tracking_params = {
        "dt": 300,
        "dxy": 500,
        "v_max": 30,
        "memory": 0,
        "time_cell_min": time_cell_min,
        "method_linking": method_linking,
        "subnetwork_size": 15,
    }

    orig_tracking = tobac.tracking.linking_trackpy(
        features, None, save_unfiltered_cell=True, **tracking_params
    )

    def check_identical(appended):
        assert "cell_unfiltered" in appended
        assert tobac.testing.check_tracking_identical(orig_tracking, appended)
        assert tobac.testing.check_tracking_identical(
            orig_tracking, appended, cell_column="cell_unfiltered"
        )
        kept = appended["cell"] != -1
        assert (
            appended.loc[kept, "cell"] == appended.loc[kept, "cell_unfiltered"]
        ).all()

    first_two_times_df = features[features["frame"] < 2]
    initial_tracking_append = tobac.tracking.linking_trackpy(
        first_two_times_df, None, save_unfiltered_cell=True, **tracking_params
    )

    # append everything at once
    check_identical(
        tobac.tracking.append_tracks_trackpy(
            initial_tracking_append, features, **tracking_params
        )
    )

    # append one at a time with the full dataframe
    curr_tracking_append = initial_tracking_append
    for i in range(3, max(features["frame"]) + 2):
        curr_tracking_append = tobac.tracking.append_tracks_trackpy(
            curr_tracking_append, features[features["frame"] < i], **tracking_params
        )
    check_identical(curr_tracking_append)

    # append one at a time with only individual times
    curr_tracking_append = initial_tracking_append
    for i in range(2, max(features["frame"]) + 1):
        curr_tracking_append = tobac.tracking.append_tracks_trackpy(
            curr_tracking_append, features[features["frame"] == i], **tracking_params
        )
    check_identical(curr_tracking_append)


def test_append_tracks_no_unfiltered_cell_output():
    """
    Test that append tracking does not add the unfiltered cell column when
    the input tracks do not have it.
    """
    features = _generate_random_features(2032, 200, 200, 3, 4)
    tracking_params = {
        "dt": 300,
        "dxy": 500,
        "v_max": 30,
        "time_cell_min": 300,
        "method_linking": "predict",
    }
    initial_tracking = tobac.tracking.linking_trackpy(
        features[features["frame"] < 2], None, **tracking_params
    )
    appended = tobac.tracking.append_tracks_trackpy(
        initial_tracking, features, **tracking_params
    )
    assert "cell_unfiltered" not in appended


def _add_latlon_to_features(
    features: pd.DataFrame, base_lat: float, base_lon: float, dxy: float
) -> pd.DataFrame:
    """Add latitude/longitude columns to a feature dataframe, treating hdim_1/hdim_2 as
    north/east offsets of dxy meters from a base latitude/longitude."""
    planet_radius = 6378137.0
    features = features.copy()
    features["latitude"] = base_lat + np.rad2deg(
        features["hdim_1"] * dxy / planet_radius
    )
    lon = base_lon + np.rad2deg(features["hdim_2"] * dxy / planet_radius) / np.cos(
        np.deg2rad(base_lat)
    )
    # wrap to [-180, 180)
    features["longitude"] = (lon + 180) % 360 - 180
    return features


@pytest.mark.parametrize(
    "seed, hdim1_max, hdim2_max, n_features, n_times, base_lat, base_lon",
    [
        (2032, 200, 200, 3, 4, 40, -90),
        (201532, 100, 100, 20, 6, -30, 20),
        (201532, 100, 100, 20, 10, 60, 179.8),
    ],
)
@pytest.mark.parametrize(
    "method_linking, time_cell_min",
    [
        ("random", 1200),
        ("predict", 300),
        ("predict", 1200),
    ],
)
@pytest.mark.parametrize(
    "adaptive_step, adaptive_stop_multiplier", [(None, None), (0.9, 0.1)]
)
def test_append_tracks_latlon(
    seed: int,
    hdim1_max: int,
    hdim2_max: int,
    n_features: int,
    n_times: int,
    base_lat: float,
    base_lon: float,
    method_linking: str,
    time_cell_min: float,
    adaptive_step: float,
    adaptive_stop_multiplier: float,
):
    """
    Test that append tracking with use_latlon reproduces linking_trackpy_latlon,
    including across the dateline and with adaptive search.
    """
    features = _add_latlon_to_features(
        _generate_random_features(seed, hdim1_max, hdim2_max, n_features, n_times),
        base_lat,
        base_lon,
        dxy=500,
    )
    tracking_params = {
        "dt": 300,
        "v_max": 30,
        "memory": 0,
        "time_cell_min": time_cell_min,
        "method_linking": method_linking,
        "subnetwork_size": 15,
        "adaptive_step": adaptive_step,
        "adaptive_stop_multiplier": adaptive_stop_multiplier,
    }

    orig_tracking = tobac.tracking.linking_trackpy_latlon(
        features, stubs=None, save_unfiltered_cell=True, **tracking_params
    )
    # lat/lon tracking should not leave behind any temporary columns
    assert set(orig_tracking.columns) == set(features.columns) | {
        "cell",
        "time_cell",
        "cell_unfiltered",
    }

    initial_tracking = tobac.tracking.linking_trackpy_latlon(
        features[features["frame"] < 2],
        stubs=None,
        save_unfiltered_cell=True,
        **tracking_params,
    )

    # append everything at once
    appended = tobac.tracking.append_tracks_trackpy(
        initial_tracking, features, use_latlon=True, **tracking_params
    )
    assert set(appended.columns) == set(orig_tracking.columns)
    assert tobac.testing.check_tracking_identical(orig_tracking, appended)

    # append one time at a time
    curr_tracking_append = initial_tracking
    for i in range(2, max(features["frame"]) + 1):
        curr_tracking_append = tobac.tracking.append_tracks_trackpy(
            curr_tracking_append,
            features[features["frame"] == i],
            use_latlon=True,
            **tracking_params,
        )
    assert tobac.testing.check_tracking_identical(orig_tracking, curr_tracking_append)
    assert tobac.testing.check_tracking_identical(
        orig_tracking, curr_tracking_append, cell_column="cell_unfiltered"
    )


def test_append_tracks_latlon_errors():
    """Test that append tracking with use_latlon raises appropriate errors."""
    features = _add_latlon_to_features(
        _generate_random_features(2032, 200, 200, 3, 4), 40, -90, dxy=500
    )
    initial_tracking = tobac.tracking.linking_trackpy_latlon(
        features[features["frame"] < 2], dt=300, v_max=30
    )

    # need exactly one of v_max and d_max
    with pytest.raises(ValueError):
        tobac.tracking.append_tracks_trackpy(
            initial_tracking, features, dt=300, v_max=30, d_max=30, use_latlon=True
        )
    # PBCs are not used with lat/lon
    with pytest.raises(ValueError):
        tobac.tracking.append_tracks_trackpy(
            initial_tracking,
            features,
            dt=300,
            v_max=30,
            use_latlon=True,
            PBC_flag="hdim_2",
            min_h2=0,
            max_h2=200,
        )
    # d_min is not supported
    with pytest.raises(ValueError):
        tobac.tracking.append_tracks_trackpy(
            initial_tracking, features, dt=300, d_min=30, use_latlon=True
        )
    # new features need lat/lon too
    with pytest.raises(ValueError):
        tobac.tracking.append_tracks_trackpy(
            initial_tracking,
            features.drop(columns=["latitude"]),
            dt=300,
            v_max=30,
            use_latlon=True,
        )
    # adaptive_stop is not supported with lat/lon
    with pytest.raises(ValueError):
        tobac.tracking.append_tracks_trackpy(
            initial_tracking,
            features,
            dt=300,
            v_max=30,
            use_latlon=True,
            adaptive_step=0.9,
            adaptive_stop=0.1,
        )
    # adaptive_step and adaptive_stop_multiplier must both be set
    with pytest.raises(ValueError):
        tobac.tracking.append_tracks_trackpy(
            initial_tracking,
            features,
            dt=300,
            v_max=30,
            use_latlon=True,
            adaptive_step=0.9,
        )
    # adaptive_stop_multiplier must be between 0 and 1
    with pytest.raises(ValueError):
        tobac.tracking.append_tracks_trackpy(
            initial_tracking,
            features,
            dt=300,
            v_max=30,
            use_latlon=True,
            adaptive_step=0.9,
            adaptive_stop_multiplier=1.4,
        )
    # adaptive_stop_multiplier is only for lat/lon
    with pytest.raises(ValueError):
        tobac.tracking.append_tracks_trackpy(
            initial_tracking,
            features,
            dt=300,
            dxy=500,
            v_max=30,
            adaptive_step=0.9,
            adaptive_stop_multiplier=0.1,
        )
    # dxy is required without lat/lon
    with pytest.raises(ValueError):
        tobac.tracking.append_tracks_trackpy(
            initial_tracking, features, dt=300, v_max=30
        )

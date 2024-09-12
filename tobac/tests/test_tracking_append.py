"""
Test for the trackpy tracking functions that append one track to another track
"""

import tobac.testing
import tobac.tracking
import pytest
from pandas.testing import assert_frame_equal


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

    assert_frame_equal(orig_tracking, append_all_tracking)

    # let's try to append one by one, with the full dataframe
    curr_tracking_append = tobac.tracking.linking_trackpy(
        first_two_times_df, None, **tracking_params
    )
    for i in range(3, max(all_feats["frame"]) + 2):
        curr_times_df = all_feats[all_feats["frame"] < i]
        curr_tracking_append = tobac.tracking.append_tracks_trackpy(
            curr_tracking_append, curr_times_df, **tracking_params
        )
    assert_frame_equal(curr_tracking_append, orig_tracking)

    # let's try to append one by one, with only individual times
    curr_tracking_append = tobac.tracking.linking_trackpy(
        first_two_times_df, None, **tracking_params
    )
    for i in range(2, max(all_feats["frame"]) + 1):
        curr_times_df = all_feats[all_feats["frame"] == i]
        curr_tracking_append = tobac.tracking.append_tracks_trackpy(
            curr_tracking_append, curr_times_df, **tracking_params
        )
    assert_frame_equal(curr_tracking_append, orig_tracking)

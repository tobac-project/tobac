"""
Test for the trackpy tracking functions that append one track to another track
"""

import tobac.testing
import tobac.tracking
import pytest


@pytest.mark.parametrize(
    "features_points, dt, dxy, v_max",
    [
        (
            (
                (0, 0),
                (1, 1),
            ),
            60,
            1000,
            30,
        )
    ],
)
def test_append_tracking_simple_tracks(
    features_points: tuple[tuple[float]], dt: float, dxy: float, v_max: float
):
    """
    Function to test (with a simple set of feature points) whether append_tracks_trackpy and
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
    )"""

    pass

"""
Audit of the testing functions that produce our test data.
Who's watching the watchmen, basically.
"""
import pytest
from tobac.testing import (
    generate_single_feature,
    get_single_pbc_coordinate,
)
import tobac.testing as tbtest
from collections import Counter
import pandas as pd
from pandas.testing import assert_frame_equal
import datetime
import numpy as np


def test_make_feature_blob():
    """Tests ```tobac.testing.make_feature_blob```
    Currently runs the following tests:
    Creates a blob in the right location and cuts off without PBCs
    Blob extends off PBCs for all dimensions when appropriate
    """

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
        PBC_flag="none",
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
        PBC_flag="none",
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
        PBC_flag="none",
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
        PBC_flag="none",
    )
    assert np.all(out_blob[4:6, 4:6, 7:10] == 1)
    assert np.all(out_blob[4:6, 4:6, 0:1] == 0)
    # There should be exactly 4 points of value 1.
    assert np.sum(out_blob) == 12 and np.min(out_blob) == 0

    for PBC_condition in ["hdim_1", "hdim_2", "both"]:
        # Now test simple cases with PBCs
        # 2D test
        out_blob = tbtest.make_feature_blob(
            np.zeros((10, 10)),
            h1_loc=5,
            h2_loc=5,
            h1_size=2,
            h2_size=2,
            shape="rectangle",
            amplitude=1,
            PBC_flag=PBC_condition,
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
            PBC_flag=PBC_condition,
        )
        assert np.all(out_blob[4:6, 4:6, 4:6] == 1)
        # There should be exactly 8 points of value 1.
        assert np.sum(out_blob) == 8 and np.min(out_blob) == 0

    # Test that it wraps around on the hdim_1 positive side
    for PBC_condition in ["hdim_2", "both"]:
        out_blob = tbtest.make_feature_blob(
            np.zeros((10, 10)),
            h1_loc=5,
            h2_loc=9,
            h1_size=2,
            h2_size=4,
            shape="rectangle",
            amplitude=1,
            PBC_flag=PBC_condition,
        )
        assert np.all(out_blob[4:6, 7:10] == 1)
        assert np.all(out_blob[4:6, 0:1] == 1)
        # There should be exactly 4 points of value 1.
        assert np.sum(out_blob) == 8 and np.min(out_blob) == 0

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
            PBC_flag=PBC_condition,
        )
        assert np.all(out_blob[4:6, 4:6, 7:10] == 1)
        assert np.all(out_blob[4:6, 4:6, 0:1] == 1)
        # There should be exactly 4 points of value 1.
        assert np.sum(out_blob) == 16 and np.min(out_blob) == 0


def test_get_single_pbc_coordinate():
    """Tests ```tobac.testing.get_single_pbc_coordinate```.
    Currently runs the following tests:
    Point within bounds with all PBC conditions
    Point off bounds on each side
    Invalid point
    """

    # Test points that do not need to be adjusted for all PBC conditions
    for PBC_condition in ["none", "hdim_1", "hdim_2", "both"]:
        assert get_single_pbc_coordinate(0, 10, 0, 10, 3, 3, PBC_condition) == (3, 3)
        assert get_single_pbc_coordinate(0, 10, 0, 10, 0, 0, PBC_condition) == (0, 0)
        assert get_single_pbc_coordinate(0, 10, 0, 10, 9, 9, PBC_condition) == (9, 9)

    # Test points off bounds on each side
    # First points off min/max of hdim_1 for the two that allow it
    for PBC_condition in ["hdim_1", "both"]:
        assert get_single_pbc_coordinate(0, 10, 0, 10, -3, 3, PBC_condition) == (7, 3)
        assert get_single_pbc_coordinate(0, 10, 0, 10, 12, 3, PBC_condition) == (2, 3)
        assert get_single_pbc_coordinate(0, 10, 0, 10, 10, 3, PBC_condition) == (0, 3)
        assert get_single_pbc_coordinate(0, 10, 0, 10, 20, 3, PBC_condition) == (0, 3)
        assert get_single_pbc_coordinate(0, 10, 0, 10, -30, 3, PBC_condition) == (0, 3)

    # Now test points off min/max of hdim_1 for the two that don't allow it (expect raise error)
    for PBC_condition in ["none", "hdim_2"]:
        with pytest.raises(ValueError):
            get_single_pbc_coordinate(0, 10, 0, 10, -3, 3, PBC_condition)
        with pytest.raises(ValueError):
            get_single_pbc_coordinate(0, 10, 0, 10, 12, 3, PBC_condition)
        with pytest.raises(ValueError):
            get_single_pbc_coordinate(0, 10, 0, 10, 10, 3, PBC_condition)

    # Now test points off min/max of hdim_2 for the two that allow it
    for PBC_condition in ["hdim_2", "both"]:
        assert get_single_pbc_coordinate(0, 10, 0, 10, 3, -3, PBC_condition) == (3, 7)
        assert get_single_pbc_coordinate(0, 10, 0, 10, 3, 12, PBC_condition) == (3, 2)
        assert get_single_pbc_coordinate(0, 10, 0, 10, 3, 10, PBC_condition) == (3, 0)

    # Now test hdim_2 min/max for the two that don't allow it
    for PBC_condition in ["none", "hdim_1"]:
        with pytest.raises(ValueError):
            get_single_pbc_coordinate(0, 10, 0, 10, 3, -3, PBC_condition)
        with pytest.raises(ValueError):
            get_single_pbc_coordinate(0, 10, 0, 10, 3, 12, PBC_condition)
        with pytest.raises(ValueError):
            get_single_pbc_coordinate(0, 10, 0, 10, 3, 10, PBC_condition)

    # Now test hdim_1 and hdim_2 min/max for 'both'
    assert get_single_pbc_coordinate(0, 11, 0, 10, -3, -3, "both") == (8, 7)
    assert get_single_pbc_coordinate(0, 10, 0, 10, 12, 12, "both") == (2, 2)

    # Now test hdim_1 and hdim/2 min/max for the three that don't allow it
    for PBC_condition in ["none", "hdim_1", "hdim_2"]:
        with pytest.raises(ValueError):
            get_single_pbc_coordinate(0, 11, 0, 10, -3, -3, PBC_condition)
        with pytest.raises(ValueError):
            get_single_pbc_coordinate(0, 10, 0, 10, 12, 12, PBC_condition)


def test_generate_single_feature():
    """Tests the `generate_single_feature` function.
    Currently runs the following tests:
    A single feature is generated

    """

    # Testing a simple 3D case
    expected_df = pd.DataFrame.from_dict(
        [
            {
                "hdim_1": 1,
                "hdim_2": 1,
                "vdim": 1,
                "frame": 0,
                "feature": 1,
                "idx": 0,
                "time": datetime.datetime(2022, 1, 1, 0, 0),
            }
        ]
    )

    assert_frame_equal(
        generate_single_feature(
            1, 1, start_v=1, frame_start=0, max_h1=1000, max_h2=1000
        ).sort_index(axis=1),
        expected_df.sort_index(axis=1),
    )

    # Testing a simple 2D case
    expected_df = pd.DataFrame.from_dict(
        [
            {
                "hdim_1": 1,
                "hdim_2": 1,
                "frame": 0,
                "feature": 1,
                "idx": 0,
                "time": datetime.datetime(2022, 1, 1, 0, 0),
            }
        ]
    )
    assert_frame_equal(
        generate_single_feature(
            1, 1, frame_start=0, max_h1=1000, max_h2=1000
        ).sort_index(axis=1),
        expected_df.sort_index(axis=1),
    )

    # Testing a simple 2D case with movement
    expected_df = pd.DataFrame.from_dict(
        [
            {
                "hdim_1": 1,
                "hdim_2": 1,
                "frame": 0,
                "feature": 1,
                "idx": 0,
                "time": datetime.datetime(2022, 1, 1, 0, 0),
            },
            {
                "hdim_1": 2,
                "hdim_2": 2,
                "frame": 1,
                "feature": 2,
                "idx": 0,
                "time": datetime.datetime(2022, 1, 1, 0, 5),
            },
            {
                "hdim_1": 3,
                "hdim_2": 3,
                "frame": 2,
                "feature": 3,
                "idx": 0,
                "time": datetime.datetime(2022, 1, 1, 0, 10),
            },
            {
                "hdim_1": 4,
                "hdim_2": 4,
                "frame": 3,
                "feature": 4,
                "idx": 0,
                "time": datetime.datetime(2022, 1, 1, 0, 15),
            },
        ]
    )
    assert_frame_equal(
        generate_single_feature(
            1,
            1,
            frame_start=0,
            num_frames=4,
            spd_h1=1,
            spd_h2=1,
            max_h1=1000,
            max_h2=1000,
        ).sort_index(axis=1),
        expected_df.sort_index(axis=1),
    )

    # Testing a simple 3D case with movement
    expected_df = pd.DataFrame.from_dict(
        [
            {
                "hdim_1": 1,
                "hdim_2": 1,
                "vdim": 1,
                "frame": 0,
                "feature": 1,
                "idx": 0,
                "time": datetime.datetime(2022, 1, 1, 0, 0),
            },
            {
                "hdim_1": 2,
                "hdim_2": 2,
                "vdim": 2,
                "frame": 1,
                "feature": 2,
                "idx": 0,
                "time": datetime.datetime(2022, 1, 1, 0, 5),
            },
            {
                "hdim_1": 3,
                "hdim_2": 3,
                "vdim": 3,
                "frame": 2,
                "feature": 3,
                "idx": 0,
                "time": datetime.datetime(2022, 1, 1, 0, 10),
            },
            {
                "hdim_1": 4,
                "hdim_2": 4,
                "vdim": 4,
                "frame": 3,
                "feature": 4,
                "idx": 0,
                "time": datetime.datetime(2022, 1, 1, 0, 15),
            },
        ]
    )
    assert_frame_equal(
        generate_single_feature(
            1,
            1,
            start_v=1,
            frame_start=0,
            num_frames=4,
            spd_h1=1,
            spd_h2=1,
            spd_v=1,
            max_h1=1000,
            max_h2=1000,
        ).sort_index(axis=1),
        expected_df.sort_index(axis=1),
    )

    # Testing a simple 3D case with movement that passes the hdim_1 boundary
    expected_df = pd.DataFrame.from_dict(
        [
            {
                "hdim_1": 1,
                "hdim_2": 1,
                "vdim": 1,
                "frame": 0,
                "feature": 1,
                "idx": 0,
                "time": datetime.datetime(2022, 1, 1, 0, 0),
            },
            {
                "hdim_1": 5,
                "hdim_2": 2,
                "vdim": 2,
                "frame": 1,
                "feature": 2,
                "idx": 0,
                "time": datetime.datetime(2022, 1, 1, 0, 5),
            },
            {
                "hdim_1": 9,
                "hdim_2": 3,
                "vdim": 3,
                "frame": 2,
                "feature": 3,
                "idx": 0,
                "time": datetime.datetime(2022, 1, 1, 0, 10),
            },
            {
                "hdim_1": 3,
                "hdim_2": 4,
                "vdim": 4,
                "frame": 3,
                "feature": 4,
                "idx": 0,
                "time": datetime.datetime(2022, 1, 1, 0, 15),
            },
        ]
    )
    assert_frame_equal(
        generate_single_feature(
            1,
            1,
            start_v=1,
            min_h1=0,
            max_h1=10,
            min_h2=0,
            max_h2=10,
            frame_start=0,
            num_frames=4,
            spd_h1=4,
            spd_h2=1,
            spd_v=1,
            PBC_flag="hdim_1",
        ).sort_index(axis=1),
        expected_df.sort_index(axis=1),
    )

    # Testing a simple 3D case with movement that passes the hdim_1 boundary
    expected_df = pd.DataFrame.from_dict(
        [
            {
                "hdim_1": 1,
                "hdim_2": 1,
                "vdim": 1,
                "frame": 0,
                "feature": 1,
                "idx": 0,
                "time": datetime.datetime(2022, 1, 1, 0, 0),
            },
            {
                "hdim_1": 5,
                "hdim_2": 2,
                "vdim": 2,
                "frame": 1,
                "feature": 2,
                "idx": 0,
                "time": datetime.datetime(2022, 1, 1, 0, 5),
            },
            {
                "hdim_1": 9,
                "hdim_2": 3,
                "vdim": 3,
                "frame": 2,
                "feature": 3,
                "idx": 0,
                "time": datetime.datetime(2022, 1, 1, 0, 10),
            },
            {
                "hdim_1": 3,
                "hdim_2": 4,
                "vdim": 4,
                "frame": 3,
                "feature": 4,
                "idx": 0,
                "time": datetime.datetime(2022, 1, 1, 0, 15),
            },
        ]
    )
    assert_frame_equal(
        generate_single_feature(
            1,
            1,
            start_v=1,
            min_h1=0,
            max_h1=10,
            min_h2=0,
            max_h2=10,
            frame_start=0,
            num_frames=4,
            spd_h1=4,
            spd_h2=1,
            spd_v=1,
            PBC_flag="hdim_1",
        ).sort_index(axis=1),
        expected_df.sort_index(axis=1),
    )

    # Testing a simple 3D case with movement that passes the hdim_2 boundary
    expected_df = pd.DataFrame.from_dict(
        [
            {
                "hdim_1": 1,
                "hdim_2": 1,
                "vdim": 1,
                "frame": 0,
                "feature": 1,
                "idx": 0,
                "time": datetime.datetime(2022, 1, 1, 0, 0),
            },
            {
                "hdim_1": 2,
                "hdim_2": 5,
                "vdim": 2,
                "frame": 1,
                "feature": 2,
                "idx": 0,
                "time": datetime.datetime(2022, 1, 1, 0, 5),
            },
            {
                "hdim_1": 3,
                "hdim_2": 9,
                "vdim": 3,
                "frame": 2,
                "feature": 3,
                "idx": 0,
                "time": datetime.datetime(2022, 1, 1, 0, 10),
            },
            {
                "hdim_1": 4,
                "hdim_2": 3,
                "vdim": 4,
                "frame": 3,
                "feature": 4,
                "idx": 0,
                "time": datetime.datetime(2022, 1, 1, 0, 15),
            },
        ]
    )
    assert_frame_equal(
        generate_single_feature(
            1,
            1,
            start_v=1,
            min_h1=0,
            max_h1=10,
            min_h2=0,
            max_h2=10,
            frame_start=0,
            num_frames=4,
            spd_h1=1,
            spd_h2=4,
            spd_v=1,
            PBC_flag="hdim_2",
        ).sort_index(axis=1),
        expected_df.sort_index(axis=1),
    )

    # Testing a simple 3D case with movement that passes the hdim_1 and hdim_2 boundaries
    expected_df = pd.DataFrame.from_dict(
        [
            {
                "hdim_1": 1,
                "hdim_2": 1,
                "vdim": 1,
                "frame": 0,
                "feature": 1,
                "idx": 0,
                "time": datetime.datetime(2022, 1, 1, 0, 0),
            },
            {
                "hdim_1": 6,
                "hdim_2": 5,
                "vdim": 2,
                "frame": 1,
                "feature": 2,
                "idx": 0,
                "time": datetime.datetime(2022, 1, 1, 0, 5),
            },
            {
                "hdim_1": 1,
                "hdim_2": 9,
                "vdim": 3,
                "frame": 2,
                "feature": 3,
                "idx": 0,
                "time": datetime.datetime(2022, 1, 1, 0, 10),
            },
            {
                "hdim_1": 6,
                "hdim_2": 3,
                "vdim": 4,
                "frame": 3,
                "feature": 4,
                "idx": 0,
                "time": datetime.datetime(2022, 1, 1, 0, 15),
            },
        ]
    )
    assert_frame_equal(
        generate_single_feature(
            1,
            1,
            start_v=1,
            min_h1=0,
            max_h1=10,
            min_h2=0,
            max_h2=10,
            frame_start=0,
            num_frames=4,
            spd_h1=5,
            spd_h2=4,
            spd_v=1,
            PBC_flag="both",
        ).sort_index(axis=1),
        expected_df.sort_index(axis=1),
    )


@pytest.mark.parametrize(
    "in_pt,in_sz,axis_size,out_pts",
    [
        (3, 0, (0, 5), (3, 3)),
        (3, 3, (0, 5), (2, 5)),
    ],
)
def test_get_start_end_of_feat_nopbc(in_pt, in_sz, axis_size, out_pts):
    """Tests ```tobac.testing.get_start_end_of_feat```"""
    assert (
        tbtest.get_start_end_of_feat(in_pt, in_sz, axis_size[0], axis_size[1])
        == out_pts
    )


"""
I acknowledge that this is a little confusing for the expected outputs, especially for the 3D.
"""


@pytest.mark.parametrize(
    "min_max_coords, lengths, expected_outs",
    [
        ((0, 3), (4,), [0, 1, 2, 3]),
        (
            (0, 3, 0, 3),
            (4, 4),
            [
                [
                    [
                        0,
                    ]
                    * 4,
                    [1] * 4,
                    [2] * 4,
                    [3] * 4,
                ],
                [[0, 1, 2, 3]] * 4,
            ],
        ),
        (
            (0, 1, 0, 1, 0, 1),
            (2, 2, 2),
            [
                [
                    [[0] * 2] * 2,
                    [[1] * 2] * 2,
                ],
                [[[0, 0], [1, 1]], [[0, 0], [1, 1]]],
                [[[0, 1], [0, 1]], [[0, 1], [0, 1]]],
            ],
        ),
    ],
)
def test_generate_grid_coords(min_max_coords, lengths, expected_outs):
    """Tests ```tobac.testing.generate_grid_coords```
    Parameters
    ----------
    min_max_coords: array-like, either length 2, length 4, or length 6.
        The minimum and maximum values in each dimension as:
        (min_dim1, max_dim1, min_dim2, max_dim2, min_dim3, max_dim3) to use
        all 3 dimensions. You can omit any dimensions that you aren't using.
    lengths: array-like, either length 1, 2, or 3.
        The lengths of values in each dimension. Length must equal 1/2 the length
        of min_max_coords.
    expected_outs: array-like, either 1D, 2D, or 3D
        The expected output
    """

    out_grid = tbtest.generate_grid_coords(min_max_coords, lengths)
    assert np.all(np.isclose(out_grid, np.array(expected_outs)))

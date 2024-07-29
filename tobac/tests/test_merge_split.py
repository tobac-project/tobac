"""Module to test tobac.merge_split
"""

import pandas as pd
import numpy as np
import pytest

import datetime

import tobac.testing as tbtest
import tobac.tracking as tbtrack
import tobac.merge_split as mergesplit


def test_merge_split_MEST():
    """Tests tobac.merge_split.merge_split_cells by
    generating two cells, colliding them into one another,
    and merging them.
    """

    cell_1 = tbtest.generate_single_feature(
        1,
        1,
        min_h1=0,
        max_h1=100,
        min_h2=0,
        max_h2=100,
        frame_start=0,
        num_frames=2,
        spd_h1=20,
        spd_h2=20,
        start_date=datetime.datetime(2022, 1, 1, 0),
    )
    cell_1["feature"] = [1, 3]

    cell_2 = tbtest.generate_single_feature(
        1,
        100,
        min_h1=0,
        max_h1=101,
        min_h2=0,
        max_h2=101,
        frame_start=0,
        num_frames=2,
        spd_h1=20,
        spd_h2=-20,
        start_date=datetime.datetime(2022, 1, 1, 0),
    )
    cell_2["feature"] = [2, 4]
    cell_3 = tbtest.generate_single_feature(
        30,
        50,
        min_h1=0,
        max_h1=100,
        min_h2=0,
        max_h2=100,
        frame_start=2,
        num_frames=2,
        spd_h1=20,
        spd_h2=0,
        start_date=datetime.datetime(2022, 1, 1, 0, 10),
    )
    cell_3["feature"] = [5, 6]
    features = pd.concat([cell_1, cell_2, cell_3])
    output = tbtrack.linking_trackpy(features, None, 1, 1, v_max=40)

    dist_between = np.sqrt(
        np.power(
            output[output["frame"] == 1].iloc[0]["hdim_1"]
            - output[output["frame"] == 1].iloc[1]["hdim_1"],
            2,
        )
        + np.power(
            output[output["frame"] == 1].iloc[0]["hdim_2"]
            - output[output["frame"] == 1].iloc[1]["hdim_2"],
            2,
        )
    )

    # Test a successful merger
    mergesplit_output_merged = mergesplit.merge_split_MEST(
        output, dxy=10, distance=(dist_between + 50) * 10
    )

    # These cells should have merged together.
    assert len(mergesplit_output_merged["track"]) == 1

    # Test an unsuccessful merger.
    mergesplit_output_unmerged = mergesplit.merge_split_MEST(
        output, dxy=10, distance=(dist_between - 50) * 10
    )

    # These cells should NOT have merged together.
    print(mergesplit_output_unmerged["track"])
    assert len(mergesplit_output_unmerged["track"]) == 2


def test_merge_split_MEST_PBC():
    """
    Test PBC handling for merge_split_MEST
    """
    test_features = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "hdim_1": [1, 89, 1, 99],
            "hdim_2": [50, 50, 50, 50],
            "cell": [1, 2, 1, 2],
            "frame": [0, 0, 1, 1],
            "time": [
                datetime.datetime(2000, 1, 1),
                datetime.datetime(2000, 1, 1),
                datetime.datetime(2000, 1, 1, 0, 5),
                datetime.datetime(2000, 1, 1, 0, 5),
            ],
        }
    )
    # Test without PBCs
    mergesplit_output_no_pbc = mergesplit.merge_split_MEST(
        test_features,
        dxy=1,
        distance=25,
    )

    assert len(mergesplit_output_no_pbc["track"]) == 2

    # Test with PBC in hdim_1, cells should merge
    mergesplit_output_hdim_1_pbc = mergesplit.merge_split_MEST(
        test_features,
        dxy=1,
        distance=25,
        PBC_flag="hdim_1",
        min_h1=0,
        max_h1=100,
        min_h2=0,
        max_h2=100,
    )

    assert len(mergesplit_output_hdim_1_pbc["track"]) == 1

    # Test with PBC in hdim_2, cells should not merge
    mergesplit_output_hdim_2_pbc = mergesplit.merge_split_MEST(
        test_features,
        dxy=1,
        distance=25,
        PBC_flag="hdim_2",
        min_h1=0,
        max_h1=100,
        min_h2=0,
        max_h2=100,
    )

    assert len(mergesplit_output_hdim_2_pbc["track"]) == 2


def test_merge_split_MEST_frame_len():
    """
    Test the frame_len parameter of merge_split_MEST
    """
    test_features = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "hdim_1": [50, 50, 50, 50],
            "hdim_2": [50, 50, 50, 50],
            "cell": [1, 1, 2, 2],
            "frame": [0, 1, 3, 4],
            "time": [
                datetime.datetime(2000, 1, 1),
                datetime.datetime(2000, 1, 1),
                datetime.datetime(2000, 1, 1, 0, 5),
                datetime.datetime(2000, 1, 1, 0, 5),
            ],
        }
    )

    # Test with short frame_len, expect no link
    mergesplit_output = mergesplit.merge_split_MEST(
        test_features,
        dxy=1,
        distance=25,
        frame_len=1,
    )
    assert len(mergesplit_output["track"]) == 2

    # Test with longer frame_len, expect link
    mergesplit_output = mergesplit.merge_split_MEST(
        test_features,
        dxy=1,
        distance=25,
        frame_len=2,
    )
    assert len(mergesplit_output["track"]) == 1


def test_merge_split_MEST_no_cell():
    """
    Test merge/split in cases with features with no cell
    """
    test_features = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "hdim_1": [25, 30, 50],
            "hdim_2": [25, 30, 50],
            "cell": [1, -1, 1],
            "frame": [0, 0, 1],
            "time": [
                datetime.datetime(2000, 1, 1),
                datetime.datetime(2000, 1, 1),
                datetime.datetime(2000, 1, 1, 0, 5),
            ],
        }
    )

    mergesplit_output = mergesplit.merge_split_MEST(
        test_features,
        dxy=1,
        distance=25,
    )

    assert len(mergesplit_output["track"]) == 1

    assert mergesplit_output["feature_parent_cell_id"].values[1] == -1


def test_merge_split_MEST_3D():
    """
    Test merge/split support for 3D tracks and dz input
    """

    test_features = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "vdim": [1, 2, 1, 2],
            "hdim_1": [50, 40, 50, 40],
            "hdim_2": [50, 50, 50, 50],
            "cell": [1, 2, 1, 2],
            "frame": [0, 0, 1, 1],
            "time": [
                datetime.datetime(2000, 1, 1),
                datetime.datetime(2000, 1, 1),
                datetime.datetime(2000, 1, 1, 0, 5),
                datetime.datetime(2000, 1, 1, 0, 5),
            ],
        }
    )

    # Test that failing to provide dz causes an error
    with pytest.raises(ValueError):
        mergesplit.merge_split_MEST(
            test_features,
            dxy=1,
            distance=20,
        )

    # Test with dz=10, expect merge
    mergesplit_output_3d_merge = mergesplit.merge_split_MEST(
        test_features,
        dxy=1,
        dz=10,
        distance=20,
    )

    assert len(mergesplit_output_3d_merge["track"]) == 1

    # Test with dz=25, expect no merge
    mergesplit_output_3d_nomerge = mergesplit.merge_split_MEST(
        test_features,
        dxy=1,
        dz=25,
        distance=20,
    )

    assert len(mergesplit_output_3d_nomerge["track"]) == 2

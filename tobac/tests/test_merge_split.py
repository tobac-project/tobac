"""Module to test tobac.merge_split
"""

import pandas as pd
import numpy as np

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
        output, dxy=1, distance=dist_between + 50
    )

    # These cells should have merged together.
    assert len(mergesplit_output_merged["track"]) == 1

    # Test an unsuccessful merger.
    mergesplit_output_unmerged = mergesplit.merge_split_MEST(
        output, dxy=1, distance=dist_between - 50
    )

    # These cells should NOT have merged together.
    print(mergesplit_output_unmerged["track"])
    assert len(mergesplit_output_unmerged["track"]) == 2

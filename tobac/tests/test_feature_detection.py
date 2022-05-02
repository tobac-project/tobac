import tobac.testing
import tobac.testing as tbtest
import tobac.feature_detection as feat_detect
import pytest


def test_feature_detection_multithreshold_timestep():
    """
    Tests ```tobac.feature_detection.feature_detection_multithreshold_timestep
    """
    import numpy as np
    from tobac import testing
    from tobac import feature_detection

    # start by building a simple dataset with a single feature and seeing
    # if we identify it

    test_dset_size = (50, 50)
    test_hdim_1_pt = 20.0
    test_hdim_2_pt = 20.0
    test_hdim_1_sz = 5
    test_hdim_2_sz = 5
    test_amp = 2
    test_threshs = [
        1.5,
    ]
    test_min_num = 2

    test_data = np.zeros(test_dset_size)
    test_data = testing.make_feature_blob(
        test_data,
        test_hdim_1_pt,
        test_hdim_2_pt,
        h1_size=test_hdim_1_sz,
        h2_size=test_hdim_2_sz,
        amplitude=test_amp,
    )
    test_data_iris = testing.make_dataset_from_arr(test_data, data_type="iris")
    fd_output = feature_detection.feature_detection_multithreshold_timestep(
        test_data_iris, 0, threshold=test_threshs, min_num=test_min_num
    )

    # Make sure we have only one feature
    assert len(fd_output.index) == 1
    # Make sure that the location of the feature is correct
    assert fd_output.iloc[0]["hdim_1"] == pytest.approx(test_hdim_1_pt)
    assert fd_output.iloc[0]["hdim_2"] == pytest.approx(test_hdim_2_pt)


@pytest.mark.parametrize(
    "feature_1_loc, feature_2_loc, dxy, dz, min_distance,"
    " add_x_coords, add_y_coords,"
    "add_z_coords, PBC_flag, expect_feature_1, expect_feature_2",
    [
        (
            (0, 0, 0, 4, 1),
            (1, 1, 1, 4, 1),
            1000,
            100,
            1,
            False,
            False,
            False,
            "none",
            True,
            True,
        ),
        (
            (0, 0, 0, 4, 1),
            (1, 1, 1, 3, 1),
            1000,
            100,
            5000,
            False,
            False,
            False,
            "none",
            True,
            False,
        ),
        (
            (0, 0, 0, 4, 2),
            (1, 1, 1, 10, 1),
            1000,
            100,
            5000,
            False,
            False,
            False,
            "none",
            True,
            False,
        ),
    ],
)
def test_filter_min_distance(
    feature_1_loc,
    feature_2_loc,
    dxy,
    dz,
    min_distance,
    add_x_coords,
    add_y_coords,
    add_z_coords,
    PBC_flag,
    expect_feature_1,
    expect_feature_2,
):
    """Tests tobac.feature_detection.filter_min_distance
    Parameters
    ----------
    feature_1_loc: tuple, length of  4 or 5
        Feature 1 location, num, and threshold value (assumes a 100 x 100 x 100 grid).
        Assumes z, y, x, num, threshold_value for 3D where num is the size/ 'num'
        column of the feature and threshold_value is the threshold_value.
        If 2D, assumes y, x, num, threshold_value.
    feature_2_loc: tuple, length of  4 or 5
        Feature 2 location, same format and length as `feature_1_loc`
    dxy: float or None
        Horizontal grid spacing
    dz: float or None
        Vertical grid spacing (constant)
    min_distance: float
        Minimum distance between features (m)
    add_x_coords: bool
        Whether or not to add x coordinates
    add_y_coords: bool
        Whether or not to add y coordinates
    add_z_coords: bool
        Whether or not to add z coordinates
    PBC_flag : str('none', 'hdim_1', 'hdim_2', 'both')
        Sets whether to use periodic boundaries, and if so in which directions.
        'none' means that we do not have periodic boundaries
        'hdim_1' means that we are periodic along hdim1
        'hdim_2' means that we are periodic along hdim2
        'both' means that we are periodic along both horizontal dimensions
    expect_feature_1: bool
        True if we expect feature 1 to remain, false if we expect it gone.
    expect_feature_2: bool
        True if we expect feature 2 to remain, false if we expect it gone.
    """
    import pandas as pd
    import numpy as np

    h1_max = 100
    h2_max = 100
    z_max = 100

    assumed_dxy = 100
    assumed_dz = 100

    x_coord_name = "projection_coord_x"
    y_coord_name = "projection_coord_y"
    z_coord_name = "projection_coord_z"

    is_3D = len(feature_1_loc) == 5
    start_size_loc = 3 if is_3D else 2
    start_h1_loc = 1 if is_3D else 0
    feat_opts_f1 = {
        "start_h1": feature_1_loc[start_h1_loc],
        "start_h2": feature_1_loc[start_h1_loc + 1],
        "max_h1": h1_max,
        "max_h2": h2_max,
        "feature_size": feature_1_loc[start_size_loc],
        "threshold_val": feature_1_loc[start_size_loc + 1],
        "feature_num": 1,
    }

    feat_opts_f2 = {
        "start_h1": feature_2_loc[start_h1_loc],
        "start_h2": feature_2_loc[start_h1_loc + 1],
        "max_h1": h1_max,
        "max_h2": h2_max,
        "feature_size": feature_2_loc[start_size_loc],
        "threshold_val": feature_2_loc[start_size_loc + 1],
        "feature_num": 2,
    }
    if is_3D:
        feat_opts_f1["start_v"] = feature_1_loc[0]
        feat_opts_f2["start_v"] = feature_2_loc[0]

    feat_1_interp = tbtest.generate_single_feature(**feat_opts_f1)
    feat_2_interp = tbtest.generate_single_feature(**feat_opts_f2)

    feat_combined = pd.concat([feat_1_interp, feat_2_interp], ignore_index=True)

    filter_dist_opts = dict()

    if add_x_coords:
        feat_combined[x_coord_name] = feat_combined["hdim_2"] * assumed_dxy
        filter_dist_opts["x_coordinate_name"] = x_coord_name
    if add_y_coords:
        feat_combined[y_coord_name] = feat_combined["hdim_1"] * assumed_dxy
        filter_dist_opts["y_coordinate_name"] = y_coord_name
    if add_z_coords and is_3D:
        feat_combined[z_coord_name] = feat_combined["vdim"] * assumed_dz
        filter_dist_opts["z_coordinate_name"] = z_coord_name

    filter_dist_opts = {
        "features": feat_combined,
        "dxy": dxy,
        "dz": dz,
        "min_distance": min_distance,
        "PBC_flag": PBC_flag,
    }

    out_feats = feat_detect.filter_min_distance(**filter_dist_opts)

    assert expect_feature_1 == (np.sum(out_feats["feature"] == 1) == 1)
    assert expect_feature_2 == (np.sum(out_feats["feature"] == 2) == 1)


@pytest.mark.parametrize(
    "test_dset_size, vertical_axis_num, "
    "vertical_coord_name,"
    " vertical_coord_opt, expected_raise",
    [
        ((1, 20, 30, 40), 1, "altitude", "auto", False),
        ((1, 20, 30, 40), 2, "altitude", "auto", False),
        ((1, 20, 30, 40), 3, "altitude", "auto", False),
        ((1, 20, 30, 40), 1, "air_pressure", "air_pressure", False),
        ((1, 20, 30, 40), 1, "air_pressure", "auto", True),
        ((1, 20, 30, 40), 1, "model_level_number", "auto", False),
        ((1, 20, 30, 40), 1, "altitude", "auto", False),
        ((1, 20, 30, 40), 1, "geopotential_height", "auto", False),
    ],
)
def test_feature_detection_multiple_z_coords(
    test_dset_size,
    vertical_axis_num,
    vertical_coord_name,
    vertical_coord_opt,
    expected_raise,
):
    """Tests ```tobac.feature_detection.feature_detection_multithreshold```
    with different axes

    Parameters
    ----------
    test_dset_size: tuple(int, int, int, int)
        Size of the test dataset
    vertical_axis_num: int (0-2, inclusive)
        Which axis in test_dset_size is the vertical axis
    vertical_coord_name: str
        Name of the vertical coordinate.
    vertical_coord_opt: str
        What to pass in as the vertical coordinate option to segmentation_timestep
    expected_raise: bool
        True if we expect a ValueError to be raised, false otherwise
    """
    import numpy as np

    # First, just check that input and output shapes are the same.
    test_dxy = 1000
    test_vdim_pt_1 = 8
    test_hdim_1_pt_1 = 12
    test_hdim_2_pt_1 = 12
    test_data = np.zeros(test_dset_size)
    test_data[0, 0:5, 0:5, 0:5] = 3
    common_dset_opts = {
        "in_arr": test_data,
        "data_type": "iris",
        "z_dim_name": vertical_coord_name,
    }
    if vertical_axis_num == 1:
        test_data_iris = tbtest.make_dataset_from_arr(
            time_dim_num=0, z_dim_num=1, y_dim_num=2, x_dim_num=3, **common_dset_opts
        )
    elif vertical_axis_num == 2:
        test_data_iris = tbtest.make_dataset_from_arr(
            time_dim_num=0, z_dim_num=2, y_dim_num=1, x_dim_num=3, **common_dset_opts
        )
    elif vertical_axis_num == 3:
        test_data_iris = tbtest.make_dataset_from_arr(
            time_dim_num=0, z_dim_num=3, y_dim_num=1, x_dim_num=2, **common_dset_opts
        )

    if not expected_raise:
        out_df = feat_detect.feature_detection_multithreshold(
            field_in=test_data_iris,
            dxy=test_dxy,
            threshold=[
                1.5,
            ],
            vertical_coord=vertical_coord_opt,
        )
        # Check that the vertical coordinate is returned.
        print(out_df.columns)
        assert vertical_coord_name in out_df
    else:
        # Expecting a raise
        with pytest.raises(ValueError):
            out_seg_mask, out_df = feat_detect.feature_detection_multithreshold(
                field_in=test_data_iris,
                dxy=test_dxy,
                threshold=[
                    1.5,
                ],
                vertical_coord=vertical_coord_opt,
            )

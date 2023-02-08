import pytest
import tobac.testing as testing
import tobac.segmentation as seg
import numpy as np


def test_segmentation_timestep_2D_feature_2D_seg():
    """Tests `tobac.segmentation.segmentation_timestep` with a 2D
    input feature and a 2D segmentation array
    """
    # Before we can run segmentation, we must run feature detection.

    # start by building a simple dataset with a single feature

    test_dset_size = (50, 50)
    test_hdim_1_pt = 20.0
    test_hdim_2_pt = 20.0
    test_hdim_1_sz = 5
    test_hdim_2_sz = 5
    test_dxy = 1000
    hdim_1_start_feat = int(np.ceil(test_hdim_1_pt - test_hdim_1_sz / 2))
    hdim_1_end_feat = int(np.ceil(test_hdim_1_pt + test_hdim_1_sz / 2))
    hdim_2_start_feat = int(np.ceil(test_hdim_2_pt - test_hdim_2_sz / 2))
    hdim_2_end_feat = int(np.ceil(test_hdim_2_pt + test_hdim_2_sz / 2))

    test_amp = 2

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
    # Generate dummy feature dataset
    test_feature_ds = testing.generate_single_feature(
        start_h1=20.0, start_h2=20.0, max_h1=1000, max_h2=1000
    )

    out_seg_mask, out_df = seg.segmentation_timestep(
        field_in=test_data_iris,
        features_in=test_feature_ds,
        dxy=test_dxy,
        threshold=1.5,
    )

    # Make sure that all labeled points are segmented
    assert np.all(
        out_seg_mask.core_data()[
            hdim_1_start_feat:hdim_1_end_feat, hdim_2_start_feat:hdim_2_end_feat
        ]
        == np.ones((test_hdim_1_sz, test_hdim_2_sz))
    )


def test_segmentation_timestep_level():
    """Tests `tobac.segmentation.segmentation_timestep` with a 2D
    input feature and a 3D segmentation array, specifying the `level` parameter.
    """
    # Before we can run segmentation, we must run feature detection.

    # start by building a simple dataset with a single feature

    test_dset_size = (20, 50, 50)
    test_hdim_1_pt = 20.0
    test_hdim_2_pt = 20.0
    test_vdim_pt = 2
    test_hdim_1_sz = 5
    test_hdim_2_sz = 5
    test_vdim_sz = 3
    test_dxy = 1000

    vdim_start_feat = int(np.ceil(test_vdim_pt - test_vdim_sz / 2))
    vdim_end_feat = int(np.ceil(test_vdim_pt + test_vdim_sz / 2))
    hdim_1_start_feat = int(np.ceil(test_hdim_1_pt - test_hdim_1_sz / 2))
    hdim_1_end_feat = int(np.ceil(test_hdim_1_pt + test_hdim_1_sz / 2))
    hdim_2_start_feat = int(np.ceil(test_hdim_2_pt - test_hdim_2_sz / 2))
    hdim_2_end_feat = int(np.ceil(test_hdim_2_pt + test_hdim_2_sz / 2))

    test_amp = 2

    test_data = np.zeros(test_dset_size)
    test_data = testing.make_feature_blob(
        test_data,
        test_hdim_1_pt,
        test_hdim_2_pt,
        test_vdim_pt,
        h1_size=test_hdim_1_sz,
        h2_size=test_hdim_2_sz,
        v_size=test_vdim_sz,
        amplitude=test_amp,
    )

    # Make a second feature, above the first.

    delta_height = 8
    test_data = testing.make_feature_blob(
        test_data,
        test_hdim_1_pt,
        test_hdim_2_pt,
        test_vdim_pt + delta_height,
        h1_size=test_hdim_1_sz,
        h2_size=test_hdim_2_sz,
        v_size=test_vdim_sz,
        amplitude=test_amp,
    )

    test_data_iris = testing.make_dataset_from_arr(
        test_data, data_type="iris", z_dim_num=0, y_dim_num=1, x_dim_num=2
    )
    # Generate dummy feature dataset
    test_feature_ds = testing.generate_single_feature(
        start_h1=20.0, start_h2=20.0, max_h1=1000, max_h2=1000
    )

    out_seg_mask, out_df = seg.segmentation_timestep(
        field_in=test_data_iris,
        features_in=test_feature_ds,
        dxy=test_dxy,
        threshold=1.5,
        seed_3D_flag="column",
    )
    out_seg_mask_arr = out_seg_mask.core_data()
    # Make sure that all labeled points are segmented, before setting specific levels
    assert np.all(
        out_seg_mask_arr[
            vdim_start_feat:vdim_end_feat,
            hdim_1_start_feat:hdim_1_end_feat,
            hdim_2_start_feat:hdim_2_end_feat,
        ]
        == np.ones((test_vdim_sz, test_hdim_1_sz, test_hdim_2_sz))
    )
    assert np.all(
        out_seg_mask_arr[
            vdim_start_feat + delta_height : vdim_end_feat + delta_height,
            hdim_1_start_feat:hdim_1_end_feat,
            hdim_2_start_feat:hdim_2_end_feat,
        ]
        == np.ones((test_vdim_sz, test_hdim_1_sz, test_hdim_2_sz))
    )

    # now set specific levels
    out_seg_mask, out_df = seg.segmentation_timestep(
        field_in=test_data_iris,
        features_in=test_feature_ds,
        dxy=test_dxy,
        level=slice(vdim_start_feat, vdim_end_feat),
        threshold=1.5,
        seed_3D_flag="column",
    )
    out_seg_mask_arr = out_seg_mask.core_data()
    # Make sure that all labeled points are segmented, before setting specific levels
    assert np.all(
        out_seg_mask_arr[
            vdim_start_feat:vdim_end_feat,
            hdim_1_start_feat:hdim_1_end_feat,
            hdim_2_start_feat:hdim_2_end_feat,
        ]
        == np.ones((test_vdim_sz, test_hdim_1_sz, test_hdim_2_sz))
    )
    assert np.all(
        out_seg_mask_arr[
            vdim_start_feat + delta_height : vdim_end_feat + delta_height,
            hdim_1_start_feat:hdim_1_end_feat,
            hdim_2_start_feat:hdim_2_end_feat,
        ]
        == np.zeros((test_vdim_sz, test_hdim_1_sz, test_hdim_2_sz))
    )


@pytest.mark.parametrize(
    "blob_size, shift_pts, seed_3D_size" ", expected_both_segmented",
    [
        ((3, 3, 3), (0, 0, 4), 3, False),
        ((3, 3, 3), (0, 0, 4), 5, False),
        ((3, 3, 3), (0, 0, 4), 7, True),
    ],
)
def test_segmentation_timestep_3d_seed_box_nopbcs(
    blob_size, shift_pts, seed_3D_size, expected_both_segmented
):
    """Tests ```tobac.segmentation.segmentation_timestep```
    to make sure that the 3D seed box works.
    Parameters
    ----------
    blob_size: tuple(int, int, int)
        Size of the initial blob to add to the domain in (z, y, x) space.
        We strongly recommend that these be *odd* numbers.
    shift_pts: tuple(int, int, int)
        Number of points *relative to the center* to shift the blob in
        (z, y, x) space.
    seed_3D_size: int or tuple
        Seed size to pass to tobac
    expected_both_segmented: bool
        True if we expect both features to be segmented, false
        if we don't expect them both to be segmented

    """

    import numpy as np

    """
    The best way to do this I think is to create two blobs near (but not touching)
    each other, varying the seed_3D_size so that they are either segmented together
    or not segmented together. 
    """
    test_dset_size = (20, 50, 50)
    test_hdim_1_pt_1 = 20.0
    test_hdim_2_pt_1 = 20.0
    test_vdim_pt_1 = 8
    test_dxy = 1000
    test_amp = 2

    test_data = np.zeros(test_dset_size)
    test_data = testing.make_feature_blob(
        test_data,
        test_hdim_1_pt_1,
        test_hdim_2_pt_1,
        test_vdim_pt_1,
        h1_size=blob_size[1],
        h2_size=blob_size[2],
        v_size=blob_size[0],
        amplitude=test_amp,
    )

    # Make a second feature
    test_data = testing.make_feature_blob(
        test_data,
        test_hdim_1_pt_1 + shift_pts[1],
        test_hdim_2_pt_1 + shift_pts[2],
        test_vdim_pt_1 + shift_pts[0],
        h1_size=blob_size[1],
        h2_size=blob_size[2],
        v_size=blob_size[0],
        amplitude=test_amp,
    )

    test_data_iris = testing.make_dataset_from_arr(
        test_data, data_type="iris", z_dim_num=0, y_dim_num=1, x_dim_num=2
    )
    # Generate dummy feature dataset only on the first feature.
    test_feature_ds = testing.generate_single_feature(
        start_v=test_vdim_pt_1,
        start_h1=test_hdim_1_pt_1,
        start_h2=test_hdim_2_pt_1,
        max_h1=1000,
        max_h2=1000,
    )

    out_seg_mask, out_df = seg.segmentation_timestep(
        field_in=test_data_iris,
        features_in=test_feature_ds,
        dxy=test_dxy,
        threshold=1.5,
        seed_3D_flag="box",
        seed_3D_size=seed_3D_size,
    )

    second_point_seg = out_seg_mask.core_data()[
        int(test_vdim_pt_1 + shift_pts[0]),
        int(test_hdim_1_pt_1 + shift_pts[1]),
        int(test_hdim_2_pt_1 + shift_pts[2]),
    ]
    # We really only need to check the center point here for this test.
    seg_point_overlaps = second_point_seg == 1
    assert seg_point_overlaps == expected_both_segmented


@pytest.mark.parametrize(
    "test_dset_size, vertical_axis_num, "
    "vertical_coord_name,"
    " vertical_coord_opt, expected_raise",
    [
        ((20, 30, 40), 0, "altitude", "auto", False),
        ((20, 30, 40), 1, "altitude", "auto", False),
        ((20, 30, 40), 2, "altitude", "auto", False),
        ((20, 30, 40), 0, "air_pressure", "air_pressure", False),
        ((20, 30, 40), 0, "air_pressure", "auto", True),
        ((20, 30, 40), 0, "model_level_number", "auto", False),
        ((20, 30, 40), 0, "altitude", "auto", False),
        ((20, 30, 40), 0, "geopotential_height", "auto", False),
    ],
)
def test_different_z_axes(
    test_dset_size,
    vertical_axis_num,
    vertical_coord_name,
    vertical_coord_opt,
    expected_raise,
):
    """Tests ```tobac.segmentation.segmentation_timestep```
    Tests:
    The output is the same no matter what order we have axes in.
    A ValueError is raised if an invalid vertical coordinate is
    passed in

    Parameters
    ----------
    test_dset_size: tuple(int, int, int)
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
    common_dset_opts = {
        "in_arr": test_data,
        "data_type": "iris",
        "z_dim_name": vertical_coord_name,
    }
    if vertical_axis_num == 0:
        test_data_iris = testing.make_dataset_from_arr(
            z_dim_num=0, y_dim_num=1, x_dim_num=2, **common_dset_opts
        )
    elif vertical_axis_num == 1:
        test_data_iris = testing.make_dataset_from_arr(
            z_dim_num=1, y_dim_num=0, x_dim_num=1, **common_dset_opts
        )
    elif vertical_axis_num == 2:
        test_data_iris = testing.make_dataset_from_arr(
            z_dim_num=2, y_dim_num=0, x_dim_num=1, **common_dset_opts
        )

    # Generate dummy feature dataset only on the first feature.
    test_feature_ds = testing.generate_single_feature(
        start_v=test_vdim_pt_1,
        start_h1=test_hdim_1_pt_1,
        start_h2=test_hdim_2_pt_1,
        max_h1=1000,
        max_h2=1000,
    )
    if not expected_raise:
        out_seg_mask, out_df = seg.segmentation_timestep(
            field_in=test_data_iris,
            features_in=test_feature_ds,
            dxy=test_dxy,
            threshold=1.5,
            vertical_coord=vertical_coord_opt,
        )
        # Check that shapes don't change.
        assert test_data.shape == out_seg_mask.core_data().shape

    else:
        # Expecting a raise
        with pytest.raises(ValueError):
            out_seg_mask, out_df = seg.segmentation_timestep(
                field_in=test_data_iris,
                features_in=test_feature_ds,
                dxy=test_dxy,
                threshold=1.5,
            )

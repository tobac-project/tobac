import pytest
import tobac.testing as testing
import tobac.segmentation as seg
import numpy as np
from tobac import segmentation, feature_detection, testing
from tobac.utils import periodic_boundaries as pbc_utils


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

    out_seg_mask, out_df = segmentation.segmentation_timestep(
        field_in=test_data_iris,
        features_in=test_feature_ds,
        dxy=test_dxy,
        threshold=1.5,
        PBC_flag="none",
    )

    # Make sure that all labeled points are segmented
    assert np.all(
        out_seg_mask.core_data()[
            hdim_1_start_feat:hdim_1_end_feat, hdim_2_start_feat:hdim_2_end_feat
        ]
        == np.ones((test_hdim_1_sz, test_hdim_2_sz))
    )

    # Now try PBCs
    # First, something stretching across hdim_1
    test_hdim_1_pt = 0.0
    test_data = np.zeros(test_dset_size)

    # Note that PBC flag here is 'both' as we still want the blob to be on both
    # sides of the boundary to see if we accidentally grab it without PBC
    # segmentation
    test_data = testing.make_feature_blob(
        test_data,
        test_hdim_1_pt,
        test_hdim_2_pt,
        h1_size=test_hdim_1_sz,
        h2_size=test_hdim_2_sz,
        amplitude=test_amp,
        PBC_flag="both",
    )

    test_data_iris = testing.make_dataset_from_arr(test_data, data_type="iris")
    # Generate dummy feature dataset
    test_feature_ds = testing.generate_single_feature(
        start_h1=test_hdim_1_pt, start_h2=test_hdim_2_pt, max_h1=1000, max_h2=1000
    )

    hdim_1_start_feat, hdim_1_end_feat = testing.get_start_end_of_feat(
        test_hdim_1_pt, test_hdim_1_sz, 0, test_dset_size[0], is_pbc=True
    )

    for pbc_option in ["none", "hdim_1", "hdim_2", "both"]:
        out_seg_mask, out_df = seg.segmentation_timestep(
            field_in=test_data_iris,
            features_in=test_feature_ds,
            dxy=test_dxy,
            threshold=test_amp - 0.5,
            PBC_flag=pbc_option,
        )
        # This will automatically give the appropriate box, and it's tested separately.
        segmented_box_expected = pbc_utils.get_pbc_coordinates(
            0,
            test_dset_size[0],
            0,
            test_dset_size[1],
            hdim_1_start_feat,
            hdim_1_end_feat,
            hdim_2_start_feat,
            hdim_2_end_feat,
            PBC_flag=pbc_option,
        )
        # Make sure that all labeled points are segmented
        for seg_box in segmented_box_expected:
            assert np.all(
                out_seg_mask.core_data()[
                    seg_box[0] : seg_box[1], seg_box[2] : seg_box[3]
                ]
                == np.ones((seg_box[1] - seg_box[0], seg_box[3] - seg_box[2]))
            )

        if pbc_option in ["none", "hdim_2"]:
            # there will only be one seg_box
            assert np.sum(
                out_seg_mask.core_data()[out_seg_mask.core_data() == 1]
            ) == np.sum(np.ones((seg_box[1] - seg_box[0], seg_box[3] - seg_box[2])))
        else:
            # We should be capturing the whole feature
            assert np.sum(
                out_seg_mask.core_data()[out_seg_mask.core_data() == 1]
            ) == np.sum(np.ones((test_hdim_1_sz, test_hdim_2_sz)))

    # Same as the above test, but for hdim_2
    # First, try the cases where we shouldn't get the points on the opposite
    # hdim_2 side
    test_hdim_1_pt = 20.0
    test_hdim_2_pt = 0.0
    test_data = np.zeros(test_dset_size)
    test_data = testing.make_feature_blob(
        test_data,
        test_hdim_1_pt,
        test_hdim_2_pt,
        h1_size=test_hdim_1_sz,
        h2_size=test_hdim_2_sz,
        amplitude=test_amp,
        PBC_flag="both",
    )
    test_data_iris = testing.make_dataset_from_arr(test_data, data_type="iris")
    # Generate dummy feature dataset
    test_feature_ds = testing.generate_single_feature(
        start_h1=test_hdim_1_pt, start_h2=test_hdim_2_pt, max_h1=1000, max_h2=1000
    )
    hdim_1_start_feat, hdim_1_end_feat = testing.get_start_end_of_feat(
        test_hdim_1_pt, test_hdim_1_sz, 0, test_dset_size[0], is_pbc=True
    )

    hdim_2_start_feat, hdim_2_end_feat = testing.get_start_end_of_feat(
        test_hdim_2_pt, test_hdim_2_sz, 0, test_dset_size[1], is_pbc=True
    )

    for pbc_option in ["none", "hdim_1", "hdim_2", "both"]:
        out_seg_mask, out_df = seg.segmentation_timestep(
            field_in=test_data_iris,
            features_in=test_feature_ds,
            dxy=test_dxy,
            threshold=test_amp - 0.5,
            PBC_flag=pbc_option,
        )
        # This will automatically give the appropriate box(es), and it's tested separately.
        segmented_box_expected = pbc_utils.get_pbc_coordinates(
            0,
            test_dset_size[0],
            0,
            test_dset_size[1],
            hdim_1_start_feat,
            hdim_1_end_feat,
            hdim_2_start_feat,
            hdim_2_end_feat,
            PBC_flag=pbc_option,
        )
        # Make sure that all labeled points are segmented
        for seg_box in segmented_box_expected:
            assert np.all(
                out_seg_mask.core_data()[
                    seg_box[0] : seg_box[1], seg_box[2] : seg_box[3]
                ]
                == np.ones((seg_box[1] - seg_box[0], seg_box[3] - seg_box[2]))
            )

        if pbc_option in ["none", "hdim_1"]:
            # there will only be one seg_box
            assert np.sum(
                out_seg_mask.core_data()[out_seg_mask.core_data() == 1]
            ) == np.sum(np.ones((seg_box[1] - seg_box[0], seg_box[3] - seg_box[2])))
        else:
            # We should be capturing the whole feature
            assert np.sum(
                out_seg_mask.core_data()[out_seg_mask.core_data() == 1]
            ) == np.sum(np.ones((test_hdim_1_sz, test_hdim_2_sz)))

    # Same as the above test, but for hdim_2
    # First, try the cases where we shouldn't get the points on the opposite
    # both sides (corner point)
    test_hdim_1_pt = 0.0
    test_hdim_2_pt = 0.0
    test_data = np.zeros(test_dset_size)
    test_data = testing.make_feature_blob(
        test_data,
        test_hdim_1_pt,
        test_hdim_2_pt,
        h1_size=test_hdim_1_sz,
        h2_size=test_hdim_2_sz,
        amplitude=test_amp,
        PBC_flag="both",
    )
    test_data_iris = testing.make_dataset_from_arr(test_data, data_type="iris")
    # Generate dummy feature dataset
    test_feature_ds = testing.generate_single_feature(
        start_h1=test_hdim_1_pt, start_h2=test_hdim_2_pt, max_h1=1000, max_h2=1000
    )
    hdim_1_start_feat, hdim_1_end_feat = testing.get_start_end_of_feat(
        test_hdim_1_pt, test_hdim_1_sz, 0, test_dset_size[0], is_pbc=True
    )

    hdim_2_start_feat, hdim_2_end_feat = testing.get_start_end_of_feat(
        test_hdim_2_pt, test_hdim_2_sz, 0, test_dset_size[1], is_pbc=True
    )

    for pbc_option in ["none", "hdim_1", "hdim_2", "both"]:
        out_seg_mask, out_df = seg.segmentation_timestep(
            field_in=test_data_iris,
            features_in=test_feature_ds,
            dxy=test_dxy,
            threshold=test_amp - 0.5,
            PBC_flag=pbc_option,
        )
        # This will automatically give the appropriate box(es), and it's tested separately.
        segmented_box_expected = pbc_utils.get_pbc_coordinates(
            0,
            test_dset_size[0],
            0,
            test_dset_size[1],
            hdim_1_start_feat,
            hdim_1_end_feat,
            hdim_2_start_feat,
            hdim_2_end_feat,
            PBC_flag=pbc_option,
        )
        # Make sure that all labeled points are segmented
        for seg_box in segmented_box_expected:
            print(pbc_option, seg_box)
            # TODO: something is wrong with this case, unclear what.
            assert np.all(
                out_seg_mask.core_data()[
                    seg_box[0] : seg_box[1], seg_box[2] : seg_box[3]
                ]
                == np.ones((seg_box[1] - seg_box[0], seg_box[3] - seg_box[2]))
            )

        # TODO: Make sure for none, hdim_1, hdim_2 that only the appropriate points are segmented


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

    out_seg_mask, out_df = segmentation.segmentation_timestep(
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
    out_seg_mask, out_df = segmentation.segmentation_timestep(
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

    PBC_opt = "none"

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

    out_seg_mask, out_df = segmentation.segmentation_timestep(
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
        ((20, 30, 40), 0, "altitude", None, False),
        ((20, 30, 40), 1, "altitude", None, False),
        ((20, 30, 40), 2, "altitude", None, False),
        ((20, 30, 40), 0, "air_pressure", "air_pressure", False),
        ((20, 30, 40), 0, "air_pressure", None, True),
        ((20, 30, 40), 0, "model_level_number", None, False),
        ((20, 30, 40), 0, "altitude", None, False),
        ((20, 30, 40), 0, "geopotential_height", None, False),
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
        out_seg_mask, out_df = segmentation.segmentation_timestep(
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
            out_seg_mask, out_df = segmentation.segmentation_timestep(
                field_in=test_data_iris,
                features_in=test_feature_ds,
                dxy=test_dxy,
                threshold=1.5,
            )


def test_segmentation_multiple_features():
    """Tests `tobac.segmentation.segmentation_timestep` with a 2D input containing multiple features with different areas.
    Tests specifically whether their area (ncells) is correctly calculate and assigned to the different features.
    """
    test_dset_size = (50, 50)
    test_hdim_1_pt = 20.0
    test_hdim_2_pt = 20.0
    test_hdim_1_sz = 5
    test_hdim_2_sz = 5
    size_feature1 = test_hdim_1_sz * test_hdim_2_sz
    test_amp = 2
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

    # add feature of different size
    test_hdim_1_pt = 40.0
    test_hdim_2_pt = 40.0
    test_hdim_1_sz = 10
    test_hdim_2_sz = 10
    size_feature2 = test_hdim_1_sz * test_hdim_2_sz
    test_amp = 10
    test_dxy = 1

    test_data = testing.make_feature_blob(
        test_data,
        test_hdim_1_pt,
        test_hdim_2_pt,
        h1_size=test_hdim_1_sz,
        h2_size=test_hdim_2_sz,
        amplitude=test_amp,
    )

    test_data_iris = testing.make_dataset_from_arr(test_data, data_type="iris")

    # detect both features
    fd_output = feature_detection.feature_detection_multithreshold_timestep(
        test_data_iris,
        i_time=0,
        dxy=1,
        threshold=[1, 2, 3],
        n_min_threshold=test_min_num,
        target="maximum",
    )

    # add feature IDs to data frame for one time step
    fd_output["feature"] = [1, 2]

    # perform segmentation
    out_seg_mask, out_df = segmentation.segmentation_timestep(
        field_in=test_data_iris, features_in=fd_output, dxy=test_dxy, threshold=1.5
    )
    out_seg_mask_arr = out_seg_mask.core_data()

    # assure that the number of grid cells belonging to each feature (ncells) are consistent with segmentation mask
    assert int(out_df[out_df.feature == 1].ncells.values) == size_feature1
    assert int(out_df[out_df.feature == 2].ncells.values) == size_feature2


# TODO: add more tests to make sure buddy box code is run.
# From this list right now, I'm not sure why buddy box isn't run actually.
@pytest.mark.parametrize(
    "dset_size, blob_1_loc, blob_1_size, blob_2_loc, blob_2_size,"
    "shift_domain, seed_3D_size",
    [
        ((20, 30, 40), (8, 0, 0), (5, 5, 5), (8, 3, 3), (5, 5, 5), (0, -8, -8), None),
        ((20, 30, 40), (8, 0, 0), (5, 5, 5), (8, 3, 3), (5, 5, 5), (0, -8, -8), None),
        ((20, 30, 40), (8, 1, 1), (5, 5, 5), (8, 28, 38), (5, 5, 5), (0, 15, 15), None),
        ((20, 30, 40), (8, 0, 0), (5, 5, 5), (8, 28, 38), (5, 5, 5), (0, -8, -8), None),
        (
            (20, 30, 40),
            (8, 0, 0),
            (5, 5, 5),
            (8, 28, 38),
            (5, 5, 5),
            (0, -8, -8),
            (5, 5, 5),
        ),
    ],
)
# TODO: last test fails
def test_segmentation_timestep_3d_buddy_box(
    dset_size,
    blob_1_loc,
    blob_1_size,
    blob_2_loc,
    blob_2_size,
    shift_domain,
    seed_3D_size,
):
    """Tests ```tobac.segmentation.segmentation_timestep```
    to make sure that the "buddy box" 3D PBC implementation works.
    Basic procedure: build a dataset with two features (preferrably on the corner)
    and then run segmentation, shift the points, and then run segmentation again.
    After shifting back, the results should be identical.
    Note: only tests 'both' PBC condition.
    Parameters
    ----------
    dset_size: tuple(int, int, int)
        Size of the domain (assumes z, hdim_1, hdim_2)
    blob_1_loc: tuple(int, int, int)
        Location of the first blob
    blob_1_size: tuple(int, int, int)
        Size of the first blob. Note: use odd numbers here.
    blob_2_loc: tuple(int, int, int)
        Location of the second blob
    blob_2_size: tuple(int, int, int)
        Size of the second blob. Note: use odd numbers here.
    shift_domain: tuple(int, int, int)
        How many points to shift the domain by.
    seed_3D_size: None, int, or tuple
        Seed size to pass to tobac. If None, passes in a column seed
    """

    import pandas as pd

    """
    The best way to do this I think is to create two blobs near (but not touching)
    each other, varying the seed_3D_size so that they are either segmented together
    or not segmented together. 
    """
    test_dxy = 1000
    test_amp = 2

    test_data = np.zeros(dset_size)
    test_data = testing.make_feature_blob(
        test_data,
        blob_1_loc[1],
        blob_1_loc[2],
        blob_1_loc[0],
        h1_size=blob_1_size[1],
        h2_size=blob_1_size[2],
        v_size=blob_1_size[0],
        amplitude=test_amp,
        PBC_flag="both",
    )

    # Make a second feature
    test_data = testing.make_feature_blob(
        test_data,
        blob_2_loc[1],
        blob_2_loc[2],
        blob_2_loc[0],
        h1_size=blob_2_size[1],
        h2_size=blob_2_size[2],
        v_size=blob_2_size[0],
        amplitude=test_amp,
        PBC_flag="both",
    )

    test_data_iris = testing.make_dataset_from_arr(
        test_data, data_type="iris", z_dim_num=0, y_dim_num=1, x_dim_num=2
    )
    # Generate dummy feature dataset only on the first feature.
    test_feature_ds_1 = testing.generate_single_feature(
        start_v=blob_1_loc[0],
        start_h1=blob_1_loc[1],
        start_h2=blob_1_loc[2],
        max_h1=dset_size[1],
        max_h2=dset_size[2],
        feature_num=1,
        PBC_flag="both",
    )
    test_feature_ds_2 = testing.generate_single_feature(
        start_v=blob_2_loc[0],
        start_h1=blob_2_loc[1],
        start_h2=blob_2_loc[2],
        max_h1=dset_size[1],
        max_h2=dset_size[2],
        feature_num=2,
        PBC_flag="both",
    )
    test_feature_ds = pd.concat([test_feature_ds_1, test_feature_ds_2])

    common_seg_opts = {"dxy": test_dxy, "threshold": 1.5, "PBC_flag": "both"}
    if seed_3D_size is None:
        common_seg_opts["seed_3D_flag"] = "column"
    else:
        common_seg_opts["seed_3D_flag"] = "box"
        common_seg_opts["seed_3D_size"] = seed_3D_size

    out_seg_mask, out_df = seg.segmentation_timestep(
        field_in=test_data_iris, features_in=test_feature_ds, **common_seg_opts
    )

    # Now, shift the data over and re-run segmentation.
    test_data_shifted = np.roll(test_data, shift_domain, axis=(0, 1, 2))
    test_data_iris_shifted = testing.make_dataset_from_arr(
        test_data_shifted, data_type="iris", z_dim_num=0, y_dim_num=1, x_dim_num=2
    )
    test_feature_ds_1 = testing.generate_single_feature(
        start_v=blob_1_loc[0] + shift_domain[0],
        start_h1=blob_1_loc[1] + shift_domain[1],
        start_h2=blob_1_loc[2] + shift_domain[2],
        max_h1=dset_size[1],
        max_h2=dset_size[2],
        feature_num=1,
        PBC_flag="both",
    )
    test_feature_ds_2 = testing.generate_single_feature(
        start_v=blob_2_loc[0] + shift_domain[0],
        start_h1=blob_2_loc[1] + shift_domain[1],
        start_h2=blob_2_loc[2] + shift_domain[2],
        max_h1=dset_size[1],
        max_h2=dset_size[2],
        feature_num=2,
        PBC_flag="both",
    )
    test_feature_ds_shifted = pd.concat([test_feature_ds_1, test_feature_ds_2])
    out_seg_mask_shifted, out_df = seg.segmentation_timestep(
        field_in=test_data_iris_shifted,
        features_in=test_feature_ds_shifted,
        **common_seg_opts
    )

    # Now, shift output back.
    out_seg_reshifted = np.roll(
        out_seg_mask_shifted.core_data(),
        tuple((-x for x in shift_domain)),
        axis=(0, 1, 2),
    )

    assert np.all(out_seg_mask.core_data() == out_seg_reshifted)


@pytest.mark.parametrize(
    "dset_size, feat_1_loc, feat_2_loc," "shift_domain, seed_3D_size",
    [
        ((20, 30, 40), (8, 0, 0), (8, 3, 3), (0, -8, -8), None),
        ((20, 30, 40), (8, 0, 0), (8, 3, 3), (0, -8, -8), None),
        ((20, 30, 40), (8, 1, 1), (8, 28, 38), (0, 15, 15), None),
        ((20, 30, 40), (8, 0, 0), (8, 28, 38), (0, -8, -8), None),
        ((20, 30, 40), (8, 0, 0), (8, 28, 38), (0, -8, -8), (5, 5, 5)),
    ],
)
def test_add_markers_pbcs(
    dset_size, feat_1_loc, feat_2_loc, shift_domain, seed_3D_size
):
    """Tests ```tobac.segmentation.add_markers```
    to make sure that adding markers works and is consistent across PBCs
    Parameters
    ----------
    dset_size: tuple(int, int, int) or (int, int)
        Size of the domain (assumes z, hdim_1, hdim_2) or (hdim_1, hdim_2)
    feat_1_loc: tuple, same length as dset_size
        Location of the first blob
    feat_2_loc: tuple, same length as dset_size
        Location of the second blob
    shift_domain: tuple, same length as dset_size
        How many points to shift the domain by.
    seed_3D_size: None, int, or tuple
        Seed size to pass to tobac. If None, passes in a column seed
    """

    import pandas as pd

    if len(dset_size) == 2:
        is_3D = False
        start_h1_ax = 0
    else:
        is_3D = True
        start_h1_ax = 1

    common_feat_opts = {
        "PBC_flag": "both",
        "max_h1": dset_size[start_h1_ax],
        "max_h2": dset_size[start_h1_ax + 1],
    }

    # Generate dummy feature dataset only on the first feature.
    test_feature_ds_1 = testing.generate_single_feature(
        start_v=feat_1_loc[0],
        start_h1=feat_1_loc[1],
        start_h2=feat_1_loc[2],
        feature_num=1,
        **common_feat_opts
    )
    test_feature_ds_2 = testing.generate_single_feature(
        start_v=feat_2_loc[0],
        start_h1=feat_2_loc[1],
        start_h2=feat_2_loc[2],
        feature_num=2,
        **common_feat_opts
    )
    test_feature_ds = pd.concat([test_feature_ds_1, test_feature_ds_2])

    common_marker_opts = dict()
    common_marker_opts["PBC_flag"] = "both"

    if seed_3D_size is None:
        common_marker_opts["seed_3D_flag"] = "column"
    else:
        common_marker_opts["seed_3D_flag"] = "box"
        common_marker_opts["seed_3D_size"] = seed_3D_size

    marker_arr = seg.add_markers(
        test_feature_ds, np.zeros(dset_size), **common_marker_opts
    )

    # Now, shift the data over and re-run markers.
    test_feature_ds_1 = testing.generate_single_feature(
        start_v=feat_1_loc[0] + shift_domain[0],
        start_h1=feat_1_loc[1] + shift_domain[1],
        start_h2=feat_1_loc[2] + shift_domain[2],
        feature_num=1,
        **common_feat_opts
    )
    test_feature_ds_2 = testing.generate_single_feature(
        start_v=feat_2_loc[0] + shift_domain[0],
        start_h1=feat_2_loc[1] + shift_domain[1],
        start_h2=feat_2_loc[2] + shift_domain[2],
        feature_num=2,
        **common_feat_opts
    )
    test_feature_ds_shifted = pd.concat([test_feature_ds_1, test_feature_ds_2])

    marker_arr_shifted = seg.add_markers(
        test_feature_ds_shifted, np.zeros(dset_size), **common_marker_opts
    )

    # Now, shift output back.
    marker_arr_reshifted = np.roll(
        marker_arr_shifted, tuple((-x for x in shift_domain)), axis=(0, 1, 2)
    )

    assert np.all(marker_arr == marker_arr_reshifted)


@pytest.mark.parametrize(
    "PBC_flag",
    [
        ("none"),
        ("hdim_1"),
        ("hdim_2"),
        ("both"),
    ],
)
def test_empty_segmentation(PBC_flag):
    """Tests ```tobac.segmentation.segmentation_timestep``` with an
    empty/zeroed out array

    """

    h1_size = 100
    h2_size = 100
    v_size = 5
    test_dxy = 1000
    test_feature = testing.generate_single_feature(
        start_v=1,
        start_h1=1,
        start_h2=1,
        max_h1=h1_size,
        max_h2=h2_size,
        feature_num=1,
        PBC_flag=PBC_flag,
    )

    seg_arr = np.zeros((v_size, h1_size, h2_size))
    seg_opts = {
        "dxy": test_dxy,
        "threshold": 1.5,
        "PBC_flag": PBC_flag,
        "segment_number_unassigned": 0,
        "segment_number_below_threshold": -1,
    }
    test_data_iris = testing.make_dataset_from_arr(
        seg_arr, data_type="iris", z_dim_num=0, y_dim_num=1, x_dim_num=2
    )

    out_seg_mask, out_df = seg.segmentation_timestep(
        field_in=test_data_iris, features_in=test_feature, **seg_opts
    )

    assert np.all(out_seg_mask.core_data() == -1)


def test_pbc_snake_segmentation():
    """
    Test that a "snake" feature that crosses PBCs multiple times is recognized as a single feature
    """

    test_arr = np.zeros((50, 50))
    test_arr[::4, 0] = 2
    test_arr[1::4, 0] = 2
    test_arr[3::4, 0] = 2

    test_arr[1::4, 49] = 2
    test_arr[2::4, 49] = 2
    test_arr[3::4, 49] = 2

    test_data_iris = testing.make_dataset_from_arr(test_arr, data_type="iris")
    fd_output = feature_detection.feature_detection_multithreshold_timestep(
        test_data_iris,
        0,
        threshold=[1, 2, 3],
        n_min_threshold=2,
        dxy=1,
        target="maximum",
        PBC_flag="hdim_2",
    )
    fd_output["feature"] = [1]

    seg_output, seg_feats = segmentation.segmentation_timestep(
        test_data_iris,
        fd_output,
        1,
        threshold=1,
        PBC_flag="hdim_2",
        seed_3D_flag="box",
        seed_3D_size=3,
        segment_number_unassigned=0,
        segment_number_below_threshold=-1,
    )

    correct_seg_arr = np.full((50, 50), -1, dtype=np.int32)
    feat_num = 1
    correct_seg_arr[::4, 0] = feat_num
    correct_seg_arr[1::4, 0] = feat_num
    correct_seg_arr[3::4, 0] = feat_num

    correct_seg_arr[1::4, 49] = feat_num
    correct_seg_arr[2::4, 49] = feat_num
    correct_seg_arr[3::4, 49] = feat_num
    np.where(correct_seg_arr == 0)
    seg_out_arr = seg_output.core_data()
    assert np.all(correct_seg_arr == seg_out_arr)

    # test hdim_1
    test_data_iris = testing.make_dataset_from_arr(test_arr.T, data_type="iris")
    fd_output = feature_detection.feature_detection_multithreshold_timestep(
        test_data_iris,
        0,
        threshold=[1, 2, 3],
        n_min_threshold=2,
        dxy=1,
        target="maximum",
        PBC_flag="hdim_1",
    )
    fd_output["feature"] = [1]

    seg_output, seg_feats = segmentation.segmentation_timestep(
        test_data_iris,
        fd_output,
        1,
        threshold=1,
        PBC_flag="hdim_1",
        seed_3D_flag="box",
        seed_3D_size=3,
        segment_number_unassigned=0,
        segment_number_below_threshold=-1,
    )
    seg_out_arr = seg_output.core_data()

    assert np.all(correct_seg_arr.T == seg_out_arr)


def test_max_distance():
    """
    Tests that max_distance works for both PBCs and normal segmentation
    """

    test_arr = np.zeros((50, 50))
    test_arr[:, :] = 10

    fd_output = testing.generate_single_feature(5, 5, max_h1=50, max_h2=50)

    test_data_iris = testing.make_dataset_from_arr(test_arr, data_type="iris")

    seg_output, seg_feats = segmentation.segmentation_timestep(
        test_data_iris,
        fd_output,
        1,
        threshold=1,
        PBC_flag="none",
        max_distance=1,
    )

    correct_seg_arr = np.full((50, 50), 0, dtype=np.int32)
    feat_num: int = 1
    correct_seg_arr[4:7, 5] = feat_num
    correct_seg_arr[5, 4:7] = feat_num

    seg_out_arr = seg_output.core_data()
    assert np.all(correct_seg_arr == seg_out_arr)

    test_arr = np.zeros((50, 50))
    test_arr[:, :20] = 10
    test_arr[:, 45:] = 10

    fd_output = testing.generate_single_feature(0, 0, max_h1=50, max_h2=50)

    test_data_iris = testing.make_dataset_from_arr(test_arr, data_type="iris")

    with pytest.raises(NotImplementedError):
        seg_output, seg_feats = segmentation.segmentation_timestep(
            test_data_iris,
            fd_output,
            1,
            threshold=1,
            PBC_flag="hdim_2",
            max_distance=1,
        )
    """    
    correct_seg_arr = np.full((50, 50), 0, dtype=np.int32)
    feat_num: int = 1
    correct_seg_arr[0:3, 0] = feat_num
    correct_seg_arr[0, 0:3] = feat_num
    correct_seg_arr[:, 20:45] = -1

    seg_out_arr = seg_output.core_data()
    assert np.all(correct_seg_arr == seg_out_arr)
    """


@pytest.mark.parametrize(
    ("below_thresh", "above_thresh", "error"),
    ((0, 0, False), (0, -1, False), (-5, -10, False), (20, 30, True)),
)
def test_seg_alt_unseed_num(below_thresh, above_thresh, error):
    """
    Tests ```segmentation.segmentation_timestep``` to
    make sure that the unseeded regions are labeled appropriately.

    """
    test_arr = np.zeros((50, 50))
    test_arr[0:10, 0:10] = 10
    test_arr[40:50, 40:50] = 10

    fd_output = testing.generate_single_feature(5, 5, max_h1=50, max_h2=50)

    test_data_iris = testing.make_dataset_from_arr(test_arr, data_type="iris")
    if error:
        with pytest.raises(ValueError):
            seg_output, seg_feats = segmentation.segmentation_timestep(
                test_data_iris,
                fd_output,
                1,
                threshold=1,
                PBC_flag="none",
                segment_number_below_threshold=below_thresh,
                segment_number_unassigned=above_thresh,
            )
    else:
        seg_output, seg_feats = segmentation.segmentation_timestep(
            test_data_iris,
            fd_output,
            1,
            threshold=1,
            PBC_flag="none",
            segment_number_below_threshold=below_thresh,
            segment_number_unassigned=above_thresh,
        )

        correct_seg_arr = np.full((50, 50), below_thresh, dtype=np.int32)
        feat_num: int = 1
        correct_seg_arr[0:10, 0:10] = feat_num
        correct_seg_arr[10:40, 10:40] = below_thresh
        correct_seg_arr[40:50, 40:50] = above_thresh

        seg_out_arr = seg_output.core_data()
        assert np.all(correct_seg_arr == seg_out_arr)

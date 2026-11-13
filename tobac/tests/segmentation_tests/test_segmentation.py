import warnings
import pytest
import tobac.segmentation as seg
import tobac.segmentation.watershed_segmentation as watershed_segmentation
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import pytest
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
        out_seg_mask, out_df = segmentation.segmentation_timestep(
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
        out_seg_mask, out_df = segmentation.segmentation_timestep(
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
        out_seg_mask, out_df = segmentation.segmentation_timestep(
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
        statistic={"features_mean": np.mean},
    )

    # add feature IDs to data frame for one time step
    fd_output["feature"] = [1, 2]

    # perform segmentation
    out_seg_mask, out_df = segmentation.segmentation_timestep(
        field_in=test_data_iris,
        features_in=fd_output,
        dxy=test_dxy,
        threshold=1.5,
        statistic={"segments_mean": np.mean},
    )
    out_seg_mask_arr = out_seg_mask.core_data()

    # assure that the number of grid cells belonging to each feature (ncells) are consistent with segmentation mask
    assert int(out_df[out_df.feature == 1].ncells.values[0]) == size_feature1
    assert int(out_df[out_df.feature == 2].ncells.values[0]) == size_feature2
    # assure that bulk statistic columns are created in output (one column added after segmentation)
    assert out_df.columns.size - fd_output.columns.size > 1
    # assure that statistics are calculated everywhere where an area for ncells is found
    assert (
        out_df.ncells[out_df["ncells"] > 0].shape
        == out_df.ncells[out_df["features_mean"] > 0].shape
    )


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

    out_seg_mask, out_df = segmentation.segmentation_timestep(
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
    out_seg_mask_shifted, out_df = segmentation.segmentation_timestep(
        field_in=test_data_iris_shifted,
        features_in=test_feature_ds_shifted,
        **common_seg_opts,
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
        ((30, 40), (0, 0), (3, 3), (-8, -8), None),
        ((30, 40), (0, 0), (3, 3), (-8, -8), None),
        ((30, 40), (1, 1), (28, 38), (15, 15), None),
        ((30, 40), (0, 0), (28, 38), (-8, -8), None),
        ((30, 40), (0, 0), (28, 38), (-8, -8), (5, 5)),
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
    if is_3D:
        test_feature_ds_1 = testing.generate_single_feature(
            start_v=feat_1_loc[0],
            start_h1=feat_1_loc[1],
            start_h2=feat_1_loc[2],
            feature_num=1,
            **common_feat_opts,
        )
        test_feature_ds_2 = testing.generate_single_feature(
            start_v=feat_2_loc[0],
            start_h1=feat_2_loc[1],
            start_h2=feat_2_loc[2],
            feature_num=2,
            **common_feat_opts,
        )
    else:
        test_feature_ds_1 = testing.generate_single_feature(
            start_h1=feat_1_loc[0],
            start_h2=feat_1_loc[1],
            feature_num=1,
            **common_feat_opts,
        )
        test_feature_ds_2 = testing.generate_single_feature(
            start_h1=feat_2_loc[0],
            start_h2=feat_2_loc[1],
            feature_num=2,
            **common_feat_opts,
        )
    test_feature_ds = pd.concat([test_feature_ds_1, test_feature_ds_2])

    common_marker_opts = dict()
    common_marker_opts["PBC_flag"] = "both"

    if seed_3D_size is None:
        common_marker_opts["seed_3D_flag"] = "column"
    else:
        common_marker_opts["seed_3D_flag"] = "box"
        common_marker_opts["seed_3D_size"] = seed_3D_size

    marker_arr = segmentation.add_markers(
        test_feature_ds, np.zeros(dset_size), **common_marker_opts
    )

    # Now, shift the data over and re-run markers.
    if is_3D:
        test_feature_ds_1 = testing.generate_single_feature(
            start_v=feat_1_loc[0] + shift_domain[0],
            start_h1=feat_1_loc[1] + shift_domain[1],
            start_h2=feat_1_loc[2] + shift_domain[2],
            feature_num=1,
            **common_feat_opts,
        )
        test_feature_ds_2 = testing.generate_single_feature(
            start_v=feat_2_loc[0] + shift_domain[0],
            start_h1=feat_2_loc[1] + shift_domain[1],
            start_h2=feat_2_loc[2] + shift_domain[2],
            feature_num=2,
            **common_feat_opts,
        )
    else:
        test_feature_ds_1 = testing.generate_single_feature(
            start_h1=feat_1_loc[0] + shift_domain[0],
            start_h2=feat_1_loc[1] + shift_domain[1],
            feature_num=1,
            **common_feat_opts,
        )
        test_feature_ds_2 = testing.generate_single_feature(
            start_h1=feat_2_loc[0] + shift_domain[0],
            start_h2=feat_2_loc[1] + shift_domain[1],
            feature_num=2,
            **common_feat_opts,
        )

    test_feature_ds_shifted = pd.concat([test_feature_ds_1, test_feature_ds_2])

    marker_arr_shifted = segmentation.add_markers(
        test_feature_ds_shifted, np.zeros(dset_size), **common_marker_opts
    )

    # Now, shift output back.
    if is_3D:
        marker_arr_reshifted = np.roll(
            marker_arr_shifted, tuple((-x for x in shift_domain)), axis=(0, 1, 2)
        )
    else:
        marker_arr_reshifted = np.roll(
            marker_arr_shifted, tuple((-x for x in shift_domain)), axis=(0, 1)
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

    out_seg_mask, out_df = segmentation.segmentation_timestep(
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

    # PBC + max_distance is covered by the test_max_distance_pbc_* tests below.


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


def test_segmentation_return_cells():
    """Test segmentation with the return_cells option"""
    test_data = np.zeros([3, 4, 5], dtype=int)
    test_data[:, 1:3, 1:4] = 2

    test_data = xr.DataArray(
        test_data,
        dims=("time", "y", "x"),
        coords=dict(
            time=pd.date_range(
                datetime(2000, 1, 1, 0), datetime(2000, 1, 1, 2), periods=3
            )
        ),
        attrs=dict(units="feature"),
    )

    test_features = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "frame": [0, 1, 2],
            "time": pd.date_range(
                datetime(2000, 1, 1, 0), datetime(2000, 1, 1, 2), periods=3
            ),
            "hdim_1": [1.5, 1.5, 1.5],
            "hdim_2": [2, 2, 2],
            "cell": [1, 1, 1],
        }
    )

    cell_mask, _ = segmentation.segmentation(
        test_features, test_data, 1, threshold=1, return_cells=True
    )

    assert np.all(cell_mask.values[test_data.values == 2] == 1)
    assert np.all(cell_mask.values[test_data.values == 0] == 0)


def test_segmentation_return_cells_stubs():
    """Test segmentation with the return_cells option and stubs option"""
    test_data = np.zeros([3, 4, 5], dtype=int)
    test_data[:, 1:3, 1:4] = 2

    test_data = xr.DataArray(
        test_data,
        dims=("time", "y", "x"),
        coords=dict(
            time=pd.date_range(
                datetime(2000, 1, 1, 0), datetime(2000, 1, 1, 2), periods=3
            )
        ),
        attrs=dict(units="feature"),
    )

    test_features = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "frame": [0, 1, 2],
            "time": pd.date_range(
                datetime(2000, 1, 1, 0), datetime(2000, 1, 1, 2), periods=3
            ),
            "hdim_1": [1.5, 1.5, 1.5],
            "hdim_2": [2, 2, 2],
            "cell": [1, 1, -1],
        }
    )

    # Without stubs
    cell_mask, _ = segmentation.segmentation(
        test_features, test_data, 1, threshold=1, return_cells=True
    )

    assert np.all(cell_mask[:-1].values[test_data[:-1].values == 2] == 1)
    assert np.all(cell_mask[-1].values[test_data[-1].values == 2] == -1)
    assert np.all(cell_mask.values[test_data.values == 0] == 0)

    # With stubs
    cell_mask, _ = segmentation.segmentation(
        test_features, test_data, 1, threshold=1, return_cells=True, stubs=-1
    )

    assert np.all(cell_mask[:-1].values[test_data[:-1].values == 2] == 1)
    assert np.all(cell_mask[-1].values[test_data[-1].values == 2] == 0)
    assert np.all(cell_mask.values[test_data.values == 0] == 0)


def test_segmentation_return_cells_no_cell_column():
    """Test segmentation with the return_cells raise the correct error if the
    input features has no cell column
    """
    test_data = np.zeros([3, 4, 5], dtype=int)
    test_data[:, 1:3, 1:4] = 2

    test_data = xr.DataArray(
        test_data,
        dims=("time", "y", "x"),
        coords=dict(
            time=pd.date_range(
                datetime(2000, 1, 1, 0), datetime(2000, 1, 1, 2), periods=3
            )
        ),
        attrs=dict(units="feature"),
    )

    test_features = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "frame": [0, 1, 2],
            "time": pd.date_range(
                datetime(2000, 1, 1, 0), datetime(2000, 1, 1, 2), periods=3
            ),
            "hdim_1": [1.5, 1.5, 1.5],
            "hdim_2": [2, 2, 2],
        }
    )

    with pytest.raises(
        ValueError,
        match="`cell` column not found in features input, please perform tracking on this data before performing segmentation with *",
    ):
        cell_mask, _ = segmentation.segmentation(
            test_features, test_data, 1, threshold=1, return_cells=True
        )


def test_segmentation_returns_early_for_empty_features():
    """
    Tests that segmentation exits early and returns an empty segmentation mask and the empty features input when no features are provided.
    """
    from tobac.utils.internal import make_empty_features_dataframe

    field = xr.DataArray(
        np.zeros((2, 10, 12), dtype=float),
        dims=("time", "y", "x"),
        coords={"time": pd.date_range("2000-01-01", periods=2, freq="2min")},
        name="test_field",
    )

    df = make_empty_features_dataframe(is_3D=False)
    seg_mask, features_out = segmentation.segmentation(
        features=df,
        field=field,
        dxy=1000.0,
        threshold=1.0,
    )

    # mask exists and is all zeros
    assert seg_mask.shape == field.shape
    assert np.all(seg_mask.data == 0)

    # features returned unchanged and still empty
    assert features_out is df
    assert features_out.empty

    sef_mask_none, features_out_none = segmentation.segmentation(
        features=None,
        field=field,
        dxy=1000.0,
        threshold=1.0,
    )

    assert np.all(sef_mask_none.data == 0)
    assert features_out_none is None


@pytest.mark.parametrize(
    "seg_func",
    [
        watershed_segmentation.segmentation,
        watershed_segmentation.segmentation_2D,
        watershed_segmentation.segmentation_3D,
    ],
)
@pytest.mark.parametrize(
    "suppress_warnings, expect_warning",
    [
        (False, True),
        (True, False),
        ("default", True),
    ],
)
def test_segmentation_suppress_warnings(seg_func, suppress_warnings, expect_warning):
    """Test that the 1xMxN warning is issued when suppress_warnings=False
    and suppressed when suppress_warnings=True
    """
    test_dset_size = (1, 50, 50)
    test_data = np.zeros(test_dset_size)
    test_data_xarray = testing.make_dataset_from_arr(
        test_data, data_type="xarray", time_dim_num=0, y_dim_num=1, x_dim_num=2
    )
    features = testing.generate_single_feature(
        start_h1=10, start_h2=10, max_h1=50, max_h2=50
    )

    kwargs = {"dxy": 1000.0, "threshold": 1.5}
    if suppress_warnings != "default":
        kwargs["suppress_warnings"] = suppress_warnings

    warning_msg = (
        "As of v1.6.0, segmentation with time length 1 will return time as a coordinate"
    )

    if expect_warning:
        with pytest.warns(UserWarning, match=warning_msg):
            seg_func(features, test_data_xarray, **kwargs)
    else:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            seg_func(features, test_data_xarray, **kwargs)
            matching_warnings = [
                w
                for w in record
                if issubclass(w.category, UserWarning) and warning_msg in str(w.message)
            ]
            assert len(matching_warnings) == 0


def test_add_markers_3d_box_overlap_closest_feature_wins():
    """Regression test: in 3D box seeding, when two seed boxes overlap, each
    seed point must be assigned to the feature that is closest in full 3D
    distance, as documented in the ``add_markers`` docstring ("if two seed
    boxes overlap, the feature that is seeded will be the closer feature").
    """
    dset_size = (10, 20, 20)  # (z, hdim_1, hdim_2)
    seed_3D_size = 9
    features = pd.DataFrame(
        [
            {"feature": 1, "vdim": 4.0, "hdim_1": 10.0, "hdim_2": 10.0},
            {"feature": 2, "vdim": 5.0, "hdim_1": 14.0, "hdim_2": 10.0},
        ]
    )
    marker_arr = segmentation.add_markers(
        features,
        np.zeros(dset_size, dtype=np.int32),
        seed_3D_flag="box",
        seed_3D_size=seed_3D_size,
        PBC_flag="none",
    )

    # Independently compute the expected assignment: every seed-box point goes
    # to the feature with the smallest squared distance in all 3 dimensions.
    offsets = np.arange(-seed_3D_size / 2, seed_3D_size / 2)
    expected = np.zeros(dset_size, dtype=np.int32)
    best_dist = np.full(dset_size, np.inf)
    for _, row in features.iterrows():
        centre = np.array([row["vdim"], row["hdim_1"], row["hdim_2"]])
        for dz in offsets:
            for dh1 in offsets:
                for dh2 in offsets:
                    loc = np.ceil(centre + [dz, dh1, dh2]).astype(int)
                    loc = np.clip(loc, 0, np.array(dset_size) - 1)
                    dist = ((loc - centre) ** 2).sum()
                    if dist < best_dist[tuple(loc)]:
                        best_dist[tuple(loc)] = dist
                        expected[tuple(loc)] = row["feature"]
    assert np.all(marker_arr == expected)

    # Specific point where vertical-only distance picks the wrong feature:
    # (z=4, h1=13, h2=10) is closer to feature 2 in 3D (dist^2 = 2) than to
    # feature 1 (dist^2 = 9), even though it is closer to feature 1 in z alone.
    assert marker_arr[4, 13, 10] == 2


@pytest.mark.parametrize("PBC_flag", ["hdim_2", "both"])
def test_segmentation_timestep_pbc_does_not_unassign_cells(PBC_flag):
    """Regression test: the PBC "buddy box" re-segmentation must not overwrite
    cells that were already assigned to a feature with the unassigned marker (0).
    Uses a 2D field with blobs straddling the periodic boundary and a blob on
    the non-periodic wall, with column seeding.
    """
    ny, nx = 40, 50
    y, x = np.mgrid[0:ny, 0:nx]
    field_vals = np.zeros((ny, nx))
    # blob straddling the hdim_2 boundary
    field_vals += (
        np.exp(-(((y - 20) ** 2 + np.minimum((x - 49) ** 2, (x - 0) ** 2)) / 8.0)) * 1.2
    )
    # blob straddling the hdim_1 boundary
    field_vals += (
        np.exp(-((np.minimum((y - 39) ** 2, (y - 0) ** 2) + (x - 25) ** 2) / 8.0)) * 1.1
    )
    # interior blob
    field_vals += np.exp(-(((y - 10) ** 2 + (x - 12) ** 2) / 8.0))
    field = xr.DataArray(
        field_vals, dims=("y", "x"), coords={"y": np.arange(ny), "x": np.arange(nx)}
    )
    features = pd.DataFrame(
        [
            {"feature": 1, "hdim_1": 20.0, "hdim_2": 49.0},
            {"feature": 2, "hdim_1": 39.0, "hdim_2": 25.0},
            {"feature": 3, "hdim_1": 0.0, "hdim_2": 25.0},
            {"feature": 4, "hdim_1": 39.0, "hdim_2": 49.0},
            {"feature": 5, "hdim_1": 10.2, "hdim_2": 12.4},
        ]
    )
    threshold = 0.3
    seg_mask, seg_feats = segmentation.segmentation_timestep(
        field_in=field,
        features_in=features,
        dxy=1000.0,
        threshold=threshold,
        PBC_flag=PBC_flag,
    )
    mask = np.asarray(seg_mask.values)

    # No above-threshold cell may be left unassigned (mask == 0)
    above_threshold = field_vals > threshold
    assert not np.any(above_threshold & (mask == 0))

    # Golden ncells, identical to upstream RC_v1.6.x
    assert dict(zip(seg_feats["feature"], seg_feats["ncells"])) == {
        1: 44,
        2: 22,
        3: 22,
        4: 0,
        5: 29,
    }
    if PBC_flag == "hdim_2":
        # Cells that must be assigned across/on the boundary
        assert mask[0, 22] == 3
        assert mask[17, 0] == 1
        assert mask[18, 48] == 1


def test_add_markers_column_seed_casts_to_int():
    """Regression test pinning the column-seeding pixel indexing: hdim
    coordinates are offset by 0.5 and cast to int (``int(x + 0.5)``, truncation
    toward zero for negative values), then wrapped via modulo — the behaviour
    of RC_v1.6.x. The vectorised implementation must reproduce this exactly,
    in particular it must NOT use ``np.round`` (ties-to-even).
    """
    ny, nx = 50, 60
    features = pd.DataFrame(
        [
            {"feature": 1, "hdim_1": 10.5, "hdim_2": 12.5},
            {"feature": 2, "hdim_1": 11.5, "hdim_2": 13.5},
            {"feature": 3, "hdim_1": -0.6, "hdim_2": 2.4},
            {"feature": 4, "hdim_1": 2.5, "hdim_2": 7.5},
        ]
    )
    marker_arr = segmentation.add_markers(
        features, np.zeros((ny, nx), dtype=np.int32), seed_3D_flag="column"
    )
    # int(x + 0.5) cast: 10.5->11, 12.5->13, 11.5->12, 13.5->14, 2.5->3, 7.5->8
    # negative truncates toward zero: -0.6 -> int(-0.1) -> 0
    expected = {(11, 13): 1, (12, 14): 2, (0, 2): 3, (3, 8): 4}
    for (iy, ix), feat in expected.items():
        assert marker_arr[iy, ix] == feat
    assert np.count_nonzero(marker_arr) == len(expected)


def test_max_distance_pbc_interior_feature_matches_no_pbc():
    """max_distance filtering with PBCs enabled must give identical results to
    no PBCs when the feature is far from the boundary (no wrap-around involved),
    and must keep exactly the disk of radius ceil(max_distance/dxy) around the
    marker.
    """
    ny = nx = 50
    test_arr = np.full((ny, nx), 10.0)
    fd_output = testing.generate_single_feature(25, 25, max_h1=ny, max_h2=nx)
    test_data_iris = testing.make_dataset_from_arr(test_arr, data_type="iris")

    seg_none, _ = segmentation.segmentation_timestep(
        test_data_iris, fd_output, 1, threshold=1, PBC_flag="none", max_distance=3
    )
    seg_pbc, _ = segmentation.segmentation_timestep(
        test_data_iris, fd_output, 1, threshold=1, PBC_flag="both", max_distance=3
    )
    assert np.all(seg_none.core_data() == seg_pbc.core_data())

    y, x = np.mgrid[0:ny, 0:nx]
    expected = np.where((y - 25) ** 2 + (x - 25) ** 2 <= 9, 1, 0)
    assert np.all(np.asarray(seg_pbc.core_data()) == expected)


@pytest.mark.parametrize(
    "PBC_flag, feat_h1, feat_h2",
    [("hdim_1", 0.0, 25.0), ("hdim_2", 25.0, 49.0), ("both", 0.0, 49.0)],
)
def test_max_distance_pbc_boundary_feature_wraps(PBC_flag, feat_h1, feat_h2):
    """max_distance filtering with PBCs must use the periodic distance: cells
    across the boundary that are within max_distance of the marker must remain
    assigned to the feature. Uses a uniform above-threshold field with a single
    feature on the boundary, so the expected mask is a disk of radius
    ceil(max_distance/dxy) around the marker in periodic distance.
    """
    ny = nx = 50
    test_arr = np.full((ny, nx), 10.0)
    fd_output = testing.generate_single_feature(feat_h1, feat_h2, max_h1=ny, max_h2=nx)
    test_data_iris = testing.make_dataset_from_arr(test_arr, data_type="iris")

    max_distance = 3.0
    seg_output, seg_feats = segmentation.segmentation_timestep(
        test_data_iris,
        fd_output,
        1,
        threshold=1,
        PBC_flag=PBC_flag,
        max_distance=max_distance,
    )
    mask = np.asarray(seg_output.core_data())

    # PBC-aware distance to the (int-cast) marker position
    y, x = np.mgrid[0:ny, 0:nx]
    my, mx = int(feat_h1 + 0.5), int(feat_h2 + 0.5)
    dy = np.abs(y - my)
    dx = np.abs(x - mx)
    if PBC_flag in ("hdim_1", "both"):
        dy = np.minimum(dy, ny - dy)
    if PBC_flag in ("hdim_2", "both"):
        dx = np.minimum(dx, nx - dx)
    dist = np.sqrt(dy**2 + dx**2)
    expected = np.where(dist <= max_distance, 1, 0)
    assert np.all(mask == expected)


def test_max_distance_pbc_blob_across_boundary():
    """max_distance filtering with PBCs on a blob that straddles the boundary:
    only cells within the periodic distance of the marker are kept, on both
    sides of the boundary. Below-threshold and beyond-max-distance cells are 0
    with the default segment numbers.
    """
    ny = nx = 50
    test_arr = np.zeros((ny, nx))
    test_arr[23:28, 47:50] = 10.0
    test_arr[23:28, 0:3] = 10.0
    fd_output = testing.generate_single_feature(25, 49, max_h1=ny, max_h2=nx)
    test_data_iris = testing.make_dataset_from_arr(test_arr, data_type="iris")

    seg_output, seg_feats = segmentation.segmentation_timestep(
        test_data_iris, fd_output, 1, threshold=1, PBC_flag="hdim_2", max_distance=2
    )
    mask = np.asarray(seg_output.core_data())

    y, x = np.mgrid[0:ny, 0:nx]
    dx = np.minimum(np.abs(x - 49), nx - np.abs(x - 49))
    dist = np.sqrt((y - 25) ** 2 + dx**2)
    above = test_arr > 1
    expected = np.where(above & (dist <= 2), 1, 0)
    assert np.all(mask == expected)
    # Cells across the boundary within max_distance must be assigned
    assert mask[25, 0] == 1
    assert mask[24, 0] == 1
    # Cells across the boundary beyond max_distance must not be assigned
    assert mask[25, 3] == 0


def test_max_distance_pbc_3d_column_seed():
    """3D max_distance filtering with PBCs: with column seeding the markers form
    a full column, so the kept region is a cylinder of radius
    ceil(max_distance/dxy) around the column in periodic horizontal distance,
    across the boundary. Feature is on the (49, 49) corner with PBCs in both
    horizontal dimensions.
    """
    nz, ny, nx = 6, 50, 50
    test_arr = np.full((nz, ny, nx), 10.0)
    fd_output = testing.generate_single_feature(49, 49, start_v=3, max_h1=ny, max_h2=nx)
    test_data_iris = testing.make_dataset_from_arr(
        test_arr, data_type="iris", z_dim_num=0, y_dim_num=1, x_dim_num=2
    )

    seg_output, seg_feats = segmentation.segmentation_timestep(
        test_data_iris, fd_output, 1, threshold=1, PBC_flag="both", max_distance=3
    )
    mask = np.asarray(seg_output.core_data())

    y, x = np.mgrid[0:ny, 0:nx]
    dy = np.minimum(np.abs(y - 49), ny - np.abs(y - 49))
    dx = np.minimum(np.abs(x - 49), nx - np.abs(x - 49))
    dist = np.sqrt(dy**2 + dx**2)
    expected = np.where(dist <= 3, 1, 0)[np.newaxis, :, :].repeat(nz, axis=0)
    assert np.all(mask == expected)

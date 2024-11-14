"""Tests for time padding of segmentation
"""
import datetime
import pytest
from typing import Optional
import numpy as np
import tobac.testing as tb_test
import tobac.segmentation.watershed_segmentation as watershed_segmentation
import tobac.feature_detection as feature_detection


@pytest.mark.parametrize(
    "time_pad_setting, time_offset, expect_output,",
    [
        (datetime.timedelta(seconds=1), datetime.timedelta(seconds=0), True),
        (datetime.timedelta(seconds=1), datetime.timedelta(seconds=2), False),
        (datetime.timedelta(seconds=0), datetime.timedelta(seconds=2), False),
        (datetime.timedelta(seconds=3), datetime.timedelta(seconds=2), True),
        (datetime.timedelta(seconds=2), datetime.timedelta(seconds=1), True),
        (datetime.timedelta(seconds=0), datetime.timedelta(seconds=0), True),
        (None, datetime.timedelta(seconds=0), True),
        (None, datetime.timedelta(seconds=1), False),
    ],
)
def test_watershed_segmentation_time_pad(
    time_pad_setting: Optional[datetime.timedelta],
    time_offset: datetime.timedelta,
    expect_output: bool,
):
    """Tests tobac.watershed_segmentation for time padding working correctly."""
    test_dset_size = (50, 50)
    test_hdim_1_pt = 20.0
    test_hdim_2_pt = 20.0
    test_hdim_1_sz = 5
    test_hdim_2_sz = 5
    size_feature1 = test_hdim_1_sz * test_hdim_2_sz
    test_amp = 2
    test_min_num = 2

    test_data = np.zeros(test_dset_size)
    test_data = tb_test.make_feature_blob(
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

    test_data = tb_test.make_feature_blob(
        test_data,
        test_hdim_1_pt,
        test_hdim_2_pt,
        h1_size=test_hdim_1_sz,
        h2_size=test_hdim_2_sz,
        amplitude=test_amp,
    )

    test_data_xarray = tb_test.make_dataset_from_arr(test_data, data_type="xarray")
    test_data_xarray = test_data_xarray.assign_coords(
        time=datetime.datetime(2020, 1, 1)
    )

    test_data_xarray = test_data_xarray.expand_dims("time")
    # detect both features
    fd_output = feature_detection.feature_detection_multithreshold(
        test_data_xarray,
        i_time=0,
        dxy=1,
        threshold=[1, 2, 3],
        n_min_threshold=test_min_num,
        target="maximum",
        statistic={"features_mean": np.mean},
    )

    # add feature IDs to data frame for one time step
    fd_output["feature"] = [1, 2]
    fd_output.loc[:, "time"] += time_offset

    # perform segmentation
    out_seg_mask, out_df = watershed_segmentation.segmentation(
        field=test_data_xarray,
        features=fd_output,
        dxy=test_dxy,
        threshold=1.5,
        time_padding=time_pad_setting,
    )
    out_seg_mask_arr = out_seg_mask
    if expect_output:
        # assure that the number of grid cells belonging to each feature (ncells) are consistent with segmentation mask
        assert np.sum(out_seg_mask_arr == 1) == size_feature1
        assert np.sum(out_seg_mask_arr == 2) == size_feature2
    else:
        assert np.sum(out_seg_mask_arr == 1) == 0
        assert np.sum(out_seg_mask_arr == 2) == 0

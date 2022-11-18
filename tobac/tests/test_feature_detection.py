import tobac.testing as tbtest
import tobac.feature_detection as feat_detect
import pytest
import numpy as np
from pandas.testing import assert_frame_equal


@pytest.mark.parametrize(
    "test_threshs, dxy, wavelength_filtering",
    [([1.5], -1, None), ([1.5], 10000, (100 * 1000, 500 * 1000))],
)
def test_feature_detection_multithreshold_timestep(
    test_threshs, dxy, wavelength_filtering
):
    """
    Tests ```tobac.feature_detection.feature_detection_multithreshold_timestep```
    """

    # start by building a simple dataset with a single feature and seeing
    # if we identify it

    test_dset_size = (50, 50)
    test_hdim_1_pt = 20.0
    test_hdim_2_pt = 20.0
    test_hdim_1_sz = 5
    test_hdim_2_sz = 5
    test_amp = 2
    test_min_num = 2

    test_data = np.zeros(test_dset_size)
    test_data = tbtest.make_feature_blob(
        test_data,
        test_hdim_1_pt,
        test_hdim_2_pt,
        h1_size=test_hdim_1_sz,
        h2_size=test_hdim_2_sz,
        amplitude=test_amp,
    )
    test_data_iris = tbtest.make_dataset_from_arr(test_data, data_type="iris")
    fd_output = feat_detect.feature_detection_multithreshold_timestep(
        test_data_iris,
        0,
        threshold=test_threshs,
        n_min_threshold=test_min_num,
        dxy=dxy,
        wavelength_filtering=wavelength_filtering,
    )

    # Make sure we have only one feature
    assert len(fd_output.index) == 1
    # Make sure that the location of the feature is correct
    assert fd_output.iloc[0]["hdim_1"] == pytest.approx(test_hdim_1_pt)
    assert fd_output.iloc[0]["hdim_2"] == pytest.approx(test_hdim_2_pt)


@pytest.mark.parametrize(
    "test_threshs, min_distance, dxy", [([1, 2, 3], 100000, 10000)]
)
def test_filter_min_distance(test_threshs, min_distance, dxy):
    """
    Tests ```tobac.feature_detection.filter_min_distance```
    """
    # start by building a simple dataset with two features close to each other

    test_dset_size = (50, 50)
    test_hdim_1_pt = 20.0
    test_hdim_2_pt = 20.0
    test_hdim_1_sz = 5
    test_hdim_2_sz = 5
    test_amp = 5
    test_min_num = 2

    test_data = np.zeros(test_dset_size)
    test_data = tbtest.make_feature_blob(
        test_data,
        test_hdim_1_pt,
        test_hdim_2_pt,
        h1_size=test_hdim_1_sz,
        h2_size=test_hdim_2_sz,
        amplitude=test_amp,
    )

    ## add another blob with smaller value
    test_hdim_1_pt2 = 25.0
    test_hdim_2_pt2 = 25.0
    test_hdim_1_sz2 = 2
    test_hdim_2_sz2 = 2
    test_amp2 = 3
    test_data = tbtest.make_feature_blob(
        test_data,
        test_hdim_1_pt2,
        test_hdim_2_pt2,
        h1_size=test_hdim_1_sz2,
        h2_size=test_hdim_2_sz2,
        amplitude=test_amp2,
    )
    test_data_iris = tbtest.make_dataset_from_arr(test_data, data_type="iris")

    # identify these features
    fd_output = feat_detect.feature_detection_multithreshold_timestep(
        test_data_iris,
        0,
        threshold=test_threshs,
        n_min_threshold=test_min_num,
        min_distance=min_distance,
        dxy=dxy,
    )

    # check if it function to filter
    fd_filtered = feat_detect.filter_min_distance(fd_output, dxy, min_distance)

    # Make sure we have only one feature (small feature in minimum distance should be removed )
    assert len(fd_output.index) == 2
    assert len(fd_filtered.index) == 1
    # Make sure that the locations of the features is correct (should correspond to locations of first feature)
    assert fd_filtered.iloc[0]["hdim_1"] == pytest.approx(test_hdim_1_pt)
    assert fd_filtered.iloc[0]["hdim_2"] == pytest.approx(test_hdim_2_pt)


@pytest.mark.parametrize(
    "position_threshold", [("center"), ("extreme"), ("weighted_diff"), ("weighted_abs")]
)
def test_feature_detection_position(position_threshold):
    """
    Tests to make sure that all feature detection position_thresholds work.
    """

    test_dset_size = (50, 50)

    test_data = np.zeros(test_dset_size)

    test_data[0:5, 0:5] = 3
    test_threshs = [
        1.5,
    ]
    test_min_num = 2

    test_data_iris = tbtest.make_dataset_from_arr(test_data, data_type="iris")

    fd_output = feat_detect.feature_detection_multithreshold_timestep(
        test_data_iris,
        0,
        threshold=test_threshs,
        n_min_threshold=test_min_num,
        position_threshold=position_threshold,
    )

    pass


@pytest.mark.parametrize(
    "test_threshs, target",
    [
        (([1, 2, 3], [3, 2, 1], [1, 3, 2]), "maximum"),
        (([1, 2, 3], [3, 2, 1], [1, 3, 2]), "minimum"),
    ],
)
def test_feature_detection_threshold_sort(test_threshs, target):
    """Tests that feature detection is consistent regardless of what order they are in"""
    test_dset_size = (50, 50)
    test_hdim_1_pt = 20.0
    test_hdim_2_pt = 20.0
    test_hdim_1_sz = 5
    test_hdim_2_sz = 5
    test_amp = 2
    test_min_num = 2

    test_data = np.zeros(test_dset_size)
    test_data = tbtest.make_feature_blob(
        test_data,
        test_hdim_1_pt,
        test_hdim_2_pt,
        h1_size=test_hdim_1_sz,
        h2_size=test_hdim_2_sz,
        amplitude=test_amp,
    )
    test_data_iris = tbtest.make_dataset_from_arr(test_data, data_type="iris")
    fd_output_first = feat_detect.feature_detection_multithreshold_timestep(
        test_data_iris,
        0,
        threshold=test_threshs[0],
        n_min_threshold=test_min_num,
        dxy=1,
        target=target,
    )

    for thresh_test in test_threshs[1:]:
        fd_output_test = feat_detect.feature_detection_multithreshold_timestep(
            test_data_iris,
            0,
            threshold=thresh_test,
            n_min_threshold=test_min_num,
            dxy=1,
            target=target,
        )
        assert_frame_equal(fd_output_first, fd_output_test)

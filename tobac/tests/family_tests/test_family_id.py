import tobac
import tobac.testing as tbtest
import tobac.feature_detection as feat_detect
import tobac.segmentation as tb_seg
import pytest
import numpy as np

import tobac.merge_split.families as tb_fam


def test_family_id_from_data_basic():
    """
    Tests that family id
    (tobac.merge_split_families.feature_family_id.identify_feature_families_from_data)
    functions at all. Not comprehensive, but a decent look at it.
    Returns
    -------
    """

    test_dset_size = (100, 100)
    # make a few points that overlap one another so we end up with multiple
    # features that should agglomerate to one family
    test_hdim_1_pts = [20.0, 23, 15, 17]
    test_hdim_2_pts = [20.0, 23, 15, 17]
    test_hdim_1_szs = [5, 5, 5, 5]
    test_hdim_2_szs = [5, 5, 5, 5]
    test_amps = [3, 5, 2, 5]

    test_data = np.zeros(test_dset_size)

    for test_h1, test_h2, test_sz1, test_sz2, test_amp in zip(
        test_hdim_1_pts, test_hdim_2_pts, test_hdim_1_szs, test_hdim_2_szs, test_amps
    ):
        test_data = tbtest.make_feature_blob(
            test_data,
            test_h1,
            test_h2,
            h1_size=test_sz1,
            h2_size=test_sz2,
            amplitude=test_amp,
        )
    test_data = np.expand_dims(test_data, 0)
    test_data_xr = tbtest.make_dataset_from_arr(
        test_data, data_type="xarray", time_dim_num=0
    )

    test_threshs = [1.5, 2.5, 4.5]
    n_min_threshold = 1
    dxy = 100

    fd_output = feat_detect.feature_detection_multithreshold(
        test_data_xr,
        dxy=dxy,
        threshold=test_threshs,
        n_min_threshold=n_min_threshold,
    )

    assert len(fd_output) == 2, f"Expected 2 features, but got {len(fd_output)}"

    # detect families from data

    families_fd, stats_fd = tb_fam.identify_feature_families_from_data(
        fd_output, test_data_xr, threshold=1.5
    )
    assert (
        len(families_fd) == 2
    ), f"families: Expected 2 features, but got {len(fd_output)}"
    assert (
        len(stats_fd) == 1
    ), f"family stats: Expected 1 family, but got {len(fd_output)}"
    assert np.all(families_fd["feature_family_id"].values == [1, 1])


def test_family_id_from_seg_basic():
    """
    Tests that family id
    (tobac.merge_split_families.feature_family_id.identify_feature_families_from_segmentation)
    functions work at all. Not comprehensive, but a decent look at it.
    Returns
    -------
    """

    test_dset_size = (100, 100)
    # make a few points that overlap one another so we end up with multiple
    # features that should agglomerate to one family
    test_hdim_1_pts = [20.0, 23, 15, 17]
    test_hdim_2_pts = [20.0, 23, 15, 17]
    test_hdim_1_szs = [5, 5, 5, 5]
    test_hdim_2_szs = [5, 5, 5, 5]
    test_amps = [3, 5, 2, 5]

    test_data = np.zeros(test_dset_size)

    for test_h1, test_h2, test_sz1, test_sz2, test_amp in zip(
        test_hdim_1_pts, test_hdim_2_pts, test_hdim_1_szs, test_hdim_2_szs, test_amps
    ):
        test_data = tbtest.make_feature_blob(
            test_data,
            test_h1,
            test_h2,
            h1_size=test_sz1,
            h2_size=test_sz2,
            amplitude=test_amp,
        )
    test_data = np.expand_dims(test_data, 0)
    test_data_xr = tbtest.make_dataset_from_arr(
        test_data, data_type="xarray", time_dim_num=0
    )

    test_threshs = [1.5, 2.5, 4.5]
    n_min_threshold = 1
    dxy = 100

    fd_output = feat_detect.feature_detection_multithreshold(
        test_data_xr,
        dxy=dxy,
        threshold=test_threshs,
        n_min_threshold=n_min_threshold,
    )

    assert len(fd_output) == 2, f"Expected 2 features, but got {len(fd_output)}"

    # run segmentation

    seg_grid, seg_feats = tb_seg.segmentation_3D(
        fd_output, test_data_xr, dxy=dxy, threshold=1.5
    )

    # detect families from data

    families_fd, stats_fd = tb_fam.identify_feature_families_from_segmentation(
        seg_feats, seg_grid
    )
    assert (
        len(families_fd) == 2
    ), f"families: Expected 2 features, but got {len(fd_output)}"
    assert (
        len(stats_fd) == 1
    ), f"family stats: Expected 1 family, but got {len(fd_output)}"
    assert np.all(families_fd["feature_family_id"].values == [1, 1])


def test_family_id_data_features_unlinked():
    """
    tests identify_feature_families_from_data
    Ensures that features whose positions fall outside of the feature area
    are still in the output but are unlinked (-1)
    """

    test_dset_size = (100, 100)
    # pretty darn circular feature
    test_hdim_1_pts = [
        60.0,
        58.7,
        55.0,
        50.0,
        45.0,
        41.3,
        40.0,
        41.3,
        45.0,
        50.0,
        55.0,
        58.7,
    ]
    test_hdim_2_pts = [
        50.0,
        55.0,
        58.7,
        60.0,
        58.7,
        55.0,
        50.0,
        45.0,
        41.3,
        40.0,
        41.3,
        45.0,
    ]
    test_hdim_1_szs = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    test_hdim_2_szs = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    test_amps = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

    test_data = np.zeros(test_dset_size)

    for test_h1, test_h2, test_sz1, test_sz2, test_amp in zip(
        test_hdim_1_pts, test_hdim_2_pts, test_hdim_1_szs, test_hdim_2_szs, test_amps
    ):
        test_data = tbtest.make_feature_blob(
            test_data,
            test_h1,
            test_h2,
            h1_size=test_sz1,
            h2_size=test_sz2,
            amplitude=test_amp,
        )
    test_data = np.expand_dims(test_data, 0)
    test_data_xr = tbtest.make_dataset_from_arr(
        test_data, data_type="xarray", time_dim_num=0
    )

    test_threshs = [1.5, 2.5, 4.5]
    n_min_threshold = 1
    dxy = 100

    fd_output = feat_detect.feature_detection_multithreshold(
        test_data_xr,
        dxy=dxy,
        threshold=test_threshs,
        n_min_threshold=n_min_threshold,
    )
    assert len(fd_output) == 1, f"Expected 1 feature, but got {len(fd_output)}"

    # detect families from data

    families_fd, stats_fd = tb_fam.identify_feature_families_from_data(
        fd_output, test_data_xr, threshold=1.5, unlinked_family_id=-1
    )
    assert (
        len(families_fd) == 1
    ), f"families: Expected 1 feature, but got {len(fd_output)}"
    assert (
        len(stats_fd) == 0
    ), f"family stats: Expected 0 families, but got {len(fd_output)}"
    assert np.all(families_fd["feature_family_id"].values == [])

    families_fd, stats_fd = tb_fam.identify_feature_families_from_data(
        fd_output, test_data_xr, threshold=1.5, unlinked_family_id=None
    )
    assert (
        len(families_fd) == 0
    ), f"families: Expected 0 features, but got {len(fd_output)}"
    assert (
        len(stats_fd) == 0
    ), f"family stats: Expected 0 families, but got {len(fd_output)}"

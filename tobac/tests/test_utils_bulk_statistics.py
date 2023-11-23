
import numpy as np
import tobac
import tobac.utils as tb_utils
import tobac.testing as tb_test

def test_bulk_statistics():
    """
    Test to assure that bulk statistics for identified features are computed as expected.

    """

    ### Test 2D data with time dimension
    test_data = tb_test.make_simple_sample_data_2D().core_data()
    common_dset_opts = {
        "in_arr": test_data,
        "data_type": "iris",
    }
    test_data_iris = tb_test.make_dataset_from_arr(
        time_dim_num=0, y_dim_num=1, x_dim_num=2, **common_dset_opts
    )

    # detect features
    threshold = 7
    # test_data_iris = testing.make_dataset_from_arr(test_data, data_type="iris")
    fd_output = tobac.feature_detection.feature_detection_multithreshold(
        test_data_iris,
        dxy=1000,
        threshold=[threshold],
        n_min_threshold=100,
        target="maximum",
    )

    # perform segmentation with bulk statistics
    stats = {
        "segment_max": np.max,
        "segment_min": min,
        "percentiles": (np.percentile, {"q": 95}),
    }
    out_seg_mask, out_df = tobac.segmentation.segmentation_2D(
        fd_output, test_data_iris, dxy=1000, threshold=threshold, statistics=stats
    )

    #### checks

    #  assure that bulk statistics in postprocessing give same result
    out_segmentation = tb_utils.get_statistics_from_mask(
        out_seg_mask, test_data_iris, features=out_df, statistic=stats
    )
    assert out_segmentation.equals(out_df)

    # assure that column names in new dataframe correspond to keys in statistics dictionary
    for key in stats.keys():
        assert key in out_df.columns

    # assure that statistics bring expected result
    for frame in out_df.frame.values:
        assert out_df[out_df.frame == frame].segment_max.values[0] == np.max(
            test_data[frame]
        )

    ### Test the same with 3D data
    test_data_iris = tb_test.make_sample_data_3D_3blobs()

    # detect features in test dataset
    fd_output = tobac.feature_detection.feature_detection_multithreshold(
        test_data_iris,
        dxy=1000,
        threshold=[threshold],
        n_min_threshold=100,
        target="maximum",
    )

    # perform segmentation with bulk statistics
    stats = {
        "segment_max": np.max,
        "segment_min": min,
        "percentiles": (np.percentile, {"q": 95}),
    }
    out_seg_mask, out_df = tobac.segmentation.segmentation_3D(
        fd_output, test_data_iris, dxy=1000, threshold=threshold, statistics=stats
    )

    ##### checks #####

    #  assure that bulk statistics in postprocessing give same result
    out_segmentation = tb_utils.get_statistics_from_mask(
        out_seg_mask, test_data_iris, features=out_df, statistic=stats
    )
    assert out_segmentation.equals(out_df)

    # assure that column names in new dataframe correspond to keys in statistics dictionary
    for key in stats.keys():
        assert key in out_df.columns

    # assure that statistics bring expected result
    for frame in out_df.frame.values:
        assert out_df[out_df.frame == frame].segment_max.values[0] == np.max(
            test_data_iris.data[frame]
        )

from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
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
        fd_output, test_data_iris, dxy=1000, threshold=threshold, statistic=stats
    )

    #### checks

    #  assure that bulk statistics in postprocessing give same result
    out_segmentation = tb_utils.get_statistics_from_mask(
        out_df, out_seg_mask, test_data_iris, statistic=stats
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
        fd_output, test_data_iris, dxy=1000, threshold=threshold, statistic=stats
    )

    ##### checks #####

    #  assure that bulk statistics in postprocessing give same result
    out_segmentation = tb_utils.get_statistics_from_mask(
        out_df, out_seg_mask, test_data_iris, statistic=stats
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


def test_bulk_statistics_multiple_fields():
    """
    Test that multiple field input to bulk_statistics works as intended
    """

    test_labels = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 2, 0],
                [0, 1, 0, 2, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 3, 0, 0, 0],
                [0, 3, 0, 4, 0],
                [0, 3, 0, 4, 0],
                [0, 0, 0, 0, 0],
            ],
        ],
        dtype=int,
    )

    test_labels = xr.DataArray(
        test_labels,
        dims=("time", "y", "x"),
        coords={
            "time": [datetime(2000, 1, 1), datetime(2000, 1, 1, 0, 5)],
            "y": np.arange(5),
            "x": np.arange(5),
        },
    )

    test_values = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 2, 0],
                [0, 2, 0, 2, 0],
                [0, 3, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0],
                [0, 3, 0, 3, 0],
                [0, 4, 0, 2, 0],
                [0, 0, 0, 0, 0],
            ],
        ]
    )

    test_values = xr.DataArray(
        test_values, dims=test_labels.dims, coords=test_labels.coords
    )

    test_weights = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ],
        ]
    )

    test_weights = xr.DataArray(
        test_weights, dims=test_labels.dims, coords=test_labels.coords
    )

    test_features = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "frame": [0, 0, 1, 1],
            "time": [
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
                datetime(2000, 1, 1, 0, 5),
                datetime(2000, 1, 1, 0, 5),
            ],
        }
    )

    statistics_mean = {"mean": np.mean}

    expected_mean_result = np.array([2, 2, 3, 2.5])

    bulk_statistics_output = tb_utils.get_statistics_from_mask(
        test_features, test_labels, test_values, statistic=statistics_mean
    )

    statistics_weighted_mean = {
        "weighted_mean": (lambda x, y: np.average(x, weights=y))
    }

    expected_weighted_mean_result = np.array([3, 2, 2, 2.5])

    bulk_statistics_output = tb_utils.get_statistics_from_mask(
        bulk_statistics_output,
        test_labels,
        test_values,
        test_weights,
        statistic=statistics_weighted_mean,
    )

    assert np.all(bulk_statistics_output["mean"] == expected_mean_result)
    assert np.all(
        bulk_statistics_output["weighted_mean"] == expected_weighted_mean_result
    )

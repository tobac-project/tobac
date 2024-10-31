from datetime import datetime
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import tobac
import tobac.utils as tb_utils
import tobac.testing as tb_test


@pytest.mark.parametrize("statistics_unsmoothed", [(False), (True)])
def test_bulk_statistics_fd(statistics_unsmoothed):
    """
    Assure that bulk statistics in feature detection work, both on smoothed and raw data
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
    stats = {"feature_max": np.max}

    # detect features
    threshold = 7
    fd_output = tobac.feature_detection.feature_detection_multithreshold(
        test_data_iris,
        dxy=1000,
        threshold=[threshold],
        n_min_threshold=100,
        target="maximum",
        statistic=stats,
        statistics_unsmoothed=statistics_unsmoothed,
    )

    assert "feature_max" in fd_output.columns


@pytest.mark.parametrize(
    "id_column, index",
    [
        ("feature", [1]),
        ("feature_id", [1]),
        ("cell", [1]),
    ],
)
def test_bulk_statistics(id_column, index):
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
    out_df = out_df.rename(columns={"feature": id_column})

    #  assure that bulk statistics in postprocessing give same result
    out_segmentation = tb_utils.get_statistics_from_mask(
        out_df, out_seg_mask, test_data_iris, statistic=stats, id_column=id_column
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
    out_df = out_df.rename(columns={"feature": id_column})
    #  assure that bulk statistics in postprocessing give same result
    out_segmentation = tb_utils.get_statistics_from_mask(
        out_df, out_seg_mask, test_data_iris, statistic=stats, id_column=id_column
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


def test_bulk_statistics_missing_segments():
    """
    Test that output feature dataframe contains all the same time steps even though for some timesteps,
    the statistics have not been calculated (in the case of unmatching labels or no segment labels for a given feature)
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
        fd_output, test_data_iris, dxy=1000, threshold=threshold
    )

    # specify some timesteps we set to zero
    timesteps_to_zero = [1, 3, 10]  # 0-based indexing
    modified_data = out_seg_mask.data.copy()
    # Set values to zero for the specified timesteps
    for timestep in timesteps_to_zero:
        modified_data[timestep, :, :] = 0  # Set all values for this timestep to zero

    #  assure that bulk statistics in postprocessing give same result
    out_segmentation = tb_utils.get_statistics_from_mask(
        out_df, out_seg_mask, test_data_iris, statistic=stats
    )

    assert out_df.time.unique().size == out_segmentation.time.unique().size


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


def test_bulk_statistics_time_invariant_field():
    """
    Some fields, such as area, are time invariant, and so passing an array with
    a time dimension is memory inefficient. Here we test if
    `get_statistics_from_mask` works if an input field has no time dimension,
    by passing the whole field to `get_statistics` rather than a time slice.
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

    test_areas = np.array(
        [
            [0.25, 0.5, 0.75, 1, 1],
            [0.25, 0.5, 0.75, 1, 1],
            [0.25, 0.5, 0.75, 1, 1],
            [0.25, 0.5, 0.75, 1, 1],
            [0.25, 0.5, 0.75, 1, 1],
        ]
    )

    test_areas = xr.DataArray(
        test_areas,
        dims=("y", "x"),
        coords={
            "y": np.arange(5),
            "x": np.arange(5),
        },
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

    statistics_sum = {"sum": np.sum}

    expected_sum_result = np.array([1.5, 2, 1.5, 2])

    bulk_statistics_output = tb_utils.get_statistics_from_mask(
        test_features, test_labels, test_areas, statistic=statistics_sum
    )

    assert np.all(bulk_statistics_output["sum"] == expected_sum_result)


def test_bulk_statistics_broadcasting():
    """
    Test whether field broadcasting works for bulk_statistics, with both leading and trailing dimensions tested
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
            [0.25, 0.5, 0.75, 1, 1],
            [1.25, 1.5, 1.75, 2, 2],
        ]
    )

    test_values = xr.DataArray(
        test_values,
        dims=("time", "x"),
        coords={"time": test_labels.time, "x": test_labels.x},
    )

    test_weights = np.array([0, 0, 1, 0, 0]).reshape([5, 1])

    test_weights = xr.DataArray(
        test_weights, dims=("y", "z"), coords={"y": test_labels.y}
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

    statistics_sum = {"sum": np.sum}

    expected_sum_result = np.array([1.5, 2, 4.5, 4])

    bulk_statistics_output = tb_utils.get_statistics_from_mask(
        test_features, test_labels, test_values, statistic=statistics_sum
    )

    statistics_weighted_sum = {"weighted_sum": (lambda x, y: np.sum(x * y))}

    expected_weighted_sum_result = np.array([0.5, 1, 1.5, 2])

    bulk_statistics_output = tb_utils.get_statistics_from_mask(
        bulk_statistics_output,
        test_labels,
        test_values,
        test_weights,
        statistic=statistics_weighted_sum,
    )

    assert np.all(bulk_statistics_output["sum"] == expected_sum_result)
    assert np.all(
        bulk_statistics_output["weighted_sum"] == expected_weighted_sum_result
    )


def test_get_statistics_collapse_axis():
    """
    Test the collapse_axis keyword of get_statistics
    """
    test_labels = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 2, 0],
            [0, 1, 0, 2, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=int,
    )

    test_values = np.array([0.25, 0.5, 0.75, 1, 1])

    test_features = pd.DataFrame(
        {
            "feature": [1, 2],
            "frame": [0, 0],
            "time": [
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
            ],
        }
    )
    statistics_sum = {"sum": np.sum}

    expected_sum_result_axis0 = np.array([0.5, 1])
    output_collapse_axis0 = tb_utils.get_statistics(
        test_features,
        test_labels,
        test_values,
        statistic=statistics_sum,
        collapse_axis=0,
    )
    assert np.all(output_collapse_axis0["sum"] == expected_sum_result_axis0)

    expected_sum_result_axis1 = np.array([2.25, 1.75])
    output_collapse_axis1 = tb_utils.get_statistics(
        test_features,
        test_labels,
        test_values,
        statistic=statistics_sum,
        collapse_axis=1,
    )
    assert np.all(output_collapse_axis1["sum"] == expected_sum_result_axis1)

    # Check that attempting broadcast raises a ValueError
    with pytest.raises(ValueError):
        _ = tb_utils.get_statistics(
            test_features,
            test_labels,
            test_values.reshape([5, 1]),
            statistic=statistics_sum,
            collapse_axis=0,
        )

    # Check that attempting to collapse all axes raises a ValueError:
    with pytest.raises(ValueError):
        _ = tb_utils.get_statistics(
            test_features,
            test_labels,
            test_values,
            statistic=statistics_sum,
            collapse_axis=[0, 1],
        )

    # Test with collpasing multiple axes
    test_labels = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 1, 0, 2, 0],
                [0, 1, 0, 2, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ],
        dtype=int,
    )
    test_values = np.array([0.5, 1])
    expected_sum_result_axis12 = np.array([1.5, 0.5])
    output_collapse_axis12 = tb_utils.get_statistics(
        test_features,
        test_labels,
        test_values,
        statistic=statistics_sum,
        collapse_axis=[1, 2],
    )
    assert np.all(output_collapse_axis12["sum"] == expected_sum_result_axis12)


def test_get_statistics_from_mask_collapse_dim():
    """
    Test the collapse_dim keyword of get_statistics_from_mask
    """

    test_labels = np.array(
        [
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
                    [0, 1, 0, 0, 0],
                    [0, 1, 0, 3, 0],
                    [0, 1, 0, 3, 0],
                    [0, 0, 0, 0, 0],
                ],
            ],
        ],
        dtype=int,
    )

    test_labels = xr.DataArray(
        test_labels,
        dims=("time", "z", "y", "x"),
        coords={
            "time": [datetime(2000, 1, 1)],
            "z": np.arange(2),
            "y": np.arange(5),
            "x": np.arange(5),
        },
    )

    test_values = np.ones([5, 5])

    test_values = xr.DataArray(
        test_values,
        dims=("x", "y"),
        coords={
            "y": np.arange(5),
            "x": np.arange(5),
        },
    )

    test_features = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "frame": [0, 0, 0],
            "time": [
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
            ],
        }
    )

    statistics_sum = {"sum": np.sum}

    expected_sum_result = np.array([3, 2, 2])

    # Test over a single dim
    statistics_output = tb_utils.get_statistics_from_mask(
        test_features,
        test_labels,
        test_values,
        statistic=statistics_sum,
        collapse_dim="z",
    )

    assert np.all(statistics_output["sum"] == expected_sum_result)

    test_values = np.ones([2])

    test_values = xr.DataArray(
        test_values,
        dims=("z",),
        coords={
            "z": np.arange(2),
        },
    )

    expected_sum_result = np.array([2, 1, 1])

    # Test over multiple dims
    statistics_output = tb_utils.get_statistics_from_mask(
        test_features,
        test_labels,
        test_values,
        statistic=statistics_sum,
        collapse_dim=("x", "y"),
    )

    assert np.all(statistics_output["sum"] == expected_sum_result)

    # Test that collapse_dim not in labels raises an error
    with pytest.raises(ValueError):
        _ = statistics_output = tb_utils.get_statistics_from_mask(
            test_features,
            test_labels,
            test_values,
            statistic=statistics_sum,
            collapse_dim="not_a_dim",
        )

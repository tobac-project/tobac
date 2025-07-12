"""Tests for utils.mask"""

from datetime import datetime
import warnings
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tobac.utils.mask import convert_feature_mask_to_cells


def test_convert_feature_mask_to_cells_single_cell():
    """Test basic functionality of convert_feature_mask_to_cells with a single
    tracked cell
    """
    test_data = np.zeros([3, 4, 5], dtype=int)
    test_data[0, 1:3, 1:4] = 1
    test_data[1, 1:3, 1:4] = 2
    test_data[2, 1:3, 1:4] = 3

    test_mask = xr.DataArray(
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
            "cell": [1, 1, 1],
        }
    )

    cell_mask = convert_feature_mask_to_cells(test_features, test_mask)

    # Test all cell mask values are 0 or 1
    assert np.all(np.isin(cell_mask.values, [0, 1]))

    # Test all cell mask values where the feature mask is not zero are 1
    assert np.all(cell_mask.values[test_mask.values != 0] == 1)

    # Test all cell mask values where the feature mask is zero are 0
    assert np.all(cell_mask.values[test_mask.values == 0] == 0)

    # Test coords are the same
    assert cell_mask.coords.keys() == test_mask.coords.keys()


def test_convert_feature_mask_to_cells_multiple_cells():
    """Test functionality of convert_feature_mask_to_cells with multiple cells
    and non-consecutive feature and cell values
    """
    test_data = np.zeros([3, 4, 5], dtype=int)
    test_data[0, 1:3, 1:4] = 1
    test_data[1, 1:3, 1:4] = 2
    test_data[1, 3:, 3:] = 5
    test_data[2, 3:, 3:] = 6

    test_mask = xr.DataArray(
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
            "feature": [1, 2, 5, 6],
            "frame": [0, 1, 1, 2],
            "time": [
                datetime(2000, 1, 1, 0),
                datetime(2000, 1, 1, 1),
                datetime(2000, 1, 1, 1),
                datetime(2000, 1, 1, 2),
            ],
            "cell": [1, 1, 3, 3],
        }
    )

    cell_mask = convert_feature_mask_to_cells(test_features, test_mask)

    # Test all cell mask values are 0, 1, or 3
    assert np.all(np.isin(cell_mask.values, [0, 1, 3]))

    # Test all cell mask values where the feature mask is 1 or 2 are 1
    assert np.all(cell_mask.values[np.isin(test_mask.values, [1, 2])] == 1)

    # Test all cell mask values where the feature mask is 5 or 6 are 3
    assert np.all(cell_mask.values[np.isin(test_mask.values, [5, 6])] == 3)

    # Test all cell mask values where the feature mask is zero are 0
    assert np.all(cell_mask.values[test_mask.values == 0] == 0)

    # Test coords are the same
    assert cell_mask.coords.keys() == test_mask.coords.keys()


def test_convert_feature_mask_to_cells_mismatched_mask():
    """
    Test a situation when the user provides a mask that does not correspond to
    the given feature dataframe, and has additional values. This should raise a
    ValueError and inform the user of the problem.
    """

    test_data = np.zeros([3, 4, 5], dtype=int)
    test_data[0, 1:3, 1:4] = 1
    test_data[1, 1:3, 1:4] = 2
    test_data[2, 1:3, 1:4] = 4

    test_mask = xr.DataArray(
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
            "cell": [1, 1, 1],
        }
    )

    with pytest.raises(
        ValueError, match="Values in feature_mask are not present in features*"
    ):
        cell_mask = convert_feature_mask_to_cells(test_features, test_mask)


def test_convert_feature_mask_to_cells_no_cell_column():
    """
    Test correct error handling when convert_feature_mask_to_cells is given a
    features dataframe with no cell column
    """
    test_data = np.zeros([3, 4, 5], dtype=int)
    test_data[0, 1:3, 1:4] = 1
    test_data[1, 1:3, 1:4] = 2
    test_data[2, 1:3, 1:4] = 3

    test_mask = xr.DataArray(
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
        }
    )

    with pytest.raises(ValueError, match="`cell` column not found in features input*"):
        cell_mask = convert_feature_mask_to_cells(test_features, test_mask)


def test_convert_feature_mask_to_cells_stub_value():
    """
    Test filtering of stub values from cell_mask
    """
    test_data = np.zeros([3, 4, 5], dtype=int)
    test_data[0, 1:3, 1:4] = 1
    test_data[1, 1:3, 1:4] = 2
    test_data[2, 1:3, 1:4] = 3

    test_mask = xr.DataArray(
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
            "cell": [1, 1, -1],
        }
    )

    cell_mask = convert_feature_mask_to_cells(test_features, test_mask)

    # Test that without providing a stub value the stub feature is relabelled to -1
    assert np.all(cell_mask.values[test_mask.values == 3] == -1)

    cell_mask = convert_feature_mask_to_cells(test_features, test_mask, stubs=-1)

    # Test that providing a stub value the stub feature is relabelled to 0
    assert np.all(cell_mask.values[test_mask.values == 3] == 0)

    cell_mask = convert_feature_mask_to_cells(test_features, test_mask, stubs=-999)

    # Test that providing a different stub value the stub feature is relabelled to -1
    assert np.all(cell_mask.values[test_mask.values == 3] == -1)


def test_convert_feature_mask_to_cells_no_input_mutation():
    """Test that convert_feature_mask_to_cells does not alter the input features 
    and mask
    """
    test_data = np.zeros([3, 4, 5], dtype=int)
    test_data[0, 1:3, 1:4] = 1
    test_data[1, 1:3, 1:4] = 2
    test_data[2, 1:3, 1:4] = 3

    test_mask = xr.DataArray(
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
            "cell": [1, 1, -1],
        }
    )

    mask_copy = test_mask.copy(deep=True)
    features_copy = test_features.copy(deep=True)

    cell_mask = convert_feature_mask_to_cells(test_features, test_mask, stubs=-1)

    # Test dataframe is the same
    pd.testing.assert_frame_equal(
        test_features, features_copy
    )
    
    # Test mask is the same
    assert mask_copy.equals(test_mask)
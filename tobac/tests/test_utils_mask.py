"""Tests for utils.mask"""

from datetime import datetime
import warnings
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tobac.utils.mask import (
    convert_cell_mask_to_features,
    convert_cell_mask_to_tracks,
    convert_feature_mask_to_cells,
    convert_feature_mask_to_tracks,
)


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
    pd.testing.assert_frame_equal(test_features, features_copy)

    # Test mask is the same
    assert mask_copy.equals(test_mask)


def test_convert_feature_mask_to_cells_inplace():
    """Test that convert_feature_mask_to_cells does alter the input mask when
    the inplace keyword is used
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

    cell_mask = convert_feature_mask_to_cells(
        test_features, test_mask, stubs=-1, inplace=True
    )

    # Test dataframe is the same
    pd.testing.assert_frame_equal(test_features, features_copy)

    # Test mask is the same
    assert cell_mask.equals(test_mask)
    assert not mask_copy.equals(test_mask)


def test_convert_cell_mask_to_features_single_timestep():
    """Test basic functionality of convert_cell_mask_to_features with a single
    tracked cell and timestep
    """
    test_data = np.zeros([1, 4, 5], dtype=int)
    test_data[0, 1:3, 1:4] = 1

    test_mask = xr.DataArray(
        test_data,
        dims=("time", "y", "x"),
        coords=dict(time=[datetime(2000, 1, 1, 0)]),
        attrs=dict(units="feature"),
    )

    test_features = pd.DataFrame(
        {
            "feature": [2],
            "frame": [0],
            "time": [datetime(2000, 1, 1, 0)],
            "cell": [1],
        }
    )

    feature_mask = convert_cell_mask_to_features(test_features, test_mask)

    # Test all feature mask values are 0 or 2
    assert np.all(np.isin(feature_mask.values, [0, 2]))

    # Test all feature mask values where the cell mask is not zero are 2
    assert np.all(feature_mask.values[test_mask.values != 0] == 2)

    # Test all cell mask values where the feature mask is zero are 0
    assert np.all(feature_mask.values[test_mask.values == 0] == 0)

    # Test coords are the same
    assert feature_mask.coords.keys() == test_mask.coords.keys()


def test_convert_cell_mask_to_features_single_cell():
    """Test basic functionality of convert_cell_mask_to_features with a single
    tracked cell
    """
    test_data = np.zeros([3, 4, 5], dtype=int)
    test_data[0, 1:3, 1:4] = 1
    test_data[1, 1:3, 1:4] = 1
    test_data[2, 1:3, 1:4] = 1

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

    feature_mask = convert_cell_mask_to_features(test_features, test_mask)

    # Test all feature mask values where the cell mask is not zero are in test_features.feature
    assert np.all(
        np.isin(feature_mask.values[test_mask.values != 0], test_features.feature)
    )

    # Test all cell mask values where the feature mask is zero are 0
    assert np.all(feature_mask.values[test_mask.values == 0] == 0)

    # Test coords are the same
    assert feature_mask.coords.keys() == test_mask.coords.keys()


def test_convert_cell_mask_to_features_multiple_cells():
    """Test functionality of convert_cell_mask_to_features with multiple cells
    and non-consecutive feature and cell values
    """
    test_data = np.zeros([3, 4, 5], dtype=int)
    test_data[0, 1:3, 1:4] = 1
    test_data[1, 1:3, 1:4] = 1
    test_data[1, 3:, 3:] = 3
    test_data[2, 3:, 3:] = 3

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

    feature_mask = convert_cell_mask_to_features(test_features, test_mask)

    # Test all feature mask values where the cell mask is not zero are in test_features.feature
    assert np.all(
        np.isin(feature_mask.values[test_mask.values != 0], test_features.feature)
    )

    # Test all cell mask values where the cell mask is 1 are 1 or 2
    assert np.all(np.isin(feature_mask.values[test_mask.values == 1], [1, 2]))

    # Test all cell mask values where the cell mask is 3 are 5 or 6
    assert np.all(np.isin(feature_mask.values[test_mask.values == 3], [5, 6]))

    # Test all cell mask values where the feature mask is zero are 0
    assert np.all(feature_mask.values[test_mask.values == 0] == 0)

    # Test coords are the same
    assert feature_mask.coords.keys() == test_mask.coords.keys()


def test_convert_cell_mask_to_features_0_cell():
    """Test functionality of convert_feature_mask_to_cells when a cell has the
    value 0
    """
    test_data = np.zeros([3, 4, 5], dtype=int)
    test_data[0, 1:3, 1:4] = 1
    test_data[1, 1:3, 1:4] = 1
    test_data[2, 1:3, 1:4] = 0

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

    feature_mask = convert_cell_mask_to_features(test_features, test_mask)

    # Test all feature mask values where the cell mask is not zero are in test_features.feature
    assert np.all(
        np.isin(feature_mask.values[test_mask.values != 0], test_features.feature)
    )

    # Test all cell mask values where the feature mask is zero are 0
    assert np.all(feature_mask.values[test_mask.values == 0] == 0)

    # Test coords are the same
    assert feature_mask.coords.keys() == test_mask.coords.keys()


def test_convert_cell_mask_to_features_stub_cell():
    """Test functionality of convert_feature_mask_to_cells when a cell has a
    stub value but cell mask is 0
    """
    test_data = np.zeros([3, 4, 5], dtype=int)
    test_data[0, 1:3, 1:4] = 1
    test_data[1, 1:3, 1:4] = -1
    test_data[1, 3:, 3:] = -1
    test_data[2, 3:, 3:] = 3

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
            "cell": [1, -1, -1, 3],
        }
    )

    # Test without stub value provided the correct error is raised.
    with pytest.raises(
        ValueError,
        match="Duplicate cell values found for a single timestep in features. This may be because there are stub cells *",
    ):
        feature_mask = convert_cell_mask_to_features(test_features, test_mask)

    feature_mask = convert_cell_mask_to_features(test_features, test_mask, stubs=-1)

    # Test all feature mask values where the cell mask is not zero are in test_features.feature
    assert np.all(
        np.isin(feature_mask.values[test_mask.values > 0], test_features.feature)
    )

    # Test all cell mask values where the feature mask is zero or the stub value are 0
    assert np.all(feature_mask.values[np.isin(test_mask.values, [0, -1])] == 0)

    # Test coords are the same
    assert feature_mask.coords.keys() == test_mask.coords.keys()

    with pytest.raises(
        ValueError,
        match="Duplicate cell values found for a single timestep in features that does not match the provided stub value*",
    ):
        feature_mask = convert_cell_mask_to_features(
            test_features, test_mask, stubs=-999
        )


def test_convert_cell_mask_to_features_mismatched_cell():
    """Test functionality of convert_feature_mask_to_cells when a cell exists in
    the mask that does not occur in the features dataframe
    """
    test_data = np.zeros([3, 4, 5], dtype=int)
    test_data[0, 1:3, 1:4] = 1
    test_data[1, 1:3, 1:4] = 1
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
            "cell": [1, 1, 2],
        }
    )

    with pytest.raises(
        ValueError, match="Cell values in cell_mask are not present in features, *"
    ):
        feature_mask = convert_cell_mask_to_features(test_features, test_mask)


def test_convert_cell_mask_to_features_no_cell_column():
    """
    Test correct error handling when convert_cell_mask_to_features is given a
    features dataframe with no cell column
    """
    test_data = np.zeros([3, 4, 5], dtype=int)
    test_data[0, 1:3, 1:4] = 1
    test_data[1, 1:3, 1:4] = 1
    test_data[2, 1:3, 1:4] = 1

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
        feature_mask = convert_cell_mask_to_features(test_features, test_mask)


def test_convert_cell_mask_to_features_no_input_mutation():
    """Test that convert_cell_mask_to_features does not alter the input features
    and mask
    """
    test_data = np.zeros([3, 4, 5], dtype=int)
    test_data[0, 1:3, 1:4] = 1
    test_data[1, 1:3, 1:4] = 1
    test_data[2, 1:3, 1:4] = -1

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

    feature_mask = convert_cell_mask_to_features(test_features, test_mask, stubs=-1)

    # Test dataframe is the same
    pd.testing.assert_frame_equal(test_features, features_copy)

    # Test mask is the same
    assert mask_copy.equals(test_mask)


def test_convert_cell_mask_to_features_inplace():
    """Test that convert_cell_mask_to_features does alter the input mask when
    the inplace keyword is used
    """
    test_data = np.zeros([3, 4, 5], dtype=int)
    test_data[0, 1:3, 1:4] = 1
    test_data[1, 1:3, 1:4] = 1
    test_data[2, 1:3, 1:4] = -1

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

    feature_mask = convert_cell_mask_to_features(
        test_features, test_mask, stubs=-1, inplace=True
    )

    # Test dataframe is the same
    pd.testing.assert_frame_equal(test_features, features_copy)

    # Test mask is the same
    assert feature_mask.equals(test_mask)
    assert not mask_copy.equals(test_mask)


def test_convert_feature_mask_to_tracks_single_track():
    """Test basic functionality of convert_feature_mask_to_tracks with a single
    track
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
            "track": [1, 1, 1],
        }
    )

    track_mask = convert_feature_mask_to_tracks(test_features, test_mask)

    # Test all cell mask values are 0 or 1
    assert np.all(np.isin(track_mask.values, [0, 1]))

    # Test all cell mask values where the feature mask is not zero are 1
    assert np.all(track_mask.values[test_mask.values != 0] == 1)

    # Test all cell mask values where the feature mask is zero are 0
    assert np.all(track_mask.values[test_mask.values == 0] == 0)

    # Test coords are the same
    assert track_mask.coords.keys() == test_mask.coords.keys()


def test_convert_feature_mask_to_tracks_multiple_tracks():
    """Test functionality of convert_feature_mask_to_tracks with multiple tracks
    and non-consecutive feature and track values
    """
    test_data = np.zeros([3, 4, 5], dtype=int)
    test_data[0, 1:3, 1:4] = 1
    test_data[1, 1:3, 1:4] = 2
    test_data[1, 3:, 3:] = 5
    test_data[2, 3:, 3:] = 6
    test_data[2, 0, 0] = 8
    test_data[2, 0, 1] = 9

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
            "feature": [1, 2, 5, 6, 8, 9],
            "frame": [0, 1, 1, 2, 2, 2],
            "time": [
                datetime(2000, 1, 1, 0),
                datetime(2000, 1, 1, 1),
                datetime(2000, 1, 1, 1),
                datetime(2000, 1, 1, 2),
                datetime(2000, 1, 1, 2),
                datetime(2000, 1, 1, 2),
            ],
            "cell": [1, 1, 3, 3, 4, 5],
            "track": [1, 1, 2, 2, 4, 4],
        }
    )

    track_mask = convert_feature_mask_to_tracks(test_features, test_mask)

    # Test all track mask values are 0, 1, 2 or 4
    assert np.all(np.isin(track_mask.values, [0, 1, 2, 4]))

    # Test all track mask values where the feature mask is 1 or 2 are 1
    assert np.all(track_mask.values[np.isin(test_mask.values, [1, 2])] == 1)

    # Test all track mask values where the feature mask is 5 or 6 are 2
    assert np.all(track_mask.values[np.isin(test_mask.values, [5, 6])] == 2)

    # Test all track mask values where the feature mask is 8 or 9 are 4
    assert np.all(track_mask.values[np.isin(test_mask.values, [8, 9])] == 4)

    # Test all track mask values where the feature mask is zero are 0
    assert np.all(track_mask.values[test_mask.values == 0] == 0)

    # Test coords are the same
    assert track_mask.coords.keys() == test_mask.coords.keys()


def test_convert_feature_mask_to_tracks_mismatched_mask():
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
            "track": [1, 1, 1],
        }
    )

    with pytest.raises(
        ValueError, match="Values in feature_mask are not present in features*"
    ):
        track_mask = convert_feature_mask_to_tracks(test_features, test_mask)


def test_convert_feature_mask_to_tracks_no_track_column():
    """
    Test correct error handling when convert_feature_mask_to_tracks is given a
    features dataframe with no track column
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

    with pytest.raises(ValueError, match="`track` column not found in features input*"):
        track_mask = convert_feature_mask_to_tracks(test_features, test_mask)


def test_convert_feature_mask_to_tracks_stub_value():
    """
    Test filtering of stub values from track_mask
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
            "track": [1, 1, -1],
        }
    )

    track_mask = convert_feature_mask_to_tracks(test_features, test_mask)

    # Test that without providing a stub value the stub feature is relabelled to -1
    assert np.all(track_mask.values[test_mask.values == 3] == -1)

    track_mask = convert_feature_mask_to_tracks(test_features, test_mask, stubs=-1)

    # Test that providing a stub value the stub feature is relabelled to 0
    assert np.all(track_mask.values[test_mask.values == 3] == 0)

    track_mask = convert_feature_mask_to_tracks(test_features, test_mask, stubs=-999)

    # Test that providing a different stub value the stub feature is relabelled to -1
    assert np.all(track_mask.values[test_mask.values == 3] == -1)


def test_convert_feature_mask_to_tracks_no_input_mutation():
    """Test that convert_feature_mask_to_tracks does not alter the input features
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
            "track": [1, 1, -1],
        }
    )

    mask_copy = test_mask.copy(deep=True)
    features_copy = test_features.copy(deep=True)

    track_mask = convert_feature_mask_to_tracks(test_features, test_mask, stubs=-1)

    # Test dataframe is the same
    pd.testing.assert_frame_equal(test_features, features_copy)

    # Test mask is the same
    assert mask_copy.equals(test_mask)


def test_convert_feature_mask_to_tracks_inplace():
    """Test that convert_feature_mask_to_tracks does alter the input mask when
    the inplace keyword is used
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
            "track": [1, 1, -1],
        }
    )

    mask_copy = test_mask.copy(deep=True)
    features_copy = test_features.copy(deep=True)

    track_mask = convert_feature_mask_to_tracks(
        test_features, test_mask, stubs=-1, inplace=True
    )

    # Test dataframe is the same
    pd.testing.assert_frame_equal(test_features, features_copy)

    # Test mask is the same
    assert track_mask.equals(test_mask)
    assert not mask_copy.equals(test_mask)


# Cell to track tests


def test_convert_cell_mask_to_tracks_single_track():
    """Test basic functionality of convert_cell_mask_to_tracks with a single
    track
    """
    test_data = np.zeros([3, 4, 5], dtype=int)
    test_data[0, 1:3, 1:4] = 1
    test_data[1, 1:3, 1:4] = 1
    test_data[2, 1:3, 1:4] = 1

    test_mask = xr.DataArray(
        test_data,
        dims=("time", "y", "x"),
        coords=dict(
            time=pd.date_range(
                datetime(2000, 1, 1, 0), datetime(2000, 1, 1, 2), periods=3
            )
        ),
        attrs=dict(units="cell"),
    )

    test_features = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "frame": [0, 1, 2],
            "time": pd.date_range(
                datetime(2000, 1, 1, 0), datetime(2000, 1, 1, 2), periods=3
            ),
            "cell": [1, 1, 1],
            "track": [1, 1, 1],
        }
    )

    track_mask = convert_cell_mask_to_tracks(test_features, test_mask)

    # Test all cell mask values are 0 or 1
    assert np.all(np.isin(track_mask.values, [0, 1]))

    # Test all cell mask values where the cell mask is not zero are 1
    assert np.all(track_mask.values[test_mask.values != 0] == 1)

    # Test all cell mask values where the cell mask is zero are 0
    assert np.all(track_mask.values[test_mask.values == 0] == 0)

    # Test coords are the same
    assert track_mask.coords.keys() == test_mask.coords.keys()


def test_convert_cell_mask_to_tracks_multiple_tracks():
    """Test functionality of convert_cell_mask_to_tracks with multiple tracks
    and non-consecutive cell and track values
    """
    test_data = np.zeros([3, 4, 5], dtype=int)
    test_data[0, 1:3, 1:4] = 1
    test_data[1, 1:3, 1:4] = 1
    test_data[1, 3:, 3:] = 3
    test_data[2, 3:, 3:] = 3
    test_data[2, 0, 0] = 4
    test_data[2, 0, 1] = 5

    test_mask = xr.DataArray(
        test_data,
        dims=("time", "y", "x"),
        coords=dict(
            time=pd.date_range(
                datetime(2000, 1, 1, 0), datetime(2000, 1, 1, 2), periods=3
            )
        ),
        attrs=dict(units="cell"),
    )

    test_features = pd.DataFrame(
        {
            "feature": [1, 2, 5, 6, 8, 9],
            "frame": [0, 1, 1, 2, 2, 2],
            "time": [
                datetime(2000, 1, 1, 0),
                datetime(2000, 1, 1, 1),
                datetime(2000, 1, 1, 1),
                datetime(2000, 1, 1, 2),
                datetime(2000, 1, 1, 2),
                datetime(2000, 1, 1, 2),
            ],
            "cell": [1, 1, 3, 3, 4, 5],
            "track": [1, 1, 2, 2, 4, 4],
        }
    )

    track_mask = convert_cell_mask_to_tracks(test_features, test_mask)

    # Test all track mask values are 0, 1, 2 or 4
    assert np.all(np.isin(track_mask.values, [0, 1, 2, 4]))

    # Test all track mask values where the cell mask is 1 are 1
    assert np.all(track_mask.values[np.isin(test_mask.values, [1])] == 1)

    # Test all track mask values where the cell mask is 3 are 2
    assert np.all(track_mask.values[np.isin(test_mask.values, [3])] == 2)

    # Test all track mask values where the cell mask is 4 or 5 are 4
    assert np.all(track_mask.values[np.isin(test_mask.values, [4, 5])] == 4)

    # Test all track mask values where the cell mask is zero are 0
    assert np.all(track_mask.values[test_mask.values == 0] == 0)

    # Test coords are the same
    assert track_mask.coords.keys() == test_mask.coords.keys()


def test_convert_cell_mask_to_tracks_mismatched_mask():
    """
    Test a situation when the user provides a mask that does not correspond to
    the given cell dataframe, and has additional values. This should raise a
    ValueError and inform the user of the problem.
    """

    test_data = np.zeros([3, 4, 5], dtype=int)
    test_data[0, 1:3, 1:4] = 1
    test_data[1, 1:3, 1:4] = 2
    test_data[2, 1:3, 1:4] = 1

    test_mask = xr.DataArray(
        test_data,
        dims=("time", "y", "x"),
        coords=dict(
            time=pd.date_range(
                datetime(2000, 1, 1, 0), datetime(2000, 1, 1, 2), periods=3
            )
        ),
        attrs=dict(units="cell"),
    )

    test_features = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "frame": [0, 1, 2],
            "time": pd.date_range(
                datetime(2000, 1, 1, 0), datetime(2000, 1, 1, 2), periods=3
            ),
            "cell": [1, 1, 1],
            "track": [1, 1, 1],
        }
    )

    with pytest.raises(
        ValueError, match="Values in cell_mask are not present in features*"
    ):
        track_mask = convert_cell_mask_to_tracks(test_features, test_mask)


def test_convert_cell_mask_to_tracks_no_track_column():
    """
    Test correct error handling when convert_cell_mask_to_tracks is given a
    features dataframe with no track column
    """
    test_data = np.zeros([3, 4, 5], dtype=int)
    test_data[0, 1:3, 1:4] = 1
    test_data[1, 1:3, 1:4] = 1
    test_data[2, 1:3, 1:4] = 1

    test_mask = xr.DataArray(
        test_data,
        dims=("time", "y", "x"),
        coords=dict(
            time=pd.date_range(
                datetime(2000, 1, 1, 0), datetime(2000, 1, 1, 2), periods=3
            )
        ),
        attrs=dict(units="cell"),
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

    with pytest.raises(ValueError, match="`track` column not found in features input*"):
        track_mask = convert_cell_mask_to_tracks(test_features, test_mask)


def test_convert_cell_mask_to_tracks_stub_value():
    """
    Test filtering of stub values from track_mask
    """
    test_data = np.zeros([3, 4, 5], dtype=int)
    test_data[0, 1:3, 1:4] = 1
    test_data[1, 1:3, 1:4] = 1
    test_data[2, 1:3, 1:4] = -1

    test_mask = xr.DataArray(
        test_data,
        dims=("time", "y", "x"),
        coords=dict(
            time=pd.date_range(
                datetime(2000, 1, 1, 0), datetime(2000, 1, 1, 2), periods=3
            )
        ),
        attrs=dict(units="cell"),
    )

    test_features = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "frame": [0, 1, 2],
            "time": pd.date_range(
                datetime(2000, 1, 1, 0), datetime(2000, 1, 1, 2), periods=3
            ),
            "cell": [1, 1, -1],
            "track": [1, 1, -1],
        }
    )

    track_mask = convert_cell_mask_to_tracks(test_features, test_mask)

    # Test that without providing a stub value the stub cell is relabelled to -1
    assert np.all(track_mask.values[test_mask.values == -1] == -1)

    track_mask = convert_cell_mask_to_tracks(test_features, test_mask, stubs=-1)

    # Test that providing a stub value the stub cell is relabelled to 0
    assert np.all(track_mask.values[test_mask.values == -1] == 0)

    track_mask = convert_cell_mask_to_tracks(test_features, test_mask, stubs=-999)

    # Test that providing a different stub value the stub cell is relabelled to -1
    assert np.all(track_mask.values[test_mask.values == -1] == -1)


def test_convert_cell_mask_to_tracks_no_input_mutation():
    """Test that convert_cell_mask_to_tracks does not alter the input features
    and mask
    """
    test_data = np.zeros([3, 4, 5], dtype=int)
    test_data[0, 1:3, 1:4] = 1
    test_data[1, 1:3, 1:4] = 1
    test_data[2, 1:3, 1:4] = -1

    test_mask = xr.DataArray(
        test_data,
        dims=("time", "y", "x"),
        coords=dict(
            time=pd.date_range(
                datetime(2000, 1, 1, 0), datetime(2000, 1, 1, 2), periods=3
            )
        ),
        attrs=dict(units="cell"),
    )

    test_features = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "frame": [0, 1, 2],
            "time": pd.date_range(
                datetime(2000, 1, 1, 0), datetime(2000, 1, 1, 2), periods=3
            ),
            "cell": [1, 1, -1],
            "track": [1, 1, -1],
        }
    )

    mask_copy = test_mask.copy(deep=True)
    features_copy = test_features.copy(deep=True)

    track_mask = convert_cell_mask_to_tracks(test_features, test_mask, stubs=-1)

    # Test dataframe is the same
    pd.testing.assert_frame_equal(test_features, features_copy)

    # Test mask is the same
    assert mask_copy.equals(test_mask)


def test_convert_cell_mask_to_tracks_inplace():
    """Test that convert_cell_mask_to_tracks does alter the input mask when
    the inplace keyword is used
    """
    test_data = np.zeros([3, 4, 5], dtype=int)
    test_data[0, 1:3, 1:4] = 1
    test_data[1, 1:3, 1:4] = 1
    test_data[2, 1:3, 1:4] = -1

    test_mask = xr.DataArray(
        test_data,
        dims=("time", "y", "x"),
        coords=dict(
            time=pd.date_range(
                datetime(2000, 1, 1, 0), datetime(2000, 1, 1, 2), periods=3
            )
        ),
        attrs=dict(units="cell"),
    )

    test_features = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "frame": [0, 1, 2],
            "time": pd.date_range(
                datetime(2000, 1, 1, 0), datetime(2000, 1, 1, 2), periods=3
            ),
            "cell": [1, 1, -1],
            "track": [1, 1, -1],
        }
    )

    mask_copy = test_mask.copy(deep=True)
    features_copy = test_features.copy(deep=True)

    track_mask = convert_cell_mask_to_tracks(
        test_features, test_mask, stubs=-1, inplace=True
    )

    # Test dataframe is the same
    pd.testing.assert_frame_equal(test_features, features_copy)

    # Test mask is the same
    assert track_mask.equals(test_mask)
    assert not mask_copy.equals(test_mask)

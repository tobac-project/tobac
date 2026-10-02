import pytest
import numpy as np
import pandas as pd
import xarray as xr
from tobac.tracking.overlap import linking_overlap


class TestLinkingOverlap:
    """Test suite for linking_overlap function."""

    @pytest.fixture
    def simple_2d_data(self):
        """Create simple 2D test data with two time steps."""
        # Create mask with two time steps
        time = pd.date_range("2020-01-01", periods=2, freq="h")
        hdim_1 = np.arange(10)
        hdim_2 = np.arange(10)

        # Time 0: feature 1 at (2:4, 2:4), feature 2 at (6:8, 6:8)
        mask_t0 = np.zeros((10, 10), dtype=int)
        mask_t0[2:4, 2:4] = 1
        mask_t0[6:8, 6:8] = 2

        # Time 1: feature 3 at (3:5, 3:5), feature 4 at (7:9, 7:9)
        mask_t1 = np.zeros((10, 10), dtype=int)
        mask_t1[3:5, 3:5] = 3
        mask_t1[7:9, 7:9] = 4

        mask_data = np.stack([mask_t0, mask_t1])
        mask = xr.DataArray(
            mask_data,
            coords={"time": time, "hdim_1": hdim_1, "hdim_2": hdim_2},
            dims=["time", "hdim_1", "hdim_2"],
        )

        # Create features dataframe
        features = pd.DataFrame(
            {
                "feature": [1, 2, 3, 4],
                "frame": [0, 0, 1, 1],
                "time": [time[0], time[0], time[1], time[1]],
                "hdim_1": [2.5, 6.5, 3.5, 7.5],
                "hdim_2": [2.5, 6.5, 3.5, 7.5],
            }
        )

        return features, mask

    @pytest.fixture
    def simple_3d_data(self):
        """Create simple 3D test data with two time steps."""
        time = pd.date_range("2020-01-01", periods=2, freq="h")
        vdim = np.arange(5)
        hdim_1 = np.arange(10)
        hdim_2 = np.arange(10)

        # Time 0: feature 1 at (1:3, 2:4, 2:4), feature 2 at (2:4, 6:8, 6:8)
        mask_t0 = np.zeros((5, 10, 10), dtype=int)
        mask_t0[1:3, 2:4, 2:4] = 1
        mask_t0[2:4, 6:8, 6:8] = 2

        # Time 1: feature 3 at (1:3, 3:5, 3:5), feature 4 at (2:4, 7:9, 7:9)
        mask_t1 = np.zeros((5, 10, 10), dtype=int)
        mask_t1[1:3, 3:5, 3:5] = 3
        mask_t1[2:4, 7:9, 7:9] = 4

        mask_data = np.stack([mask_t0, mask_t1])
        mask = xr.DataArray(
            mask_data,
            coords={"time": time, "vdim": vdim, "hdim_1": hdim_1, "hdim_2": hdim_2},
            dims=["time", "vdim", "hdim_1", "hdim_2"],
        )

        # Create features dataframe
        features = pd.DataFrame(
            {
                "feature": [1, 2, 3, 4],
                "frame": [0, 0, 1, 1],
                "time": [time[0], time[0], time[1], time[1]],
                "vdim": [1.5, 2.5, 1.5, 2.5],
                "hdim_1": [2.5, 6.5, 3.5, 7.5],
                "hdim_2": [2.5, 6.5, 3.5, 7.5],
            }
        )

        return features, mask

    @pytest.fixture
    def non_overlapping_data(self):
        """Create test data where features don't overlap between time steps."""
        time = pd.date_range("2020-01-01", periods=2, freq="h")
        hdim_1 = np.arange(20)
        hdim_2 = np.arange(20)

        # Time 0: feature 1 at (2:5, 2:5)
        mask_t0 = np.zeros((20, 20), dtype=int)
        mask_t0[2:5, 2:5] = 1

        # Time 1: feature 2 at (15:18, 15:18) - far from origin
        mask_t1 = np.zeros((20, 20), dtype=int)
        mask_t1[15:18, 15:18] = 1

        mask_data = np.stack([mask_t0, mask_t1])
        mask = xr.DataArray(
            mask_data,
            coords={"time": time, "hdim_1": hdim_1, "hdim_2": hdim_2},
            dims=["time", "hdim_1", "hdim_2"],
        )

        features = pd.DataFrame(
            {
                "feature": [1, 2],
                "frame": [0, 1],
                "time": [time[0], time[1]],
                "hdim_1": [3.0, 16.0],
                "hdim_2": [3.0, 16.0],
            }
        )

        return features, mask

    def test_basic_linking_2d(self, simple_2d_data):
        """Test basic feature linking in 2D."""
        features, mask = simple_2d_data
        features_copy, mask_copy = features.copy(), mask.copy()
        result = linking_overlap(features, mask)

        # Check output structure
        assert "cell" in result.columns
        assert "time_cell" in result.columns
        assert len(result) == len(features)
        pd.testing.assert_index_equal(result.index, features.index)

        # check input is not modified
        pd.testing.assert_frame_equal(features, features_copy)
        xr.testing.assert_equal(mask, mask_copy)

        # Check that features at subsequent time steps have the same cell id
        assert (
            result.cell[result.feature == 1].values
            == result.cell[result.feature == 3].values
        )
        assert (
            result.cell[result.feature == 2].values
            == result.cell[result.feature == 4].values
        )

        # Check that cells are different and have positive values
        assert np.all(result.cell > 0)
        assert (
            result.cell[result.feature == 1].values
            != result.cell[result.feature == 2].values
        )

    def test_basic_linking_3d(self, simple_3d_data):
        """Test basic feature linking in 3D."""
        features, mask = simple_3d_data
        features_copy, mask_copy = features.copy(), mask.copy()
        result = linking_overlap(features, mask, vertical_coord="vdim")

        assert "cell" in result.columns
        assert "time_cell" in result.columns
        assert len(result) == len(features)
        pd.testing.assert_index_equal(result.index, features.index)

        pd.testing.assert_frame_equal(features, features_copy)
        xr.testing.assert_equal(mask, mask_copy)

        # Check that features at subsequent time steps have the same cell id
        assert (
            result.cell[result.feature == 1].values
            == result.cell[result.feature == 3].values
        )
        assert (
            result.cell[result.feature == 2].values
            == result.cell[result.feature == 4].values
        )

        # Check that cells are different and have positive values
        assert np.all(result.cell > 0)
        assert (
            result.cell[result.feature == 1].values
            != result.cell[result.feature == 2].values
        )

    def test_non_overlapping_features(self, non_overlapping_data):
        """Test linking when features don't overlap."""
        features, mask = non_overlapping_data
        result = linking_overlap(features, mask)

        # Feature at t1 should have different cell ID if no overlap
        cell_t0 = result.loc[result["time"] == features["time"].iloc[0], "cell"].values[
            0
        ]
        cell_t1 = result.loc[result["time"] == features["time"].iloc[1], "cell"].values[
            0
        ]

        # Both should be unassigned
        assert (cell_t0) == -1 and (cell_t1) == -1

    def test_stubs_filtering(self, simple_2d_data):
        """Test that stub cells are filtered correctly."""
        features, mask = simple_2d_data

        # With stubs=2, cells with only 1 time step should be unassigned
        result = linking_overlap(features, mask, stubs=2)

        # Count cells with more than 1 occurrence
        cell_counts = result["cell"].value_counts()
        assert all(count >= 2 for count in cell_counts.values if count != -1)

    def test_cell_numbering(self, simple_2d_data):
        """Test that cell numbering starts from cell_number_start."""
        features, mask = simple_2d_data

        result = linking_overlap(features, mask, cell_number_start=10)

        # Cell IDs should start from 10
        assigned_cells = result.loc[result["cell"] != -1, "cell"].unique()
        if len(assigned_cells) > 0:
            assert assigned_cells.min() >= 10

    def test_unassigned_cell_value(self, non_overlapping_data):
        """Test that unassigned cells use correct value."""
        features, mask = non_overlapping_data

        result = linking_overlap(features, mask, cell_number_unassigned=-999)

        # Unassigned cells should have value -999
        assert (-999 in result["cell"].values) or all(result["cell"] >= 0)

    def test_time_cell_calculation(self, simple_2d_data):
        """Test that time_cell is calculated correctly."""
        features, mask = simple_2d_data

        result = linking_overlap(features, mask)

        # time_cell should be relative to cell start
        for cell_id in result["cell"].unique():
            if cell_id == -1:
                continue
            cell_mask = result["cell"] == cell_id
            cell_times = result.loc[cell_mask, "time_cell"].values

            # Should be monotonically increasing
            time_diffs = np.diff(cell_times.astype("timedelta64[s]").astype(float))
            assert np.all(time_diffs > 0)

    def test_minimum_overlap_threshold(self, simple_2d_data):
        """Test minimum_overlap parameter."""
        features, mask = simple_2d_data

        # With higher minimum overlap, features should not link
        result = linking_overlap(features, mask, minimum_overlap=2)

        assert np.all(result.cell == -1)

    def test_minimum_relative_overlap(self, simple_2d_data):
        """Test minimum_relative_overlap parameter."""
        features, mask = simple_2d_data

        # Test with different relative overlap thresholds
        result_high = linking_overlap(features, mask, minimum_relative_overlap=0.26)
        result_low = linking_overlap(features, mask, minimum_relative_overlap=0.25)

        # With higher minimum relative overlap, features should not link
        assert np.all(result_high.cell == -1)

        # With lower minimum relative overlap, features should not link
        assert np.all(result_low.cell > 0)
        assert (
            result_low.cell[result_low.feature == 1].values
            == result_low.cell[result_low.feature == 3].values
        )
        assert (
            result_low.cell[result_low.feature == 2].values
            == result_low.cell[result_low.feature == 4].values
        )

    def test_translate_method_constant(self, simple_2d_data):
        """Test constant velocity translation method."""
        features, mask = simple_2d_data

        # test with zero velocity and minimum overlap, shouldn't link
        result = linking_overlap(
            features,
            mask,
            translate_method="constant",
            velocity_constant=np.array([0, 0]),
            minimum_overlap=2,
        )
        assert np.all(result.cell == -1)

        # test with non-zero velocity and minimum overlap, should link
        result = linking_overlap(
            features,
            mask,
            translate_method="constant",
            velocity_constant=np.array([1.0 / 3600, 1.0 / 3600]),
            minimum_overlap=2,
        )
        assert np.all(result.cell > 0)
        assert (
            result.cell[result.feature == 1].values
            == result.cell[result.feature == 3].values
        )
        assert (
            result.cell[result.feature == 2].values
            == result.cell[result.feature == 4].values
        )

    def test_translate_method_drift(self, simple_2d_data):
        """Test drift translation method."""
        features, mask = simple_2d_data

        result = linking_overlap(
            features,
            mask,
            translate_method="drift",
        )

        assert "cell" in result.columns
        assert len(result) == len(features)

    def test_translate_method_predict(self, simple_2d_data):
        """Test predict translation method."""
        features, mask = simple_2d_data

        result = linking_overlap(
            features,
            mask,
            translate_method="predict",
            velocity_method="constant",
            velocity_constant=np.array([1.0, 1.0]),
        )

        assert "cell" in result.columns
        assert len(result) == len(features)

    def test_output_preserves_input_columns(self, simple_2d_data):
        """Test that output preserves input feature columns."""
        features, mask = simple_2d_data

        result = linking_overlap(features, mask)

        # Original columns should be preserved
        for col in features.columns:
            assert col in result.columns

    def test_output_dataframe_structure(self, simple_2d_data):
        """Test output DataFrame structure."""
        features, mask = simple_2d_data

        result = linking_overlap(features, mask)

        # Check types and structure
        assert isinstance(result, pd.DataFrame)
        assert result["cell"].dtype in [np.int32, np.int64, int]
        assert pd.api.types.is_timedelta64_dtype(result["time_cell"])

    def test_pbc_flag_none(self, simple_2d_data):
        """Test with no periodic boundary conditions."""
        features, mask = simple_2d_data

        result = linking_overlap(features, mask, PBC_flag="none")

        assert len(result) == len(features)

    def test_pbc_flag_hdim_1(self, simple_2d_data):
        """Test with periodic boundary conditions in hdim_1."""
        features, mask = simple_2d_data

        result = linking_overlap(features, mask, PBC_flag="hdim_1")

        assert len(result) == len(features)

    def test_pbc_flag_both(self, simple_2d_data):
        """Test with periodic boundary conditions in both dimensions."""
        features, mask = simple_2d_data

        result = linking_overlap(features, mask, PBC_flag="both")

        assert len(result) == len(features)

    def test_empty_features(self):
        """Test with empty features dataframe."""
        features = pd.DataFrame(
            {
                "feature": [],
                "time": [],
                "hdim_1": [],
                "hdim_2": [],
            }
        )

        mask = xr.DataArray(
            np.zeros((1, 10, 10), dtype=int),
            coords={
                "time": pd.date_range("2020-01-01", periods=1),
                "hdim_1": np.arange(10),
                "hdim_2": np.arange(10),
            },
            dims=["time", "hdim_1", "hdim_2"],
        )

        result = linking_overlap(features, mask)

        assert len(result) == 0
        assert "cell" in result.columns

    def test_single_timestep(self):
        """Test with single time step."""
        time = [pd.Timestamp("2020-01-01")]
        hdim_1 = np.arange(10)
        hdim_2 = np.arange(10)

        mask_data = np.zeros((1, 10, 10), dtype=int)
        mask_data[0, 2:4, 2:4] = 1

        mask = xr.DataArray(
            mask_data,
            coords={"time": time, "hdim_1": hdim_1, "hdim_2": hdim_2},
            dims=["time", "hdim_1", "hdim_2"],
        )

        features = pd.DataFrame(
            {
                "feature": [1],
                "time": time,
                "hdim_1": [3.0],
                "hdim_2": [3.0],
            }
        )

        result = linking_overlap(features, mask)

        assert len(result) == 1
        assert "cell" in result.columns

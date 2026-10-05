import numpy as np
import pandas as pd
import xarray as xr
import pytest

import tobac.feature_detection as feat_detection
from tobac.utils.features_to_field import features_to_interest_field


def make_template():
    """
    Create a template interest field with time, x, and y dimensions.
    """
    time = pd.date_range("2000-01-01", periods=5, freq="1h")
    x = np.arange(50)
    y = np.arange(60)

    data = xr.DataArray(
        np.zeros((len(time), len(x), len(y))),
        dims=("time", "x", "y"),
        coords={
            "time": time,
            "x": x,
            "y": y,
        },
        name="interest",
    )
    return data


def test_empty_features_returns_zero_field():
    """
    Test that empty features return an output field filled with zeros.
    """
    template = make_template()

    features = pd.DataFrame(columns=["frame", "hdim_1", "hdim_2", "threshold_value"])

    out = features_to_interest_field(
        features,
        template,
        position_mode="hdim",
        amp_from="threshold_value",
    )

    assert out.shape == template.shape
    assert float(out.max()) == 0.0
    assert float(out.min()) == 0.0


def test_amplitude_scaling_from_threshold():
    """
    Test that the output amplitude is scaled correctly from threshold_value.
    """
    template = make_template()

    features = pd.DataFrame(
        [
            {
                "frame": 0,
                "hdim_1": 20.0,
                "hdim_2": 30.0,
                "threshold_value": 0.3,
            }
        ]
    )

    out = features_to_interest_field(
        features,
        template,
        position_mode="hdim",
        thresh_from="threshold_value",
        amp_from=None,
        amp_factor=2.0,
        default_size=3.0,
        size_from=None,
        mode="max",
    )

    assert np.isclose(float(out.max()), 0.6, atol=1e-6)


def test_blob_center_is_at_feature_position():
    """
    Test that the blob maximum is located at the feature position.
    """
    template = make_template()

    features = pd.DataFrame(
        [
            {
                "frame": 2,
                "hdim_1": 10.0,
                "hdim_2": 15.0,
                "threshold_value": 1.0,
            }
        ]
    )

    out = features_to_interest_field(
        features,
        template,
        position_mode="hdim",
        thresh_from="threshold_value",
        amp_from=None,
        amp_factor=1.0,
        default_size=2.0,
        size_from=None,
        mode="max",
    )

    val = out.isel(time=2, x=10, y=15).item()
    assert np.isclose(val, 1.0, atol=1e-6)


def test_add_vs_max_overlap():
    """
    Test that overlapping blobs are combined differently for add and max modes.
    """
    template = make_template()

    features = pd.DataFrame(
        [
            {"frame": 0, "hdim_1": 25.0, "hdim_2": 30.0, "threshold_value": 1.0},
            {"frame": 0, "hdim_1": 25.0, "hdim_2": 30.0, "threshold_value": 1.0},
        ]
    )

    out_add = features_to_interest_field(
        features,
        template,
        position_mode="hdim",
        thresh_from="threshold_value",
        amp_from=None,
        amp_factor=1.0,
        default_size=2.0,
        size_from=None,
        mode="add",
    )

    out_max = features_to_interest_field(
        features,
        template,
        position_mode="hdim",
        thresh_from="threshold_value",
        amp_from=None,
        amp_factor=1.0,
        default_size=2.0,
        size_from=None,
        mode="max",
    )

    assert float(out_add.max()) > float(out_max.max())
    assert np.isclose(float(out_max.max()), 1.0, atol=1e-6)


def test_area_changes_blob_width():
    """
    Test that blob width is derived from area when size_from is set.
    """
    template = make_template()

    features = pd.DataFrame(
        [
            {
                "frame": 0,
                "hdim_1": 25.0,
                "hdim_2": 30.0,
                "threshold_value": 1.0,
                "num": 100.0,
            }
        ]
    )

    out = features_to_interest_field(
        features,
        template,
        position_mode="hdim",
        thresh_from="threshold_value",
        amp_from=None,
        amp_factor=2.0,
        default_size=1.0,
        size_from="num",
        mode="max",
    )

    center = out.isel(time=0, x=25, y=30).item()
    off = out.isel(time=0, x=25, y=35).item()

    assert off > 0.0
    assert off < center


def test_invalid_position_mode_raises():
    """
    Test that an invalid position_mode raises a ValueError.
    """
    template = make_template()

    features = pd.DataFrame(
        [{"frame": 0, "hdim_1": 10.0, "hdim_2": 10.0, "threshold_value": 1.0}]
    )

    with pytest.raises(ValueError):
        features_to_interest_field(
            features,
            template,
            position_mode="invalid",
        )


@pytest.mark.parametrize("blob", ["gaussian", "tophat", "cone"])
def test_reconstructed_field_recovers_feature_position_and_size(blob):
    """
    Round-trip test: reconstruct an interest field from input features and
    verify that the field maximum for each time step is located at (or very
    close to) the original feature position.

    Three well-separated features are placed at different time steps to avoid
    cross-contamination. The area (num) column drives the blob size so the
    test exercises the size-from-area code path.
    """
    template = make_template()

    input_features = pd.DataFrame(
        [
            {
                "frame": 0,
                "hdim_1": 10.0,
                "hdim_2": 15.0,
                "threshold_value": 1.0,
                "num": 50.0,
            },
            {
                "frame": 1,
                "hdim_1": 25.0,
                "hdim_2": 40.0,
                "threshold_value": 1.0,
                "num": 100.0,
            },
            {
                "frame": 2,
                "hdim_1": 35.0,
                "hdim_2": 20.0,
                "threshold_value": 1.0,
                "num": 200.0,
            },
        ]
    )

    out = features_to_interest_field(
        input_features,
        template,
        position_mode="hdim",
        thresh_from="threshold_value",
        amp_from=None,
        amp_factor=2.0,
        size_from="num",
        blob=blob,
        mode="max",
    )

    threshold = (
        1.0  # features have amp=2*threshold_value=2.0, so 0.5 is well below peak
    )

    out_features = feat_detection.feature_detection_multithreshold(
        out,
        dxy=1,
        threshold=[threshold],
    )

    distance_tolerance = 1.5  # pixel tolerance for position match
    relative_size_tolerance = 0.1  # relative tolerance for size match

    for _, row in input_features.iterrows():
        frame = row["frame"]
        true_x = row["hdim_1"]
        true_y = row["hdim_2"]

        candidates = out_features[out_features["frame"] == frame]

        if candidates.empty:
            raise AssertionError(f"No features detected in frame {frame}")

        candidates["distance"] = np.sqrt(
            (candidates["hdim_1"] - true_x) ** 2 + (candidates["hdim_2"] - true_y) ** 2
        )

        min_distance = candidates["distance"].min()

        assert (
            min_distance <= distance_tolerance
        ), f"Feature at frame {frame} has no match within {distance_tolerance} pixels (min distance: {min_distance})"

        # size check: the detected num (pixel count) of the closest feature
        # should be within a relative tolerance of the input num
        closest = candidates.loc[candidates["distance"].idxmin()]
        detected_num = float(closest["num"])
        input_num = float(row["num"])
        rel_err = abs(detected_num - input_num) / input_num
        assert rel_err <= relative_size_tolerance, (
            f"blob={blob}, frame={frame}: detected num={detected_num:.1f} "
            f"differs from input num={input_num:.1f} by {rel_err*100:.1f}%"
        )

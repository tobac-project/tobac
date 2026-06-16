import numpy as np
import pandas as pd
import xarray as xr
import pytest
import tobac


def test_find_mappings_for_chunk():
    """Assure that find_mappings_for_chunk returns the minimum parent label where child == index for each index"""
    chunk = {
        "parent": np.array([9, 3, 8, 2, 7], dtype=np.int64),
        "child": np.array([1, 1, 2, 2, 2], dtype=np.int64),
        "index": np.array([1, 2], dtype=np.int64),
    }

    out = tobac.multivariate.find_mappings_for_chunk(chunk, nan_val=99)

    assert (out == np.array([3, 2], dtype=np.int64)).all()


def test_drop_repeated_mappings():
    """Assure that drop_repeated_mappings maps each child label uniquely and to the minimum parent label, recording the laternatives"""

    child_to_parent, parent_to_child = tobac.multivariate.drop_repeated_mappings(
        [1, 1, 2, 2, 2],
        [8, 10, 7, 9, 6],
        nan_val=999,
    )

    assert child_to_parent == {1: 8, 2: 6}
    assert parent_to_child == {8: 8, 10: 8, 7: 6, 9: 6, 6: 6}


def make_sample_tracks():
    parent_mask = xr.DataArray(
        np.array(
            [
                [
                    [3, 3, 3, 0, 0, 0, 0, 0],
                    [3, 3, 3, 0, 0, 0, 0, 0],
                    [3, 3, 3, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 8, 8, 8],
                    [0, 0, 0, 0, 0, 8, 8, 8],
                    [0, 0, 0, 0, 0, 8, 8, 8],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 3, 3, 3, 0, 0, 0, 0],
                    [0, 3, 3, 3, 0, 0, 0, 0],
                    [0, 3, 3, 3, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 8, 8, 8, 0],
                    [0, 0, 0, 0, 8, 8, 8, 0],
                    [0, 0, 0, 0, 8, 8, 8, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 3, 3, 3, 0, 0, 0],
                    [0, 0, 3, 3, 3, 0, 0, 0],
                    [0, 0, 3, 3, 3, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 8, 8, 8, 0, 0],
                    [0, 0, 0, 8, 8, 8, 0, 0],
                    [0, 0, 0, 8, 8, 8, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 3, 3, 3, 0, 0],
                    [0, 0, 0, 3, 3, 3, 0, 0],
                    [0, 0, 0, 3, 3, 3, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 9, 9, 9, 0, 0, 0],
                    [0, 0, 9, 9, 9, 0, 0, 0],
                    [0, 0, 9, 9, 9, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
            ],
            dtype=np.int64,
        ),
        dims=("time", "x", "y"),
        coords={
            "time": np.arange(4),
            "x": np.arange(8),
            "y": np.arange(8),
        },
        name="cell",
    )
    child_mask = xr.DataArray(
        np.array(
            [
                [
                    [1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 2, 2, 0, 0, 0, 0],
                    [0, 0, 2, 2, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 3, 3, 0, 0],
                    [0, 0, 0, 0, 3, 3, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 2, 2, 0, 0, 0],
                    [0, 0, 0, 2, 2, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 3, 3, 0, 0],
                    [0, 0, 0, 0, 3, 3, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
            ],
            dtype=np.int64,
        ),
        dims=("time", "x", "y"),
        coords={
            "time": np.arange(4),
            "x": np.arange(8),
            "y": np.arange(8),
        },
        name="cell",
    )
    parent_tracks = pd.DataFrame({"cell": [3, 8, 9]})
    child_tracks = pd.DataFrame({"cell": [1, 2, 3]})
    return child_mask, child_tracks, parent_mask, parent_tracks


def test_get_multivariate_label_maps():
    """Assure that get_multivariate_label_maps works as expected"""

    child_mask, child_tracks, parent_mask, parent_tracks = make_sample_tracks()
    expected_child = pd.DataFrame({"cell": [1, 2, 3], "multivariate": [3, 3, 8]})
    expected_parent = pd.DataFrame({"cell": [3, 8, 9], "multivariate": [3, 8, 8]})
    child_out, parent_out = tobac.multivariate.get_multivariate_label_maps(
        child_mask,
        child_tracks,
        parent_mask,
        parent_tracks,
    )

    assert (expected_child == child_out).all().all()
    assert (expected_parent == parent_out).all().all()


def test_apply_multivariate_label_maps():
    """Assure that apply_multivariate_label_maps works as expected"""

    child_mask, child_tracks, parent_mask, parent_tracks = make_sample_tracks()
    child_label_maps = pd.DataFrame({"cell": [1, 2, 3], "multivariate": [3, 3, 8]})
    parent_label_maps = pd.DataFrame({"cell": [3, 8, 9], "multivariate": [3, 8, 8]})

    expected_parent = xr.DataArray(
        np.array(
            [
                [
                    [3, 3, 3, 0, 0, 0, 0, 0],
                    [3, 3, 3, 0, 0, 0, 0, 0],
                    [3, 3, 3, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 8, 8, 8],
                    [0, 0, 0, 0, 0, 8, 8, 8],
                    [0, 0, 0, 0, 0, 8, 8, 8],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 3, 3, 3, 0, 0, 0, 0],
                    [0, 3, 3, 3, 0, 0, 0, 0],
                    [0, 3, 3, 3, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 8, 8, 8, 0],
                    [0, 0, 0, 0, 8, 8, 8, 0],
                    [0, 0, 0, 0, 8, 8, 8, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 3, 3, 3, 0, 0, 0],
                    [0, 0, 3, 3, 3, 0, 0, 0],
                    [0, 0, 3, 3, 3, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 8, 8, 8, 0, 0],
                    [0, 0, 0, 8, 8, 8, 0, 0],
                    [0, 0, 0, 8, 8, 8, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 3, 3, 3, 0, 0],
                    [0, 0, 0, 3, 3, 3, 0, 0],
                    [0, 0, 0, 3, 3, 3, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 8, 8, 8, 0, 0, 0],
                    [0, 0, 8, 8, 8, 0, 0, 0],
                    [0, 0, 8, 8, 8, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
            ],
            dtype=np.int64,
        ),
        dims=("time", "x", "y"),
        coords={
            "time": np.arange(4),
            "x": np.arange(8),
            "y": np.arange(8),
        },
        name="multivariate",
    )

    expected_child = xr.DataArray(
        np.array(
            [
                [
                    [3, 3, 0, 0, 0, 0, 0, 0],
                    [3, 3, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 3, 3, 0, 0, 0, 0, 0],
                    [0, 3, 3, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 3, 3, 0, 0, 0, 0],
                    [0, 0, 3, 3, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 8, 8, 0, 0],
                    [0, 0, 0, 0, 8, 8, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 3, 3, 0, 0, 0],
                    [0, 0, 0, 3, 3, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 8, 8, 0, 0],
                    [0, 0, 0, 0, 8, 8, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
            ],
            dtype=np.int64,
        ),
        dims=("time", "x", "y"),
        coords={
            "time": np.arange(4),
            "x": np.arange(8),
            "y": np.arange(8),
        },
        name="multivariate",
    )

    child_out, parent_out = tobac.multivariate.apply_multivariate_label_maps(
        child_mask,
        child_label_maps,
        parent_mask,
        parent_label_maps,
    )

    assert (child_out == expected_child).all()
    assert (parent_out == expected_parent).all()

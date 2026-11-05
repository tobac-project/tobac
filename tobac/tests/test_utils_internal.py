import tobac.utils.internal as internal_utils
import tobac.testing as tbtest

import pytest
import numpy as np
import xarray as xr
import pandas.api.types as ptypes


@pytest.mark.parametrize(
    "dset_type, time_axis, vertical_axis, expected_out",
    [
        ("iris", 0, 1, (2, 3)),
        ("iris", -1, 0, (1, 2)),
        ("iris", 0, -1, (1, 2)),
        ("iris", 0, 2, (1, 3)),
        ("iris", 3, 0, (1, 2)),
        ("iris", 0, 3, (1, 2)),
        ("iris", 1, 2, (0, 3)),
        ("xarray", 0, 1, (2, 3)),
        ("xarray", 0, 2, (1, 3)),
        ("xarray", 3, 0, (1, 2)),
        ("xarray", 0, 3, (1, 2)),
        ("xarray", 1, 2, (0, 3)),
    ],
)
def test_find_hdim_axes_3D(dset_type, time_axis, vertical_axis, expected_out):
    """Tests tobac.utils.internal.file_hdim_axes_3D

    Parameters
    ----------
    dset_type: str{"xarray" or "iris"}
        type of the dataset to generate
    time_axis: int
        axis number of the time coordinate (or -1 to not have one)
    vertical_axis: int
        axis number of the vertical coordinate (or -1 to not have one)
    expected_out: tuple (int, int)
        expected output
    """
    ndims = 2 + (1 if time_axis >= 0 else 0) + (1 if vertical_axis >= 0 else 0)
    test_dset_size = [2] * ndims

    test_data = np.zeros(test_dset_size)

    dset_opts = {
        "in_arr": test_data,
        "data_type": dset_type,
    }
    if time_axis >= 0:
        dset_opts["time_dim_num"] = time_axis
    if vertical_axis >= 0:
        dset_opts["z_dim_num"] = vertical_axis
        dset_opts["z_dim_name"] = "altitude"

    y_set = False
    for dim_number in range(ndims):
        if time_axis != dim_number and vertical_axis != dim_number:
            if not y_set:
                dset_opts["y_dim_num"] = dim_number
                y_set = True
            else:
                dset_opts["x_dim_num"] = dim_number

    cube_test = tbtest.make_dataset_from_arr(**dset_opts)

    out_coords = internal_utils.find_hdim_axes_3D(cube_test)

    assert out_coords == expected_out


@pytest.mark.parametrize(
    "lat_name, lon_name, lat_name_test, lon_name_test, expected_result",
    [
        ("lat", "lon", None, None, ("lat", "lon")),
        ("lat", "long", None, None, ("lat", "long")),
        ("lat", "altitude", None, None, ("lat", None)),
        ("lat", "longitude", "lat", "longitude", ("lat", "longitude")),
    ],
)
def test_detect_latlon_coord_name(
    lat_name, lon_name, lat_name_test, lon_name_test, expected_result
):
    """Tests tobac.utils.internal.detect_latlon_coord_name"""

    in_arr = np.empty((50, 50))
    lat_vals = np.empty(50)
    lon_vals = np.empty(50)

    in_xr = xr.Dataset(
        {"data": ((lat_name, lon_name), in_arr)},
        coords={lat_name: lat_vals, lon_name: lon_vals},
    )
    out_lat_name, out_lon_name = internal_utils.detect_latlon_coord_name(
        in_xr["data"].to_iris(), lat_name_test, lon_name_test
    )
    assert out_lat_name == expected_result[0]
    assert out_lon_name == expected_result[1]


def test_make_empty_features_dataframe_2d():
    """
    Test that the empty 2D feature DataFrame has the correct columns and dtypes.
    """
    df = internal_utils.make_empty_features_dataframe(is_3D=False)

    expected_cols = [
        "frame",
        "idx",
        "hdim_1",
        "hdim_2",
        "num",
        "threshold_value",
        "feature",
        "time",
        "timestr",
        "y",
        "x",
        "latitude",
        "longitude",
    ]

    assert list(df.columns) == expected_cols
    assert df.empty

    assert ptypes.is_integer_dtype(df["frame"].dtype)
    assert ptypes.is_integer_dtype(df["idx"].dtype)
    assert ptypes.is_float_dtype(df["hdim_1"].dtype)
    assert ptypes.is_float_dtype(df["hdim_2"].dtype)
    assert ptypes.is_integer_dtype(df["num"].dtype)
    assert ptypes.is_integer_dtype(df["threshold_value"].dtype)
    assert ptypes.is_integer_dtype(df["feature"].dtype)

    assert ptypes.is_datetime64_ns_dtype(df["time"].dtype)
    assert df["timestr"].dtype == object

    assert ptypes.is_float_dtype(df["y"].dtype)
    assert ptypes.is_float_dtype(df["x"].dtype)
    assert ptypes.is_float_dtype(df["latitude"].dtype)
    assert ptypes.is_float_dtype(df["longitude"].dtype)


def test_make_empty_features_dataframe_3d_schema_and_dtypes():
    """
    Test that the empty 3D feature DataFrame has the correct columns and dtypes.
    """
    df = internal_utils.make_empty_features_dataframe(is_3D=True)

    expected_cols = [
        "frame",
        "idx",
        "vdim",
        "hdim_1",
        "hdim_2",
        "num",
        "threshold_value",
        "feature",
        "time",
        "timestr",
        "z",
        "y",
        "x",
        "latitude",
        "longitude",
    ]
    assert list(df.columns) == expected_cols
    assert df.empty

    assert ptypes.is_integer_dtype(df["frame"].dtype)
    assert ptypes.is_integer_dtype(df["idx"].dtype)
    assert ptypes.is_float_dtype(df["vdim"].dtype)
    assert ptypes.is_float_dtype(df["hdim_1"].dtype)
    assert ptypes.is_float_dtype(df["hdim_2"].dtype)
    assert ptypes.is_integer_dtype(df["num"].dtype)
    assert ptypes.is_integer_dtype(df["threshold_value"].dtype)
    assert ptypes.is_integer_dtype(df["feature"].dtype)

    assert ptypes.is_datetime64_ns_dtype(df["time"].dtype)
    assert df["timestr"].dtype == object

    assert ptypes.is_float_dtype(df["z"].dtype)
    assert ptypes.is_float_dtype(df["y"].dtype)
    assert ptypes.is_float_dtype(df["x"].dtype)
    assert ptypes.is_float_dtype(df["latitude"].dtype)
    assert ptypes.is_float_dtype(df["longitude"].dtype)


def _brute_force_pbc_labels(mask, PBC_flag, connectivity):
    """Label a 2D mask pixel by pixel, linking points across periodic walls in the
    same row/column, and opposite corners when doubly periodic."""
    import scipy.sparse
    import scipy.sparse.csgraph
    import skimage.measure

    in_domain = skimage.measure.label(mask, connectivity=connectivity)
    ny, nx = mask.shape
    idx = np.arange(ny * nx).reshape(ny, nx)
    links = [(idx[mask], idx[mask])]
    for label in np.unique(in_domain[in_domain > 0]):
        points = idx[in_domain == label]
        links.append((points, np.full_like(points, points[0])))
    if PBC_flag in ("hdim_1", "both"):
        links.append((idx[0, :], idx[-1, :]))
    if PBC_flag in ("hdim_2", "both"):
        links.append((idx[:, 0], idx[:, -1]))
    if PBC_flag == "both":
        links.append((idx[[0, 0], [0, -1]], idx[[-1, -1], [-1, 0]]))
    side_1 = np.concatenate([link[0] for link in links])
    side_2 = np.concatenate([link[1] for link in links])
    both_set = mask.ravel()[side_1] & mask.ravel()[side_2]
    graph = scipy.sparse.coo_matrix(
        (np.ones(both_set.sum()), (side_1[both_set], side_2[both_set])),
        shape=(ny * nx, ny * nx),
    )
    _, component = scipy.sparse.csgraph.connected_components(graph, directed=False)
    return np.where(mask, component.reshape(ny, nx) + 1, 0)


@pytest.mark.parametrize("PBC_flag", ["none", "hdim_1", "hdim_2", "both"])
@pytest.mark.parametrize("connectivity", [1, 2])
@pytest.mark.parametrize("seed", range(5))
def test_label_with_pbcs_matches_brute_force(PBC_flag, connectivity, seed):
    """
    Tests that label_with_pbcs merges labels across periodic boundaries, including
    chains of labels that are only connected through several boundary crossings.
    """
    from tobac.utils.internal.label_functions import label_with_pbcs

    rng = np.random.default_rng(seed)
    mask = rng.random((30, 40)) < 0.45

    labels, num_labels = label_with_pbcs(
        mask, PBC_flag=PBC_flag, connectivity=connectivity
    )
    labels = labels[0]
    expected = _brute_force_pbc_labels(mask, PBC_flag, connectivity)

    assert num_labels == len(np.unique(expected)) - 1
    assert np.array_equal(labels > 0, mask)
    # same partition of points into labels
    pairs = np.unique(np.stack([labels[mask], expected[mask]]), axis=1)
    assert len(np.unique(pairs[0])) == len(np.unique(pairs[1])) == pairs.shape[1]


def test_label_with_pbcs_chain():
    """
    Tests that labels linked through a chain of boundary crossings are merged, and
    that the merged label is the lowest label in the chain.
    """
    from tobac.utils.internal.label_functions import label_with_pbcs

    mask = np.zeros((10, 10), dtype=bool)
    # two separate areas on the west wall, joined by one area on the east wall
    mask[1:3, 0:2] = True
    mask[5:7, 0:2] = True
    mask[1:7, 8:10] = True
    # an area that doesn't touch any wall
    mask[4, 4] = True

    labels, num_labels = label_with_pbcs(mask, PBC_flag="hdim_2", connectivity=1)
    labels = labels[0]
    assert num_labels == 2
    assert np.all(labels[mask & (np.arange(10) != 4)[np.newaxis, :]] == 1)
    assert labels[4, 4] not in (0, 1)

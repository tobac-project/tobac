"""Containing methods to make simple sample data for testing.

"""

import datetime
import numpy as np
from xarray import DataArray
import pandas as pd
from collections import Counter
from .utils import periodic_boundaries as pbc_utils


def make_simple_sample_data_2D(data_type="iris"):
    """Create a simple dataset to use in tests.

    The grid has a grid spacing of 1km in both horizontal directions
    and 100 grid cells in x direction and 500 in y direction.
    Time resolution is 1 minute and the total length of the dataset is
    100 minutes around a abritraty date (2000-01-01 12:00).
    The longitude and latitude coordinates are added as 2D aux
    coordinates and arbitrary, but in realisitic range.
    The data contains a single blob travelling on a linear trajectory
    through the dataset for part of the time.

    Parameters
    ----------
    data_type : {'iris', 'xarray'}, optional
        Choose type of the dataset that will be produced.
        Default is 'iris'

    Returns
    -------
    sample_data : iris.cube.Cube or xarray.DataArray
    """

    from iris.cube import Cube
    from iris.coords import DimCoord, AuxCoord

    t_0 = datetime.datetime(2000, 1, 1, 12, 0, 0)

    x = np.arange(0, 100e3, 1000)
    y = np.arange(0, 50e3, 1000)
    t = t_0 + np.arange(0, 100, 1) * datetime.timedelta(minutes=1)
    xx, yy = np.meshgrid(x, y)

    t_temp = np.arange(0, 60, 1)
    track1_t = t_0 + t_temp * datetime.timedelta(minutes=1)
    x_0_1 = 10e3
    y_0_1 = 10e3
    track1_x = x_0_1 + 30 * t_temp * 60
    track1_y = y_0_1 + 14 * t_temp * 60
    track1_magnitude = 10 * np.ones(track1_x.shape)

    data = np.zeros((t.shape[0], y.shape[0], x.shape[0]))
    for i_t, t_i in enumerate(t):
        if np.any(t_i in track1_t):
            x_i = track1_x[track1_t == t_i]
            y_i = track1_y[track1_t == t_i]
            mag_i = track1_magnitude[track1_t == t_i]
            data[i_t] = data[i_t] + mag_i * np.exp(
                -np.power(xx - x_i, 2.0) / (2 * np.power(10e3, 2.0))
            ) * np.exp(-np.power(yy - y_i, 2.0) / (2 * np.power(10e3, 2.0)))

    t_start = datetime.datetime(1970, 1, 1, 0, 0)
    t_points = (t - t_start).astype("timedelta64[ms]").astype(int) / 1000
    t_coord = DimCoord(
        t_points,
        standard_name="time",
        var_name="time",
        units="seconds since 1970-01-01 00:00",
    )
    x_coord = DimCoord(
        x, standard_name="projection_x_coordinate", var_name="x", units="m"
    )
    y_coord = DimCoord(
        y, standard_name="projection_y_coordinate", var_name="y", units="m"
    )
    lat_coord = AuxCoord(
        24 + 1e-5 * xx, standard_name="latitude", var_name="latitude", units="degree"
    )
    lon_coord = AuxCoord(
        150 + 1e-5 * yy, standard_name="longitude", var_name="longitude", units="degree"
    )
    sample_data = Cube(
        data,
        dim_coords_and_dims=[(t_coord, 0), (y_coord, 1), (x_coord, 2)],
        aux_coords_and_dims=[(lat_coord, (1, 2)), (lon_coord, (1, 2))],
        var_name="w",
        units="m s-1",
    )

    if data_type == "xarray":
        sample_data = DataArray.from_iris(sample_data)

    return sample_data


def make_sample_data_2D_3blobs(data_type="iris"):
    """Create a simple dataset to use in tests.

    The grid has a grid spacing of 1km in both horizontal directions
    and 100 grid cells in x direction and 200 in y direction.
    Time resolution is 1 minute and the total length of the dataset is
    100 minutes around a arbitrary date (2000-01-01 12:00).
    The longitude and latitude coordinates are added as 2D aux
    coordinates and arbitrary, but in realisitic range.
    The data contains three individual blobs travelling on a linear
    trajectory through the dataset for part of the time.

    Parameters
    ----------
    data_type : {'iris', 'xarray'}, optional
        Choose type of the dataset that will be produced.
        Default is 'iris'

    Returns
    -------
    sample_data : iris.cube.Cube or xarray.DataArray
    """

    from iris.cube import Cube
    from iris.coords import DimCoord, AuxCoord

    t_0 = datetime.datetime(2000, 1, 1, 12, 0, 0)

    x = np.arange(0, 100e3, 1000)
    y = np.arange(0, 200e3, 1000)
    t = t_0 + np.arange(0, 100, 1) * datetime.timedelta(minutes=1)
    xx, yy = np.meshgrid(x, y)

    t_temp = np.arange(0, 60, 1)
    track1_t = t_0 + t_temp * datetime.timedelta(minutes=1)
    x_0_1 = 10e3
    y_0_1 = 10e3
    track1_x = x_0_1 + 30 * t_temp * 60
    track1_y = y_0_1 + 14 * t_temp * 60
    track1_magnitude = 10 * np.ones(track1_x.shape)

    t_temp = np.arange(0, 30, 1)
    track2_t = t_0 + (t_temp + 40) * datetime.timedelta(minutes=1)
    x_0_2 = 20e3
    y_0_2 = 10e3
    track2_x = x_0_2 + 24 * (t_temp * 60) ** 2 / 1000
    track2_y = y_0_2 + 12 * t_temp * 60
    track2_magnitude = 20 * np.ones(track2_x.shape)

    t_temp = np.arange(0, 20, 1)
    track3_t = t_0 + (t_temp + 50) * datetime.timedelta(minutes=1)
    x_0_3 = 70e3
    y_0_3 = 110e3
    track3_x = x_0_3 + 20 * (t_temp * 60) ** 2 / 1000
    track3_y = y_0_3 + 20 * t_temp * 60
    track3_magnitude = 15 * np.ones(track3_x.shape)

    data = np.zeros((t.shape[0], y.shape[0], x.shape[0]))
    for i_t, t_i in enumerate(t):
        if np.any(t_i in track1_t):
            x_i = track1_x[track1_t == t_i]
            y_i = track1_y[track1_t == t_i]
            mag_i = track1_magnitude[track1_t == t_i]
            data[i_t] = data[i_t] + mag_i * np.exp(
                -np.power(xx - x_i, 2.0) / (2 * np.power(10e3, 2.0))
            ) * np.exp(-np.power(yy - y_i, 2.0) / (2 * np.power(10e3, 2.0)))
        if np.any(t_i in track2_t):
            x_i = track2_x[track2_t == t_i]
            y_i = track2_y[track2_t == t_i]
            mag_i = track2_magnitude[track2_t == t_i]
            data[i_t] = data[i_t] + mag_i * np.exp(
                -np.power(xx - x_i, 2.0) / (2 * np.power(10e3, 2.0))
            ) * np.exp(-np.power(yy - y_i, 2.0) / (2 * np.power(10e3, 2.0)))
        if np.any(t_i in track3_t):
            x_i = track3_x[track3_t == t_i]
            y_i = track3_y[track3_t == t_i]
            mag_i = track3_magnitude[track3_t == t_i]
            data[i_t] = data[i_t] + mag_i * np.exp(
                -np.power(xx - x_i, 2.0) / (2 * np.power(10e3, 2.0))
            ) * np.exp(-np.power(yy - y_i, 2.0) / (2 * np.power(10e3, 2.0)))
    t_start = datetime.datetime(1970, 1, 1, 0, 0)
    t_points = (t - t_start).astype("timedelta64[ms]").astype(int) / 1000
    t_coord = DimCoord(
        t_points,
        standard_name="time",
        var_name="time",
        units="seconds since 1970-01-01 00:00",
    )
    x_coord = DimCoord(
        x, standard_name="projection_x_coordinate", var_name="x", units="m"
    )
    y_coord = DimCoord(
        y, standard_name="projection_y_coordinate", var_name="y", units="m"
    )
    lat_coord = AuxCoord(
        24 + 1e-5 * xx, standard_name="latitude", var_name="latitude", units="degree"
    )
    lon_coord = AuxCoord(
        150 + 1e-5 * yy, standard_name="longitude", var_name="longitude", units="degree"
    )
    sample_data = Cube(
        data,
        dim_coords_and_dims=[(t_coord, 0), (y_coord, 1), (x_coord, 2)],
        aux_coords_and_dims=[(lat_coord, (1, 2)), (lon_coord, (1, 2))],
        var_name="w",
        units="m s-1",
    )

    if data_type == "xarray":
        sample_data = DataArray.from_iris(sample_data)

    return sample_data


def make_sample_data_2D_3blobs_inv(data_type="iris"):
    """Create a version of the dataset with switched coordinates.

    Create a version of the dataset created in the function
    make_sample_cube_2D, but with switched coordinate order for the
    horizontal coordinates for tests to ensure that this does not
    affect the results.

    Parameters
    ----------
    data_type : {'iris', 'xarray'}, optional
        Choose type of the dataset that will be produced.
        Default is 'iris'

    Returns
    -------
    sample_data : iris.cube.Cube or xarray.DataArray
    """

    from iris.cube import Cube
    from iris.coords import DimCoord, AuxCoord

    t_0 = datetime.datetime(2000, 1, 1, 12, 0, 0)
    x = np.arange(0, 100e3, 1000)
    y = np.arange(0, 200e3, 1000)
    t = t_0 + np.arange(0, 100, 1) * datetime.timedelta(minutes=1)
    yy, xx = np.meshgrid(y, x)

    t_temp = np.arange(0, 60, 1)
    track1_t = t_0 + t_temp * datetime.timedelta(minutes=1)
    x_0_1 = 10e3
    y_0_1 = 10e3
    track1_x = x_0_1 + 30 * t_temp * 60
    track1_y = y_0_1 + 14 * t_temp * 60
    track1_magnitude = 10 * np.ones(track1_x.shape)

    t_temp = np.arange(0, 30, 1)
    track2_t = t_0 + (t_temp + 40) * datetime.timedelta(minutes=1)
    x_0_2 = 20e3
    y_0_2 = 10e3
    track2_x = x_0_2 + 24 * (t_temp * 60) ** 2 / 1000
    track2_y = y_0_2 + 12 * t_temp * 60
    track2_magnitude = 20 * np.ones(track2_x.shape)

    t_temp = np.arange(0, 20, 1)
    track3_t = t_0 + (t_temp + 50) * datetime.timedelta(minutes=1)
    x_0_3 = 70e3
    y_0_3 = 110e3
    track3_x = x_0_3 + 20 * (t_temp * 60) ** 2 / 1000
    track3_y = y_0_3 + 20 * t_temp * 60
    track3_magnitude = 15 * np.ones(track3_x.shape)

    data = np.zeros((t.shape[0], x.shape[0], y.shape[0]))
    for i_t, t_i in enumerate(t):
        if np.any(t_i in track1_t):
            x_i = track1_x[track1_t == t_i]
            y_i = track1_y[track1_t == t_i]
            mag_i = track1_magnitude[track1_t == t_i]
            data[i_t] = data[i_t] + mag_i * np.exp(
                -np.power(xx - x_i, 2.0) / (2 * np.power(10e3, 2.0))
            ) * np.exp(-np.power(yy - y_i, 2.0) / (2 * np.power(10e3, 2.0)))
        if np.any(t_i in track2_t):
            x_i = track2_x[track2_t == t_i]
            y_i = track2_y[track2_t == t_i]
            mag_i = track2_magnitude[track2_t == t_i]
            data[i_t] = data[i_t] + mag_i * np.exp(
                -np.power(xx - x_i, 2.0) / (2 * np.power(10e3, 2.0))
            ) * np.exp(-np.power(yy - y_i, 2.0) / (2 * np.power(10e3, 2.0)))
        if np.any(t_i in track3_t):
            x_i = track3_x[track3_t == t_i]
            y_i = track3_y[track3_t == t_i]
            mag_i = track3_magnitude[track3_t == t_i]
            data[i_t] = data[i_t] + mag_i * np.exp(
                -np.power(xx - x_i, 2.0) / (2 * np.power(10e3, 2.0))
            ) * np.exp(-np.power(yy - y_i, 2.0) / (2 * np.power(10e3, 2.0)))

    t_start = datetime.datetime(1970, 1, 1, 0, 0)
    t_points = (t - t_start).astype("timedelta64[ms]").astype(int) / 1000

    t_coord = DimCoord(
        t_points,
        standard_name="time",
        var_name="time",
        units="seconds since 1970-01-01 00:00",
    )
    x_coord = DimCoord(
        x, standard_name="projection_x_coordinate", var_name="x", units="m"
    )
    y_coord = DimCoord(
        y, standard_name="projection_y_coordinate", var_name="y", units="m"
    )
    lat_coord = AuxCoord(
        24 + 1e-5 * xx, standard_name="latitude", var_name="latitude", units="degree"
    )
    lon_coord = AuxCoord(
        150 + 1e-5 * yy, standard_name="longitude", var_name="longitude", units="degree"
    )

    sample_data = Cube(
        data,
        dim_coords_and_dims=[(t_coord, 0), (y_coord, 2), (x_coord, 1)],
        aux_coords_and_dims=[(lat_coord, (1, 2)), (lon_coord, (1, 2))],
        var_name="w",
        units="m s-1",
    )

    if data_type == "xarray":
        sample_data = DataArray.from_iris(sample_data)

    return sample_data


def make_sample_data_3D_3blobs(data_type="iris", invert_xy=False):
    """Create a simple dataset to use in tests.

    The grid has a grid spacing of 1km in both horizontal directions
    and 100 grid cells in x direction and 200 in y direction.
    Time resolution is 1 minute and the total length of the dataset is
    100 minutes around a abritraty date (2000-01-01 12:00).
    The longitude and latitude coordinates are added as 2D aux
    coordinates and arbitrary, but in realisitic range.
    The data contains three individual blobs travelling on a linear
    trajectory through the dataset for part of the time.

    Parameters
    ----------
    data_type : {'iris', 'xarray'}, optional
        Choose type of the dataset that will be produced.
        Default is 'iris'

    invert_xy : bool, optional
        Flag to determine wether to switch x and y coordinates
        Default is False

    Returns
    -------
    sample_data : iris.cube.Cube or xarray.DataArray
    """

    from iris.cube import Cube
    from iris.coords import DimCoord, AuxCoord

    t_0 = datetime.datetime(2000, 1, 1, 12, 0, 0)

    x = np.arange(0, 100e3, 1000)
    y = np.arange(0, 200e3, 1000)
    z = np.arange(0, 20e3, 1000)

    t = t_0 + np.arange(0, 50, 2) * datetime.timedelta(minutes=1)

    t_temp = np.arange(0, 60, 1)
    track1_t = t_0 + t_temp * datetime.timedelta(minutes=1)
    x_0_1 = 10e3
    y_0_1 = 10e3
    z_0_1 = 4e3
    track1_x = x_0_1 + 30 * t_temp * 60
    track1_y = y_0_1 + 14 * t_temp * 60
    track1_magnitude = 10 * np.ones(track1_x.shape)

    t_temp = np.arange(0, 30, 1)
    track2_t = t_0 + (t_temp + 40) * datetime.timedelta(minutes=1)
    x_0_2 = 20e3
    y_0_2 = 10e3
    z_0_2 = 6e3
    track2_x = x_0_2 + 24 * (t_temp * 60) ** 2 / 1000
    track2_y = y_0_2 + 12 * t_temp * 60
    track2_magnitude = 20 * np.ones(track2_x.shape)

    t_temp = np.arange(0, 20, 1)
    track3_t = t_0 + (t_temp + 50) * datetime.timedelta(minutes=1)
    x_0_3 = 70e3
    y_0_3 = 110e3
    z_0_3 = 8e3
    track3_x = x_0_3 + 20 * (t_temp * 60) ** 2 / 1000
    track3_y = y_0_3 + 20 * t_temp * 60
    track3_magnitude = 15 * np.ones(track3_x.shape)

    if invert_xy == False:
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
        y_dim = 2
        x_dim = 3
        data = np.zeros((t.shape[0], z.shape[0], y.shape[0], x.shape[0]))

    else:
        zz, xx, yy = np.meshgrid(z, x, y, indexing="ij")
        x_dim = 2
        y_dim = 3
        data = np.zeros((t.shape[0], z.shape[0], x.shape[0], y.shape[0]))

    for i_t, t_i in enumerate(t):
        if np.any(t_i in track1_t):
            x_i = track1_x[track1_t == t_i]
            y_i = track1_y[track1_t == t_i]
            z_i = z_0_1
            mag_i = track1_magnitude[track1_t == t_i]
            data[i_t] = data[i_t] + mag_i * np.exp(
                -np.power(xx - x_i, 2.0) / (2 * np.power(10e3, 2.0))
            ) * np.exp(-np.power(yy - y_i, 2.0) / (2 * np.power(10e3, 2.0))) * np.exp(
                -np.power(zz - z_i, 2.0) / (2 * np.power(5e3, 2.0))
            )
        if np.any(t_i in track2_t):
            x_i = track2_x[track2_t == t_i]
            y_i = track2_y[track2_t == t_i]
            z_i = z_0_2
            mag_i = track2_magnitude[track2_t == t_i]
            data[i_t] = data[i_t] + mag_i * np.exp(
                -np.power(xx - x_i, 2.0) / (2 * np.power(10e3, 2.0))
            ) * np.exp(-np.power(yy - y_i, 2.0) / (2 * np.power(10e3, 2.0))) * np.exp(
                -np.power(zz - z_i, 2.0) / (2 * np.power(5e3, 2.0))
            )

        if np.any(t_i in track3_t):
            x_i = track3_x[track3_t == t_i]
            y_i = track3_y[track3_t == t_i]
            z_i = z_0_3
            mag_i = track3_magnitude[track3_t == t_i]
            data[i_t] = data[i_t] + mag_i * np.exp(
                -np.power(xx - x_i, 2.0) / (2 * np.power(10e3, 2.0))
            ) * np.exp(-np.power(yy - y_i, 2.0) / (2 * np.power(10e3, 2.0))) * np.exp(
                -np.power(zz - z_i, 2.0) / (2 * np.power(5e3, 2.0))
            )

    t_start = datetime.datetime(1970, 1, 1, 0, 0)
    t_points = (t - t_start).astype("timedelta64[ms]").astype(int) / 1000
    t_coord = DimCoord(
        t_points,
        standard_name="time",
        var_name="time",
        units="seconds since 1970-01-01 00:00",
    )
    z_coord = DimCoord(z, standard_name="geopotential_height", var_name="z", units="m")
    y_coord = DimCoord(
        y, standard_name="projection_y_coordinate", var_name="y", units="m"
    )
    x_coord = DimCoord(
        x, standard_name="projection_x_coordinate", var_name="x", units="m"
    )
    lat_coord = AuxCoord(
        24 + 1e-5 * xx[0], standard_name="latitude", var_name="latitude", units="degree"
    )
    lon_coord = AuxCoord(
        150 + 1e-5 * yy[0],
        standard_name="longitude",
        var_name="longitude",
        units="degree",
    )
    sample_data = Cube(
        data,
        dim_coords_and_dims=[
            (t_coord, 0),
            (z_coord, 1),
            (y_coord, y_dim),
            (x_coord, x_dim),
        ],
        aux_coords_and_dims=[(lat_coord, (2, 3)), (lon_coord, (2, 3))],
        var_name="w",
        units="m s-1",
    )

    if data_type == "xarray":
        sample_data = DataArray.from_iris(sample_data)

    return sample_data


def make_dataset_from_arr(
    in_arr,
    data_type="xarray",
    time_dim_num=None,
    z_dim_num=None,
    z_dim_name="altitude",
    y_dim_num=0,
    x_dim_num=1,
):
    """Makes a dataset (xarray or iris) for feature detection/segmentation from
    a raw numpy/dask/etc. array.

    Parameters
    ----------
    in_arr: array-like
        The input array to convert to iris/xarray

    data_type: str('xarray' or 'iris'), optional
        Type of the dataset to return
        Default is 'xarray'

    time_dim_num: int or None, optional
        What axis is the time dimension on, None for a single timestep
        Default is None

    z_dim_num: int or None, optional
        What axis is the z dimension on, None for a 2D array
    z_dim_name: str
        What the z dimension name is named
    y_dim_num: int
        What axis is the y dimension on, typically 0 for a 2D array
        Default is 0

    x_dim_num: int, optional
        What axis is the x dimension on, typically 1 for a 2D array
        Default is 1

    Returns
    -------
    Iris or xarray dataset with everything we need for feature detection/tracking.

    """

    import xarray as xr
    import iris

    has_time = time_dim_num is not None

    is_3D = z_dim_num is not None
    output_arr = xr.DataArray(in_arr)
    if is_3D:
        z_max = in_arr.shape[z_dim_num]

    if has_time:
        time_min = datetime.datetime(2022, 1, 1)
        time_num = in_arr.shape[time_dim_num]

    if data_type == "xarray":
        return output_arr
    elif data_type == "iris":
        out_arr_iris = output_arr.to_iris()

        if is_3D:
            out_arr_iris.add_dim_coord(
                iris.coords.DimCoord(np.arange(0, z_max), standard_name=z_dim_name),
                z_dim_num,
            )
        if has_time:
            out_arr_iris.add_dim_coord(
                iris.coords.DimCoord(
                    pd.date_range(start=time_min, periods=time_num)
                    .values.astype("datetime64[s]")
                    .astype(int),
                    standard_name="time",
                    units="seconds since epoch",
                ),
                time_dim_num,
            )
        return out_arr_iris
    else:
        raise ValueError("data_type must be 'xarray' or 'iris'")


def make_feature_blob(
    in_arr,
    h1_loc,
    h2_loc,
    v_loc=None,
    h1_size=1,
    h2_size=1,
    v_size=1,
    shape="rectangle",
    amplitude=1,
    PBC_flag="none",
):
    """Function to make a defined "blob" in location (zloc, yloc, xloc) with
    user-specified shape and amplitude. Note that this function will
    round the size and locations to the nearest point within the array.

    Parameters
    ----------
    in_arr: array-like
        input array to add the "blob" to

    h1_loc: float
        Center hdim_1 location of the blob, required

    h2_loc: float
        Center hdim_2 location of the blob, required

    v_loc: float, optional
        Center vdim location of the blob, optional. If this is None, we assume that the
        dataset is 2D.
        Default is None

    h1_size: float, optional
        Size of the bubble in array coordinates in hdim_1
        Default is 1

    h2_size: float, optional
        Size of the bubble in array coordinates in hdim_2
        Default is 1

    v_size: float, optional
        Size of the bubble in array coordinates in vdim
        Default is 1

    shape: str('rectangle'), optional
        The shape of the blob that is added. For now, this is just rectangle
        'rectangle' adds a rectangular/rectangular prism bubble with constant amplitude `amplitude`.
        Default is "rectangle"

    amplitude: float, optional
        Maximum amplitude of the blob
        Default is 1

    PBC_flag : str('none', 'hdim_1', 'hdim_2', 'both')
        Sets whether to use periodic boundaries, and if so in which directions.
        'none' means that we do not have periodic boundaries
        'hdim_1' means that we are periodic along hdim1
        'hdim_2' means that we are periodic along hdim2
        'both' means that we are periodic along both horizontal dimensions

    Returns
    -------
    array-like
        An array with the same type as `in_arr` that has the blob added.
    """

    import xarray as xr

    # Check if z location is there and set our 3D-ness based on this.
    if v_loc is None:
        is_3D = False
        start_loc = 0
        start_v = None
        end_v = None

    else:
        is_3D = True
        start_loc = 1
        v_min = 0
        v_max = in_arr.shape[start_loc]
        start_v = int(np.ceil(max(v_min, v_loc - v_size / 2)))
        end_v = int(np.ceil(min(v_max - 1, v_loc + v_size / 2)))
        if v_size > v_max - v_min:
            raise ValueError("v_size larger than domain size")

    # Get min/max coordinates for hdim_1 and hdim_2
    # Min is inclusive, end is exclusive
    h1_min = 0
    h1_max = in_arr.shape[start_loc]

    h2_min = 0
    h2_max = in_arr.shape[start_loc + 1]

    if (h1_size > h1_max - h1_min) or (h2_size > h2_max - h2_min):
        raise ValueError("Horizontal size larger than domain size")

    # let's get start/end x/y/z
    start_h1 = int(np.ceil(h1_loc - h1_size / 2))
    end_h1 = int(np.ceil(h1_loc + h1_size / 2))

    start_h2 = int(np.ceil(h2_loc - h2_size / 2))
    end_h2 = int(np.ceil(h2_loc + h2_size / 2))

    # get the coordinate sets
    coords_to_fill = pbc_utils.get_pbc_coordinates(
        h1_min,
        h1_max,
        h2_min,
        h2_max,
        start_h1,
        end_h1,
        start_h2,
        end_h2,
        PBC_flag=PBC_flag,
    )
    if shape == "rectangle":
        for coord_box in coords_to_fill:
            in_arr = set_arr_2D_3D(
                in_arr,
                amplitude,
                coord_box[0],
                coord_box[1],
                coord_box[2],
                coord_box[3],
                start_v,
                end_v,
            )
        return in_arr


def set_arr_2D_3D(
    in_arr, value, start_h1, end_h1, start_h2, end_h2, start_v=None, end_v=None
):
    """Function to set part of `in_arr` for either 2D or 3D points to `value`.
    If `start_v` and `end_v` are not none, we assume that the array is 3D. If they
    are none, we will set the array as if it is a 2D array.

    Parameters
    ----------
    in_arr: array-like
        Array of values to set

    value: int, float, or array-like of size (end_v-start_v, end_h1-start_h1, end_h2-start_h2)
        The value to assign to in_arr. This will work to assign an array, but the array
        must have the same dimensions as the size specified in the function.

    start_h1: int
        Start index to set for hdim_1

    end_h1: int
        End index to set for hdim_1 (exclusive, so it acts like [start_h1:end_h1])

    start_h2: int
        Start index to set for hdim_2

    end_h2: int
        End index to set for hdim_2

    start_v: int, optional
        Start index to set for vdim
        Default is None

    end_v: int, optional
        End index to set for vdim
        Default is None

    Returns
    -------
    array-like
        in_arr with the new values set.
    """

    if start_v is not None and end_v is not None:
        in_arr[start_v:end_v, start_h1:end_h1, start_h2:end_h2] = value
    else:
        in_arr[start_h1:end_h1, start_h2:end_h2] = value

    return in_arr


def get_single_pbc_coordinate(
    h1_min, h1_max, h2_min, h2_max, h1_coord, h2_coord, PBC_flag="none"
):
    """Function to get the PBC-adjusted coordinate for an original non-PBC adjusted
    coordinate.

    Parameters
    ----------
    h1_min: int
        Minimum point in hdim_1
    h1_max: int
        Maximum point in hdim_1
    h2_min: int
        Minimum point in hdim_2
    h2_max: int
        Maximum point in hdim_2
    h1_coord: int
        hdim_1 query coordinate
    h2_coord: int
        hdim_2 query coordinate
    PBC_flag : str('none', 'hdim_1', 'hdim_2', 'both')
        Sets whether to use periodic boundaries, and if so in which directions.
        'none' means that we do not have periodic boundaries
        'hdim_1' means that we are periodic along hdim1
        'hdim_2' means that we are periodic along hdim2
        'both' means that we are periodic along both horizontal dimensions

    Returns
    -------
    tuple
        Returns a tuple of (hdim_1, hdim_2).

    Raises
    ------
    ValueError
        Raises a ValueError if the point is invalid (e.g., h1_coord < h1_min
        when PBC_flag = 'none')
    """
    # Avoiding duplicating code here, so throwing this into a loop.
    is_pbc = [False, False]
    if PBC_flag in ["hdim_1", "both"]:
        is_pbc[0] = True
    if PBC_flag in ["hdim_2", "both"]:
        is_pbc[1] = True

    out_coords = list()

    for point_query, dim_min, dim_max, dim_pbc in zip(
        [h1_coord, h2_coord], [h1_min, h2_min], [h1_max, h2_max], is_pbc
    ):
        dim_size = dim_max - dim_min
        if point_query >= dim_min and point_query < dim_max:
            out_coords.append(point_query)
        # off at least one domain
        elif point_query < dim_min:
            if not dim_pbc:
                raise ValueError("Point invalid!")
            if abs(point_query // dim_size) > 1:
                # we are more than one interval length away from the periodic boundary
                point_query = point_query + (
                    dim_size * (abs(point_query // dim_size) - 1)
                )

            out_coords.append(point_query + dim_size)
        elif point_query >= dim_max:
            if not dim_pbc:
                raise ValueError("Point invalid!")
            if abs(point_query // dim_size) > 1:
                # we are more than one interval length away from the periodic boundary
                point_query = point_query - (
                    dim_size * (abs(point_query // dim_size) - 1)
                )
            out_coords.append(point_query - dim_size)

    return tuple(out_coords)


def generate_single_feature(
    start_h1,
    start_h2,
    start_v=None,
    spd_h1=1,
    spd_h2=1,
    spd_v=1,
    min_h1=0,
    max_h1=None,
    min_h2=0,
    max_h2=None,
    num_frames=1,
    dt=datetime.timedelta(minutes=5),
    start_date=datetime.datetime(2022, 1, 1, 0),
    PBC_flag="none",
    frame_start=0,
    feature_num=1,
    feature_size=None,
    threshold_val=None,
):
    """Function to generate a dummy feature dataframe to test the tracking functionality

    Parameters
    ----------
    start_h1: float
        Starting point of the feature in hdim_1 space

    start_h2: float
        Starting point of the feature in hdim_2 space

    start_v: float, optional
        Starting point of the feature in vdim space (if 3D). For 2D, set to None.
        Default is None

    spd_h1: float, optional
        Speed (per frame) of the feature in hdim_1
        Default is 1

    spd_h2: float, optional
        Speed (per frame) of the feature in hdim_2
        Default is 1

    spd_v: float, optional
        Speed (per frame) of the feature in vdim
        Default is 1

    min_h1: int, optional
        Minimum value of hdim_1 allowed. If PBC_flag is not 'none', then
        this will be used to know when to wrap around periodic boundaries.
        If PBC_flag is 'none', features will disappear if they are above/below
        these bounds.
        Default is 0

    max_h1: int, optional
        Similar to min_h1, but the max value of hdim_1 allowed.
        Default is 1000

    min_h2: int, optional
        Similar to min_h1, but the minimum value of hdim_2 allowed.
        Default is 0

    max_h2: int, optional
        Similar to min_h1, but the maximum value of hdim_2 allowed.
        Default is 1000

    num_frames: int, optional
        Number of frames to generate
        Default is 1

    dt: datetime.timedelta, optional
        Difference in time between each frame
        Default is datetime.timedelta(minutes=5)

    start_date: datetime.datetime, optional
        Start datetime
        Default is datetime.datetime(2022, 1, 1, 0)

    PBC_flag : str('none', 'hdim_1', 'hdim_2', 'both')
        Sets whether to use periodic boundaries, and if so in which directions.
        'none' means that we do not have periodic boundaries
        'hdim_1' means that we are periodic along hdim1
        'hdim_2' means that we are periodic along hdim2
        'both' means that we are periodic along both horizontal dimensions
    frame_start: int
        Number to start the frame at
        Default is 1

    feature_num: int, optional
        What number to start the feature at
        Default is 1
    feature_size: int or None
        'num' column in output; feature size
        If None, doesn't set this column
    threshold_val: float or None
        Threshold value of this feature
    """

    if max_h1 is None or max_h2 is None:
        raise ValueError("Max coords must be specified.")

    out_list_of_dicts = list()
    curr_h1 = start_h1
    curr_h2 = start_h2
    curr_v = start_v
    curr_dt = start_date
    is_3D = not (start_v is None)
    for i in range(num_frames):
        curr_dict = dict()
        curr_h1, curr_h2 = get_single_pbc_coordinate(
            min_h1, max_h1, min_h2, max_h2, curr_h1, curr_h2, PBC_flag
        )
        curr_dict["hdim_1"] = curr_h1
        curr_dict["hdim_2"] = curr_h2
        curr_dict["frame"] = frame_start + i
        curr_dict["idx"] = 0
        if curr_v is not None:
            curr_dict["vdim"] = curr_v
            curr_v += spd_v
        curr_dict["time"] = curr_dt
        curr_dict["feature"] = feature_num + i
        if feature_size is not None:
            curr_dict["num"] = feature_size
        if threshold_val is not None:
            curr_dict["threshold_value"] = threshold_val
        curr_h1 += spd_h1
        curr_h2 += spd_h2
        curr_dt += dt
        out_list_of_dicts.append(curr_dict)

    return pd.DataFrame.from_dict(out_list_of_dicts)


def get_start_end_of_feat(center_point, size, axis_min, axis_max, is_pbc=False):
    """Gets the start and ending points for a feature given a size and PBC
    conditions

    Parameters
    ----------
    center_point: float
        The center point of the feature
    size: float
        The size of the feature in this dimension
    axis_min: int
        Minimum point on the axis (usually 0)
    axis_max: int
        Maximum point on the axis (exclusive). This is 1 after
        the last real point on the axis, such that axis_max - axis_min
        is the size of the axis
    is_pbc: bool
        True if we should give wrap around points, false if we shouldn't.

    Returns
    -------
    tuple (start_point, end_point)
    Note that if is_pbc is True, start_point can be less than axis_min and
    end_point can be greater than or equal to axis_max. This is designed to be used with
    ```get_pbc_coordinates```
    """
    import numpy as np

    min_pt = int(np.ceil(center_point - size / 2))
    max_pt = int(np.ceil(center_point + size / 2))
    # adjust points for boundaries, if needed.
    if min_pt < axis_min and not is_pbc:
        min_pt = axis_min
    if max_pt > axis_max and not is_pbc:
        max_pt = axis_max

    return (min_pt, max_pt)


def generate_grid_coords(min_max_coords, lengths):
    """Generates a grid of coordinates, such as fake lat/lons for testing.

    Parameters
    ----------
    min_max_coords: array-like, either length 2, length 4, or length 6.
        The minimum and maximum values in each dimension as:
        (min_dim1, max_dim1, min_dim2, max_dim2, min_dim3, max_dim3) to use
        all 3 dimensions. You can omit any dimensions that you aren't using.
    lengths: array-like, either length 1, 2, or 3.
        The lengths of values in each dimension. Length must equal 1/2 the length
        of min_max_coords.

    Returns
    -------
    1, 2, or 3 array-likes
        array-like of grid coordinates in the number of dimensions requested
        and with the number of arrays specified (meshed coordinates)

    """
    import numpy as np

    if len(min_max_coords) != len(lengths) * 2:
        raise ValueError(
            "The length of min_max_coords must be exactly 2 times"
            " the length of lengths."
        )

    if len(lengths) == 1:
        return np.mgrid[
            min_max_coords[0] : min_max_coords[1] : complex(imag=lengths[0])
        ]

    if len(lengths) == 2:
        return np.mgrid[
            min_max_coords[0] : min_max_coords[1] : complex(imag=lengths[0]),
            min_max_coords[2] : min_max_coords[3] : complex(imag=lengths[1]),
        ]

    if len(lengths) == 3:
        return np.mgrid[
            min_max_coords[0] : min_max_coords[1] : complex(imag=lengths[0]),
            min_max_coords[2] : min_max_coords[3] : complex(imag=lengths[1]),
            min_max_coords[4] : min_max_coords[5] : complex(imag=lengths[2]),
        ]


def lists_equal_without_order(a, b):
    """
    This will make sure the inner list contain the same,
    but doesn't account for duplicate groups.
    from: https://stackoverflow.com/questions/31501909/assert-list-of-list-equality-without-order-in-python/31502000
    """
    for l1 in a:
        check_counter = Counter(l1)
        if not any(Counter(l2) == check_counter for l2 in b):
            return False
    return True

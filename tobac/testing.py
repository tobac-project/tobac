import datetime
import numpy as np
from xarray import DataArray


def make_simple_sample_data_2D(data_type="iris"):
    """
    function creating a simple dataset to use in tests for tobac.
    The grid has a grid spacing of 1km in both horizontal directions and 100 grid cells in x direction and 500 in y direction.
    Time resolution is 1 minute and the total length of the dataset is 100 minutes around a abritraty date (2000-01-01 12:00).
    The longitude and latitude coordinates are added as 2D aux coordinates and arbitrary, but in realisitic range.
    The data contains a single blob travelling on a linear trajectory through the dataset for part of the time.

    :param data_type: 'iris' or 'xarray' to chose the type of dataset to produce
    :return: sample dataset as an Iris.Cube.cube or xarray.DataArray

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
    from iris.cube import Cube
    from iris.coords import DimCoord, AuxCoord

    """
    function creating a simple dataset to use in tests for tobac. 
    The grid has a grid spacing of 1km in both horizontal directions and 100 grid cells in x direction and 200 in y direction.
    Time resolution is 1 minute and the total length of the dataset is 100 minutes around a abritraty date (2000-01-01 12:00). 
    The longitude and latitude coordinates are added as 2D aux coordinates and arbitrary, but in realisitic range.
    The data contains a three individual blobs travelling on a linear trajectory through the dataset for part of the time.
    
    :param data_type: 'iris' or 'xarray' to chose the type of dataset to produce
    :return: sample dataset as an Iris.Cube.cube or xarray.DataArray

    """

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
    """
    function creating a version of the dataset created in the function make_sample_cube_2D, but with switched coordinate order for the horizontal coordinates
    for tests to ensure that this does not affect the results

    :param data_type: 'iris' or 'xarray' to chose the type of dataset to produce
    :return: sample dataset as an Iris.Cube.cube or xarray.DataArray

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
    from iris.cube import Cube
    from iris.coords import DimCoord, AuxCoord

    """
    function creating a simple dataset to use in tests for tobac. 
    The grid has a grid spacing of 1km in both horizontal directions and 100 grid cells in x direction and 200 in y direction.
    Time resolution is 1 minute and the total length of the dataset is 100 minutes around a abritraty date (2000-01-01 12:00). 
    The longitude and latitude coordinates are added as 2D aux coordinates and arbitrary, but in realisitic range.
    The data contains a three individual blobs travelling on a linear trajectory through the dataset for part of the time.
    
    :param data_type: 'iris' or 'xarray' to chose the type of dataset to produce
    :return: sample dataset as an Iris.Cube.cube or xarray.DataArray

    """

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

"""
Tests for the tobac.utils.general module.

"""


import numpy as np
from iris.coords import DimCoord
from iris.cube import Cube
from tobac.utils.general import get_spacings
import datetime
import cf_units
import pandas as pd
import pytest


# define hourly time range for random cube
timerange = pd.date_range(
    datetime.datetime(2021, 1, 1), datetime.datetime(2021, 12, 1), freq="h"
)
numtimes = np.array(())
for t in np.arange(len(timerange)):
    numtimes = np.append(
        numtimes,
        cf_units.date2num(
            timerange[t], "hours since 1970-01-01 00:00:00", cf_units.CALENDAR_STANDARD
        ),
    )
unit = cf_units.Unit("hours since 1970-01-01 00:00:00", cf_units.CALENDAR_STANDARD)
times = DimCoord(numtimes, standard_name="time", units=unit)

# lat, lon grid with 1 degree spacing
latitude = DimCoord(np.arange(-90, 90, 1), standard_name="latitude", units="degrees")

longitude = DimCoord(np.arange(0, 360, 1), standard_name="longitude", units="degrees")


# pseudo cartesian grid
y_coord = DimCoord(
    np.arange(0, 500 * 1000, 55 * 1000),
    standard_name="projection_y_coordinate",
    units="metres",
)

x_coord = DimCoord(
    np.arange(0, 1000 * 1000, 55 * 1000),
    standard_name="projection_x_coordinate",
    units="metres",
)


cube_lon_lat = Cube(
    np.zeros((times.shape[0], latitude.shape[0], longitude.shape[0]), np.float32),
    dim_coords_and_dims=[(times, 0), (latitude, 1), (longitude, 2)],
)

cube_cartesian = Cube(
    np.zeros((times.shape[0], y_coord.shape[0], x_coord.shape[0]), np.float32),
    dim_coords_and_dims=[(times, 0), (y_coord, 1), (x_coord, 2)],
)

# list of empty cubes with different grid features (can be extended with more variants)
CUBES = [cube_lon_lat, cube_cartesian]


@pytest.mark.parametrize("cube", CUBES)
def test_get_spacings(cube):
    """
    Assert that grid and time spacings are derived correctly.
    Test for cartesian and lat/lon grids.
    """
    dxy, dt = get_spacings(cube)
    assert dt == 3600
    assert int(dxy / 10000) == 5

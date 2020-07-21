import numpy as np
import warnings
import xarray as xr

from scipy import ndimage
from datetime import datetime

try:
    import pyproj
    _PYPROJ_AVAILABLE = True
except ImportError:
    _PYPROJ_AVAILABLE = False

def cartesian_to_geographic_aeqd(x, y, lon_0, lat_0, R=6370997.):
    """
    Azimuthal equidistant Cartesian to geographic coordinate transform.

    Transform a set of Cartesian/Cartographic coordinates (x, y) to
    geographic coordinate system (lat, lon) using a azimuthal equidistant
    map projection [1]_.

    .. math::

        lat = \\arcsin(\\cos(c) * \\sin(lat_0) +
                       (y * \\sin(c) * \\cos(lat_0) / \\rho))

        lon = lon_0 + \\arctan2(
            x * \\sin(c),
            \\rho * \\cos(lat_0) * \\cos(c) - y * \\sin(lat_0) * \\sin(c))

        \\rho = \\sqrt(x^2 + y^2)

        c = \\rho / R

    Where x, y are the Cartesian position from the center of projection;
    lat, lon the corresponding latitude and longitude; lat_0, lon_0 are the
    latitude and longitude of the center of the projection; R is the radius of
    the earth (defaults to ~6371 km). lon is adjusted to be between -180 and
    180.

    Parameters
    ----------
    x, y : array-like
        Cartesian coordinates in the same units as R, typically meters.
    lon_0, lat_0 : float
        Longitude and latitude, in degrees, of the center of the projection.
    R : float, optional
        Earth radius in the same units as x and y. The default value is in
        units of meters.

    Returns
    -------
    lon, lat : array
        Longitude and latitude of Cartesian coordinates in degrees.

    References
    ----------
    .. [1] Snyder, J. P. Map Projections--A Working Manual. U. S. Geological
        Survey Professional Paper 1395, 1987, pp. 191-202.

    """
    x = np.atleast_1d(np.asarray(x))
    y = np.atleast_1d(np.asarray(y))

    lat_0_rad = np.deg2rad(lat_0)
    lon_0_rad = np.deg2rad(lon_0)

    rho = np.sqrt(x*x + y*y)
    c = rho / R

    with warnings.catch_warnings():
        # division by zero may occur here but is properly addressed below so
        # the warnings can be ignored
        warnings.simplefilter("ignore", RuntimeWarning)
        lat_rad = np.arcsin(np.cos(c) * np.sin(lat_0_rad) +
                            y * np.sin(c) * np.cos(lat_0_rad) / rho)
    lat_deg = np.rad2deg(lat_rad)
    # fix cases where the distance from the center of the projection is zero
    lat_deg[rho == 0] = lat_0

    x1 = x * np.sin(c)
    x2 = rho*np.cos(lat_0_rad)*np.cos(c) - y*np.sin(lat_0_rad)*np.sin(c)
    lon_rad = lon_0_rad + np.arctan2(x1, x2)
    lon_deg = np.rad2deg(lon_rad)
    # Longitudes should be from -180 to 180 degrees
    lon_deg[lon_deg > 180] -= 360.
    lon_deg[lon_deg < -180] += 360.

    return lon_deg, lat_deg

def cartesian_to_geographic(grid_ds):
    """
    Cartesian to Geographic coordinate transform.

    Transform a set of Cartesian/Cartographic coordinates (x, y) to a
    geographic coordinate system (lat, lon) using pyproj or a build in
    Azimuthal equidistant projection.

    Parameters
    ----------
    grid_ds: xarray DataSet
        Cartesian coordinates in meters unless R is defined in different units
        in the projparams parameter.

    Returns
    -------
    lon, lat : array
        Longitude and latitude of the Cartesian coordinates in degrees.

    """
    projparams = grid_ds.ProjectionCoordinateSystem
    x = grid_ds.x.values
    y = grid_ds.y.values
    z = grid_ds.z.values
    z, y, x = np.meshgrid(z, y, x, indexing='ij')
    if projparams.attrs['grid_mapping_name'] == 'azimuthal_equidistant':
        # Use Py-ART's Azimuthal equidistance projection
        lat_0 = projparams.attrs['latitude_of_projection_origin']
        lon_0 = projparams.attrs['longitude_of_projection_origin']
        if 'semi_major_axis' in projparams:
            R = projparams.attrs['semi_major_axis']
            lon, lat = cartesian_to_geographic_aeqd(x, y, lon_0, lat_0, R)
        else:
            lon, lat = cartesian_to_geographic_aeqd(x, y, lon_0, lat_0)
    else:
        # Use pyproj for the projection
        # check that pyproj is available
        if not _PYPROJ_AVAILABLE:
            raise MissingOptionalDependency(
                "PyProj is required to use cartesian_to_geographic "
                "with a projection other than pyart_aeqd but it is not "
                "installed")
        proj = pyproj.Proj(projparams)
        lon, lat = proj(x, y, inverse=True)
    return lon, lat


def add_lat_lon_grid(grid_ds):
    lon, lat = cartesian_to_geographic(grid_ds)
    grid_ds["point_latitude"] = xr.DataArray(lat, dims=["z", "y", "x"])
    grid_ds["point_latitude"].attrs["long_name"] = "Latitude"
    grid_ds["point_latitude"].attrs["units"] = "degrees"
    grid_ds["point_longitude"] = xr.DataArray(lon, dims=["z", "y", "x"])
    grid_ds["point_longitude"].attrs["long_name"] = "Latitude"
    grid_ds["point_longitude"].attrs["units"] = "degrees"
    return grid_ds

def parse_grid_datetime(my_ds):
    year = my_ds['time'].dt.year
    month = my_ds['time'].dt.month
    day = my_ds['time'].dt.day
    hour = my_ds['time'].dt.hour
    minute = my_ds['time'].dt.minute
    second = my_ds['time'].dt.second
    return datetime(year=year, month=month, day=day,
                    hour=hour, minute=minute, second=second)


def get_vert_projection(grid, thresh=40):
    """ Returns boolean vertical projection from grid. """
    return np.any(grid > thresh, axis=0)


def get_filtered_frame(grid, min_size, thresh):
    """ Returns a labeled frame from gridded radar data. Smaller objects
    are removed and the rest are labeled. """
    if len(grid.shape) == 3:
        echo_height = get_vert_projection(grid, thresh)
    else:
        echo_height = grid > thresh
    labeled_echo = ndimage.label(echo_height)[0]
    frame = clear_small_echoes(labeled_echo, min_size)
    return frame


def clear_small_echoes(label_image, min_size):
    """ Takes in binary image and clears objects less than min_size. """
    flat_image = label_image.flatten()
    flat_image = flat_image[flat_image > 0]
    unique_elements, size_table = np.unique(flat_image, return_counts=True)
    small_objects = unique_elements[size_table < min_size]

    for obj in small_objects:
        label_image[label_image == obj] = 0
    label_image = ndimage.label(label_image)
    return label_image[0]


def get_grid_alt(grid_z, alt_meters=1500):
    """ Returns z-index closest to alt_meters. """
    return np.argmin(np.abs(grid_z - alt_meters))


def extract_grid_data(grid_obj, field, grid_size, params):
    """ Returns filtered grid frame and raw grid slice at global shift
    altitude. """
    min_size = params['MIN_SIZE'] / np.prod(grid_size[1:]/1000)
    masked = grid_obj.variables[field].fillna(0).values
    gs_alt = params['GS_ALT']
    raw = masked[get_grid_alt(grid_obj.z.values, gs_alt), :, :]
    frame = get_filtered_frame(masked, min_size, params['FIELD_THRESH'])
    return raw, frame


def get_grid_size(grid_obj):
    z_len = grid_obj.z.values[-1] - grid_obj.z.values[0]
    x_len = grid_obj.x.values[-1] - grid_obj.x.values[0]
    y_len = grid_obj.y.values[-1] - grid_obj.y.values[0]
    z_size = z_len / (grid_obj.z.values.shape[0] - 1)
    x_size = x_len / (grid_obj.x.values.shape[0] - 1)
    y_size = y_len / (grid_obj.y.values.shape[0] - 1)
    return np.array([z_size, y_size, x_size])


def extract_grid_data_2d(grid_obj, field, params):
    grid_size = np.array(grid_obj[field].values.shape)
    min_size = params['MIN_SIZE']
    masked = grid_obj.variables[field].values
    masked = masked > params['FIELD_THRESH']
    print(np.sum(masked))
    raw = masked
    labeled_echo = ndimage.label(masked)[0]
    frame = clear_small_echoes(labeled_echo, min_size)
    return raw, frame    


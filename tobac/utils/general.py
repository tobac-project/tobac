"""Provide essential methods.

"""
import logging

from .convert import xarray_to_iris, iris_to_xarray


@xarray_to_iris
def add_coordinates(t, variable_cube, coord_interp_kind):
    """Add coordinates from the tracking cube to the trajectories.

    Coordinates: time, longitude&latitude, x&y dimensions.

    Parameters
    ----------
    t : pandas.DataFrame
        Trajectories/features from feature detection or linking step

    variable_cube : iris.cube.Cube
        Input data used for the tracking to transfer coodinate information to resulting DataFrame

    coord_interp_kind: str
        The kind of interpolation for coordinates.

    Returns
    -------
    t : pandas.DataFrame
        Trajectories with added coordinated.

    """

    import numpy as np
    from scipy.interpolate import interp2d, interp1d

    logging.debug("start adding coordinates from cube")

    # pull time as datetime object and timestr from input data and add it to DataFrame:
    t["time"] = None
    t["timestr"] = None

    logging.debug("adding time coordinate")

    time_in = variable_cube.coord("time")
    time_in_datetime = time_in.units.num2date(time_in.points)

    t["time"] = time_in_datetime[t["frame"]]
    t["timestr"] = [
        x.strftime("%Y-%m-%d %H:%M:%S") for x in time_in_datetime[t["frame"]]
    ]

    # Get list of all coordinates in input cube except for time (already treated):
    coord_names = [coord.name() for coord in variable_cube.coords()]
    coord_names.remove("time")

    logging.debug("time coordinate added")

    # chose right dimension for horizontal axis based on time dimension:
    ndim_time = variable_cube.coord_dims("time")[0]
    if ndim_time == 0:
        hdim_1 = 1
        hdim_2 = 2
    elif ndim_time == 1:
        hdim_1 = 0
        hdim_2 = 2
    elif ndim_time == 2:
        hdim_1 = 0
        hdim_2 = 1

    # create vectors to use to interpolate from pixels to coordinates
    dimvec_1 = np.arange(variable_cube.shape[hdim_1])
    dimvec_2 = np.arange(variable_cube.shape[hdim_2])

    # loop over coordinates in input data:
    for coord in coord_names:
        logging.debug("adding coord: " + coord)
        # interpolate 2D coordinates:
        if variable_cube.coord(coord).ndim == 1:
            # check the interpolation method is valid
            if coord_interp_kind not in ["linear", "nearest", "nearest-up",
                                         "zero", "slinear", "quadratic",
                                         "cubic", "previous", "next"]:
                raise ValueError(
                    "coord_interp_kind must be linear, nearest, nearest-up, \
                     zero, slinear, quadratic, \
                     cubic, previous, or next for 1D coord"
                )

            if variable_cube.coord_dims(coord) == (hdim_1,):
                f = interp1d(
                    dimvec_1,
                    variable_cube.coord(coord).points,
                    kind=coord_interp_kind,
                    fill_value="extrapolate",
                )
                coordinate_points = f(t["hdim_1"])

            if variable_cube.coord_dims(coord) == (hdim_2,):
                f = interp1d(
                    dimvec_2,
                    variable_cube.coord(coord).points,
                    kind=coord_interp_kind,
                    fill_value="extrapolate",
                )
                coordinate_points = f(t["hdim_2"])

        # interpolate 2D coordinates:
        elif variable_cube.coord(coord).ndim == 2:
            # check the interpolation method is valid
            if coord_interp_kind not in ["linear", "cubic", "quintic"]:
                raise ValueError(
                    "coord_interp_kind must be linear, cubic, or quintic for 2D coord"
                )

            if variable_cube.coord_dims(coord) == (hdim_1, hdim_2):
                f = interp2d(dimvec_2, dimvec_1, variable_cube.coord(coord).points, kind=coord_interp_kind)
                coordinate_points = [f(a, b) for a, b in zip(t["hdim_2"], t["hdim_1"])]

            if variable_cube.coord_dims(coord) == (hdim_2, hdim_1):
                f = interp2d(dimvec_1, dimvec_2, variable_cube.coord(coord).points, kind=coord_interp_kind)
                coordinate_points = [f(a, b) for a, b in zip(t["hdim_1"], t["hdim_2"])]

        # interpolate 3D coordinates:
        # mainly workaround for wrf latitude and longitude (to be fixed in future)

        elif variable_cube.coord(coord).ndim == 3:
            # check the interpolation method is valid
            if coord_interp_kind not in ["linear", "cubic", "quintic"]:
                raise ValueError(
                    "coord_interp_kind must be linear, cubic, or quintic for 3D coord"
                )

            if variable_cube.coord_dims(coord) == (ndim_time, hdim_1, hdim_2):
                f = interp2d(
                    dimvec_2, dimvec_1, variable_cube[0, :, :].coord(coord).points,
                    kind=coord_interp_kind
                )
                coordinate_points = [f(a, b) for a, b in zip(t["hdim_2"], t["hdim_1"])]

            if variable_cube.coord_dims(coord) == (ndim_time, hdim_2, hdim_1):
                f = interp2d(
                    dimvec_1, dimvec_2, variable_cube[0, :, :].coord(coord).points,
                    kind=coord_interp_kind
                )
                coordinate_points = [f(a, b) for a, b in zip(t["hdim_1"], t["hdim_2"])]

            if variable_cube.coord_dims(coord) == (hdim_1, ndim_time, hdim_2):
                f = interp2d(
                    dimvec_2, dimvec_1, variable_cube[:, 0, :].coord(coord).points,
                    kind=coord_interp_kind
                )
                coordinate_points = [f(a, b) for a, b in zip(t["hdim_2"], t["hdim_1"])]

            if variable_cube.coord_dims(coord) == (hdim_1, hdim_2, ndim_time):
                f = interp2d(
                    dimvec_2, dimvec_1, variable_cube[:, :, 0].coord(coord).points,
                    kind=coord_interp_kind
                )
                coordinate_points = [f(a, b) for a, b in zip(t["hdim_2"], t["hdim1"])]

            if variable_cube.coord_dims(coord) == (hdim_2, ndim_time, hdim_1):
                f = interp2d(
                    dimvec_1, dimvec_2, variable_cube[:, 0, :].coord(coord).points,
                    kind=coord_interp_kind
                )
                coordinate_points = [f(a, b) for a, b in zip(t["hdim_1"], t["hdim_2"])]

            if variable_cube.coord_dims(coord) == (hdim_2, hdim_1, ndim_time):
                f = interp2d(
                    dimvec_1, dimvec_2, variable_cube[:, :, 0].coord(coord).points,
                    kind=coord_interp_kind
                )
                coordinate_points = [f(a, b) for a, b in zip(t["hdim_1"], t["hdim_2"])]

        # write resulting array or list into DataFrame:
        t[coord] = coordinate_points

        logging.debug("added coord: " + coord)
    return t


@xarray_to_iris
def get_bounding_box(x, buffer=1):
    """Calculates the bounding box of a ndarray.

    https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array

    Parameters
    ----------
    x

    buffer : int, optional
        Default is 1.

    Returns
    -------
    bbox

    Notes
    -----
    unsure about anything
    """

    from numpy import delete, arange, diff, nonzero, array

    mask = x == 0

    bbox = []
    all_axis = arange(x.ndim)
    # loop over dimensions
    for kdim in all_axis:
        nk_dim = delete(all_axis, kdim)
        mask_i = mask.all(axis=tuple(nk_dim))
        dmask_i = diff(mask_i)
        idx_i = nonzero(dmask_i)[0]
        # for case where there is no value in idx_i
        if len(idx_i) == 0:
            idx_i = array([0, x.shape[kdim] - 1])
        # for case where there is only one value in idx_i
        elif len(idx_i) == 1:
            idx_i = array([idx_i, idx_i])
        # make sure there is two values in idx_i
        elif len(idx_i) > 2:
            idx_i = array([idx_i[0], idx_i[-1]])
        # caluclate min and max values for idx_i and append them to list
        idx_min = max(0, idx_i[0] + 1 - buffer)
        idx_max = min(x.shape[kdim] - 1, idx_i[1] + 1 + buffer)
        bbox.append([idx_min, idx_max])
    return bbox


@xarray_to_iris
def get_spacings(field_in, grid_spacing=None, time_spacing=None):
    """Determine spatial and temporal grid spacing of input data

    Parameters
    ----------
    field_in : iris.cube.Cube
        Input field where to get spacings.

    grid_spacing : float, optional
        Grid spacing in input data. Default is None.

    time_spacing : float, optional
        Time resolution of input data. Default is None.

    Returns
    -------
    dxy : float
        Grid spacing in metres

    dt : float
        Time resolution in seconds

    Raises
    ------
    ValueError
        If input_cube does not contail projection_x_coord and
        projection_y_coord or keyword argument grid_spacing.

    """

    import numpy as np
    from copy import deepcopy
    from math import cos, sin, asin, sqrt, radians

    # set horizontal grid spacing of input data
    # If cartesian x and y corrdinates are present, use these to determine dxy (vertical grid spacing used to transfer pixel distances to real distances):
    coord_names = [coord.name() for coord in field_in.coords()]

    if (
        "projection_x_coordinate" in coord_names
        and "projection_y_coordinate" in coord_names
    ) and (grid_spacing is None):
        x_coord = deepcopy(field_in.coord("projection_x_coordinate"))
        x_coord.convert_units("metre")
        dx = np.diff(field_in.coord("projection_y_coordinate")[0:2].points)[0]
        y_coord = deepcopy(field_in.coord("projection_y_coordinate"))
        y_coord.convert_units("metre")
        dy = np.diff(field_in.coord("projection_y_coordinate")[0:2].points)[0]
        dxy = 0.5 * (dx + dy)
    elif grid_spacing is not None:
        dxy = grid_spacing

    # estimate grid spacing from lats and lons using the haversine
    elif ("longitude" in coord_names and "latitude" in coord_names) and (
        grid_spacing is None
    ):
        # get min and max values of lats and lons
        lat1 = np.min(field_in.coord("latitude").points)
        lat2 = np.max(field_in.coord("latitude").points)
        # convert decimal degrees to radians
        lat1, lat2 = map(radians, [lat1, lat2])

        # loop through coords to check dimension order 
        for i, coord in enumerate(coord_names):
            if coord == 'latitude' and field_in.coord(coord).ndim == 2:
                lat_axis = i 
            elif coord == 'longitude' and field_in.coord(coord).ndim == 2:
                lon_axis = i 
            else:
                # for 1D lats and lons  
                lat_axis, lon_axis = -1, -1 
                
        # re-write to corresponding axis in 2D coords (needed if other dimensions precede lats and lons)
        if lon_axis > lat_axis:
            lat_axis, lon_axis  = -2,  -1 
        elif lon_axis < lat_axis:
            lat_axis, lon_axis = -1 , -2
        dlat = np.diff(field_in.coord("latitude").points, axis= lat_axis ).mean()
        dlon = np.diff(field_in.coord("longitude").points, axis = lon_axis).mean()
        # haversine formula 
        dlat, dlon =  map(radians, [dlon, dlat])
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        km = 6371 * c
        dxy = np.round(km * 1000)
    else:
        raise ValueError(
            "no information about grid spacing, need either input cube with projection_x_coord and projection_y_coord, latitude and longitude  or keyword argument grid_spacing"
        )

    # set horizontal grid spacing of input data
    if time_spacing is None:
        # get time resolution of input data from first to steps of input cube:
        time_coord = field_in.coord("time")
        dt = (
            time_coord.units.num2date(time_coord.points[1])
            - time_coord.units.num2date(time_coord.points[0])
        ).seconds
    elif time_spacing is not None:
        # use value of time_spacing for dt:
        dt = time_spacing
    return dxy, dt


def spectral_filtering(
    dxy, field_in, lambda_min, lambda_max, return_transfer_function=False
):
    """
    This function creates and applies a 2D transfer function that can be used as a bandpass filter to remove
    certain wavelengths of an atmospheric field (e.g. vorticity).

    Parameters:
    -----------

    dxy : float
        grid spacing in m


    field_in: numpy.array
        2D field with input data

    lambda_min: float
        minimum wavelength in km


    lambda_max: float
        maximum wavelength in km

    return_transfer_function: boolean, optional
        default: False. If set to True, then the 1D transfer function are returned.

    Returns:
    --------

    filtered_field: numpy.array
        spectrally filtered 2D field of data

    transfer_function: tuple
        Two 2D fields, where the first one corresponds to the wavelengths of the domain and the second one
        to the 2D transfer function of the bandpass filter. Only returned, if return_transfer_function is True.

    """
    import numpy as np
    from scipy import signal
    from scipy import fft

    # check if valid value for dxy is given
    if dxy <= 0:
        raise ValueError(
            "Invalid value for dxy. Please provide the grid spacing in meter."
        )

    # convert grid spacing to km to get same units as given wavelengths
    dxy = dxy / 1000

    # get number of grid cells in x and y direction
    Ni = field_in.shape[-2]
    Nj = field_in.shape[-1]
    # wavenumber space
    m, n = np.meshgrid(np.arange(Ni), np.arange(Nj), indexing="ij")

    # if domain is squared:
    if Ni == Nj:
        wavenumber = np.sqrt(m**2 + n**2)
        lambda_mn = (2 * Ni * (dx)) / wavenumber

    # if domain is a rectangle:

    # alpha is the normalized wavenumber in wavenumber space
    alpha = np.sqrt(m**2 / Nj**2 + n**2 / Ni**2)
    # compute wavelengths for target grid in km
    lambda_mn = 2 * dxy / alpha

    ############### create a 2D bandpass filter (butterworth) #######################
    b, a = signal.iirfilter(
        2,
        [1 / lambda_max, 1 / lambda_min],
        btype="band",
        ftype="butter",
        fs=1 / dxy,
        output="ba",
    )
    w, h = signal.freqz(b, a, 1 / lambda_mn.flatten(), fs=1 / dxy)
    transfer_function = np.reshape(abs(h), lambda_mn.shape)

    # 2-dimensional discrete cosine transformation to convert data to spectral space
    spectral = fft.dctn(field_in.data)
    filtered = spectral * transfer_function
    # inverse discrete cosine transformation
    filtered_field = fft.idctn(filtered)

    if return_transfer_function is True:
        return (lambda_mn, transfer_function), filtered_field
    else:
        return filtered_field


# @xarray_to_iris
# def constraint_cell(track,mask_cell,width=None,x=None,):
#     from iris import Constraint
#     import numpy as np
#
#     time_coord=mask_cell.coord('time')
#     time_units=time_coord.units
#
#     def time_condition(cell):
#         return time_units.num2date(track.head(n=1)['time']) <= cell <= time_units.num2date(track.tail(n=1)['time'])
#
#     constraint_time=Constraint(time=time_condition)
##     mask_cell_i=mask_cell.extract(constraint_time)
#     mask_cell_surface_i=mask_cell_surface.extract(constraint_time)
#
#     x_dim=mask_cell_surface_i.coord_dims('projection_x_coordinate')[0]
#     y_dim=mask_cell_surface_i.coord_dims('projection_y_coordinate')[0]
#     x_coord=mask_cell_surface_i.coord('projection_x_coordinate')
#     y_coord=mask_cell_surface_i.coord('projection_y_coordinate')
#
#     if (mask_cell_surface_i.core_data()>0).any():
#         box_mask_i=get_bounding_box(mask_cell_surface_i.core_data(),buffer=1)
#
#         box_mask=[[x_coord.points[box_mask_i[x_dim][0]],x_coord.points[box_mask_i[x_dim][1]]],
#                  [y_coord.points[box_mask_i[y_dim][0]],y_coord.points[box_mask_i[y_dim][1]]]]
#     else:
#         box_mask=[[np.nan,np.nan],[np.nan,np.nan]]
#
#         x_min=box_mask[0][0]
#         x_max=box_mask[0][1]
#         y_min=box_mask[1][0]
#         y_max=box_mask[1][1]
#     constraint_x=Constraint(projection_x_coordinate=lambda cell: int(x_min) < cell < int(x_max))
#     constraint_y=Constraint(projection_y_coordinate=lambda cell: int(y_min) < cell < int(y_max))
#
#     constraint=constraint_time & constraint_x & constraint_y
#     return constraint

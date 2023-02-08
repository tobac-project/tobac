"""General tobac utilities

"""
import copy
import logging
from . import internal as internal_utils
import numpy as np
import sklearn
import sklearn.neighbors


def add_coordinates(t, variable_cube):
    """Add coordinates from the input cube of the feature detection
    to the trajectories/features.

    Parameters
    ----------
    t : pandas.DataFrame
        Trajectories/features from feature detection or linking step.

    variable_cube : iris.cube.Cube
        Input data used for the tracking with coordinate information
        to transfer to the resulting DataFrame. Needs to contain the
        coordinate 'time'.

    Returns
    -------
    t : pandas.DataFrame
        Trajectories with added coordinates.

    """

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
            if variable_cube.coord_dims(coord) == (hdim_1,):
                f = interp1d(
                    dimvec_1,
                    variable_cube.coord(coord).points,
                    fill_value="extrapolate",
                )
                coordinate_points = f(t["hdim_1"])

            if variable_cube.coord_dims(coord) == (hdim_2,):
                f = interp1d(
                    dimvec_2,
                    variable_cube.coord(coord).points,
                    fill_value="extrapolate",
                )
                coordinate_points = f(t["hdim_2"])

        # interpolate 2D coordinates:
        elif variable_cube.coord(coord).ndim == 2:
            if variable_cube.coord_dims(coord) == (hdim_1, hdim_2):
                f = interp2d(dimvec_2, dimvec_1, variable_cube.coord(coord).points)
                coordinate_points = [f(a, b) for a, b in zip(t["hdim_2"], t["hdim_1"])]

            if variable_cube.coord_dims(coord) == (hdim_2, hdim_1):
                f = interp2d(dimvec_1, dimvec_2, variable_cube.coord(coord).points)
                coordinate_points = [f(a, b) for a, b in zip(t["hdim_1"], t["hdim_2"])]

        # interpolate 3D coordinates:
        # mainly workaround for wrf latitude and longitude (to be fixed in future)

        elif variable_cube.coord(coord).ndim == 3:
            if variable_cube.coord_dims(coord) == (ndim_time, hdim_1, hdim_2):
                f = interp2d(
                    dimvec_2, dimvec_1, variable_cube[0, :, :].coord(coord).points
                )
                coordinate_points = [f(a, b) for a, b in zip(t["hdim_2"], t["hdim_1"])]

            if variable_cube.coord_dims(coord) == (ndim_time, hdim_2, hdim_1):
                f = interp2d(
                    dimvec_1, dimvec_2, variable_cube[0, :, :].coord(coord).points
                )
                coordinate_points = [f(a, b) for a, b in zip(t["hdim_1"], t["hdim_2"])]

            if variable_cube.coord_dims(coord) == (hdim_1, ndim_time, hdim_2):
                f = interp2d(
                    dimvec_2, dimvec_1, variable_cube[:, 0, :].coord(coord).points
                )
                coordinate_points = [f(a, b) for a, b in zip(t["hdim_2"], t["hdim_1"])]

            if variable_cube.coord_dims(coord) == (hdim_1, hdim_2, ndim_time):
                f = interp2d(
                    dimvec_2, dimvec_1, variable_cube[:, :, 0].coord(coord).points
                )
                coordinate_points = [f(a, b) for a, b in zip(t["hdim_2"], t["hdim1"])]

            if variable_cube.coord_dims(coord) == (hdim_2, ndim_time, hdim_1):
                f = interp2d(
                    dimvec_1, dimvec_2, variable_cube[:, 0, :].coord(coord).points
                )
                coordinate_points = [f(a, b) for a, b in zip(t["hdim_1"], t["hdim_2"])]

            if variable_cube.coord_dims(coord) == (hdim_2, hdim_1, ndim_time):
                f = interp2d(
                    dimvec_1, dimvec_2, variable_cube[:, :, 0].coord(coord).points
                )
                coordinate_points = [f(a, b) for a, b in zip(t["hdim_1"], t["hdim_2"])]

        # write resulting array or list into DataFrame:
        t[coord] = coordinate_points

        logging.debug("added coord: " + coord)
    return t


def get_bounding_box(x, buffer=1):
    """Finds the bounding box of a ndarray, i.e. the smallest
    bounding rectangle for nonzero values as explained here:
    https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array

    Parameters
    ----------
    x : numpy.ndarray
        Array for which the bounding box is to be determined.

    buffer : int, optional
        Number to set a buffer between the nonzero values and
        the edges of the box. Default is 1.

    Returns
    -------
    bbox : list
        Dimensionwise list of the indices representing the edges
        of the bounding box.

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


def get_spacings(field_in, grid_spacing=None, time_spacing=None):
    """Determine spatial and temporal grid spacing of the
    input data.

    Parameters
    ----------
    field_in : iris.cube.Cube
        Input field where to get spacings.

    grid_spacing : float, optional
        Manually sets the grid spacing if specified.
        Default is None.

    time_spacing : float, optional
        Manually sets the time spacing if specified.
        Default is None.

    Returns
    -------
    dxy : float
        Grid spacing in metres.

    dt : float
        Time resolution in seconds.

    Raises
    ------
    ValueError
        If input_cube does not contain projection_x_coord and
        projection_y_coord or keyword argument grid_spacing.

    """

    from copy import deepcopy

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
    else:
        raise ValueError(
            "no information about grid spacing, need either input cube with projection_x_coord and projection_y_coord or keyword argument grid_spacing"
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
    """This function creates and applies a 2D transfer function that
    can be used as a bandpass filter to remove certain wavelengths
    of an atmospheric input field (e.g. vorticity, IVT, etc).

    Parameters:
    -----------
    dxy : float
        Grid spacing in m.

    field_in: numpy.array
        2D field with input data.

    lambda_min: float
        Minimum wavelength in m.

    lambda_max: float
        Maximum wavelength in m.

    return_transfer_function: boolean, optional
        default: False. If set to True, then the 2D transfer function and
        the corresponding wavelengths are returned.

    Returns:
    --------
    filtered_field: numpy.array
        Spectrally filtered 2D field of data (with same shape as input data).

    transfer_function: tuple
        Two 2D fields, where the first one corresponds to the wavelengths
        in the spectral space of the domain and the second one to the 2D
        transfer function of the bandpass filter. Only returned, if
        return_transfer_function is True.
    """

    from scipy import signal
    from scipy import fft

    # check if valid value for dxy is given
    if dxy <= 0:
        raise ValueError(
            "Invalid value for dxy. Please provide the grid spacing in meter."
        )

    # get number of grid cells in x and y direction
    Ni = field_in.shape[-2]
    Nj = field_in.shape[-1]
    # wavenumber space
    m, n = np.meshgrid(np.arange(Ni), np.arange(Nj), indexing="ij")

    # if domain is squared:
    if Ni == Nj:
        wavenumber = np.sqrt(m**2 + n**2)
        lambda_mn = (2 * Ni * (dxy)) / wavenumber
    else:
        # if domain is a rectangle:
        # alpha is the normalized wavenumber in wavenumber space
        alpha = np.sqrt(m**2 / Ni**2 + n**2 / Nj**2)
        # compute wavelengths for target grid in m
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
    # multiplication of spectral coefficients with transfer function
    filtered = spectral * transfer_function
    # inverse discrete cosine transformation to go back from spectral to original space
    filtered_field = fft.idctn(filtered)

    if return_transfer_function is True:
        return (lambda_mn, transfer_function), filtered_field
    else:
        return filtered_field


def combine_tobac_feats(list_of_feats, preserve_old_feat_nums=None):
    """Function to combine a list of tobac feature detection dataframes
    into one combined dataframe that can be used for tracking
    or segmentation.
    Parameters
    ----------
    list_of_feats: array-like of Pandas DataFrames
        A list of dataframes (generated, for example, by
        running feature detection on multiple nodes).
    preserve_old_feat_nums: str or None
        The column name to preserve old feature numbers in. If None, these
        old numbers will be deleted. Users may want to enable this feature
        if they have run segmentation with the separate dataframes and
        therefore old feature numbers.
    Returns
    -------
    pd.DataFrame
        One combined DataFrame.
    """
    import pandas as pd

    # first, let's just combine these.
    combined_df = pd.concat(list_of_feats)
    # Then, sort by time first, then by feature number
    combined_df = combined_df.sort_values(["time", "feature"])
    all_times = sorted(combined_df["time"].unique())
    # Loop through current times
    start_feat_num = combined_df["feature"].min()
    # Save the old feature numbers if requested.
    if preserve_old_feat_nums is not None:
        combined_df[preserve_old_feat_nums] = combined_df["feature"]

    for frame_num, curr_time in enumerate(all_times):
        # renumber the frame number
        combined_df.loc[combined_df["time"] == curr_time, "frame"] = frame_num
        # renumber the features
        curr_row_count = len(combined_df.loc[combined_df["time"] == curr_time])
        feat_num_arr = np.arange(start_feat_num, start_feat_num + curr_row_count)
        combined_df.loc[combined_df["time"] == curr_time, "feature"] = feat_num_arr
        start_feat_num = np.max(feat_num_arr) + 1

    combined_df = combined_df.reset_index(drop=True)
    return combined_df


@internal_utils.irispandas_to_xarray
def transform_feature_points(
    features,
    new_dataset,
    latitude_name="auto",
    longitude_name="auto",
):
    """Function to transform input feature dataset horizontal grid points to a different grid.
    The typical use case for this function is to transform detected features to perform
    segmentation on a different grid.

    The existing feature dataset must have some latitude/longitude coordinates associated
    with each feature, and the new_dataset must have latitude/longitude available with
    the same name.

    Parameters
    ----------
    features: pd.DataFrame
        Input feature dataframe
    new_dataset: iris.cube.Cube or xarray
        The dataset to transform the
    latitude_name: str
        The name of the latitude coordinate. If "auto", tries to auto-detect.
    longitude_name: str
        The name of the longitude coordinate. If "auto", tries to auto-detect.

    Returns
    -------
    transformed_features: pd.DataFrame
        A new feature dataframe, with the coordinates transformed to
        the new grid, suitable for use in segmentation

    """

    lat_coord, lon_coord = internal_utils.detect_latlon_coord_name(
        new_dataset, latitude_name=latitude_name, longitude_name=longitude_name
    )

    if lat_coord not in features or lon_coord not in features:
        raise ValueError("Cannot find latitude and/or longitude coordinate")

    lat_vals_new = new_dataset[lat_coord].values
    lon_vals_new = new_dataset[lon_coord].values

    if len(lat_vals_new.shape) != len(lon_vals_new.shape):
        raise ValueError(
            "Cannot work with lat/lon coordinates of unequal dimensionality"
        )

    # the lat/lons must be a 2D grid, so if they aren't, make them one.
    if len(lat_vals_new.shape) == 1:
        lat_vals_new, lon_vals_new = np.meshgrid(lat_vals_new, lon_vals_new)

    # we have to convert to radians because scikit-learn's haversine
    # requires that the input be in radians.
    flat_lats = np.deg2rad(lat_vals_new.flatten())
    flat_lons = np.deg2rad(lon_vals_new.flatten())
    ll_tree = sklearn.neighbors.BallTree(
        np.array([flat_lats, flat_lons]).T, metric="haversine"
    )
    new_h1 = list()
    new_h2 = list()
    # there is almost certainly room for speedup in here.
    for index in features["index"]:
        dist, closest_pt = ll_tree.query(
            [
                [
                    np.deg2rad(features["latitude"][index]),
                    np.deg2rad(features["longitude"][index]),
                ]
            ]
        )
        unraveled_h2, unraveled_h1 = np.unravel_index(
            closest_pt[0][0], np.shape(lat_vals_new)
        )

        new_h1.append(unraveled_h1)
        new_h2.append(unraveled_h2)
    ret_features = copy.deepcopy(features)
    ret_features["hdim_1"] = ("index", new_h1)
    ret_features["hdim_2"] = ("index", new_h2)

    return ret_features

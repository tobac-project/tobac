"""General tobac utilities

"""
import copy
import logging
from typing import Union

import iris.cube
from typing import Callable
import pandas as pd

from . import internal as internal_utils
import numpy as np
import sklearn
import sklearn.neighbors
import datetime
import xarray as xr
import warnings


def add_coordinates(
    t: pd.DataFrame, variable_cube: Union[xr.DataArray, iris.cube.Cube]
) -> pd.DataFrame:
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
    if isinstance(variable_cube, iris.cube.Cube):
        return internal_utils.iris_utils.add_coordinates(t, variable_cube)
    if isinstance(variable_cube, xr.DataArray):
        raise NotImplementedError("add_coordinates not implemented for xarray.")
    raise ValueError(
        "add_coordinates only supports xarray.DataArray and iris.cube.Cube"
    )


def add_coordinates_3D(
    t: pd.DataFrame,
    variable_cube: Union[xr.DataArray, iris.cube.Cube],
    vertical_coord: Union[str, int] = None,
    vertical_axis: Union[int, None] = None,
    assume_coords_fixed_in_time: bool = True,
):
    """Function adding coordinates from the tracking cube to the trajectories
        for the 3D case: time, longitude&latitude, x&y dimensions, and altitude

    Parameters
    ----------
    t:             pandas DataFrame
                   trajectories/features
    variable_cube: iris.cube.Cube
        Cube (usually the one you are tracking on) at least conaining the dimension of 'time'.
        Typically, 'longitude','latitude','x_projection_coordinate','y_projection_coordinate',
        and 'altitude' (if 3D) are the coordinates that we expect, although this function
        will happily interpolate along any dimension coordinates you give.
    vertical_coord: str or int
        Name or axis number of the vertical coordinate. If None, tries to auto-detect.
        If it is a string, it looks for the coordinate or the dimension name corresponding
        to the string. If it is an int, it assumes that it is the vertical axis.
        Note that if you only have a 2D or 3D coordinate for altitude, you must
        pass in an int.
    vertical_axis: int or None
        Axis number of the vertical.
    assume_coords_fixed_in_time: bool
        If true, it assumes that the coordinates are fixed in time, even if the
        coordinates say they vary in time. This is, by default, True, to preserve
        legacy functionality. If False, it assumes that if a coordinate says
        it varies in time, it takes the coordinate at its word.

    Returns
    -------
    pandas DataFrame
                   trajectories with added coordinates
    """
    if isinstance(variable_cube, iris.cube.Cube):
        return internal_utils.iris_utils.add_coordinates_3D(
            t, variable_cube, vertical_coord, vertical_axis, assume_coords_fixed_in_time
        )
    if isinstance(variable_cube, xr.DataArray):
        raise NotImplementedError("add_coordinates_3D not implemented for xarray.")
    raise ValueError(
        "add_coordinates_3D only supports xarray.DataArray and iris.cube.Cube"
    )


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
    """WARNING: This function has been deprecated and will be removed in a future
    release, please use 'combine_feature_dataframes' instead

    Function to combine a list of tobac feature detection dataframes
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
        One combined DataFrame."""
    import warnings

    warnings.warn(
        "This function has been deprecated and will be removed in a future release, please use 'combine_feature_dataframes' instead",
        DeprecationWarning,
    )

    return combine_feature_dataframes(
        list_of_feats, old_feature_column_name=preserve_old_feat_nums
    )


def combine_feature_dataframes(
    feature_df_list,
    renumber_features=True,
    old_feature_column_name=None,
    sort_features_by=None,
):
    """Function to combine a list of tobac feature detection dataframes
    into one combined dataframe that can be used for tracking
    or segmentation.
    Parameters
    ----------
    feature_df_list: array-like of Pandas DataFrames
        A list of dataframes (generated, for example, by
        running feature detection on multiple nodes).
    renumber_features: bool, optional (default: True)
        If true, features are renumber with contiguous integers. If false, the
        old feature numbers will be retained, but an exception will be raised if
        there are any non-unique feature numbers. If you have non-unique feature
        numbers and want to preserve them, use the old_feature_column_name to
        save the old feature numbers to under a different column name.
    old_feature_column_name: str or None, optional (default: None)
        The column name to preserve old feature numbers in. If None, these
        old numbers will be deleted. Users may want to enable this feature
        if they have run segmentation with the separate dataframes and
        therefore old feature numbers.
    sort_features_by: list, str or None, optional (default: None)
        The sorting order to pass to Dataframe.sort_values for the merged
        dataframe. If None, will default to ["frame", "idx"] if
        renumber_features is True, or "feature" if renumber_features is False.

    Returns
    -------
    pd.DataFrame
        One combined DataFrame.
    """
    import pandas as pd

    # first, let's just combine these.
    combined_df = pd.concat(feature_df_list)

    if not renumber_features and np.any(
        np.bincount(combined_df["feature"] + np.nanmin(combined_df["feature"])) > 1
    ):
        error = ValueError(
            "Non-unique feature values detected. Combining feature dataframes with original feature numbers cannot be performed because duplicate feature numbers exist, please use 'renumber_features=True'. If you would like to preserve the original feature numbers, please use the 'old_feature_column_name' keyword to define a new column for these values in the returned dataframe"
        )
        # error.add_note(
        #     "Combining feature dataframes with original feature numbers cannot be performed because duplicate feature numbers exist, please use 'renumber_features=True'"
        # )
        # error.add_note(
        #     "If you would like to preserve the original feature numbers, please use the 'old_feature_column_name' keyword to define a new column for these values in the returned dataframe"
        # )
        raise error

    if sort_features_by is None:
        if renumber_features:
            sort_features_by = ["frame", "idx"]
        else:
            sort_features_by = "feature"
    # # Then, sort by time first, then by feature number
    # combined_df = combined_df.sort_values(["time", "feature"])
    # Save the old feature numbers if requested.
    if old_feature_column_name is not None:
        combined_df[old_feature_column_name] = combined_df["feature"]
    # count_per_time = combined_feats.groupby('time')['index'].count()
    combined_df["frame"] = combined_df["time"].rank(method="dense").astype(int) - 1
    combined_sorted = combined_df.sort_values(sort_features_by, ignore_index=True)
    if renumber_features:
        combined_sorted["feature"] = np.arange(1, len(combined_sorted) + 1)
    combined_sorted = combined_sorted.reset_index(drop=True)
    return combined_sorted


def get_statistics(
    labels: np.ndarray[int],
    *fields: tuple[np.ndarray],
    features: pd.DataFrame,
    func_dict: dict[str, tuple[Callable]] = {"ncells": np.count_nonzero},
    index: None | list[int] = None,
    default: None | float = None,
    id_column: str = "feature",
) -> pd.DataFrame:
    """
    Get bulk statistics for objects (e.g. features or segmented features) given a labelled mask of the objects
    and any input field with the same dimensions.

    The statistics are added as a new column to the existing feature dataframe. Users can specify which statistics are computed by
    providing a dictionary with the column name of the metric and the respective function.

    Parameters
    ----------
    labels : np.ndarray[int]
        Mask with labels of each regions to apply function to (e.g. output of segmentation for a specific timestep)
    *fields : tuple[np.ndarray]
        Fields to give as arguments to each function call. Must have the same shape as labels.
    features: pd.DataFrame
        Dataframe with features or segmented features (output from feature detection or segmentation)
        can be for the specific timestep or for the whole dataset
    func_dict: dict[str, Callable], optional (default: {'ncells':np.count_nonzero})
        Dictionary with function(s) to apply over each region as values and the name of the respective statistics as keys
        default is to just count the number of cells associated with each feature and write it to the feature dataframe
    index: None | list[int], optional (default: None)
        list of indexes of regions in labels to apply function to. If None, will
            default to all integers between the minimum and the maximum value in labels
    default: None | float, optional (default: None)
        default value to return in a region that has no values
    id_column: str, optional (default: "feature")
       Name of the column in feature dataframe that contains IDs that match with the labels in mask. The default is the column "feature".

     Returns:
     -------
     features: pd.DataFrame
         Updated feature dataframe with bulk statistics for each feature saved in a new column
    """
    # raise error if mask and input data dimensions do not match
    for field in fields:
        if labels.shape != field.shape:
            raise ValueError("Input labels and field do not have the same shape")

    # mask must contain positive values to calculate statistics
    if labels[labels > 0].size > 0:

        if index is None:
            index = range(
                int(np.nanmin(labels[labels > 0])), int(np.nanmax(labels) + 1)
            )
        else:
            # get the statistics only for specified feature objects
            if np.max(index) > np.max(labels):
                raise ValueError("Index contains values that are not in labels!")

        # set negative markers to 0 as they are unsegmented
        labels[labels < 0] = 0
        bins = np.cumsum(np.bincount(labels.ravel()))
        argsorted = np.argsort(labels.ravel())

        # apply each function given per func_dict for the labeled regions sorted in ascending order
        for stats_name in func_dict.keys():
            # initiate new column in feature dataframe if it does not already exist
            if stats_name not in features.columns:
                features[stats_name] = None
                # if function is given as a tuple, take the input parameters provided
            if type(func_dict[stats_name]) is tuple:
                func = func_dict[stats_name][0]
                # check that key word arguments are provided as dictionary
                if not type(func_dict[stats_name][1]) is dict:
                    raise TypeError(
                        "Tuple must contain dictionary with key word arguments for function."
                    )
                else:
                    kwargs = func_dict[stats_name][1]
                    # default needs to be sequence when function output is array-like
                    output = func(np.random.rand(1, 10), **kwargs)
                    if hasattr(output, "__len__"):
                        default = np.full(output.shape, default)
                    stats = np.array(
                        [
                            func(
                                *[
                                    field.ravel()[argsorted[bins[i - 1] : bins[i]]]
                                    for field in fields
                                ],
                                **kwargs,
                            )
                            if bins[i] > bins[i - 1]
                            else default
                            for i in index
                        ]
                    )
            # otherwise apply function on region without any input parameter
            else:
                func = func_dict[stats_name]
                # default needs to be sequence when function output is array-like
                output = func(np.random.rand(1, 10))
                if hasattr(output, "__len__"):
                    default = np.full(output.shape, default)

                stats = np.array(
                    [
                        func(
                            *[
                                field.ravel()[argsorted[bins[i - 1] : bins[i]]]
                                for field in fields
                            ]
                        )
                        if bins[i] > bins[i - 1]
                        else default
                        for i in index
                    ]
                )

            # add results of computed statistics to feature dataframe with column name given per func_dict
            for idx, label in enumerate(np.unique(labels[labels > 0])):

                # test if values are scalars
                if not hasattr(stats[idx], "__len__"):
                    # if yes, we can just assign the value to the new column and row of the respective feature
                    features.loc[features[id_column] == label, stats_name] = stats[idx]
                    # if stats output is array-like it has to be added in a different way
                else:
                    df = pd.DataFrame({stats_name: [stats[idx]]})
                    # get row index rather than pd.Dataframe index value since we need to use .iloc indexing
                    row_idx = np.where(features[id_column] == label)[0]
                    features.iloc[
                        row_idx,
                        features.columns.get_loc(stats_name),
                    ] = df.apply(lambda r: tuple(r), axis=1)

    return features


@internal_utils.iris_to_xarray
def get_statistics_from_mask(
    segmentation_mask: xr.DataArray,
    *fields: xr.DataArray,
    features: pd.DataFrame,
    func_dict: dict[str, tuple[Callable]] = {"Mean": np.mean},
    index: None | list[int] = None,
    default: None | float = None,
    id_column: str = "feature",
) -> pd.DataFrame:
    """
    Derives bulk statistics for each object in the segmentation mask.


    Parameters:
    -----------
    segmentation_mask : xr.DataArray
        Segmentation mask output
    *fields : xr.DataArray[np.ndarray]
        Field(s) with input data. Needs to have the same dimensions as the segmentation mask.
    features: pd.DataFrame
        Dataframe with segmented features (output from feature detection or segmentation).
        Timesteps must not be exactly the same as in segmentation mask but all labels in the mask need to be present in the feature dataframe.
    func_dict: dict[str, Callable], optional (default: {'ncells':np.count_nonzero})
        Dictionary with function(s) to apply over each region as values and the name of the respective statistics as keys
        default is to just count the number of cells associated with each feature and write it to the feature dataframe
    index: None | list[int], optional (default: None)
        list of indexes of regions in labels to apply function to. If None, will
            default to all integers between 1 and the maximum value in labels
    default: None | float, optional (default: None)
        default value to return in a region that has no values
    id_column: str, optional (default: "feature")
       Name of the column in feature dataframe that contains IDs that match with the labels in mask. The default is the column "feature".


     Returns:
     -------
     features: pd.DataFrame
         Updated feature dataframe with bulk statistics for each feature saved in a new column
    """
    # check that mask and input data have the same dimensions
    for field in fields:
        if segmentation_mask.shape != field.shape:
            raise ValueError("Input labels and field do not have the same shape")

    # warning when feature labels are not unique in dataframe
    if not features.feature.is_unique:
        raise logging.warning(
            "Feature labels are not unique which may cause unexpected results for the computation of bulk statistics."
        )

    # get bulk statistics for each timestep
    for tt in pd.to_datetime(segmentation_mask.time):
        # select specific timestep
        segmentation_mask_t = segmentation_mask.sel(time=tt).data
        field_t = field.sel(time=tt).data

        # make sure that the labels in the segmentation mask exist in feature dataframe
        if (
            np.intersect1d(np.unique(segmentation_mask_t), features.feature).size
            > np.unique(segmentation_mask_t).size
        ):
            raise ValueError(
                "The labels of the segmentation mask and the feature dataframe do not seem to match. Please make sure you provide the correct input feature dataframe to calculate the bulk statistics. "
            )
        else:
            # make sure that features are not double-defined
            features = get_statistics(
                segmentation_mask_t,
                field_t,
                features=features,
                func_dict=func_dict,
                default=default,
                index=index,
                id_column=id_column,
            )

    return features


@internal_utils.irispandas_to_xarray
def transform_feature_points(
    features,
    new_dataset,
    latitude_name=None,
    longitude_name=None,
    altitude_name=None,
    max_time_away=None,
    max_space_away=None,
    max_vspace_away=None,
    warn_dropped_features=True,
):
    """Function to transform input feature dataset horizontal grid points to a different grid.
    The typical use case for this function is to transform detected features to perform
    segmentation on a different grid.

    The existing feature dataset must have some latitude/longitude coordinates associated
    with each feature, and the new_dataset must have latitude/longitude available with
    the same name. Note that due to xarray/iris incompatibilities, we suggest that the
    input coordinates match the standard_name from Iris.

    Parameters
    ----------
    features: pd.DataFrame
        Input feature dataframe
    new_dataset: iris.cube.Cube or xarray
        The dataset to transform the
    latitude_name: str
        The name of the latitude coordinate. If None, tries to auto-detect.
    longitude_name: str
        The name of the longitude coordinate. If None, tries to auto-detect.
    altitude_name: str
        The name of the altitude coordinate. If None, tries to auto-detect.
    max_time_away: datetime.timedelta
        The maximum time delta to associate feature points away from.
    max_space_away: float
        The maximum horizontal distance (in meters) to transform features to.
    max_vspace_away: float
        The maximum vertical distance (in meters) to transform features to.
    warn_dropped_features: bool
        Whether or not to print a warning message if one of the max_* options is
        going to result in features that are dropped.
    Returns
    -------
    transformed_features: pd.DataFrame
        A new feature dataframe, with the coordinates transformed to
        the new grid, suitable for use in segmentation

    """
    from .. import analysis as tb_analysis

    RADIUS_EARTH_M = 6371000
    is_3D = "vdim" in features
    if is_3D:
        vert_coord = internal_utils.find_vertical_axis_from_coord(
            new_dataset, altitude_name
        )

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
        lon_vals_new, lat_vals_new = np.meshgrid(lon_vals_new, lat_vals_new)

    # we have to convert to radians because scikit-learn's haversine
    # requires that the input be in radians.
    flat_lats = np.deg2rad(lat_vals_new.ravel())
    flat_lons = np.deg2rad(lon_vals_new.ravel())

    # we have to drop NaN values.
    either_nan = np.logical_or(np.isnan(flat_lats), np.isnan(flat_lons))
    # we need to remember where these values are in the array so that we can
    # appropriately unravel them.
    loc_arr_trimmed = np.where(np.logical_not(either_nan))[0]
    flat_lats_nona = flat_lats[~either_nan]
    flat_lons_nona = flat_lons[~either_nan]
    ll_tree = sklearn.neighbors.BallTree(
        np.array([flat_lats_nona, flat_lons_nona]).T, metric="haversine"
    )

    ret_features = copy.deepcopy(features)

    # there is almost certainly room for speedup in here.
    rad_lats = np.deg2rad(features[lat_coord])
    rad_lons = np.deg2rad(features[lon_coord])
    dists, closest_pts = ll_tree.query(np.column_stack((rad_lats, rad_lons)))
    unraveled_h1, unraveled_h2 = np.unravel_index(
        loc_arr_trimmed[closest_pts[:, 0]], np.shape(lat_vals_new)
    )

    ret_features["hdim_1"] = ("index", unraveled_h1)
    ret_features["hdim_2"] = ("index", unraveled_h2)

    # now interpolate vertical, if available.
    if is_3D and max_space_away is not None and max_vspace_away is not None:
        alt_tree = sklearn.neighbors.BallTree(
            new_dataset[vert_coord].values[:, np.newaxis]
        )
        alt_dists, closest_alt_pts = alt_tree.query(
            features[vert_coord].values[:, np.newaxis]
        )
        ret_features["vdim"] = ("index", closest_alt_pts[:, 0])

        dist_cond = xr.DataArray(
            np.logical_or(
                (dists[:, 0] * RADIUS_EARTH_M) < max_space_away,
                alt_dists[:, 0] < max_vspace_away,
            ),
            dims="index",
        )
    elif max_space_away is not None:
        dist_cond = xr.DataArray(
            (dists[:, 0] * RADIUS_EARTH_M) < max_space_away, dims="index"
        )

    if max_space_away is not None or max_vspace_away is not None:
        ret_features = ret_features.where(dist_cond, drop=True)

    # force times to match, where appropriate.
    if "time" in new_dataset.coords and max_time_away is not None:
        # this is necessary due to the iris/xarray/pandas weirdness that we have.
        old_feat_times = ret_features["time"].astype("datetime64[s]")
        new_dataset_times = new_dataset["time"].astype("datetime64[s]")
        closest_times = np.min(np.abs(old_feat_times - new_dataset_times), axis=1)
        closest_time_locs = np.abs(old_feat_times - new_dataset_times).argmin(axis=1)
        # force to seconds to deal with iris not accepting ms
        ret_features["time"] = new_dataset["time"][closest_time_locs].astype(
            "datetime64[s]"
        )
        ret_features = ret_features.where(
            closest_times < np.timedelta64(max_time_away), drop=True
        )

    if warn_dropped_features:
        returned_features = ret_features["feature"]
        all_features = features["feature"]
        removed_features = np.delete(
            all_features, np.where(np.any(all_features == returned_features))
        )
        warnings.warn(
            "Dropping feature numbers: " + str(removed_features.values), UserWarning
        )

    return ret_features


def standardize_track_dataset(TrackedFeatures, Mask, Projection=None):
    """
    CAUTION: this function is experimental. No data structures output are guaranteed to be supported in future versions of tobac.
    Combine a feature mask with the feature data table into a common dataset.
    returned by tobac.segmentation
    with the TrackedFeatures dataset returned by tobac.linking_trackpy.
    Also rename the variables to be more descriptive and comply with cf-tree.
    Convert the default cell parent ID  to an integer table.
    Add a cell dimension to reflect
    Projection is an xarray DataArray
    TODO: Add metadata attributes
    Parameters
    ----------
    TrackedFeatures : xarray.core.dataset.Dataset
        xarray dataset of tobac Track information, the xarray dataset returned by tobac.tracking.linking_trackpy
    Mask: xarray.core.dataset.Dataset
        xarray dataset of tobac segmentation mask information, the xarray dataset returned
        by tobac.segmentation.segmentation
    Projection : xarray.core.dataarray.DataArray, default = None
        array.DataArray of the original input dataset (gridded nexrad data for example).
        If using gridded nexrad data, this can be input as: data['ProjectionCoordinateSystem']
        An example of the type of information in the dataarray includes the following attributes:
        latitude_of_projection_origin :29.471900939941406
        longitude_of_projection_origin :-95.0787353515625
        _CoordinateTransformType :Projection
        _CoordinateAxes :x y z time
        _CoordinateAxesTypes :GeoX GeoY Height Time
        grid_mapping_name :azimuthal_equidistant
        semi_major_axis :6370997.0
        inverse_flattening :298.25
        longitude_of_prime_meridian :0.0
        false_easting :0.0
        false_northing :0.0
    Returns
    -------
    ds : xarray.core.dataset.Dataset
        xarray dataset of merged Track and Segmentation Mask datasets with renamed variables.
    """
    import xarray as xr

    feature_standard_names = {
        # new variable name, and long description for the NetCDF attribute
        "frame": (
            "feature_time_index",
            "positional index of the feature along the time dimension of the mask, from 0 to N-1",
        ),
        "hdim_1": (
            "feature_hdim1_coordinate",
            "position of the feature along the first horizontal dimension in grid point space; a north-south coordinate for dim order (time, y, x)."
            "The numbering is consistent with positional indexing of the coordinate, but can be"
            "fractional, to account for a centroid not aligned to the grid.",
        ),
        "hdim_2": (
            "feature_hdim2_coordinate",
            "position of the feature along the second horizontal dimension in grid point space; an east-west coordinate for dim order (time, y, x)"
            "The numbering is consistent with positional indexing of the coordinate, but can be"
            "fractional, to account for a centroid not aligned to the grid.",
        ),
        "idx": ("feature_id_this_frame",),
        "num": (
            "feature_grid_cell_count",
            "Number of grid points that are within the threshold of this feature",
        ),
        "threshold_value": (
            "feature_threshold_max",
            "Feature number within that frame; starts at 1, increments by 1 to the number of features for each frame, and resets to 1 when the frame increments",
        ),
        "feature": (
            "feature",
            "Unique number of the feature; starts from 1 and increments by 1 to the number of features",
        ),
        "time": (
            "feature_time",
            "time of the feature, consistent with feature_time_index",
        ),
        "timestr": (
            "feature_time_str",
            "String representation of the feature time, YYYY-MM-DD HH:MM:SS",
        ),
        "projection_y_coordinate": (
            "feature_projection_y_coordinate",
            "y position of the feature in the projection given by ProjectionCoordinateSystem",
        ),
        "projection_x_coordinate": (
            "feature_projection_x_coordinate",
            "x position of the feature in the projection given by ProjectionCoordinateSystem",
        ),
        "lat": ("feature_latitude", "latitude of the feature"),
        "lon": ("feature_longitude", "longitude of the feature"),
        "ncells": (
            "feature_ncells",
            "number of grid cells for this feature (meaning uncertain)",
        ),
        "areas": ("feature_area",),
        "isolated": ("feature_isolation_flag",),
        "num_objects": ("number_of_feature_neighbors",),
        "cell": ("feature_parent_cell_id",),
        "time_cell": ("feature_parent_cell_elapsed_time",),
        "segmentation_mask": ("2d segmentation mask",),
    }
    new_feature_var_names = {
        k: feature_standard_names[k][0]
        for k in feature_standard_names.keys()
        if k in TrackedFeatures.variables.keys()
    }

    #     TrackedFeatures = TrackedFeatures.drop(["cell_parent_track_id"])
    # Combine Track and Mask variables. Use the 'feature' variable as the coordinate variable instead of
    # the 'index' variable and call the dimension 'feature'
    ds = xr.merge(
        [
            TrackedFeatures.swap_dims({"index": "feature"})
            .drop("index")
            .rename_vars(new_feature_var_names),
            Mask,
        ]
    )

    # Add the projection data back in
    if Projection is not None:
        ds["ProjectionCoordinateSystem"] = Projection

    return ds

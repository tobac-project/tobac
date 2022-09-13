import logging
import functools


def column_mask_from2D(mask_2D, cube, z_coord="model_level_number"):
    """function to turn 2D watershedding mask into a 3D mask of selected columns
    Input:
    cube:              iris.cube.Cube
                       data cube
    mask_2D:           iris.cube.Cube
                       2D cube containing mask (int id for tacked volumes 0 everywhere else)
    z_coord:           str
                       name of the vertical coordinate in the cube
    Output:
    mask_2D:           iris.cube.Cube
                       3D cube containing columns of 2D mask (int id for tacked volumes 0 everywhere else)
    """
    from copy import deepcopy

    mask_3D = deepcopy(cube)
    mask_3D.rename("segmentation_mask")
    dim = mask_3D.coord_dims(z_coord)[0]
    for i in range(len(mask_3D.coord(z_coord).points)):
        slc = [slice(None)] * len(mask_3D.shape)
        slc[dim] = slice(i, i + 1)
        mask_out = mask_3D[slc]
        mask_3D.data[slc] = mask_2D.core_data()
    return mask_3D


def mask_cube_cell(variable_cube, mask, cell, track):
    """Mask cube for tracked volume of an individual cell
    Input:
    variable_cube:     iris.cube.Cube
                       unmasked data cube
    mask:              iris.cube.Cube
                       cube containing mask (int id for tacked volumes 0 everywhere else)
    cell:          int
                       interger id of cell to create masked cube for
    Output:
    variable_cube_out: iris.cube.Cube
                       Masked cube with data for respective cell
    """
    from copy import deepcopy

    variable_cube_out = deepcopy(variable_cube)
    feature_ids = track.loc[track["cell"] == cell, "feature"].values
    variable_cube_out = mask_cube_features(variable_cube, mask, feature_ids)
    return variable_cube_out


def mask_cube_all(variable_cube, mask):
    """Mask cube for untracked volume
    Input:
    variable_cube:     iris.cube.Cube
                       unmasked data cube
    mask:              iris.cube.Cube
                       cube containing mask (int id for tacked volumes 0 everywhere else)
    Output:
    variable_cube_out: iris.cube.Cube
                       Masked cube for untracked volume
    """
    from dask.array import ma
    from copy import deepcopy

    variable_cube_out = deepcopy(variable_cube)
    variable_cube_out.data = ma.masked_where(
        mask.core_data() == 0, variable_cube_out.core_data()
    )
    return variable_cube_out


def mask_cube_untracked(variable_cube, mask):
    """Mask cube for untracked volume
    Input:
    variable_cube:     iris.cube.Cube
                       unmasked data cube
    mask:              iris.cube.Cube
                       cube containing mask (int id for tacked volumes 0 everywhere else)
    Output:
    variable_cube_out: iris.cube.Cube
                       Masked cube for untracked volume
    """
    from dask.array import ma
    from copy import deepcopy

    variable_cube_out = deepcopy(variable_cube)
    variable_cube_out.data = ma.masked_where(
        mask.core_data() > 0, variable_cube_out.core_data()
    )
    return variable_cube_out


def mask_cube(cube_in, mask):
    """Mask cube where mask is larger than zero
    Input:
    cube_in:           iris.cube.Cube
                       unmasked data cube
    mask:              numpy.ndarray or dask.array
                       mask to use for masking, >0 where cube is supposed to be masked
    Output:
    cube_out:          iris.cube.Cube
                       Masked cube
    """
    from dask.array import ma
    from copy import deepcopy

    cube_out = deepcopy(cube_in)
    cube_out.data = ma.masked_where(mask != 0, cube_in.core_data())
    return cube_out


def mask_cell(mask, cell, track, masked=False):
    """create mask for specific cell
    Input:
    mask:              iris.cube.Cube
                       cube containing mask (int id for tacked volumes 0 everywhere else)
    Output:
    variable_cube_out: numpy.ndarray
                       Masked cube for untracked volume
    """
    feature_ids = track.loc[track["cell"] == cell, "feature"].values
    mask_i = mask_features(mask, feature_ids, masked=masked)
    return mask_i


def mask_cell_surface(mask, cell, track, masked=False, z_coord="model_level_number"):
    """Create surface projection of mask for individual cell
    Input:
    mask:              iris.cube.Cube
                       cube containing mask (int id for tacked volumes 0 everywhere else)
    Output:
    variable_cube_out: iris.cube.Cube
                       Masked cube for untracked volume
    """
    feature_ids = track.loc[track["cell"] == cell, "feature"].values
    mask_i_surface = mask_features_surface(
        mask, feature_ids, masked=masked, z_coord=z_coord
    )
    return mask_i_surface


def mask_cell_columns(mask, cell, track, masked=False, z_coord="model_level_number"):
    """Create mask with entire columns for individual cell
    Input:
    mask:              iris.cube.Cube
                       cube containing mask (int id for tacked volumes 0 everywhere else)
    Output:
    variable_cube_out: iris.cube.Cube
                       Masked cube for untracked volume
    """
    feature_ids = track.loc[track["cell"] == cell].loc["feature"]
    mask_i = mask_features_columns(mask, feature_ids, masked=masked, z_coord=z_coord)
    return mask_i


def mask_cube_features(variable_cube, mask, feature_ids):
    """Mask cube for tracked volume of an individual cell
    Input:
    variable_cube:     iris.cube.Cube
                       unmasked data cube
    mask:              iris.cube.Cube
                       cube containing mask (int id for tacked volumes 0 everywhere else)
    cell:          int
                       interger id of cell to create masked cube for
    Output:
    variable_cube_out: iris.cube.Cube
                       Masked cube with data for respective cell
    """
    from dask.array import ma, isin
    from copy import deepcopy

    variable_cube_out = deepcopy(variable_cube)
    variable_cube_out.data = ma.masked_where(
        ~isin(mask.core_data(), feature_ids), variable_cube_out.core_data()
    )
    return variable_cube_out


def mask_features(mask, feature_ids, masked=False):
    """create mask for specific features
    Input:
    mask:              iris.cube.Cube
                       cube containing mask (int id for tacked volumes 0 everywhere else)
    Output:
    variable_cube_out: numpy.ndarray
                       Masked cube for untracked volume
    """
    from dask.array import ma, isin
    from copy import deepcopy

    mask_i = deepcopy(mask)
    mask_i_data = mask_i.core_data()
    mask_i_data[~isin(mask_i.core_data(), feature_ids)] = 0
    if masked:
        mask_i.data = ma.masked_equal(mask_i.core_data(), 0)
    return mask_i


def mask_features_surface(
    mask, feature_ids, masked=False, z_coord="model_level_number"
):
    """create surface mask for individual features
    Input:
    mask:              iris.cube.Cube
                       cube containing mask (int id for tacked volumes 0 everywhere else)
    Output:
    variable_cube_out: iris.cube.Cube
                       Masked cube for untracked volume
    """
    from iris.analysis import MAX
    from dask.array import ma, isin
    from copy import deepcopy

    mask_i = deepcopy(mask)
    #     mask_i.data=[~isin(mask_i.data,feature_ids)]=0
    mask_i_data = mask_i.core_data()
    mask_i_data[~isin(mask_i.core_data(), feature_ids)] = 0
    mask_i_surface = mask_i.collapsed(z_coord, MAX)
    if masked:
        mask_i_surface.data = ma.masked_equal(mask_i_surface.core_data(), 0)
    return mask_i_surface


def mask_all_surface(mask, masked=False, z_coord="model_level_number"):
    """create surface mask for individual features
    Input:
    mask:              iris.cube.Cube
                       cube containing mask (int id for tacked volumes 0 everywhere else)
    Output:
    mask_i_surface:    iris.cube.Cube (2D)
                       Mask with 1 below features and 0 everywhere else
    """
    from iris.analysis import MAX
    from dask.array import ma, isin
    from copy import deepcopy

    mask_i = deepcopy(mask)
    mask_i_surface = mask_i.collapsed(z_coord, MAX)
    mask_i_surface_data = mask_i_surface.core_data()
    mask_i_surface[mask_i_surface_data > 0] = 1
    if masked:
        mask_i_surface.data = ma.masked_equal(mask_i_surface.core_data(), 0)
    return mask_i_surface


# def mask_features_columns(mask,feature_ids,masked=False,z_coord='model_level_number'):
#     ''' Mask cube for untracked volume
#     Input:
#     variable_cube:     iris.cube.Cube
#                        unmasked data cube
#     mask:              iris.cube.Cube
#                        cube containing mask (int id for tacked volumes 0 everywhere else)
#     Output:
#     variable_cube_out: iris.cube.Cube
#                        Masked cube for untracked volume
#     '''
#     from iris.analysis import MAX
#     import numpy as np
#     from copy import deepcopy
#     mask_i=deepcopy(mask)
#     mask_i.data[~np.isin(mask_i.data,feature_ids)]=0
#     mask_i_surface=mask_i.collapsed(z_coord,MAX)
#     for cube_slice in mask_i.slices(['time','x','y']):
#             cube_slice.data=mask_i_surface.core_data()
#     if masked:
#         mask_i.data=np.ma.array(mask_i.data,mask=mask_i.data)
#     return mask_i


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


def add_coordinates(t, variable_cube):
    import numpy as np

    """ Function adding coordinates from the tracking cube to the trajectories: time, longitude&latitude, x&y dimensions
    Input:
    t:             pandas DataFrame
                   trajectories/features
    variable_cube: iris.cube.Cube 
                   Cube containing the dimensions 'time','longitude','latitude','x_projection_coordinate','y_projection_coordinate', usually cube that the tracking is performed on
    Output:
    t:             pandas DataFrame 
                   trajectories with added coordinated
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
    from numpy import delete, arange, diff, nonzero, array

    """ Calculates the bounding box of a ndarray
    https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    """
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
    import numpy as np
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


def get_label_props_in_dict(labels):
    """Function to get the label properties into a dictionary format.

    Parameters
    ----------
    labels:    2D array-like
        comes from the `skimage.measure.label` function

    Returns
    -------
    dict
        output from skimage.measure.regionprops in dictionary format, where they key is the label number
    """
    import skimage.measure

    region_properties_raw = skimage.measure.regionprops(labels)
    region_properties_dict = {
        region_prop.label: region_prop for region_prop in region_properties_raw
    }

    return region_properties_dict


def get_indices_of_labels_from_reg_prop_dict(region_property_dict):
    """Function to get the x and y indices (as well as point count) of
    all labeled regions.

    Parameters
    ----------
    region_property_dict:    dict of region_property objects
        This dict should come from the get_label_props_in_dict function.

    Returns
    -------
    dict (key: label number, int)
        The number of points in the label number
    dict (key: label number, int)
        the y indices in the label number
    dict (key: label number, int)
        the x indices in the label number

    Raises
    ------
    ValueError
        a ValueError is raised if there are no regions in the region property dict

    """
    import numpy as np

    if len(region_property_dict) == 0:
        raise ValueError("No regions!")

    y_indices = dict()
    x_indices = dict()
    curr_loc_indices = dict()

    # loop through all skimage identified regions
    for region_prop_key in region_property_dict:
        region_prop = region_property_dict[region_prop_key]
        index = region_prop.label
        curr_y_ixs, curr_x_ixs = np.transpose(region_prop.coords)

        y_indices[index] = curr_y_ixs
        x_indices[index] = curr_x_ixs
        curr_loc_indices[index] = len(curr_y_ixs)

    return (curr_loc_indices, y_indices, x_indices)


def iris_to_xarray(func):
    """Decorator that converts all input of a function that is in the form of 
    Iris cubes into xarray DataArrays and converts all output in xarray 
    DataArrays back into Iris cubes.

    Parameters
    ----------
    func : function
        Function to be decorated

    Returns
    -------
    wrapper : function
        Function including decorator
    """

    import iris
    import xarray

    def wrapper(*args, **kwargs):
        # print(kwargs)
        if any([type(arg) == iris.cube.Cube for arg in args]) or any(
            [type(arg) == iris.cube.Cube for arg in kwargs]
        ):
            # print("converting iris to xarray and back")
            args = tuple(
                [
                    xarray.DataArray.from_iris(arg)
                    if type(arg) == iris.cube.Cube
                    else arg
                    for arg in args
                ]
            )
            kwargs = kwargs.update(
                zip(
                    kwargs.keys(),
                    [
                        xarray.DataArray.from_iris(arg)
                        if type(arg) == iris.cube.Cube
                        else arg
                        for arg in kwargs.values()
                    ],
                )
            )

            output = func(*args, **kwargs)
            if type(output) == tuple:
                output = tuple(
                    [
                        xarray.DataArray.to_iris(output_item)
                        if type(output_item) == xarray.DataArray
                        else output_item
                        for output_item in output
                    ]
                )
            else:
                output = xarray.DataArray.to_iris(output)

        else:
            output = func(*args, **kwargs)
        return output

    return wrapper


def xarray_to_iris(func):
    """Decorator that converts all input of a function that is in the form of 
    xarray DataArrays Iris cubes into and converts all output in Iris cubes 
    back into xarray DataArrays.

    Parameters
    ----------
    func : function
        Function to be decorated.

    Returns
    -------
    wrapper : function
        Function including decorator.
    """

    import iris
    import xarray

    def wrapper(*args, **kwargs):
        # print(args)
        # print(kwargs)
        if any([type(arg) == xarray.DataArray for arg in args]) or any(
            [type(arg) == xarray.DataArray for arg in kwargs]
        ):
            # print("converting xarray to iris and back")
            args = tuple(
                [
                    xarray.DataArray.to_iris(arg)
                    if type(arg) == xarray.DataArray
                    else arg
                    for arg in args
                ]
            )
            if kwargs:
                kwargs_new = dict(
                    zip(
                        kwargs.keys(),
                        [
                            xarray.DataArray.to_iris(arg)
                            if type(arg) == xarray.DataArray
                            else arg
                            for arg in kwargs.values()
                        ],
                    )
                )
            else:
                kwargs_new = kwargs
            # print(args)
            # print(kwargs)
            output = func(*args, **kwargs_new)
            if type(output) == tuple:
                output = tuple(
                    [
                        xarray.DataArray.from_iris(output_item)
                        if type(output_item) == iris.cube.Cube
                        else output_item
                        for output_item in output
                    ]
                )
            else:
                if type(output) == iris.cube.Cube:
                    output = xarray.DataArray.from_iris(output)

        else:
            output = func(*args, **kwargs)
        # print(output)
        return output

    return wrapper


def irispandas_to_xarray(func):
    """Decorator that converts all input of a function that is in the form of 
    Iris cubes into xarray DataArrays and converts all output in xarray 
    DataArrays back into Iris cubes.

    Parameters
    ----------
    func : function
        Function to be decorated.

    Returns
    -------
    wrapper : function
        Function including decorator.
    """
    import iris
    import xarray
    import pandas as pd

    def wrapper(*args, **kwargs):
        # print(kwargs)
        if any(
            [(type(arg) == iris.cube.Cube or type(arg) == pd.DataFrame) for arg in args]
        ) or any(
            [
                (type(arg) == iris.cube.Cube or type(arg) == pd.DataFrame)
                for arg in kwargs
            ]
        ):
            # print("converting iris to xarray and back")
            args = tuple(
                [
                    xarray.DataArray.from_iris(arg)
                    if type(arg) == iris.cube.Cube
                    else arg.to_xarray
                    if type(arg) == pd.DataFrame
                    else arg
                    for arg in args
                ]
            )
            kwargs = kwargs.update(
                zip(
                    kwargs.keys(),
                    [
                        xarray.DataArray.from_iris(arg)
                        if type(arg) == iris.cube.Cube
                        else arg.to_xarray
                        if type(arg) == pd.DataFrame
                        else arg
                        for arg in kwargs.values()
                    ],
                )
            )

            output = func(*args, **kwargs)
            if type(output) == tuple:
                output = tuple(
                    [
                        xarray.DataArray.to_iris(output_item)
                        if type(output_item) == xarray.DataArray
                        else output_item.to_dataframe()
                        if type(output_item) == xarray.Dataset
                        else output_item
                        for output_item in output
                    ]
                )
            else:
                if type(output) == xarray.DataArray:
                    output = xarray.DataArray.to_iris(output)
                elif type(output) == xarray.Dataset:
                    output.to_dataframe()

        else:
            output = func(*args, **kwargs)
        return output

    return wrapper


def xarray_to_irispandas(func):
    """Decorator that converts all input of a function that is in the form of 
    xarray DataArrays Iris cubes into and converts all output in Iris cubes 
    back into xarray DataArrays.

    Parameters
    ----------
    func : function
        Function to be decorated.

    Returns
    -------
    wrapper : function
        Function including decorator.
    """

    import iris
    import xarray
    import pandas as pd

    def wrapper(*args, **kwargs):
        # print(args)
        # print(kwargs)
        if any(
            [
                (type(arg) == xarray.DataArray or type(arg) == xarray.Dataset)
                for arg in args
            ]
        ) or any(
            [
                (type(arg) == xarray.DataArray or type(arg) == xarray.Dataset)
                for arg in kwargs
            ]
        ):
            # print("converting xarray to iris and back")
            args = tuple(
                [
                    xarray.DataArray.to_iris(arg)
                    if type(arg) == xarray.DataArray
                    else arg.to_dataframe()
                    if type(arg) == xarray.Dataset
                    else arg
                    for arg in args
                ]
            )
            if kwargs:
                kwargs_new = dict(
                    zip(
                        kwargs.keys(),
                        [
                            xarray.DataArray.to_iris(arg)
                            if type(arg) == xarray.DataArray
                            else arg.to_dataframe()
                            if type(arg) == xarray.Dataset
                            else arg
                            for arg in kwargs.values()
                        ],
                    )
                )
            else:
                kwargs_new = kwargs
            # print(args)
            # print(kwargs)
            output = func(*args, **kwargs_new)
            if type(output) == tuple:
                output = tuple(
                    [
                        xarray.DataArray.from_iris(output_item)
                        if type(output_item) == iris.cube.Cube
                        else output_item.to_xarray()
                        if type(output_item) == pd.DataFrame
                        else output_item
                        for output_item in output
                    ]
                )
            else:
                if type(output) == iris.cube.Cube:
                    output = xarray.DataArray.from_iris(output)
                elif type(output) == pd.DataFrame:
                    output.to_xarray()

        else:
            output = func(*args, **kwargs)
        # print(output)
        return output

    return wrapper
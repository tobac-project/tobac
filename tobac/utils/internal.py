"""Internal tobac utilities
"""


def get_label_props_in_dict(labels):
    """Function to get the label properties into a dictionary format.

    Parameters
    ----------
    labels : 2D array-like
        Output of the `skimage.measure.label` function.

    Returns
    -------
    region_properties_dict: dict
        Output from skimage.measure.regionprops in dictionary
        format, where they key is the label number.
    """

    import skimage.measure

    region_properties_raw = skimage.measure.regionprops(labels)
    region_properties_dict = {
        region_prop.label: region_prop for region_prop in region_properties_raw
    }

    return region_properties_dict


def get_indices_of_labels_from_reg_prop_dict(region_property_dict):
    """Function to get the x, y, and z indices (as well as point count) of all labeled regions.
    Parameters
    ----------
    region_property_dict : dict of region_property objects
        This dict should come from the get_label_props_in_dict function.
    Returns
    -------
    curr_loc_indices : dict
        The number of points in the label number (key: label number).
    z_indices : dict
        The z indices in the label number. If a 2D property dict is passed, this value is not returned.
    y_indices : dict
        The y indices in the label number (key: label number).
    x_indices : dict
        The x indices in the label number (key: label number).
    Raises
    ------
    ValueError
        A ValueError is raised if there are no regions in the region
        property dict.
    """

    import numpy as np

    if len(region_property_dict) == 0:
        raise ValueError("No regions!")

    z_indices = dict()
    y_indices = dict()
    x_indices = dict()
    curr_loc_indices = dict()
    is_3D = False

    # loop through all skimage identified regions
    for region_prop_key in region_property_dict:
        region_prop = region_property_dict[region_prop_key]
        index = region_prop.label
        if len(region_prop.coords[0]) >= 3:
            is_3D = True
            curr_z_ixs, curr_y_ixs, curr_x_ixs = np.transpose(region_prop.coords)
            z_indices[index] = curr_z_ixs
        else:
            curr_y_ixs, curr_x_ixs = np.transpose(region_prop.coords)
            z_indices[index] = -1

        y_indices[index] = curr_y_ixs
        x_indices[index] = curr_x_ixs
        curr_loc_indices[index] = len(curr_y_ixs)
    # print("indices found")
    if is_3D:
        return [curr_loc_indices, z_indices, y_indices, x_indices]
    else:
        return [curr_loc_indices, y_indices, x_indices]


def iris_to_xarray(func):
    """Decorator that converts all input of a function that is in the form of
    Iris cubes into xarray DataArrays and converts all outputs with type
    xarray DataArrays back into Iris cubes.

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
            [type(arg) == iris.cube.Cube for arg in kwargs.values()]
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
            kwargs_new = dict(
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
            # print(args)
            # print(kwargs)
            output = func(*args, **kwargs_new)
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
    xarray DataArrays into Iris cubes and converts all outputs with type
    Iris cubes back into xarray DataArrays.

    Parameters
    ----------
    func : function
        Function to be decorated.

    Returns
    -------
    wrapper : function
        Function including decorator.

    Examples
    --------
    >>> segmentation_xarray = xarray_to_iris(segmentation)

    This line creates a new function that can process xarray fields and
    also outputs fields in xarray format, but otherwise works just like
    the original function:

    >>> mask_xarray, features = segmentation_xarray(
        features, data_xarray, dxy, threshold
        )
    """

    import iris
    import xarray

    def wrapper(*args, **kwargs):
        # print(args)
        # print(kwargs)
        if any([type(arg) == xarray.DataArray for arg in args]) or any(
            [type(arg) == xarray.DataArray for arg in kwargs.values()]
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
    Iris cubes/pandas Dataframes into xarray DataArrays/xarray Datasets and
    converts all outputs with the type xarray DataArray/xarray Dataset
    back into Iris cubes/pandas Dataframes.

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
                for arg in kwargs.values()
            ]
        ):
            # print("converting iris to xarray and back")
            args = tuple(
                [
                    xarray.DataArray.from_iris(arg)
                    if type(arg) == iris.cube.Cube
                    else arg.to_xarray()
                    if type(arg) == pd.DataFrame
                    else arg
                    for arg in args
                ]
            )
            kwargs = dict(
                zip(
                    kwargs.keys(),
                    [
                        xarray.DataArray.from_iris(arg)
                        if type(arg) == iris.cube.Cube
                        else arg.to_xarray()
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
                    output = output.to_dataframe()

        else:
            output = func(*args, **kwargs)
        return output

    return wrapper


def xarray_to_irispandas(func):
    """Decorator that converts all input of a function that is in the form of
    DataArrays/xarray Datasets into xarray Iris cubes/pandas Dataframes and
    converts all outputs with the type Iris cubes/pandas Dataframes back into
    xarray DataArray/xarray Dataset.

    Parameters
    ----------
    func : function
        Function to be decorated.

    Returns
    -------
    wrapper : function
        Function including decorator.

    Examples
    --------
    >>> linking_trackpy_xarray = xarray_to_irispandas(
        linking_trackpy
        )

    This line creates a new function that can process xarray inputs and
    also outputs in xarray formats, but otherwise works just like the
    original function:

    >>> track_xarray = linking_trackpy_xarray(
        features_xarray, field_xarray, dt, dx
        )
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
                for arg in kwargs.values()
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
                    output = output.to_xarray()

        else:
            output = func(*args, **kwargs)
        # print(output)
        return output

    return wrapper


def njit_if_available(func, **kwargs):
    """Decorator to wrap a function with numba.njit if available.
    If numba isn't available, it just returns the function.

    Parameters
    ----------
    func: function object
        Function to wrap with njit
    kwargs:
        Keyword arguments to pass to numba njit
    """
    try:
        from numba import njit

        return njit(func, kwargs)
    except ModuleNotFoundError:
        return func


def find_vertical_axis_from_coord(variable_cube, vertical_coord="auto"):
    """Function to find the vertical coordinate in the iris cube

    Parameters
    ----------
    variable_cube: iris.cube
        Input variable cube, containing a vertical coordinate.
    vertical_coord: str
        Vertical coordinate name. If `auto`, this function tries to auto-detect.

    Returns
    -------
    str
        the vertical coordinate name

    Raises
    ------
    ValueError
        Raised if the vertical coordinate isn't found in the cube.
    """

    list_coord_names = [coord.name() for coord in variable_cube.coords()]

    if vertical_coord == "auto":
        list_vertical = ["z", "model_level_number", "altitude", "geopotential_height"]
        # find the intersection
        all_vertical_axes = list(set(list_coord_names) & set(list_vertical))
        if len(all_vertical_axes) >= 1:
            return all_vertical_axes[0]
        else:
            raise ValueError(
                "Cube lacks suitable automatic vertical coordinate (z, model_level_number, altitude, or geopotential_height)"
            )
    elif vertical_coord in list_coord_names:
        return vertical_coord
    else:
        raise ValueError("Please specify vertical coordinate found in cube")


def find_dataframe_vertical_coord(variable_dataframe, vertical_coord="auto"):
    """Function to find the vertical coordinate in the iris cube

    Parameters
    ----------
    variable_dataframe: pandas.DataFrame
        Input variable cube, containing a vertical coordinate.
    vertical_coord: str
        Vertical coordinate name. If `auto`, this function tries to auto-detect.

    Returns
    -------
    str
        the vertical coordinate name

    Raises
    ------
    ValueError
        Raised if the vertical coordinate isn't found in the cube.
    """

    if vertical_coord == "auto":
        list_vertical = ["z", "model_level_number", "altitude", "geopotential_height"]
        all_vertical_axes = list(set(variable_dataframe.columns) & set(list_vertical))
        if len(all_vertical_axes) == 1:
            return all_vertical_axes[0]
        else:
            raise ValueError("Please specify vertical coordinate")

    else:
        if vertical_coord in variable_dataframe.columns:
            return vertical_coord
        else:
            raise ValueError("Please specify vertical coordinate")

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
    """Function to get the x and y indices (as well as point count) of
    all labeled regions.

    Parameters
    ----------
    region_property_dict : dict of region_property objects
        This dict should come from the get_label_props_in_dict function.

    Returns
    -------
    curr_loc_indices : dict
        The number of points in the label number (key: label number).

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

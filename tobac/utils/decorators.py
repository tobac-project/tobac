"""Decorators for use with other tobac functions"""

from __future__ import annotations
import functools
import warnings

import numpy as np
from numpy import ma
import pandas as pd
import xarray as xr
import iris.cube


def convert_cube_to_dataarray(cube):
    """
    Convert an iris cube to an xarray dataarray, averting error for integer dtype cubes in xarray<v2023.06

    Parameters
    ----------
    cube : iris.cube.Cube
        Iris data cube

    Returns
    -------
    dataarray : xr.DataArray
        dataarray converted from cube. If the cube's core data is a masked array and has integer dtype,
        the returned datarray will have a numpy array with masked values filled with the minimum value for
        that integer dtype. Otherwise the data will be identical to that produced using xr.DataArray.from_iris
    """
    if isinstance(cube.core_data(), ma.core.MaskedArray) and np.issubdtype(
        cube.core_data().dtype, np.integer
    ):
        return xr.DataArray.from_iris(
            cube.copy(cube.core_data().filled(np.iinfo(cube.core_data().dtype).min))
        )
    return xr.DataArray.from_iris(cube)


def _conv_kwargs_iris_to_xarray(conv_kwargs: dict):
    """
    Internal function to convert iris cube kwargs to xarray dataarrays

    Parameters
    ----------
    conv_kwargs : dict
        Input kwargs to convert

    Returns
    -------
    dict
        Output keyword arguments without any Iris Cubes
    """
    return {
        key: convert_cube_to_dataarray(arg) if isinstance(arg, iris.cube.Cube) else arg
        for key, arg in zip(conv_kwargs.keys(), conv_kwargs.values())
    }


def _conv_kwargs_irispandas_to_xarray(conv_kwargs: dict):
    """
    Internal function to convert iris cube and pandas dataframe kwargs to xarray dataarrays

    Parameters
    ----------
    conv_kwargs : dict
        Input kwargs to convert

    Returns
    -------
    dict
        Output keyword arguments without any Iris Cubes or pandas dataframes

    """
    return {
        key: (
            convert_cube_to_dataarray(arg)
            if isinstance(arg, iris.cube.Cube)
            else arg.to_xarray() if isinstance(arg, pd.DataFrame) else arg
        )
        for key, arg in zip(conv_kwargs.keys(), conv_kwargs.values())
    }


def _conv_kwargs_xarray_to_iris(conv_kwargs: dict):
    """
    Internal function to convert  xarray dataarray kwargs back to iris cubes

    Parameters
    ----------
    conv_kwargs : dict
        Input kwargs to convert

    Returns
    -------
    dict
        Output keyword arguments with all xarray dataarrays converted back to
        iris cubes
    """
    return {
        key: (
            xr.DataArray.to_iris(arg).copy(arg.data)
            if isinstance(arg, xr.DataArray)
            else arg
        )
        for key, arg in zip(conv_kwargs.keys(), conv_kwargs.values())
    }


def _conv_kwargs_xarray_to_irispandas(conv_kwargs: dict):
    """
    Internal function to convert xarray dataarrays back to iris cubes/pandas dataframes

    Parameters
    ----------
    conv_kwargs : dict
        Input kwargs to convert

    Returns
    -------
    dict
        Output keyword arguments with all xarray dataarrays converted back to
        iris cubes
    """
    return {
        key: (
            xr.DataArray.to_iris(arg).copy(arg.data)
            if isinstance(arg, xr.DataArray)
            else arg.to_dataframe() if isinstance(arg, xr.Dataset) else arg
        )
        for key, arg in zip(conv_kwargs.keys(), conv_kwargs.values())
    }


def iris_to_xarray(save_iris_info: bool = False):
    def iris_to_xarray_i(func):
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
        import iris.cube
        import xarray

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # print(kwargs)

            if save_iris_info:
                if any([(type(arg) == iris.cube.Cube) for arg in args]) or any(
                    [(type(arg) == iris.cube.Cube) for arg in kwargs.values()]
                ):
                    kwargs["converted_from_iris"] = True
                else:
                    kwargs["converted_from_iris"] = False

            if any([type(arg) == iris.cube.Cube for arg in args]) or any(
                [type(arg) == iris.cube.Cube for arg in kwargs.values()]
            ):
                # print("converting iris to xarray and back")
                args = tuple(
                    [
                        (
                            convert_cube_to_dataarray(arg)
                            if type(arg) == iris.cube.Cube
                            else arg
                        )
                        for arg in args
                    ]
                )
                kwargs_new = _conv_kwargs_iris_to_xarray(kwargs)
                # print(args)
                # print(kwargs)
                output = func(*args, **kwargs_new)
                if type(output) == tuple:
                    output = tuple(
                        [
                            (
                                output_item.to_iris().copy(output_item.data)
                                if type(output_item) == xarray.DataArray
                                else output_item
                            )
                            for output_item in output
                        ]
                    )
                elif type(output) == xarray.DataArray:
                    output = output.to_iris().copy(output.data)
                # if output is neither tuple nor an xr.DataArray

            else:
                output = func(*args, **kwargs)
            return output

        return wrapper

    return iris_to_xarray_i


def xarray_to_iris():
    def xarray_to_iris_i(func):
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
        >>> segmentation_xarray_conv = xarray_to_iris()
        >>> segmentation_xarray = segmentation_xarray_conv(segmentation)

        This line creates a new function that can process xarray fields and
        also outputs fields in xarray format, but otherwise works just like
        the original function:

        >>> mask_xarray, features = segmentation_xarray(
            features, data_xarray, dxy, threshold
            )
        """

        import iris
        import xarray

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # print(args)
            # print(kwargs)
            if any([type(arg) == xarray.DataArray for arg in args]) or any(
                [type(arg) == xarray.DataArray for arg in kwargs.values()]
            ):
                # print("converting xarray to iris and back")
                args = tuple(
                    [
                        (
                            arg.to_iris().copy(arg.data)
                            if type(arg) == xarray.DataArray
                            else arg
                        )
                        for arg in args
                    ]
                )
                if kwargs:
                    kwargs_new = _conv_kwargs_xarray_to_iris(kwargs)
                else:
                    kwargs_new = kwargs
                # print(args)
                # print(kwargs)
                output = func(*args, **kwargs_new)
                if type(output) == tuple:
                    output = tuple(
                        [
                            (
                                xarray.DataArray.from_iris(output_item)
                                if type(output_item) == iris.cube.Cube
                                else output_item
                            )
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

    return xarray_to_iris_i


def irispandas_to_xarray(save_iris_info: bool = False):
    def irispandas_to_xarray_i(func):
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
        import iris.cube
        import xarray
        import pandas as pd

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # pass if we did an iris conversion.
            if save_iris_info:
                if any([(type(arg) == iris.cube.Cube) for arg in args]) or any(
                    [(type(arg) == iris.cube.Cube) for arg in kwargs.values()]
                ):
                    kwargs["converted_from_iris"] = True
                else:
                    kwargs["converted_from_iris"] = False

            # print(kwargs)
            if any(
                [
                    (type(arg) == iris.cube.Cube or type(arg) == pd.DataFrame)
                    for arg in args
                ]
            ) or any(
                [
                    (type(arg) == iris.cube.Cube or type(arg) == pd.DataFrame)
                    for arg in kwargs.values()
                ]
            ):
                # print("converting iris to xarray and back")
                args = tuple(
                    [
                        (
                            convert_cube_to_dataarray(arg)
                            if type(arg) == iris.cube.Cube
                            else arg.to_xarray() if type(arg) == pd.DataFrame else arg
                        )
                        for arg in args
                    ]
                )
                kwargs = _conv_kwargs_irispandas_to_xarray(kwargs)

                output = func(*args, **kwargs)
                if type(output) == tuple:
                    output = tuple(
                        [
                            (
                                output_item.to_iris().copy(output_item.data)
                                if type(output_item) == xarray.DataArray
                                else (
                                    output_item.to_dataframe()
                                    if type(output_item) == xarray.Dataset
                                    else output_item
                                )
                            )
                            for output_item in output
                        ]
                    )
                else:
                    if type(output) == xarray.DataArray:
                        output = output.to_iris().copy(output.data)
                    elif type(output) == xarray.Dataset:
                        output = output.to_dataframe()

            else:
                output = func(*args, **kwargs)
            return output

        return wrapper

    return irispandas_to_xarray_i


def xarray_to_irispandas():
    def xarray_to_irispandas_i(func):
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

        @functools.wraps(func)
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
                        (
                            xarray.DataArray.to_iris(arg).copy(arg.data)
                            if type(arg) == xarray.DataArray
                            else (
                                arg.to_dataframe()
                                if type(arg) == xarray.Dataset
                                else arg
                            )
                        )
                        for arg in args
                    ]
                )
                if kwargs:
                    kwargs_new = _conv_kwargs_xarray_to_irispandas(kwargs)
                else:
                    kwargs_new = kwargs
                # print(args)
                # print(kwargs)
                output = func(*args, **kwargs_new)
                if type(output) == tuple:
                    output = tuple(
                        [
                            (
                                xarray.DataArray.from_iris(output_item)
                                if type(output_item) == iris.cube.Cube
                                else (
                                    output_item.to_xarray()
                                    if type(output_item) == pd.DataFrame
                                    else output_item
                                )
                            )
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

    return xarray_to_irispandas_i


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
    except KeyboardInterrupt as kie:
        raise
    except Exception as exc:
        warnings.warn(
            "Numba not able to be imported; periodic boundary calculations will be slower."
            "Exception raised: " + repr(exc)
        )
        return func

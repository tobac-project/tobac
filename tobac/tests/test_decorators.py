"""
Tests for tobac.utils.decorators
"""

import numpy as np
import pandas as pd
import xarray as xr
import iris

from tobac.utils import decorators


def test_convert_cube_to_dataarray():
    test_da_float = xr.DataArray(np.arange(15, dtype=float).reshape(3, 5) + 0.5)
    test_da_int = xr.DataArray(np.arange(15, dtype=int).reshape(3, 5))

    assert np.all(
        decorators.convert_cube_to_dataarray(test_da_float.to_iris())
        == test_da_float.values
    )
    assert np.all(
        decorators.convert_cube_to_dataarray(test_da_int.to_iris())
        == test_da_int.values
    )


def test_conv_kwargs_iris_to_xarray():
    assert decorators._conv_kwargs_iris_to_xarray({}) == {}
    assert decorators._conv_kwargs_iris_to_xarray(dict(test_int=1)) == dict(test_int=1)

    test_da = xr.DataArray(np.arange(5))

    test_xr_kwarg = decorators._conv_kwargs_iris_to_xarray(dict(test_xr=test_da))
    assert isinstance(test_xr_kwarg["test_xr"], xr.DataArray)

    test_iris_kwarg = decorators._conv_kwargs_iris_to_xarray(
        dict(test_iris=test_da.to_iris())
    )
    assert isinstance(test_iris_kwarg["test_iris"], xr.DataArray)


def test_conv_kwargs_irispandas_to_xarray():
    assert decorators._conv_kwargs_irispandas_to_xarray({}) == {}
    assert decorators._conv_kwargs_irispandas_to_xarray(dict(test_int=1)) == dict(
        test_int=1
    )

    test_da = xr.DataArray(np.arange(5))

    test_xr_kwarg = decorators._conv_kwargs_irispandas_to_xarray(dict(test_xr=test_da))
    assert isinstance(test_xr_kwarg["test_xr"], xr.DataArray)

    test_iris_kwarg = decorators._conv_kwargs_irispandas_to_xarray(
        dict(test_iris=test_da.to_iris())
    )
    assert isinstance(test_iris_kwarg["test_iris"], xr.DataArray)

    test_ds = xr.Dataset({"test": test_da})
    test_ds_kwarg = decorators._conv_kwargs_irispandas_to_xarray(dict(test_xr=test_ds))
    assert isinstance(test_ds_kwarg["test_xr"], xr.Dataset)

    test_pd_kwarg = decorators._conv_kwargs_irispandas_to_xarray(
        dict(test_pd=test_ds.to_pandas())
    )
    assert isinstance(test_pd_kwarg["test_pd"], xr.Dataset)


def test_conv_kwargs_xarray_to_iris():
    assert decorators._conv_kwargs_xarray_to_iris({}) == {}
    assert decorators._conv_kwargs_xarray_to_iris(dict(test_int=1)) == dict(test_int=1)

    test_da = xr.DataArray(np.arange(5))

    test_xr_kwarg = decorators._conv_kwargs_xarray_to_iris(dict(test_xr=test_da))
    assert isinstance(test_xr_kwarg["test_xr"], iris.cube.Cube)

    test_iris_kwarg = decorators._conv_kwargs_xarray_to_iris(
        dict(test_iris=test_da.to_iris())
    )
    assert isinstance(test_iris_kwarg["test_iris"], iris.cube.Cube)


def test_conv_kwargs_xarray_to_irispandas():
    assert decorators._conv_kwargs_xarray_to_irispandas({}) == {}
    assert decorators._conv_kwargs_xarray_to_irispandas(dict(test_int=1)) == dict(
        test_int=1
    )

    test_da = xr.DataArray(np.arange(5))

    test_xr_kwarg = decorators._conv_kwargs_xarray_to_irispandas(dict(test_xr=test_da))
    assert isinstance(test_xr_kwarg["test_xr"], iris.cube.Cube)

    test_iris_kwarg = decorators._conv_kwargs_xarray_to_irispandas(
        dict(test_iris=test_da.to_iris())
    )
    assert isinstance(test_iris_kwarg["test_iris"], iris.cube.Cube)

    test_ds = xr.Dataset({"test": test_da})
    test_ds_kwarg = decorators._conv_kwargs_xarray_to_irispandas(dict(test_xr=test_ds))
    assert isinstance(test_ds_kwarg["test_xr"], pd.DataFrame)

    test_pd_kwarg = decorators._conv_kwargs_xarray_to_irispandas(
        dict(test_pd=test_ds.to_pandas())
    )
    assert isinstance(test_pd_kwarg["test_pd"], pd.DataFrame)


@decorators.iris_to_xarray(save_iris_info=True)
def _test_iris_to_xarray(*args, **kwargs):
    return kwargs["converted_from_iris"]


def test_iris_to_xarray():
    test_da = xr.DataArray(np.arange(5))

    assert _test_iris_to_xarray(test_da) == False
    assert _test_iris_to_xarray(kwarg_xr=test_da) == False

    assert _test_iris_to_xarray(test_da.to_iris()) == True
    assert _test_iris_to_xarray(kwarg_ir=test_da.to_iris()) == True


@decorators.irispandas_to_xarray(save_iris_info=True)
def _test_irispandas_to_xarray(*args, **kwargs):
    return kwargs["converted_from_iris"]


def test_irispandas_to_xarray():
    test_da = xr.DataArray(np.arange(5))

    assert _test_irispandas_to_xarray(test_da) == False
    assert _test_irispandas_to_xarray(kwarg_xr=test_da) == False

    assert _test_irispandas_to_xarray(test_da.to_iris()) == True
    assert _test_irispandas_to_xarray(kwarg_ir=test_da.to_iris()) == True


@decorators.xarray_to_irispandas()
def _test_xarray_to_irispandas(*args, **kwargs):
    return args, kwargs


def test_xarray_to_irispandas():
    test_da = xr.DataArray(np.arange(5, dtype=float))

    assert isinstance(_test_xarray_to_irispandas(test_da)[0][0], iris.cube.Cube)
    assert _test_xarray_to_irispandas(test_da)[1] == {}

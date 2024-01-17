"""Tests for the iris/xarray conversion decorators"""

import pytest
import tobac
import tobac.testing
import xarray
import iris
from iris.cube import Cube
import pandas as pd
from pandas.testing import assert_frame_equal
from copy import deepcopy
from tobac.utils.decorators import (
    xarray_to_iris,
    iris_to_xarray,
    xarray_to_irispandas,
    irispandas_to_xarray,
)


@pytest.mark.parametrize(
    "decorator, input_types, expected_internal_types, expected_output_type",
    [
        (
            xarray_to_iris,
            [xarray.DataArray, xarray.DataArray],
            [Cube, Cube],
            xarray.DataArray,
        ),
        (xarray_to_iris, [Cube, Cube], [Cube, Cube], Cube),
        (xarray_to_iris, [Cube, xarray.DataArray], [Cube, Cube], xarray.DataArray),
        (xarray_to_iris, [xarray.DataArray, Cube], [Cube, Cube], xarray.DataArray),
        (iris_to_xarray, [Cube, Cube], [xarray.DataArray, xarray.DataArray], Cube),
        (
            iris_to_xarray,
            [xarray.DataArray, xarray.DataArray],
            [xarray.DataArray, xarray.DataArray],
            xarray.DataArray,
        ),
        (
            iris_to_xarray,
            [xarray.DataArray, Cube],
            [xarray.DataArray, xarray.DataArray],
            Cube,
        ),
        (
            iris_to_xarray,
            [Cube, xarray.DataArray],
            [xarray.DataArray, xarray.DataArray],
            Cube,
        ),
        (
            xarray_to_irispandas,
            [xarray.DataArray, xarray.DataArray],
            [Cube, Cube],
            xarray.DataArray,
        ),
        (xarray_to_irispandas, [Cube, Cube], [Cube, Cube], Cube),
        (
            xarray_to_irispandas,
            [Cube, xarray.DataArray],
            [Cube, Cube],
            xarray.DataArray,
        ),
        (
            xarray_to_irispandas,
            [xarray.DataArray, Cube],
            [Cube, Cube],
            xarray.DataArray,
        ),
        (
            xarray_to_irispandas,
            [xarray.Dataset, xarray.Dataset],
            [pd.DataFrame, pd.DataFrame],
            xarray.Dataset,
        ),
        (
            xarray_to_irispandas,
            [pd.DataFrame, pd.DataFrame],
            [pd.DataFrame, pd.DataFrame],
            pd.DataFrame,
        ),
        (
            xarray_to_irispandas,
            [xarray.Dataset, pd.DataFrame],
            [pd.DataFrame, pd.DataFrame],
            xarray.Dataset,
        ),
        (
            xarray_to_irispandas,
            [pd.DataFrame, xarray.Dataset],
            [pd.DataFrame, pd.DataFrame],
            xarray.Dataset,
        ),
        (
            xarray_to_irispandas,
            [xarray.Dataset, xarray.DataArray],
            [pd.DataFrame, Cube],
            xarray.Dataset,
        ),
        (
            irispandas_to_xarray,
            [Cube, Cube],
            [xarray.DataArray, xarray.DataArray],
            Cube,
        ),
        (
            irispandas_to_xarray,
            [xarray.DataArray, xarray.DataArray],
            [xarray.DataArray, xarray.DataArray],
            xarray.DataArray,
        ),
        (
            irispandas_to_xarray,
            [xarray.DataArray, Cube],
            [xarray.DataArray, xarray.DataArray],
            Cube,
        ),
        (
            irispandas_to_xarray,
            [Cube, xarray.DataArray],
            [xarray.DataArray, xarray.DataArray],
            Cube,
        ),
        (
            irispandas_to_xarray,
            [pd.DataFrame, pd.DataFrame],
            [xarray.Dataset, xarray.Dataset],
            pd.DataFrame,
        ),
        (
            irispandas_to_xarray,
            [xarray.Dataset, xarray.Dataset],
            [xarray.Dataset, xarray.Dataset],
            xarray.Dataset,
        ),
        (
            irispandas_to_xarray,
            [pd.DataFrame, xarray.Dataset],
            [xarray.Dataset, xarray.Dataset],
            pd.DataFrame,
        ),
        (
            irispandas_to_xarray,
            [xarray.Dataset, pd.DataFrame],
            [xarray.Dataset, xarray.Dataset],
            pd.DataFrame,
        ),
        (
            irispandas_to_xarray,
            [pd.DataFrame, Cube],
            [xarray.Dataset, xarray.DataArray],
            pd.DataFrame,
        ),
    ],
)
def test_converting(
    decorator, input_types, expected_internal_types, expected_output_type
):
    """Testing the conversions of the decorators internally and for the output"""

    def test_function_kwarg(test_input, kwarg=None):
        assert (
            type(test_input) == expected_internal_types[0]
        ), "Expected internal type {}, got {} for {}".format(
            expected_internal_types[0], type(test_input), decorator.__name__
        )
        assert (
            type(kwarg) == expected_internal_types[1]
        ), "Expected internal type {}, got {} for {} as keyword argument".format(
            expected_internal_types[1], type(kwarg), decorator.__name__
        )
        return test_input

    def test_function_tuple_output(test_input, kwarg=None):
        return (test_input, test_input)

    decorated_function_kwarg = decorator(test_function_kwarg)
    decorated_function_tuple = decorator(test_function_tuple_output)

    if input_types[0] == xarray.DataArray:
        data = xarray.DataArray.from_iris(tobac.testing.make_simple_sample_data_2D())
    elif input_types[0] == Cube:
        data = tobac.testing.make_simple_sample_data_2D()
    elif input_types[0] == xarray.Dataset:
        data = tobac.testing.generate_single_feature(
            1, 1, max_h1=100, max_h2=100
        ).to_xarray()
    elif input_types[0] == pd.DataFrame:
        data = tobac.testing.generate_single_feature(1, 1, max_h1=100, max_h2=100)

    if input_types[1] == xarray.DataArray:
        kwarg = xarray.DataArray.from_iris(tobac.testing.make_simple_sample_data_2D())
    elif input_types[1] == Cube:
        kwarg = tobac.testing.make_simple_sample_data_2D()
    elif input_types[1] == xarray.Dataset:
        kwarg = tobac.testing.generate_single_feature(
            1, 1, max_h1=100, max_h2=100
        ).to_xarray()
    elif input_types[1] == pd.DataFrame:
        kwarg = tobac.testing.generate_single_feature(1, 1, max_h1=100, max_h2=100)

    output = decorated_function_kwarg(data, kwarg=kwarg)
    tuple_output = decorated_function_tuple(data, kwarg=kwarg)

    assert (
        type(output) == expected_output_type
    ), "Expected output type {}, got {} for {}".format(
        expected_output_type, type(output), decorator.__name__
    )
    assert (
        type(tuple_output[0]) == expected_output_type
    ), "Expected output type {}, but got {} for {} (1st tuple output(".format(
        expected_output_type, type(tuple_output[0]), decorator.__name__
    )
    assert (
        type(tuple_output[1]) == expected_output_type
    ), "Expected output type {}, but got {} for {} (2nd tuple output(".format(
        expected_output_type, type(tuple_output[1]), decorator.__name__
    )


def test_xarray_workflow():
    """Test comparing the outputs of the standard functions of tobac for a test dataset
    with the output of the same functions decorated with tobac.utils.xarray_to_iris"""

    data = tobac.testing.make_sample_data_2D_3blobs()
    data_xarray = xarray.DataArray.from_iris(deepcopy(data))

    # Testing the get_spacings utility
    get_spacings_xarray = xarray_to_iris(tobac.utils.get_spacings)
    dxy, dt = tobac.utils.get_spacings(data)
    dxy_xarray, dt_xarray = get_spacings_xarray(data_xarray)

    assert dxy == dxy_xarray
    assert dt == dt_xarray

    # Testing feature detection
    feature_detection_xarray = xarray_to_iris(
        tobac.feature_detection.feature_detection_multithreshold
    )
    features = tobac.feature_detection.feature_detection_multithreshold(
        data, dxy, threshold=1.0
    )
    features_xarray = feature_detection_xarray(data_xarray, dxy_xarray, threshold=1.0)

    assert_frame_equal(features, features_xarray)

    # Testing the segmentation
    segmentation_xarray = xarray_to_iris(tobac.segmentation.segmentation)
    mask, features = tobac.segmentation.segmentation(features, data, dxy, threshold=1.0)
    mask_xarray, features_xarray = segmentation_xarray(
        features_xarray, data_xarray, dxy_xarray, threshold=1.0
    )

    assert (mask.data == mask_xarray.to_iris().data).all()

    # testing tracking
    tracking_xarray = xarray_to_iris(tobac.tracking.linking_trackpy)
    track = tobac.tracking.linking_trackpy(features, data, dt, dxy, v_max=100.0)
    track_xarray = tracking_xarray(
        features_xarray, data_xarray, dt_xarray, dxy_xarray, v_max=100.0
    )

    assert_frame_equal(track, track_xarray)

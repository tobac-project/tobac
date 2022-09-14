import tobac
import tobac.testing
import xarray
from pandas.testing import assert_frame_equal
from copy import deepcopy


def test_xarray_to_iris():
    """Testing tobac.utils.xarray_to_iris for the standard steps of tobac with a
    test dataset"""

    data = tobac.testing.make_sample_data_2D_3blobs()
    data_xarray = xarray.DataArray.from_iris(deepcopy(data))

    # Testing the get_spacings utility
    get_spacings_xarray = tobac.utils.xarray_to_iris(tobac.utils.get_spacings)
    dxy, dt = tobac.utils.get_spacings(data)
    dxy_xarray, dt_xarray = get_spacings_xarray(data_xarray)

    assert dxy == dxy_xarray
    assert dt == dt_xarray

    # Testing feature detection
    feature_detection_xarray = tobac.utils.xarray_to_iris(
        tobac.feature_detection.feature_detection_multithreshold
    )
    features = tobac.feature_detection.feature_detection_multithreshold(data, dxy, 1.0)
    features_xarray = feature_detection_xarray(data_xarray, dxy_xarray, 1.0)

    assert_frame_equal(features, features_xarray)

    # Testing the segmentation
    segmentation_xarray = tobac.utils.xarray_to_iris(tobac.segmentation.segmentation)
    mask, features = tobac.segmentation.segmentation(features, data, dxy, 1.0)
    mask_xarray, features_xarray = segmentation_xarray(
        features_xarray, data_xarray, dxy_xarray, 1.0
    )

    assert (mask.data == mask_xarray.to_iris().data).all()

    # testing tracking
    tracking_xarray = tobac.utils.xarray_to_iris(tobac.tracking.linking_trackpy)
    track = tobac.tracking.linking_trackpy(features, data, dt, dxy, v_max=100.0)
    track_xarray = tracking_xarray(
        features_xarray, data_xarray, dt_xarray, dxy_xarray, v_max=100.0
    )

    assert_frame_equal(track, track_xarray)


def test_xarray_to_irispandas():
    """Testing tobac.utils.xarray_to_irispandas for the standard steps of tobac with a
    test dataset"""

    data = tobac.testing.make_sample_data_2D_3blobs()
    data_xarray = xarray.DataArray.from_iris(deepcopy(data))

    # Testing the get_spacings utility
    get_spacings_xarray = tobac.utils.xarray_to_irispandas(tobac.utils.get_spacings)
    dxy, dt = tobac.utils.get_spacings(data)
    dxy_xarray, dt_xarray = get_spacings_xarray(data_xarray)

    assert dxy == dxy_xarray
    assert dt == dt_xarray

    # Testing feature detection
    feature_detection_xarray = tobac.utils.xarray_to_irispandas(
        tobac.feature_detection.feature_detection_multithreshold
    )
    features = tobac.feature_detection.feature_detection_multithreshold(data, dxy, 1.0)
    features_xarray = feature_detection_xarray(data_xarray, dxy_xarray, 1.0)

    assert_frame_equal(features, features_xarray)

    # Testing the segmentation
    segmentation_xarray = tobac.utils.xarray_to_irispandas(
        tobac.segmentation.segmentation
    )
    mask, features = tobac.segmentation.segmentation(features, data, dxy, 1.0)
    mask_xarray, features_xarray = segmentation_xarray(
        features_xarray, data_xarray, dxy_xarray, 1.0
    )

    assert (mask.data == mask_xarray.to_iris().data).all()

    # testing tracking
    tracking_xarray = tobac.utils.xarray_to_irispandas(tobac.tracking.linking_trackpy)
    track = tobac.tracking.linking_trackpy(features, data, dt, dxy, v_max=100.0)
    track_xarray = tracking_xarray(
        features_xarray, data_xarray, dt_xarray, dxy_xarray, v_max=100.0
    )

    assert_frame_equal(track, track_xarray, check_names=False)

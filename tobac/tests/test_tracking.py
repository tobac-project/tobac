'''
Test for the trackpy tracking functions
Who's watching the watchmen, basically.
'''
from pyexpat import features
import pytest
import tobac.testing
import tobac.tracking
import copy
from pandas.util.testing import assert_frame_equal
import numpy as np

def test_linking_trackpy():
    '''Function to test ```tobac.tracking.linking_trackpy```
    Currently tests:
    2D tracking 
    3D tracking
    3D tracking with PBCs
    '''

    # Test 2D tracking of a simple moving feature
    test_feature = tobac.testing.generate_single_feature(1, 1, 
                        min_h1 = 0, max_h1 = 100, min_h2 = 0, max_h2 = 100,
                        frame_start = 0, num_frames=5,
                        spd_h1 = 1, spd_h2 = 1, PBC_flag='none')
    
    expected_out_feature = copy.deepcopy(test_feature)
    expected_out_feature['cell'] = 1.0

    actual_out_feature = tobac.tracking.linking_trackpy(
        test_feature, None, 5, 1000, 
        v_max = 10000, method_linking='predict',
        PBC_flag = 'none'
    )
    # Just want to remove the time_cell column here. 
    actual_out_feature = actual_out_feature[['hdim_1', 'hdim_2', 'frame', 'feature', 'time', 'cell']]

    assert_frame_equal(expected_out_feature.sort_index(axis=1), actual_out_feature.sort_index(axis=1))

    # Test 3D tracking of a simple moving feature
    test_feature = tobac.testing.generate_single_feature(1, 1, start_v = 1,
                        min_h1 = 0, max_h1 = 100, min_h2 = 0, max_h2 = 100,
                        frame_start = 0, num_frames=5,
                        spd_h1 = 1, spd_h2 = 1, spd_v = 1, PBC_flag='none')
    
    expected_out_feature = copy.deepcopy(test_feature)
    expected_out_feature['cell'] = 1.0

    actual_out_feature = tobac.tracking.linking_trackpy(
        test_feature, None, 5, 1000, dz=1000,
        v_max = 10000, method_linking='predict',
        PBC_flag = 'none', vertical_coord=None
    )
    # Just want to remove the time_cell column here. 
    actual_out_feature = actual_out_feature[['hdim_1', 'hdim_2', 'vdim', 'frame', 'feature', 'time', 'cell']]

    assert_frame_equal(expected_out_feature.sort_index(axis=1), actual_out_feature.sort_index(axis=1))


    # Test 3D tracking of a simple moving feature with periodic boundaries on hdim_1
    test_feature = tobac.testing.generate_single_feature(1, 1, start_v = 1,
                        min_h1 = 0, max_h1 = 10, min_h2 = 0, max_h2 = 10,
                        frame_start = 0, num_frames=8,
                        spd_h1 = 3, spd_h2 = 1, spd_v = 1, PBC_flag='hdim_1')
    
    expected_out_feature = copy.deepcopy(test_feature)
    expected_out_feature['cell'] = 1.0

    actual_out_feature = tobac.tracking.linking_trackpy(
        test_feature, None, 1, 1, dz=1, min_h1 = 0, max_h1 = 10, min_h2 = 0, max_h2 = 10,
        v_max = 4, method_linking='predict', vertical_coord=None,
        PBC_flag = 'hdim_1'
    )
    # Just want to remove the time_cell column here. 
    actual_out_feature = actual_out_feature[['hdim_1', 'hdim_2', 'vdim', 'frame', 'feature', 'time', 'cell']]

    assert_frame_equal(expected_out_feature.sort_index(axis=1), actual_out_feature.sort_index(axis=1))

    # Test 3D tracking of a simple moving feature with periodic boundaries on hdim_2
    test_feature = tobac.testing.generate_single_feature(1, 1, start_v = 1,
                        min_h1 = 0, max_h1 = 10, min_h2 = 0, max_h2 = 10,
                        frame_start = 0, num_frames=8,
                        spd_h1 = 1, spd_h2 = 3, spd_v = 1, PBC_flag='hdim_2')
    
    expected_out_feature = copy.deepcopy(test_feature)
    expected_out_feature['cell'] = 1.0

    actual_out_feature = tobac.tracking.linking_trackpy(
        test_feature, None, 1, 1, dz=1, min_h1 = 0, max_h1 = 10, min_h2 = 0, max_h2 = 10,
        v_max = 4, method_linking='predict', vertical_coord=None,
        PBC_flag = 'hdim_2'
    )
    # Just want to remove the time_cell column here. 
    actual_out_feature = actual_out_feature[['hdim_1', 'hdim_2', 'vdim', 'frame', 'feature', 'time', 'cell']]

    assert_frame_equal(expected_out_feature.sort_index(axis=1), actual_out_feature.sort_index(axis=1))

    # Test 3D tracking of a simple moving feature with periodic boundaries on both hdim_1 and hdim_2
    test_feature = tobac.testing.generate_single_feature(1, 1, start_v = 1,
                        min_h1 = 0, max_h1 = 10, min_h2 = 0, max_h2 = 10,
                        frame_start = 0, num_frames=8,
                        spd_h1 = 3, spd_h2 = 3, spd_v = 0, PBC_flag='both')
    
    expected_out_feature = copy.deepcopy(test_feature)
    expected_out_feature['cell'] = 1.0

    actual_out_feature = tobac.tracking.linking_trackpy(
        test_feature, None, 1, 1,dz=1, min_h1 = 0, max_h1 = 10, min_h2 = 0, max_h2 = 10,
        v_max = 5, method_linking='predict', vertical_coord=None,
        PBC_flag = 'both'
    )
    # Just want to remove the time_cell column here. 
    actual_out_feature = actual_out_feature[['hdim_1', 'hdim_2', 'vdim', 'frame', 'feature', 'time', 'cell']]

    assert_frame_equal(expected_out_feature.sort_index(axis=1), actual_out_feature.sort_index(axis=1))


def test_build_distance_function():
    '''Tests ```tobac.tracking.build_distance_function``` 
    Currently tests:
    that this produces an object that is suitable to call from trackpy
    '''

    test_func = tobac.tracking.build_distance_function(0, 10, 0, 10, 'both')
    assert (test_func(np.array((0,9,9)), np.array((0,0,0))) == pytest.approx(1.4142135))


@pytest.mark.parametrize("point_init, speed, dxy, actual_dz, v_max,"
                                "use_dz, features_connected", 
                        [((0,0,0), (1,0,0), 1000, 100, 200, True, True), 
                         ((0,0,0), (1,0,0), 1000, 100, 200, False, True), 
                         ((0,0,0), (5,0,0), 1000, 100, 200, True, False), 
                         ((0,0,0), (5,0,0), 1000, 100, 200, False, False), 
                         ]
)
def test_3D_tracking_min_dist_z(point_init, speed, dxy, actual_dz, v_max,
                                use_dz, features_connected):
    '''Tests ```tobac.tracking.linking_trackpy``` with 
    points in z with varying distances between them.
    
    Parameters
    ----------
    point_init: 3D array-like 
        Initial point (z, y, x)
    speed: 3D array-like
        Speed of the feature (z, y, x)
    dxy: float
        grid spacing for dx and dy
    actual_dz: float
        grid spacing for Z
    use_dz: bool
        True to use the passed in constant dz, False
        to use the calculated vertical coordinates
    features_connected: bool
        Do we expect the features to be connected?
    '''


    test_feature = tobac.testing.generate_single_feature(
                        start_h1 = point_init[1], start_h2 = point_init[2], 
                        start_v = point_init[0],
                        min_h1 = 0, max_h1 = 100, min_h2 = 0, max_h2 = 100,
                        frame_start = 0, num_frames=2,
                        spd_h1 = speed[1], spd_h2 = speed[2], spd_v=speed[0],
                        PBC_flag='none')
    if not use_dz:
        test_feature['z'] = test_feature['vdim']*actual_dz
    
    expected_out_feature = copy.deepcopy(test_feature)

    if features_connected:
        expected_out_feature['cell'] = 1.0
    else:
        expected_out_feature['cell'] = np.nan

    common_params = {
        'features': test_feature,
        'field_in': None,
        'dt': 1,
        'time_cell_min': 1,
        'dxy': dxy,
        'v_max': v_max,
        'method_linking': 'predict',
    }
    if use_dz: 
        common_params['dz'] = actual_dz
        common_params['vertical_coord'] = None
    else:
        common_params['vertical_coord'] = 'z'

    actual_out_feature = tobac.tracking.linking_trackpy(
        **common_params
    )
    # Just want to remove the time_cell column here. 
    actual_out_feature = actual_out_feature.drop('time_cell', axis=1)
    assert_frame_equal(expected_out_feature.sort_index(axis=1), actual_out_feature.sort_index(axis=1))




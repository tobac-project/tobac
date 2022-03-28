'''
Test for the trackpy tracking functions
Who's watching the watchmen, basically.
'''
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
        test_feature, None, 5, 1000, 
        v_max = 10000, method_linking='predict',
        PBC_flag = 'none'
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
        test_feature, None, 1, 1, min_h1 = 0, max_h1 = 10, min_h2 = 0, max_h2 = 10,
        v_max = 4, method_linking='predict',
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
        test_feature, None, 1, 1, min_h1 = 0, max_h1 = 10, min_h2 = 0, max_h2 = 10,
        v_max = 4, method_linking='predict',
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
        test_feature, None, 1, 1, min_h1 = 0, max_h1 = 10, min_h2 = 0, max_h2 = 10,
        v_max = 5, method_linking='predict',
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



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


def test_calc_distance_coords_pbc():
    '''Tests ```tobac.tracking.calc_distance_coords_pbc```
    Currently tests:
    two points in normal space 
    Periodicity along hdim_1, hdim_2, and corners
    '''

    # Test first two points in normal space with varying PBC conditions
    for PBC_condition in ['none', 'hdim_1', 'hdim_2', 'both']:
        assert (tobac.tracking.calc_distance_coords_pbc(np.array((0,0,0)), np.array((0,0,0)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(0))
        assert (tobac.tracking.calc_distance_coords_pbc(np.array((0,0,0)), np.array((0,0,1)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(1))
        assert (tobac.tracking.calc_distance_coords_pbc(np.array((0,0)), np.array((0,1)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(1))
        assert (tobac.tracking.calc_distance_coords_pbc(np.array((0,0,0)), np.array((3,3,1)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(4.3588989, rel=1e-3))

    # Now test two points that will be closer along the hdim_1 boundary for cases without PBCs
    for PBC_condition in ['hdim_1', 'both']:
        assert (tobac.tracking.calc_distance_coords_pbc(np.array((0,0,0)), np.array((0,9,0)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(1))
        assert (tobac.tracking.calc_distance_coords_pbc(np.array((0,9,0)), np.array((0,0,0)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(1))
        assert (tobac.tracking.calc_distance_coords_pbc(np.array((8,0)), np.array((0,0)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(2))
        assert (tobac.tracking.calc_distance_coords_pbc(np.array((4,0,4)), np.array((3,7,3)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(3.3166247))
        assert (tobac.tracking.calc_distance_coords_pbc(np.array((4,0,4)), np.array((3,7,3)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(3.3166247))



    # Test the same points, except without PBCs
    for PBC_condition in ['none', 'hdim_2']:
        assert (tobac.tracking.calc_distance_coords_pbc(np.array((0,0,0)), np.array((0,9,0)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(9))
        assert (tobac.tracking.calc_distance_coords_pbc(np.array((0,9,0)), np.array((0,0,0)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(9))
        assert (tobac.tracking.calc_distance_coords_pbc(np.array((8,0)), np.array((0,0)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(8))

    # Now test two points that will be closer along the hdim_2 boundary for cases without PBCs
    for PBC_condition in ['hdim_2', 'both']:
        assert (tobac.tracking.calc_distance_coords_pbc(np.array((0,0,0)), np.array((0,0,9)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(1))
        assert (tobac.tracking.calc_distance_coords_pbc(np.array((0,0,9)), np.array((0,0,0)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(1))
        assert (tobac.tracking.calc_distance_coords_pbc(np.array((0,8)), np.array((0,0)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(2))

    # Test the same points, except without PBCs
    for PBC_condition in ['none', 'hdim_1']:
        assert (tobac.tracking.calc_distance_coords_pbc(np.array((0,0,0)), np.array((0,0,9)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(9))
        assert (tobac.tracking.calc_distance_coords_pbc(np.array((0,0,9)), np.array((0,0,0)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(9))
        assert (tobac.tracking.calc_distance_coords_pbc(np.array((0,8)), np.array((0,0)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(8))

    # Test points that will be closer for the both
    PBC_condition = 'both'
    assert (tobac.tracking.calc_distance_coords_pbc(np.array((0,9,9)), np.array((0,0,0)), 0, 10, 0, 10, PBC_condition)
        == pytest.approx(1.4142135, rel=1e-3))
    assert (tobac.tracking.calc_distance_coords_pbc(np.array((0,0,9)), np.array((0,9,0)), 0, 10, 0, 10, PBC_condition)
        == pytest.approx(1.4142135, rel=1e-3))

    # Test the corner points for no PBCs
    PBC_condition = 'none'
    assert (tobac.tracking.calc_distance_coords_pbc(np.array((0,9,9)), np.array((0,0,0)), 0, 10, 0, 10, PBC_condition)
        == pytest.approx(12.727922, rel=1e-3))
    assert (tobac.tracking.calc_distance_coords_pbc(np.array((0,0,9)), np.array((0,9,0)), 0, 10, 0, 10, PBC_condition)
        == pytest.approx(12.727922, rel=1e-3))
    
    # Test the corner points for hdim_1 and hdim_2
    for PBC_condition in ['hdim_1', 'hdim_2']:
        assert (tobac.tracking.calc_distance_coords_pbc(np.array((0,9,9)), np.array((0,0,0)), 0, 10, 0, 10, PBC_condition)
            == pytest.approx(9.055385))
        assert (tobac.tracking.calc_distance_coords_pbc(np.array((0,0,9)), np.array((0,9,0)), 0, 10, 0, 10, PBC_condition)
            == pytest.approx(9.055385))



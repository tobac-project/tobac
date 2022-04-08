import pytest
import tobac.testing
import tobac.utils as tb_utils
from collections import Counter


def lists_equal_without_order(a, b):
    """
    This will make sure the inner list contain the same, 
    but doesn't account for duplicate groups.
    from: https://stackoverflow.com/questions/31501909/assert-list-of-list-equality-without-order-in-python/31502000
    """
    for l1 in a:
        check_counter = Counter(l1)
        if not any(Counter(l2) == check_counter for l2 in b):
            return False
    return True


def test_get_label_props_in_dict():
    '''Testing ```tobac.feature_detection.get_label_props_in_dict``` for both 2D and 3D cases.
    '''
    import skimage.measure as skim
    test_3D_data = tobac.testing.make_sample_data_3D_3blobs(data_type='xarray')
    test_2D_data = tobac.testing.make_sample_data_2D_3blobs(data_type='xarray')


    # make sure it works for 3D data
    labels_3D = skim.label(test_3D_data.values[0])
    
    output_3D = tb_utils.get_label_props_in_dict(labels_3D)
    
    #make sure it is a dict
    assert type(output_3D) is dict
    #make sure we get at least one output, there should be at least one label.
    assert len(output_3D) > 0

    # make sure it works for 2D data
    labels_2D = skim.label(test_2D_data.values[0])
    
    output_2D = tb_utils.get_label_props_in_dict(labels_2D)
    
    #make sure it is a dict
    assert type(output_2D) is dict
    #make sure we get at least one output, there should be at least one label.
    assert len(output_2D) > 0


def test_get_indices_of_labels_from_reg_prop_dict():
    '''Testing ```tobac.feature_detection.get_indices_of_labels_from_reg_prop_dict``` for 2D and 3D cases.
    '''
    import skimage.measure as skim
    import numpy as np
    test_3D_data = tobac.testing.make_sample_data_3D_3blobs(data_type='xarray')
    test_2D_data = tobac.testing.make_sample_data_2D_3blobs(data_type='xarray')


    # make sure it works for 3D data
    labels_3D = skim.label(test_3D_data.values[0])
    nx_3D = test_3D_data.values[0].shape[2]
    ny_3D = test_3D_data.values[0].shape[1]
    nz_3D = test_3D_data.values[0].shape[0]

    labels_2D = skim.label(test_2D_data.values[0])
    nx_2D = test_2D_data.values[0].shape[1]
    ny_2D = test_2D_data.values[0].shape[0]
    
    region_props_3D = tb_utils.get_label_props_in_dict(labels_3D)
    region_props_2D = tb_utils.get_label_props_in_dict(labels_2D)

    #get_indices_of_labels_from_reg_prop_dict 
    
    [curr_loc_indices, z_indices, y_indices, x_indices] = tb_utils.get_indices_of_labels_from_reg_prop_dict(region_props_3D)

    for index_key in curr_loc_indices:
        # there should be at least one value in each.
        assert curr_loc_indices[index_key] > 0
        
        assert np.all(z_indices[index_key] >= 0) and np.all(z_indices[index_key] < nz_3D)
        assert np.all(x_indices[index_key] >= 0) and np.all(x_indices[index_key] < nx_3D)
        assert np.all(y_indices[index_key] >= 0) and np.all(y_indices[index_key] < ny_3D)
    
    [curr_loc_indices, y_indices, x_indices] = tb_utils.get_indices_of_labels_from_reg_prop_dict(region_props_2D)

    for index_key in curr_loc_indices:
        # there should be at least one value in each.
        assert curr_loc_indices[index_key] > 0
        
        assert np.all(x_indices[index_key] >= 0) and np.all(x_indices[index_key] < nx_2D)
        assert np.all(y_indices[index_key] >= 0) and np.all(y_indices[index_key] < ny_2D)
    


def test_calc_distance_coords_pbc():
    '''Tests ```tobac.utils.calc_distance_coords_pbc```
    Currently tests:
    two points in normal space 
    Periodicity along hdim_1, hdim_2, and corners
    '''
    import numpy as np

    # Test first two points in normal space with varying PBC conditions
    for PBC_condition in ['none', 'hdim_1', 'hdim_2', 'both']:
        assert (tb_utils.calc_distance_coords_pbc(np.array((0,0,0)), np.array((0,0,0)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(0))
        assert (tb_utils.calc_distance_coords_pbc(np.array((0,0,0)), np.array((0,0,1)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(1))
        assert (tb_utils.calc_distance_coords_pbc(np.array((0,0)), np.array((0,1)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(1))
        assert (tb_utils.calc_distance_coords_pbc(np.array((0,0,0)), np.array((3,3,1)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(4.3588989, rel=1e-3))

    # Now test two points that will be closer along the hdim_1 boundary for cases without PBCs
    for PBC_condition in ['hdim_1', 'both']:
        assert (tb_utils.calc_distance_coords_pbc(np.array((0,0,0)), np.array((0,9,0)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(1))
        assert (tb_utils.calc_distance_coords_pbc(np.array((0,9,0)), np.array((0,0,0)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(1))
        assert (tb_utils.calc_distance_coords_pbc(np.array((8,0)), np.array((0,0)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(2))
        assert (tb_utils.calc_distance_coords_pbc(np.array((4,0,4)), np.array((3,7,3)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(3.3166247))
        assert (tb_utils.calc_distance_coords_pbc(np.array((4,0,4)), np.array((3,7,3)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(3.3166247))



    # Test the same points, except without PBCs
    for PBC_condition in ['none', 'hdim_2']:
        assert (tb_utils.calc_distance_coords_pbc(np.array((0,0,0)), np.array((0,9,0)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(9))
        assert (tb_utils.calc_distance_coords_pbc(np.array((0,9,0)), np.array((0,0,0)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(9))
        assert (tb_utils.calc_distance_coords_pbc(np.array((8,0)), np.array((0,0)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(8))

    # Now test two points that will be closer along the hdim_2 boundary for cases without PBCs
    for PBC_condition in ['hdim_2', 'both']:
        assert (tb_utils.calc_distance_coords_pbc(np.array((0,0,0)), np.array((0,0,9)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(1))
        assert (tb_utils.calc_distance_coords_pbc(np.array((0,0,9)), np.array((0,0,0)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(1))
        assert (tb_utils.calc_distance_coords_pbc(np.array((0,8)), np.array((0,0)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(2))

    # Test the same points, except without PBCs
    for PBC_condition in ['none', 'hdim_1']:
        assert (tb_utils.calc_distance_coords_pbc(np.array((0,0,0)), np.array((0,0,9)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(9))
        assert (tb_utils.calc_distance_coords_pbc(np.array((0,0,9)), np.array((0,0,0)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(9))
        assert (tb_utils.calc_distance_coords_pbc(np.array((0,8)), np.array((0,0)), 0, 10, 0, 10, PBC_condition)
                == pytest.approx(8))

    # Test points that will be closer for the both
    PBC_condition = 'both'
    assert (tb_utils.calc_distance_coords_pbc(np.array((0,9,9)), np.array((0,0,0)), 0, 10, 0, 10, PBC_condition)
        == pytest.approx(1.4142135, rel=1e-3))
    assert (tb_utils.calc_distance_coords_pbc(np.array((0,0,9)), np.array((0,9,0)), 0, 10, 0, 10, PBC_condition)
        == pytest.approx(1.4142135, rel=1e-3))

    # Test the corner points for no PBCs
    PBC_condition = 'none'
    assert (tb_utils.calc_distance_coords_pbc(np.array((0,9,9)), np.array((0,0,0)), 0, 10, 0, 10, PBC_condition)
        == pytest.approx(12.727922, rel=1e-3))
    assert (tb_utils.calc_distance_coords_pbc(np.array((0,0,9)), np.array((0,9,0)), 0, 10, 0, 10, PBC_condition)
        == pytest.approx(12.727922, rel=1e-3))
    
    # Test the corner points for hdim_1 and hdim_2
    for PBC_condition in ['hdim_1', 'hdim_2']:
        assert (tb_utils.calc_distance_coords_pbc(np.array((0,9,9)), np.array((0,0,0)), 0, 10, 0, 10, PBC_condition)
            == pytest.approx(9.055385))
        assert (tb_utils.calc_distance_coords_pbc(np.array((0,0,9)), np.array((0,9,0)), 0, 10, 0, 10, PBC_condition)
            == pytest.approx(9.055385))


@pytest.mark.parametrize("loc_1, loc_2, bounds, PBC_flag, expected_dist", 
                         [((0,0,0), (0,0,9), (0, 10, 0, 10), 'both', 1), 
                          ]
)
def test_calc_distance_coords_pbc_param(loc_1, loc_2, bounds, PBC_flag, expected_dist):
    '''Tests ```tobac.utils.calc_distance_coords_pbc``` in a parameterized way
    
    Parameters
    ----------
    loc_1: tuple
        First point location, either in 2D or 3D space (assumed z, h1, h2)
    loc_2: tuple
        Second point location, either in 2D or 3D space (assumed z, h1, h2)
    bounds: tuple
        hdim_1/hdim_2 bounds as (h1_min, h1_max, h2_min, h2_max)
    PBC_flag : {'none', 'hdim_1', 'hdim_2', 'both'}
        Sets whether to use periodic boundaries, and if so in which directions.
        'none' means that we do not have periodic boundaries
        'hdim_1' means that we are periodic along hdim1
        'hdim_2' means that we are periodic along hdim2
        'both' means that we are periodic along both horizontal dimensions
    expected_dist: float
        Expected distance between the two points
    '''
    import numpy as np

    assert (tb_utils.calc_distance_coords_pbc(np.array(loc_1), np.array(loc_2), bounds[0], bounds[1], 
            bounds[2], bounds[3], PBC_flag)== pytest.approx(expected_dist))


def test_get_pbc_coordinates():
    '''Tests tobac.util.get_pbc_coordinates. 
    Currently runs the following tests:
    For an invalid PBC_flag, we raise an error
    For PBC_flag of 'none', we truncate the box and give a valid box. 

    '''

    with pytest.raises(ValueError):
        tb_utils.get_pbc_coordinates(0, 10, 0, 10, 1, 4, 1, 4, 'c')

    # Test PBC_flag of none

    assert (tb_utils.get_pbc_coordinates(0, 10, 0, 10, 1, 4, 1, 4, 'none') == [(1, 4, 1, 4),])
    assert (tb_utils.get_pbc_coordinates(0, 10, 0, 10, -1, 4, 1, 4, 'none') == [(0, 4, 1, 4),])
    assert (tb_utils.get_pbc_coordinates(0, 10, 0, 10, 1, 12, 1, 4, 'none') == [(1, 10, 1, 4),])
    assert (tb_utils.get_pbc_coordinates(0, 10, 0, 10, 1, 12, -1, 4, 'none') == [(1, 10, 0, 4),])

    # Test PBC_flag with hdim_1 
    # Simple case, no PBC overlapping
    assert (tb_utils.get_pbc_coordinates(0, 10, 0, 10, 1, 4, 1, 4, 'hdim_1') == [(1, 4, 1, 4),])
    # PBC going on the min side
    assert (tb_utils.get_pbc_coordinates(0, 10, 0, 10, -1, 4, 1, 4, 'hdim_1') == [(0, 4, 1, 4), (9, 10, 1, 4)])
    # PBC going on the min side; should be truncated in hdim_2.
    assert (tb_utils.get_pbc_coordinates(0, 10, 0, 10, -1, 4, -1, 4, 'hdim_1') == [(0, 4, 0, 4), (9, 10, 0, 4)])
    # PBC going on the max side only
    assert (tb_utils.get_pbc_coordinates(0, 10, 0, 10, 4, 12, 1, 4, 'hdim_1') == [(4, 10, 1, 4), (0, 2, 1, 4)])
    # PBC overlapping
    assert (tb_utils.get_pbc_coordinates(0, 10, 0, 10, -4, 12, 1, 4, 'hdim_1') == [(0, 10, 1, 4),])

    # Test PBC_flag with hdim_2
    # Simple case, no PBC overlapping
    assert (tb_utils.get_pbc_coordinates(0, 10, 0, 10, 1, 4, 1, 4, 'hdim_2') == [(1, 4, 1, 4),])
    # PBC going on the min side
    assert (tb_utils.get_pbc_coordinates(0, 10, 0, 10, 1, 4, -1, 4, 'hdim_2') == [(1, 4, 0, 4), (1, 4, 9, 10)])
    # PBC going on the min side with truncation in hdim_1
    assert (tb_utils.get_pbc_coordinates(0, 10, 0, 10, -4, 4, -1, 4, 'hdim_2') == [(0, 4, 0, 4), (0, 4, 9, 10)])
    # PBC going on the max side 
    assert (tb_utils.get_pbc_coordinates(0, 10, 0, 10, 1, 4, 4, 12, 'hdim_2') == [(1, 4, 4, 10), (1, 4, 0, 2)])
    # PBC overlapping
    assert (tb_utils.get_pbc_coordinates(0, 10, 0, 10, 1, 4, -4, 12, 'hdim_2') == [(1, 4, 0, 10),])

    # Test PBC_flag with both
    # Simple case, no PBC overlapping
    assert (tb_utils.get_pbc_coordinates(0, 10, 0, 10, 1, 4, 1, 4, 'both') == [(1, 4, 1, 4),])
    # hdim_1 only testing
    # PBC on the min side of hdim_1 only
    assert (tb_utils.get_pbc_coordinates(0, 10, 0, 10, -1, 4, 1, 4, 'both') == [(0, 4, 1, 4), (9, 10, 1, 4)])
    # PBC on the max side of hdim_1 only
    assert (tb_utils.get_pbc_coordinates(0, 10, 0, 10, 4, 12, 1, 4, 'both') == [(4, 10, 1, 4), (0, 2, 1, 4)])
    # PBC overlapping on max side of hdim_1 only
    assert (tb_utils.get_pbc_coordinates(0, 10, 0, 10, -4, 12, 1, 4, 'both') == [(0, 10, 1, 4),])
    # hdim_2 only testing
    # PBC on the min side of hdim_2 only
    assert (tb_utils.get_pbc_coordinates(0, 10, 0, 10, 1, 4, -1, 4, 'both') == [(1, 4, 0, 4), (1, 4, 9, 10)])
    # PBC on the max side of hdim_2 only
    assert (tb_utils.get_pbc_coordinates(0, 10, 0, 10, 1, 4, 4, 12, 'both') == [(1, 4, 4, 10), (1, 4, 0, 2)])
    #  PBC overlapping on max side of hdim_2 only
    assert (tb_utils.get_pbc_coordinates(0, 10, 0, 10, 1, 4, -4, 12, 'both') == [(1, 4, 0, 10),])
    # hdim_1 and hdim_2 testing simultaneous
    # both larger than the actual domain 
    assert (tb_utils.get_pbc_coordinates(0, 10, 0, 10, -1, 12, -4, 14, 'both') == [(0, 10, 0, 10),])
    # min in hdim_1 and hdim_2 
    assert (lists_equal_without_order(tb_utils.get_pbc_coordinates(0, 10, 0, 10, -3, 3, -4, 2, 'both'), [(0, 3, 0, 2), (0, 3, 6, 10), (7, 10, 6, 10), (7, 10, 0, 2)]))
    # max in hdim_1, min in hdim_2
    assert (lists_equal_without_order(tb_utils.get_pbc_coordinates(0, 10, 0, 10, 5, 12, -4, 2, 'both'), [(5, 10, 0, 2), (5, 10, 6, 10), (0, 2, 6, 10), (0, 2, 0, 2)]))
    # max in hdim_1 and hdim_2
    assert (lists_equal_without_order(tb_utils.get_pbc_coordinates(0, 10, 0, 10, 5, 12, 7, 15, 'both'), [(5, 10, 7, 10), (5, 10, 0, 5), (0, 2, 0, 5), (0, 2, 7, 10)]))
    # min in hdim_1, max in hdim_2
    assert (lists_equal_without_order(tb_utils.get_pbc_coordinates(0, 10, 0, 10, -3, 3, 7, 15, 'both'), [(0, 3, 7, 10), (0, 3, 0, 5), (7, 10, 0, 5), (7, 10, 7, 10)]))

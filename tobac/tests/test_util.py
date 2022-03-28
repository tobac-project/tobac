import pytest
import tobac.testing
import tobac.utils as tb_utils

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



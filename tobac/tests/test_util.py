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
    


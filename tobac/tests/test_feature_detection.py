import tobac.testing
import tobac.feature_detection as feat_detect
import pytest

def test_get_label_props_in_dict():
    '''Testing ```tobac.feature_detection.get_label_props_in_dict``` for both 2D and 3D cases.
    '''
    import skimage.measure as skim
    test_3D_data = tobac.testing.make_sample_data_3D_3blobs(data_type='xarray')
    test_2D_data = tobac.testing.make_sample_data_2D_3blobs(data_type='xarray')


    # make sure it works for 3D data
    labels_3D = skim.label(test_3D_data.values[0])
    
    output_3D = feat_detect.get_label_props_in_dict(labels_3D)
    
    #make sure it is a dict
    assert type(output_3D) is dict
    #make sure we get at least one output, there should be at least one label.
    assert len(output_3D) > 0

    # make sure it works for 2D data
    labels_2D = skim.label(test_2D_data.values[0])
    
    output_2D = feat_detect.get_label_props_in_dict(labels_2D)
    
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
    
    region_props_3D = feat_detect.get_label_props_in_dict(labels_3D)
    region_props_2D = feat_detect.get_label_props_in_dict(labels_2D)

    #get_indices_of_labels_from_reg_prop_dict 
    
    [curr_loc_indices, z_indices, y_indices, x_indices] = feat_detect.get_indices_of_labels_from_reg_prop_dict(region_props_3D)

    for index_key in curr_loc_indices:
        # there should be at least one value in each.
        assert curr_loc_indices[index_key] > 0
        
        assert np.all(z_indices[index_key] >= 0) and np.all(z_indices[index_key] < nz_3D)
        assert np.all(x_indices[index_key] >= 0) and np.all(x_indices[index_key] < nx_3D)
        assert np.all(y_indices[index_key] >= 0) and np.all(y_indices[index_key] < ny_3D)
    
    [curr_loc_indices, y_indices, x_indices] = feat_detect.get_indices_of_labels_from_reg_prop_dict(region_props_2D)

    for index_key in curr_loc_indices:
        # there should be at least one value in each.
        assert curr_loc_indices[index_key] > 0
        
        assert np.all(x_indices[index_key] >= 0) and np.all(x_indices[index_key] < nx_2D)
        assert np.all(y_indices[index_key] >= 0) and np.all(y_indices[index_key] < ny_2D)
    



def test_feature_detection_multithreshold_timestep():
    '''
    Tests ```tobac.feature_detection.feature_detection_multithreshold_timestep
    '''
    import numpy as np
    from tobac import testing
    from tobac import feature_detection

    # start by building a simple dataset with a single feature and seeing
    # if we identify it

    test_dset_size = (50, 50)
    test_hdim_1_pt = 20.0
    test_hdim_2_pt = 20.0
    test_hdim_1_sz = 5
    test_hdim_2_sz = 5
    test_amp = 2
    test_threshs = [
        1.5,
    ]
    test_min_num = 2

    test_data = np.zeros(test_dset_size)
    test_data = testing.make_feature_blob(
        test_data,
        test_hdim_1_pt,
        test_hdim_2_pt,
        h1_size=test_hdim_1_sz,
        h2_size=test_hdim_2_sz,
        amplitude=test_amp,
    )
    test_data_iris = testing.make_dataset_from_arr(test_data, data_type="iris")
    fd_output = feature_detection.feature_detection_multithreshold_timestep(
        test_data_iris, 0, threshold=test_threshs, min_num=test_min_num
    )

    # Make sure we have only one feature
    assert len(fd_output.index) == 1
    # Make sure that the location of the feature is correct
    assert fd_output.iloc[0]["hdim_1"] == pytest.approx(test_hdim_1_pt)
    assert fd_output.iloc[0]["hdim_2"] == pytest.approx(test_hdim_2_pt)
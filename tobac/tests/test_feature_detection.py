import tobac.testing
import tobac.feature_detection as feat_detect
import pytest



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
    test_threshs = [1.5, ]
    test_min_num = 2

    test_data = np.zeros(test_dset_size)
    test_data = testing.make_feature_blob(test_data, test_hdim_1_pt, test_hdim_2_pt, h1_size=test_hdim_1_sz, h2_size=test_hdim_2_sz, amplitude=test_amp)
    test_data_iris = testing.make_dataset_from_arr(test_data, data_type='iris')
    fd_output = feature_detection.feature_detection_multithreshold_timestep(test_data_iris, 0, 
        threshold=test_threshs, min_num=test_min_num)

    # Make sure we have only one feature
    assert len(fd_output.index) == 1
    # Make sure that the location of the feature is correct
    assert fd_output.iloc[0]['hdim_1'] == pytest.approx(test_hdim_1_pt)
    assert fd_output.iloc[0]['hdim_2'] == pytest.approx(test_hdim_2_pt)
    

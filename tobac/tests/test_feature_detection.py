import tobac.testing
import tobac.feature_detection as feat_detect




def test_feature_detection_multithreshold_timestep():
    '''
    Tests ```tobac.feature_detection.feature_detection_multithreshold_timestep
    '''
    import numpy as np
    from tobac import testing
    from tobac import feature_detection 
    # start by building a simple dataset with a single feature and seeing 
    # if we identify it
    
    test_data = np.zeros((50,50))
    test_data = testing.make_feature_blob(test_data, 20, 20, h1_size=5, h2_size=5, amplitude=2)
    test_data_iris = testing.make_dataset_from_arr(test_data, data_type='iris')
    fd_output = feature_detection.feature_detection_multithreshold_timestep(test_data_iris, 0, 
        threshold=[1.5,], min_num=2)

    # Make sure we have only one feature
    assert len(fd_output.index) == 1
    # Make sure that the location of the feature is correct
    assert fd_output.iloc[0]['hdim_1'] == 19.5
    assert fd_output.iloc[0]['hdim_2'] == 19.5
    
    pass
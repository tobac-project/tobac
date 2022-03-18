import tobac.testing as testing
import tobac.segmentation as seg

def test_segmentation_timestep_2D_feature_2D_seg():
    ''' Tests `tobac.segmentation.segmentation_timestep` with a 2D
    input feature and a 2D segmentation array
    '''
    # Before we can run segmentation, we must run feature detection. 

    # start by building a simple dataset with a single feature 
    import numpy as np
    
    test_dset_size = (50, 50)
    test_hdim_1_pt = 20.0
    test_hdim_2_pt = 20.0
    test_hdim_1_sz = 5
    test_hdim_2_sz = 5
    hdim_1_start_feat = int(np.ceil(test_hdim_1_pt - test_dset_size[0] / 2))
    hdim_1_end_feat = int(np.ceil(test_hdim_1_pt + test_dset_size[0] / 2))

    hdim_2_start_feat = int(np.ceil(test_hdim_2_pt - test_dset_size[1] / 2))
    hdim_2_end_feat = int(np.ceil(test_hdim_2_pt + test_dset_size[1] / 2))

    test_amp = 2

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
    # Generate dummy feature dataset
    test_feature_ds = testing.generate_single_feature(start_h1 = 20.0, start_h2 = 20.0)
    
    out_seg_mask, out_df = seg.segmentation_timestep(test_data_iris, test_feature_ds,
                                                        threshold = 1.5, PBC_flag='none', )
    
    # Make sure that all labeled points are segmented
    assert np.all(out_seg_mask[hdim_1_start_feat:hdim_1_end_feat, 
    hdim_2_start_feat:hdim_2_end_feat] == np.ones((test_hdim_1_sz, test_hdim_2_sz)))
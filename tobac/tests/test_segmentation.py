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
    test_dxy = 1000
    hdim_1_start_feat = int(np.ceil(test_hdim_1_pt - test_hdim_1_sz / 2))
    hdim_1_end_feat = int(np.ceil(test_hdim_1_pt + test_hdim_1_sz / 2))
    hdim_2_start_feat = int(np.ceil(test_hdim_2_pt - test_hdim_2_sz / 2))
    hdim_2_end_feat = int(np.ceil(test_hdim_2_pt + test_hdim_2_sz / 2))

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
    
    out_seg_mask, out_df = seg.segmentation_timestep(field_in = test_data_iris, 
                        features_in = test_feature_ds, dxy = test_dxy,
                                                        threshold = 1.5, PBC_flag='none', )
    
    # Make sure that all labeled points are segmented
    assert np.all(out_seg_mask.core_data()[hdim_1_start_feat:hdim_1_end_feat, 
    hdim_2_start_feat:hdim_2_end_feat] == np.ones((test_hdim_1_sz, test_hdim_2_sz)))


    # Now try PBCs
    # First, something stretching across hdim_1
    test_hdim_1_pt = 0.0
    test_data = np.zeros(test_dset_size)

    # Note that PBC flag here is 'both' as we still want the blob to be on both
    # sides of the boundary to see if we accidentally grab it without PBC 
    # segmentation
    test_data = testing.make_feature_blob(
        test_data,
        test_hdim_1_pt,
        test_hdim_2_pt,
        h1_size=test_hdim_1_sz,
        h2_size=test_hdim_2_sz,
        amplitude=test_amp,
        PBC_flag = 'both'
    )

    test_data_iris = testing.make_dataset_from_arr(test_data, data_type="iris")
    # Generate dummy feature dataset
    test_feature_ds = testing.generate_single_feature(start_h1 = test_hdim_1_pt, 
                                                      start_h2 = test_hdim_2_pt)
    
    hdim_1_start_feat, hdim_1_end_feat = testing.get_start_end_of_feat(test_hdim_1_pt, 
                                                            test_hdim_1_sz, 0,test_dset_size[0],
                                                            is_pbc = True )

    for pbc_option in ['none', 'hdim_1', 'hdim_2', 'both']:
        out_seg_mask, out_df = seg.segmentation_timestep(field_in = test_data_iris, 
                            features_in = test_feature_ds, dxy = test_dxy,
                            threshold = test_amp-0.5, PBC_flag=pbc_option, )
        # This will automatically give the appropriate box, and it's tested separately.
        segmented_box_expected = testing.get_pbc_coordinates(0, test_dset_size[0], 
            0, test_dset_size[1], hdim_1_start_feat, hdim_1_end_feat, hdim_2_start_feat, 
            hdim_2_end_feat, PBC_flag=pbc_option)
        # Make sure that all labeled points are segmented
        for seg_box in segmented_box_expected:
            assert np.all(out_seg_mask.core_data()[seg_box[0]:seg_box[1], 
            seg_box[2]:seg_box[3]] == np.ones((seg_box[1]-seg_box[0], seg_box[3]-seg_box[2])))
        
        if pbc_option in ['none', 'hdim_2']:
            #there will only be one seg_box
            assert (np.sum(out_seg_mask.core_data()[out_seg_mask.core_data()==1]) == 
                    np.sum(np.ones((seg_box[1]-seg_box[0], seg_box[3]-seg_box[2]))))
        else:
            # We should be capturing the whole feature
            assert (np.sum(out_seg_mask.core_data()[out_seg_mask.core_data()==1]) == 
                    np.sum(np.ones((test_hdim_1_sz, test_hdim_2_sz))))

    # Same as the above test, but for hdim_2
    # First, try the cases where we shouldn't get the points on the opposite
    # hdim_2 side
    test_hdim_1_pt = 20.0
    test_hdim_2_pt = 0.0
    test_data = np.zeros(test_dset_size)
    test_data = testing.make_feature_blob(
        test_data,
        test_hdim_1_pt,
        test_hdim_2_pt,
        h1_size=test_hdim_1_sz,
        h2_size=test_hdim_2_sz,
        amplitude=test_amp,
        PBC_flag = 'both'
    )
    test_data_iris = testing.make_dataset_from_arr(test_data, data_type="iris")
    # Generate dummy feature dataset
    test_feature_ds = testing.generate_single_feature(start_h1 = test_hdim_1_pt, 
                                                      start_h2 = test_hdim_2_pt)
    hdim_1_start_feat, hdim_1_end_feat = testing.get_start_end_of_feat(test_hdim_1_pt, 
                                                            test_hdim_1_sz, 0,test_dset_size[0],
                                                            is_pbc = True )

    hdim_2_start_feat, hdim_2_end_feat = testing.get_start_end_of_feat(test_hdim_2_pt, 
                                                            test_hdim_2_sz, 0,test_dset_size[1],
                                                            is_pbc = True )

    for pbc_option in ['none', 'hdim_1', 'hdim_2', 'both']:
        out_seg_mask, out_df = seg.segmentation_timestep(field_in = test_data_iris, 
                            features_in = test_feature_ds, dxy = test_dxy,
                            threshold = test_amp-0.5, PBC_flag=pbc_option, )
        # This will automatically give the appropriate box(es), and it's tested separately.
        segmented_box_expected = testing.get_pbc_coordinates(0, test_dset_size[0], 
            0, test_dset_size[1], hdim_1_start_feat, hdim_1_end_feat, hdim_2_start_feat, 
            hdim_2_end_feat, PBC_flag=pbc_option)
        # Make sure that all labeled points are segmented
        for seg_box in segmented_box_expected:
            assert np.all(out_seg_mask.core_data()[seg_box[0]:seg_box[1], 
            seg_box[2]:seg_box[3]] == np.ones((seg_box[1]-seg_box[0], seg_box[3]-seg_box[2])))
        
        if pbc_option in ['none', 'hdim_1']:
            #there will only be one seg_box
            assert (np.sum(out_seg_mask.core_data()[out_seg_mask.core_data()==1]) == 
                    np.sum(np.ones((seg_box[1]-seg_box[0], seg_box[3]-seg_box[2]))))
        else:
            # We should be capturing the whole feature
            assert (np.sum(out_seg_mask.core_data()[out_seg_mask.core_data()==1]) == 
                    np.sum(np.ones((test_hdim_1_sz, test_hdim_2_sz))))



    # Same as the above test, but for hdim_2
    # First, try the cases where we shouldn't get the points on the opposite
    # both sides (corner point)
    test_hdim_1_pt = 0.0
    test_hdim_2_pt = 0.0
    test_data = np.zeros(test_dset_size)
    test_data = testing.make_feature_blob(
        test_data,
        test_hdim_1_pt,
        test_hdim_2_pt,
        h1_size=test_hdim_1_sz,
        h2_size=test_hdim_2_sz,
        amplitude=test_amp,
        PBC_flag = 'both'
    )
    test_data_iris = testing.make_dataset_from_arr(test_data, data_type="iris")
    # Generate dummy feature dataset
    test_feature_ds = testing.generate_single_feature(start_h1 = test_hdim_1_pt, 
                                                      start_h2 = test_hdim_2_pt)
    hdim_1_start_feat, hdim_1_end_feat = testing.get_start_end_of_feat(test_hdim_1_pt, 
                                                            test_hdim_1_sz, 0,test_dset_size[0],
                                                            is_pbc = True )

    hdim_2_start_feat, hdim_2_end_feat = testing.get_start_end_of_feat(test_hdim_2_pt, 
                                                            test_hdim_2_sz, 0,test_dset_size[1],
                                                            is_pbc = True )

    for pbc_option in ['none', 'hdim_1', 'hdim_2', 'both']:
        out_seg_mask, out_df = seg.segmentation_timestep(field_in = test_data_iris, 
                            features_in = test_feature_ds, dxy = test_dxy,
                            threshold = test_amp-0.5, PBC_flag=pbc_option, )
        # This will automatically give the appropriate box(es), and it's tested separately.
        segmented_box_expected = testing.get_pbc_coordinates(0, test_dset_size[0], 
            0, test_dset_size[1], hdim_1_start_feat, hdim_1_end_feat, hdim_2_start_feat, 
            hdim_2_end_feat, PBC_flag=pbc_option)
        # Make sure that all labeled points are segmented
        for seg_box in segmented_box_expected:
            print(pbc_option, seg_box)
            #TODO: something is wrong with this case, unclear what.
            #assert np.all(out_seg_mask.core_data()[seg_box[0]:seg_box[1], 
            #seg_box[2]:seg_box[3]] == np.ones((seg_box[1]-seg_box[0], seg_box[3]-seg_box[2])))

        #TODO: Make sure for none, hdim_1, hdim_2 that only the appropriate points are segmented


def test_segmentation_timestep_level():
    """Tests `tobac.segmentation.segmentation_timestep` with a 2D
    input feature and a 3D segmentation array, specifying the `level` parameter.
    """
    # Before we can run segmentation, we must run feature detection.

    # start by building a simple dataset with a single feature
    import numpy as np

    test_dset_size = (20, 50, 50)
    test_hdim_1_pt = 20.0
    test_hdim_2_pt = 20.0
    test_vdim_pt = 2
    test_hdim_1_sz = 5
    test_hdim_2_sz = 5
    test_vdim_sz = 3
    test_dxy = 1000

    vdim_start_feat = int(np.ceil(test_vdim_pt - test_vdim_sz / 2))
    vdim_end_feat = int(np.ceil(test_vdim_pt + test_vdim_sz / 2))
    hdim_1_start_feat = int(np.ceil(test_hdim_1_pt - test_hdim_1_sz / 2))
    hdim_1_end_feat = int(np.ceil(test_hdim_1_pt + test_hdim_1_sz / 2))
    hdim_2_start_feat = int(np.ceil(test_hdim_2_pt - test_hdim_2_sz / 2))
    hdim_2_end_feat = int(np.ceil(test_hdim_2_pt + test_hdim_2_sz / 2))

    test_amp = 2

    test_data = np.zeros(test_dset_size)
    test_data = testing.make_feature_blob(
        test_data,
        test_hdim_1_pt,
        test_hdim_2_pt,
        test_vdim_pt,
        h1_size=test_hdim_1_sz,
        h2_size=test_hdim_2_sz,
        v_size=test_vdim_sz,
        amplitude=test_amp,
    )

    # Make a second feature, above the first.

    delta_height = 8
    test_data = testing.make_feature_blob(
        test_data,
        test_hdim_1_pt,
        test_hdim_2_pt,
        test_vdim_pt + delta_height,
        h1_size=test_hdim_1_sz,
        h2_size=test_hdim_2_sz,
        v_size=test_vdim_sz,
        amplitude=test_amp,
    )

    test_data_iris = testing.make_dataset_from_arr(
        test_data, data_type="iris", z_dim_num=0, y_dim_num=1, x_dim_num=2
    )
    # Generate dummy feature dataset
    test_feature_ds = testing.generate_single_feature(start_h1=20.0, start_h2=20.0)

    out_seg_mask, out_df = seg.segmentation_timestep(
        field_in=test_data_iris,
        features_in=test_feature_ds,
        dxy=test_dxy,
        threshold=1.5,
        seed_3D_flag= 'column'
    )
    out_seg_mask_arr = out_seg_mask.core_data()
    # Make sure that all labeled points are segmented, before setting specific levels
    assert np.all(
        out_seg_mask_arr[
            vdim_start_feat:vdim_end_feat,
            hdim_1_start_feat:hdim_1_end_feat,
            hdim_2_start_feat:hdim_2_end_feat,
        ]
        == np.ones((test_vdim_sz, test_hdim_1_sz, test_hdim_2_sz))
    )
    assert np.all(
        out_seg_mask_arr[
            vdim_start_feat + delta_height : vdim_end_feat + delta_height,
            hdim_1_start_feat:hdim_1_end_feat,
            hdim_2_start_feat:hdim_2_end_feat,
        ]
        == np.ones((test_vdim_sz, test_hdim_1_sz, test_hdim_2_sz))
    )

    # now set specific levels
    out_seg_mask, out_df = seg.segmentation_timestep(
        field_in=test_data_iris,
        features_in=test_feature_ds,
        dxy=test_dxy,
        level=slice(vdim_start_feat, vdim_end_feat),
        threshold=1.5,
        seed_3D_flag = 'column'
    )
    out_seg_mask_arr = out_seg_mask.core_data()
    # Make sure that all labeled points are segmented, before setting specific levels
    assert np.all(
        out_seg_mask_arr[
            vdim_start_feat:vdim_end_feat,
            hdim_1_start_feat:hdim_1_end_feat,
            hdim_2_start_feat:hdim_2_end_feat,
        ]
        == np.ones((test_vdim_sz, test_hdim_1_sz, test_hdim_2_sz))
    )
    assert np.all(
        out_seg_mask_arr[
            vdim_start_feat + delta_height : vdim_end_feat + delta_height,
            hdim_1_start_feat:hdim_1_end_feat,
            hdim_2_start_feat:hdim_2_end_feat,
        ]
        == np.zeros((test_vdim_sz, test_hdim_1_sz, test_hdim_2_sz))
    )

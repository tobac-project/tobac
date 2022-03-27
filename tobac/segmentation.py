import logging

from numpy import transpose
from . import utils as tb_utils
        
def transfm_pbc_point(in_dim, dim_min, dim_max):
    '''Function to transform a PBC-feature point for contiguity
    
    Parameters
    ----------
    in_dim : int
        Input coordinate to adjust
    dim_min : int
        Minimum point for the dimension
    dim_max : int
        Maximum point for the dimension (inclusive)
    
    Returns
    -------
    int
        The transformed point
    
    '''
    if in_dim < ((dim_min+dim_max)/2):
        return in_dim+dim_max+1
    else:
        return in_dim

def segmentation_3D(features,field,dxy,threshold=3e-3,target='maximum',level=None,method='watershed',max_distance=None,PBC_flag='none',seed_3D_flag='column'):
    return segmentation(features,field,dxy,threshold=threshold,target=target,level=level,method=method,max_distance=max_distance,PBC_flag=PBC_flag,seed_3D_flag='column')

def segmentation_2D(features,field,dxy,threshold=3e-3,target='maximum',level=None,method='watershed',max_distance=None,PBC_flag='none',seed_3D_flag='column'):
    return segmentation(features,field,dxy,threshold=threshold,target=target,level=level,method=method,max_distance=max_distance,PBC_flag=PBC_flag,seed_3D_flag='column')


def segmentation_timestep(field_in,features_in,dxy,threshold=3e-3,target='maximum',level=None,method='watershed',max_distance=None,vertical_coord='auto',PBC_flag='none',seed_3D_flag='column', seed_3D_size=5):    
    """Function performing watershedding for an individual timestep of the data
    
    Parameters
    ----------
    features:   pandas.DataFrame 
                features for one specific point in time
    field:      iris.cube.Cube 
                input field to perform the watershedding on (2D or 3D for one specific point in time)
    threshold:  float 
                threshold for the watershedding field to be used for the mas
    target:     string
                switch to determine if algorithm looks strating from maxima or minima in input field (maximum: starting from maxima (default), minimum: starting from minima)
    level       slice
                levels at which to seed the cells for the watershedding algorithm
    method:     string
                flag determining the algorithm to use (currently watershedding implemented)
    max_distance: float
                  maximum distance from a marker allowed to be classified as belonging to that cell
    PBC_flag:   string
                options: 'none' (default), 'hdim_1', 'hdim_2', 'both'
                flag indicating whether to use PBC treatment or not
                note to self: should be expanded to account for singly periodic boundaries also
                rather than just doubly periodic
    seed_3D_flag: str('column', 'box')
        Seed 3D field at feature positions with either the full column (default)
         or a box of user-set size
    seed_3D_size: int or tuple (dimensions equal to dimensions of `field`)
        This sets the size of the seed box when `seed_3D_flag` is 'box'. If it's an 
        integer, the seed box is identical in all dimensions. If it's a tuple, it specifies the 
        seed area for each dimension separately. Note: we recommend the use
        of odd numbers for this. If you give an even number, your seed box will be 
        biased and not centered around the feature. 
    
    Returns
    -------
    iris.cube.Cube
                      cloud mask, 0 outside and integer numbers according to track inside the clouds
    pandas.DataFrame
                  feature dataframe including the number of cells (2D or 3D) in the segmented area/volume of the feature at the timestep
    """
    from skimage.segmentation import watershed
    import skimage.measure
    from scipy.ndimage import distance_transform_edt
    from copy import deepcopy
    import numpy as np
    import iris

    # How many dimensions are we using?
    if field_in.ndim==2:
        hdim_1_axis = 0
        hdim_2_axis = 1
        is_3D_seg = False
    elif field_in.ndim == 3:
        is_3D_seg = True
        # Find which coordinate is the z coordinate
        list_coord_names=[coord.name() for coord in field_in.coords()]
        #determine vertical axis:
        if vertical_coord=='auto':
            list_vertical=['z','model_level_number','altitude','geopotential_height']
            # TODO: there surely must be a better way to handle this
            vertical_axis = None
            for coord_name in list_vertical:
                if coord_name in list_coord_names:
                    vertical_axis=coord_name
                    break
            if vertical_axis is None:
                raise ValueError('Please specify vertical coordinate')
        elif vertical_coord in list_coord_names:
            vertical_axis=vertical_coord
        else:
            raise ValueError('Please specify vertical coordinate')
        ndim_vertical=field_in.coord_dims(vertical_axis)
        if len(ndim_vertical)>1:
            raise ValueError('please specify 1 dimensional vertical coordinate')
        vertical_coord_axis = ndim_vertical[0]
        # Once we know the vertical coordinate, we can resolve the 
        # horizontal coordinates
        # To make things easier, we will transpose the axes
        # so that they are consistent. 
        if vertical_coord_axis == 0:
            hdim_1_axis = 1
            hdim_2_axis = 2
        elif vertical_coord_axis == 1:
            hdim_1_axis = 0
            hdim_2_axis = 2
        elif vertical_coord_axis == 2:
            hdim_1_axis = 0
            hdim_2_axis = 1
    else:
        raise ValueError('Segmentation routine only possible with 2 or 3 spatial dimensions')

    
    # copy feature dataframe for output 
    features_out=deepcopy(features_in)
    # Create cube of the same dimensions and coordinates as input data to store mask:        
    segmentation_out=1*field_in
    segmentation_out.rename('segmentation_mask')
    segmentation_out.units=1

    # Get raw array from input data:
    data=field_in.core_data()
    is_3D_seg = len(data.shape)==3
    # To make things easier, we will transpose the axes
    # so that they are consistent: z, hdim_1, hdim_2
    # We only need to do this for 3D. 
    transposed_data = False
    if is_3D_seg:
        if vertical_coord_axis == 1:
            data = np.transpose(data, axes=(1, 0, 2))
            transposed_data = True
        elif vertical_coord_axis == 2:
            data = np.transpose(data, axes=(2, 0, 1))
            transposed_data = True

    
    #Set level at which to create "Seed" for each feature in the case of 3D watershedding:
    # If none, use all levels (later reduced to the ones fulfilling the theshold conditions)
    if level==None:
        level=slice(None)

    # transform max_distance in metres to distance in pixels:
    if max_distance is not None:
        max_distance_pixel=np.ceil(max_distance/dxy)

    # mask data outside region above/below threshold and invert data if tracking maxima:
    if target == 'maximum':
        unmasked=data>threshold
        data_segmentation=-1*data
    elif target == 'minimum':
        unmasked=data<threshold
        data_segmentation=data
    else:
        raise ValueError('unknown type of target')

    # set markers at the positions of the features:
    markers = np.zeros(unmasked.shape).astype(np.int32)
    if not is_3D_seg: #2D watershedding        
        for index, row in features_in.iterrows():
            markers[int(row['hdim_1']), int(row['hdim_2'])]=row['feature']

    elif is_3D_seg: #3D watershedding
        
        # We need to generate seeds in 3D. 
        # TODO: I think it would be easier to transpose the input to always be
        # z, h1, h2. 
        if (seed_3D_flag == 'column'):
            for index, row in features_in.iterrows():
                markers[level,int(row['hdim_1']), int(row['hdim_2'])]=row['feature']
                    
        elif (seed_3D_flag == 'box'):
            z_len = data.shape[0]
            h1_len = data.shape[1]
            h2_len = data.shape[2]
        
            # Get the size of the seed box from the input parameter
            try:
                seed_z = seed_3D_size[0]
                seed_h1 = seed_3D_size[1]
                seed_h2 = seed_3D_size[2]
            except TypeError:
                # Not iterable, assume int. 
                seed_z = seed_3D_size
                seed_h1 = seed_3D_size
                seed_h2 = seed_3D_size

            # Can we use our testing function to generate 3D boxes (with PBC awareness)
            # for a faster version of this?
            for index, row in features_in.iterrows():
                try: 
                    row['vdim']
                except KeyError:
                    raise ValueError("For Box seeding, you must have a 3D input"
                    "source.")
                
                # Because we don't support PBCs on the vertical axis,
                # this is simple- just go in the seed_z/2 points around the 
                # vdim of the feature, up to the limits of the array. 
                z_seed_start = int(np.max([0, np.ceil(row['vdim']-seed_z/2)]))
                z_seed_end = int(np.min([z_len, np.ceil(row['vdim']+seed_z/2)]))
                
                # For the horizontal dimensions, it's more complicated if we have
                # PBCs. 
                hdim_1_min = int(np.ceil(row['hdim_1'] - seed_h1/2))
                hdim_1_max = int(np.ceil(row['hdim_1'] + seed_h1/2))
                hdim_2_min = int(np.ceil(row['hdim_2'] - seed_h2/2))
                hdim_2_max = int(np.ceil(row['hdim_2'] + seed_h2/2))

                all_seed_boxes = tb_utils.get_pbc_coordinates(
                    hdim_1_min, hdim_1_max, hdim_2_min, hdim_2_max,
                    0, h1_len, 0, h2_len, PBC_flag= PBC_flag
                )
                for seed_box in all_seed_boxes:
                    markers[z_seed_start:z_seed_end,
                            seed_box[0]:seed_box[1], 
                            seed_box[2]:seed_box[3]]=row['feature']
                                            

    # set markers in cells not fulfilling threshold condition to zero:
    markers[~unmasked]=0
    #marker_vals = np.unique(markers)
  
    # Turn into np arrays (not necessary for markers) as dask arrays don't yet seem to work for watershedding algorithm
    data_segmentation=np.array(data_segmentation)
    unmasked=np.array(unmasked)

    # perform segmentation:
    if method=='watershed':
        segmentation_mask = watershed(np.array(data_segmentation),markers.astype(np.int32), mask=unmasked)

    else:                
        raise ValueError('unknown method, must be watershed')

    # remove everything from the individual masks that is more than max_distance_pixel away from the markers
    if max_distance is not None:
        D=distance_transform_edt((markers==0).astype(int))
        segmentation_mask[np.bitwise_and(segmentation_mask>0, D>max_distance_pixel)]=0
        
    #mask all segmentation_mask points below threshold as -1
    #to differentiate from those unmasked points NOT filled by watershedding
    # TODO: allow user to specify
    segmentation_mask[~unmasked] = -1
    
    #saves/prints below for testing
    seg_m_data = segmentation_mask[:]
        
    
    hdim1_min = 0
    hdim1_max = segmentation_mask.shape[hdim_1_axis] - 1
    hdim2_min = 0
    hdim2_max = segmentation_mask.shape[hdim_2_axis] - 1
    
    # all options that involve dealing with periodic boundaries
    pbc_options = ['hdim_1', 'hdim_2', 'both']
    # Only run this if we need to deal with PBCs
    if PBC_flag in pbc_options:

        # read in labeling/masks and region-finding functions
        reg_props_dict = tb_utils.get_label_props_in_dict(seg_m_data)

        if not is_3D_seg:
            # let's transpose segmentation_mask to a 1,y,x array to make calculations etc easier.
            segmentation_mask = segmentation_mask[np.newaxis, :, :]
            unmasked = unmasked[np.newaxis, :, :]
            data_segmentation = data_segmentation[np.newaxis, :, :]
            vertical_coord_axis = 0
            hdim_1_axis = 1
            hdim_2_axis = 2


        seg_mask_unseeded = np.zeros(segmentation_mask.shape)


        # Return all indices where segmentation field == 0
        # meaning unfilled but above threshold
        # TODO: is there a way to do this without np.where?
        vdim_unf,hdim1_unf,hdim2_unf = np.where(segmentation_mask==0)
        seg_mask_unseeded[vdim_unf,hdim1_unf,hdim2_unf]=1            
    
        # create labeled field of unfilled, unseeded features
        labels_unseeded,label_num = skimage.measure.label(seg_mask_unseeded, return_num=True)
        
        markers_2 = np.zeros(data_segmentation.shape).astype(np.int32)
        
        #new, shorter PBC marker seeding approach
        #loop thru LB points
        #then check if fillable region (labels_unseeded > 0)
        #then check if point on other side of boundary is > 0 in segmentation_mask
        '''
        "First pass" at seeding features across the boundaries. This first pass will bring in
        eligible (meaning values that are higher than threshold) but not previously watershedded 
        points across the boundary by seeding them with the appropriate feature across the boundary.

        Later, we will run the second pass or "buddy box" approach that handles cases where points across the boundary
        have been watershedded already. 
        '''

        # TODO: clean up code. 
        if PBC_flag == 'hdim_1' or PBC_flag == 'both':
            for vdim_ind in range(0,segmentation_mask.shape[0]):
                for hdim1_ind in [hdim1_min,hdim1_max]:
                    for hdim2_ind in range(hdim2_min,hdim2_max):
                
                
                        if(labels_unseeded[vdim_ind,hdim1_ind,hdim2_ind] == 0):
                            continue
                        else:
                            if hdim1_ind == 0:
                                if (segmentation_mask[vdim_ind,hdim1_max,hdim2_ind]<=0):
                                    continue
                                else:
                                    markers_2[vdim_ind,hdim1_ind,hdim2_ind] = segmentation_mask[vdim_ind,hdim1_max,hdim2_ind]
                            elif hdim1_ind == hdim1_max:
                                if (segmentation_mask[vdim_ind,hdim1_min,hdim2_ind]<=0):
                                    continue
                                else:
                                    markers_2[vdim_ind,hdim1_ind,hdim2_ind] = segmentation_mask[vdim_ind,hdim1_min,hdim2_ind]
        if PBC_flag == 'hdim_2' or PBC_flag == 'both':
            # TODO: This seems quite slow, is there scope for further speedup?
            for vdim_ind in range(0,segmentation_mask.shape[0]):
                for hdim1_ind in range(hdim1_min,hdim1_max):
                    for hdim2_ind in [hdim2_min,hdim2_max]:
                
                        if(labels_unseeded[vdim_ind,hdim1_ind,hdim2_ind] == 0):
                            continue
                        else:
                            if hdim2_ind == hdim2_min:
                                if (segmentation_mask[vdim_ind,hdim1_ind,hdim2_max]<=0):
                                    continue
                                else:
                                    markers_2[vdim_ind,hdim1_ind,hdim2_ind] = segmentation_mask[vdim_ind,hdim1_ind,hdim2_max]
                                    #print(z_ind,y_ind,x_ind)
                                    #print("seeded")
                            elif hdim2_ind == hdim2_max:
                                if (segmentation_mask[vdim_ind,hdim1_ind,hdim2_min]<=0):
                                    continue
                                else:
                                    markers_2[vdim_ind,hdim1_ind,hdim2_ind] = segmentation_mask[vdim_ind,hdim1_ind,hdim2_min]
                                    #print(z_ind,y_ind,x_ind)
                                    #print("seeded")
            
        # Deal with the opposite corner only
        if PBC_flag == 'both':
            # TODO: This seems quite slow, is there scope for further speedup?
            for vdim_ind in range(0,segmentation_mask.shape[0]):
                for hdim1_ind in [hdim1_min, hdim1_max]:
                    for hdim2_ind in [hdim2_min,hdim2_max]:
                        # If this point is unseeded and unlabeled
                        if(labels_unseeded[vdim_ind,hdim1_ind,hdim2_ind] == 0):
                            continue
                        
                        # Find the opposite point in hdim1 space
                        hdim1_opposite_corner = (hdim1_min if hdim1_ind == hdim1_max else hdim1_max)
                        hdim2_opposite_corner = (hdim2_min if hdim2_ind == hdim2_max else hdim2_max)
                        if segmentation_mask[vdim_ind, hdim1_opposite_corner, hdim2_opposite_corner] <= 0:
                            continue

                        markers_2[vdim_ind, hdim1_ind, hdim2_ind] = segmentation_mask[vdim_ind,hdim1_opposite_corner,hdim2_opposite_corner]

        markers_2[~unmasked]=0
        
        if method=='watershed':
            segmentation_mask_2 = watershed(data_segmentation,markers_2.astype(np.int32), mask=unmasked)
        else:                
            raise ValueError('unknown method, must be watershed')

        # remove everything from the individual masks that is more than max_distance_pixel away from the markers
        if max_distance is not None:
            D=distance_transform_edt((markers==0).astype(int))
            segmentation_mask_2[np.bitwise_and(segmentation_mask_2>0, D>max_distance_pixel)]=0
    
        # Sum up original mask and secondary PBC-mask for full PBC segmentation
        segmentation_mask_3=segmentation_mask + segmentation_mask_2
                    
        # Secondary seeding complete, now blending periodic boundaries
        # keep segmentation mask fields for now so we can save these all later
        # for demos of changes
                
        #update mask coord regions

        '''
        Now, start the second round of watershedding- the "buddy box" approach
        buddies contains features of interest and any neighbors that across the boundary or in 
        physical contact with that label
        '''
        # TODO: this can cause a crash if there are no segmentation regions        
        reg_props_dict = tb_utils.get_label_props_in_dict(segmentation_mask_3)

        curr_reg_inds, z_reg_inds, y_reg_inds, x_reg_inds= tb_utils.get_indices_of_labels_from_reg_prop_dict(reg_props_dict)

        wall_labels = np.array([])

        # TODO: move indices around with specific z axis
        w_wall = np.unique(segmentation_mask_3[:,:,0])
        wall_labels = np.append(wall_labels,w_wall)

        # TODO: add test case that tests buddy box
        #e_wall = np.unique(segmentation_mask_3[:,:,-1])
        #wall_labels = np.append(wall_labels,e_wall)

        #n_wall = np.unique(segmentation_mask_3[:,-1,:])
        #wall_labels = np.append(wall_labels,n_wall)

        s_wall = np.unique(segmentation_mask_3[:,0,:])
        wall_labels = np.append(wall_labels,s_wall)

        wall_labels = np.unique(wall_labels)
        wall_labels = wall_labels[(wall_labels) > 0].astype(int)
        #print(wall_labels)
        
        # Loop through all segmentation mask labels on the wall 
        for cur_idx in wall_labels:
            
            vdim_indices = z_reg_inds[cur_idx]
            hdim1_indices = y_reg_inds[cur_idx]
            hdim2_indices = x_reg_inds[cur_idx]
    
            #start buddies array with feature of interest
            buddies = np.array([cur_idx],dtype=int)
            # Loop through all points in the segmentation mask that we're intertested in
            for label_z, label_y, label_x in zip(vdim_indices, hdim1_indices, hdim2_indices):
                    
                # check if this is the special case of being a corner point. 
                # if it's doubly periodic AND on both x and y boundaries, it's a corner point
                # and we have to look at the other corner. 
                # here, we will only look at the corner point and let the below deal with x/y only. 
                if PBC_flag == 'both' and (np.any(label_y == [hdim1_min,hdim1_max]) and np.any(label_x == [hdim2_min,hdim2_max])):
                        
                    #adjust x and y points to the other side
                    y_val_alt = tb_utils.adjust_pbc_point(label_y, hdim1_min, hdim1_max)
                    x_val_alt = tb_utils.adjust_pbc_point(label_x, hdim2_min, hdim2_max)
                    label_on_corner = segmentation_mask_3[label_z,y_val_alt,x_val_alt]

                    if((label_on_corner > 0)):
                        #add opposite-corner buddy if it exists
                        buddies = np.append(buddies,label_on_corner)
                
                    
                # on the hdim1 boundary and periodic on hdim1
                if (PBC_flag == 'hdim_1' or PBC_flag == 'both') and np.any(label_y == [hdim1_min,hdim1_max]):                        
                    y_val_alt = tb_utils.adjust_pbc_point(label_y, hdim1_min, hdim1_max)

                    #get the label value on the opposite side
                    label_alt = segmentation_mask_3[label_z,y_val_alt,label_x]
                        
                    #if it's labeled and not already been dealt with
                    if((label_alt > 0)):
                        #add above/below buddy if it exists
                        buddies = np.append(buddies,label_alt)
                   
                if (PBC_flag == 'hdim_2' or PBC_flag == 'both') and np.any(label_x == [hdim2_min,hdim2_max]):                        
                    x_val_alt = tb_utils.adjust_pbc_point(label_x, hdim2_min, hdim2_max)

                    #get the seg value on the opposite side
                    label_alt = segmentation_mask_3[label_z,label_y,x_val_alt]
                        
                    #if it's labeled and not already been dealt with
                    if((label_alt > 0)):
                        #add left/right buddy if it exists
                        buddies = np.append(buddies,label_alt)
                        
            
            buddies = np.unique(buddies)
            
            if np.all(buddies==cur_idx):
                continue
            else:
                inter_buddies,feat_inds,buddy_inds=np.intersect1d(features_in.feature.values[:],buddies,return_indices=True)

            # Get features that are needed for the buddy box
            buddy_features = deepcopy(features_in.iloc[feat_inds])
        
            #create arrays to contain points of all buddies
            #and their transpositions/transformations
            #for use in Buddy Box space

            #z,y,x points in the grid domain with no transformations
            #NOTE: when I think about it, not sure if these are really needed
            # as we use the y_a1/x_a1 points for the data transposition
            # to the buddy box rather than these and their z2/y2/x2 counterparts
            buddy_z = np.array([],dtype=int)
            buddy_y = np.array([],dtype=int)
            buddy_x = np.array([],dtype=int)

            # z,y,x points from the grid domain WHICH MAY OR MAY NOT BE TRANSFORMED
            # so as to be continuous/contiguous across a grid boundary for that dimension
            #(e.g., instead of [1496,1497,0,1,2,3] it would be [1496,1497,1498,1499,1500,1501])
            buddy_z2 = np.array([],dtype=int)
            buddy_y2 = np.array([],dtype=int)
            buddy_x2 = np.array([],dtype=int)

            # These are just for feature positions and are in z2/y2/x2 space
            # (may or may not be within real grid domain)
            # so that when the buddy box is constructed, seeding is done properly
            # in the buddy box space

            #NOTE: We may not need this, as we already do this editing the buddy_features df
            # and an iterrows call through this is what's used to actually seed the buddy box
            buddy_zf = np.array([],dtype=int)
            buddy_yf = np.array([],dtype=int)
            buddy_xf = np.array([],dtype=int)
        
            buddy_looper = 0
            
            #loop thru buddies
            for buddy in buddies:
                
                #isolate feature from set of buddies
                buddy_feat = features_in[features_in['feature'] == buddy]

                #transform buddy feature position if needed for positioning in z2/y2/x2 space
                #MAY be redundant with what is done just below here        
                yf2 = transfm_pbc_point(int(buddy_feat.hdim_1), hdim1_min, hdim1_max)
                xf2 = transfm_pbc_point(int(buddy_feat.hdim_2), hdim2_min, hdim2_max)

                #edit value in buddy_features dataframe 
                buddy_features.hdim_1.values[buddy_looper] = transfm_pbc_point(float(buddy_feat.hdim_1), hdim1_min, hdim1_max)
                buddy_features.hdim_2.values[buddy_looper] = transfm_pbc_point(float(buddy_feat.hdim_2), hdim2_min, hdim2_max)
            
                #print(int(buddy_feat.vdim),yf2,xf2)
                #display(buddy_features)
                
                #again, this may be redundant as I don't think we use buddy_zf/yf/xf after this
                #in favor of iterrows thru the updated buddy_features
                buddy_zf = np.append(buddy_zf,int(buddy_feat.vdim))
                buddy_yf = np.append(buddy_yf,yf2)
                buddy_xf = np.append(buddy_xf,xf2)
            
                buddy_looper = buddy_looper+1
                # Create 1:1 map through actual domain points and continuous/contiguous points
                # used to identify buddy box dimension lengths for its construction
                for z,y,x in zip(z_reg_inds[buddy],y_reg_inds[buddy],x_reg_inds[buddy]):
                
                    buddy_z = np.append(buddy_z,z)
                    buddy_y = np.append(buddy_y,y)
                    buddy_x = np.append(buddy_x,x)
                                
                    y2 = transfm_pbc_point(y, hdim1_min, hdim1_max)
                    x2 = transfm_pbc_point(x, hdim2_min, hdim2_max)
                
                    buddy_z2 = np.append(buddy_z2,z)
                    buddy_y2 = np.append(buddy_y2,y2)
                    buddy_x2 = np.append(buddy_x2,x2)
                    
            # Buddy Box!
            # Indentify mins and maxes of buddy box continuous points range
            # so that box of correct size can be constructred
            bbox_zstart = int(np.min(buddy_z2))
            bbox_ystart = int(np.min(buddy_y2))
            bbox_xstart = int(np.min(buddy_x2))
            bbox_zend = int(np.max(buddy_z2)+1)
            bbox_yend = int(np.max(buddy_y2)+1)
            bbox_xend = int(np.max(buddy_x2)+1)
    
            bbox_zsize = bbox_zend - bbox_zstart
            bbox_ysize = bbox_yend - bbox_ystart
            bbox_xsize = bbox_xend - bbox_xstart
        
            #print(bbox_zsize,bbox_ysize,bbox_xsize)
    
            # Creation of actual Buddy Box space for transposition
            # of data in domain and re-seeding with Buddy feature markers
            buddy_rgn = np.zeros((bbox_zsize, bbox_ysize, bbox_xsize))
            ind_ctr = 0
        
            #need to loop thru ALL z,y,x inds in buddy box
            #not just the ones that have nonzero seg mask values

            # "_a1" points are re-transformations from the continuous buddy box points
            # back to original grid/domain space to ensure that the correct data are
            # copied to the proper Buddy Box locations
            for z in range(bbox_zstart,bbox_zend):
                for y in range(bbox_ystart,bbox_yend):
                    for x in range(bbox_xstart,bbox_xend):
                        z_a1 = z
                        if y > hdim1_max:
                            y_a1 = y - (hdim1_max + 1)
                        else:
                            y_a1 = y
                        
                        if x > hdim2_max:
                            x_a1 = x - (hdim2_max + 1)
                        else:
                            x_a1 = x

                        buddy_rgn[z-bbox_zstart,y-bbox_ystart,x-bbox_xstart] = field_in.data[z_a1,y_a1,x_a1]
                    
            
            #construction of iris cube corresponding to buddy box and its data
            #for marker seeding and watershedding of buddy box
            rgn_cube = iris.cube.Cube(data=buddy_rgn)
        
            coord_system=None
            # TODO: clean this up
            h2_coord=iris.coords.DimCoord(np.arange(bbox_xsize), long_name='hdim_2', units='1', bounds=None, attributes=None, coord_system=coord_system)
            h1_coord=iris.coords.DimCoord(np.arange(bbox_ysize), long_name='hdim_1', units='1', bounds=None, attributes=None, coord_system=coord_system)
            v_coord=iris.coords.DimCoord(np.arange(bbox_zsize), long_name='vdim', units='1', bounds=None, attributes=None, coord_system=coord_system)
        
            rgn_cube.add_dim_coord(h2_coord,2)
            rgn_cube.add_dim_coord(h1_coord,1)
            rgn_cube.add_dim_coord(v_coord,0)
            #rgn_cube.add_dim_coord(itime,0)
        
            rgn_cube.units = 'kg kg-1'
        
            print(rgn_cube)
            #print(rgn_cube.vdim)
        
            #Update buddy_features feature positions to correspond to buddy box space
            #rather than domain space or continuous/contiguous point space
            for buddy_looper in range(0,len(buddy_features)):
                buddy_features.vdim.values[buddy_looper] = buddy_features.vdim.values[buddy_looper] - bbox_zstart
                buddy_features.hdim_1.values[buddy_looper] = buddy_features.hdim_1.values[buddy_looper] - bbox_ystart
                buddy_features.hdim_2.values[buddy_looper] = buddy_features.hdim_2.values[buddy_looper] - bbox_xstart
            
            # Create cube of the same dimensions and coordinates as Buddy Box to store updated mask:        
            buddies_out=1*rgn_cube
            buddies_out.rename('buddies_mask')
            buddies_out.units=1

            #Create dask array from input data:
            #data=rgn_cube.core_data()
            buddy_data = buddy_rgn

            #All of the below is, I think, the same overarching segmentation procedure as in the original
            #segmentation approach until the line which states
            # "#transform seg_mask_4 data back to original mask after PBC first-pass ("segmentation_mask_3")"
            # It's just performed on the buddy box and its data rather than our full domain

            #Set level at which to create "Seed" for each feature in the case of 3D watershedding:
            # If none, use all levels (later reduced to the ones fulfilling the theshold conditions)
            if level==None:
                level=slice(None)

            # transform max_distance in metres to distance in pixels:
            if max_distance is not None:
                max_distance_pixel=np.ceil(max_distance/dxy)
                #note - this doesn't consider vertical distance in pixels

            # mask data outside region above/below threshold and invert data if tracking maxima:
            if target == 'maximum':
                unmasked_buddies=buddy_data>threshold
                buddy_segmentation=-1*buddy_data
            elif target == 'minimum':
                unmasked_buddies=buddy_data<threshold
                buddy_segmentation=buddy_data
            else:
                raise ValueError('unknown type of target')

            # set markers at the positions of the features:
            buddy_markers = np.zeros(unmasked_buddies.shape).astype(np.int32)
    
            if rgn_cube.ndim==2: #2D watershedding        
                for index, row in buddy_features.iterrows():
                    buddy_markers[int(row['hdim_1']), int(row['hdim_2'])]=row['feature']

            elif rgn_cube.ndim==3: #3D watershedding
        
        
                list_coord_names=[coord.name() for coord in rgn_cube.coords()]
                #determine vertical axis:
                print(list_coord_names)
                if vertical_coord=='auto':
                    list_vertical=['vdim','z','model_level_number','altitude','geopotential_height']
                    for coord_name in list_vertical:
                        if coord_name in list_coord_names:
                            vertical_axis=coord_name
                            print(vertical_axis)
                            break
                elif vertical_coord in list_coord_names:
                    vertical_axis=vertical_coord
                else:
                    raise ValueError('Please specify vertical coordinate')
                ndim_vertical=rgn_cube.coord_dims(vertical_axis)
        
                if len(ndim_vertical)>1:
                    raise ValueError('please specify 1 dimensional vertical coordinate')
                z_len = len(rgn_cube.coord('vdim').points)
                y_len = len(rgn_cube.coord('hdim_1').points)
                x_len = len(rgn_cube.coord('hdim_2').points)
        
        
                for index, row in buddy_features.iterrows():
                #creation of 5x5x5 point ranges for 3D marker seeding
                #and PBC flags for cross-boundary seeding - nixing this idea for now, but described in PBC Segmentation notes
            
                    # TODO: fix point ranges here. 
                    # TODO: why is this repeated?
                    if(int(row['vdim']) >=2 and int(row['vdim']) <= z_len-3):
                        z_list = np.arange(int(row['vdim']-2),int(row['vdim']+3))
                    elif(int(row['vdim']) < 2):
                        z_list = np.arange(0,5)
                    else:
                        z_list = np.arange(z_len-5,z_len)
                
                    if(int(row['hdim_1']) >=2 and int(row['hdim_1']) <= y_len-3):
                        y_list = np.arange(int(row['hdim_1']-2),int(row['hdim_1']+3))
                    elif(int(row['hdim_1']) < 2):
                        y_list = np.arange(0,5)
                        #PBC_y_chk = 1
                    else:
                        y_list = np.arange(y_len-5,y_len)
                        #PBC_y_chk = 1
                
                    if(int(row['hdim_2']) >=2 and int(row['hdim_2']) <= x_len-3):
                        x_list = np.arange(int(row['hdim_2']-2),int(row['hdim_2']+3))
                    elif(int(row['hdim_2']) < 2):
                        x_list = np.arange(0,5)
                        #PBC_x_chk = 1
                    else:
                        x_list = np.arange(x_len-5,x_len)
                        #PBC_x_chk = 1
                
                    #loop thru 5x5x5 z times y times x range
                    for k in range(0,5):
                        for j in range(0,5):
                            for i in range(0,5):
                        
                                if ndim_vertical[0]==0:
                                    buddy_markers[z_list[k],y_list[j],x_list[i]]=row['feature']
                                elif ndim_vertical[0]==1:
                                    buddy_markers[y_list[j],z_list[k],x_list[i]]=row['feature']
                                elif ndim_vertical[0]==2:
                                    buddy_markers[y_list[j],x_list[i],z_list[k]]=row['feature']
                                    
            else:
                raise ValueError('Segmentations routine only possible with 2 or 3 spatial dimensions')

            # set markers in cells not fulfilling threshold condition to zero:
            print(np.unique(buddy_markers))
            buddy_markers[~unmasked_buddies]=0
    
            marker_vals = np.unique(buddy_markers)
  
            # Turn into np arrays (not necessary for markers) as dask arrays don't yet seem to work for watershedding algorithm
            buddy_segmentation=np.array(buddy_segmentation)
            unmasked_buddies=np.array(unmasked_buddies)

            # perform segmentation:
            if method=='watershed':
                segmentation_mask_4 = watershed(np.array(buddy_segmentation),buddy_markers.astype(np.int32), mask=unmasked_buddies)
                
            else:                
                raise ValueError('unknown method, must be watershed')

            # remove everything from the individual masks that is more than max_distance_pixel away from the markers
            if max_distance is not None:
                D=distance_transform_edt((markers==0).astype(int))
                segmentation_mask_4[np.bitwise_and(segmentation_mask_4>0, D>max_distance_pixel)]=0

    
            #mask all segmentation_mask points below threshold as -1
            #to differentiate from those unmasked points NOT filled by watershedding
            print(np.unique(segmentation_mask_4))
            segmentation_mask_4[~unmasked_buddies] = -1
            
            
            #transform seg_mask_4 data back to original mask after PBC first-pass ("segmentation_mask_3")
            #print(np.unique(test_mask3.data))
        
            #loop through buddy box inds and analogous seg mask inds
            for z_val in range(bbox_zstart,bbox_zend):
                z_seg = z_val - bbox_zstart
                z_val_o = z_val
                for y_val in range(bbox_ystart,bbox_yend):
                    y_seg = y_val - bbox_ystart
                    #y_val_o = y_val
                    if y_val > hdim1_max:
                        y_val_o = y_val - (hdim1_max+1)
                    else:
                        y_val_o = y_val
                    for x_val in range(bbox_xstart,bbox_xend):
                        x_seg = x_val - bbox_xstart
                        #x_val_o = x_val
                        if x_val > hdim2_max:
                            x_val_o = x_val - (hdim2_max+1)
                        else:
                            x_val_o = x_val
                            #print(z_seg,y_seg,x_seg)
                            #print(z_val,y_val,x_val)
                    
                        #fix to
                        #overwrite IF:
                        #1) feature of interest
                        #2) changing to/from feature of interest or adjacent segmented feature
                    
                        #We don't want to overwrite other features that may be in the
                        #buddy box if not contacting the intersected seg field
                        
                        if (np.any(segmentation_mask_3[z_val_o,y_val_o,x_val_o]==buddies) and np.any(segmentation_mask_4.data[z_seg,y_seg,x_seg]==buddies)):
                            #only do updating procedure if old and new values both in buddy set
                            #and values are different
                            if(segmentation_mask_3[z_val_o,y_val_o,x_val_o] != segmentation_mask_4.data[z_seg,y_seg,x_seg]):
                                segmentation_mask_3[z_val_o,y_val_o,x_val_o] = segmentation_mask_4.data[z_seg,y_seg,x_seg]
                                #print("updated")
        if not is_3D_seg:
            segmentation_mask_3 = segmentation_mask_3[0]  

        segmentation_mask = segmentation_mask_3

    if transposed_data:
        segmentation_mask = np.transpose(segmentation_mask, axes = 
                                        [vertical_coord_axis, hdim_1_axis, hdim_2_axis])

    # Finished PBC checks and new PBC updated segmentation now in segmentation_mask. 
    #Write resulting mask into cube for output
    segmentation_out.data = segmentation_mask

    # count number of grid cells asoociated to each tracked cell and write that into DataFrame:
    values, count = np.unique(segmentation_mask, return_counts=True)
    counts=dict(zip(values, count))
    ncells=np.zeros(len(features_out))
    for i,(index,row) in enumerate(features_out.iterrows()):
        if row['feature'] in counts.keys():
            ncells=counts[row['feature']]
    features_out['ncells']=ncells

    return segmentation_out,features_out

def segmentation(features,field,dxy,threshold=3e-3,target='maximum',level=None,method='watershed',max_distance=None,vertical_coord='auto',PBC_flag='none',seed_3D_flag='column'):
    """
    Function using watershedding or random walker to determine cloud volumes associated with tracked updrafts
    
    Parameters:
    features:         pandas.DataFrame 
                   output from trackpy/maketrack
    field:      iris.cube.Cube 
                   containing the field to perform the watershedding on 
    threshold:  float 
                   threshold for the watershedding field to be used for the mask
                   
    target:        string
                   Switch to determine if algorithm looks strating from maxima or minima in input field (maximum: starting from maxima (default), minimum: starting from minima)

    level          slice
                   levels at which to seed the cells for the watershedding algorithm
    method:        str ('method')
                   flag determining the algorithm to use (currently watershedding implemented)
                   
    max_distance: float
                  Maximum distance from a marker allowed to be classified as belonging to that cell
                  
    PBC_flag:     string
                  string flag of 'none', 'hdim_1', 'hdim_2', or 'both' indicating which lateral boundaries are periodic
                  
    seed_3D_flag: string
                  string flag of 'column' (default) or 'box' which determines the method of seeding feature positions for 3D watershedding
    
    Output:
    segmentation_out: iris.cube.Cube
                   Cloud mask, 0 outside and integer numbers according to track inside the cloud
    """
    import pandas as pd
    from iris.cube import CubeList
    
    logging.info('Start watershedding 3D')

    # check input for right dimensions: 
    if not (field.ndim==3 or field.ndim==4):
        raise ValueError('input to segmentation step must be 3D or 4D including a time dimension')
    if 'time' not in [coord.name() for coord in field.coords()]:
        raise ValueError("input to segmentation step must include a dimension named 'time'")

    # CubeList and list to store individual segmentation masks and feature DataFrames with information about segmentation
    segmentation_out_list=CubeList()
    features_out_list=[]
                         
    #loop over individual input timesteps for segmentation:
    #OR do segmentation on single timestep
    #print(field)
    field_time=field.slices_over('time')
    #print(field_time)
    #print(enumerate(field_time))
    time_len = len(field.coord('time').points[:])
    print(time_len)
    
    for i,field_i in enumerate(field_time):
        time_i=field_i.coord('time').units.num2date(field_i.coord('time').points[0])
        features_i=features.loc[features['time']==time_i]
        #print(time_i)
        #print(field_i)
        #print(features_i)
        segmentation_out_i,features_out_i=segmentation_timestep(field_i,features_i,dxy,threshold=threshold,target=target,level=level,method=method,max_distance=max_distance,vertical_coord=vertical_coord,PBC_flag=PBC_flag,seed_3D_flag=seed_3D_flag)                 
        segmentation_out_list.append(segmentation_out_i)
        features_out_list.append(features_out_i)
        logging.debug('Finished segmentation for '+time_i.strftime('%Y-%m-%d_%H:%M:%S'))
    
    #if time_len > 1:
    
        
            
    #else:
    #    time_i=field.coord('time').units.num2date(field.coord('time').points[0])
    #    features_i=features.loc[features['time']==time_i]
    #    print(time_i)
    #    print(field)
    #    print(features_i)
    #    segmentation_out_i,features_out_i=segmentation_timestep(field,features_i,dxy,threshold=threshold,target=target,level=level,method=method,max_distance=max_distance,vertical_coord=vertical_coord,PBC_flag=PBC_flag)                 
    #    segmentation_out_list.append(segmentation_out_i)
    #    features_out_list.append(features_out_i)
    #    logging.debug('Finished segmentation for '+time_i.strftime('%Y-%m-%d_%H:%M:%S'))

    #Merge output from individual timesteps:
    segmentation_out=segmentation_out_list.merge_cube()
    features_out=pd.concat(features_out_list)
    
    logging.debug('Finished segmentation')
    return segmentation_out,features_out

def watershedding_3D(track,field_in,**kwargs):
    kwargs.pop('method',None)
    return segmentation_3D(track,field_in,method='watershed',**kwargs)

def watershedding_2D(track,field_in,**kwargs):
    kwargs.pop('method',None)
    return segmentation_2D(track,field_in,method='watershed',**kwargs)

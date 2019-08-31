def segmentation_3D(features,field,dxy,threshold=3e-3,target='maximum',level=None,method='watershed',max_distance=None):
    return segmentation(features,field,dxy,threshold=threshold,target=target,level=level,method=method,max_distance=max_distance)

def segmentation_2D(features,field,dxy,threshold=3e-3,target='maximum',level=None,method='watershed',max_distance=None):
    return segmentation(features,field,dxy,threshold=threshold,target=target,level=level,method=method,max_distance=max_distance)


def segmentation_timestep(field_i,features_i,dxy,threshold=3e-3,target='maximum',level=None,method='watershed',max_distance=None):    # Create cube of the same dimensions and coordinates as input data to store mask:        
    from skimage.morphology import watershed
    # from skimage.segmentation import random_walker
    from scipy.ndimage import distance_transform_edt

    
    features_out_i=features_i
    segmentation_out_i=1*field_i
    segmentation_out_i.rename('segmentation_mask')
    segmentation_out_i.units=1

    data_i=field_i.data

    # mask data outside region above/below threshold and invert data if tracking maxima:
    if target == 'maximum':
        unmasked=data_i>threshold
        data_i_segmentation=-1*data_i
    elif target == 'minimum':
        unmasked=data_i<threshold
        data_i_segmentation=data_i
    else:
        raise ValueError('unknown type of target')

    markers = np.zeros_like(unmasked).astype(np.int32)

    if field_i.ndim==2: #2D watershedding        

        for index, row in features_i.iterrows():
            markers[int(row['hdim_1']), int(row['hdim_2'])]=row['feature']

    elif field_i.ndim==3: #3D watershedding
        
        list_coord_names=[coord.name() for coord in field_i.coords()]
        #determin vertical axis:
        if vertical_coord=='auto':
            list_vertical=['z','model_level_number','altitude','geopotential_height']
            for coord_name in list_vertical:
                if coord_name in list_coord_names:
                    vertical_axis=coord_name
                    break
        elif vertical_coord in list_coord_names:
            vertical_axis=vertical_coord
        else:
            raise ValueError('Plese specify vertical coordinate')
            
                
        ndim_vertical=field_i.coord_dims(vertical_axis)
        if len(ndim_vertical)>1:
            raise ValueError('please specify 1 dimensional vertical coordinate')
            
        for index, row in features_i.iterrows():
            if ndim_vertical==0:
                markers[:,int(row['hdim_1']), int(row['hdim_2'])]=row['feature']
            elif ndim_vertical==1:
                markers[int(row['hdim_1']),:, int(row['hdim_2'])]=row['feature']
            elif ndim_vertical==2:
                markers[int(row['hdim_1']), int(row['hdim_2']),:]=row['feature']
    else:
        raise ValueError('Segmentations routine only possible with 2 or 3 spatial dimensions')
    
    markers[~unmasked]=0


    if method=='watershed':
        segmentation_mask_i = watershed(data_i_segmentation,markers.astype(np.int32), mask=unmasked)
#        elif method=='random_walker':
#           segmentation_mask_i=random_walker(data_i_segmentation, markers.astype(np.int32),
#                                beta=130, mode='bf', tol=0.001, copy=True, multichannel=False, return_full_prob=False, spacing=None)
    else:                
        raise ValueError('unknown method, must be watershed')

    # remove everything from the individual masks that is more than max_distance_pixel away from the markers
    if max_distance is not None:
        for feature in features_i['feature']:
            D=distance_transform_edt((markers!=feature).astype(int))
            segmentation_mask_i[np.bitwise_and(segmentation_mask_i==feature, D>max_distance_pixel)]=0

    #Write resulting mass into Cube and append to CubeList collecting masks for individual timesteps
    segmentation_out_i.data=segmentation_mask_i


    # count number of grid cells asoociated to each tracked cell and write that into DataFrame:
    values, count = np.unique(segmentation_mask_i, return_counts=True)
    counts=dict(zip(values, count))
    for index, row in features_out_i.iterrows():
        if row['feature'] in counts.keys():
            features_out_i.loc[index,'ncells']=counts[row['feature']]

    return segmentation_out_i,features_out_i

def segmentation(features,field,dxy,threshold=3e-3,target='maximum',level=None,method='watershed',max_distance=None):
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
    
    Output:
    segmentation_out: iris.cube.Cube
                   Cloud mask, 0 outside and integer numbers according to track inside the clouds
    
    """
    import numpy as np
    import logging
    from iris.cube import CubeList
    
    logging.info('Start watershedding 3D')
    
    if not field.ndim==3 or field.ndim==4:
        raise ValueError('input to segmentation step must be 3D or 4D including a time dimension')
    if 'time' not in [coord.name() for coord in field.coords()]:
        raise ValueError("input to segmentation step must include a dimension named 'time'")

    
    #Set level at which to create "Seed" for each cloud and threshold in total water content:
    # If none, use all levels (later reduced to the ones fulfilling the theshold conditions)
    if level==None:
        level=slice(None)
        
    if max_distance is not None:
        max_distance_pixel=np.ceil(max_distance/dxy)
    
    # CubeList to store individual segmentation masks
    segmentation_out_list=CubeList()
    features_out_list=[]
                         
    features['ncells']=0
    field_time=field.slices_over('time')
    for i,field_i in enumerate(field_time):        
        time_i=field_i.coord('time').units.num2date(field_i.coord('time').points[0])
        features_i=features[features['time']==time_i]
        segmentation_out_i,features_out_i=segmentation_timestep(field_i,features_i,dxy,threshold=threshold,target=target,level=level,method=method,max_distance=max_distance)                 
        segmentation_out_list.append(segmentation_out_i)           
        features_out_list.append(features_out_i)           

        
        logging.debug('Finished segmentation for '+time_i.strftime('%Y-%m-%d_%H:%M:%S'))
                         
    segmentation_out=segmentation_out_list.merge_cube()
    features_out=pd.concat(features_out_list)
    

    logging.debug('Finished segmentation')
    return segmentation_out,features_out
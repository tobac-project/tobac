import logging

def segmentation_3D(features,field,dxy,threshold=3e-3,target='maximum',level=None,method='watershed',max_distance=None):
    return segmentation(features,field,dxy,threshold=threshold,target=target,level=level,method=method,max_distance=max_distance)

def segmentation_2D(features,field,dxy,threshold=3e-3,target='maximum',level=None,method='watershed',max_distance=None):
    return segmentation(features,field,dxy,threshold=threshold,target=target,level=level,method=method,max_distance=max_distance)


def segmentation_timestep(field_in,features_in,dxy,threshold=3e-3,target='maximum',level=None,method='watershed',max_distance=None,vertical_coord='auto'):    
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
    
    Returns
    -------
    iris.cube.Cube
                      cloud mask, 0 outside and integer numbers according to track inside the clouds
    pandas.DataFrame
                  feature dataframe including the number of cells (2D or 3D) in the segmented area/volume of the feature at the timestep
    """
    from skimage.morphology import watershed
    # from skimage.segmentation import random_walker
    from scipy.ndimage import distance_transform_edt
    from copy import deepcopy
    import numpy as np

    # copy feature dataframe for output 
    features_out=deepcopy(features_in)
    # Create cube of the same dimensions and coordinates as input data to store mask:        
    segmentation_out=1*field_in
    segmentation_out.rename('segmentation_mask')
    segmentation_out.units=1

    #Create dask array from input data:
    data=field_in.core_data()
    
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
    if field_in.ndim==2: #2D watershedding        
        for index, row in features_in.iterrows():
            markers[int(row['hdim_1']), int(row['hdim_2'])]=row['feature']

    elif field_in.ndim==3: #3D watershedding
        list_coord_names=[coord.name() for coord in field_in.coords()]
        #determine vertical axis:
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
        ndim_vertical=field_in.coord_dims(vertical_axis)
        if len(ndim_vertical)>1:
            raise ValueError('please specify 1 dimensional vertical coordinate')
        for index, row in features_in.iterrows():
            if ndim_vertical[0]==0:
                markers[:,int(row['hdim_1']), int(row['hdim_2'])]=row['feature']
            elif ndim_vertical[0]==1:
                markers[int(row['hdim_1']),:, int(row['hdim_2'])]=row['feature']
            elif ndim_vertical[0]==2:
                markers[int(row['hdim_1']), int(row['hdim_2']),:]=row['feature']
    else:
        raise ValueError('Segmentations routine only possible with 2 or 3 spatial dimensions')

    # set markers in cells not fulfilling threshold condition to zero:
    markers[~unmasked]=0
  
    # Turn into np arrays (not necessary for markers) as dask arrays don't yet seem to work for watershedding algorithm
    data_segmentation=np.array(data_segmentation)
    unmasked=np.array(unmasked)

    # perform segmentation:
    if method=='watershed':
        segmentation_mask = watershed(np.array(data_segmentation),markers.astype(np.int32), mask=unmasked)
#    elif method=='random_walker':
#        segmentation_mask=random_walker(data_segmentation, markers.astype(np.int32),
#                                          beta=130, mode='bf', tol=0.001, copy=True, multichannel=False, return_full_prob=False, spacing=None)
    else:                
        raise ValueError('unknown method, must be watershed')

    # remove everything from the individual masks that is more than max_distance_pixel away from the markers
    if max_distance is not None:
        D=distance_transform_edt((markers==0).astype(int))
        segmentation_mask[np.bitwise_and(segmentation_mask>0, D>max_distance_pixel)]=0

    #Write resulting mask into cube for output
    segmentation_out.data=segmentation_mask

    # count number of grid cells asoociated to each tracked cell and write that into DataFrame:
    values, count = np.unique(segmentation_mask, return_counts=True)
    counts=dict(zip(values, count))
    ncells=np.zeros(len(features_out))
    for i,(index,row) in enumerate(features_out.iterrows()):
        if row['feature'] in counts.keys():
            ncells=counts[row['feature']]
    features_out['ncells']=ncells

    return segmentation_out,features_out

def segmentation(features,field,dxy,threshold=3e-3,target='maximum',level=None,method='watershed',max_distance=None,vertical_coord='auto'):
    """Function using watershedding or random walker to determine cloud volumes associated with tracked updrafts
    
    Parameters
    ----------
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
    
    Returns
    -------
    iris.cube.Cube
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
    field_time=field.slices_over('time')
    for i,field_i in enumerate(field_time):
        time_i=field_i.coord('time').units.num2date(field_i.coord('time').points[0])
        features_i=features.loc[features['time']==time_i]
        segmentation_out_i,features_out_i=segmentation_timestep(field_i,features_i,dxy,threshold=threshold,target=target,level=level,method=method,max_distance=max_distance,vertical_coord=vertical_coord)                 
        segmentation_out_list.append(segmentation_out_i)
        features_out_list.append(features_out_i)
        logging.debug('Finished segmentation for '+time_i.strftime('%Y-%m-%d_%H:%M:%S'))

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

def segmentation_3D(track,field,threshold=3e-3,target='maximum',level=None,method='watershed'):
    """
    Function using watershedding or random walker to determine cloud volumes associated with tracked updrafts
    
    Parameters:
    track:         pandas.DataFrame 
                   output from trackpy/maketrack
    field_in:      iris.cube.Cube 
                   containing the field to perform the watershedding on 
    threshold:  float 
                   threshold for the watershedding field to be used for the mask
                   
    target:        string
                   Switch to determine if algorithm looks strating from maxima or minima in input field (maximum: starting from maxima (default), minimum: starting from minima)

    level          slice
                   levels at which to seed the cells for the watershedding algorithm
    method:        str ('method')
                   flag determining the algorithm to use (currently watershedding implemented)

    Output:
    segmentation_out: iris.cube.Cube
                   Cloud mask, 0 outside and integer numbers according to track inside the clouds
    
    """
    
    import numpy as np
    from skimage.morphology import watershed
#    from skimage.segmentation import random_walker
    import logging
    from iris.cube import CubeList
    from iris.util import new_axis
    logging.info('Start watershedding 3D')

    #Set level at which to create "Seed" for each cloud and threshold in total water content:
    # If none, use all levels (later reduced to the ones fulfilling the theshold conditions)
    if level==None:
        level=slice(None)
    
    # CubeList to store individual segmentation masks
    segmentation_out_list=CubeList()
    
    track['ncells']=0
    field_time=field.slices_over('time')
    for i,field_i in enumerate(field_time):

        # Create cube of the same dimensions and coordinates as input data to store mask:        
        segmentation_out_i=1*field_i
        segmentation_out_i.rename('segmentation_mask')
        segmentation_out_i.units=1

#        data_i=field_i.core_data()
        data_i=field_i.data

        time_i=field_i.coord('time').units.num2date(field_i.coord('time').points[0])
        tracks_i=track[track['time']==time_i]
        
        # mask data outside region above/below threshold and invert data if tracking maxima:
        if target == 'maximum':
            unmasked=data_i>threshold
            data_i_segmentation=-1*data_i
        elif target == 'minimum':
            unmasked=data_i<threshold
            data_i_segmentation=data_i
        else:
            raise ValueError('unknown type of target')

            raise ValueError('unknown type of target')
        markers = np.zeros_like(unmasked).astype(np.int32)
        for index, row in tracks_i.iterrows():
             markers[:,int(row['hdim_1']), int(row['hdim_2'])]=row.cell
        markers[~unmasked]=0
        
        if method=='watershed':
            res1 = watershed(data_i_segmentation,markers.astype(np.int32), mask=unmasked)
#        elif method=='random_walker':
#             res1=random_walker(data_i_segmentation, markers.astype(np.int32),
#                                beta=130, mode='bf', tol=0.001, copy=True, multichannel=False, return_full_prob=False, spacing=None)
        else:                
            raise ValueError('unknown method, must be watershed')
            
        #Write resulting mass into Cube and append to CubeList collecting masks for individual timesteps
        
        segmentation_out_i.data=res1
        
        # using merge throws error, so cubes with time promoted to DimCoord and using concatenate:
#        segmentation_out_list.append(segmentation_out_i)
        segmentation_out_i_temp=new_axis(segmentation_out_i, scalar_coord='time')
        segmentation_out_list.append(segmentation_out_i_temp)

        # count number of grid cells asoociated to each tracked cell and write that into DataFrame:
        values, count = np.unique(res1, return_counts=True)
        counts=dict(zip(values, count))
        for index, row in tracks_i.iterrows():
            if row['cell'] in counts.keys():
                track.loc[index,'ncells']=counts[row['cell']]
        
        logging.debug('Finished segmentation 3D for '+time_i.strftime('%Y-%m-%d_%H:%M:%S'))
    #merge individual masks in CubeList into one Cube:    
    # using merge throws error, so cubes with time promoted to DimCoord and using concatenate:
#    segmentation_out=segmentation_out_list.merge_cube()
    segmentation_out=segmentation_out_list.concatenate_cube()

    logging.debug('Finished segmentation 3D')
    return segmentation_out,track
            
def segmentation_2D(track,field,threshold=0,target='maximum',method='watershed'):
    """
    Function using watershedding or random walker to determine cloud volumes associated with tracked updrafts
    Parameters:
    track:         pandas.DataFrame 
                   output from trackpy/maketrack
    field_in:      iris.cube.Cube
                   containing the 3D (time,x,y) field to perform the watershedding on 
    threshold:     float 
                   threshold for the watershedding field to be used for the mask
    target:        string
                   Switch to determine if algorithm looks strating from maxima or minima in input field (maximum: starting from maxima (default), minimum: starting from minima)
    method:        str ('method')
                   flag determining the algorithm to use (currently watershedding implemented)
    
    Output:
    segmentation_out: iris.cube.Cube
                   Cloud mask, 0 outside and integer numbers according to track inside the clouds
    
    """  
    import numpy as np
    from skimage.morphology import watershed
#    from skimage.segmentation import random_walker
    import logging
    from iris.cube import CubeList
    from iris.util import new_axis

    logging.info('Start wateshedding 2D')

    # CubeList to store individual segmentation masks
    segmentation_out_list=CubeList()

    track['ncells']=0

    field_time=field.slices_over('time')
    for i,field_i in enumerate(field_time):
        
        # Create cube of the same dimensions and coordinates as input data to store mask:        
        segmentation_out_i=1*field_i
        segmentation_out_i.rename('segmentation_mask')
        segmentation_out_i.units=1
        
        data_i=field_i.core_data()
        time_i=field_i.coord('time').units.num2date(field_i.coord('time').points[0])
        tracks_i=track[track['time']==time_i]
        
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
        for index, row in tracks_i.iterrows():
            markers[int(row['hdim_1']), int(row['hdim_2'])]=row.cell
        markers[~unmasked]=0

        if method=='watershed':
            res1 = watershed(data_i_segmentation,markers.astype(np.int32), mask=unmasked)
#        elif method=='random_walker':
#            #res1 = random_walker(Mask, markers,mode='cg')
#             res1=random_walker(data_i_segmentation, markers.astype(np.int32),
#                                beta=130, mode='bf', tol=0.001, copy=True, multichannel=False, return_full_prob=False, spacing=None)
        else:
            raise ValueError('unknown method, must be watershed')
            
        segmentation_out_i.data=res1
        # using merge throws error, so cubes with time promoted to DimCoord and using concatenate:
#        segmentation_out_list.append(segmentation_out_i)
        segmentation_out_i_temp=new_axis(segmentation_out_i, scalar_coord='time')
        segmentation_out_list.append(segmentation_out_i_temp)
        
        # count number of grid cells asoociated to each tracked cell and write that into DataFrame:
        values, count = np.unique(res1, return_counts=True)
        counts=dict(zip(values, count))
        for index, row in tracks_i.iterrows():
            if row['cell'] in counts.keys():
                track.loc[index,'ncells']=counts[row['cell']]
        logging.debug('Finished segmentation 2D for '+time_i.strftime('%Y-%m-%d_%H:%M:%S'))
    
    #merge individual masks in CubeList into one Cube:    
    # using merge throws error, so cubes with time promoted to DimCoord and using concatenate:
#    segmentation_out=segmentation_out_list.merge_cube()
    segmentation_out=segmentation_out_list.concatenate_cube()

    logging.debug('Finished segmentation 2D')

    return segmentation_out,track

#functions for backwards compatibility

def watershedding_3D(track,field_in,**kwargs):
    kwargs.pop('method',None)
    return segmentation_3D(track,field_in,method='watershed',**kwargs)

def watershedding_2D(track,field_in,**kwargs):
    kwargs.pop('method',None)
    return segmentation_2D(track,field_in,method='watershed',**kwargs)
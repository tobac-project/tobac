def segmentation_3D(track,field,threshold=3e-3,target='maximum',level=None,compactness=0,method='watershed'):
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
                   levels at which to seed the particles for the watershedding algorithm
    compactness    float
                   parameter describing the compactness of the resulting volume
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
        segmentation_out_i.rename('watershedding_output_mask')
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
             markers[:,int(row['hdim_1']), int(row['hdim_2'])]=row.particle
        markers[~unmasked]=0
        
        if method=='watershed':
            res1 = watershed(data_i_segmentation,markers.astype(np.int32), mask=unmasked,compactness=compactness)
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
            if row['particle'] in counts.keys():
                track.loc[index,'ncells']=counts[row['particle']]
        
        logging.debug('Finished segmentation 3D for '+time_i.strftime('%Y-%m-%d_%H:%M:%S'))
    #merge individual masks in CubeList into one Cube:    
    # using merge throws error, so cubes with time promoted to DimCoord and using concatenate:
#    segmentation_out=segmentation_out_list.merge_cube()
    segmentation_out=segmentation_out_list.concatenate_cube()

    logging.debug('Finished segmentation 3D')
    return segmentation_out,track
            
def segmentation_2D(track,field,threshold=0,target='maximum',compactness=0,method='watershed'):
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
    compactness    float
                   parameter describing the compactness of the resulting volume
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
        segmentation_out_i.rename('watershedding_output_mask')
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
            markers[int(row['hdim_1']), int(row['hdim_2'])]=row.particle
        markers[~unmasked]=0

        if method=='watershed':
            res1 = watershed(data_i_segmentation,markers.astype(np.int32), mask=unmasked,compactness=compactness)
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
            if row['particle'] in counts.keys():
                track.loc[index,'ncells']=counts[row['particle']]
        logging.debug('Finished wateshedding 2D for '+time_i.strftime('%Y-%m-%d_%H:%M:%S'))
    
    #merge individual masks in CubeList into one Cube:    
    # using merge throws error, so cubes with time promoted to DimCoord and using concatenate:
#    segmentation_out=segmentation_out_list.merge_cube()
    segmentation_out=segmentation_out_list.concatenate_cube()

    logging.debug('Finished segmentation 3D')

    return segmentation_out,track

#functions for backwards compatibility

def watershedding_3D(track,field_in,**kwargs):
    kwargs.pop('method',None)
    return segmentation_3D(track,field_in,method='watershed',**kwargs)

def watershedding_2D(track,field_in,**kwargs):
    kwargs.pop('method',None)
    return segmentation_2D(track,field_in,method='watershed',**kwargs)

def column_mask_from2D(mask_2D,cube,z_coord='model_level_number'):
    '''     function to turn 2D watershedding mask into a 3D mask of selected columns
    Input:
    cube:              iris.cube.Cube 
                       data cube
    mask_2D:           iris.cube.Cube 
                       2D cube containing mask (int id for tacked volumes 0 everywhere else)
    z_coord:           str
                       name of the vertical coordinate in the cube
    Output:
    mask_2D:           iris.cube.Cube 
                       3D cube containing columns of 2D mask (int id for tacked volumes 0 everywhere else)
    '''
    from copy import deepcopy
    mask_3D=deepcopy(cube)
    mask_3D.rename('watershedding_output_mask')
    dim=mask_3D.coord_dims(z_coord)[0]
    for i in range(len(mask_3D.coord(z_coord).points)):
        slc = [slice(None)] * len(mask_3D.shape)
        print(i)
        print(dim)
        slc[dim] = slice(i,i+1)    
        print(slc)
        print(mask_2D.shape)
        mask_out=mask_3D[slc]
        print(mask_out.shape)
        mask_3D.data[slc]=mask_2D.core_data()
    return mask_3D


def mask_cube_particle(variable_cube,mask,particle):
    ''' Mask cube for tracked volume of an individual cell   
    Input:
    variable_cube:     iris.cube.Cube 
                       unmasked data cube
    mask:              iris.cube.Cube 
                       cube containing mask (int id for tacked volumes 0 everywhere else)
    particle:          int
                       interger id of cell to create masked cube for
    Output:
    variable_cube_out: iris.cube.Cube 
                       Masked cube with data for respective cell
    '''
    import numpy as np 
    from copy import deepcopy
    variable_cube_out=deepcopy(variable_cube)
    mask_i=mask.data!=particle
    variable_cube_out.data=np.ma.array(variable_cube_out.data,mask=mask_i)    
    return variable_cube_out

def mask_cube_untracked(variable_cube,mask):
    ''' Mask cube for untracked volume 
    Input:
    variable_cube:     iris.cube.Cube 
                       unmasked data cube
    mask:              iris.cube.Cube 
                       cube containing mask (int id for tacked volumes 0 everywhere else)
    Output:
    variable_cube_out: iris.cube.Cube 
                       Masked cube for untracked volume
    '''
    import numpy as np 
    from copy import deepcopy
    variable_cube_out=deepcopy(variable_cube)
    mask_i=mask.data!=0
    variable_cube_out.data=np.ma.array(variable_cube_out.data,mask=mask_i)    
    return variable_cube_out

def mask_cube(cube_in,mask):
    ''' Mask cube where mask is larger than zero
    Input:
    cube_in:     iris.cube.Cube 
                       unmasked data cube
    mask:              numpy.ndarray or dask.array 
                       mask to use for masking, >0 where cube is supposed to be masked
    Output:
    cube_out:          iris.cube.Cube 
                       Masked cube
    '''
    from numpy import ones_like,ma
    from copy import deepcopy
    mask_array=ones_like(cube_in.data,dtype=bool)
    mask_array[mask>0]=False
    mask_array[mask==0]=True
    cube_out=deepcopy(cube_in)
    cube_out.data=ma.array(cube_in.data,mask=mask_array)
    return cube_out

def mask_particle(Mask,particle,masked=False):
    ''' create mask for specific particle
    Input:
    variable_cube:     iris.cube.Cube 
                       unmasked data cube
    mask:              iris.cube.Cube 
                       cube containing mask (int id for tacked volumes 0 everywhere else)
    Output:
    variable_cube_out: numpy.ndarray 
                       Masked cube for untracked volume
    '''
    import numpy as np 
    from copy import deepcopy
    Mask_i=deepcopy(Mask)
    Mask_i.data[Mask_i.data!=particle]=0
    if masked:
        Mask_i.data=np.ma.array(Mask_i.data,mask=Mask_i.data)
    return Mask_i   

def mask_particle_surface(Mask,particle,masked=False,z_coord='model_level_number'):
    ''' Mask cube for untracked volume 
    Input:
    variable_cube:     iris.cube.Cube 
                       unmasked data cube
    mask:              iris.cube.Cube 
                       cube containing mask (int id for tacked volumes 0 everywhere else)
    Output:
    variable_cube_out: iris.cube.Cube 
                       Masked cube for untracked volume
    '''
    from iris.analysis import MAX
    import numpy as np 
    from copy import deepcopy
    Mask_i=deepcopy(Mask)
    Mask_i.data[Mask_i.data!=particle]=0
    for coord in  Mask_i.coords():
        if coord.ndim>1 and Mask_i.coord_dims(z_coord)[0] in Mask_i.coord_dims(coord):
            Mask_i.remove_coord(coord.name())
    Mask_i_surface=Mask_i.collapsed(z_coord,MAX)
    if masked:
        Mask_i_surface.data=np.ma.array(Mask_i_surface.data,mask=Mask_i_surface.data)
    return Mask_i_surface    

def mask_particle_columns(Mask,particle,masked=False,z_coord='model_level_number'):
    ''' Mask cube for untracked volume 
    Input:
    variable_cube:     iris.cube.Cube 
                       unmasked data cube
    mask:              iris.cube.Cube 
                       cube containing mask (int id for tacked volumes 0 everywhere else)
    Output:
    variable_cube_out: iris.cube.Cube 
                       Masked cube for untracked volume
    '''
    from iris.analysis import MAX
    import numpy as np 
    from copy import deepcopy
    Mask_i=deepcopy(Mask)
    Mask_i.data[Mask_i.data!=particle]=0
    for coord in  Mask_i.coords():
        if coord.ndim>1 and Mask_i.coord_dims(z_coord)[0] in Mask_i.coord_dims(coord):
            Mask_i.remove_coord(coord.name())
    Mask_i_surface=Mask_i.collapsed(z_coord,MAX)
    for cube_slice in Mask_i.slices(['time','x','y']):
            cube_slice.data=Mask_i_surface.core_data()
    if masked:
        Mask_i.data=np.ma.array(Mask_i.data,mask=Mask_i.data)
    return Mask_i


#def constraint_cell(track,mask_particle,width=None,x=None,):
#     from iris import Constraint
#     import numpy as np
#    
#     time_coord=mask_particle.coord('time')
#     time_units=time_coord.units
#    
#     def time_condition(cell):
#         return time_units.num2date(track.head(n=1)['time']) <= cell <= time_units.num2date(track.tail(n=1)['time'])
#
#     constraint_time=Constraint(time=time_condition)
##     mask_particle_i=mask_particle.extract(constraint_time)
#     mask_particle_surface_i=mask_particle_surface.extract(constraint_time)
#    
#     x_dim=mask_particle_surface_i.coord_dims('projection_x_coordinate')[0]
#     y_dim=mask_particle_surface_i.coord_dims('projection_y_coordinate')[0]
#     x_coord=mask_particle_surface_i.coord('projection_x_coordinate')
#     y_coord=mask_particle_surface_i.coord('projection_y_coordinate')
#    
#     if (mask_particle_surface_i.core_data()>0).any():
#         box_mask_i=get_bounding_box(mask_particle_surface_i.core_data(),buffer=1)
#
#         box_mask=[[x_coord.points[box_mask_i[x_dim][0]],x_coord.points[box_mask_i[x_dim][1]]],
#                  [y_coord.points[box_mask_i[y_dim][0]],y_coord.points[box_mask_i[y_dim][1]]]]
#     else:
#         box_mask=[[np.nan,np.nan],[np.nan,np.nan]]
#
#         x_min=box_mask[0][0]
#         x_max=box_mask[0][1]
#         y_min=box_mask[1][0]
#         y_max=box_mask[1][1]
#     constraint_x=Constraint(projection_x_coordinate=lambda cell: int(x_min) < cell < int(x_max))
#     constraint_y=Constraint(projection_y_coordinate=lambda cell: int(y_min) < cell < int(y_max))
#
#     constraint=constraint_time & constraint_x & constraint_y
#     return constraint


def get_bounding_box(x,buffer=1):
    from numpy import delete,arange,diff,nonzero,array
    """ Calculates the bounding box of a ndarray
    https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    """
    mask = x == 0

    bbox = []
    all_axis = arange(x.ndim)
    #loop over dimensions
    for kdim in all_axis:
        nk_dim = delete(all_axis, kdim)
        mask_i = mask.all(axis=tuple(nk_dim))
        dmask_i = diff(mask_i)
        idx_i = nonzero(dmask_i)[0]
        # for case where there is no value in idx_i
        if len(idx_i) == 0:
            idx_i=array([0,x.shape[kdim]-1])
        # for case where there is only one value in idx_i
        if len(idx_i) == 1:
            idx_i=array([idx_i,idx_i])
        # make sure there is two values in idx_i
        if len(idx_i) != 2:
            raise ValueError('Algorithm failed, {} does not have 2 elements!'.format(idx_i))
        # caluclate min and max values for idx_i and append them to list
        idx_min=max(0,idx_i[0]+1-buffer)
        idx_max=min(x.shape[kdim]-1,idx_i[1]+1+buffer)
        bbox.append([idx_min, idx_max])
    return bbox

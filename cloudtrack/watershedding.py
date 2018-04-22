def watershedding_3D(Track,Field_in,threshold=3e-3,target='maximum',level=None,compactness=0,method='watershed'):
    """
    Function using watershedding to determine cloud volumes associated with tracked updrafts
    
    Parameters:
    Track:         pandas.DataFrame 
                   output from trackpy/maketrack
    Field_in:      iris.cube.Cube 
                   containing the field to perform the watershedding on 
    threshold:  float 
                   threshold for the watershedding field to be used for the mask
                   
    target:        string
                   Switch to determine if algorithm looks strating from maxima or minima in input field (maximum: starting from maxima (default), minimum: starting from minima)

    level          slice
                   levels at which to seed the particles for the watershedding algorithm
    compactness    float
                   parameter describing the compactness of the resulting volume
    
    Output:
    Watershed_out: iris.cube.Cube
                   Cloud mask, 0 outside and integer numbers according to track inside the clouds
    
    """
    
    import numpy as np
    import copy
    from skimage.morphology import watershed
    from skimage.segmentation import random_walker
    from iris.analysis import MIN,MAX
    import logging
#    from scipy.ndimage.measurements import watershed_ift

    #Set level at which to create "Seed" for each cloud and threshold in total water content:
    # If none, use all levels (later reduced to the ones fulfilling the theshold conditions)
    if level==None:
        level=slice(None)
         
    Watershed_out=copy.deepcopy(Field_in)
    Watershed_out.rename('watershedding_output_mask')
    Watershed_out.data[:]=0
    Watershed_out.units=1
    cooridinates=Field_in.coords(dim_coords=True)
    maximum_value=Field_in.collapsed(cooridinates,MAX).data
    minimum_value=Field_in.collapsed(cooridinates,MIN).data
    range_value=maximum_value-minimum_value
    Track['ncells']=0
    for i, time in enumerate(Field_in.coord('time').points):        
#        print('doing watershedding for',WC.coord('time').units.num2date(time).strftime('%Y-%m-%d %H:%M:%S'))
        Tracks_i=Track[Track['frame']==i]
        data_i=Field_in[i,:].data
 
        if target == 'maximum':
            unmasked=data_i>threshold
        elif target == 'minimum':
            unmasked=data_i<threshold
        else:
            raise ValueError('unknown type of target')
        markers = np.zeros_like(unmasked).astype(np.int32)
        for index, row in Tracks_i.iterrows():
            markers[:,int(row['hdim_1']), int(row['hdim_2'])]=row.particle
        markers[~unmasked]=0
        maximum_value=np.amax(data_i)
        minimum_value=np.amin(data_i)
        range_value=maximum_value-minimum_value
        if target == 'maximum':
            data_i_watershed=1500-(data_i-minimum_value)*1000/range_value
        elif target == 'minimum':
            data_i_watershed=1500-(maximum_value-data_i)*1000/range_value
        else:
            raise ValueError('unknown type of target')

        data_i_watershed[~unmasked]=2000
        data_i_watershed=data_i_watershed.astype(np.uint32)
        #res1 = watershed_ift(data_i_watershed, markers)
        
        if method=='watershed':
            res1 = watershed(data_i_watershed,markers.astype(np.int32), mask=unmasked,compactness=compactness)
        elif method=='random_walker':
            #res1 = random_walker(Mask, markers,mode='cg')
             res1=random_walker(data_i_watershed, markers.astype(np.int32), beta=130, mode='bf', tol=0.001, copy=True, multichannel=False, return_full_prob=False, spacing=None)
        else:
            print('unknown method')
        Watershed_out.data[i,:]=res1
        values, count = np.unique(res1, return_counts=True)
        counts=dict(zip(values, count))

        for index, row in Tracks_i.iterrows():
            if row['particle'] in counts.keys():
                Track.loc[index,'ncells']=counts[row['particle']]
                
    return Watershed_out,Track
            
def watershedding_2D(Track,Field_in,threshold=0,target='maximum',compactness=0,method='watershed'):
    """
    Function using watershedding to determine cloud volumes associated with tracked updrafts
    
    Parameters:
    :param CommonData or CommonDataList data: Data to collocate
    
    :param pandas.DataFrame Track: output from trackpy/maketrack
    :param iris.cube.Cube Field_in: containing the field to perform the watershedding on 
    :param float threshold: threshold for the watershedding field to be used for the mask
    :param string target:Switch to determine if algorithm looks strating from maxima or minima in input field ('maximum': starting from maxima (default), 'minimum': starting from minima)
    :param slice level: levels at which to seed the particles for the watershedding algorithm
    :param float compactness: parameter describing the compactness of the resulting volume
    
    Output:
        
    :return iris.cube.Cube Watershed_out: Cloud mask, 0 outside and integer numbers according to track inside the clouds
    """
    
    import numpy as np
    import copy
    from skimage.morphology import watershed
    from skimage.segmentation import random_walker
    from iris.analysis import MIN,MAX

#    from scipy.ndimage.measurements import watershed_ift

    Watershed_out=copy.deepcopy(Field_in)
    Watershed_out.rename('watershedding_output_mask')
    Watershed_out.data[:]=0
    Watershed_out.units=1
    cooridinates=Field_in.coords(dim_coords=True)
    maximum_value=Field_in.collapsed(cooridinates,MAX).data
    minimum_value=Field_in.collapsed(cooridinates,MIN).data
    range_value=maximum_value-minimum_value
    
    Track['ncells']=0

    for i, time in enumerate(Field_in.coord('time').points):        
#        print('doing watershedding for',WC.coord('time').units.num2date(time).strftime('%Y-%m-%d %H:%M:%S'))
        Tracks_i=Track[Track['frame']==i]
        data_i=Field_in[i,:].data        
        
        if target == 'maximum':
            unmasked=data_i>threshold
        elif target == 'minimum':
            unmasked=data_i<threshold
        else:
            raise ValueError('unknown type of target')
        markers = np.zeros_like(unmasked).astype(np.int16)
        for index, row in Tracks_i.iterrows():
            markers[int(row['hdim_2']), int(row['hdim_1'])]=row.particle
        markers[~unmasked]=0
        if target == 'maximum':
            data_i_watershed=1000-(data_i-minimum_value)*1000/range_value
        elif target == 'minimum':
            data_i_watershed=1000-(maximum_value-data_i)*1000/range_value
        else:
            raise ValueError('unknown type of target')

        data_i_watershed[~unmasked]=2000
        data_i_watershed=data_i_watershed.astype(np.uint16)

        data_i_watershed=data_i_watershed.astype(np.uint16)
        #res1 = watershed_ift(data_i_watershed, markers)
        
        if method=='watershed':
            res1 = watershed(data_i_watershed,markers.astype(np.int8), mask=unmasked,compactness=compactness)
        elif method=='random_walker':
            #res1 = random_walker(Mask, markers,mode='cg')
              res1=random_walker(data_i_watershed, markers.astype(np.int8), beta=130, mode='bf', tol=0.001, copy=True, multichannel=False, return_full_prob=False, spacing=None)
        else:
            print('unknown method')
        Watershed_out.data[i,:]=res1
        
        values, count = np.unique(res1, return_counts=True)
        counts=dict(zip(values, count))

        for index, row in Tracks_i.iterrows():
            if row['particle'] in counts.keys():
                Track.loc[index,'ncells']=counts[row['particle']]

    return Watershed_out,Track

def mask_cube_particle(variable_cube,Mask,particle):
    import numpy as np 
    from copy import deepcopy
    variable_cube_out=deepcopy(variable_cube)
    mask=Mask.data!=particle
    variable_cube_out.data=np.ma.array(variable_cube_out.data,mask=mask)    
    return variable_cube_out

def mask_cube_untracked(variable_cube,Mask):
    import numpy as np 
    from copy import deepcopy
    variable_cube_out=deepcopy(variable_cube)
    mask=Mask.data!=0
    variable_cube_out.data=np.ma.array(variable_cube_out.data,mask=mask)    
    return variable_cube_out

def mask_cube(cube_in,mask):
    from numpy import ones_like,ma
    from copy import deepcopy
    mask_array=ones_like(cube_in.data,dtype=bool)
    mask_array[mask>0]=False
    mask_array[mask==0]=True
    cube_out=deepcopy(cube_in)
    cube_out.data=ma.array(cube_in.data,mask=mask_array)
    return cube_out

def mask_particle(Mask,particle,masked=False):
    import numpy as np 
    from copy import deepcopy
    Mask_i=deepcopy(Mask)
    Mask_i.data[Mask_i.data!=particle]=0
    if masked:
        Mask_i.data=np.ma.array(Mask_i.data,mask=Mask_i.data)
    return Mask_i   

def mask_particle_surface(Mask,particle,masked=False,z_coord=None):
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

def mask_particle_columns(Mask,particle,masked=False,z_coord=None):
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


# def constraint_cell(Track,mask_particle,width=None,x=None,):
#     from iris import Constraint
#     import numpy as np
    
#     time_coord=mask.coord('time')
#     time_units=time_coord.units
    
#     def time_condition(cell):
#         return time_units.num2date(Track.head(n=1)['time']) <= cell <= time_units.num2date(Track.tail(n=1)['time'])

#     constraint_time=iris.Constraint(time=time_condition)
#     mask_particle_i=mask_particle.extract(constraint_time)
#     mask_particle_surface_i=mask_particle_surface.extract(constraint_time)
    
#     x_dim=mask_particle_surface_i.coord_dims('projection_x_coordinate')[0]
#     y_dim=mask_particle_surface_i.coord_dims('projection_y_coordinate')[0]
#     x_coord=mask_particle_surface_i.coord('projection_x_coordinate')
#     y_coord=mask_particle_surface_i.coord('projection_y_coordinate')
    

#     logging.debug('min mask_particle_surface_i'+str(np.amin(mask_particle_surface_i.core_data())))
#     logging.debug('max mask_particle_surface_i'+str(np.amax(mask_particle_surface_i.core_data())))
#     logging.debug('shape mask_particle_surface_i'+str(mask_particle_surface_i.shape))

#     if (mask_particle_surface_i.core_data()>0).any():
#         box_mask_i=get_bounding_box(mask_particle_surface_i.core_data(),buffer=1)

#         box_mask=[[x_coord.points[box_mask_i[x_dim][0]],x_coord.points[box_mask_i[x_dim][1]]],
#                  [y_coord.points[box_mask_i[y_dim][0]],y_coord.points[box_mask_i[y_dim][1]]]]
#     else:
#         box_mask=[[np.nan,np.nan],[np.nan,np.nan]]

#         x_min=box_mask[0][0]
#         x_max=box_mask[0][1]
#         y_min=box_mask[1][0]
#         y_max=box_mask[1][1]
#     constraint_x=Constraint(projection_x_coordinate=lambda cell: int(x_min) < cell < int(x_max))
#     constraint_y=Constraint(projection_y_coordinate=lambda cell: int(y_min) < cell < int(y_max))

#     constraint=constraint_time & constraint_x & constraint_y


# def get_bounding_box(x,buffer=1):
#     """ Calculates the bounding box of a ndarray
#     https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
#     """
    
#     mask = x == 0

#     bbox = []
#     all_axis = np.arange(x.ndim)
#     logging.debug("all_axis "+str(all_axis))
#     for kdim in all_axis:
#         nk_dim = np.delete(all_axis, kdim)
#         mask_i = mask.all(axis=tuple(nk_dim))
#         dmask_i = np.diff(mask_i)
#         idx_i = np.nonzero(dmask_i)[0]
#         logging.debug("kfim, idx: "+str(kdim)+' , ' +str(idx_i))
#         if len(idx_i) == 1:
#             idx_i=np.array([idx_i,idx_i])
#         if len(idx_i) != 2:
#             raise ValueError('Algorithm failed, {} does not have 2 elements!'.format(idx_i))
#         idx_min=max(0,idx_i[0]+1-buffer)
#         idx_max=min(x.shape[kdim]-1,idx_i[1]+1+buffer)
#         logging.debug("kfim, idx_min: "+str(kdim)+' , '+str(idx_min))
#         logging.debug("kdim, idx_max: "+str(kdim)+' , '+str(idx_max))

#         # bbox.append(slice(idx_min, idx_max))
#         bbox.append([idx_min, idx_max])

#     return bbox

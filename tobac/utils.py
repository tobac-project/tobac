import logging
import numpy as np


def column_mask_from2D(mask_2D,cube,z_coord='model_level_number'):
    '''function to turn 2D watershedding mask into a 3D mask of selected columns
    Parameters
    ----------
    cube:              iris.cube.Cube 
                       data cube
    mask_2D:           iris.cube.Cube 
                       2D cube containing mask (int id for tacked volumes 0 everywhere else)
    z_coord:           str
                       name of the vertical coordinate in the cube
    Returns
    -------
    iris.cube.Cube 
                       3D cube containing columns of 2D mask (int id for tacked volumes 0 everywhere else)
    '''
    from copy import deepcopy
    mask_3D=deepcopy(cube)
    mask_3D.rename('segmentation_mask')
    dim=mask_3D.coord_dims(z_coord)[0]
    for i in range(len(mask_3D.coord(z_coord).points)):
        slc = [slice(None)] * len(mask_3D.shape)
        slc[dim] = slice(i,i+1)    
        mask_out=mask_3D[slc]
        mask_3D.data[slc]=mask_2D.core_data()
    return mask_3D


def mask_cube_cell(variable_cube,mask,cell,track):
    '''Mask cube for tracked volume of an individual cell   
    
    Parameters
    ----------
    variable_cube:     iris.cube.Cube 
                       unmasked data cube
    mask:              iris.cube.Cube 
                       cube containing mask (int id for tacked volumes 0 everywhere else)
    cell:          int
                       interger id of cell to create masked cube for
    
    Returns
    -------
    iris.cube.Cube 
                       Masked cube with data for respective cell
    '''
    from copy import deepcopy
    variable_cube_out=deepcopy(variable_cube)
    feature_ids=track.loc[track['cell']==cell,'feature'].values
    variable_cube_out=mask_cube_features(variable_cube,mask,feature_ids)
    return variable_cube_out

def mask_cube_all(variable_cube,mask):
    ''' Mask cube for untracked volume 
    
    Parameters
    ----------
    variable_cube:     iris.cube.Cube 
                       unmasked data cube
    mask:              iris.cube.Cube 
                       cube containing mask (int id for tacked volumes 0 everywhere else)
    
    Returns
    -------
    variable_cube_out: iris.cube.Cube 
                       Masked cube for untracked volume
    '''
    from dask.array import ma
    from copy import deepcopy
    variable_cube_out=deepcopy(variable_cube)
    variable_cube_out.data=ma.masked_where(mask.core_data()==0,variable_cube_out.core_data())    
    return variable_cube_out

def mask_cube_untracked(variable_cube,mask):
    '''Mask cube for untracked volume 
    
    Parameters
    ----------
    variable_cube:     iris.cube.Cube 
                       unmasked data cube
    mask:              iris.cube.Cube 
                       cube containing mask (int id for tacked volumes 0 everywhere else)
    
    Returns
    -------
    variable_cube_out: iris.cube.Cube 
                       Masked cube for untracked volume
    '''
    from dask.array import ma
    from copy import deepcopy
    variable_cube_out=deepcopy(variable_cube)
    variable_cube_out.data=ma.masked_where(mask.core_data()>0,variable_cube_out.core_data())    
    return variable_cube_out

def mask_cube(cube_in,mask):
    ''' Mask cube where mask is larger than zero
    
    Parameters
    ----------
    cube_in:           iris.cube.Cube 
                       unmasked data cube
    mask:              numpy.ndarray or dask.array 
                       mask to use for masking, >0 where cube is supposed to be masked
    
    Returns
    -------
    iris.cube.Cube 
                       Masked cube
    '''
    from dask.array import ma
    from copy import deepcopy
    cube_out=deepcopy(cube_in)
    cube_out.data=ma.masked_where(mask!=0,cube_in.core_data())
    return cube_out

def mask_cell(mask,cell,track,masked=False):
    '''create mask for specific cell
    
    Parameters
    ----------
    mask:              iris.cube.Cube 
                       cube containing mask (int id for tacked volumes 0 everywhere else)
    
    Returns
    -------
    numpy.ndarray 
                       Masked cube for untracked volume
    '''
    feature_ids=track.loc[track['cell']==cell,'feature'].values
    mask_i=mask_features(mask,feature_ids,masked=masked)
    return mask_i   

def mask_cell_surface(mask,cell,track,masked=False,z_coord='model_level_number'):
    '''Create surface projection of mask for individual cell
    
    Parameters
    ----------
    mask:              iris.cube.Cube 
                       cube containing mask (int id for tacked volumes 0 everywhere else)
    
    Returns
    -------
    iris.cube.Cube 
                       Masked cube for untracked volume
    '''
    feature_ids=track.loc[track['cell']==cell,'feature'].values
    mask_i_surface=mask_features_surface(mask,feature_ids,masked=masked,z_coord=z_coord)
    return mask_i_surface

def mask_cell_columns(mask,cell,track,masked=False,z_coord='model_level_number'):
    '''Create mask with entire columns for individual cell
    
    Parameters
    ----------
    mask:              iris.cube.Cube 
                       cube containing mask (int id for tacked volumes 0 everywhere else)
    
    Returns
    -------
    iris.cube.Cube 
                       Masked cube for untracked volume
    '''
    feature_ids=track.loc[track['cell']==cell].loc['feature']
    mask_i=mask_features_columns(mask,feature_ids,masked=masked,z_coord=z_coord)
    return mask_i

def mask_cube_features(variable_cube,mask,feature_ids):
    ''' Mask cube for tracked volume of an individual cell   
    
    Parameters
    ----------
    variable_cube:     iris.cube.Cube 
                       unmasked data cube
    mask:              iris.cube.Cube 
                       cube containing mask (int id for tacked volumes 0 everywhere else)
    cell:          int
                       interger id of cell to create masked cube for
    
    Returns
    -------
    iris.cube.Cube 
                       Masked cube with data for respective cell
    '''
    from dask.array import ma,isin
    from copy import deepcopy
    variable_cube_out=deepcopy(variable_cube)
    variable_cube_out.data=ma.masked_where(~isin(mask.core_data(),feature_ids),variable_cube_out.core_data())    
    return variable_cube_out

def mask_features(mask,feature_ids,masked=False):
    '''create mask for specific features
    
    Parameters
    ----------
    mask:              iris.cube.Cube 
                       cube containing mask (int id for tacked volumes 0 everywhere else)
    
    Returns
    -------
    numpy.ndarray 
                       Masked cube for untracked volume
    '''
    from dask.array import ma,isin
    from copy import deepcopy
    mask_i=deepcopy(mask)
    mask_i_data=mask_i.core_data()
    mask_i_data[~isin(mask_i.core_data(),feature_ids)]=0
    if masked:    
        mask_i.data=ma.masked_equal(mask_i.core_data(),0)
    return mask_i   

def mask_features_surface(mask,feature_ids,masked=False,z_coord='model_level_number'):
    ''' create surface mask for individual features 
    
    Parameters
    ----------
    mask:              iris.cube.Cube 
                       cube containing mask (int id for tacked volumes 0 everywhere else)
    
    Returns
    -------
    iris.cube.Cube 
                       Masked cube for untracked volume
    '''
    from iris.analysis import MAX
    from dask.array import ma,isin
    from copy import deepcopy
    mask_i=deepcopy(mask)
#     mask_i.data=[~isin(mask_i.data,feature_ids)]=0
    mask_i_data=mask_i.core_data()
    mask_i_data[~isin(mask_i.core_data(),feature_ids)]=0
    mask_i_surface=mask_i.collapsed(z_coord,MAX)
    if masked:
        mask_i_surface.data=ma.masked_equal(mask_i_surface.core_data(),0)
    return mask_i_surface    

def mask_all_surface(mask,masked=False,z_coord='model_level_number'):
    ''' create surface mask for individual features 
    
    Parameters
    ----------
    mask:              iris.cube.Cube 
                       cube containing mask (int id for tacked volumes 0 everywhere else)
    
    Returns
    -------
    mask_i_surface:    iris.cube.Cube (2D)
                       Mask with 1 below features and 0 everywhere else
    '''
    from iris.analysis import MAX
    from dask.array import ma,isin
    from copy import deepcopy
    mask_i=deepcopy(mask)
    mask_i_surface=mask_i.collapsed(z_coord,MAX)
    mask_i_surface_data=mask_i_surface.core_data()
    mask_i_surface[mask_i_surface_data>0]=1
    if masked:
        mask_i_surface.data=ma.masked_equal(mask_i_surface.core_data(),0)
    return mask_i_surface    


# def mask_features_columns(mask,feature_ids,masked=False,z_coord='model_level_number'):
#     ''' Mask cube for untracked volume 
#     Input:
#     variable_cube:     iris.cube.Cube 
#                        unmasked data cube
#     mask:              iris.cube.Cube 
#                        cube containing mask (int id for tacked volumes 0 everywhere else)
#     Output:
#     variable_cube_out: iris.cube.Cube 
#                        Masked cube for untracked volume
#     '''
#     from iris.analysis import MAX
#     import numpy as np 
#     from copy import deepcopy
#     mask_i=deepcopy(mask)
#     mask_i.data[~np.isin(mask_i.data,feature_ids)]=0
#     mask_i_surface=mask_i.collapsed(z_coord,MAX)
#     for cube_slice in mask_i.slices(['time','x','y']):
#             cube_slice.data=mask_i_surface.core_data()
#     if masked:
#         mask_i.data=np.ma.array(mask_i.data,mask=mask_i.data)
#     return mask_i




#def constraint_cell(track,mask_cell,width=None,x=None,):
#     from iris import Constraint
#     import numpy as np
#    
#     time_coord=mask_cell.coord('time')
#     time_units=time_coord.units
#    
#     def time_condition(cell):
#         return time_units.num2date(track.head(n=1)['time']) <= cell <= time_units.num2date(track.tail(n=1)['time'])
#
#     constraint_time=Constraint(time=time_condition)
##     mask_cell_i=mask_cell.extract(constraint_time)
#     mask_cell_surface_i=mask_cell_surface.extract(constraint_time)
#    
#     x_dim=mask_cell_surface_i.coord_dims('projection_x_coordinate')[0]
#     y_dim=mask_cell_surface_i.coord_dims('projection_y_coordinate')[0]
#     x_coord=mask_cell_surface_i.coord('projection_x_coordinate')
#     y_coord=mask_cell_surface_i.coord('projection_y_coordinate')
#    
#     if (mask_cell_surface_i.core_data()>0).any():
#         box_mask_i=get_bounding_box(mask_cell_surface_i.core_data(),buffer=1)
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
    
def add_coordinates(t,variable_cube):
    import numpy as np
    '''Function adding coordinates from the tracking cube to the trajectories
        for the 2D case: time, longitude&latitude, x&y dimensions
    
    Parameters
    ----------
    t:             pandas DataFrame
                   trajectories/features
    variable_cube: iris.cube.Cube 
        Cube (usually the one you are tracking on) at least conaining the dimension of 'time'. 
        Typically, 'longitude','latitude','x_projection_coordinate','y_projection_coordinate', 
        are the coordinates that we expect, although this function
        will happily interpolate along any dimension coordinates you give. 
    
    Returns
    -------
    pandas DataFrame 
                   trajectories with added coordinates
    '''
    from scipy.interpolate import interp2d, interp1d

    logging.debug('start adding coordinates from cube')

    # pull time as datetime object and timestr from input data and add it to DataFrame:    
    t['time']=None
    t['timestr']=None
    
    
    logging.debug('adding time coordinate')

    time_in=variable_cube.coord('time')
    time_in_datetime=time_in.units.num2date(time_in.points)
    
    t["time"]=time_in_datetime[t['frame']]
    t["timestr"]=[x.strftime('%Y-%m-%d %H:%M:%S') for x in time_in_datetime[t['frame']]]

    # Get list of all coordinates in input cube except for time (already treated):
    coord_names=[coord.name() for coord in  variable_cube.coords()]
    coord_names.remove('time')
    
    logging.debug('time coordinate added')

    # chose right dimension for horizontal axis based on time dimension:    
    ndim_time=variable_cube.coord_dims('time')[0]
    if ndim_time==0:
        hdim_1=1
        hdim_2=2
    elif ndim_time==1:
        hdim_1=0
        hdim_2=2
    elif ndim_time==2:
        hdim_1=0
        hdim_2=1
    
    # create vectors to use to interpolate from pixels to coordinates
    dimvec_1=np.arange(variable_cube.shape[hdim_1])
    dimvec_2=np.arange(variable_cube.shape[hdim_2])

    # loop over coordinates in input data:
    for coord in coord_names:
        logging.debug('adding coord: '+ coord)
        # interpolate 2D coordinates:
        if variable_cube.coord(coord).ndim==1:

            if variable_cube.coord_dims(coord)==(hdim_1,):
                f=interp1d(dimvec_1,variable_cube.coord(coord).points,fill_value="extrapolate")
                coordinate_points=f(t['hdim_1'])

            if variable_cube.coord_dims(coord)==(hdim_2,):
                f=interp1d(dimvec_2,variable_cube.coord(coord).points,fill_value="extrapolate")
                coordinate_points=f(t['hdim_2'])

        # interpolate 2D coordinates:
        elif variable_cube.coord(coord).ndim==2:

            if variable_cube.coord_dims(coord)==(hdim_1,hdim_2):
                f=interp2d(dimvec_2,dimvec_1,variable_cube.coord(coord).points)
                coordinate_points=[f(a,b) for a,b in zip(t['hdim_2'],t['hdim_1'])]

            if variable_cube.coord_dims(coord)==(hdim_2,hdim_1):
                f=interp2d(dimvec_1,dimvec_2,variable_cube.coord(coord).points)
                coordinate_points=[f(a,b) for a,b in zip(t['hdim_1'],t['hdim_2'])]

        # interpolate 3D coordinates:            
        # mainly workaround for wrf latitude and longitude (to be fixed in future)
        
        elif variable_cube.coord(coord).ndim==3:

            if variable_cube.coord_dims(coord)==(ndim_time,hdim_1,hdim_2):
                f=interp2d(dimvec_2,dimvec_1,variable_cube[0,:,:].coord(coord).points)
                coordinate_points=[f(a,b) for a,b in zip(t['hdim_2'],t['hdim_1'])]
            
            if variable_cube.coord_dims(coord)==(ndim_time,hdim_2,hdim_1):
                f=interp2d(dimvec_1,dimvec_2,variable_cube[0,:,:].coord(coord).points)
                coordinate_points=[f(a,b) for a,b in zip(t['hdim_1'],t['hdim_2'])]

        
            if variable_cube.coord_dims(coord)==(hdim_1,ndim_time,hdim_2):
                f=interp2d(dimvec_2,dimvec_1,variable_cube[:,0,:].coord(coord).points)
                coordinate_points=[f(a,b) for a,b in zip(t['hdim_2'],t['hdim_1'])]

            if variable_cube.coord_dims(coord)==(hdim_1,hdim_2,ndim_time):
                f=interp2d(dimvec_2,dimvec_1,variable_cube[:,:,0].coord(coord).points)
                coordinate_points=[f(a,b) for a,b in zip(t['hdim_2'],t['hdim1'])]

                    
            if variable_cube.coord_dims(coord)==(hdim_2,ndim_time,hdim_1):
                f=interp2d(dimvec_1,dimvec_2,variable_cube[:,0,:].coord(coord).points)
                coordinate_points=[f(a,b) for a,b in zip(t['hdim_1'],t['hdim_2'])]

            if variable_cube.coord_dims(coord)==(hdim_2,hdim_1,ndim_time):
                f=interp2d(dimvec_1,dimvec_2,variable_cube[:,:,0].coord(coord).points)
                coordinate_points=[f(a,b) for a,b in zip(t['hdim_1'],t['hdim_2'])]

        # write resulting array or list into DataFrame:
        t[coord]=coordinate_points

        logging.debug('added coord: '+ coord)
    return t

def add_coordinates_3D(t,variable_cube):
    import numpy as np
    '''Function adding coordinates from the tracking cube to the trajectories
        for the 3D case: time, longitude&latitude, x&y dimensions, and altitude
    
    Parameters
    ----------
    t:             pandas DataFrame
                   trajectories/features
    variable_cube: iris.cube.Cube 
        Cube (usually the one you are tracking on) at least conaining the dimension of 'time'. 
        Typically, 'longitude','latitude','x_projection_coordinate','y_projection_coordinate', 
        and 'altitude' (if 3D) are the coordinates that we expect, although this function
        will happily interpolate along any dimension coordinates you give. 
    
    Returns
    -------
    pandas DataFrame 
                   trajectories with added coordinates
    '''
    from scipy.interpolate import interp2d, interp1d

    logging.debug('start adding coordinates from cube')

    # pull time as datetime object and timestr from input data and add it to DataFrame:    
    t['time']=None
    t['timestr']=None
    
    
    logging.debug('adding time coordinate')

    time_in=variable_cube.coord('time')
    time_in_datetime=time_in.units.num2date(time_in.points)
    
    t["time"]=time_in_datetime[t['frame']]
    t["timestr"]=[x.strftime('%Y-%m-%d %H:%M:%S') for x in time_in_datetime[t['frame']]]

    # Get list of all coordinates in input cube except for time (already treated):
    coord_names=[coord.name() for coord in  variable_cube.coords()]
    coord_names.remove('time')
    
    logging.debug('time coordinate added')

    # chose right dimension for horizontal and vertical axes based on time dimension:    
    ndim_time=variable_cube.coord_dims('time')[0]
    if ndim_time==0:
        vdim=1
        hdim_1=2
        hdim_2=3
    elif ndim_time==1:
        vdim=0
        hdim_1=2
        hdim_2=3
    elif ndim_time==2:
        vdim=0
        hdim_1=1
        hdim_2=3
    elif ndim_time==3:
        vdim=0
        hdim_1=1
        hdim_2=2
    
    # create vectors to use to interpolate from pixels to coordinates
    dimvec_1=np.arange(variable_cube.shape[vdim])
    dimvec_2=np.arange(variable_cube.shape[hdim_1])
    dimvec_3=np.arange(variable_cube.shape[hdim_2])

    # loop over coordinates in input data:
    for coord in coord_names:
        logging.debug('adding coord: '+ coord)
        # interpolate 1D coordinates:
        if variable_cube.coord(coord).ndim==1:
            
            if variable_cube.coord_dims(coord)==(vdim,):
                f=interp1d(dimvec_1,variable_cube.coord(coord).points,fill_value="extrapolate")
                coordinate_points=f(t['vdim'])

            if variable_cube.coord_dims(coord)==(hdim_1,):
                f=interp1d(dimvec_2,variable_cube.coord(coord).points,fill_value="extrapolate")
                coordinate_points=f(t['hdim_1'])

            if variable_cube.coord_dims(coord)==(hdim_2,):
                f=interp1d(dimvec_3,variable_cube.coord(coord).points,fill_value="extrapolate")
                coordinate_points=f(t['hdim_2'])

        # interpolate 2D coordinates:
        elif variable_cube.coord(coord).ndim==2:

            if variable_cube.coord_dims(coord)==(hdim_1,hdim_2):
                f=interp2d(dimvec_3,dimvec_2,variable_cube.coord(coord).points)
                coordinate_points=[f(a,b) for a,b in zip(t['hdim_2'],t['hdim_1'])]

            if variable_cube.coord_dims(coord)==(hdim_2,hdim_1):
                f=interp2d(dimvec_2,dimvec_3,variable_cube.coord(coord).points)
                coordinate_points=[f(a,b) for a,b in zip(t['hdim_1'],t['hdim_2'])]

        # interpolate 3D coordinates:            
        # mainly workaround for wrf latitude and longitude (to be fixed in future)
        
        elif variable_cube.coord(coord).ndim==3:

            if variable_cube.coord_dims(coord)==(ndim_time,hdim_1,hdim_2):
                f=interp2d(dimvec_2,dimvec_1,variable_cube[0,:,:].coord(coord).points)
                coordinate_points=[f(a,b) for a,b in zip(t['hdim_2'],t['hdim_1'])]
            
            if variable_cube.coord_dims(coord)==(ndim_time,hdim_2,hdim_1):
                f=interp2d(dimvec_1,dimvec_2,variable_cube[0,:,:].coord(coord).points)
                coordinate_points=[f(a,b) for a,b in zip(t['hdim_1'],t['hdim_2'])]

        
            if variable_cube.coord_dims(coord)==(hdim_1,ndim_time,hdim_2):
                f=interp2d(dimvec_2,dimvec_1,variable_cube[:,0,:].coord(coord).points)
                coordinate_points=[f(a,b) for a,b in zip(t['hdim_2'],t['hdim_1'])]

            if variable_cube.coord_dims(coord)==(hdim_1,hdim_2,ndim_time):
                f=interp2d(dimvec_2,dimvec_1,variable_cube[:,:,0].coord(coord).points)
                coordinate_points=[f(a,b) for a,b in zip(t['hdim_2'],t['hdim1'])]

                    
            if variable_cube.coord_dims(coord)==(hdim_2,ndim_time,hdim_1):
                f=interp2d(dimvec_1,dimvec_2,variable_cube[:,0,:].coord(coord).points)
                coordinate_points=[f(a,b) for a,b in zip(t['hdim_1'],t['hdim_2'])]

            if variable_cube.coord_dims(coord)==(hdim_2,hdim_1,ndim_time):
                f=interp2d(dimvec_1,dimvec_2,variable_cube[:,:,0].coord(coord).points)
                coordinate_points=[f(a,b) for a,b in zip(t['hdim_1'],t['hdim_2'])]

        # write resulting array or list into DataFrame:
        t[coord]=coordinate_points

        logging.debug('added coord: '+ coord)
    return t




def get_bounding_box(x,buffer=1):
    from numpy import delete,arange,diff,nonzero,array
    """Calculates the bounding box of a ndarray
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
        elif len(idx_i) == 1:
            idx_i=array([idx_i,idx_i])
        # make sure there is two values in idx_i
        elif len(idx_i) > 2:
            idx_i=array([idx_i[0],idx_i[-1]])
        # caluclate min and max values for idx_i and append them to list
        idx_min=max(0,idx_i[0]+1-buffer)
        idx_max=min(x.shape[kdim]-1,idx_i[1]+1+buffer)
        bbox.append([idx_min, idx_max])
    return bbox

def get_spacings(field_in,grid_spacing=None,time_spacing=None):
    import numpy as np
    from copy import deepcopy
    # set horizontal grid spacing of input data
    # If cartesian x and y corrdinates are present, use these to determine dxy (vertical grid spacing used to transfer pixel distances to real distances):
    coord_names=[coord.name() for coord in  field_in.coords()]
    
    if (('projection_x_coordinate' in coord_names and 'projection_y_coordinate' in coord_names) and  (grid_spacing is None)):
        x_coord=deepcopy(field_in.coord('projection_x_coordinate'))
        x_coord.convert_units('metre')
        dx=np.diff(field_in.coord('projection_y_coordinate')[0:2].points)[0]
        y_coord=deepcopy(field_in.coord('projection_y_coordinate'))
        y_coord.convert_units('metre')
        dy=np.diff(field_in.coord('projection_y_coordinate')[0:2].points)[0]
        dxy=0.5*(dx+dy)
    elif grid_spacing is not None:
        dxy=grid_spacing
    else:
        ValueError('no information about grid spacing, need either input cube with projection_x_coord and projection_y_coord or keyword argument grid_spacing')
    
    # set horizontal grid spacing of input data
    if (time_spacing is None):    
        # get time resolution of input data from first to steps of input cube:
        time_coord=field_in.coord('time')
        dt=(time_coord.units.num2date(time_coord.points[1])-time_coord.units.num2date(time_coord.points[0])).seconds
    elif (time_spacing is not None):
        # use value of time_spacing for dt:
        dt=time_spacing
    return dxy,dt

def get_label_props_in_dict(labels):
    '''Function to get the label properties into a dictionary format.
    
    Parameters
    ----------
    labels:    2D or 3D array-like
        comes from the `skimage.measure.label` function
    
    Returns
    -------
    dict
        output from skimage.measure.regionprops in dictionary format, where they key is the label number
    '''
    import skimage.measure
    
    region_properties_raw = skimage.measure.regionprops(labels)
    region_properties_dict = dict()
    for region_prop in region_properties_raw:
        region_properties_dict[region_prop.label] = region_prop
    
    return region_properties_dict

def get_indices_of_labels_from_reg_prop_dict(region_property_dict):
    '''Function to get the x, y, and z indices (as well as point count) of all labeled regions.
 
    Parameters
    ----------
    region_property_dict:    dict of region_property objects
        This dict should come from the get_label_props_in_dict function.

    Returns
    -------
    dict (key: label number, int)
        The number of points in the label number
    dict (key: label number, int)
        The z indices in the label number. If a 2D property dict is passed, this value is not returned
    dict (key: label number, int)
        the y indices in the label number
    dict (key: label number, int)
        the x indices in the label number
    
    Raises
    ------
    ValueError
        a ValueError is raised if there are no regions in the region property dict

    '''
    
    import skimage.measure
    import numpy as np

    if len(region_property_dict) ==0:
        raise ValueError("No regions!")


    z_indices = dict()
    y_indices = dict()
    x_indices = dict()
    curr_loc_indices = dict()
    is_3D = False
        
    #loop through all skimage identified regions
    for region_prop_key in region_property_dict:
        region_prop = region_property_dict[region_prop_key]
        index = region_prop.label
        if len(region_prop.coords[0])>=3:
            is_3D = True
            curr_z_ixs, curr_y_ixs, curr_x_ixs = np.transpose(region_prop.coords)
            z_indices[index] = curr_z_ixs
        else:
            curr_y_ixs, curr_x_ixs = np.transpose(region_prop.coords)
            z_indices[index] = -1

        y_indices[index] = curr_y_ixs
        x_indices[index] = curr_x_ixs
        curr_loc_indices[index] = len(curr_y_ixs)
                        
    #print("indices found")
    if is_3D:
        return [curr_loc_indices, z_indices, y_indices, x_indices]
    else: 
        return [curr_loc_indices, y_indices, x_indices]

def adjust_pbc_point(in_dim, dim_min, dim_max):
    '''Function to adjust a point to the other boundary for PBCs
    
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
        The adjusted point on the opposite boundary
    
    Raises
    ------
    ValueError
        If in_dim isn't on one of the boundary points
    '''
    if in_dim == dim_min:
        return dim_max
    elif in_dim == dim_max:
        return dim_min
    else:
        raise ValueError("In adjust_pbc_point, in_dim isn't on a boundary.")


def get_pbc_coordinates(h1_min, h1_max, h2_min, h2_max, 
                        h1_start_coord, h1_end_coord, h2_start_coord, h2_end_coord,
                        PBC_flag = 'none'):
    '''Function to get the *actual* coordinate boxes of interest given a set of shifted 
    coordinates with periodic boundaries. 
    
    For example, if you pass in [as h1_start_coord, h1_end_coord, h2_start_coord, h2_end_coord] 
    (-3, 5, 2,6) with PBC_flag of 'both' or 'hdim_1', h1_max of 10, and h1_min of 0
    this function will return: [(0,5,2,6), (7,10,2,6)].

    If you pass in something outside the bounds of the array, this will truncate your
    requested box. For example, if you pass in [as h1_start_coord, h1_end_coord, h2_start_coord, h2_end_coord] 
    (-3, 5, 2,6) with PBC_flag of 'none' or 'hdim_2', this function will return:
    [(0,5,2,6)], assuming h1_min is 0. 

    For cases where PBC_flag is 'both' and we have a corner case, it is possible
    to get overlapping boundaries. For example, if you pass in (-6, 5, -6, 5)

    Parameters
    ----------
    h1_min: int
        Minimum array value in hdim_1, typically 0.
    h1_max: int
        Maximum array value in hdim_1 (exclusive). h1_max - h1_min should be the size in h1.
    h2_min: int
        Minimum array value in hdim_2, typically 0.
    h2_max: int
        Maximum array value in hdim_2 (exclusive). h2_max - h2_min should be the size in h2. 
    h1_start_coord: int
        Start coordinate in hdim_1. Can be < h1_min if dealing with PBCs.
    h1_end_coord: int
        End coordinate in hdim_1. Can be >= h1_max if dealing with PBCs.
    h2_start_coord: int
        Start coordinate in hdim_2. Can be < h2_min if dealing with PBCs.
    h2_end_coord: int
        End coordinate in hdim_2. Can be >= h2_max if dealing with PBCs.
    PBC_flag : str('none', 'hdim_1', 'hdim_2', 'both')
        Sets whether to use periodic boundaries, and if so in which directions.
        'none' means that we do not have periodic boundaries
        'hdim_1' means that we are periodic along hdim1
        'hdim_2' means that we are periodic along hdim2
        'both' means that we are periodic along both horizontal dimensions

    Returns
    -------
    list of tuples
        A list of tuples containing (h1_start, h1_end, h2_start, h2_end) of each of the
        boxes needed to encompass the coordinates.
    '''

    if PBC_flag not in ['none', 'hdim_1', 'hdim_2', 'both']:
        raise ValueError("PBC_flag must be 'none', 'hdim_1', 'hdim_2', or 'both'")
    

    h1_start_coords = list()
    h1_end_coords = list()
    h2_start_coords = list()
    h2_end_coords = list()


    # In both of these cases, we just need to truncate the hdim_1 points. 
    if PBC_flag in ['none', 'hdim_2']:
        h1_start_coords.append(max(h1_min, h1_start_coord))
        h1_end_coords.append(min(h1_max, h1_end_coord))
    
    
    # In both of these cases, we only need to truncate the hdim_2 points.
    if PBC_flag in ['none', 'hdim_1']:
        h2_start_coords.append(max(h2_min, h2_start_coord))
        h2_end_coords.append(min(h2_max, h2_end_coord))

    # If the PBC flag is none, we can just return.
    if PBC_flag == 'none':
        return [(h1_start_coords[0], h1_end_coords[0], h2_start_coords[0], h2_end_coords[0])]

    # We have at least one periodic boundary.         

    # hdim_1 boundary is periodic. 
    if PBC_flag in ['hdim_1', 'both']:
        if (h1_end_coord - h1_start_coord) >= (h1_max - h1_min):
            # In this case, we have selected the full h1 length of the domain,
            # so we set the start and end coords to just that.
            h1_start_coords.append(h1_min)
            h1_end_coords.append(h1_max)

        # We know we only have either h1_end_coord > h1_max or h1_start_coord < h1_min
        # and not both. If both are true, the previous if statement should trigger.
        elif h1_start_coord < h1_min:
            # First set of h1 start coordinates
            h1_start_coords.append(h1_min)
            h1_end_coords.append(h1_end_coord)
            # Second set of h1 start coordinates
            pts_from_begin = h1_min - h1_start_coord
            h1_start_coords.append(h1_max - pts_from_begin)
            h1_end_coords.append(h1_max)

        elif h1_end_coord > h1_max:
            h1_start_coords.append(h1_start_coord)
            h1_end_coords.append(h1_max)
            pts_from_end = h1_end_coord - h1_max
            h1_start_coords.append(h1_min)
            h1_end_coords.append(h1_min + pts_from_end)

        # We have no PBC-related issues, actually
        else:
            h1_start_coords.append(h1_start_coord)
            h1_end_coords.append(h1_end_coord)
    
    if PBC_flag in ['hdim_2', 'both']:
        if (h2_end_coord - h2_start_coord) >= (h2_max - h2_min):
            # In this case, we have selected the full h2 length of the domain,
            # so we set the start and end coords to just that.
            h2_start_coords.append(h2_min)
            h2_end_coords.append(h2_max)

        # We know we only have either h1_end_coord > h1_max or h1_start_coord < h1_min
        # and not both. If both are true, the previous if statement should trigger.
        elif h2_start_coord < h2_min:
            # First set of h1 start coordinates
            h2_start_coords.append(h2_min)
            h2_end_coords.append(h2_end_coord)
            # Second set of h1 start coordinates
            pts_from_begin = h2_min - h2_start_coord
            h2_start_coords.append(h2_max - pts_from_begin)
            h2_end_coords.append(h2_max)

        elif h2_end_coord > h2_max:
            h2_start_coords.append(h2_start_coord)
            h2_end_coords.append(h2_max)
            pts_from_end = h2_end_coord - h2_max
            h2_start_coords.append(h2_min)
            h2_end_coords.append(h2_min + pts_from_end)

        # We have no PBC-related issues, actually
        else:
            h2_start_coords.append(h2_start_coord)
            h2_end_coords.append(h2_end_coord)

    out_coords = list()
    for h1_start_coord_single, h1_end_coord_single in zip(h1_start_coords, h1_end_coords):
        for h2_start_coord_single, h2_end_coord_single in zip(h2_start_coords, h2_end_coords):
            out_coords.append((h1_start_coord_single, h1_end_coord_single, h2_start_coord_single, h2_end_coord_single))
    return out_coords

def njit_if_available(func, **kwargs):
    '''Decorator to wrap a function with numba.njit if available.
    If numba isn't available, it just returns the function. 
    '''
    try:
        from numba import njit
        return njit(func, kwargs)
    except ModuleNotFoundError:
        return func


@njit_if_available
def calc_distance_coords_pbc(coords_1, coords_2, min_h1, max_h1, min_h2, max_h2,
                             PBC_flag):
    '''Function to calculate the distance between cartesian
    coordinate set 1 and coordinate set 2. Note that we assume both
    coordinates are within their min/max already. 

    Parameters
    ----------
    coords_1: 2D or 3D array-like
        Set of coordinates passed in from trackpy of either (vdim, hdim_1, hdim_2)
        coordinates or (hdim_1, hdim_2) coordinates.
    coords_2: 2D or 3D array-like
        Similar to coords_1, but for the second pair of coordinates
    min_h1: int
        Minimum point in hdim_1
    max_h1: int
        Maximum point in hdim_1, exclusive. max_h1-min_h1 should be the size. 
    min_h2: int
        Minimum point in hdim_2
    max_h2: int
        Maximum point in hdim_2, exclusive. max_h2-min_h2 should be the size. 
    PBC_flag : str('none', 'hdim_1', 'hdim_2', 'both')
        Sets whether to use periodic boundaries, and if so in which directions.
        'none' means that we do not have periodic boundaries
        'hdim_1' means that we are periodic along hdim1
        'hdim_2' means that we are periodic along hdim2
        'both' means that we are periodic along both horizontal dimensions
    
    Returns
    -------
    float
        Distance between coords_1 and coords_2 in cartesian space.

    '''
    
    is_3D = len(coords_1)== 3
    size_h1 = max_h1 - min_h1
    size_h2 = max_h2 - min_h2

    if not is_3D:
        # Let's make the accounting easier.
        coords_1 = np.array((0, coords_1[0], coords_1[1]))
        coords_2 = np.array((0, coords_2[0], coords_2[1]))

    if PBC_flag in ['hdim_1', 'both']:
        mod_h1 = size_h1
    else:
        mod_h1 = 0
    if PBC_flag in ['hdim_2', 'both']:
        mod_h2 = size_h2
    else:
        mod_h2 = 0
    max_dims = np.array((0, mod_h1, mod_h2))
    deltas = np.abs(coords_1 - coords_2)
    deltas = np.where(deltas > 0.5 * max_dims, deltas - max_dims, deltas)
    return np.sqrt(np.sum(deltas**2))

'''Provide essential methods.

'''
import logging

from .convert import xarray_to_iris, iris_to_xarray

@xarray_to_iris
def add_coordinates(t,variable_cube):
    '''Add coordinates from the tracking cube to the trajectories.

    Coordinates: time, longitude&latitude, x&y dimensions.

    Parameters
    ----------
    t : pandas.DataFrame
        Trajectories/features from trackpy.

    variable_cube : iris.cube.Cube
        Cube containing the dimensions 'time', 'longitude', 'latitude',
        'x_projection_coordinate', 'y_projection_coordinate', usually
        cube that the tracking is performed on.

    Returns
    -------
    t : pandas.DataFrame
        Trajectories with added coordinated.

    Notes
    -----
    where appropriate replace description of variable_cube with
    description above
    '''

    import numpy as np
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

@xarray_to_iris
def get_bounding_box(x,buffer=1):
    '''Calculates the bounding box of a ndarray.

    https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array

    Parameters
    ----------
    x

    buffer : int, optional
        Default is 1.

    Returns
    -------
    bbox

    Notes
    -----
    unsure about anything
    '''

    from numpy import delete,arange,diff,nonzero,array

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

@xarray_to_iris
def get_spacings(field_in,grid_spacing=None,time_spacing=None):
    '''
    Parameters
    ----------
    field_in : iris.cube.Cube
        Input field where to get spacings.

    grid_spacing : float, optional
        Grid spacing in input data. Default is None.

    time_spacing : float, optional
	Time resolution of input data. Default is None.

    Returns
    -------
    dxy : float
	Grid spacing.

    dt : float
	Time resolution.

    Raises
    ------
    ValueError
        If input_cube does not contail projection_x_coord and
        projection_y_coord or keyword argument grid_spacing.

    Notes
    -----
    need short summary
    '''

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

# @xarray_to_iris
# def constraint_cell(track,mask_cell,width=None,x=None,):
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

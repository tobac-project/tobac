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
    mask_3D.rename('segmentation_mask')
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


def mask_cube_cell(variable_cube,mask,cell):
    ''' Mask cube for tracked volume of an individual cell   
    Input:
    variable_cube:     iris.cube.Cube 
                       unmasked data cube
    mask:              iris.cube.Cube 
                       cube containing mask (int id for tacked volumes 0 everywhere else)
    cell:          int
                       interger id of cell to create masked cube for
    Output:
    variable_cube_out: iris.cube.Cube 
                       Masked cube with data for respective cell
    '''
    import numpy as np 
    from copy import deepcopy
    variable_cube_out=deepcopy(variable_cube)
    mask_i=mask.data!=cell
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

def mask_cell(Mask,cell,masked=False):
    ''' create mask for specific cell
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
    Mask_i.data[Mask_i.data!=cell]=0
    if masked:
        Mask_i.data=np.ma.array(Mask_i.data,mask=Mask_i.data)
    return Mask_i   

def mask_cell_surface(Mask,cell,masked=False,z_coord='model_level_number'):
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
    Mask_i.data[Mask_i.data!=cell]=0
    for coord in  Mask_i.coords():
        if coord.ndim>1 and Mask_i.coord_dims(z_coord)[0] in Mask_i.coord_dims(coord):
            Mask_i.remove_coord(coord.name())
    Mask_i_surface=Mask_i.collapsed(z_coord,MAX)
    if masked:
        Mask_i_surface.data=np.ma.array(Mask_i_surface.data,mask=Mask_i_surface.data)
    return Mask_i_surface    

def mask_cell_columns(Mask,cell,masked=False,z_coord='model_level_number'):
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
    Mask_i.data[Mask_i.data!=cell]=0
    for coord in  Mask_i.coords():
        if coord.ndim>1 and Mask_i.coord_dims(z_coord)[0] in Mask_i.coord_dims(coord):
            Mask_i.remove_coord(coord.name())
    Mask_i_surface=Mask_i.collapsed(z_coord,MAX)
    for cube_slice in Mask_i.slices(['time','x','y']):
            cube_slice.data=Mask_i_surface.core_data()
    if masked:
        Mask_i.data=np.ma.array(Mask_i.data,mask=Mask_i.data)
    return Mask_i


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

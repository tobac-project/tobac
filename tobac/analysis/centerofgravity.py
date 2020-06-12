'''Identify center of gravity and mass for analysis.
'''

import logging
import os

from tobac.utils.convert import xarray_to_iris,iris_to_xarray


@xarray_to_iris
def cog_cell(cell,Tracks=None,M_total=None,M_liquid=None,
             M_frozen=None,
             Mask=None,
             savedir=None):
    '''
    Parameters
    ----------
    cell : int
        Integer id of cell to create masked cube for output.

    Tracks : optional
        Default is None.

    M_total : subset of cube, optional
        Default is None.

    M_liquid : subset of cube, optional
        Default is None.

    M_frozen : subset of cube, optional
        Default is None.

    savedir : str
        Default is None.

    Returns
    -------
    None

    Notes
    -----
    unsure about anything
    needs a short summary
    '''
    
    from iris import Constraint
    logging.debug('Start calculating COG for '+str(cell))
    Track=Tracks[Tracks['cell']==cell]
    constraint_time=Constraint(time=lambda cell: Track.head(1)['time'].values[0] <= cell <= Track.tail(1)['time'].values[0])
    M_total_i=M_total.extract(constraint_time)
    M_liquid_i=M_liquid.extract(constraint_time)
    M_frozen_i=M_frozen.extract(constraint_time)
    Mask_i=Mask.extract(constraint_time)

    savedir_cell=os.path.join(savedir,'cells',str(int(cell)))
    os.makedirs(savedir_cell,exist_ok=True)
    savefile_COG_total_i=os.path.join(savedir_cell,'COG_total'+'_'+str(int(cell))+'.h5')
    savefile_COG_liquid_i=os.path.join(savedir_cell,'COG_liquid'+'_'+str(int(cell))+'.h5')
    savefile_COG_frozen_i=os.path.join(savedir_cell,'COG_frozen'+'_'+str(int(cell))+'.h5')
    
    Tracks_COG_total_i=calculate_cog(Track,M_total_i,Mask_i)
#   Tracks_COG_total_list.append(Tracks_COG_total_i)
    logging.debug('COG total loaded for ' +str(cell))
    
    Tracks_COG_liquid_i=calculate_cog(Track,M_liquid_i,Mask_i)
#   Tracks_COG_liquid_list.append(Tracks_COG_liquid_i)
    logging.debug('COG liquid loaded for ' +str(cell))
    Tracks_COG_frozen_i=calculate_cog(Track,M_frozen_i,Mask_i)
#   Tracks_COG_frozen_list.append(Tracks_COG_frozen_i)
    logging.debug('COG frozen loaded for ' +str(cell))
    
    Tracks_COG_total_i.to_hdf(savefile_COG_total_i,'table')
    Tracks_COG_liquid_i.to_hdf(savefile_COG_liquid_i,'table')
    Tracks_COG_frozen_i.to_hdf(savefile_COG_frozen_i,'table')
    logging.debug('individual COG calculated and saved to '+ savedir_cell)



@xarray_to_iris
def calculate_cog(tracks,mass,mask):
    '''Caluclate center of gravity and mass forech tracked cell.

    Parameters
    ----------
    tracks : pandas.DataFrame
         DataFrame containing trajectories of cell centers.

    mass : iris.cube.Cube
        Cube of quantity (need coordinates 'time',
        'geopotential_height','projection_x_coordinate' and
        'projection_y_coordinate').

    mask : iris.cube.Cube
        Cube containing mask (int > where belonging to cloud volume,
        0 everywhere else).

    Returns
    -------
    tracks_out : pandas.DataFrame
        Dataframe containing t, x, y, z positions of center of gravity
        and total cloud mass each tracked cells at each timestep.
    '''

    from tobac.utils import mask_cube_cell
    from iris import Constraint
    
    logging.info('start calculating centre of gravity for tracked cells')

    tracks_out=tracks[['time','frame','cell','time_cell']]

    for i_row,row in tracks_out.iterrows():        
        cell=row['cell']
        constraint_time=Constraint(time=row['time'])
        mass_i=mass.extract(constraint_time)
        mask_i=mask.extract(constraint_time)
        mass_masked_i=mask_cube_cell(mass_i,mask_i,cell)
        x_M,y_M,z_M,mass_M=center_of_gravity(mass_masked_i)
        tracks_out.loc[i_row,'x_M']=float(x_M)
        tracks_out.loc[i_row,'y_M']=float(y_M)
        tracks_out.loc[i_row,'z_M']=float(z_M)
        tracks_out.loc[i_row,'mass']=float(mass_M)
        
    logging.info('Finished calculating centre of gravity for tracked cells')

    return tracks_out
    
@xarray_to_iris
def calculate_cog_untracked(mass,mask):
    '''Caluclate center of gravity and mass for untracked domain parts.

    Parameters
    ----------
    mass : iris.cube.Cube
        Cube of quantity (need coordinates 'time',
        'geopotential_height','projection_x_coordinate' and
        'projection_y_coordinate').

    mask : iris.cube.Cube
        Cube containing mask (int > where belonging to cloud volume,
        0 everywhere else).

    Returns
    -------
    tracks_out : pandas.DataFrame
        Dataframe containing t, x, y, z positions of center of gravity
        and total cloud mass for untracked part of domain.
    '''

    from pandas import DataFrame
    from tobac.utils import mask_cube_untracked
    from iris import Constraint
    
    logging.info('start calculating centre of gravity for untracked parts of the domain')
    tracks_out=DataFrame()
    time_coord=mass.coord('time')
    tracks_out['frame']=range(len(time_coord.points))
    for i_row,row in tracks_out.iterrows():
        time_i=time_coord.units.num2date(time_coord[int(row['frame'])].points[0])
        constraint_time=Constraint(time=time_i)
        mass_i=mass.extract(constraint_time)
        mask_i=mask.extract(constraint_time)
        mass_untracked_i=mask_cube_untracked(mass_i,mask_i)
        x_M,y_M,z_M,mass_M=center_of_gravity(mass_untracked_i)
        tracks_out.loc[i_row,'time']=time_i
        tracks_out.loc[i_row,'x_M']=float(x_M)
        tracks_out.loc[i_row,'y_M']=float(y_M)
        tracks_out.loc[i_row,'z_M']=float(z_M)
        tracks_out.loc[i_row,'mass']=float(mass_M)
    
    logging.info('Finished calculating centre of gravity for untracked parts of the domain')
    
    return tracks_out

@xarray_to_iris
def calculate_cog_domain(mass):
    '''Caluclate center of gravity and mass for entire domain.

    Parameters
    ----------
    mass : iris.cube.Cube
        Cube of quantity (need coordinates 'time',
        'geopotential_height','projection_x_coordinate' and
        'projection_y_coordinate').

    Returns
    -------
    tracks_out : pandas.DataFrame
        Dataframe containing t, x, y, z positions of center of gravity
        and total cloud mass.
    '''

    from pandas import DataFrame
    from iris import Constraint
    
    logging.info('start calculating centre of gravity for entire domain')

    time_coord=mass.coord('time')

    tracks_out=DataFrame()
    tracks_out['frame']=range(len(time_coord.points))
    for i_row,row in tracks_out.iterrows():  
        time_i=time_coord.units.num2date(time_coord[int(row['frame'])].points[0])
        constraint_time=Constraint(time=time_i)
        mass_i=mass.extract(constraint_time)
        x_M,y_M,z_M,mass_M=center_of_gravity(mass_i)
        tracks_out.loc[i_row,'time']=time_i
        tracks_out.loc[i_row,'x_M']=float(x_M)
        tracks_out.loc[i_row,'y_M']=float(y_M)
        tracks_out.loc[i_row,'z_M']=float(z_M)
        tracks_out.loc[i_row,'mass']=float(mass_M)
        
    logging.info('Finished calculating centre of gravity for entire domain')

    return tracks_out

@xarray_to_iris
def center_of_gravity(cube_in):
    '''Caluclate center of gravity and sum of quantity.

    Parameters
    ----------
    cube_in : iris.cube.Cube
        Cube (potentially masked) of quantity (need coordinates
        'geopotential_height','projection_x_coordinate' and
        'projection_y_coordinate').

    Returns
    -------
    x : float
        X position of center of gravity.

    y : float
        Y position of center of gravity.

    z : float
        Z position of center of gravity.

    variable_sum : float
        Sum of quantity of over unmasked part of the cube.
    '''

    from iris.analysis import SUM
    import numpy as np
    cube_sum=cube_in.collapsed(['bottom_top','south_north','west_east'],SUM)
    z=cube_in.coord('geopotential_height')    
    x=cube_in.coord('projection_x_coordinate')
    y=cube_in.coord('projection_y_coordinate')
    dimensions_collapse=['model_level_number','x','y']
    for coord in  cube_in.coords():
        if (coord.ndim>1 and (cube_in.coord_dims(dimensions_collapse[0])[0] in cube_in.coord_dims(coord) or cube_in.coord_dims(dimensions_collapse[1])[0] in cube_in.coord_dims(coord) or cube_in.coord_dims(dimensions_collapse[2])[0] in cube_in.coord_dims(coord))):
                    cube_in.remove_coord(coord.name())
    if cube_sum.data > 0:
        x=((cube_in*x).collapsed(['model_level_number','x','y'],SUM)/cube_sum).data
        y=((cube_in*y).collapsed(['model_level_number','x','y'],SUM)/cube_sum).data
        z=((cube_in*z.points).collapsed(['model_level_number','x','y'],SUM)/cube_sum).data
    else:
        x=np.nan
        y=np.nan
        z=np.nan
    variable_sum=cube_sum.data
    return(x,y,z,variable_sum)

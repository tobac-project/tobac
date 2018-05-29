def calculate_cog(Tracks,mass,Mask):
    from .watershedding import mask_cube_particle
    from iris import Constraint
    Tracks_out=Tracks[['time','frame','particle','time_cell']]
    for i_row,row in Tracks_out.iterrows():        
        particle=row['particle']
        constraint_time=Constraint(time=row['time'])
        mass_i=mass.extract(constraint_time)
        Mask_i=Mask.extract(constraint_time)
        mass_masked_i=mask_cube_particle(mass_i,Mask_i,particle)
        x_M,y_M,z_M,mass_M=center_of_gravity(mass_masked_i)
        Tracks_out.loc[i_row,'x_M']=float(x_M)
        Tracks_out.loc[i_row,'y_M']=float(y_M)
        Tracks_out.loc[i_row,'z_M']=float(z_M)
        Tracks_out.loc[i_row,'mass']=float(mass_M)
    return Tracks_out
    
def calculate_cog_untracked(mass,Mask):
    import pandas as pd
    from .watershedding import mask_cube_untracked
    from iris import Constraint
    Tracks_out=pd.DataFrame()
    time_coord=mass.coord('time')
    Tracks_out['frame']=range(len(time_coord.points))
    for i_row,row in Tracks_out.iterrows():
        time_i=time_coord.units.num2date(time_coord[int(row['frame'])].points[0])
        constraint_time=Constraint(time=time_i)
        mass_i=mass.extract(constraint_time)
        Mask_i=Mask.extract(constraint_time)
        mass_untracked_i=mask_cube_untracked(mass_i,Mask_i)
        x_M,y_M,z_M,mass_M=center_of_gravity(mass_untracked_i)
        Tracks_out.loc[i_row,'time']=time_i
        Tracks_out.loc[i_row,'x_M']=float(x_M)
        Tracks_out.loc[i_row,'y_M']=float(y_M)
        Tracks_out.loc[i_row,'z_M']=float(z_M)
        Tracks_out.loc[i_row,'mass']=float(mass_M)
    return Tracks_out

def calculate_cog_domain(mass):
    import pandas as pd
    from iris import Constraint
    time_coord=mass.coord('time')

    Tracks_out=pd.DataFrame()
    Tracks_out['frame']=range(len(time_coord.points))
    for i_row,row in Tracks_out.iterrows():  
        time_i=time_coord.units.num2date(time_coord[int(row['frame'])].points[0])
        constraint_time=Constraint(time=time_i)
        mass_i=mass.extract(constraint_time)
        x_M,y_M,z_M,mass_M=center_of_gravity(mass_i)
        Tracks_out.loc[i_row,'time']=time_i
        Tracks_out.loc[i_row,'x_M']=float(x_M)
        Tracks_out.loc[i_row,'y_M']=float(y_M)
        Tracks_out.loc[i_row,'z_M']=float(z_M)
        Tracks_out.loc[i_row,'mass']=float(mass_M)
    return Tracks_out

def center_of_gravity(mass_in):
    from iris.analysis import SUM
    import numpy as np
    Mass=mass_in.collapsed(['bottom_top','south_north','west_east'],SUM)
    z=mass_in.coord('geopotential_height')    
    x=mass_in.coord('projection_x_coordinate')
    y=mass_in.coord('projection_y_coordinate')
    dimensions_collapse=['model_level_number','x','y']
    for coord in  mass_in.coords():
        if (coord.ndim>1 and (mass_in.coord_dims(dimensions_collapse[0])[0] in mass_in.coord_dims(coord) or mass_in.coord_dims(dimensions_collapse[1])[0] in mass_in.coord_dims(coord) or mass_in.coord_dims(dimensions_collapse[2])[0] in mass_in.coord_dims(coord))):
                    mass_in.remove_coord(coord.name())
    if Mass.data > 0:
        x_M=((mass_in*x).collapsed(['model_level_number','x','y'],SUM)/Mass).data
        y_M=((mass_in*y).collapsed(['model_level_number','x','y'],SUM)/Mass).data
        z_M=((mass_in*z.points).collapsed(['model_level_number','x','y'],SUM)/Mass).data
    else:
        x_M=np.nan
        y_M=np.nan
        z_M=np.nan
    Mass_M=Mass.data
    return(x_M,y_M,z_M,Mass_M)

    


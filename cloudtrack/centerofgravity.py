def calculate_cog(Tracks,mass,Mask):
    import numpy as np
    from .watershedding import mask_cube_particle
    Tracks_out=Tracks[['time','frame','particle']]
    Tracks_out['x_M']=np.nan
    Tracks_out['y_M']=np.nan
    Tracks_out['z_M']=np.nan
    Tracks_out['mass']=np.nan
    for i_row,row in Tracks_out.iterrows():        
        particle=row['particle']
#        print('frame: ',row['frame'],'particle: ',row['particle'])
        mass_i=mask_cube_particle(mass[row['frame']],Mask[row['frame']],particle)
        x_M,y_M,z_M,mass_M=center_of_gravity(mass_i)
        Tracks_out.loc[i_row,'x_M']=float(x_M)
        Tracks_out.loc[i_row,'y_M']=float(y_M)
        Tracks_out.loc[i_row,'z_M']=float(z_M)
        Tracks_out.loc[i_row,'mass']=float(mass_M)
    return Tracks_out
    
def calculate_cog_untracked(mass,Mask):
    import numpy as np
    import pandas as pd
    from .watershedding import mask_cube_untracked
    Tracks_out=pd.DataFrame()
    Tracks_out['frame']=range(len(mass.coord('time').points))
    Tracks_out['x_M']=np.nan
    Tracks_out['y_M']=np.nan
    Tracks_out['z_M']=np.nan
    Tracks_out['mass']=np.nan
    for i_row,row in Tracks_out.iterrows():  
        i_time=int(row['frame'])
        mass_i=mask_cube_untracked(mass[i_time],Mask[i_time])
        x_M,y_M,z_M,mass_M=center_of_gravity(mass_i)
        Tracks_out.loc[i_row,'x_M']=float(x_M)
        Tracks_out.loc[i_row,'y_M']=float(y_M)
        Tracks_out.loc[i_row,'z_M']=float(z_M)
        Tracks_out.loc[i_row,'mass']=float(mass_M)
    return Tracks_out

def calculate_cog_domain(mass):
    import numpy as np
    import pandas as pd
    Tracks_out=pd.DataFrame()
    Tracks_out['frame']=range(len(mass.coord('time').points))
    Tracks_out['x_M']=np.nan
    Tracks_out['y_M']=np.nan
    Tracks_out['z_M']=np.nan
    Tracks_out['mass']=np.nan
    for i_row,row in Tracks_out.iterrows():  
        i_time=int(row['frame'])
        mass_i=mass[i_time]
        x_M,y_M,z_M,mass_M=center_of_gravity(mass_i)
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
    y=mass_in.coord('projection_y_coordinate')
    mass_in.remove_coord('latitude')
    mass_in.remove_coord('longitude')
    mass_in.remove_coord('geopotential_height')
    x=mass_in.coord('projection_x_coordinate')
    if Mass.data > 0:
        x_M=((mass_in*x).collapsed(['bottom_top','south_north','west_east'],SUM)/Mass).data
        y_M=((mass_in*y).collapsed(['bottom_top','south_north','west_east'],SUM)/Mass).data
        z_M=((mass_in*z.points).collapsed(['bottom_top','south_north','west_east'],SUM)/Mass).data
    else:
        x_M=np.nan
        y_M=np.nan
        z_M=np.nan
    Mass_M=Mass.data
    return(x_M,y_M,z_M,Mass_M)

    


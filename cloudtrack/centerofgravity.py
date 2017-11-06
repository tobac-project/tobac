def calculate_cog(Tracks,mass,Mask):
    from watershedding import mask_cube_particle
    Tracks_out=Tracks['time','frame','particle']
    Tracks_out['x_M']=None
    Tracks_out['y_M']=None
    Tracks_out['z_M']=None
    Tracks_out['mass']=None
    for i_row,row in Tracks_out.iterrows():        
        particle=row['particle']
        print(particle)
        mass_i=mask_cube_particle(mass['frame'],Mask,particle)
        x_M,y_M,z_M,mass=center_of_gravity(mass_i)
    Tracks_out[i_row,'x_M']=x_M
    Tracks_out[i_row,'y_M']=y_M
    Tracks_out[i_row,'z_M']=z_M
    Tracks_out[i_row,'mass']=mass
    return Tracks_out
    
def center_of_gravity(mass):
    from iris.analysis import SUM
    Mass=mass.collapsed(['bottom_top','south_north','west_east'],SUM)
    z=mass.coord('geopotential_height')
    x=mass.coord('west_east')
    y=mass.coord('south_north')
    mass.remove_coord('latitude')
    mass.remove_coord('longitude')
    mass.remove_coord('geopotential_height')
    x_M=((mass*x).collapsed(['bottom_top','south_north','west_east'],SUM)/Mass).data
    y_M=((mass*y).collapsed(['bottom_top','south_north','west_east'],SUM)/Mass).data
    z_M=((mass*z.points).collapsed(['bottom_top','south_north','west_east'],SUM)/Mass).data
    Mass=Mass.data
    return(x_M,y_M,z_M,Mass)

    


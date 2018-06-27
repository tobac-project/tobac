import pandas as pd
import numpy as np
import logging
import os

from .watershedding import mask_particle,mask_particle_surface,mask_cube_particle

def cell_statistics(Input_cubes,Track,Mask,dimensions,aggregators,output_path='./',output_name='Profiles',**kwargs):
    from iris.cube import Cube,CubeList
    from iris.coords import AuxCoord
    from iris import Constraint,save
    # If input is single cube, turn into cubelist
    if type(Input_cubes) is Cube:
        Input_cubes=CubeList([Input_cubes])
            
    dimensions=['x','y']
    for particle in np.unique(Track['particle']):
        logging.debug('Start calculating profiles for particle '+str(particle))
        Track_i=Track[Track['particle']==particle]
        def time_condition(cell):
            return Track_i.loc[Track_i.index[0],'time'] <= cell <= Track_i.loc[Track_i.index[-1],'time']
        constraint_time=Constraint(time=time_condition)
        constraint=constraint_time
        Mask_i=Mask.extract(constraint)
        Mask_particle_i=mask_particle(Mask_i,particle,masked=False)
#       Mask_particle_surface_i=mask_particle_surface(Mask_w_i,particle,masked=False,z_coord='model_level_number')
        Input_cubes_i=Input_cubes.extract(constraint)
        minutes=(Track_i['time_cell']/pd.Timedelta(minutes=1)).as_matrix()
        latitude=Track_i['latitude'].as_matrix()
        longitude=Track_i['longitude'].as_matrix()
        minutes_coord=AuxCoord(minutes,long_name='cell_time',units='min')
        latitude_coord=AuxCoord(latitude,long_name='latitude',units='degrees')
        longitude_coord=AuxCoord(longitude,long_name='longitude',units='degrees')

        cubes_profile={}
        for aggregator in aggregators:
            cubes_profile[aggregator.name()]=CubeList()
        
        for cube in Input_cubes_i:
            for coord in  cube.coords():
                if (coord.ndim>1 and (cube.coord_dims(dimensions[0])[0] in cube.coord_dims(coord) or cube.coord_dims(dimensions[1])[0] in cube.coord_dims(coord))):
                    cube.remove_coord(coord.name())

            cube.add_aux_coord(minutes_coord,data_dims=cube.coord_dims('time'))
            cube.add_aux_coord(latitude_coord,data_dims=cube.coord_dims('time'))
            cube.add_aux_coord(longitude_coord,data_dims=cube.coord_dims('time'))
            cube_masked=mask_cube_particle(cube,Mask_particle_i,particle)
            
            
            for aggregator in aggregators:
                    
                cubes_profile[aggregator.name()].append(cube_masked.collapsed(dimensions,aggregator,**kwargs))

        for aggregator_name in cubes_profile.keys():
            os.makedirs(os.path.join(output_path,output_name,aggregator_name),exist_ok=True)
            savefile=os.path.join(output_path,output_name,aggregator_name,output_name+'_'+ aggregator_name+'_'+str(int(particle))+'.nc')
            save(cubes_profile[aggregator_name],savefile)

def cog_cell(particle,Tracks=None,M_total=None,M_liquid=None,
             M_frozen=None,
             Mask=None,
             savedir=None):
    
    
    from iris import Constraint
    logging.debug('Start calculating COG for '+str(particle))
    Track=Tracks[Tracks['particle']==particle]
    constraint_time=Constraint(time=lambda cell: Track.head(1)['time'].as_matrix()[0] <= cell <= Track.tail(1)['time'].as_matrix()[0])
    M_total_i=M_total.extract(constraint_time)
    M_liquid_i=M_liquid.extract(constraint_time)
    M_frozen_i=M_frozen.extract(constraint_time)
    Mask_i=Mask.extract(constraint_time)

    savedir_cell=os.path.join(savedir,'cells',str(int(particle)))
    os.makedirs(savedir_cell,exist_ok=True)
    savefile_COG_total_i=os.path.join(savedir_cell,'COG_total'+'_'+str(int(particle))+'.h5')
    savefile_COG_liquid_i=os.path.join(savedir_cell,'COG_liquid'+'_'+str(int(particle))+'.h5')
    savefile_COG_frozen_i=os.path.join(savedir_cell,'COG_frozen'+'_'+str(int(particle))+'.h5')
    
    Tracks_COG_total_i=calculate_cog(Track,M_total_i,Mask_i)
#   Tracks_COG_total_list.append(Tracks_COG_total_i)
    logging.debug('COG total loaded for ' +str(particle))
    
    Tracks_COG_liquid_i=calculate_cog(Track,M_liquid_i,Mask_i)
#   Tracks_COG_liquid_list.append(Tracks_COG_liquid_i)
    logging.debug('COG liquid loaded for ' +str(particle))
    Tracks_COG_frozen_i=calculate_cog(Track,M_frozen_i,Mask_i)
#   Tracks_COG_frozen_list.append(Tracks_COG_frozen_i)
    logging.debug('COG frozen loaded for ' +str(particle))
    
    Tracks_COG_total_i.to_hdf(savefile_COG_total_i,'table')
    Tracks_COG_liquid_i.to_hdf(savefile_COG_liquid_i,'table')
    Tracks_COG_frozen_i.to_hdf(savefile_COG_frozen_i,'table')
    logging.debug('individual COG calculated and saved to '+ savedir_cell)

import trackpy as tp
import numpy as np
#from skimage.morphology import watershed
#from skimage.segmentation import random_walker
#from scipy.ndimage.measurements import watershed_ift
from scipy.interpolate import interp2d, interp1d
import pandas as pd

#from pathos.multiprocessing import ProcessingPool as Pool
import logging

def fill_gaps(t,order=1,extrapolate=0,frame_max=None,x_max=None,y_max=None):
    from scipy.interpolate import InterpolatedUnivariateSpline
    t_grouped=t.groupby('particle')
    t_list=[]
    if frame_max==None:
        frame_max=t['frame'].max()
    for particle,track in t_grouped:        
        
        frame_in=track['frame'].as_matrix()
        x_in=track['x'].as_matrix()
        y_in=track['y'].as_matrix()
        s_x = InterpolatedUnivariateSpline(frame_in, x_in, k=order)
        s_y = InterpolatedUnivariateSpline(frame_in, y_in, k=order)
        track=track.set_index('frame', drop=False)
        
        index_min=min(track.index)-extrapolate
        index_min=max(index_min,0)
        index_max=max(track.index)+extrapolate
        index_max=min(index_max,frame_max)
        new_index=range(index_min,index_max)
        track=track.reindex(new_index)
        track['frame']=new_index
#        track=track.interpolate(method='linear')
        frame_out=track['frame'].as_matrix()
        x_out=s_x(frame_out)
        y_out=s_y(frame_out)
        track['x']=x_out
        track['y']=y_out
        track['particle']=particle
        t_list.append(track)
    t_out=pd.concat(t_list)
    t_out=t_out.loc[(t_out['x']<x_max) & (t_out['y']<y_max) &(t_out['x']>0) & (t_out['y']>0)]         
    t_out=t_out.reset_index(drop=True)
    return t_out


def add_coordinates(t,variable_cube):     
    ''' Function adding coordinates from the tracking cube to the trajectories: time, longitude&latitude, x&y dimensions
    Input:
    t:             pandas dataframe 
                   trajectories from trackpy
    variable_cube: iris.cube.Cube 
                   Cube containing the dimensions 'time','longitude','latitude','x_projection_coordinate','y_projection_coordinate', usually cube that the tracking is performed on

    '''
    Time=variable_cube.coord('time')
    latitude=variable_cube.coord('latitude').points
    longitude=variable_cube.coord('longitude').points
    if (latitude.ndim==3):
        latitude=latitude[0]
        longitude=longitude[0]

   
#    constraint_x =iris.Constraint(projection_x_coordinate=lambda cell: projection_x_coordinate(np.floor(row['x'])) < cell < projection_x_coordinate(np.floor(row['x'])))
#    constraint_y =iris.Constraint(projection_y_coordinate=lambda cell: projection_y_coordinate(np.floor(row['y'])) < cell < projection_y_coordinate())
#    constraint_xy= constraint_x & constraint_y
    coord_names=[coord.name() for coord in  variable_cube.coords()]
    if ('time' in coord_names):
#        print('time coordinate present in variable_cube, interpolating time') 
        t['time']=None
        t['timestr']=None

        for i, row in t.iterrows():
            t.loc[i,'time']=[Time.units.num2date(Time[row['frame']].points[0])]
            t.loc[i,'timestr']=Time.units.num2date(Time[row['frame']].points[0]).strftime('%Y-%m-%d %H:%M:%S')

    if ('projection_x_coordinate' in coord_names and 'projection_y_coordinate' in coord_names):
#        print('x and y coordinates present in variable_cube, interpolating x and y') 
        t['projection_x_coordinate']=np.nan
        t['projection_y_coordinate']=np.nan
        dim_xcoord=variable_cube.coord_dims('projection_x_coordinate')[0]
        dim_ycoord=variable_cube.coord_dims('projection_y_coordinate')[0]        
        x_vec=np.arange(variable_cube.shape[dim_xcoord])
        y_vec=np.arange(variable_cube.shape[dim_ycoord])
#        print('dim_xcoord: ',dim_xcoord)
#        print('dim_ycoord: ',dim_ycoord)

        if (dim_xcoord==3 and dim_ycoord==2):
#        if (dim_xcoord==2 and dim_ycoord==3):

            f_x=interp1d(x_vec,variable_cube.coord('projection_x_coordinate').points)
            f_y=interp1d(y_vec,variable_cube.coord('projection_y_coordinate').points)
        elif (dim_xcoord==2 and dim_ycoord==3):
#        elif (dim_xcoord==3 and dim_ycoord==2):
            f_x=interp1d(x_vec,variable_cube.coord('projection_y_coordinate').points)
            f_y=interp1d(y_vec,variable_cube.coord('projection_x_coordinate').points)
            
        for i, row in t.iterrows():
#            if (dim_xcoord==2 and dim_ycoord==3):
            if (dim_xcoord==3 and dim_ycoord==2):

    #            print(row['x'])
    #            print(variable_cube.coord('projection_x_coordinate').points)
                f_x(row['x'])
                t.loc[i,'projection_x_coordinate']=float(f_x(row['x']))
                t.loc[i,'projection_y_coordinate']=float(f_y(row['y']))
#            elif (dim_xcoord==3 and dim_ycoord==2):
            elif    (dim_xcoord==2 and dim_ycoord==3):

    #            print(row['x'])
    #            print(variable_cube.coord('projection_y_coordinate').points)
    #            print(f_x(row['x']))
                t.loc[i,'projection_y_coordinate']=float(f_x(row['x']))
                t.loc[i,'projection_x_coordinate']=float(f_y(row['y']))

    if ('latitude' in coord_names and 'longitude' in coord_names):
#        print('latitude and longitude coordinates present in variable_cube, interpolating latitude and longitude') 

        t['latitude']=np.nan
        t['longitude']=np.nan
        x_vec=np.arange(latitude.shape[0])
        y_vec=np.arange(latitude.shape[1])
        f_lat=interp2d(x_vec,y_vec,latitude)
        f_lon=interp2d(x_vec,y_vec,longitude)
        for i, row in t.iterrows():
            t.loc[i,'latitude']=float(f_lat(row['x'],row['y']))
            t.loc[i,'longitude']=float(f_lon(row['x'],row['y']))#    print('dim_xcoord',dim_xcoord)
    #        t.loc[i,'latitude']=0.5*(latitude[np.floor(row['x']),np.floor(row['y'])]
    #                            +latitude[np.ceil(row['x']),np.ceil(row['y'])])
    #        t.loc[i,'longitude']=0.5*(longitude[np.floor(row['x']),np.floor(row['y'])]
    #                            +longitude[np.ceil(row['x']),np.ceil(row['y'])])
    
    return t


def maketrack(field_in,model=None,
              diameter=5000,v_max=10,memory=3,stubs=5,
              min_mass=0, min_signal=0,
              order=1,extrapolate=0
              ):
    """
    Function using watershedding to determine cloud volumes associated with tracked updrafts
    
    Parameters:
    Field_in:     iris.cube.Cube 
                  2D input field tracking is performed on
    model:        string ('WRF' or 'RAMS')
                  flag to determin which model the data is coming from (currently needed for vertical coordinate)
    diameter:     float
                  Assumed diameter of tracked objects (m)
    v_max:        float
                  Assumed maximum speed of tracked objects (m/s)
    memory:       int
                  Number of timesteps for which objects can be missed by the algorithm to still give a constistent track
    stubs:        float
                  Minumum number of timesteps for which objects have to be detected to not be filtered out as spurious
    min_mass:     float
                  Minumum "mass" of tracked object to be reached over its lifetime (not actually a physical mass, integrated maximum vertical velocity in this case so units (m/s*m^2=m^3/s)
    min_signal:   float
                  Minumum signal of tracked object to be reached over its lifetime (related to contrast between objects and background)
    order:        int
                  order if interpolation spline to fill gaps in tracking(from allowing memory to be larger than 0)
    extrapolate   int
                  number of points to extrapolate individual tracks by

    Output
    Tracks:      pandas.DataFrame
                 Tracked updrafts, one row per timestep and updraft, includes dimensions 'time','latitude','longitude','projection_x_variable', 'projection_y_variable' based on w cube. 
                 'x' and 'y' are used for watershedding in next step, not equivalent to actual x and y in the model, rather to the order of data storage in the model output
    """
    
    logger = logging.getLogger('trackpy')
    logger.propagate = False
    logger.setLevel(logging.WARNING)


    dx=np.diff(field_in.coord('projection_x_coordinate').points)[0]
    dy=np.diff(field_in.coord('projection_y_coordinate').points)[0]
    dt=np.diff(field_in.coord('time').points)[0]*24*3600

    dxy=0.5*(dx+dy)

   
    
    data=field_in.data
    #data_plot=TWP.data
        
    frames=data
    
    
    # Settings for the tracking algorithm thresholds etc..)
    diameter_pix=round(int(diameter/dxy)/2)*2+1

    memory=memory
    search_range=int(dt*v_max/dxy)
    
    #Filters:
    min_mass_pix=min_mass#/(dx*dy)
    min_signal_pix=min_signal
    
    

    
    
    
    # Settings for the tracking algorithm from initial Supercell tracking:
#    size=11
#    memory=3
#    search_range=7
#    stubs=6
#    mass=100
#    signal=20
##    
#    diameter_pix=size
#    min_signal_pix=signal
#    min_mass_pix=mass    
    
#    print('diameter_pix: ',diameter_pix)
#    print('memory: ',memory)
#    print('search_range: ',search_range)
#    print('stubs: ',stubs)
#
#    print('min_mass_pix: ',min_mass_pix)
#    print('min_signal_pix: ',min_signal_pix)
#
# Identification of features in the individual frames, i.e. timesteps
    
#    f = tp.batch(frames, diameter, invert=False).reset_index(drop=True)
    f = tp.batch(frames, diameter_pix
#                 minmass=0,
#                 maxsize=None, separation=None,
#                 noise_size=1, smoothing_size=None, threshold=None, 
#                  invert=True
#                 percentile=64, topn=None, preprocess=True, max_iterations=10,
#                 filter_before=None, filter_after=True, characterize=True,
#                 engine='auto', output=None, meta=None
                  )
# Linking of the features in the individual frames to trajectories
    
#    t = tp.link_df(f, search_range, memory=memory)
    t = tp.link_df(f, search_range, memory=memory
#                   neighbor_strategy='KDTree', link_strategy='auto',
#                   predictor=None, adaptive_stop=None, adaptive_step=0.95,
#                   copy_features=False, diagnostics=False, pos_columns=None,
#                   t_column=None, hash_size=None, box_size=None, verify_integrity=True,
#                   retain_index=False
                    )
    # Add 1 to all particle numbers to avoid problems with particle "0" in some later stages of the analysis (watershedding)
    t['particle']=t['particle']+1
                  
# Filter trajectories to exclude short trajectories that are likely to be spurious
    
    t1 = tp.filter_stubs(t,threshold=stubs)
    t1=t1.reset_index(drop=True)

# Filter trajectories based on a minimum mass (not ne been seen as sth physical) and signal within the trajectory

    condition = lambda x: (
            (x['mass'].max() > min_mass_pix) 
            &   
            (x['signal'].max() > min_signal_pix)
            )
    t2 = tp.filter(t1, condition)
    t2=t2.reset_index(drop=True)
    
    # Restrict output and further treatment to relevant columns:
    t2=t2[['x','y','frame','particle']]
    
    #Interpolate to fill the gaps in the trajectories (left from allowing memory in the linking)
    t2=fill_gaps(t2,order=order,extrapolate=extrapolate,frame_max=field_in.shape[0],x_max=field_in.shape[2],y_max=field_in.shape[3])
    
#   # Extrapolate tracks (currently not implemented)
#    t2=extrapolate_tracks(t_2,steps=2)
    
    t_final=t2
    
    t_final_out=add_coordinates(t_final,field_in)

    return t_final_out


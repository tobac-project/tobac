import logging
import numpy as np
import pandas as pd
import math


def njit_if_available(func, **kwargs):
    '''Decorator to wrap a function with numba.njit if available.
    If numba isn't available, it just returns the function. 
    '''
    try:
        from numba import njit
        return njit(func, kwargs)
    except ModuleNotFoundError:
        return func



def linking_trackpy(features,field_in,dt,dxy,
                       v_max=None,d_max=None,d_min=None,subnetwork_size=None,
                       memory=0,stubs=1,time_cell_min=None,              
                       order=1,extrapolate=0, 
                       method_linking='random',
                       adaptive_step=None,adaptive_stop=None,
                       cell_number_start=1,
                       min_h1 = None, max_h1 = None, 
                       min_h2 = None, max_h2 = None,
                       PBC_flag = 'none'
                       ):
    """Function to perform the linking of features in trajectories
    
    Parameters
    ----------
    features:     pandas.DataFrame 
                  Detected features to be linked             
    v_max:        float
                  speed at which features are allowed to move
    dt:           float
                  time resolution of tracked features
    dxy:          float
                  grid spacing of input data
    memory        int
                  number of output timesteps features allowed to vanish for to be still considered tracked
    subnetwork_size int
                    maximim size of subnetwork for linking  
    method_linking:   str('predict' or 'random')
                      flag choosing method used for trajectory linking
    min_h1: int
        Minimum hdim_1 value, required when PBC_flag is 'hdim_1' or 'both'
    max_h1: int
        Maximum hdim_1 value, required when PBC_flag is 'hdim_1' or 'both'
    min_h2: int
        Minimum hdim_2 value, required when PBC_flag is 'hdim_2' or 'both'
    max_h2: int
        Maximum hdim_2 value, required when PBC_flag is 'hdim_2' or 'both'
    PBC_flag : str('none', 'hdim_1', 'hdim_2', 'both')
        Sets whether to use periodic boundaries, and if so in which directions.
        'none' means that we do not have periodic boundaries
        'hdim_1' means that we are periodic along hdim1
        'hdim_2' means that we are periodic along hdim2
        'both' means that we are periodic along both horizontal dimensions

    Returns
    -------
    pandas.dataframe
        Pandas dataframe containing the linked features
    """
    #    from trackpy import link_df
    #    from trackpy import link_df
    import trackpy as tp
    from copy import deepcopy
#    from trackpy import filter_stubs
#    from .utils import add_coordinates

    # make sure that we have min and max for h1 and h2 if we are PBC
    if PBC_flag in ['hdim_1', 'both'] and (min_h1 is None or max_h1 is None):
        raise ValueError("For PBC tracking, must set min and max coordinates.")
    
    if PBC_flag in ['hdim_2', 'both'] and (min_h2 is None or max_h2 is None):
        raise ValueError("For PBC tracking, must set min and max coordinates.")


    # calculate search range based on timestep and grid spacing
    if v_max is not None:
        search_range=int(dt*v_max/dxy)
    
    # calculate search range based on timestep and grid spacing
    if d_max is not None:
        search_range=int(d_max/dxy)
    
    # calculate search range based on timestep and grid spacing
    if d_min is not None:
        search_range=max(search_range,int(d_min/dxy))

    if time_cell_min:
        stubs=np.floor(time_cell_min/dt)+1
    
    
    #logging.debug('stubs: '+ str(stubs))

    #logging.debug('start linking features into trajectories')
    
    
    #If subnetwork size given, set maximum subnet size
    if subnetwork_size is not None:
        tp.linking.Linker.MAX_SUB_NET_SIZE=subnetwork_size
    # deep copy to preserve features field:
    features_linking=deepcopy(features)
    # check if we are 3D or not
    
    if 'vdim' in features_linking:
        is_3D = True
        pos_columns_tp = ['vdim','hdim_1','hdim_2']
    else:
        is_3D = False
        pos_columns_tp = ['hdim_1', 'hdim_2']
    
    # Check if we have PBCs. 
    if PBC_flag in ['hdim_1', 'hdim_2', 'both']:
        # Per the trackpy docs, to specify a custom distance function 
        # which we need for PBCs, neighbor_strategy must be 'BTree'. 
        # I think this shouldn't change results, but it will degrade performance.
        neighbor_strategy = 'BTree'
        dist_func = build_distance_function(min_h1, max_h1, min_h2, max_h2, PBC_flag)

    else:
        neighbor_strategy = 'KDTree'
        dist_func = None
    
    
    if method_linking is 'random':
#     link features into trajectories:
        trajectories_unfiltered = tp.link(features_linking, 
                               search_range=search_range, 
                               memory=memory, 
                               t_column='frame',
                               pos_columns=pos_columns_tp,
                               adaptive_step=adaptive_step,adaptive_stop=adaptive_stop,
                               neighbor_strategy=neighbor_strategy, link_strategy='auto',
                               dist_func = dist_func
                               )
    elif method_linking is 'predict':

        pred = tp.predict.NearestVelocityPredict(span=1)
        trajectories_unfiltered = pred.link_df(features_linking, search_range=search_range, memory=memory,
                                 pos_columns=pos_columns_tp,
                                 t_column='frame',
                                 neighbor_strategy=neighbor_strategy, link_strategy='auto',
                                 adaptive_step=adaptive_step,adaptive_stop=adaptive_stop,
                                 dist_func = dist_func
#                                 copy_features=False, diagnostics=False,
#                                 hash_size=None, box_size=None, verify_integrity=True,
#                                 retain_index=False
                                 )
    else:
        raise ValueError('method_linking unknown')
        
        
        # Filter trajectories to exclude short trajectories that are likely to be spurious
#    trajectories_filtered = filter_stubs(trajectories_unfiltered,threshold=stubs)
#    trajectories_filtered=trajectories_filtered.reset_index(drop=True)

    # Reset particle numbers from the arbitray numbers at the end of the feature detection and linking to consecutive cell numbers
    # keep 'particle' for reference to the feature detection step.
    trajectories_unfiltered['cell']=None
    particle_num_to_cell_num = dict()
    for i_particle,particle in enumerate(pd.Series.unique(trajectories_unfiltered['particle'])):
        cell=int(i_particle+cell_number_start)
        particle_num_to_cell_num[particle] = int(cell)
    remap_particle_to_cell_vec = np.vectorize(remap_particle_to_cell_nv)
    trajectories_unfiltered['cell'] = remap_particle_to_cell_vec(particle_num_to_cell_num, trajectories_unfiltered['particle'])
    trajectories_unfiltered['cell'] = trajectories_unfiltered['cell'].astype(int)
    trajectories_unfiltered.drop(columns=['particle'],inplace=True)

    trajectories_bycell=trajectories_unfiltered.groupby('cell')
    stub_cell_nums = list()
    for cell,trajectories_cell in trajectories_bycell:
        #logging.debug("cell: "+str(cell))
        #logging.debug("feature: "+str(trajectories_cell['feature'].values))
        #logging.debug("trajectories_cell.shape[0]: "+ str(trajectories_cell.shape[0]))
        
        if trajectories_cell.shape[0] < stubs:
            #logging.debug("cell" + str(cell)+ "  is a stub ("+str(trajectories_cell.shape[0])+ "), setting cell number to Nan..")
            stub_cell_nums.append(cell)
    
    trajectories_unfiltered.loc[trajectories_unfiltered['cell'].isin(stub_cell_nums),'cell']=np.nan

    trajectories_filtered=trajectories_unfiltered


    #Interpolate to fill the gaps in the trajectories (left from allowing memory in the linking)
    trajectories_filtered_unfilled=deepcopy(trajectories_filtered)


#    trajectories_filtered_filled=fill_gaps(trajectories_filtered_unfilled,order=order,
#                                extrapolate=extrapolate,frame_max=field_in.shape[0]-1,
#                                hdim_1_max=field_in.shape[1],hdim_2_max=field_in.shape[2])
#     add coorinates from input fields to output trajectories (time,dimensions)
#    logging.debug('start adding coordinates to trajectories')
#    trajectories_filtered_filled=add_coordinates(trajectories_filtered_filled,field_in)
#     add time coordinate relative to cell initiation:
#    logging.debug('start adding cell time to trajectories')
    trajectories_filtered_filled=trajectories_filtered_unfilled
    trajectories_final=add_cell_time(trajectories_filtered_filled)

    # add coordinate to raw features identified:
    #logging.debug('start adding coordinates to detected features')
    #logging.debug('feature linking completed')

    return trajectories_final
  

def fill_gaps(t,order=1,extrapolate=0,frame_max=None,hdim_1_max=None,hdim_2_max=None):
    '''add cell time as time since the initiation of each cell   
    
    Parameters
    ----------
    t:             pandas dataframe 
                   trajectories from trackpy
    order:         int
                    Order of polynomial used to extrapolate trajectory into gaps and beyond start and end point
    extrapolate     int
                    number of timesteps to extrapolate trajectories by
    frame_max:      int
                    size of input data along time axis
    hdim_1_max:     int
                    size of input data along first horizontal axis
    hdim_2_max:     int
                    size of input data along second horizontal axis
    Returns
    -------
    pandas dataframe 
        trajectories from trackpy with with filled gaps and potentially extrapolated
    '''
    from scipy.interpolate import InterpolatedUnivariateSpline
    logging.debug('start filling gaps')
    
    t_list=[]    # empty list to store interpolated DataFrames
    
    # group by cell number and perform process for each cell individually:
    t_grouped=t.groupby('cell')
    for cell,track in t_grouped:        
        
        # Setup interpolator from existing points (of order given as keyword)      
        frame_in=track['frame'].values
        hdim_1_in=track['hdim_1'].values
        hdim_2_in=track['hdim_2'].values
        s_x = InterpolatedUnivariateSpline(frame_in, hdim_1_in, k=order)
        s_y = InterpolatedUnivariateSpline(frame_in, hdim_2_in, k=order)        
        
        # Create new index filling in gaps and possibly extrapolating:
        index_min=min(frame_in)-extrapolate
        index_min=max(index_min,0)
        index_max=max(frame_in)+extrapolate
        index_max=min(index_max,frame_max)
        new_index=range(index_min,index_max+1) # +1 here to include last value
        track=track.reindex(new_index)
        
        # Interpolate to extended index:
        frame_out=new_index
        hdim_1_out=s_x(frame_out)
        hdim_2_out=s_y(frame_out)
        
        # Replace fields in data frame with        
        track['frame']=new_index
        track['hdim_1']=hdim_1_out
        track['hdim_2']=hdim_2_out
        track['cell']=cell   
        
        # Append DataFrame to list of DataFrames
        t_list.append(track)       
    # Concatenate interpolated trajectories into one DataFrame:    
    t_out=pd.concat(t_list)
    # Restrict output trajectories to input data in time and space:
    t_out=t_out.loc[(t_out['hdim_1']<hdim_1_max) & (t_out['hdim_2']<hdim_2_max) &(t_out['hdim_1']>0) & (t_out['hdim_2']>0)]
    t_out=t_out.reset_index(drop=True)
    return t_out

def add_cell_time(t):
    ''' add cell time as time since the initiation of each cell   
    
    Parameters
    ----------
    t:             pandas  DataFrame
                   trajectories with added coordinates
    
    Returns
    -------
    t:             pandas dataframe 
                   trajectories with added cell time
    '''

    #logging.debug('start adding time relative to cell initiation')
    t_grouped=t.groupby('cell')
    
    t['time_cell'] = t['time']-t.groupby('cell')['time'].transform('min')
    t['time_cell']=pd.to_timedelta(t['time_cell'])
    return t

def remap_particle_to_cell_nv(particle_cell_map, input_particle):
    '''Remaps the particles to new cells given an input map and the current particle.
    Helper function that is designed to be vectorized with np.vectorize

    Parameters
    ----------
    particle_cell_map: dict-like
        The dictionary mapping particle number to cell number
    input_particle: key for particle_cell_map
        The particle number to remap
    
    '''
    return particle_cell_map[input_particle]

def build_distance_function(min_h1, max_h1, min_h2, max_h2, PBC_flag):
    '''Function to build a partial ```calc_distance_coords_pbc``` function 
    suitable for use with trackpy

    Parameters
    ----------
    min_h1: int
        Minimum point in hdim_1
    max_h1: int
        Maximum point in hdim_1
    min_h2: int
        Minimum point in hdim_2
    max_h2: int
        Maximum point in hdim_2
    PBC_flag : str('none', 'hdim_1', 'hdim_2', 'both')
        Sets whether to use periodic boundaries, and if so in which directions.
        'none' means that we do not have periodic boundaries
        'hdim_1' means that we are periodic along hdim1
        'hdim_2' means that we are periodic along hdim2
        'both' means that we are periodic along both horizontal dimensions
    
    Returns
    -------
    function object
        A version of calc_distance_coords_pbc suitable to be called by
        just f(coords_1, coords_2)

    '''
    import functools
    return functools.partial(calc_distance_coords_pbc, 
                             min_h1 = min_h1, max_h1 = max_h1, min_h2 = min_h2, 
                             max_h2 = max_h2, PBC_flag = PBC_flag)

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
        Maximum point in hdim_1
    min_h2: int
        Minimum point in hdim_2
    max_h2: int
        Maximum point in hdim_2
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

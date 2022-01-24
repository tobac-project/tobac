'''Provide tracking methods.

The individual features and associated area/volumes identified in
each timestep have to be linked into cloud trajectories to analyse
the time evolution of cloud properties for a better understanding of
the underlying pyhsical processes. [5]_

The implementations are structured in a way that allows for the future
addition of more complex tracking methods recording a more complex
network of relationships between cloud objects at different points in
time. [5]_

References
----------
.. [5] Heikenfeld, M., Marinescu, P. J., Christensen, M., Watson-Parris,
   D., Senf, F., van den Heever, S. C., and Stier, P.: tobac v1.0:
   towards a flexible framework for tracking and analysis of clouds in
   diverse datasets, Geosci. Model Dev. Discuss.,
   https://doi.org/10.5194/gmd-2019-105 , in review, 2019, 9f.
'''

import logging
import numpy as np
import pandas as pd

def linking_trackpy(features,field_in,dt,dxy,
                       v_max=None,d_max=None,d_min=None,subnetwork_size=None,
                       memory=0,stubs=1,time_cell_min=None,              
                       order=1,extrapolate=0, 
                       method_linking='random',
                       adaptive_step=None,adaptive_stop=None,
                       cell_number_start=1
                       ):
    '''Perform Linking of features in trajectories.

    The linking determines which of the features detected in a specific
    timestep is identical to an existing feature in the previous
    timestep. For each existing feature, the movement within a time step
    is extrapolated based on the velocities in a number previous time
    steps. The algorithm then breaks the search process down to a few
    candidate features by restricting the search to a circular search
    region centered around the predicted position of the feature in the
    next time step. For newly initialized trajectories, where no
    velocity from previous timesteps is available, the algorithm resort
    to the average velocity of the nearest tracked objects. v_max and
    d_min are given as physical quantities and then converted into
    pixel-based values used in trackpy. This allows for cloud tracking
    that is controlled by physically-based parameters that are
    independent of the temporal and spatial resolution of the input
    data. The algorithm creates a continuous track for the cloud that
    most directly follows the direction of travel of the preceding or
    following cell path. [5]_

    Parameters
    ----------
    features : xarray.Dataset
        Detected features to be linked.

    field_in : xarray.DataArray
        Input field to perform the watershedding on (2D or 3D for one
        specific point in time).

    dt : float
    	Time resolution of tracked features.

    dxy : float
        Grid spacing of the input data.

    d_max : float, optional
    	Maximum search range
        
        Default is None.

    d_min : float, optional
        Variations in the shape of the regions used to determine the
        positions of the features can lead to quasi-instantaneous shifts
        of the position of the feature by one or two grid cells even for
        a very high temporal resolution of the input data, potentially
        jeopardising the tracking procedure. To prevent this, tobac uses
        an additional minimum radius of the search range. [5]_

        Default is None.

    subnetwork_size : int, optional
        Maximum size of subnetwork for linking. Default is None.

    v_max : float, optional
        Speed at which features are allowed to move. Default is None.

    memory : int, optional
        Number of output timesteps features allowed to vanish for to
        be still considered tracked. Default is 0.
        .. warning :: This parameter should be used with caution, as it
                     can lead to erroneous trajectory linking,
                     espacially for data with low time resolution. [5]_

    stubs : int, optional
        Minimum number of timesteps of a tracked cell to be reported
        Default is 1

    time_cell_min : float, optional
    	Minimum length in time of tracked cell to be reported in minutes
        
        Default is None.

    order : int, optional
    	Order of polynomial used to extrapolate trajectory into gaps and
        ond start and end point. 
        Default is 1.

    extrapolate : int, optional
    	Number or timesteps to extrapolate trajectories. 
        Default is 0.

    method_linking : {'random', 'predict'}, optional
        Flag choosing method used for trajectory linking.
        Default is 'random'.

    adaptive_step : float, optional
        Reduce search range by multiplying it by this factor.
        
    adaptive_stop : float, optional
        If not None, when encountering an oversize subnet, retry by progressively
        reducing search_range until the subnet is solvable. If search_range
        becomes <= adaptive_stop, give up and raise a SubnetOversizeException.
        Default is None
        
    cell_number_start : int, optional
        Cell number for first tracked cell. 
        Default is 1

    Returns
    -------
    trajectories_final : xarray.Dataset
        This enables filtering the resulting trajectories, e.g. to
        reject trajectories that are only partially captured at the
        boundaries of the input field both in space and time. [5]_

    Raises
    ------
    ValueError
        If method_linking is neither 'random' nor 'predict'.
    '''

    #    from trackpy import link_df
    import trackpy as tp
    from copy import deepcopy
#    from trackpy import filter_stubs
#    from tobac.utils import add_coordinates

    # convert to iris/pandas
    field_in=field_in.to_iris()
    features=features.to_dataframe()

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
    
    
    logging.debug('stubs: '+ str(stubs))

    logging.debug('start linking features into trajectories')
    
    
    #If subnetwork size given, set maximum subnet size
    if subnetwork_size is not None:
        tp.linking.Linker.MAX_SUB_NET_SIZE=subnetwork_size
    # deep copy to preserve features field:
    features_linking=deepcopy(features)
    
    if method_linking == 'random':
#     link features into trajectories:
        trajectories_unfiltered = tp.link(features_linking, 
                               search_range=search_range, 
                               memory=memory, 
                               t_column='frame',
                               pos_columns=['hdim_2','hdim_1'],
                               adaptive_step=adaptive_step,adaptive_stop=adaptive_stop,
                               neighbor_strategy='KDTree', link_strategy='auto'
                               )
    elif method_linking == 'predict':

        pred = tp.predict.NearestVelocityPredict(span=1)
        trajectories_unfiltered = pred.link_df(features_linking, search_range=search_range, memory=memory,
                                 pos_columns=['hdim_1','hdim_2'],
                                 t_column='frame',
                                 neighbor_strategy='KDTree', link_strategy='auto',
                                 adaptive_step=adaptive_step,adaptive_stop=adaptive_stop
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

    logging.debug('set cell number')
    # https://stackoverflow.com/questions/51834175/
    #   assigning-incremental-values-based-on-an-unique-value-of-a-column
    trajectories_unfiltered['cell'] = pd.factorize(trajectories_unfiltered['particle'])[0] + int(cell_number_start)

    trajectories_unfiltered.drop(columns=['particle'],inplace=True)

    logging.debug('set cell to nan if count < stubs')
    # https://stackoverflow.com/questions/47813994/
    #   python-pandas-how-to-replace-values-that-have-a-count-smaller-than-x
    trajectories_unfiltered.loc[trajectories_unfiltered.groupby('cell').cell.transform('count').lt(stubs), 'cell'] = np.nan

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

    logging.debug('start adding time relative to cell initiation')
    trajectories_filtered_filled['time_cell'] = trajectories_filtered_filled.groupby('cell').time.apply(lambda x: x - x.iloc[0])

    # add coordinate to raw features identified:
    logging.debug('start adding coordinates to detected features')
    logging.debug('feature linking completed')
    
    trajectories_final = trajectories_filtered_filled.to_xarray()
    return trajectories_final

def fill_gaps(t,order=1,extrapolate=0,frame_max=None,hdim_1_max=None,hdim_2_max=None):
    '''Add cell time as time since the initiation of each cell.

    Parameters
    ----------
    t : pandas.DataFrame
	Trajectories from trackpy.

    order : int, optional
	Order of polynomial used to extrapolate trajectory into
	gaps and beyond start and end point. Default is 1.

    extrapolate : int, optional
	Number or timesteps to extrapolate trajectories. Default is 0.

    frame_max : int, optional
        Size of input data along time axis. Default is None.

    hdim_1_max, hdim2_max : int, optional
        Size of input data along first and second horizontal axis.
        Default is None.

    Returns
    -------
    t : pandas.DataFrame
        Trajectories from trackpy with with filled gaps and potentially
        extrapolated.
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


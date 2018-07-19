import numpy as np
import pandas as pd

import logging
def maketrack(field_in,
              grid_spacing=None,time_spacing=None,
              target='maximum',              
              v_max=10,memory=3,stubs=5,              
              order=1,extrapolate=0,              
              method_detection="threshold",
              position_threshold='center',
              diameter=5000,min_mass=0, min_signal=0,parameters_features=False,
              threshold=1, min_num=0,              
              method_linking="random",            
              cell_number_start=1,              
              subnetwork_size=None              
              ):
    """
    Function using watershedding to determine cloud volumes associated with tracked updrafts
    
    Parameters:
    field_in:     iris.cube.Cube 
                  2D input field tracking is performed on
    grid_spacing: float
                  grid spacing in input data (m)
    time_spacing: float
                  time resolution of input data (s)
    diameter:     float
                  Assumed diameter of tracked objects (m)
    target        string
                  Switch to determine if algorithm looks for maxima or minima in input field (maximum: look for maxima (default), minimum: look for minima)
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
    min_num:      int
                  Minumum number of cells above threshold in the feature to be tracked
    order:        int
                  order if interpolation spline to fill gaps in tracking(from allowing memory to be larger than 0)
    extrapolate   int
                  number of points to extrapolate individual tracks by
    method_detection: str('trackpy' or 'threshold')
                      flag choosing method used for feature detection
    position_threshold: str('extreme', 'weighted_diff', 'weighted_abs' or 'center')
                      flag choosing method used for the position of the tracked feature
    method_linking:   str('predict' or 'random')
                      flag choosing method used for trajectory linking

    Output
    Tracks:      pandas.DataFrame
                 Tracked updrafts, one row per timestep and updraft, includes dimensions 'time','latitude','longitude','projection_x_variable', 'projection_y_variable' based on w cube. 
                 'hdim_1' and 'hdim_2' are used for watershedding step.
    """
    from copy import deepcopy
    from trackpy import filter_stubs,filter
    
    logger = logging.getLogger('trackpy')
    logger.propagate = False
    logger.setLevel(logging.WARNING)
    
    ### Prepare Tracking

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
        
    ### Start Tracking
    # Feature detection:
    if method_detection=="trackpy":
        features=feature_detection_trackpy(field_in,diameter=diameter,dxy=dxy,target=target)
        features_filtered = features.drop(features[features['num'] < min_num].index)

        features_filtered = deepcopy(features)

    elif method_detection == "threshold":
        features=feature_detection_threshold(field_in,threshold=threshold,dxy=dxy,target=target)
        features_filtered = features.drop(features[features['num'] < min_num].index)

    else:
        raise ValueError('method_detection unknown, has to be either trackpy or threshold')

    # Link the features in the individual frames to trajectories:
    trajectories_unfiltered=trajectory_linking(features_filtered,v_max=v_max,dt=dt,dxy=dxy,memory=memory,
                                               subnetwork_size=subnetwork_size,
                                               method_linking=method_linking)

    # Filter trajectories to exclude short trajectories that are likely to be spurious
    trajectories_filtered = filter_stubs(trajectories_unfiltered,threshold=stubs)
    trajectories_filtered=trajectories_filtered.reset_index(drop=True)


    if method_detection=="trackpy":    
        # Filter trajectories based on a minimum mass (not ne been seen as sth physical) and signal within the trajectory
        #Filters:
        min_mass_pix=min_mass/(dxy*dxy)
        min_signal_pix=min_signal

        condition = lambda x: (
                (x['mass'].max() > min_mass_pix) 
                &   
                (x['signal'].max() > min_signal_pix)
                )
        trajectories_filtered = filter(trajectories_filtered, condition)
        trajectories_filtered=trajectories_filtered.reset_index(drop=True)
        
        # Restrict output and further treatment to relevant columns:
        trajectories_filtered['mass']=trajectories_filtered['mass']*(dxy*dxy)
        trajectories_filtered['size']=trajectories_filtered['size']*(dxy)
    
        if not parameters_features:
            trajectories_filtered=trajectories_filtered.drop(['mass','signal','size','ecc'],axis=1)
        
    # Reset particle numbers from the arbitray numbers at the end of the feature detection and linking to consecutive numbers
    # keep 'particle' for reference to the feature detection step.
    trajectories_filtered['particle_old']=trajectories_filtered['particle']
    for i,particle in enumerate(pd.Series.unique(trajectories_filtered['particle_old'])):
        particle_new=int(i+cell_number_start)
        trajectories_filtered.loc[trajectories_filtered['particle_old']==particle,'particle']=particle_new


    #Interpolate to fill the gaps in the trajectories (left from allowing memory in the linking)
    trajectories_filtered_unfilled=deepcopy(trajectories_filtered)
    trajectories_filtered_unfilled=add_coordinates(trajectories_filtered_unfilled,field_in)

    
    trajectories_filtered=fill_gaps(trajectories_filtered,order=order,
                                    extrapolate=extrapolate,frame_max=field_in.shape[0],
                                    hdim_1_max=field_in.shape[1],hdim_2_max=field_in.shape[2])
    # add coorinates from input fields to output trajectories (time,dimensions)
    logging.debug('start adding coordinates to trajectories')
    trajectories_filtered=add_coordinates(trajectories_filtered,field_in)
    # add time coordinate relative to cell initiation:
    logging.debug('start adding cell time to trajectories')
    trajectories_filtered=add_cell_time(trajectories_filtered)

    # add coordinate to raw features identified:
    logging.debug('start adding coordinates to detected features')
    features_identified=add_coordinates(features,field_in)

    logging.debug('Finished tracking')

    return trajectories_filtered, features_identified, trajectories_filtered_unfilled

def feature_detection_trackpy(field_in,diameter,dxy,target='maximum'):
    from trackpy import locate
    diameter_pix=round(int(diameter/dxy)/2)*2+1
    
    logging.debug('start feature detection based on trackpy function')

    # set invert to True when tracking minima, False when tracking maxima
    if target=='maximum':
        invert= False
    elif target =='minimum':
        invert= True
    else:
        raise ValueError('target has to be either maximum or minimum')

    # locate features for each timestep and then combine
    list_features=[]   
    data_time=field_in.slices_over('time')
    for i,data_i in enumerate(data_time):
        f_i=locate(data_i.data, diameter_pix, invert=invert,
                 minmass=0, maxsize=None, separation=None,
#                 noise_size=1, smoothing_size=None, threshold=None, 
#                  invert=True
#                 percentile=64, topn=None, preprocess=True, max_iterations=10,
#                 filter_before=None, filter_after=True, characterize=True,
#                 engine='auto', output=None, meta=None
                  )
        f_i['frame']=int(i)
        list_features.append(f_i)        
    logging.debug('feature detection: merging DataFrames')
    features=pd.concat(list_features)
    features.rename(columns={'y':'hdim_1', 'x':'hdim_2'}, inplace =True)
    logging.debug('feature detection completed')
    features=features.reset_index(drop=True)

    return features

def feature_detection_threshold(field_in,threshold,dxy,target='maximum', position_threshold='center'):
    ''' Function to perform feature detection based on contiguous regions above/below a threshold
    Input:
    field_in:      iris.cube.Cube
                   2D field to perform the tracking on (needs to have coordinate 'time' along one of its dimensions)
    
    threshold:     float
                   threshold value used to select target regions to track
    dxy:           float
                   grid spacing of the input data (m)
    target:        str ('minimum' or 'maximum')
                   flag to determine if tracking is targetting minima or maxima in the data
    position_threshold: str('extreme', 'weighted_diff', 'weighted_abs' or 'center')
                      flag choosing method used for the position of the tracked feature

    Output:
    features:      pandas DataFrame 
                   detected features
    '''
    
    from skimage import filters, measure

    logging.debug('start feature detection based on thresholds')
    logging.debug('target: '+str(target))

    # locate features for each timestep and then combine:
    list_features=[]
        
    # loop over timesteps for feature identification:
    data_time=field_in.slices_over('time')
    for i,data_i in enumerate(data_time):
        time_i=data_i.coord('time').units.num2date(data_i.coord('time').points[0])
        track_data = data_i.data
        # if looking for minima, set values above threshold to 0 and scale by data minimum:
        if target is 'maximum':
            mask=1*(track_data >= threshold)
            blobs=mask

        # if looking for minima, set values above threshold to 0 and scale by data minimum:
        elif target is 'minimum':            
            mask=1*(track_data <= threshold)  # only include values greater than threshold    
            blobs=mask
            
        # detect individual regions, label  and count the number of pixels included:
        blobs_labels = measure.label(blobs, background=0)
        values, count = np.unique(blobs_labels[:,:].ravel(), return_counts=True)

        counts=dict(zip(values, count))
        for j in np.arange(1,len(values)):        
            cur_idx = values[j];
            region=blobs_labels[:,:] == cur_idx
            [a,b] = np.nonzero(region)
          
            
            if position_threshold=='center':
                # get position as geometrical centre of identified region:
                hdim1_index=np.mean(a)
                hdim2_index=np.mean(b)
            
            elif position_threshold=='extreme':
                #get positin as max/min position inside the identified region:
                if target is 'maximum':
                    index=np.argmax(track_data[region])
                    hdim1_index=a[index]
                    hdim2_index=b[index]
                    
                if target is 'minimum':
                    index=np.argmin(track_data[region])
                    hdim1_index=a[index]
                    hdim2_index=b[index]
                    
            elif position_threshold=='weighted_diff':
                # get position as centre of identified region, weighted by difference from the threshold:
                hdim1_index=np.average(a,abs(track_data[region]-threshold))
                hdim2_index=np.average(a,abs(track_data[region]-threshold))
                
            elif position_threshold=='weighted_abs':
                # get position as centre of identified region, weighted by absolute values if the field:
                hdim1_index=np.average(a,abs(track_data[region]))
                hdim2_index=np.average(a,abs(track_data[region]))

            else:
                raise ValueError('position_threshold must be center or extreme')
                
            data_frame={'frame': int(i),'hdim_1': hdim1_index,'hdim_2':hdim2_index,'num':counts[cur_idx]}
            f_i=pd.DataFrame(data=data_frame,index=[i])
            list_features.append(f_i)
        logging.debug('Finished feature detection for '+time_i.strftime('%Y-%m-%d_%H:%M:%S'))

            
    logging.debug('feature detection: merging DataFrames')
    # concatenate features from different timesteps into one pandas DataFrame, if not features are detected raise error
    if not list_features:
        raise ValueError('No features detected')
    features=pd.concat(list_features)    
    features=features.reset_index(drop=True)
    
    logging.debug('feature detection completed')
    return features

def trajectory_linking(features,v_max,dt,dxy,memory,subnetwork_size=None,method_linking='random'):
#    from trackpy import link_df
    import trackpy as tp
    from copy import deepcopy
    """
    Function to perform the linking of features in trajectories
    
    Parameters:
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
    method_detection: str('trackpy' or 'threshold')
                      flag choosing method used for feature detection
    method_linking:   str('predict' or 'random')
                      flag choosing method used for trajectory linking
    """
    # calculate search range based on timestep and grid spacing
    search_range=int(dt*v_max/dxy)
    
    logging.debug('start linking features into trajectories')
    
    
    #If subnetwork size given, set maximum subnet size
    if subnetwork_size is not None:
        tp.linking.Linker.MAX_SUB_NET_SIZE=subnetwork_size
    # deep copy to preserve features field:
    features_linking=deepcopy(features)
    
    
    if method_linking is 'random':
#     link features into trajectories:
        trajectories = tp.link(features_linking, 
                               search_range=search_range, 
                               memory=memory, 
                              t_column='frame',
                              pos_columns=['hdim_2','hdim_1'])
    elif method_linking is 'predict':

        pred = tp.predict.NearestVelocityPredict(span=1)
        trajectories = pred.link_df(features_linking, search_range=search_range, memory=memory,
                                 pos_columns=['hdim_1','hdim_2'],
                                 t_column='frame',
                                 neighbor_strategy='KDTree', link_strategy='auto',
                                 adaptive_stop=None, adaptive_step=0.95,
#                                 copy_features=False, diagnostics=False,
#                                 hash_size=None, box_size=None, verify_integrity=True,
#                                 retain_index=False
                                 )
    else:
        raise ValueError('method_linking unknown')
    logging.debug('feature linking completed')

    return trajectories



def fill_gaps(t,order=1,extrapolate=0,frame_max=None,hdim_1_max=None,hdim_2_max=None):
    ''' add cell time as time since the initiation of each cell   
    Input:
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
    Output:
    t:             pandas dataframe 
                   trajectories from trackpy with with filled gaps and potentially extrapolated
    '''
    from scipy.interpolate import InterpolatedUnivariateSpline
    logging.debug('start filling gaps')
    
    t_list=[]    # empty list to store interpolated DataFrames
    
    # group by cell number and perform process for each cell individually:
    t_grouped=t.groupby('particle')
    for particle,track in t_grouped:        
        
        # Setup interpolator from existing points (of order given as keyword)      
        frame_in=track['frame'].values()
        hdim_1_in=track['hdim_1'].values()
        hdim_2_in=track['hdim_2'].values()
        s_x = InterpolatedUnivariateSpline(frame_in, hdim_1_in, k=order)
        s_y = InterpolatedUnivariateSpline(frame_in, hdim_2_in, k=order)        
        
        # Create new index filling in gaps and possibly extrapolating:
        index_min=min(frame_in)-extrapolate
        index_min=max(index_min,0)
        index_max=max(frame_in)+extrapolate
        index_max=min(index_max,frame_max)
        new_index=range(index_min,index_max)
        track=track.reindex(new_index)
        
        # Interpolate to extended index:
        frame_out=new_index
        hdim_1_out=s_x(frame_out)
        hdim_2_out=s_y(frame_out)
        
        # Replace fields in data frame with        
        track['frame']=new_index
        track['hdim_1']=hdim_1_out
        track['hdim_2']=hdim_2_out
        track['particle']=particle   
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
    Input:
    t:             pandas  DataFrame
                   trajectories with added coordinates
    Output:
    t:             pandas dataframe 
                   trajectories with added cell time
    '''

    logging.debug('start adding time relative to cell initiation')
    t_grouped=t.groupby('particle')
    t['time_cell']=np.nan
    for particle,track in t_grouped:
        track_0=track.head(n=1)
        for i,row in track.iterrows():
            t.loc[i,'time_cell']=row['time']-track_0.loc[track_0.index[0],'time']
    return t

def add_coordinates(t,variable_cube):     
    ''' Function adding coordinates from the tracking cube to the trajectories: time, longitude&latitude, x&y dimensions
    Input:
    t:             pandas DataFrame
                   trajectories/features
    variable_cube: iris.cube.Cube 
                   Cube containing the dimensions 'time','longitude','latitude','x_projection_coordinate','y_projection_coordinate', usually cube that the tracking is performed on
    Output:
    t:             pandas DataFrame 
                   trajectories with added coordinated
    '''
    from scipy.interpolate import interp2d, interp1d

    logging.debug('start adding coordinates from cube')

    # pull time as datetime object and timestr from input data and add it to DataFrame:    
    t['time']=None
    t['timestr']=None
    
    
    logging.debug('adding time coordinate')

    time_in=variable_cube.coord('time')
    time_in_datetime=time_in.units.num2date(time_in.points)
    
    for i, row in t.iterrows():
#        logging.debug('adding time coordinate for row '+str(i))
        t.loc[i,'time']=time_in_datetime[int(row['frame'])]
        t.loc[i,'timestr']=time_in_datetime[int(row['frame'])].strftime('%Y-%m-%d %H:%M:%S')


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
                t[coord]=np.nan            
                f=interp1d(dimvec_1,variable_cube.coord(coord).points,fill_value="extrapolate")
                for i, row in t.iterrows():
                    t.loc[i,coord]=float(f(row['hdim_1']))

            if variable_cube.coord_dims(coord)==(hdim_2,):
                t[coord]=np.nan            
                f=interp1d(dimvec_2,variable_cube.coord(coord).points,fill_value="extrapolate")
                for i, row in t.iterrows():
                    t.loc[i,coord]=float(f(row['hdim_2']))

        # interpolate 2D coordinates:
        elif variable_cube.coord(coord).ndim==2:

            t[coord]=np.nan            
            if variable_cube.coord_dims(coord)==(hdim_1,hdim_2):
                f=interp2d(dimvec_2,dimvec_1,variable_cube.coord(coord).points)
                for i, row in t.iterrows():
                    t.loc[i,coord]=float(f(row['hdim_2'],row['hdim_1']))
            if variable_cube.coord_dims(coord)==(hdim_2,hdim_1):
                f=interp2d(dimvec_1,dimvec_2,variable_cube.coord(coord).points)
                for i, row in t.iterrows():
                    t.loc[i,coord]=float(f(row['hdim_1'],row['hdim_2']))
        
        # interpolate 3D coordinates:            
        # mainly workaround for wrf latitude and longitude (to be fixed in future)
        
        elif variable_cube.coord(coord).ndim==3:

            t[coord]=np.nan
            if variable_cube.coord_dims(coord)==(ndim_time,hdim_1,hdim_2):
                f=interp2d(dimvec_2,dimvec_1,variable_cube[0,:,:].coord(coord).points)
                for i, row in t.iterrows():
                    t.loc[i,coord]=float(f(row['hdim_2'],row['hdim_1']))
            
            if variable_cube.coord_dims(coord)==(ndim_time,hdim_2,hdim_1):
                f=interp2d(dimvec_1,dimvec_2,variable_cube[0,:,:].coord(coord).points)
                for i, row in t.iterrows():
                    t.loc[i,coord]=float(f(row['hdim_1'],row['hdim_2']))
        
            if variable_cube.coord_dims(coord)==(hdim_1,ndim_time,hdim_2):
                f=interp2d(dimvec_2,dimvec_1,variable_cube[:,0,:].coord(coord).points)
                for i, row in t.iterrows():
                    t.loc[i,coord]=float(f(row['hdim_2'],row['hdim_1']))
                    
            if variable_cube.coord_dims(coord)==(hdim_1,hdim_2,ndim_time):
                f=interp2d(dimvec_2,dimvec_1,variable_cube[:,:,0].coord(coord).points)
                for i, row in t.iterrows():
                    t.loc[i,coord]=float(f(row['hdim_2'],row['hdim_1']))
                    
                    
            if variable_cube.coord_dims(coord)==(hdim_2,ndim_time,hdim_1):
                f=interp2d(dimvec_1,dimvec_2,variable_cube[:,0,:].coord(coord).points)
                for i, row in t.iterrows():
                    t.loc[i,coord]=float(f(row['hdim_1'],row['hdim_2']))
                    
            if variable_cube.coord_dims(coord)==(hdim_2,hdim_1,ndim_time):
                f=interp2d(dimvec_1,dimvec_2,variable_cube[:,:,0].coord(coord).points)
                for i, row in t.iterrows():
                    t.loc[i,coord]=float(f(row['hdim_1'],row['hdim_2']))
        logging.debug('added coord: '+ coord)

    return t


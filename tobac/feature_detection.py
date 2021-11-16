import logging
import numpy as np
import pandas as pd

def feature_position(hdim1_indices,hdim2_indeces,region,track_data,threshold_i,position_threshold, target):
    '''Function to  determine feature position
    
    Parameters
    ----------
        hdim1_indices : list
            list of indices along hdim1 (typically ```y```)
        
        hdim2_indeces : list
            List of indices of feature along hdim2 (typically ```x```)
        
        region : list
            List of 2-element tuples
        track_data : array-like
            2D array containing the data
        
        threshold_i : float
            TODO: ??
        
        position_threshold : str
            TODO: ??
        
        target : str
            TODO: ??

    Returns
    -------
        hdim1_index : float
            feature position along 1st horizontal dimension
        hdim2_index : float
            feature position along 2nd horizontal dimension
    '''
    if position_threshold=='center':
        # get position as geometrical centre of identified region:
        hdim1_index=np.mean(hdim1_indices)
        hdim2_index=np.mean(hdim2_indeces)

    elif position_threshold=='extreme':
        #get position as max/min position inside the identified region:
        if target == 'maximum':
            index=np.argmax(track_data[region])
            hdim1_index=hdim1_indices[index]
            hdim2_index=hdim2_indeces[index]

        if target == 'minimum':
            index=np.argmin(track_data[region])
            hdim1_index=hdim1_indices[index]
            hdim2_index=hdim2_indeces[index]

    elif position_threshold=='weighted_diff':
        # get position as centre of identified region, weighted by difference from the threshold:
        weights=abs(track_data[region]-threshold_i)
        if sum(weights)==0:
            weights=None
        hdim1_index=np.average(hdim1_indices,weights=weights)
        hdim2_index=np.average(hdim2_indeces,weights=weights)

    elif position_threshold=='weighted_abs':
        # get position as centre of identified region, weighted by absolute values if the field:
        weights=abs(track_data[region])
        if sum(weights)==0:
            weights=None
        hdim1_index=np.average(hdim1_indices,weights=weights)
        hdim2_index=np.average(hdim2_indeces,weights=weights)
    else:
        raise ValueError('position_threshold must be center,extreme,weighted_diff or weighted_abs')
    return hdim1_index,hdim2_index

def test_overlap(region_inner,region_outer):
    '''function to test for overlap between two regions (TODO: probably scope for further speedup here)

    Parameters
    ----------
    region_1 : list
        list of 2-element tuples defining the indeces of all cell in the region
    region_2 : list
        list of 2-element tuples defining the indeces of all cell in the region

    Returns
    -------
    bool
        True if there are any shared points between the two regions
    '''
    overlap=frozenset(region_outer).isdisjoint(region_inner)
    return not overlap

def remove_parents(features_thresholds,regions_i,regions_old):
    '''function to remove features whose regions surround newly detected feature regions

    Parameters
    ----------
    features_thresholds : pandas.DataFrame
        Dataframe containing detected features
    regions_i : dict
        dictionary containing the regions above/below threshold for the newly detected feature (feature ids as keys)
    regions_old : dict
        dictionary containing the regions above/below threshold from previous threshold (feature ids as keys)

    Returns
    -------
    pandas.DataFrame
        Dataframe containing detected features excluding those that are superseded by newly detected ones
    '''
    list_remove=[]
    for idx_i,region_i in regions_i.items():    
        for idx_old,region_old in regions_old.items():
            if test_overlap(regions_old[idx_old],regions_i[idx_i]):
                list_remove.append(idx_old)
    list_remove=list(set(list_remove))    
    # remove parent regions:
    if features_thresholds is not None:
        features_thresholds=features_thresholds[~features_thresholds['idx'].isin(list_remove)]

    return features_thresholds

def feature_detection_threshold(data_i,i_time,
                                threshold=None,
                                min_num=0,
                                target='maximum',
                                position_threshold='center',
                                sigma_threshold=0.5,
                                n_erosion_threshold=0,
                                n_min_threshold=0,
                                min_distance=0,
                                idx_start=0):
    '''function to find features based on individual threshold value

    Parameters
    ----------
    data_i : iris.cube.Cube
        2D field to perform the feature detection (single timestep)
    i_time : int
        number of the current timestep
    threshold : float
        threshold value used to select target regions to track
    target : str ('minimum' or 'maximum')
        flag to determine if tracking is targetting minima or maxima in the data
    position_threshold : str('extreme', 'weighted_diff', 'weighted_abs' or 'center')
        flag choosing method used for the position of the tracked feature
    sigma_threshold : float
        standard deviation for intial filtering step
    n_erosion_threshold : int
        number of pixel by which to erode the identified features
    n_min_threshold : int
        minimum number of identified features
    min_distance : float
        minimum distance between detected features (m)
    idx_start : int
        feature id to start with
    Returns
    -------
    pandas DataFrame 
        detected features for individual threshold
    dict
        dictionary containing the regions above/below threshold used for each feature (feature ids as keys)
    '''
    from skimage.measure import label
    from skimage.morphology import binary_erosion

    # if looking for minima, set values above threshold to 0 and scale by data minimum:
    if target == 'maximum':
        mask=1*(data_i >= threshold)
        # if looking for minima, set values above threshold to 0 and scale by data minimum:
    elif target == 'minimum': 
        mask=1*(data_i <= threshold)  
    # only include values greater than threshold
    # erode selected regions by n pixels 
    if n_erosion_threshold>0:
        selem=np.ones((n_erosion_threshold,n_erosion_threshold))
        mask=binary_erosion(mask,selem).astype(np.int64)
        # detect individual regions, label  and count the number of pixels included:
    labels = label(mask, background=0)
    values, count = np.unique(labels[:,:].ravel(), return_counts=True)
    values_counts=dict(zip(values, count))
    # Filter out regions that have less pixels than n_min_threshold
    values_counts={k:v for k, v in values_counts.items() if v>n_min_threshold}
    #check if not entire domain filled as one feature
    if 0 in values_counts:       
    #Remove background counts:
        values_counts.pop(0)       
        #create empty list to store individual features for this threshold
        list_features_threshold=[]
        #create empty dict to store regions for individual features for this threshold
        regions=dict()
        #create emptry list of features to remove from parent threshold value
        #loop over individual regions:       
        for cur_idx,count in values_counts.items():
            region=labels[:,:] == cur_idx
            [hdim1_indices,hdim2_indeces]= np.nonzero(region)
            #write region for individual threshold and feature to dict
            region_i=list(zip(hdim1_indices,hdim2_indeces))
            regions[cur_idx+idx_start]=region_i
            # Determine feature position for region by one of the following methods:
            hdim1_index,hdim2_index=feature_position(hdim1_indices,hdim2_indeces,region,data_i,threshold,position_threshold,target)
            #create individual DataFrame row in tracky format for identified feature
            list_features_threshold.append({'frame': int(i_time),
                                            'idx':cur_idx+idx_start,
                                            'hdim_1': hdim1_index,
                                            'hdim_2':hdim2_index,
                                            'num':count,
                                            'threshold_value':threshold})
        features_threshold=pd.DataFrame(list_features_threshold)
    else:
        features_threshold=pd.DataFrame()
        regions=dict()
            
    return features_threshold, regions
    
def feature_detection_multithreshold_timestep(data_i,i_time,
                                              threshold=None,
                                              min_num=0,
                                              target='maximum',
                                              position_threshold='center',
                                              sigma_threshold=0.5,
                                              n_erosion_threshold=0,
                                              n_min_threshold=0,
                                              min_distance=0,
                                              feature_number_start=1
                                              ):
    '''function to find features in each timestep based on iteratively finding regions above/below a set of thresholds
    
    Parameters
    ----------
    data_i : iris.cube.Cube
        2D field to perform the feature detection (single timestep)
    i_time : int
        number of the current timestep 
    threshold : list of floats
        threshold values used to select target regions to track
    dxy : float
        grid spacing of the input data (m)
    target : str ('minimum' or 'maximum')
        flag to determine if tracking is targetting minima or maxima in the data
    position_threshold : str('extreme', 'weighted_diff', 'weighted_abs' or 'center')
        flag choosing method used for the position of the tracked feature
    sigma_threshold : float
        standard deviation for intial filtering step
    n_erosion_threshold : int
        number of pixel by which to erode the identified features
    n_min_threshold : int
        minimum number of identified features
    min_distance : float
        minimum distance between detected features (m)
    feature_number_start : int
        feature number to start with
    Returns
    -------
    pandas DataFrame 
        detected features for individual timestep
    '''
    from scipy.ndimage.filters import gaussian_filter

    track_data = data_i.core_data()

    track_data=gaussian_filter(track_data, sigma=sigma_threshold) #smooth data slightly to create rounded, continuous field
    # create empty lists to store regions and features for individual timestep
    features_thresholds=pd.DataFrame()
    for i_threshold,threshold_i in enumerate(threshold):
        if (i_threshold>0 and not features_thresholds.empty):
            idx_start=features_thresholds['idx'].max()+1
        else:
            idx_start=0
        features_threshold_i,regions_i=feature_detection_threshold(track_data,i_time,
                                                        threshold=threshold_i,
                                                        sigma_threshold=sigma_threshold,
                                                        min_num=min_num,
                                                        target=target,
                                                        position_threshold=position_threshold,
                                                        n_erosion_threshold=n_erosion_threshold,
                                                        n_min_threshold=n_min_threshold,
                                                        min_distance=min_distance,
                                                        idx_start=idx_start
                                                        )
        if any([x is not None for x in features_threshold_i]):
            features_thresholds=features_thresholds.append(features_threshold_i)

        # For multiple threshold, and features found both in the current and previous step, remove "parent" features from Dataframe
        if (i_threshold>0 and not features_thresholds.empty and regions_old):
            # for each threshold value: check if newly found features are surrounded by feature based on less restrictive threshold
            features_thresholds=remove_parents(features_thresholds,regions_i,regions_old)
        regions_old=regions_i

        logging.debug('Finished feature detection for threshold '+str(i_threshold) + ' : ' + str(threshold_i) )
    return features_thresholds

def feature_detection_multithreshold(field_in,
                                     dxy,
                                     threshold=None,
                                     min_num=0,
                                     target='maximum',
                                     position_threshold='center',
                                     sigma_threshold=0.5,
                                     n_erosion_threshold=0,
                                     n_min_threshold=0,
                                     min_distance=0,
                                     feature_number_start=1
                                     ):
    '''Function to perform feature detection based on contiguous regions above/below a threshold
    
    Parameters
    ----------
    field_in:      iris.cube.Cube
                   2D field to perform the tracking on (needs to have coordinate 'time' along one of its dimensions)
    
    thresholds:    list of floats
                   threshold values used to select target regions to track
    dxy:           float
                   grid spacing of the input data (m)
    target:        str ('minimum' or 'maximum')
                   flag to determine if tracking is targetting minima or maxima in the data
    position_threshold: str('extreme', 'weighted_diff', 'weighted_abs' or 'center')
                      flag choosing method used for the position of the tracked feature
    sigma_threshold: float
                     standard deviation for intial filtering step
    n_erosion_threshold: int
                         number of pixel by which to erode the identified features
    n_min_threshold: int
                     minimum number of identified features
    min_distance:  float
                   minimum distance between detected features (m)
    
    Returns
    -------
    pandas DataFrame 
                   detected features
    '''
    from .utils import add_coordinates

    logging.debug('start feature detection based on thresholds')
    
    # create empty list to store features for all timesteps
    list_features_timesteps=[]

    # loop over timesteps for feature identification:
    data_time=field_in.slices_over('time')
    
    # if single threshold is put in as a single value, turn it into a list
    if type(threshold) in [int,float]:
        threshold=[threshold]    
    
    for i_time,data_i in enumerate(data_time):
        time_i=data_i.coord('time').units.num2date(data_i.coord('time').points[0])
        features_thresholds=feature_detection_multithreshold_timestep(data_i,i_time,
                                                            threshold=threshold,
                                                            sigma_threshold=sigma_threshold,
                                                            min_num=min_num,
                                                            target=target,
                                                            position_threshold=position_threshold,
                                                            n_erosion_threshold=n_erosion_threshold,
                                                            n_min_threshold=n_min_threshold,
                                                            min_distance=min_distance,
                                                            feature_number_start=feature_number_start
                                                           )
        #check if list of features is not empty, then merge features from different threshold values 
        #into one DataFrame and append to list for individual timesteps:
        if not features_thresholds.empty:
            #Loop over DataFrame to remove features that are closer than distance_min to each other:
            if (min_distance > 0):
                features_thresholds=filter_min_distance(features_thresholds,dxy,min_distance)
        list_features_timesteps.append(features_thresholds)
        
        logging.debug('Finished feature detection for ' + time_i.strftime('%Y-%m-%d_%H:%M:%S'))

    logging.debug('feature detection: merging DataFrames')
    # Check if features are detected and then concatenate features from different timesteps into one pandas DataFrame
    # If no features are detected raise error
    if any([not x.empty for x in list_features_timesteps]):
        features=pd.concat(list_features_timesteps, ignore_index=True)   
        features['feature']=features.index+feature_number_start
    #    features_filtered = features.drop(features[features['num'] < min_num].index)
    #    features_filtered.drop(columns=['idx','num','threshold_value'],inplace=True)
        features=add_coordinates(features,field_in)
    else:
        features=None
        logging.info('No features detected')
    logging.debug('feature detection completed')
    return features

def filter_min_distance(features,dxy,min_distance):
    '''Function to perform feature detection based on contiguous regions above/below a threshold
    
    Parameters
    ----------
    features:      pandas DataFrame 
                   features
    dxy:           float
                   horzontal grid spacing (m)
    min_distance:  float
                   minimum distance between detected features (m)
    
    Returns
    -------
    pandas DataFrame 
                   features
    '''
    from itertools import combinations
    remove_list_distance=[]
    #create list of tuples with all combinations of features at the timestep:
    indeces=combinations(features.index.values,2)
    #Loop over combinations to remove features that are closer together than min_distance and keep larger one (either higher threshold or larger area)
    for index_1,index_2 in indeces:
        if index_1 is not index_2:
            features.loc[index_1,'hdim_1']
            distance=dxy*np.sqrt((features.loc[index_1,'hdim_1']-features.loc[index_2,'hdim_1'])**2+(features.loc[index_1,'hdim_2']-features.loc[index_2,'hdim_2'])**2)
            if distance <= min_distance:
#                        logging.debug('distance<= min_distance: ' + str(distance))
                if features.loc[index_1,'threshold_value']>features.loc[index_2,'threshold_value']:
                    remove_list_distance.append(index_2)
                elif features.loc[index_1,'threshold_value']<features.loc[index_2,'threshold_value']:
                    remove_list_distance.append(index_1)
                elif features.loc[index_1,'threshold_value']==features.loc[index_2,'threshold_value']:
                    if features.loc[index_1,'num']>features.loc[index_2,'num']:
                        remove_list_distance.append(index_2)
                    elif features.loc[index_1,'num']<features.loc[index_2,'num']:
                        remove_list_distance.append(index_1)
                    elif features.loc[index_1,'num']==features.loc[index_2,'num']:
                        remove_list_distance.append(index_2)
    features=features[~features.index.isin(remove_list_distance)]
    return features

#--------------------------------------
#new functions for 3D feature detection
#--------------------------------------

def feature_detection_multithreshold_3D(field_in,
                                     dxy,
                                     dz,
                                     threshold=None,
                                     min_num=0,
                                     target='maximum',
                                     position_threshold='center',
                                     sigma_threshold=0.5,
                                     n_erosion_threshold=0,
                                     n_min_threshold=0,
                                     min_distance=0,
                                     feature_number_start=1
                                     ):
    ''' Function to perform feature detection based on contiguous regions above/below a threshold
    Input:
    field_in:      iris.cube.Cube
                   3D field to perform the tracking on (needs to have coordinate 'time' along one of its dimensions)
    
    thresholds:    list of floats
                   threshold values used to select target regions to track
    dxy:           float
                   grid spacing of the input data (m)
    dz:            float
                   array, grid spacing of input data in vertical (m)
    target:        str ('minimum' or 'maximum')
                   flag to determine if tracking is targetting minima or maxima in the data
    position_threshold: str('extreme', 'weighted_diff', 'weighted_abs' or 'center')
                      flag choosing method used for the position of the tracked feature
    sigma_threshold: float
                     standard deviation for intial filtering step
    n_erosion_threshold: int
                         number of pixel by which to erode the identified features
    n_min_threshold: int
                     minimum number of identified features
    min_distance:  float
                   minimum distance between detected features (m)
    Output:
    features:      pandas DataFrame 
                   detected features
    '''
    #from .utils import add_coordinates
    #"add_coordinates_3D"  defined and used here instead

    logging.debug('start feature detection based on thresholds')
    
    # create empty list to store features for all timesteps
    list_features_timesteps=[]

    # loop over timesteps for feature identification:
    data_time=field_in.slices_over('time')
    
    # if single threshold is put in as a single value, turn it into a list
    if type(threshold) in [int,float]:
        threshold=[threshold]    
    
    for i_time,data_i in enumerate(data_time):
        print("feature detection multithreshold_3D loop, i_time: ",i_time)
        time_i=data_i.coord('time').units.num2date(data_i.coord('time').points[0])
        print("Calling feature_detection_multithreshold_timestep_3D")
        features_thresholds=feature_detection_multithreshold_timestep_3D(data_i,i_time,
                                                            threshold=threshold,
                                                            sigma_threshold=sigma_threshold,
                                                            min_num=min_num,
                                                            target=target,
                                                            position_threshold=position_threshold,
                                                            n_erosion_threshold=n_erosion_threshold,
                                                            n_min_threshold=n_min_threshold,
                                                            min_distance=min_distance,
                                                            feature_number_start=feature_number_start
                                                           )
        #check if list of features is not empty, then merge features from different threshold values 
        #into one DataFrame and append to list for individual timesteps:
        if not features_thresholds.empty:
            #Loop over DataFrame to remove features that are closer than distance_min to each other:
            if (min_distance > 0):
                features_thresholds=filter_min_distance_3D(features_thresholds,dxy,dz,min_distance)
        list_features_timesteps.append(features_thresholds)

        print(features_thresholds)
        
        logging.debug('Finished feature detection for ' + time_i.strftime('%Y-%m-%d_%H:%M:%S'))
        print('Finished feature detection for ' + time_i.strftime('%Y-%m-%d_%H:%M:%S'))

    logging.debug('feature detection: merging DataFrames')
    # Check if features are detected and then concatenate features from different timesteps into one pandas DataFrame
    # If no features are detected raise error
    if any([not x.empty for x in list_features_timesteps]):
        features=pd.concat(list_features_timesteps, ignore_index=True)   
        features['feature']=features.index+feature_number_start
    #    features_filtered = features.drop(features[features['num'] < min_num].index)
    #    features_filtered.drop(columns=['idx','num','threshold_value'],inplace=True)
        features=add_coordinates_3D(features,field_in)
    else:
        features=None
        logging.info('No features detected')
    logging.debug('feature detection completed')
    print('feature detection completed')
    return features

def feature_detection_multithreshold_timestep_3D(data_i,i_time,
                                              threshold=None,
                                              min_num=0,
                                              target='maximum',
                                              position_threshold='center',
                                              sigma_threshold=0.5,
                                              n_erosion_threshold=0,
                                              n_min_threshold=0,
                                              min_distance=0,
                                              feature_number_start=1
                                              ):
    '''
    function to find features in each timestep based on iteratively finding regions above/below a set of thresholds
    Input:
    data_i:      iris.cube.Cube
                 3D field to perform the feature detection (single timestep)
    i_time:      int
                 number of the current timestep 
    
    threshold:    list of floats
                  threshold values used to select target regions to track
    dxy:          float
                  grid spacing of the input data (m)
    target:       str ('minimum' or 'maximum')
                  flag to determine if tracking is targetting minima or maxima in the data
    position_threshold: str('extreme', 'weighted_diff', 'weighted_abs' or 'center')
                      flag choosing method used for the position of the tracked feature
    sigma_threshold: float
                     standard deviation for intial filtering step
    n_erosion_threshold: int
                         number of pixel by which to erode the identified features
    n_min_threshold: int
                     minimum number of identified features
    min_distance:  float
                   minimum distance between detected features (m)
    feature_number_start: int
                          feature number to start with
    Output:
    features_threshold:      pandas DataFrame 
                             detected features for individual timestep
    '''
    from scipy.ndimage.filters import gaussian_filter

    track_data = data_i.core_data()

    track_data=gaussian_filter(track_data, sigma=sigma_threshold) #smooth data slightly to create rounded, continuous field
    # create empty lists to store regions and features for individual timestep
    features_thresholds=pd.DataFrame()
    for i_threshold,threshold_i in enumerate(threshold):
        if (i_threshold>0 and not features_thresholds.empty):
            idx_start=features_thresholds['idx'].max()+1
        else:
            idx_start=0
        features_threshold_i,regions_i=feature_detection_threshold_3D(track_data,i_time,
                                                        threshold=threshold_i,
                                                        sigma_threshold=sigma_threshold,
                                                        min_num=min_num,
                                                        target=target,
                                                        position_threshold=position_threshold,
                                                        n_erosion_threshold=n_erosion_threshold,
                                                        n_min_threshold=n_min_threshold,
                                                        min_distance=min_distance,
                                                        idx_start=idx_start
                                                        )
        if any([x is not None for x in features_threshold_i]):
            features_thresholds=features_thresholds.append(features_threshold_i)

        # For multiple threshold, and features found both in the current and previous step, remove "parent" features from Dataframe
        if (i_threshold>0 and not features_thresholds.empty and regions_old):
            # for each threshold value: check if newly found features are surrounded by feature based on less restrictive threshold
            features_thresholds=remove_parents_3D(features_thresholds,regions_i,regions_old)
        regions_old=regions_i

        logging.debug('Finished feature detection for threshold '+str(i_threshold) + ' : ' + str(threshold_i) )
    return features_thresholds


def feature_detection_threshold_3D(data_i,i_time,
                                threshold=None,
                                min_num=0,
                                target='maximum',
                                position_threshold='center',
                                sigma_threshold=0.5,
                                n_erosion_threshold=0,
                                n_min_threshold=0,
                                min_distance=0,
                                idx_start=0):
    '''
    function to find features based on individual threshold value:
    Input:
    data_i:      iris.cube.Cube
                 3D field to perform the feature detection (single timestep)
    i_time:      int
                 number of the current timestep
    threshold:    float
                  threshold value used to select target regions to track
    target:       str ('minimum' or 'maximum')
                  flag to determine if tracking is targetting minima or maxima in the data
    position_threshold: str('extreme', 'weighted_diff', 'weighted_abs' or 'center')
                      flag choosing method used for the position of the tracked feature
    sigma_threshold: float
                     standard deviation for intial filtering step
    n_erosion_threshold: int
                         number of pixel by which to erode the identified features
    n_min_threshold: int
                     minimum number of identified features
    min_distance:  float
                   minimum distance between detected features (m)
    idx_start: int
               feature id to start with
    Output:
    features_threshold:      pandas DataFrame 
                             detected features for individual threshold
    regions:                 dict
                             dictionary containing the regions above/below threshold used for each feature (feature ids as keys)
    '''
    from skimage.measure import label
    from skimage.morphology import binary_erosion
    import operator
    # if looking for minima, set values above threshold to 0 and scale by data minimum:
    if target == 'maximum':
        mask=1*(data_i >= threshold)
        # if looking for minima, set values above threshold to 0 and scale by data minimum:
    elif target == 'minimum': 
        mask=1*(data_i <= threshold)  
    # only include values greater than threshold
    # erode selected regions by n pixels 
    if n_erosion_threshold>0:
        selem=np.ones((n_erosion_threshold,n_erosion_threshold))
        mask=binary_erosion(mask,selem).astype(np.int64)
        # detect individual regions, label  and count the number of pixels included:
    labels, num_labels = label(mask, background=0, return_num=True)
    labels_shape = labels.shape
    #values, count = np.unique(labels[:,:,:].ravel(), return_counts=True)
    #values_counts=dict(zip(values, count))
    
    max_init_size_numba = labels.shape[0]*5
    # Filter out regions that have less pixels than n_min_threshold
    #values_counts={k:v for k, v in values_counts.items() if v>n_min_threshold}
    #check if not entire domain filled as one feature
    #if 0 in values_counts:       
    #Remove background counts:
    #values_counts.pop(0)       
    #create empty list to store individual features for this threshold
    list_features_threshold=[]
    #create empty dict to store regions for individual features for this threshold
    regions=dict()

    vdim_indyces_dict = dict()
    hdim1_indices_dict = dict()
    hdim2_indeces_dict = dict()
    labels = np.array(labels)
    #print(type(values))
    if(num_labels>0):
        [last_idx, vdim_indyces_dict,hdim1_indices_dict,hdim2_indeces_dict] = get_indices_of_labels(labels, np.array(list(range(1,num_labels+1))), max_init_size_numba)
        #print(last_idx, num_labels)
        #print(values, count)
        #print(np.array(list(values_counts.keys())))
    #create emptry list of features to remove from parent threshold value
    #loop over individual regions:       
    for cur_idx in range(1,num_labels+1):
        max_cur_idx = last_idx[cur_idx][0]
        if max_cur_idx<=n_min_threshold:
            continue

        #print("In feature_detection_threshold_3D, cur_idx: ", cur_idx)
        #region=labels[:,:,:] == cur_idx
        region = np.full(labels_shape, False)
        #[vdim_indyces,hdim1_indices,hdim2_indeces]= np.nonzero(region)
        [vdim_indyces,hdim1_indices,hdim2_indeces]= [vdim_indyces_dict[cur_idx][:max_cur_idx],hdim1_indices_dict[cur_idx][:max_cur_idx],hdim2_indeces_dict[cur_idx][:max_cur_idx]]
        #print(region.shape, np.array(list(zip(vdim_indyces,hdim1_indices,hdim2_indeces))))
        region[vdim_indyces,hdim1_indices,hdim2_indeces] = True

        #print(vdim_indyces, hdim1_indices, hdim2_indeces)
        #print("Original indices shape: ", np.shape(vdim_indyces), np.shape(hdim1_indices), np.shape(hdim2_indeces))
        #region=labels[:,:,:] == cur_idx
        #all_matching_indices = np.argwhere(labels==cur_idx)
        #[vdim_indyces,hdim1_indices,hdim2_indeces]=all_matching_indices.T
        #print(all_matching_indices)
        #print("New indices shape: ", np.shape(vdim_indyces), np.shape(hdim1_indices), np.shape(hdim2_indeces))

        #write region for individual threshold and feature to dict
        region_i=list(zip(vdim_indyces,hdim1_indices,hdim2_indeces))
        regions[cur_idx+idx_start]=region_i
        # Determine feature position for region by one of the following methods:
        vdim_index,hdim1_index,hdim2_index=feature_position_3D(vdim_indyces,hdim1_indices,hdim2_indeces,region,data_i,threshold,position_threshold,target)
        #create individual DataFrame row in tracky format for identified feature
        list_features_threshold.append({'frame': int(i_time),
                                        'idx':cur_idx+idx_start,
                                        'vdim': vdim_index,
                                        'hdim_1': hdim1_index,
                                        'hdim_2':hdim2_index,
                                        'num':max_cur_idx,
                                        'threshold_value':threshold})
    features_threshold=pd.DataFrame(list_features_threshold)
    #else:
    #    features_threshold=pd.DataFrame()
    #    regions=dict()
            
    return features_threshold, regions


from numba import jit
from numba.typed import Dict
import numba.core.types
import numba

# Make array type.  Type-expression is not supported in jit
# functions.

@jit(nopython=True)
def my_np_resize(a, new_size):
    new = np.zeros(new_size, a.dtype)
    new[:a.size] = a
    return new


int_array = numba.core.types.int64[:]
int_array_3d = numba.core.types.int64[:,:,:]
@jit((int_array_3d, int_array, numba.core.types.int64),nopython=True)
def get_indices_of_labels(labels, indices, max_init_size):

    '''
    Returns 3 dicts of label indices
    '''
    label_shape =labels.shape
    z_shape =  label_shape[0]
    y_shape =  label_shape[1]
    x_shape =  label_shape[2]
    z_indices = Dict.empty(
        key_type=numba.core.types.int64,
        value_type=int_array,
    )
    x_indices = Dict.empty(
        key_type=numba.core.types.int64,
        value_type=int_array,
    )
    y_indices = Dict.empty(
        key_type=numba.core.types.int64,
        value_type=int_array,
    )
    
    curr_loc_indices = Dict.empty(
        key_type=numba.core.types.int64,
        value_type=int_array,
    )
    i = 0 
    for index in indices:
        #pass
        curr_loc_indices[index] = np.array([0,])
        z_indices[index] = np.empty((max_init_size,),dtype=np.int64)
        x_indices[index] = np.empty((max_init_size,),dtype=np.int64)
        y_indices[index] = np.empty((max_init_size,),dtype=np.int64)

        
    for z in range(z_shape):
        for y in range(y_shape):
            for x in range(x_shape):
                curr_label = labels[z,y,x] 
                for index in indices:
                    if curr_label == index:
                        curr_loc_ix = curr_loc_indices[index][0]
                        if curr_loc_ix == z_indices[index].size:
                            curr_arr_sz = z_indices[index].size
                            z_indices[index]= my_np_resize(z_indices[index], curr_arr_sz*2)
                            x_indices[index]= my_np_resize(x_indices[index], curr_arr_sz*2)
                            y_indices[index]= my_np_resize(y_indices[index], curr_arr_sz*2)
                            
                        z_indices[index][curr_loc_ix] = z
                        x_indices[index][curr_loc_ix] = x
                        y_indices[index][curr_loc_ix] = y
                        curr_loc_indices[index][0]+=1
                        
    return [curr_loc_indices, z_indices, y_indices, x_indices]

def feature_position_3D(vdim_indyces,hdim1_indices,hdim2_indeces,region,track_data,threshold_i,position_threshold, target):
    '''
    function to  determine feature position
    Input:
        vdim_indyces:     list
    
        hdim1_indices:    list
        
        hdim2_indeces:    list
        
        region:    list
                   list of 2-element tuples
        track_data:     numpy.ndarray
                        2D numpy array containing the data
        
        threshold_i:    float
        
        position_threshold:    str
        
        target:    str

    Output:
        vdim_index:     float
                        feature position along vertical dimension
        hdim1_index:    float
                        feature position along 1st horizontal dimension
        hdim2_index:    float
                        feature position along 2nd horizontal dimension
    '''
    if position_threshold=='center':
        # get position as geometrical centre of identified region:
        vdim_index=np.mean(vdim_indyces)
        hdim1_index=np.mean(hdim1_indices)
        hdim2_index=np.mean(hdim2_indeces)

    elif position_threshold=='extreme':
        #get position as max/min position inside the identified region:
        if target == 'maximum':
            index=np.argmax(track_data[region])
            vdim_index=vdim_indyces[index]
            hdim1_index=hdim1_indices[index]
            hdim2_index=hdim2_indeces[index]

        if target == 'minimum':
            index=np.argmin(track_data[region])
            vdim_index=vdim_indyces[index]
            hdim1_index=hdim1_indices[index]
            hdim2_index=hdim2_indeces[index]

    elif position_threshold=='weighted_diff':
        # get position as centre of identified region, weighted by difference from the threshold:
        weights=abs(track_data[region]-threshold_i)
        if sum(weights)==0:
            weights=None
        vdim_index=np.average(vdim_indyces,weights=weights)
        hdim1_index=np.average(hdim1_indices,weights=weights)
        hdim2_index=np.average(hdim2_indeces,weights=weights)

    elif position_threshold=='weighted_abs':
        # get position as centre of identified region, weighted by absolute values if the field:
        weights=abs(track_data[region])
        if sum(weights)==0:
            weights=None
        vdim_index=np.average(vdim_indyces,weights=weights)
        hdim1_index=np.average(hdim1_indices,weights=weights)
        hdim2_index=np.average(hdim2_indeces,weights=weights)
    else:
        raise ValueError('position_threshold must be center,extreme,weighted_diff or weighted_abs')
    return vdim_index,hdim1_index,hdim2_index

def remove_parents_3D(features_thresholds,regions_i,regions_old):
    '''
    function to remove features whose regions surround newly detected feature regions
    Input:
        features_thresholds:    pandas.DataFrame
                                Dataframe containing detected features
    regions_i:                  dict
                                dictionary containing the regions above/below threshold for the newly detected feature (feature ids as keys)
    regions_old:                dict
                                dictionary containing the regions above/below threshold from previous threshold (feature ids as keys)
    Output:
        features_thresholds     pandas.DataFrame
                                Dataframe containing detected features excluding those that are superseded by newly detected ones
    '''
    list_remove=[]
    for idx_i,region_i in regions_i.items():    
        for idx_old,region_old in regions_old.items():
            if test_overlap(regions_old[idx_old],regions_i[idx_i]):
                list_remove.append(idx_old)
    list_remove=list(set(list_remove))    
    # remove parent regions:
    if features_thresholds is not None:
        features_thresholds=features_thresholds[~features_thresholds['idx'].isin(list_remove)]

    return features_thresholds

def filter_min_distance_3D(features,dxy,dz,min_distance):
    ''' Function to perform feature detection based on contiguous regions above/below a threshold
    Input:    
    features:      pandas DataFrame 
                   features
    dxy:           float
                   horzontal grid spacing (m)
    dz:            float array
                   vertical grid spacing (m)
    min_distance:  float
                   minimum distance between detected features (m)
    Output:
    features:      pandas DataFrame 
                   features
    '''
    from itertools import combinations
    remove_list_distance=[]
    #create list of tuples with all combinations of features at the timestep:
    indeces=combinations(features.index.values,2)
    #Loop over combinations to remove features that are closer together than min_distance and keep larger one (either higher threshold or larger area)
    for index_1,index_2 in indeces:
        if index_1 is not index_2:
            #features.loc[index_1,'hdim_1']
            #distance=dxy*np.sqrt((features.loc[index_1,'hdim_1']-features.loc[index_2,'hdim_1'])**2+(features.loc[index_1,'hdim_2']-features.loc[index_2,'hdim_2'])**2)
            distance=np.sqrt((features.loc[index_1,'projection_x_coordinate']-features.loc[index_2,'projection_x_coordinate'])**2 + (features.loc[index_1,'projection_y_coordinate']-features.loc[index_2,'projection_y_coordinate'])**2 + (features.loc[index_1,'altitude']-features.loc[index_2,'altitude'])**2)
            
            if distance <= min_distance:
#                        logging.debug('distance<= min_distance: ' + str(distance))
                if features.loc[index_1,'threshold_value']>features.loc[index_2,'threshold_value']:
                    remove_list_distance.append(index_2)
                elif features.loc[index_1,'threshold_value']<features.loc[index_2,'threshold_value']:
                    remove_list_distance.append(index_1)
                elif features.loc[index_1,'threshold_value']==features.loc[index_2,'threshold_value']:
                    if features.loc[index_1,'num']>features.loc[index_2,'num']:
                        remove_list_distance.append(index_2)
                    elif features.loc[index_1,'num']<features.loc[index_2,'num']:
                        remove_list_distance.append(index_1)
                    elif features.loc[index_1,'num']==features.loc[index_2,'num']:
                        remove_list_distance.append(index_2)
    features=features[~features.index.isin(remove_list_distance)]
    return features

def add_coordinates_3D(t,variable_cube):
    import numpy as np
    ''' Function adding coordinates from the tracking cube to the trajectories: time, longitude&latitude, x&y&z dimensions
    Input:
    t:             pandas DataFrame
                   trajectories/features
    variable_cube: iris.cube.Cube 
                   Cube containing the dimensions 'time','longitude','latitude','x_projection_coordinate','y_projection_coordinate','altitude' usually cube that the tracking is performed on
    Output:
    t:             pandas DataFrame 
                   trajectories with added coordinates
    '''
    from scipy.interpolate import interp2d, interp1d

    logging.debug('start adding coordinates from cube')

    # pull time as datetime object and timestr from input data and add it to DataFrame:    
    t['time']=None
    t['timestr']=None
    
    
    logging.debug('adding time coordinate')

    time_in=variable_cube.coord('time')
    time_in_datetime=time_in.units.num2date(time_in.points)
    
    t["time"]=time_in_datetime[t['frame']]
    t["timestr"]=[x.strftime('%Y-%m-%d %H:%M:%S') for x in time_in_datetime[t['frame']]]

    # Get list of all coordinates in input cube except for time (already treated):
    coord_names=[coord.name() for coord in  variable_cube.coords()]
    coord_names.remove('time')
    
    logging.debug('time coordinate added')

    # chose right dimension for horizontal and vertical axes based on time dimension:    
    ndim_time=variable_cube.coord_dims('time')[0]
    if ndim_time==0:
        vdim=1
        hdim_1=2
        hdim_2=3
    elif ndim_time==1:
        vdim=0
        hdim_1=2
        hdim_2=3
    elif ndim_time==2:
        vdim=0
        hdim_1=1
        hdim_2=3
    elif ndim_time==3:
        vdim=0
        hdim_1=1
        hdim_2=2
    
    # create vectors to use to interpolate from pixels to coordinates
    dimvec_1=np.arange(variable_cube.shape[vdim])
    dimvec_2=np.arange(variable_cube.shape[hdim_1])
    dimvec_3=np.arange(variable_cube.shape[hdim_2])

    # loop over coordinates in input data:
    for coord in coord_names:
        logging.debug('adding coord: '+ coord)
        # interpolate 1D coordinates:
        if variable_cube.coord(coord).ndim==1:
            
            if variable_cube.coord_dims(coord)==(vdim,):
                f=interp1d(dimvec_1,variable_cube.coord(coord).points,fill_value="extrapolate")
                coordinate_points=f(t['vdim'])

            if variable_cube.coord_dims(coord)==(hdim_1,):
                f=interp1d(dimvec_2,variable_cube.coord(coord).points,fill_value="extrapolate")
                coordinate_points=f(t['hdim_1'])

            if variable_cube.coord_dims(coord)==(hdim_2,):
                f=interp1d(dimvec_3,variable_cube.coord(coord).points,fill_value="extrapolate")
                coordinate_points=f(t['hdim_2'])

        # interpolate 2D coordinates:
        elif variable_cube.coord(coord).ndim==2:

            if variable_cube.coord_dims(coord)==(hdim_1,hdim_2):
                f=interp2d(dimvec_3,dimvec_2,variable_cube.coord(coord).points)
                coordinate_points=[f(a,b) for a,b in zip(t['hdim_2'],t['hdim_1'])]

            if variable_cube.coord_dims(coord)==(hdim_2,hdim_1):
                f=interp2d(dimvec_2,dimvec_3,variable_cube.coord(coord).points)
                coordinate_points=[f(a,b) for a,b in zip(t['hdim_1'],t['hdim_2'])]

        # interpolate 3D coordinates:            
        # mainly workaround for wrf latitude and longitude (to be fixed in future)
        
        elif variable_cube.coord(coord).ndim==3:

            if variable_cube.coord_dims(coord)==(ndim_time,hdim_1,hdim_2):
                f=interp2d(dimvec_2,dimvec_1,variable_cube[0,:,:].coord(coord).points)
                coordinate_points=[f(a,b) for a,b in zip(t['hdim_2'],t['hdim_1'])]
            
            if variable_cube.coord_dims(coord)==(ndim_time,hdim_2,hdim_1):
                f=interp2d(dimvec_1,dimvec_2,variable_cube[0,:,:].coord(coord).points)
                coordinate_points=[f(a,b) for a,b in zip(t['hdim_1'],t['hdim_2'])]

        
            if variable_cube.coord_dims(coord)==(hdim_1,ndim_time,hdim_2):
                f=interp2d(dimvec_2,dimvec_1,variable_cube[:,0,:].coord(coord).points)
                coordinate_points=[f(a,b) for a,b in zip(t['hdim_2'],t['hdim_1'])]

            if variable_cube.coord_dims(coord)==(hdim_1,hdim_2,ndim_time):
                f=interp2d(dimvec_2,dimvec_1,variable_cube[:,:,0].coord(coord).points)
                coordinate_points=[f(a,b) for a,b in zip(t['hdim_2'],t['hdim1'])]

                    
            if variable_cube.coord_dims(coord)==(hdim_2,ndim_time,hdim_1):
                f=interp2d(dimvec_1,dimvec_2,variable_cube[:,0,:].coord(coord).points)
                coordinate_points=[f(a,b) for a,b in zip(t['hdim_1'],t['hdim_2'])]

            if variable_cube.coord_dims(coord)==(hdim_2,hdim_1,ndim_time):
                f=interp2d(dimvec_1,dimvec_2,variable_cube[:,:,0].coord(coord).points)
                coordinate_points=[f(a,b) for a,b in zip(t['hdim_1'],t['hdim_2'])]

        # write resulting array or list into DataFrame:
        t[coord]=coordinate_points

        logging.debug('added coord: '+ coord)
    return t

def test_overlap(region_inner,region_outer):
    '''
    function to test for overlap between two regions (probably scope for further speedup here)
    Input:
    region_1:      list
                   list of 3-element tuples defining the indeces of all cell in the region
    region_2:      list
                   list of 3-element tuples defining the indeces of all cell in the region

    Output:
    overlap:       bool
                   True if there are any shared points between the two regions
    '''
    overlap=frozenset(region_outer).isdisjoint(region_inner)
    return not overlap

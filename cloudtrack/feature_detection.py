import logging
import numpy as np
import pandas as pd

def feature_detection_multithreshold(field_in,
                                     dxy,
                                     threshold,
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
    Output:
    features:      pandas DataFrame 
                   detected features
    '''
    from skimage.measure import label
    from skimage.morphology import binary_erosion
    from scipy.ndimage.filters import gaussian_filter
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
        track_data = data_i.data
        
        track_data=gaussian_filter(track_data, sigma=sigma_threshold) #smooth data slightly to create rounded, continuous field
        # create empty lists to store regions and features for individual timestep
        regions=[]
        list_features_thresholds=[]
        for i_threshold,threshold_i in enumerate(threshold):

            # if looking for minima, set values above threshold to 0 and scale by data minimum:
            if target is 'maximum':
                mask=1*(track_data >= threshold_i)
    
            # if looking for minima, set values above threshold to 0 and scale by data minimum:
            elif target is 'minimum':            
                mask=1*(track_data <= threshold_i)  # only include values greater than threshold

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
                list_features_threshold_i=[]
                #create empty dict to store regions for individual features for this threshold
                regions.append(dict())
                #create emptry list of features to remove from parent threshold value
                list_remove=[]
                
                #loop over individual regions:            
                for cur_idx,count in values_counts.items():
                    region=labels[:,:] == cur_idx
                    [a,b]= np.nonzero(region)                    
                    #write region for individual threshold and feature to dict
                    region_i=list(zip(a,b))
                    regions[i_threshold][cur_idx]=region_i
                    # Determine feature position for region by one of the following methods:
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
                        weights=abs(track_data[region]-threshold_i)
                        if sum(weights)==0:
                            weights=None
                        hdim1_index=np.average(a,weights=weights)
                        hdim2_index=np.average(b,weights=weights)

                    elif position_threshold=='weighted_abs':
                        # get position as centre of identified region, weighted by absolute values if the field:
                        weights=abs(track_data[region])
                        if sum(weights)==0:
                            weights=None
                        hdim1_index=np.average(a,weights=weights)
                        hdim2_index=np.average(b,weights=weights)
                    else:
                        raise ValueError('position_threshold must be center,extreme,weighted_diff or weighted_abs')
                    
                    #create individual DataFrame row in tracky format for identified feature
                    list_features_threshold_i.append(pd.DataFrame(data={'frame': int(i_time),
                                                              'idx':cur_idx,
                                                              'hdim_1': hdim1_index,
                                                              'hdim_2':hdim2_index,
                                                              'num':count,
                                                              'threshold_value':threshold_i},
                                                        index=[i_time]))                
                    # For multiple threshold, record "parent" feature to be removed from Dataframe later
                    if i_threshold>0:
                        for idx,region in regions[i_threshold-1].items():
                            if (any(x in regions[i_threshold-1][idx] for x in region_i)):
                                list_remove.append(idx)

                    
                #check if list of features is not empty, then merge into DataFrame and append to list for different thresholds
                if any([x is not None for x in list_features_threshold_i]):
                    list_features_thresholds.append(pd.concat(list_features_threshold_i, ignore_index=True))
                else: 
                    list_features_thresholds.append(None)
            else:
                list_features_thresholds.append(None)
                regions.append(None)

                # If multiple thresholds, remove "parent" features from detection with previous threshold value
                    # remove duplicates drom list of features to remove from parent threshold:
            if i_threshold>0:

                list_remove=list(set(list_remove))                    
                # remove parent regions
                # remove from DataFrame of last threshold
                if list_features_thresholds[i_threshold-1] is not None:
                    list_features_thresholds[i_threshold-1]=list_features_thresholds[i_threshold-1][~list_features_thresholds[i_threshold-1]['idx'].isin(list_remove)]
                # remove from regions
                if regions[i_threshold-1] is not None:
                    for idx in list_remove:
                        regions[i_threshold-1].pop(idx)

            # finished feature detection for specific threshold value:
            logging.debug('Finished feature detection for threshold '+str(i_threshold) + ' : ' + str(threshold_i) )

        #check if list of features is not empty, then merge features from different threshold values 
        #into one DataFrame and append to list for individual timesteps:
        if any([x is not None for x in list_features_thresholds]):
            features_i_merged=pd.concat(list_features_thresholds, ignore_index=True)
            #Loop over DataFrame to remove features that are closer than distance_min to each other:
            if (min_distance > 0):
                features_i_merged=filter_min_distance(features_i_merged,dxy,min_distance)
            list_features_timesteps.append(features_i_merged)

        else:
            list_features_timesteps.append(None)
            
        logging.debug('Finished feature detection for ' + time_i.strftime('%Y-%m-%d_%H:%M:%S'))


    logging.debug('feature detection: merging DataFrames')
    # Check if features are detected and then concatenate features from different timesteps into one pandas DataFrame
    # If no features are detected raise error
    if any([x is not None for x in list_features_timesteps]):
        features=pd.concat(list_features_timesteps, ignore_index=True)   
    else:
        raise ValueError('No features detected')
    logging.debug('feature detection completed')
    features['feature']=features.index+feature_number_start
#    features_filtered = features.drop(features[features['num'] < min_num].index)
#    features_filtered.drop(columns=['idx','num','threshold_value'],inplace=True)
#    features_unfiltered=add_coordinates(features,field_in)
    features=add_coordinates(features,field_in)

    
    return features

def filter_min_distance(features,dxy,min_distance):
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

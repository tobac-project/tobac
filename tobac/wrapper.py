'''Wraps up methods in feature_dection, segmentation and tracking.

'''

import numpy as np
import logging


def tracking_wrapper(
             field_in_features,
             field_in_segmentation,
             time_spacing=None,
             grid_spacing=None,
             parameters_features=None,
             parameters_tracking=None,
             parameters_segmentation=None,
             ):
    '''
    Parameters
    ----------
    field_in_features : iris.cube.Cube

    field_in_segmentation : iris.cube.Cube

    grid_spacing : float, optional
        Grid spacing in input data. Default is None.

    time_spacing : float, optional
        Time resolution of input data. Default is None.

    parameters_features : optional
        Default is None.

    parameters_tracking : optional
        Default is None.

    parameters_segmentation : optional
        Default is None.

    Raises
    ------
    ValueError
        If method_detection is neither 'threshold' nor
        'threshold_multi'.

        If method_linking is not 'trackpy'.

    Notes
    -----
    needs short summary
    unsure about field_in_features, field_in_segmentation and
    parameters_*
    '''
    
    from .feature_detection import feature_detection_multithreshold
    from .tracking import linking_trackpy
    from .segmentation import segmentation_3D, segmentation_2D
    from .utils import get_spacings

    logger = logging.getLogger('trackpy')
    logger.propagate = False
    logger.setLevel(logging.WARNING)
    
    ### Prepare Tracking

    dxy,dt=get_spacings(field_in_features,grid_spacing=grid_spacing,time_spacing=time_spacing)
    
    ### Start Tracking
    # Feature detection:
    
    method_detection=parameters_features.pop('method_detection',None)
    if method_detection  in ["threshold","threshold_multi"]:
        features=feature_detection_multithreshold(field_in_features,**parameters_features)
    else:
        raise ValueError('method_detection unknown, has to be either threshold_multi or threshold')
        
    method_segmentation=parameters_features.pop('method_segmentation',None)

    if method_segmentation  == 'watershedding':
        if field_in_segmentation.ndim==4:
            segmentation_mask,features_segmentation=segmentation_3D(features,field_in_segmentation,**parameters_segmentation)
        if field_in_segmentation.ndim==3:
            segmentation_mask,features_segmentation=segmentation_2D(features,field_in_segmentation,**parameters_segmentation)
        

    # Link the features in the individual frames to trajectories:
    method_linking=parameters_features.pop('method_linking',None)

    if method_linking == 'trackpy':
        trajectories=linking_trackpy(features,**parameters_tracking)
        logging.debug('Finished tracking')
    else:
        raise ValueError('method_linking unknown, has to be trackpy')

    return features,segmentation_mask,trajectories


def maketrack(field_in,
              grid_spacing=None,time_spacing=None,
              target='maximum',              
              v_max=None,d_max=None,
              memory=0,stubs=5,              
              order=1,extrapolate=0,              
              method_detection="threshold",
              position_threshold='center',
              sigma_threshold=0.5,
              n_erosion_threshold=0,
              threshold=1, min_num=0,
              min_distance=0,              
              method_linking="random",            
              cell_number_start=1,              
              subnetwork_size=None,
              adaptive_stop=None,
              adaptive_step=None, 
              return_intermediate=False,
              ):

    '''Identify features and link them into trajectories.

    Parameters
    ----------
    field_in : iris.cube.Cube
        2D input field tracking is performed on.

    grid_spacing : float, optional
        Grid spacing in input data. Default is None.

    time_spacing : float, optional
        Time resolution of input data. Default is None.

    target : {'maximum', 'minimum'}
        Flag to determine if tracking is targetting minima or maxima in
        the data. Default is 'maximum'.

    v_max : float, optional
        Speed at which features are allowed to move. Default is None.

    d_max : optional
        Default is None.

    memory : int, optional
        Number of timesteps for which objects can be missed by the
        algorithm to still give a constistent track. Default is 0.

        ..warning :: This parameter should be used with caution, as it
                     can lead to erroneous trajectory linking,
                     espacially for data with low time resolution.

    stubs : float, optional
        Default is 5.

    order : int, optional
        Order if interpolation spline to fill gaps in tracking
        (from allowing memory to be larger than 0).

    method_detection: {'threshold', 'threshold_multi'}
        Flag choosing method used for feature detection. Default is
        'threshold'.

    position_threshold : {'center', 'extreme', 'weighted_diff',
                          'weighted_abs'}, optional
        Flag choosing method used for the position of the tracked
        feature. Default is 'center'.

    sigma_threshold: float, optional
        Standard deviation for intial filtering step. Default is 0.5.

    n_erosion_threshold: int, optional
        Number of pixel by which to erode the identified features.
        Default is 0.

    min_num : int, optional
        Minimum number of cells above threshold in the feature to be
        tracked. Default is 0.

    min_distance : float, optional
        Minimum distance between detected features. Default is 0.

    method_linking : {'random', 'predict'}, optional
        Flag choosing method used for trajectory linking. Default is
        'random'.

   cell_number_start : int, optional
        Default is 1.

    adaptive_step : optional
        Default is None.

    adaptive_stop : optional
        Default is None.

    subnetwork_size : int, optional
        Maximim size of subnetwork for linking. Default is None.

    return_intermediate: bool, optional
        Flag to determine if only final tracjectories are output
        (False, default) or if detected features, filtered features and
        unfilled tracks are returned additionally (True).

    Returns
    -------
    trajectories_final: pandas.DataFrame
        Tracked updrafts, one row per timestep and updraft, includes
        dimensions 'time', 'latitude', 'longitude',
        'projection_x_variable', 'projection_y_variable' based on w
        cube. 'hdim_1' and 'hdim_2' are used for segementation step.

    features : pandas.DataFrame

    Raises
    ------
    ValueError
        If input_cube does not contail projection_x_coord and
        projection_y_coord or keyword argument grid_spacing.
    
        If method_detection is neither 'threshold' nor
        'threshold_multi'.

    Notes
    -----
    features needs more information

    Optional output:             
    features_filtered: pandas.DataFrame
    
    features_unfiltered: pandas.DataFrame
    
    trajectories_filtered_unfilled: pandas.DataFrame
    '''

    from .feature_detection import feature_detection_multithreshold
    from .tracking import linking_trackpy

    from copy import deepcopy
    
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
    if method_detection in ["threshold","threshold_multi"]:
        features=feature_detection_multithreshold(field_in=field_in,
                                                  threshold=threshold,
                                                  dxy=dxy,
                                                  target=target,
                                                  position_threshold=position_threshold,
                                                  sigma_threshold=sigma_threshold,
                                                  n_erosion_threshold=n_erosion_threshold)
        features_filtered = features.drop(features[features['num'] < min_num].index)

    else:
        raise ValueError('method_detection unknown, has to be either threshold_multi or threshold')
        
    # Link the features in the individual frames to trajectories:

    trajectories=linking_trackpy(features=features_filtered,
                                 field_in=field_in,
                                 dxy=dxy,
                                 dt=dt,
                                 memory=memory,
                                 subnetwork_size=subnetwork_size,
                                 adaptive_stop=adaptive_stop,
                                 adaptive_step=adaptive_step,
                                 v_max=v_max,
                                 d_max=d_max,
                                 stubs=stubs,              
                                 order=order,extrapolate=extrapolate, 
                                 method_linking=method_linking,
                                 cell_number_start=1
                                 )

    logging.debug('Finished tracking')

    return trajectories,features

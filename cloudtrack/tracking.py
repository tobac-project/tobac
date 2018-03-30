import trackpy as tp
import numpy as np
#from skimage.morphology import watershed
#from skimage.segmentation import random_walker
#from scipy.ndimage.measurements import watershed_ift
from scipy.interpolate import interp2d, interp1d
import pandas as pd
from datetime import datetime

#from pathos.multiprocessing import ProcessingPool as Pool
import logging

def fill_gaps(t,order=1,extrapolate=0,frame_max=None,hdim_1_max=None,hdim_2_max=None):
    from scipy.interpolate import InterpolatedUnivariateSpline
    t_grouped=t.groupby('particle')
    t_list=[]
    if frame_max==None:
        frame_max=t['frame'].max()
    for particle,track in t_grouped:        
        
        frame_in=track['frame'].as_matrix()
        hdim_1_in=track['hdim_1'].as_matrix()
        hdim_2_in=track['hdim_2'].as_matrix()
        s_x = InterpolatedUnivariateSpline(frame_in, hdim_1_in, k=order)
        s_y = InterpolatedUnivariateSpline(frame_in, hdim_2_in, k=order)
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
        hdim_1_out=s_x(frame_out)
        hdim_2_out=s_y(frame_out)
        track['hdim_1']=hdim_1_out
        track['hdim_2']=hdim_2_out
        track['particle']=particle        
        # add relative time since start of track:
        track['time_cell']=np.nan
        track['time_rel']=track['time']-track.loc[0,'time']
        t_list.append(track)    
        

    t_out=pd.concat(t_list)
    t_out=t_out.loc[(t_out['hdim_1']<hdim_1_max) & (t_out['hdim_2']<hdim_2_max) &(t_out['hdim_1']>0) & (t_out['hdim_2']>0)]
    
    
    
    
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
    time_in=variable_cube.coord('time')
    ndim_time=variable_cube.coord_dims('time')[0]
    
    t['time']=None
    t['timestr']=None

    for i, row in t.iterrows():
        t.loc[i,'time']=time_in.units.num2date(time_in[int(row['frame'])].points[0])
        t.loc[i,'timestr']=time_in.units.num2date(time_in[int(row['frame'])].points[0]).strftime('%Y-%m-%d %H:%M:%S')

    coord_names=[coord.name() for coord in  variable_cube.coords()]
    coord_names.remove('time')
    
    if ndim_time==0:
        hdim_1=1
        hdim_2=2
    elif ndim_time==1:
        hdim_1=0
        hdim_2=2
    elif ndim_time==2:
        hdim_1=0
        hdim_2=1
    
    logging.debug('hdim_1: ' + str(hdim_1))
    logging.debug('hdim_2: ' + str(hdim_2))

    dimvec_1=np.arange(variable_cube.shape[hdim_1])
    dimvec_2=np.arange(variable_cube.shape[hdim_2])
    # logging.debug("%s : %s",datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'dimvec1 [x]'+str(dimvec_1))
    # logging.debug("%s : %s",datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'dimvec2 [y]'+str(dimvec_2))



    for coord in coord_names:
        if variable_cube.coord(coord).ndim==1:
            # logging.debug("%s : %s",datetime.now().strftime('%Y-%m-%d %H:%M:%S'),str(variable_cube.coord(coord)))

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




        elif variable_cube.coord(coord).ndim==2:
            # logging.debug("%s : %s",datetime.now().strftime('%Y-%m-%d %H:%M:%S'),str(variable_cube.coord(coord)))
            # logging.debug("%s : %s",datetime.now().strftime('%Y-%m-%d %H:%M:%S'),str(variable_cube.coord(coord).points.shape))

            t[coord]=np.nan            
            if variable_cube.coord_dims(coord)==(hdim_1,hdim_2):
                f=interp2d(dimvec_2,dimvec_1,variable_cube.coord(coord).points)
                for i, row in t.iterrows():
                    t.loc[i,coord]=float(f(row['hdim_1'],row['hdim_2']))
            if variable_cube.coord_dims(coord)==(hdim_2,hdim_1):
                f=interp2d(dimvec_2,dimvec_1,variable_cube.coord(coord).points)
                for i, row in t.iterrows():
                    t.loc[i,coord]=float(f(row['hdim_2'],row['hdim_1']))
        
        
        elif variable_cube.coord(coord).ndim==3:
            # logging.debug("%s : %s",datetime.now().strftime('%Y-%m-%d %H:%M:%S'),str(variable_cube.coord(coord)))
            # logging.debug("%s : %s",datetime.now().strftime('%Y-%m-%d %H:%M:%S'),str(variable_cube.coord(coord).points.shape))


            t[coord]=np.nan
            # mainly workaround for wrf latitude and longitude (possibly switch to do things by timestep)
            if variable_cube.coord_dims(coord)==(ndim_time,hdim_1,hdim_2):
                # logging.debug(str(coord))
                # logging.debug('ndim_time,hdim_1,hdim_2')
                # logging.debug('dimvec_1.shape: '+ str(dimvec_1.shape))
                # logging.debug('dimvec_2.shape: '+ str(dimvec_2.shape))
                # logging.debug('variable_cube.shape: '+str(variable_cube.shape))
                f=interp2d(dimvec_2,dimvec_1,variable_cube[0,:,:].coord(coord).points)
                for i, row in t.iterrows():
                    t.loc[i,coord]=float(f(row['hdim_2'],row['hdim_1']))
            
            if variable_cube.coord_dims(coord)==(ndim_time,hdim_2,hdim_1):
                # logging.debug(str(coord))
                # logging.debug('ndim_time,hdim_2,hdim_1')
                # logging.debug('dimvec_1.shape: '+str(dimvec_1.shape))
                # logging.debug('dimvec_2.shape: '+str(dimvec_1.shape))
                # logging.debug('variable_cube.shape: '+str(variable_cube.shape))
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
                    t.loc[i,coord]=float(f(row['hdim_1'],row['hdim_2']))
                    
                    
            if variable_cube.coord_dims(coord)==(hdim_2,ndim_time,hdim_1):
                f=interp2d(dimvec_1,dimvec_2,variable_cube[:,0,:].coord(coord).points)
                for i, row in t.iterrows():
                    t.loc[i,coord]=float(f(row['hdim_1'],row['hdim_2']))
                    
            if variable_cube.coord_dims(coord)==(hdim_2,hdim_1,ndim_time):
                f=interp2d(dimvec_1,dimvec_2,variable_cube[:,:,0].coord(coord).points)
                for i, row in t.iterrows():
                    t.loc[i,coord]=float(f(row['hdim_1'],row['hdim_2']))

    return t


def feature_detection_trackpy(field_in,diameter,dxy,target='maximum'):
    diameter_pix=round(int(diameter/dxy)/2)*2+1

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
        # logging.debug('feature detection for timestep '+ str(i))
        f_i=tp.locate(data_i.data, diameter_pix, invert=invert,
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
    features.rename(columns={'x':'hdim_1', 'y':'hdim_2'}, inplace =True)
    logging.debug('feature detection completed')
    return features


def feature_detection_blob(field_in,threshold,dxy,target='maximum'):
    # Image processing
    from skimage import filters, measure



    # set invert to True when tracking minima, False when tracking maxima
    # if target=='maximum':
    #     invert= False
    # elif target =='minimum':
    #     invert= True
    # else:
    #     raise ValueError('target has to be either maximum or minimum')

    
    # locate features for each timestep and then combine
    list_features=[]   
    data_time=field_in.slices_over('time')
    for i,data_i in enumerate(data_time):
        logging.debug('feature detection for timestep '+ str(i))
        testW = data_i.data
        testW[testW < threshold] = 0 # only include w greater than X m/s
        testW = testW/np.max(testW) # make into -1 to 1 image (for image processing)
        im = filters.gaussian(testW, sigma=0.5) #smooth data slightly to create rounded, continuous updrafts
        #im = filters.gaussian(testW, sigma=0) #smooth data slightly to create rounded more uniform updrafts
        blobs = im > im.mean() # Find updraft regions in image
        # all_labels = measure.label(blobs) # create image labels
        blobs_labels = measure.label(blobs, background=0)
        xxx = np.unique(blobs_labels[:,:].ravel())
        for j in np.arange(1,len(xxx)):        
            cur_idx = xxx[j];
            [a,b] = np.where(blobs_labels[:,:] == cur_idx)
            data_frame={'frame': int(i),'hdim_1': np.mean(a),'hdim_2':np.mean(b)}
            f_i=pd.DataFrame(data=data_frame,index=[i])
            list_features.append(f_i)
    logging.debug('feature detection: merging DataFrames')
    features=pd.concat(list_features)
    logging.debug('feature detection completed')
    return features



    
def trajectory_linking(features,v_max,dt,dxy,memory,subnetwork_size=None):
    # from trackpy import predict
    import trackpy
    search_range=int(dt*v_max/dxy)
    logging.debug('start feature linking')

    # pred = predict.NearestVelocityPredict()
    if subnetwork_size is not None:
        trackpy.linking.Linker.MAX_SUB_NET_SIZE=subnetwork_size
    trajectories = tp.link_df(features, search_range, memory=memory,pos_columns=['hdim_2','hdim_1'])
    # trajectories = pred.link_df(features, search_range, memory=memory)
               # pos_columns=['hdim_1','hdim_2'],
               # t_column='frame'
#                   neighbor_strategy='KDTree', link_strategy='auto',
#                   predictor=None, adaptive_stop=None, adaptive_step=0.95,
#                   copy_features=False, diagnostics=False, pos_columns=None,
#                   t_column=None, hash_size=None, box_size=None, verify_integrity=True,
#                   retain_index=False
                # )
    logging.debug('feature linking completed')

    return trajectories


def maketrack(field_in,grid_spacing=None,diameter=5000,target='maximum',v_max=10,memory=3,stubs=5,
              min_mass=0, min_signal=0,
              order=1,extrapolate=0,
              parameters_features=False,
              cell_number_start=1,
              subnetwork_size=None,
              method="trackpy",
              threshold=1):
    """
    Function using watershedding to determine cloud volumes associated with tracked updrafts
    
    Parameters:
    field_in:     iris.cube.Cube 
                  2D input field tracking is performed on
    grid_spacing: float
                  grid spacing in input data (m)
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

    coord_names=[coord.name() for coord in  field_in.coords()]
    if ('projection_x_coordinate' in coord_names and 'projection_y_coordinate' in coord_names):
        dx=np.diff(field_in.coord('projection_x_coordinate').points)[0]
        dy=np.diff(field_in.coord('projection_y_coordinate').points)[0]    
        dxy=0.5*(dx+dy)
    elif grid_spacing is not None:
        dxy=grid_spacing
    else:
        ValueError('no information about grid spacing, need either input cube with projection_x_coord and projection_y_coord or keyword argument grid_spacing')

    dt=np.diff(field_in.coord('time').points)[0]*24*3600

    # Settings for the tracking algorithm thresholds etc..)
    
 
    if method=="trackpy":
        features=feature_detection_trackpy(field_in,diameter=diameter,dxy=dxy,target='maximum')
    
    elif method == "blob":
        features=feature_detection_blob(field_in,threshold=threshold,dxy=dxy,target='maximum')

    # Linking of the features in the individual frames to trajectories
    trajectories_unfiltered=trajectory_linking(features,v_max=v_max,dt=dt,dxy=dxy,memory=memory,subnetwork_size=subnetwork_size)
    
    # f.rename(columns={"frame":"timestep","x":"hdim_1","y":"hdim_2"})
    # Reset particle numbers from the arbitray numbers at the end of the feature detection and linking to consecutive numbers, keep 'particle' for reference to the feature detection step.
                      
# Filter trajectories to exclude short trajectories that are likely to be spurious
    trajectories_filtered = tp.filter_stubs(trajectories_unfiltered,threshold=stubs)
    trajectories_filtered=trajectories_filtered.reset_index(drop=True)


# Filter trajectories based on a minimum mass (not ne been seen as sth physical) and signal within the trajectory
    if method=="trackpy":
        #Filters:
        min_mass_pix=min_mass/(dxy*dxy)
        min_signal_pix=min_signal

        condition = lambda x: (
                (x['mass'].max() > min_mass_pix) 
                &   
                (x['signal'].max() > min_signal_pix)
                )
        trajectories_filtered = tp.filter(trajectories_filtered, condition)
        trajectories_filtered=trajectories_filtered.reset_index(drop=True)
        
        # Restrict output and further treatment to relevant columns:
        trajectories_filtered['mass']=trajectories_filtered['mass']*(dxy*dxy)
        trajectories_filtered['size']=trajectories_filtered['size']*(dxy)
    
    
    if method=="trackpy":

        if not parameters_features:
            trajectories_filtered=trajectories_filtered.drop(['mass','signal','size','ecc'],axis=1)
        
        
    trajectories_filtered['particle_old']=trajectories_filtered['particle']
    

    for i,particle in enumerate(pd.Series.unique(trajectories_filtered['particle_old'])):
        particle_new=int(i+cell_number_start)
        # logging.debug("%s : %s",datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'i,particle,paerticle_new'+str([i,particle,particle_new]))
        trajectories_filtered.loc[trajectories_filtered['particle_old']==particle,'particle']=particle_new


    #Interpolate to fill the gaps in the trajectories (left from allowing memory in the linking)
    trajectories_filtered=fill_gaps(trajectories_filtered,order=order,extrapolate=extrapolate,frame_max=field_in.shape[0],hdim_1_max=field_in.shape[1],hdim_2_max=field_in.shape[2])
    
    trajectories_filtered=add_coordinates(trajectories_filtered,field_in)

    features_identified=add_coordinates(features,field_in)
    features_identified=features

    return trajectories_filtered, features_identified


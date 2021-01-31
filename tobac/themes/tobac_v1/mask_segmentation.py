'''Provide segmentation techniques.

Segmentation techniques are used to associate areas or volumes to each
identified feature. The segmentation is implemented using watershedding
techniques from the field of image processing with a fixed threshold
value. This value has to be set specifically for every type of input
data and application. The segmentation can be performed for both
two-dimensional and three-dimensional data. At each timestep, a marker
is set at the position (weighted mean center) of each feature identified
in the detection step in an array otherwise filled with zeros. In case
of the three-dimentional watershedding, all cells in the column above
the weighted mean center position of the identified features fulfilling
the threshold condition are set to the respective marker. The algorithm
then fills the area (2D) or volume (3D) based on the input field
starting from these markers until reaching the threshold. If two or more
cloud objects are directly connected, the border runs along the
watershed line between the two regions. This procedure creates a mask of
the same shape as the input data, with zeros at all grid points where
there is no cloud or updraft and the integer number of the associated
feature at all grid points belonging to that specific cloud/updraft.
this mask can be conveniently and efficiently used to select the volume
of each cloud object at a specific time step for further analysis or
visialization. [4]_

References
----------
.. [4] Heikenfeld, M., Marinescu, P. J., Christensen, M., Watson-Parris,
   D., Senf, F., van den Heever, S. C., and Stier, P.: tobac v1.0:
   towards a flexible framework for tracking and analysis of clouds in
   diverse datasets, Geosci. Model Dev. Discuss.,
   https://doi.org/10.5194/gmd-2019-105 , in review, 2019, 7ff.
'''

import logging
import xarray

def segmentation(features,field,dxy,threshold=3e-3,target='maximum',level=None,method='watershed',max_distance=None,vertical_coord='auto'):
    '''Use watershedding or random walker technique to determine region above
    a threshold value around initial seeding position for all time steps of
    the input data. Works both in 2D (based on single seeding point) and 3D and
    returns a mask with zeros everywhere around the identified regions and the
    feature id inside the regions.

    Calls segmentation_timestep at each individal timestep of the input data.

    Parameters
    ----------
    features : xarray.Dataset
        Output from trackpy/maketrack.

    field : xarray.DataArray
        Containing the field to perform the watershedding on.

    dxy : float
	Grid spacing of the input data.

    threshold : float, optional
        Threshold for the watershedding field to be used for the mask.
        Default is 3e-3.

    target : {'maximum', 'minimum'}, optional
        Flag to determine if tracking is targetting minima or maxima in
        the data. Default is 'maximum'.

    level : slice of iris.cube.Cube, optional
        Levels at which to seed the cells for the watershedding
        algorithm. Default is None.

    method : {'watershed'}, optional
        Flag determining the algorithm to use (currently watershedding
        implemented). 'random_walk' could be uncommented.

    max_distance : float, optional
        Maximum distance from a marker allowed to be classified as
        belonging to that cell. Default is None.

    vertical_coord : {'auto', 'z', 'model_level_number', 'altitude',
                      'geopotential_height'}, optional
        Name of the vertical coordinate for use in 3D segmentation case

    Returns
    -------
    segmentation_out : xarray.DataArray
        Cloud mask, 0 outside and integer numbers according to track
        inside the clouds.

    features_out : xarray.Dataset
        Feature dataframe including the number of cells (2D or 3D) in
        the segmented area/volume of the feature at the timestep.

    Raises
    ------
    ValueError
        If field_in.ndim is neither 3 nor 4 and 'time' is not included
        in coords.
    '''
    print(field)
    import pandas as pd
    from iris.cube import CubeList

    logging.info('Start watershedding 3D')

    # convert to iris/pandas
    field=field.to_iris()
    features=features.to_dataframe()

    # check input for right dimensions:
    if not (field.ndim==3 or field.ndim==4):
        raise ValueError('input to segmentation step must be 3D or 4D including a time dimension')
    if 'time' not in [coord.name() for coord in field.coords()]:
        raise ValueError("input to segmentation step must include a dimension named 'time'")

    # CubeList and list to store individual segmentation masks and feature DataFrames with information about segmentation
    segmentation_out_list=CubeList()
    features_out_list=[]

    #loop over individual input timesteps for segmentation:
    field_time=field.slices_over('time')
    for i,field_i in enumerate(field_time):
        time_i=field_i.coord('time').units.num2date(field_i.coord('time').points[0])
        features_i=features.loc[features['time']==time_i]
        segmentation_out_i,features_out_i=segmentation_timestep(field_i,features_i,dxy,threshold=threshold,target=target,level=level,method=method,max_distance=max_distance,vertical_coord=vertical_coord)
        segmentation_out_list.append(segmentation_out_i)
        features_out_list.append(features_out_i)
        logging.debug('Finished segmentation for '+time_i.strftime('%Y-%m-%d_%H:%M:%S'))

    #Merge output from individual timesteps:
    segmentation_out=segmentation_out_list.merge_cube()
    features_out=pd.concat(features_out_list)

    features_out=features_out.to_xarray()
    segmentation_out=xarray.DataArray.from_iris(segmentation_out)

    logging.debug('Finished segmentation')
    return segmentation_out,features_out



def segmentation_timestep(field_in,features_in,dxy,threshold=3e-3,target='maximum',level=None,method='watershed',max_distance=None,vertical_coord='auto'):
    """Perform watershedding for an individual time step of the data. Works for
    both 2D and 3D data

    Parameters
    ----------
    field_in : iris.cube.Cube
        Input field to perform the watershedding on (2D or 3D for one
        specific point in time).

    features_in : pandas.DataFrame
        Features for one specific point in time.

    dxy : float
    	Grid spacing of the input data in metres

    threshold : float, optional
        Threshold for the watershedding field to be used for the mask.
        Default is 3e-3.

    target : {'maximum', 'minimum'}, optional
        Flag to determine if tracking is targetting minima or maxima in
        the data to determine from which direction to approach the threshold
        value. Default is 'maximum'.

    level : slice of iris.cube.Cube, optional
        Levels at which to seed the cells for the watershedding
        algorithm. Default is None.

    method : {'watershed'}, optional
        Flag determining the algorithm to use (currently watershedding
        implemented). 'random_walk' could be uncommented.

    max_distance : float, optional
        Maximum distance from a marker allowed to be classified as
        belonging to that cell. Default is None.

    vertical_coord : str, optional
        Vertical coordinate in 3D input data. If 'auto', input is checked for
        one of {'z', 'model_level_number', 'altitude','geopotential_height'} as
        a likely coordinate name

    Returns
    -------
    segmentation_out : iris.cube.Cube
        Cloud mask, 0 outside and integer numbers according to track
        inside the clouds.

    features_out : pandas.DataFrame
        Feature dataframe including the number of cells (2D or 3D) in
        the segmented area/volume of the feature at the timestep.

    Raises
    ------
    ValueError
        If target is neither 'maximum' nor 'minimum'.

        If vertical_coord is not in {'auto', 'z', 'model_level_number',
                                     'altitude', geopotential_height'}.

        If there is more than one coordinate name.

        If the spatial dimension is neither 2 nor 3.

        If method is not 'watershed'.

    """

    from skimage.segmentation import watershed
    # from skimage.segmentation import random_walker
    from scipy.ndimage import distance_transform_edt
    from copy import deepcopy
    import numpy as np

    # copy feature dataframe for output
    features_out=deepcopy(features_in)
    # Create cube of the same dimensions and coordinates as input data to store mask:
    segmentation_out=1*field_in
    segmentation_out.rename('segmentation_mask')
    segmentation_out.units=1

    #Create dask array from input data:
    data=field_in.core_data()

    #Set level at which to create "Seed" for each feature in the case of 3D watershedding:
    # If none, use all levels (later reduced to the ones fulfilling the theshold conditions)
    if level==None:
        level=slice(None)

    # transform max_distance in metres to distance in pixels:
    if max_distance is not None:
        max_distance_pixel=np.ceil(max_distance/dxy)

    # mask data outside region above/below threshold and invert data if tracking maxima:
    if target == 'maximum':
        unmasked=data>threshold
        data_segmentation=-1*data
    elif target == 'minimum':
        unmasked=data<threshold
        data_segmentation=data
    else:
        raise ValueError('unknown type of target')

    # set markers at the positions of the features:
    markers = np.zeros(unmasked.shape).astype(np.int32)
    if field_in.ndim==2: #2D watershedding
        for index, row in features_in.iterrows():
            markers[int(row['hdim_1']), int(row['hdim_2'])]=row['feature']

    elif field_in.ndim==3: #3D watershedding
        list_coord_names=[coord.name() for coord in field_in.coords()]
        #determine vertical axis:
        if vertical_coord=='auto':
            list_vertical=['z','model_level_number','altitude','geopotential_height']
            for coord_name in list_vertical:
                if coord_name in list_coord_names:
                    vertical_axis=coord_name
                    break
        elif vertical_coord in list_coord_names:
            vertical_axis=vertical_coord
        else:
            raise ValueError('Plese specify vertical coordinate')
        ndim_vertical=field_in.coord_dims(vertical_axis)
        if len(ndim_vertical)>1:
            raise ValueError('please specify 1 dimensional vertical coordinate')
        for index, row in features_in.iterrows():
            if ndim_vertical[0]==0:
                markers[:,int(row['hdim_1']), int(row['hdim_2'])]=row['feature']
            elif ndim_vertical[0]==1:
                markers[int(row['hdim_1']),:, int(row['hdim_2'])]=row['feature']
            elif ndim_vertical[0]==2:
                markers[int(row['hdim_1']), int(row['hdim_2']),:]=row['feature']
    else:
        raise ValueError('Segmentations routine only possible with 2 or 3 spatial dimensions')

    # set markers in cells not fulfilling threshold condition to zero:
    markers[~unmasked]=0

    # Turn into np arrays (not necessary for markers) as dask arrays don't yet seem to work for watershedding algorithm
    data_segmentation=np.array(data_segmentation)
    unmasked=np.array(unmasked)

    # perform segmentation:
    if method=='watershed':
        segmentation_mask = watershed(np.array(data_segmentation),markers.astype(np.int32), mask=unmasked)
#    elif method=='random_walker':
#        segmentation_mask=random_walker(data_segmentation, markers.astype(np.int32),
#                                          beta=130, mode='bf', tol=0.001, copy=True, multichannel=False, return_full_prob=False, spacing=None)
    else:
        raise ValueError('unknown method, must be watershed')

    # remove everything from the individual masks that is more than max_distance_pixel away from the markers
    if max_distance is not None:
        D=distance_transform_edt((markers==0).astype(int))
        segmentation_mask[np.bitwise_and(segmentation_mask>0, D>max_distance_pixel)]=0

    #Write resulting mask into cube for output
    segmentation_out.data=segmentation_mask

    # count number of grid cells asoociated to each tracked cell and write that into DataFrame:
    values, count = np.unique(segmentation_mask, return_counts=True)
    counts=dict(zip(values, count))
    ncells=np.zeros(len(features_out))
    for i,(index,row) in enumerate(features_out.iterrows()):
        if row['feature'] in counts.keys():
            ncells=counts[row['feature']]
    features_out['ncells']=ncells

    return segmentation_out,features_out

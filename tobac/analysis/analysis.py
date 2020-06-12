'''Provide tools to analyse and visualize the tracked objects.
This module provides a set of routines that enables performing analyses
and deriving statistics for individual clouds, such as the time series
of integrated properties and vertical profiles. It also provides
routines to calculate summary statistics of the entire populatin of
tracked clouds in the cloud field like histograms of cloud areas/volumes
or cloud mass and a detailed cell lifetime analysis. These analysis
routines are all built in a modular manner. Thus, users can reuse the
most basic methods for interacting with the data structure of the
package in their own analysis procedures in Python. This includes
functions perfomring simple tasks like looping over all identified
objects or cloud trajectories and masking arrays for the analysis of
individual cloud objects. Plotting routines include both visualizations
for individual convective cells and their properties. [1]_

References
----------
.. [1] Heikenfeld, M., Marinescu, P. J., Christensen, M., Watson-Parris,
   D., Senf, F., van den Heever, S. C., and Stier, P.: tobac v1.0:
   towards a flexible framework for tracking and analysis of clouds in
   diverse datasets, Geosci. Model Dev. Discuss.,
   https://doi.org/10.5194/gmd-2019-105 , in review, 2019, 10.

Notes
-----
unsure about page numer in the reference
'''

import pandas as pd
import numpy as np
import logging
import os

from tobac.utils import mask_cell,mask_cell_surface,mask_cube_cell,get_bounding_box
from tobac.utils.convert import xarray_to_iris,iris_to_xarray

@xarray_to_iris
def cell_statistics_all(input_cubes,track,mask,aggregators,output_path='./',cell_selection=None,output_name='Profiles',width=10000,z_coord='model_level_number',dimensions=['x','y'],**kwargs):
    '''
    Parameters
    ----------
    input_cubes : iris.cube.Cube

    track : dask.dataframe.DataFrame

    mask : iris.cube.Cube
        Cube containing mask (int id for tracked volumes 0 everywhere
        else).

    aggregators

    output_path : str, optional
        Default is './'.
    
    cell_selection : optional
        Default is None.

    output_name : str, optional
        Default is 'Profiles'.

    width : int, optional
        Default is 10000.

    z_coord : str, optional
        Name of the vertical coordinate in the cube. Default is
        'model_level_number'.

    dimensions : list of str, optional
        Default is ['x', 'y'].

    **kwargs

    Returns
    -------
    None

    Notes
    -----
    unsure about anything
    needs a short summary
    '''

    if cell_selection is None:
        cell_selection=np.unique(track['cell'])
    for cell in cell_selection :
        cell_statistics(input_cubes=input_cubes,track=track, mask=mask,
                        dimensions=dimensions,aggregators=aggregators,cell=cell,
                        output_path=output_path,output_name=output_name,
                        width=width,z_coord=z_coord,**kwargs)
        
@xarray_to_iris
def cell_statistics(input_cubes,track,mask,aggregators,cell,output_path='./',output_name='Profiles',width=10000,z_coord='model_level_number',dimensions=['x','y'],**kwargs):
    '''
    Parameters
    ----------
    input_cubes : iris.cube.Cube

    track : dask.dataframe.DataFrame

    mask : iris.cube.Cube
        Cube containing mask (int id for tracked volumes 0 everywhere
        else).

    aggregators

    cell : int
        Integer id of cell to create masked cube for output.

    output_path : str, optional
        Default is './'.

    output_name : str, optional
        Default is 'Profiles'.

    width : int, optional
        Default is 10000.

    z_coord : str, optional
        Name of the vertical coordinate in the cube. Default is
        'model_level_number'.

    dimensions : list of str, optional
        Default is ['x', 'y'].

    **kwargs

    Returns
    -------
    None

    Notes
    -----
    unsure about anything
    needs a short summary
    '''
    from iris.cube import Cube,CubeList
    from iris.coords import AuxCoord
    from iris import Constraint,save    
    
    # If input is single cube, turn into cubelist
    if type(input_cubes) is Cube:
        input_cubes=CubeList([input_cubes])
    
    logging.debug('Start calculating profiles for cell '+str(cell))
    track_i=track[track['cell']==cell]
    
    cubes_profile={}
    for aggregator in aggregators:
        cubes_profile[aggregator.name()]=CubeList()
        
    for time_i in track_i['time'].values:
        constraint_time = Constraint(time=time_i)
        
        mask_i=mask.extract(constraint_time)
        mask_cell_i=mask_cell(mask_i,cell,track_i,masked=False)
        mask_cell_surface_i=mask_cell_surface(mask_i,cell,track_i,masked=False,z_coord=z_coord)

        x_dim=mask_cell_surface_i.coord_dims('projection_x_coordinate')[0]
        y_dim=mask_cell_surface_i.coord_dims('projection_y_coordinate')[0]
        x_coord=mask_cell_surface_i.coord('projection_x_coordinate')
        y_coord=mask_cell_surface_i.coord('projection_y_coordinate')
    
        if (mask_cell_surface_i.core_data()>0).any():
            box_mask_i=get_bounding_box(mask_cell_surface_i.core_data(),buffer=1)
    
            box_mask=[[x_coord.points[box_mask_i[x_dim][0]],x_coord.points[box_mask_i[x_dim][1]]],
                     [y_coord.points[box_mask_i[y_dim][0]],y_coord.points[box_mask_i[y_dim][1]]]]
        else:
            box_mask=[[np.nan,np.nan],[np.nan,np.nan]]
    
        x=track_i[track_i['time'].values==time_i]['projection_x_coordinate'].values[0]
        y=track_i[track_i['time'].values==time_i]['projection_y_coordinate'].values[0]

        box_slice=[[x-width,x+width],[y-width,y+width]]
               
        x_min=np.nanmin([box_mask[0][0],box_slice[0][0]])
        x_max=np.nanmax([box_mask[0][1],box_slice[0][1]])
        y_min=np.nanmin([box_mask[1][0],box_slice[1][0]])
        y_max=np.nanmax([box_mask[1][1],box_slice[1][1]])

        constraint_x=Constraint(projection_x_coordinate=lambda cell: int(x_min) < cell < int(x_max))
        constraint_y=Constraint(projection_y_coordinate=lambda cell: int(y_min) < cell < int(y_max))

        constraint=constraint_time & constraint_x & constraint_y
#       Mask_cell_surface_i=mask_cell_surface(Mask_w_i,cell,masked=False,z_coord='model_level_number')
        mask_cell_i=mask_cell_i.extract(constraint)
        mask_cell_surface_i=mask_cell_surface_i.extract(constraint)

        input_cubes_i=input_cubes.extract(constraint)
        for cube in input_cubes_i:
            cube_masked=mask_cube_cell(cube,mask_cell_i,cell,track_i)
            coords_remove=[]
            for coordinate in cube_masked.coords(dim_coords=False):

                if coordinate.name() not in dimensions:
                    for dim in dimensions:
                        if set(cube_masked.coord_dims(coordinate)).intersection(set(cube_masked.coord_dims(dim))):
                            coords_remove.append(coordinate.name())
            for coordinate in set(coords_remove):
                cube_masked.remove_coord(coordinate)            
            
            for aggregator in aggregators:
                cube_collapsed=cube_masked.collapsed(dimensions,aggregator,**kwargs)
                #remove all collapsed coordinates (x and y dim, scalar now) and keep only time as all these coordinates are useless
                for coordinate in cube_collapsed.coords():
                    if not cube_collapsed.coord_dims(coordinate):
                        if coordinate.name() != 'time':
                            cube_collapsed.remove_coord(coordinate)
                logging.debug(str(cube_collapsed))
                cubes_profile[aggregator.name()].append(cube_collapsed)


    minutes=(track_i['time_cell']/pd.Timedelta(minutes=1)).values
    latitude=track_i['latitude'].values
    longitude=track_i['longitude'].values
    minutes_coord=AuxCoord(minutes,long_name='cell_time',units='min')
    latitude_coord=AuxCoord(latitude,long_name='latitude',units='degrees')
    longitude_coord=AuxCoord(longitude,long_name='longitude',units='degrees')
    
    for aggregator in aggregators:
        cubes_profile[aggregator.name()]=cubes_profile[aggregator.name()].merge()
        for cube in cubes_profile[aggregator.name()]:
            cube.add_aux_coord(minutes_coord,data_dims=cube.coord_dims('time'))
            cube.add_aux_coord(latitude_coord,data_dims=cube.coord_dims('time'))
            cube.add_aux_coord(longitude_coord,data_dims=cube.coord_dims('time'))
        os.makedirs(os.path.join(output_path,output_name,aggregator.name()),exist_ok=True)
        savefile=os.path.join(output_path,output_name,aggregator.name(),output_name+'_'+ aggregator.name()+'_'+str(int(cell))+'.nc')
        save(cubes_profile[aggregator.name()],savefile)

@xarray_to_iris
def lifetime_histogram(Track,bin_edges=np.arange(0,200,20),density=False,return_values=False):
    '''
    Parameters
    ----------
    Track

    bin_edged : ndarray, optional
        Default is np.arange(0, 200, 20).

    density : bool, optional
        Default is False.

    return_values : bool, optional
        Default is False.

    Returns
    -------
    hist

    bin_edges : ndarray

    bin_centers

    minutes : float

    Notes
    -----
    unsure about anything
    needs short summary
    '''

    Track_cell=Track.groupby('cell')
    minutes=(Track_cell['time_cell'].max()/pd.Timedelta(minutes=1)).values
    hist, bin_edges = np.histogram(minutes, bin_edges,density=density)
    bin_centers=bin_edges[:-1]+0.5*np.diff(bin_edges)
    if return_values:
        return hist,bin_edges,bin_centers,minutes
    else:
        return hist,bin_edges,bin_centers
    
@xarray_to_iris
def haversine(lat1,lon1,lat2,lon2):
    '''Computes the Haversine distance in kilometers.

    Calculates the Haversine distance between two points
    (based on implementation CIS https://github.com/cedadev/cis).

    Parameters
    ----------
    lat1, lon1 : array of latitude, longitude
        First point or points as array in degrees.

    lat2, lon2 : array of latitude, longitude
        Second point or points as array in degrees.

    Returns
    -------
    arclen * RADIUS_EARTH : float
        Distance between the two points in kilometers.

    Notes
    -----
    check types

    RADIUS_EARTH = 6378.0
    '''

    RADIUS_EARTH = 6378.0
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    #print(lat1,lat2,lon1,lon2)
    arclen = 2 * np.arcsin(np.sqrt((np.sin((lat2 - lat1) / 2)) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin((lon2 - lon1) / 2)) ** 2))
    return arclen * RADIUS_EARTH

@xarray_to_iris
def calculate_distance(feature_1,feature_2,method_distance=None):
    '''Computes distance between two features.

    It is based on either lat/lon coordinates or x/y coordinates.

    Parameters
    ----------
    feature_1, feature_2 : array of latitude, longitute
        First and second feature or points as array in degrees.

    method_distance : {None, 'xy', 'latlon'}, optional
        Default is None.

    Returns
    -------
    distance : float
        Between the two features in meters.

    Notes
    -----
    check sense of types and descriptions
    '''

    if method_distance is None:
        if ('projection_x_coordinate' in feature_1) and ('projection_y_coordinate' in feature_1) and ('projection_x_coordinate' in feature_2) and ('projection_y_coordinate' in feature_2) :
            method_distance='xy'
        elif ('latitude' in feature_1) and ('longitude' in feature_1) and ('latitude' in feature_2) and ('longitude' in feature_2):
            method_distance='latlon'
        else:
            raise ValueError('either latitude/longitude or projection_x_coordinate/projection_y_coordinate have to be present to calculate distances')

    if method_distance=='xy':
            distance=np.sqrt((feature_1['projection_x_coordinate']-feature_2['projection_x_coordinate'])**2
                     +(feature_1['projection_y_coordinate']-feature_2['projection_y_coordinate'])**2)
    elif method_distance=='latlon':
            distance=1000*haversine(feature_1['latitude'],feature_1['longitude'],feature_2['latitude'],feature_2['longitude'])
    else:
        raise ValueError('method undefined')
    return distance

@xarray_to_iris
def calculate_velocity_individual(feature_old,feature_new,method_distance=None):
    '''
    Parameters
    ----------
    feature_old

    feature_new

    method_distance : {None, 'xy', 'latlon'}, optional
        Default is None.

    Notes
    -----
    feature_old and feature_new need types and descriptions
    needs a short summary
    '''

    distance=calculate_distance(feature_old,feature_new,method_distance=method_distance)
    diff_time=((feature_new['time']-feature_old['time']).total_seconds())
    velocity=distance/diff_time
    return velocity

@xarray_to_iris
def calculate_velocity(track,method_distance=None):
    '''
    Parameters
    ----------
    track

    method_distance : {None, 'xy', 'latlon'}, optional
        Default is None.

    Returns
    -------
    track

    Notes
    -----
    needs short summary, description and type of track
    '''

    for cell_i,track_i in track.groupby('cell'):
        index=track_i.index.values
        for i,index_i in enumerate(index[:-1]):
            velocity=calculate_velocity_individual(track_i.loc[index[i]],track_i.loc[index[i+1]],method_distance=method_distance)
            track.at[index_i,'v']=velocity
    return track

@xarray_to_iris
def velocity_histogram(track,bin_edges=np.arange(0,30,1),density=False,method_distance=None,return_values=False):
    '''
    Parameters
    ----------
    track

    bin_edges : ndarray, optional
        Default is np.arange(0, 30, 1).

    density : bool, optional
        Default is False.

    methods_distance : {None, 'xy', 'latlon'}, optional
        Default is None.

    return_values : bool, optional
        Default is False.

    Returns
    -------
    hist

    bin_edges : ndarray

    velocities

    Notes
    -----
    short summary, types and descriptions
    '''

    if 'v' not in track.columns:
        logging.info('calculate velocities')
        track=calculate_velocity(track)
    velocities=track['v'].values
    hist, bin_edges = np.histogram(velocities[~np.isnan(velocities)], bin_edges,density=density)
    if return_values:
        return hist,bin_edges,velocities
    else:
        return hist,bin_edges

@xarray_to_iris
def calculate_nearestneighbordistance(features,method_distance=None):
    '''

    Parameters
    ----------
    features

    method_distance : {None, 'xy', 'latlon'}, optional
        Default is None.

    Returns
    -------
    features

    Notes
    -----
    short summary, types and descriptions
    '''

    from itertools import combinations
    features['min_distance']=np.nan
    for time_i,features_i in features.groupby('time'):
            logging.debug(str(time_i))
            indeces=combinations(features_i.index.values,2)
            #Loop over combinations to remove features that are closer together than min_distance and keep larger one (either higher threshold or larger area)
            distances=[]
            for index_1,index_2 in indeces:        
                if index_1 is not index_2:                    
                    distance=calculate_distance(features_i.loc[index_1],features_i.loc[index_2],method_distance=method_distance)
                    distances.append(pd.DataFrame({'index_1':index_1,'index_2':index_2,'distance': distance}, index=[0]))
            if any([x is not None for x in distances]):
                distances=pd.concat(distances, ignore_index=True)   
                for i in features_i.index:
                    min_distance=distances.loc[(distances['index_1']==i) | (distances['index_2']==i),'distance'].min()
                    features.at[i,'min_distance']=min_distance
    return features

@xarray_to_iris
def nearestneighbordistance_histogram(features,bin_edges=np.arange(0,30000,500),density=False,method_distance=None,return_values=False):
    '''
    Parameters
    ----------
    features

    bin_edges : ndarray, optional
        Default is np.arange(0, 30000, 500).

    density : bool, optional
        Default is False.

    method_distance : {None, 'xy', 'latlon'}, optional
        Default is None.

    Returns
    -------
    hist

    bin_edges : ndarray

    distances

    Notes
    -----
    short summary, types and descriptions
    '''

    if 'min_distance' not in features.columns:
        logging.debug('calculate nearest neighbor distances')
        features=calculate_nearestneighbordistance(features,method_distance=method_distance)
    distances=features['min_distance'].values
    hist, bin_edges = np.histogram(distances[~np.isnan(distances)], bin_edges,density=density)
    if return_values:
        return hist,bin_edges,distances
    else:
        return hist,bin_edges
    
# Treatment of 2D lat/lon coordinates to be added:
# def calculate_areas_2Dlatlon(latitude_coord,longitude_coord):
#     lat=latitude_coord.core_data()
#     lon=longitude_coord.core_data()
#     area=np.zeros(lat.shape)
#     dx=np.zeros(lat.shape)
#     dy=np.zeros(lat.shape)
    
#     return area

@xarray_to_iris
def calculate_area(features,mask,method_area=None):
    '''
    Parameters
    ----------
    features

    mask : iris.cube.Cube
        Cube containing mask (int for tracked volumes 0 everywhere
        else).

    method_area : {None, 'xy', 'latlon'}, optional
        Default is None.

    Returns
    -------
    hist

    bin_edges : ndarray

    bin_centers

    areas

    Raises
    ------
    ValueError
        If neither latitude/longitude nor
        projection_x_coordinate/projection_y_coordinate are present in
        mask_coords.

        If latitude/longitude coordinates are 2D.

        If latitude/longitude shapes are not supported.

        If method is undefined, e.i. method is neither None, 'xy' nor
        'latlon'.

    Notes
    -----
    needs short summary, types and descriptions
    '''
    from tobac.utils import mask_features_surface,mask_features
    from iris import Constraint
    from iris.analysis.cartography import area_weights
    
    features['area']=np.nan
    
    mask_coords=[coord.name() for coord in mask.coords()]
    if method_area is None:
        if ('projection_x_coordinate' in mask_coords) and ('projection_y_coordinate' in mask_coords):
            method_area='xy'
        elif ('latitude' in mask_coords) and ('longitude' in mask_coords):
                method_area='latlon'
        else:
            raise ValueError('either latitude/longitude or projection_x_coordinate/projection_y_coordinate have to be present to calculate distances')
    logging.debug('calculating area using method '+ method_area)
    if method_area=='xy':
        if not (mask.coord('projection_x_coordinate').has_bounds() and mask.coord('projection_y_coordinate').has_bounds()):
            mask.coord('projection_x_coordinate').guess_bounds()
            mask.coord('projection_y_coordinate').guess_bounds()
            dx=np.diff(mask.coord('projection_x_coordinate').bounds,axis=1)
            dy=np.diff(mask.coord('projection_y_coordinate').bounds,axis=1)
        if mask.coord_dims('projection_x_coordinate')[0] < mask.coord_dims('projection_y_coordinate')[0]:
            area=np.outer(dx,dy)
        if mask.coord_dims('projection_x_coordinate')[0] > mask.coord_dims('projection_y_coordinate')[0]:
            area=np.outer(dy,dx)
    elif method_area=='latlon':
        if (mask.coord('latitude').ndim==1) and (mask.coord('latitude').ndim==1):
            if not (mask.coord('latitude').has_bounds() and mask.coord('longitude').has_bounds()):
                mask.coord('latitude').guess_bounds()
                mask.coord('longitude').guess_bounds()
            area=area_weights(mask,normalize=False)
        elif mask.coord('latitude').ndim==2 and mask.coord('longitude').ndim==2:
            raise ValueError('2D latitude/longitude coordinates not supported yet')
            # area=calculate_areas_2Dlatlon(mask.coord('latitude'),mask.coord('longitude'))
        else:
            raise ValueError('latitude/longitude coordinate shape not supported')
    else:
        raise ValueError('method undefined')
                  
    for time_i,features_i in features.groupby('time'):
        logging.debug('timestep:'+ str(time_i))
        constraint_time = Constraint(time=time_i)
        mask_i=mask.extract(constraint_time)
        for i in features_i.index:
            if len(mask_i.shape)==3:
                mask_i_surface = mask_features_surface(mask_i, features_i.loc[i,'feature'], z_coord='model_level_number')
            elif len(mask_i.shape)==2:            
                mask_i_surface=mask_features(mask_i,features_i.loc[i,'feature'])
            area_feature=np.sum(area*(mask_i_surface.data>0))
            features.at[i,'area']=area_feature
    return features

@xarray_to_iris
def area_histogram(features,mask,bin_edges=np.arange(0,30000,500),
                   density=False,method_area=None,
                   return_values=False,representative_area=False):
    '''
    Parameters
    ----------
    features

    mask : iris.cube.Cube
        Cube containing mask (int id for tracked volumes 0 everywhere
        else).

    bin_edges : ndarray, optional
        Default is np.arange(0, 30000, 500).

    density : bool, optional
        Default is False.

    representive_area: bool, optional
        Default is False.

    Returns
    -------
    hist

    bin_edges : ndarray

    bin_centers

    areas

    Notes
    -----
    short summary, types and descriptions
    '''

    if 'area' not in features.columns:
        logging.info('calculate area')
        features=calculate_area(features,mask,method_area)
    areas=features['area'].values
    # restrict to non NaN values:
    areas=areas[~np.isnan(areas)]
    if representative_area:
        weights=areas
    else:
        weights=None
    hist, bin_edges = np.histogram(areas, bin_edges,density=density,weights=weights)   
    bin_centers=bin_edges[:-1]+0.5*np.diff(bin_edges)

    if return_values:
        return hist,bin_edges,bin_centers,areas
    else:
        return hist,bin_edges,bin_centers
    
@xarray_to_iris
def histogram_cellwise(Track,variable=None,bin_edges=None,quantity='max',density=False):
    '''
    Parameters
    ----------
    Track

    variable : optional
        Default is None.

    bin_edges : ndarray, optional
        Default is None.

    quantity : {'max', 'min', 'mean'}, optional
        Default is 'max'.

    density : bool, optional
        Default is False.

    Returns
    -------
    hist

    bin_edges : ndarray

    bin_centers

    Raises
    ------
    ValueError
        If quantity is not 'max', 'min' or 'mean'.

    Notes
    -----
    short summaray, types and descriptions
    '''

    Track_cell=Track.groupby('cell')
    if quantity=='max':
        variable_cell=Track_cell[variable].max().values
    elif quantity=='min':
        variable_cell=Track_cell[variable].min().values
    elif quantity=='mean':
        variable_cell=Track_cell[variable].mean().values
    else:
        raise ValueError('quantity unknown, must be max, min or mean')
    hist, bin_edges = np.histogram(variable_cell, bin_edges,density=density)
    bin_centers=bin_edges[:-1]+0.5*np.diff(bin_edges)

    return hist,bin_edges, bin_centers

@xarray_to_iris
def histogram_featurewise(Track,variable=None,bin_edges=None,density=False):
    '''
    Parameters
    ----------
    Track

    variable : optional
        Default is None.

    bin_edges : ndarray, optional
        Default is None.

    density : bool, optional
        Default is False.

    Returns
    -------
    hist

    bin_edges : ndarray

    bin_centers

    Notes
    -----
    short summaray, types and descriptions
    '''

    hist, bin_edges = np.histogram(Track[variable].values, bin_edges,density=density)
    bin_centers=bin_edges[:-1]+0.5*np.diff(bin_edges)

    return hist,bin_edges, bin_centers

@xarray_to_iris
def calculate_overlap(track_1,track_2,min_sum_inv_distance=None,min_mean_inv_distance=None):
    '''
    Parameters
    ----------
    track_1, track_2 :

    min_sum_inv_distance : optional
        Default is None.

    min_mean_inv_distance : optional
        Default is None.

    Returns
    -------
    overlap : pandas.DataFrame

    Notes
    -----
    short summary, types and descriptions
    '''

    cells_1=track_1['cell'].unique()
#    n_cells_1_tot=len(cells_1)
    cells_2=track_2['cell'].unique()
    overlap=pd.DataFrame()
    for i_cell_1,cell_1 in enumerate(cells_1):
        for cell_2 in cells_2:
            track_1_i=track_1[track_1['cell']==cell_1]
            track_2_i=track_2[track_2['cell']==cell_2]
            track_1_i=track_1_i[track_1_i['time'].isin(track_2_i['time'])]
            track_2_i=track_2_i[track_2_i['time'].isin(track_1_i['time'])]
            if not track_1_i.empty:
                n_overlap=len(track_1_i)
                distances=[]
                for i in range(len(track_1_i)):
                    distance=calculate_distance(track_1_i.iloc[[i]],track_2_i.iloc[[i]],method_distance='xy')
                    distances.append(distance)
#                mean_distance=np.mean(distances)
                mean_inv_distance=np.mean(1/(1+np.array(distances)/1000))
#                mean_inv_squaredistance=np.mean(1/(1+(np.array(distances)/1000)**2))
                sum_inv_distance=np.sum(1/(1+np.array(distances)/1000))
#                sum_inv_squaredistance=np.sum(1/(1+(np.array(distances)/1000)**2))
                overlap=overlap.append({'cell_1':cell_1,
                                'cell_2':cell_2,
                                'n_overlap':n_overlap,
#                                'mean_distance':mean_distance,
                                'mean_inv_distance':mean_inv_distance,
#                                'mean_inv_squaredistance':mean_inv_squaredistance,
                                'sum_inv_distance':sum_inv_distance,
#                                'sum_inv_squaredistance':sum_inv_squaredistance
                               },ignore_index=True)
    if min_sum_inv_distance:
        overlap=overlap[(overlap['sum_inv_distance']>=min_sum_inv_distance)] 
    if min_mean_inv_distance:
        overlap=overlap[(overlap['mean_inv_distance']>=min_mean_inv_distance)] 

    return overlap

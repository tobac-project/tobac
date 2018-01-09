def watershedding(Track,WC,WC_threshold=3e-3,level=None,compactness=0,method='watershed'):
    """
    Function using watershedding to determine cloud volumes associated with tracked updrafts
    
    Parameters:
    Track:         pandas.DataFrame 
                   output from trackpy/maketrack
    WC:            iris.cube.Cube 
                   containing the water content field to use the watershedding on 
    WC_threshold:  float 
                   threshold for the watershedding
    level          slice
                   levels at which to seed the particles for the watershedding algorithm
    compactness    float
                   parameter describing the compactness of the resulting volume
    
    Output:
    Watershed_out: iris.cube.Cube
                   Cloud mask, 0 outside and integer numbers according to track inside the clouds
    
    """
    
    import numpy as np
    import copy
    from skimage.morphology import watershed
    from skimage.segmentation import random_walker
#    from scipy.ndimage.measurements import watershed_ift

    #Set level at which to create "Seed" for each cloud and threshold in total water content:
    # If none, use all levels (later reduced to the ones fulfilling the theshold conditions)
    if level==None:
        level=slice(None)
    
    Data=copy.deepcopy(WC)
    Data.data[:,:,:]=0
    Watershed_out=copy.deepcopy(WC)
    Watershed_out.rename('watershedding_output_mask')
    Watershed_out.data[:]=0
    Watershed_out.units=1
    for i, time in enumerate(WC.coord('time').points):        
#        print('doing watershedding for',WC.coord('time').units.num2date(time).strftime('%Y-%m-%d %H:%M:%S'))
        Tracks_i=Track[Track['frame']==i]
        data_i=WC[i,:].data        
        Mask= np.zeros_like(data_i).astype(np.int16)
        Cloudy=data_i>WC_threshold
        Mask[Cloudy]=1
        markers = np.zeros_like(Cloudy).astype(np.int16)
        for index, row in Tracks_i.iterrows():
            markers[:,round(row.y), round(row.x)]=row.particle
        data_i[~Cloudy]=0
        markers[~Cloudy]=0

        data_i_watershed=(1-data_i*10)*1000
        data_i_watershed[~Cloudy]=2000

        data_i_watershed=data_i_watershed.astype(np.uint16)
        #res1 = watershed_ift(data_i_watershed, markers)
        
        if method=='watershed':
            res1 = watershed(data_i_watershed,markers.astype(np.int8), mask=Cloudy,compactness=compactness)
        elif method=='random_walker':
            #res1 = random_walker(Mask, markers,mode='cg')
             res1=random_walker(data_i_watershed, markers.astype(np.int8), beta=130, mode='bf', tol=0.001, copy=True, multichannel=False, return_full_prob=False, spacing=None)
        else:
            print('unknown method')
        Watershed_out.data[i,:]=res1
    return Watershed_out

def mask_cube_particle(variable_cube,Mask,particle):
    import numpy as np 
    from copy import deepcopy
    variable_cube_out=deepcopy(variable_cube)
    mask=Mask.data!=particle
    variable_cube_out.data=np.ma.array(variable_cube_out.data,mask=mask)    
    return variable_cube_out

def mask_cube_untracked(variable_cube,Mask):
    import numpy as np 
    from copy import deepcopy
    variable_cube_out=deepcopy(variable_cube)
    mask=Mask.data!=0
    variable_cube_out.data=np.ma.array(variable_cube_out.data,mask=mask)    
    return variable_cube_out

def mask_particle(Mask,particle,masked=False):
    import numpy as np 
    from copy import deepcopy
    Mask_i=deepcopy(Mask)
    Mask_i.data[Mask_i.data!=particle]=0
    if masked:
        Mask_i.data=np.ma.array(Mask_i.data,mask=Mask_i.data)
    return Mask_i   

def mask_particle_surface(Mask,particle,masked=False,z_coord=None):
    from iris.analysis import MAX
    import numpy as np 
    from copy import deepcopy
    Mask_i=deepcopy(Mask)
    Mask_i.data[Mask_i.data!=particle]=0
    for coord in  Mask_i.coords():
        if coord.ndim>1 and Mask_i.coord_dims(z_coord) in Mask_i.coord_dims(coord):
            Mask_i.remove_coord(coord.name())
    Mask_i_surface=Mask_i.collapsed(z_coord,MAX)
    if masked:
        Mask_i_surface.data=np.ma.array(Mask_i_surface.data,mask=Mask_i_surface.data)
    return Mask_i_surface    
def watershedding(Track,WC,WC_threshold=3e-3,level=None):
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
    
    Output:
    Watershed_out: iris.cube.Cube
                   Cloud mask, 0 outside and integer numbers according to track inside the clouds
    
    """
    
    import numpy as np
    import copy
    from skimage.morphology import watershed
#    from skimage.segmentation import random_walker
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
        print('doing watershedding for',WC.coord('time').units.num2date(time).strftime('%Y-%m-%d %H:%M:%S'))
        Tracks_i=Track[Track['frame']==i]
        data_i=WC[i,:].data        
        print('Max TWC: ',np.amax(data_i))

        Mask= np.zeros_like(data_i).astype(np.int16)
        Cloudy=data_i>WC_threshold
        Mask[Cloudy]=1
        markers = np.zeros_like(Cloudy).astype(np.int16)
        for index, row in Tracks_i.iterrows():
            markers[:,round(row.y), round(row.x)]=row.particle
        data_i[~Cloudy]=0
        data_i_watershed=(1-data_i*10)*1000
        data_i_watershed[~Cloudy]=2000

        data_i_watershed=data_i_watershed.astype(np.uint16)
        #res1 = watershed_ift(data_i_watershed, markers)
        res1 = watershed(data_i_watershed,markers.astype(np.int8), mask=Cloudy)
        #res1 = random_walker(Mask, markers,mode='cg')
        Watershed_out.data[i,:]=res1
    return Watershed_out

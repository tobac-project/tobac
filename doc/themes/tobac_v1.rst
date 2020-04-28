tobac theme: tobac_v1
---------------------




Feature detection
---------------------

The feature detection form the first step of the analysis.

**Currently implemented methods:**

	**Multiple thresholds:**

	Features are identified as regions above or below a sequence of subsequent thresholds (if searching for eather maxima or minima in the data). Subsequently more restrictive threshold values are used to further refine the resulting features and allow for separation of features that are connected through a continuous region of less restrictive threshold values.

	.. image:: ../images/detection_multiplethresholds.png
            :width: 500 px

Linking
-------
Currently implemented methods for linking detected features into cloud tracks:

**Trackpy:**

This method uses the trackpy library (http://soft-matter.github.io/trackpy). 
This approach only takes the point-like position of the feature, e.g. determined as the weighted mean, into account and does not use any other information about the identified features into account. The linking makes use of the information from the linked features in the previous timesteps to predict the position and then searches for matching features in a search range determined by the v_max parameter.

        .. image:: ../images/linking_prediction.png
            :width: 500 px


Segmentation
----------------
The segmentation step aims at associating cloud areas (2D data) or cloud volumes (3D data) with the identified and tracked features.

**Currently implemented methods:**

        **Watershedding in 2D:**  
        Markers are set at the position of the individual feature positions identified in the detection step. Then watershedding with a fixed threshold is used to determine the area around each feature above/below that threshold value. This results in a mask with the feature id at all pixels identified as part of the clouds and zeros in all cloud free areas.

        **Watershedding in 3D:**  
	Markers are set in the entire column above the individual feature positions identified in the detection step. Then watershedding with a fixed threshold is used to determine the volume around each feature above/below that threshold value. This results in a mask with the feature id at all voxels identified as part of the clouds and zeros in all cloud free areas.


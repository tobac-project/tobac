# Data Output

The output of the different analysis steps in tobac are output as either pandas DataFrames in the case of one-dimensional data, such a lists of identified features or feature tracks or as xarray DataArrays/Iris Cubes in the case of 2D/3D/4D fields such as feature masks. Note that the dataframe output from tracking is a superset of the features dataframe.

- For information on feature detection *output* from {py:func}`tobac.feature_detection.feature_detection_multithreshold`, see ({doc}`/userguide/feature_detection/feature_detection_output`).
- For information on tracking *output* from {py:func}`tobac.tracking.linking_trackpy`, see ({doc}`/userguide/tracking/tracking_output`). 

Note that in future versions of tobac, it is planned to combine both output data types into a single hierarchical data structure containing both spatial and object information. Additional information about the planned changes can be found in the v2.0-dev branch of the main [tobac repository](https://github.com/tobac-project/tobac), as well as the [tobac roadmap](https://github.com/tobac-project/tobac-roadmap).

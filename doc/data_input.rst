*Data input and output
======================

Input data for tobac should consist of one or more fields on a common, regular grid with a time dimension and two or more spatial dimensions. The input data should also include latitude and longitude coordinates, either as 1-d or 2-d variables depending on the grid used.

Interoperability with xarray is provided by the convenient functions allowing for a transformation between the two data types.
xarray DataArays can be easily converted into iris cubes using xarray's `to__iris() <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.to_iris.html>`_ method, while the Iris cubes produced as output of tobac can be turned into xarray DataArrays using the `from__iris() <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.from_iris.html>`_ method.

For the future development of the next major version of tobac, we are envisaging moving the basic data structures from Iris cubes to xarray DataArrays for improved computing performance and interoperability with other open-source sorftware packages.

The output of the different analysis steps in tobac are output as either pandas DataFrames in the case of one-dimensional data, such a lists of identified features or cloud trajectories or as Iris cubes in the case of 2D/3D/4D fields such as cloud masks. Note that the dataframe output from tracking is a superset of the features dataframe.

(quick note on terms; “feature” is a detected object at a single time step. “cell” is a series of features linked together over multiple timesteps)

Overview of the output dataframe from feature_dection
- Frame: the index along the time dimension in which the feature was detected
- hdim_1, hdim_2…: the central index location of the feature in the spatial dimensions of the input data
- num: the number of connected pixels that meet the threshold for detection for this feature
- threshold_value: the threshold value that was used to detect this feature. When using feature_detection_multithreshold  this is the max/min (depending on       whether the threshold values are increasing (e.g. precip) or decreasing (e.g. temperature) with intensity) threshold value used.
- feature: a unique integer >0 value corresponding to each feature
- time: the date and time of the feature, in datetime format
- timestr: the date and time of the feature in string format
- latitude, longitude: the central lat/lon of the feature
- x,y, etc: these are the central location of the feature in the original dataset coordinates

Also in the tracked output:
- Cell: The cell which each feature belongs to. Is nan if the feature could not be linked into a valid trajectory
- time_cell: The time of the feature along the tracked cell, in numpy.timedelta64[ns] format

The output from segmentation is an n-dimensional array produced by segmentation  in the same coordinates of the input data. It has a single field, which provides a mask for the pixels in the data which are linked to each detected feature by the segmentation routine. Each non-zero value in the array provides the integer value of the feature which that region is attributed to.

Note that in future versions of tobac, it is planned to combine both output data types into a single hierarchical data structure containing both spatial and object information. Additional information about the planned changes can be found in the v2.0-dev project, as well as the tobac roadmap

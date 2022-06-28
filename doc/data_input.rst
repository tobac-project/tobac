.. _Data Input:
Data input
==========

Input data for tobac should consist of one or more fields on a common, regular grid with a time dimension and two or more spatial dimensions. The input data should also include latitude and longitude coordinates, either as 1-d or 2-d variables depending on the grid used.

Interoperability with xarray is provided by the convenient functions allowing for a transformation between the two data types.
xarray DataArays can be easily converted into iris cubes using xarray's `to__iris() <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.to_iris.html>`_ method, while the Iris cubes produced as output of tobac can be turned into xarray DataArrays using the `from__iris() <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.from_iris.html>`_ method.

For the future development of the next major version of tobac, we are envisaging moving the basic data structures from Iris cubes to xarray DataArrays for improved computing performance and interoperability with other open-source sorftware packages.

The output of the different analysis steps in tobac are output as either pandas DataFrames in the case of one-dimensional data, such a lists of identified features or cloud trajectories or as Iris cubes in the case of 2D/3D/4D fields such as cloud masks. Note that the dataframe output from tracking is a superset of the features dataframe.

(quick note on terms; “feature” is a detected object at a single time step. “cell” is a series of features linked together over multiple timesteps)

For information on feature detection *output*, see `Feature Detection Output`_. 

Note that in future versions of tobac, it is planned to combine both output data types into a single hierarchical data structure containing both spatial and object information. Additional information about the planned changes can be found in the v2.0-dev project, as well as the tobac roadmap

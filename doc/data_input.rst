..
    Description of the input data required.

Data input
==========

Input data for tobac should consist of one or more fields on a common, regular grid with a time dimension and two or more spatial dimensions. The input data can also include latitude and longitude coordinates, either as 1-d or 2-d variables depending on the grid used.

Interoperability with xarray is provided by the convenient functions allowing for a transformation between the two data types.
xarray DataArays can be easily converted into iris cubes using xarray's `to_iris() <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.to_iris.html>`_ method, while the Iris cubes produced as output of tobac can be turned into xarray DataArrays using the `from_iris() <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.from_iris.html>`_ method.

For the future development of the next major version of tobac (v2.0), we are moving the basic data structures from Iris cubes to xarray DataArrays for improved computing performance and interoperability with other open-source sorftware packages, including the Pangeo project.

The output of the different analysis steps in tobac are output as either pandas DataFrames in the case of one-dimensional data, such a lists of identified features or feature tracks or as Iris cubes in the case of 2D/3D/4D fields such as cloud masks. Note that the dataframe output from tracking is a superset of the features dataframe.

For information on feature detection *output*, see :doc:`feature_detection_output`. 
For information on tracking *output*, see :doc:`tracking_output`. 

Note that in future versions of tobac, it is planned to combine both output data types into a single hierarchical data structure containing both spatial and object information. Additional information about the planned changes can be found in the v2.0-dev branch of the main tobac repository, as well as the tobac roadmap.

Data input and output
======================
The output of the different analysis steps in tobac are output as either pandas DataFrames in the case of one-dimensional data, such a lists of identified features or cloud trajectoies or as Iris cubes in the case of 2D/3D/4D fields such as cloud masks.

Interoperability with xarray is provided by the convenient functions allowing for a transformation between the two data types.
xarray DataArays can be easily converted into iris cubes using xarray's `to__iris() <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.to_iris.html>`_ method, while the Iris cubes produced as output of tobac can be turned into xarray DataArrays using the`from__iris() <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.from_iris.html>`_ method.

For the future development of the next major version of tobac, we are envisaging moving the basic data structures from Iris cubes to xarray DataArrays for improved computing performance and interoperability with other open-source sorftware packages.

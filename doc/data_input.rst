Data input and output
======================
The output of the different analysis steps in tobac are output as either xarray Datasets in the case of one-dimensional data, such a lists of identified features or cloud trajectoies or as xarray DataArrays in the case of 2D/3D/4D fields such as cloud masks.

Interoperability with Iris and pandas is provided by the convenient functions allowing for a transformation between the data types.
xarray DataArays can be easily converted into iris cubes using xarray's `to__iris() <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.to_iris.html>`_ method, while the Iris cubes produced as output of tobac can be turned into xarray DataArrays using the `from__iris() <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.from_iris.html>`_ method.
xarray Datasets can be easily converted into pandas DataFrames using the Dataset's `to__dataframe() <http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_dataframe.html>`_ method, while the pandas DataFrames produced can be turned into xarray Datasets using the `to_xarray() <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_xarray.html>`_ method.

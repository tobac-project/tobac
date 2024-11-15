.. _data_input:

Data Input
==========

Input data for tobac should consist of one or more fields on a common, regular grid with a time dimension and two or three spatial dimensions. The input data can also include latitude and longitude coordinates, either as 1-d or 2-d variables depending on the grid used.

Interoperability with xarray is provided by the convenient functions allowing for a transformation between the two data types.
xarray DataArays can be easily converted into iris cubes using xarray's `to_iris() <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.to_iris.html>`_ method, while the Iris cubes produced as output of tobac can be turned into xarray DataArrays using the `from_iris() <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.from_iris.html>`_ method.

For the future development of the next major version of tobac (v2.0), we are moving the basic data structures from Iris cubes to xarray DataArrays for improved computing performance and interoperability with other open-source sorftware packages, including the Pangeo project (`https://pangeo.io/ <https://pangeo.io/>`_).

=======
3D Data
=======

As of *tobac* version 1.5.0, 3D data are now fully supported for feature detection, tracking, and segmentation. Similar to how *tobac* requires some information on the horizontal grid spacing of the data (e.g., through the :code:`dxy` parameter), some information on the vertical grid spacing is also required. This is documented in detail in the API docs, but briefly, users must specify either :code:`dz`, where the grid has uniform grid spacing, or users must specify :code:`vertical_coord`, where :code:`vertical_coord` is the name of the coordinate representing the vertical, with the same units as :code:`dxy`.

===========
Data Output
===========

The output of the different analysis steps in tobac are output as either pandas DataFrames in the case of one-dimensional data, such a lists of identified features or feature tracks or as Iris cubes in the case of 2D/3D/4D fields such as feature masks. Note that the dataframe output from tracking is a superset of the features dataframe.

For information on feature detection *output*, see :doc:`feature_detection_output`. 
For information on tracking *output*, see :doc:`tracking_output`. 

Note that in future versions of tobac, it is planned to combine both output data types into a single hierarchical data structure containing both spatial and object information. Additional information about the planned changes can be found in the v2.0-dev branch of the main tobac repository (`https://github.com/tobac-project/tobac <https://github.com/tobac-project/tobac>`_), as well as the tobac roadmap (`https://github.com/tobac-project/tobac-roadmap <https://github.com/tobac-project/tobac-roadmap>`_.

.. _data_input:

Data Input
==========

Input data for tobac should consist of one or more fields on a common, regular grid with a time dimension and two or three spatial dimensions. The input data can also include latitude and longitude coordinates, either as 1-d or 2-d variables depending on the grid used.

As of version 1.6 of tobac, xarray DataArrays are the default format for input fields, with all internal operations performed using DataArrays. Backward compatibility with Iris Cube input is maintained using a conversion wrapper. Workflows using Iris should produce identical results to previous versions, but moving forward xarray is the recommended data format.

=======
3D Data
=======

As of *tobac* version 1.5.0, 3D data are now fully supported for feature detection, tracking, and segmentation. Similar to how *tobac* requires some information on the horizontal grid spacing of the data (e.g., through the :code:`dxy` parameter), some information on the vertical grid spacing is also required. This is documented in detail in the API docs, but briefly, users must specify either :code:`dz`, where the grid has uniform grid spacing, or users must specify :code:`vertical_coord`, where :code:`vertical_coord` is the name of the coordinate representing the vertical, with the same units as :code:`dxy`.

===========
Data Output
===========

The output of the different analysis steps in tobac are output as either pandas DataFrames in the case of one-dimensional data, such a lists of identified features or feature tracks or as xarray DataArrays/Iris Cubes in the case of 2D/3D/4D fields such as feature masks. Note that the dataframe output from tracking is a superset of the features dataframe.

For information on feature detection *output*, see :doc:`feature_detection_output`. 
For information on tracking *output*, see :doc:`tracking_output`. 

Note that in future versions of tobac, it is planned to combine both output data types into a single hierarchical data structure containing both spatial and object information. Additional information about the planned changes can be found in the v2.0-dev branch of the main tobac repository (`https://github.com/tobac-project/tobac <https://github.com/tobac-project/tobac>`_), as well as the tobac roadmap (`https://github.com/tobac-project/tobac-roadmap <https://github.com/tobac-project/tobac-roadmap>`_.

.. _data_input:

Data Input
==========

Input data for tobac should consist of one or more fields on a common, regular grid with a time dimension and two or three spatial dimensions. The input data can also include latitude and longitude coordinates, either as 1-d or 2-d variables depending on the grid used.

As of version 1.6 of tobac, xarray DataArrays are the default format for input fields, with all internal operations performed using DataArrays. Backward compatibility with Iris Cube input is maintained using a conversion wrapper. Workflows using Iris should produce identical results to previous versions, but moving forward xarray is the recommended data format.

=======
3D Data
=======

As of *tobac* version 1.5.0, 3D data are now fully supported for feature detection, tracking, and segmentation. Similar to how *tobac* requires some information on the horizontal grid spacing of the data (e.g., through the :code:`dxy` parameter), some information on the vertical grid spacing is also required. This is documented in detail in the API docs, but briefly, users must specify either :code:`dz`, where the grid has uniform grid spacing, or users must specify :code:`vertical_coord`, where :code:`vertical_coord` is the name of the coordinate representing the vertical, with the same units as :code:`dxy`.


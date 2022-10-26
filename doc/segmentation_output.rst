Segmentation Output
-------------------------

Segmentation outputs a mask (`iris.cube.Cube` and in the future `xarray.DataArray`) with the same dimensions as the input field, where each segmented area has the same ID as its corresponding feature (see feature column in :doc:`feature_detection_output`).
Note that there are some cases in which a feature is not attributed to a segmented area associated with it (see :doc:`features_without_segmented_area`).

Segmentation also outputs the same `pandas` dataframe as obtained by Feature Detection but with one additional column:

.. csv-table:: tobac Segmentation Output Variables
   :file: ./segmentation_out_vars.csv
   :widths: 3, 35, 3, 3
   :header-rows: 1

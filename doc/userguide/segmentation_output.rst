.. _segmentation-output:

Segmentation Output
-------------------------

Segmentation outputs a mask (`iris.cube.Cube` and in the future `xarray.DataArray`) with the same dimensions as the input field, where each segmented area has the same ID as its corresponding feature (see `feature` column in :doc:`feature_detection_output`). Note that there are some cases in which a feature is not attributed to a segmented area associated with it (see :doc:`features_without_segmented_area`).

Segmentation also outputs the same `pandas` dataframe as obtained by Feature Detection (see :doc:`feature_detection_overview`) but with one additional column:

.. csv-table:: tobac Segmentation Output Variables
   :file: ./segmentation_out_vars.csv
   :widths: 3, 35, 3, 3
   :header-rows: 1

One can optionally get the bulk statistics of the data points belonging to each segmented feature (i.e. either the 2D area or the 3D volume assigned to the feature). This is done using the `statistics` parameter when calling :ufunc:`tobac.segmentation.segmentation` . The user-defined metrics are then added as columns to the output dataframe, for example: 

.. csv-table:: tobac Segmentation Output Variables
   :file: ./segmentation_out_vars_statistics.csv
   :widths: 3, 35, 3, 3
   :header-rows: 1

Note that these statistics refer to the data fields that are used as input for the segmentation. It is possible to run the segmentation with different input (see :doc:`transform segmentation`) data to get statistics of a feature based on different variables (e.g. get statistics of cloud top temperatures as well as rain rates for a certain storm object). 

Feature detection output
-------------------------

Feature detection outputs a `pandas` dataframe with several variables. The variables, (with column names listed in the `Variable Name` column), are described below, with units. Note that while these variables come initially from the feature detection step, segmentation and tracking also share some of these variables.

Variables that are common to all feature detection files:

.. csv-table:: tobac Feature Detection Output Variables
   :file: ./feature_detection_base_out_vars.csv
   :widths: 3, 35, 3, 3
   :header-rows: 1

Variables that are included when using 3D feature detection in addition to those above:

.. csv-table:: tobac 3D Feature Detection Output Variables
   :file: ./feature_detection_3D_out_vars.csv
   :widths: 3, 35, 3, 3
   :header-rows: 1

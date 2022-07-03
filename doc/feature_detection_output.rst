Feature detection output
-------------------------

Feature detection outputs a `pandas` dataframe with several variables. The variables, (with column names listed in the `Variable Name` column), are described below, with units. Note that while these variables come initially from the feature detection step, segmentation and tracking also share some of these variables. See :doc:`tracking_output` for the additional columns added by tracking.

Variables that are common to all feature detection files:

.. csv-table:: tobac Feature Detection Output Variables
   :file: ./feature_detection_base_out_vars.csv
   :widths: 3, 35, 3, 3
   :header-rows: 1

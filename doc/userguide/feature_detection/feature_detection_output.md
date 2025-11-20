# Feature Detection Output

Feature detection (from {py:func}`tobac.feature_detection.feature_detection_multithreshold`) outputs a [`pandas` DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) with several variables. The variables, (with column names listed in the `Variable Name` column), are described below with units. Note that while these variables come initially from the feature detection step, segmentation and tracking also share some of these variables as keys (e.g., the `feature` value acts as a universal unique key between each of these). See {doc}`/userguide/tracking/tracking_output` for the additional columns added by tracking.

Variables that are common to all feature detection files:

```{eval-rst}
.. csv-table:: `tobac` Feature Detection Output Variables
   :file: ./feature_detection_base_out_vars.csv
   :widths: 3, 35, 3, 3
   :header-rows: 1
```
Variables that are always included when using 3D feature detection in addition to those above:

```{eval-rst}
.. csv-table:: `tobac` 3D Feature Detection Output Variables
   :file: ./feature_detection_3D_out_vars.csv
   :widths: 3, 35, 3, 3
   :header-rows: 1
```

In addition, *tobac* automatically interpolates any coordinates passed in with your input. Some common coordinates added by this process include: 

```{eval-rst}
.. csv-table:: `tobac` Feature Detection Output Variables
   :file: ./feature_detection_base_out_vars_some.csv
   :widths: 3, 35, 3, 3
   :header-rows: 1
```

      
One can optionally get the bulk statistics of the data points belonging to each feature region or volume. This is done using the `statistics` parameter when calling {py:func}`tobac.feature_detection.feature_detection_multithreshold`. The user-defined metrics are then added as columns to the output dataframe.



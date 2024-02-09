Threshold Feature Detection Parameters
--------------------------------------

The proper selection of parameters used to detect features with the *tobac* multiple threshold feature detection is a critical first step in using *tobac*. This page describes the various parameters available and provides broad comments on the usage of each parameter.

A full list of parameters and descriptions can be found in the API Reference: :py:meth:`tobac.feature_detection.feature_detection_multithreshold`

=========================
Basic Operating Procedure
=========================
The *tobac* multiple threshold algorithm searches the input data (`field_in`) for contiguous regions of data greater than (with `target='maximum'`, see `Target`_) or less than (with `target='minimum'`) the selected thresholds (see `Thresholds`_). Contiguous regions (see `Minimum Threshold Number`_) are then identified as individual features, with a single point representing their location in the output (see `Position Threshold`_). Using this output (see :doc:`feature_detection_output`), segmentation (:doc:`segmentation`) and tracking (:doc:`linking`) can be run. 

.. _Target:

======
Target
======
First, you must determine whether you want to detect features on maxima or minima in your dataset. For example, if you are trying to detect clouds in IR satellite data, where clouds have relatively lower brightness temperatures than the background, you would set :code:`target='minimum'`. If, instead, you are trying to detect clouds by cloud water in model data, where an increase in mixing ratio indicates the presence of a cloud, you would set :code:`target='maximum'`. The :code:`target` parameter will determine the selection of many of the following parameters.

.. _Thresholds:

==========
Thresholds
==========
You can select to detect features on either one or multiple thresholds. The first threshold (or the single threshold) sets the minimum magnitude (either lowest value for :code:`target='maximum'` or highest value for :code:`target='minimum'`) that a feature can be detected on. For example, if you have a field made up of values lower than :code:`10`, and you set :code:`target='maximum', threshold=[10,]`, *tobac* will detect no features. The feature detection uses the threshold value in an inclusive way, which means that if you set :code:`target='maximum', threshold=[10,]` and your field does not only contain values lower than :code:`10`, it will include all values **greater than and equal to** :code:`10`. 

Including *multiple* thresholds will allow *tobac* to refine the detection of features and detect multiple features that are connected through a contiguous region of less restrictive threshold values. You can see a conceptual diagram of that here: :doc:`feature_detection_overview`. To examine how setting different thresholds can change the number of features detected, see the example in this notebook: :doc:`feature_detection/notebooks/multiple_thresholds_example`.


.. _Minimum Threshold Number:

========================
Minimum Threshold Number
========================
The minimum number of points per threshold, set by :code:`n_min_threshold`, determines how many contiguous pixels are required to be above the threshold for the feature to be detected. Setting this point very low can allow extraneous points to be detected as erroneous features, while setting this value too high will cause some real features to be missed. The default value for this parameter is :code:`0`, which will cause any values greater than the threshold after filtering to be identified as a feature. You can see a demonstration of the affect of increasing :code:`n_min_threshold` at: :doc:`feature_detection/notebooks/n_min_threshold_example`.

.. _Position Threshold:

================
Feature Position
================
There are four ways of calculating the single point used to represent feature center: arithmetic center, extreme point, difference weighting, and absolute weighting. Generally, difference weighting (:code:`position_threshold='weighted_diff'`) or absolute weighting (:code:`position_threshold='weighted_abs'`) is suggested for most atmospheric applications. An example of these four methods is shown below, and can be further explored in the example notebook: :doc:`feature_detection/notebooks/position_threshold_example`.

	.. image:: images/position_thresholds.png
            :width: 500 px

.. _Filtering Options:

=================
Filtering Options
=================
Before *tobac* detects features, two filtering options can optionally be employed. First is a multidimensional Gaussian Filter (`scipy.ndimage.gaussian_filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html>`_), with its standard deviation controlled by the :code:`sigma_threshold` parameter. It is not required that users use this filter (to turn it off, set :code:`sigma_threshold=0`), but the use of the filter is recommended for most atmospheric datasets that are not otherwise smoothed. An example of varying the :code:`sigma_threshold` parameter can be seen in the below figure, and can be explored in the example notebook: :doc:`feature_detection/notebooks/feature_detection_filtering`.

	.. image:: images/sigma_threshold_example.png
            :width: 500 px

The second filtering option is a binary erosion (`skimage.morphology.binary_erosion <https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.binary_erosion>`_), which reduces the size of features in all directions. The amount of the erosion is controlled by the :code:`n_erosion_threshold` parameter, with larger values resulting in smaller potential features. It is not required to use this feature (to turn it off, set :code:`n_erosion_threshold=0`), and its use should be considered alongside careful selection of :code:`n_min_threshold`. The default value is :code:`n_erosion_threshold=0`. 

.. _Minimum Distance:

================
Minimum Distance
================
The parameter :code:`min_distance` sets the minimum distance between two detected features. If two detected features are within :code:`min_distance` of each other, the feature with the more extreme value is kept, and the feature with the less extreme value is discarded. 

Watershedding Segmentation Parameters
-------------------------------------

Appropriate parameters must be chosen to properly use the watershedding segmentation module in *tobac*. This page gives a brief overview of parameters available in watershedding segmentation. 

A full list of parameters and descriptions can be found in the API Reference: :py:meth:`tobac.segmentation.segmentation`. 

=========================
Basic Operating Procedure
=========================
The *tobac* watershedding segmentation algorithm selects regions of the data :code:`field` with values greater than :code:`threshold` and associates those regions with the features :code:`features` detected by feature detection (see :doc:`feature_detection_overview`). This algorithm uses a *watershedding* approach, which sets the individual features as initial seed points, and then has identified regions grow from those original seed points. For further information on watershedding segmentation, see `the scikit-image documentation <https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html>`.

Note that you can run the watershedding segmentation algorithm on any variable that shares a grid with the variable detected in the feature detection step. It is not required that the variable used in feature detection be the same as the one in segmentation (e.g., you can detect updraft features and then run segmentation on total condensate). 

Segmentation can be run on 2D or 3D input data, but segmentation on 3D data using a 2D feature detection field requires careful consideration of where the vertical seeding will occur (see `Level`_).

.. _Target:
======
Target
======
The :code:`target` parameter works similarly to how it works in feature detection (see :doc:`threshold_detection_parameters`). To segment areas that are greater than :code:`threshold`, use :code:`target='maximum'`. To segment areas that are less than :code:`threshold`, use :code:`target='minimum'`. 

.. _Threshold:
=========
Threshold
=========
Unlike in multiple threshold detection in Feature Detection, Watershedding Segmentation only accepts one threshold. This value will set either the minimum (for :code:`target='maximum'`) or maximum (for :code:`target='minimum'`) value to be segmented. 

.. _Level:
======================================================
Where the 3D seeds are placed for 2D feature detection
======================================================
When running feature detection on a 2D dataset and then using these detected features to segment data in 3D, there is clearly no information on where to put the seeds in the vertical. This is currently controlled by the :code:`level` parameter. By default, this parameter is :code:`None`, which seeds the full column at every 2D detected feature point. As *tobac* does not run a continuity check, this can result in undesired behavior, such as clouds in multiple layers being detected as one large object. 
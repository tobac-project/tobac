##########################
  Compute bulk statistics
##########################

Bulk statistics allow for a wide range of properties of detected objects to be calculated during feature detection and segmentation or as a postprocessing step.
Th get_statistics_from_mask function applies one or more functions over one or more data fields for each detected object. 
For example, one could calculate the convective mass flux for each detected feature by providing fields of vertical velocity, cloud water content and area. 
Numpy-like broadcasting is supported, allowing 2D and 3D data to be combined.

.. toctree::
   :maxdepth: 1

   notebooks/compute_statistics_during_feature_detection
   notebooks/compute_statistics_during_segmentation
   notebooks/compute_statistics_postprocessing_example

Segmentation:
----------------
The segmentation step aims at associating cloud areas (2D data) or cloud volumes (3D data) with the identified and tracked features.

**Currently implemented methods:**

        **Watershedding in 2D:**

        Markers are set at the position of the individual feature positions identified in the detection step. Then watershedding with a             fixed threshold is used to determine the area around each feature above/below that threshold value. This results in a mask with the feature id at all pixels identified as part of the clouds and zeros in all cloud free areas.

        **Watershedding in 3D:**

Markers are set at the in the entire column above the individual feature positions identified in the detection step. Then watershedding with a fixed threshold is used to determine the volume around each feature above/below that threshold value. This results in a mask with the feature id at all voxels identified as part of the clouds and zeros in all cloud free areas.

**Current development:**

We are currently working on providing additional approaches and algorithms for the segmentation step. Several of these approaches will combine the feature detection and segmentation into a single st pixe  on

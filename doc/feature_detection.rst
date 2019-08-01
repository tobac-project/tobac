Feature detection
---------------------

The feature detection form the first step of the analysis.

**Currently implemented methods:**

	**Multiple thresholds:**

	Features are identified as regions above or below a sequence of subsequent thresholds (if searching for eather maxima or minima in the data). Subsequently more restrictive threshold values are used to further refine the resulting features and allow for separation of features that are connected through a continuous region of less restrictive threshold values.

	.. image:: images/detection_multiplethresholds.png
            :width: 200 px

**Current development:**
We are currently working on additional methods for the identification of cloud features in different types of datasets. Some of these methods are specific to the input data such a combination of different channels from specific satellite imagers. Some of these methods will combine the feature detection and segmentations step in one single algorithm.

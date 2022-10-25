Handling Large Datasets
-------------------------------------

Often, one desires to use *tobac* to identify and track features in large datasets ("big data"). This documentation strives to suggest various methods for doing so efficiently. Current versions of *tobac* do not allow for out-of-memory computation, meaning that these strategies may need to be employed for both computational and memory reasons. 

.. _Split Feature Detection:
=======================
Split Feature Detection
=======================
Current versions of threshold feature detection (see :doc:`feature_detection_overview`) are time independent, meaning that one can parallelize feature detection across all times (although not across space). *tobac* provides the :py:meth:`tobac.utils.combine_tobac_feats` function to combine a list of dataframes produced by a parallelization method (such as :code:`jug` or :code:`multiprocessing.pool`) into a single combined dataframe suitable to perform tracking with. 


Handling Large Datasets
-------------------------------------

Often, one desires to use *tobac* to identify and track features in large datasets ("big data"). This documentation strives to suggest various methods for doing so efficiently. Current versions of *tobac* do not support out-of-core (e.g., :code:`dask`) computation, meaning that these strategies may need to be employed for both computational and memory reasons.

.. _Split Feature Detection:

=======================
Split Feature Detection and Run in Parallel
=======================
Current versions of threshold feature detection (see :doc:`feature_detection_overview`) are time independent, meaning that one can easily parallelize feature detection across all times (although not across space). *tobac* provides the :py:meth:`tobac.utils.combine_feature_dataframes` function to combine a list of dataframes produced by a parallelization method (such as :code:`jug`,  :code:`multiprocessing.pool`, or :code:`dask.bag`) into a single combined dataframe suitable to perform tracking with.

Below is a snippet from a larger notebook demonstrating how to run feature detection in parallel ( :doc:`big_datasets_examples/notebooks/parallel_processing_tobac`).

.. code-block:: python

   # build list of tracked variables using Dask.Bag
   b = dask.bag.from_sequence([combined_ds['data'][x:x+1] for x in range(len(combined_ds['time']))], npartitions=1)
   out_feature_dfs = dask.bag.map(lambda x: tobac.feature_detection_multithreshold(x.to_iris(), 4000, **parameters_features), b).compute()
   combined_dataframes= tobac.utils.general.combine_feature_dataframes(out_feature_dfs)


.. _Split Segmentation:

======================================
Split Segmentation and Run in Parallel
======================================
Recall that the segmentation mask (see :doc:`segmentation_output`) is the same size as the input grid, which results in large files when handling large input datasets. The following strategies can help reduce the output size and make segmentation masks more useful for the analysis.

The first strategy is to only segment on features *after tracking and quality control*. While this will not directly impact performance, waiting to run segmentation on the final set of features (after discarding, e.g., non-tracked cells) can make analysis of the output segmentation dataset easier.

To enhance the speed at which segmentation runs, one can process multiple segmentation times in parallel independently, similar to feature detection. Unlike feature detection, however, there is currently no built-in *tobac* method to combine multiple segmentation times into a single file. While one can do this using typical NetCDF tools such as :code:`nccat` or with xarray utilities such as :code:`xr.concat`, you can also leave the segmentation mask output as separate files, opening them later with multiple file retrievals such as :code:`xr.open_mfdataset`.


.. _Tracking Hanging:

=====================================
Tracking Hangs with too many Features
=====================================

When tracking on a large dataset, :code:`tobac.tracking.linking_trackpy` can hang using the default parameters. This is due to the tracking library :code:`trackpy` searching for the next timestep's feature in too large of an area. This can be solved *without impact to scientific output* by lowering the :code:`subnetwork_size` parameter in :code:`tobac.tracking.linking_trackpy`.


..
   Tobac homepage

tobac - Tracking and Object-Based Analysis of Clouds
-------------------------------------------------------

**tobac** is a Python package to rapidly identify, track and analyze clouds in different types of gridded datasets, such as 3D model output from cloud-resolving model simulations or 2D data from satellite retrievals.

The software is set up in a modular way to include different algorithms for feature identification, tracking, and analyses. **tobac** is also input variable agnostic and doesn't rely on specific input variables, nor a specific grid to work.

Individual features are identified as either maxima or minima in a 2D or 3D time-varying field (see :doc:`feature_detection_overview`). An associated volume can then be determined using these features with a separate (or identical) time-varying 2D or 3D field and a threshold value (see :doc:`segmentation`). The identified objects are linked into consistent trajectories representing the cloud over its lifecycle in the tracking step. Analysis and visualization methods provide a convenient way to use and display the tracking results.

**Version 1.2 of tobac** and some example applications are described in a peer-reviewed article in Geoscientific Model Development as:

Heikenfeld, M., Marinescu, P. J., Christensen, M., Watson-Parris, D., Senf, F., van den Heever, S. C., and Stier, P.: tobac 1.2: towards a flexible framework for tracking and analysis of clouds in diverse datasets, Geosci. Model Dev., 12, 4551–4570, https://doi.org/10.5194/gmd-12-4551-2019, 2019.

**Version 1.5 of tobac** and the major enhancements that came with that version are described in the following peer-reviewed article in Geoscientific Model Development:

Sokolowsky, G. A., Freeman, S. W., Jones, W. K., Kukulies, J., Senf, F., Marinescu, P. J., Heikenfeld, M., Brunner, K. N., Bruning, E. C., Collis, S. M., Jackson, R. C., Leung, G. R., Pfeifer, N., Raut, B. A., Saleeby, S. M., Stier, P., and van den Heever, S. C.: tobac v1.5: Introducing Fast 3D Tracking, Splits and Mergers, and Other Enhancements for Identifying and Analysing Meteorological Phenomena. Geoscientific Model Development, 17(13), 5309-5330. https://doi.org/10.5194/gmd-17-5309-2024, 2024.


The project is currently being extended by several contributors to include additional workflows and algorithms using the same structure, syntax, and data formats.

.. toctree::
   :caption: Basic Information
   :maxdepth: 2

   installation
   data_input
   analysis 
   plotting
   big_datasets 
   examples
   publications

.. toctree::
   :caption: Feature Detection
   :maxdepth: 2

   feature_detection_overview
   threshold_detection_parameters
   feature_detection/index
   feature_detection_output

.. toctree::
   :caption: Segmentation
   :maxdepth: 2

   segmentation
   segmentation_parameters
   segmentation_output
   features_without_segmented_area
   transform_segmentation

.. toctree::
   :caption: Tracking
   :maxdepth: 2

   linking  
   tracking_output
   
.. toctree::
   :caption: Merge/Split
   :maxdepth: 2

   merge_split


.. toctree::
   :caption: Developer guide
   :maxdepth: 3

   code_structure
   contributing
   code_reviews
   mentoring

.. toctree:: 
   :caption: Compute bulk statistics
   :maxdepth: 2

   bulk_statistics/index

.. toctree::
   :caption: API Reference
   :maxdepth: 3

   tobac

.. toctree::
   :caption: 3D Tracking
   :maxdepth: 2

   tracking/index
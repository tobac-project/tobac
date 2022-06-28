tobac - Tracking and Object-Based Analysis of Clouds
-------------------------------------------------------

**tobac** is a Python package to identify, track and analyze clouds in different types of gridded datasets, such as 3D model output from cloud-resolving model simulations or 2D data from satellite retrievals.

The software is set up in a modular way to include different algorithms for feature identification, tracking, and analyses. **tobac** is also input variable agnostic and doesn't rely on specific input variables to work.

In the current implementation, individual features are identified as either maxima or minima in a two-dimensional time-varying field. An associated volume can then be determined using these features with a separate (or identical) time-varying 2D or 3D field and a threshold value. The identified objects are linked into consistent trajectories representing the cloud over its lifecycle in the tracking step. Analysis and visualization methods provide a convenient way to use and display the tracking results.

Version 1.2 of tobac and some example applications are described in a manuscript in Geoscientific Model Development as:

Heikenfeld, M., Marinescu, P. J., Christensen, M., Watson-Parris, D., Senf, F., van den Heever, S. C., and Stier, P.: tobac 1.2: towards a flexible framework for tracking and analysis of clouds in diverse datasets, Geosci. Model Dev., 12, 4551–4570, https://doi.org/10.5194/gmd-12-4551-2019, 2019.

The project is currently being extended by several contributors to include additional workflows and algorithms using the same structure, syntax, and data formats.

.. toctree::
   :caption: Basic Information
   :maxdepth: 2
   :numbered:

   installation
   data_input
   feature_detection
   segmentation
   linking  
   analysis 
   plotting 
   examples

.. toctree::
   :caption: Output Documentation
   :maxdepth: 2

   feature_detection_output

.. toctree::
   :caption: API Reference
   :maxdepth: 2

   tobac

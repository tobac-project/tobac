tobac - Tracking and Object-Based Analysis of Clouds
-----------

**tobac** is a Python package to identify, track and analyse clouds in different types of gridded datasets, such as 3D model output from cloud resolving model simulations or 2D data from satellite retrievals.

The software is set up in a modular way to include different algorithms for feature identification, tracking and analyses.
In the current implementation, individual features are indentified as either maxima or minima in a two dimensional time varying field. The volume/are associated with the identified object can be determined based on a time-varying 2D or 3D field and a threshold value. In the tracking step, the identified objects are linked into consistent trajectories representing the cloud over its lifecycle. Analysis and visualisation methods provide a convenient way to use and display the tracking results.

The project is currently extended by several contributors to include additional workflows and algorithms using the same structure, synthax and data formats.

.. toctree::
   :maxdepth: 2
   :numbered:
   installation
   feature_detection
   segmentation
   linking  
   analysis 
   plotting 
   examples

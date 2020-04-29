tobac - Tracking and Object-Based Analysis of Clouds
-----------

**tobac** is a Python package to identify, track and analyse clouds in different types of gridded datasets, such as 3D model output from cloud resolving model simulations or 2D data from satellite retrievals.

The software is set up in a modular way to include different algorithms for feature identification, tracking and analyses.
In the current implementation, individual features are indentified as either maxima or minima in a two dimensional time varying field. The volume/are associated with the identified object can be determined based on a time-varying 2D or 3D field and a threshold value. In the tracking step, the identified objects are linked into consistent trajectories representing the cloud over its lifecycle. Analysis and visualisation methods provide a convenient way to use and display the tracking results.

Version 1.2 of tobac and some example applications are described in a paper in the journal Geoscientific Model Development as:

Heikenfeld, M., Marinescu, P. J., Christensen, M., Watson-Parris, D., Senf, F., van den Heever, S. C., and Stier, P.: tobac v1.2: towards a flexible framework for tracking and analysis of clouds in diverse datasets, Geosci. Model Dev., `https://doi.org/10.5194/gmd-12-4551-2019 <https://doi.org/10.5194/gmd-12-4551-2019>`_ , 2019.

The project is currently extended by several contributors to include additional workflows and algorithms using the same structure, synthax and data formats.

.. toctree::
   :maxdepth: 2
   :numbered:

   installation
   data_input
   themes
   analysis 
   plotting 
   examples

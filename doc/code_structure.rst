Code structure and key design koncepts 
--------------------------------------

==================================
Modules
==================================

**tobac** aims to provide a flexible and modular framework which can be seen as a toolbox to create tracking algorithms according to the user's specific research needs. 


The **tobac** package currently consists of three main modules:

1. The :py:mod:`tobac.feature_detection` contains methods to identify features/objects in 2D or 3D (3D or 4D when including the time dimensions) gridded data. This is done by identifying contiguous regions above/below on one or multiple user-defined thresholds. The module makes use of :py:mod:`scipy.ndimage.label`, a generic image processing method that labels features in an array. The methods in this module are high-level functions that enable a fast and effective feature detection and create a user-friendly output with information on each detected feature. The method that is the most important for users is currently `tobac.feature_detection_multithreshold`. 


2. The :py:mod:`tobac.segmentation` contains methods to define the extent of the identified feature areas or volumes. This is currently done by using watershdding, but more methods are to be implemented. Note that this module can handle both 2D and 3D data. 


3. The :py:mod:`tobac.tracking` module is responsible for the code that links identified features over time. This module makes use of the python package :py:mod:`trackpy`. Note that currently the linking is based on particle tracking principles which means that only the feature center positions are needed to link features over time. Other methods such as tracking based on overlapping areas from the segmented features are to be implemented.


In addition to the main modules, there are three postprocessing modules: 

4. The :py:mod:`tobac.merge_split` module provides functionality to identify mergers and splitters in the tracking output and to add labels such that one can reconstruct the parent and child tracks of each cell. 

5. The :py:mod:`tobac.analysis` module contains methods to analyze the tracking output and derive statistics about individual tracks as well as summary statistics of the entire populations of tracks or subsets of the latter. 

6. The :py:mod:`tobac.plotting` module provides methods to create plots, in particular maps and animations, of the tracking output. 


Finally, there are two modules that are particularly important for developers:

7. The :py:mod:`tobac.utils` module is a collection of smaller not necessarily tracking-specific methods that facilitate and support the methods of the main modules. This module has multiple submodules and the most important here is the separation between methods that are more generic and could be used by users as needed (:py:mod:`tobac.utils.general`) and methods that facilitate the development of **tobac** and are therefore primarily for internal use (:py:mod:`tobac.utils.internal`). Sometimes, new features come with the need of a whole set of nee methods, so it could make sense to save these in their own module (e.g. :py:mod:`tobac.periodic_boundaries`)

8. The :py:mod:`tobac.testing` module provides support for writing of unit tests. In particular, it contains several methods to create simplified test data sets on which the various methods and parameters for feature detection, segmentation, and tracking can be tested. 

For more information on each submodule, refer to the respective source code documentation.

One thing to note is that **tobac** as of now is purely functional. The plan is, however, to move towards a more object-oriented design with base classes for the main operations such as feature detection and tracking. 

============================
Migrating to xarray and dask
============================

- Basics of xarray (xarray.Dataarray class) and dask
- How these are or could be used in tobac 
- How to work on this












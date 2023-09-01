Code structure and key design koncepts 
--------------------------------------

==================================
Modules
==================================

**tobac** aims to provide a flexible and modular framework which can be seen as a toolbox to create tracking algorithms according to the user's specific research needs. 

The **tobac** package currently consists of three **main modules**:

1. The :py:mod:`tobac.feature_detection` contains methods to identify objects (*features*) in 2D or 3D (3D or 4D when including the time dimensions) gridded data. This is done by identifying contiguous regions above or below one or multiple user-defined thresholds. The module makes use of :py:mod:`scipy.ndimage.label`, a generic image processing method that labels features in an array. The methods in :py:mod:`tobac.feature_detection` are high-level functions that enable a fast and effective feature detection and create easy-to-use output in form of a :py:mod:`pandas.DataFrame` that contains the coordinates and some basic information on each detected feature. The most high-level methods that is commonly used by users is :py:func:`tobac.feature_detection_multithreshold`. 

2. The :py:mod:`tobac.segmentation` module contains methods to define the extent of the identified feature areas or volumes. This step is needed to create a mask of the identified features because the feature detection currently only saves the center points of the features. The segmentation procedure is performed by using the watershedding method, but more methods are to be implemented in the future. Just as the feature detection, this module can handle both 2D and 3D data. 

3. The :py:mod:`tobac.tracking` module is responsible for linking identified features over time. This module makes primarily use of the python package :py:mod:`trackpy`. Note that the linking using :py:mod:`trackpy` is based on particle tracking principles which means that only the feature center positions (not the entire area or volume associated with each feature) are needed to link features over time. Other methods such as tracking based on overlapping areas from the segmented features are to be implemented.

In addition to the main modules, there are three **postprocessing modules**: 

4. The :py:mod:`tobac.merge_split` module provides functionality to identify mergers and splitters in the tracking output and to add labels such that one can reconstruct the parent and child tracks of each cell. 

5. The :py:mod:`tobac.analysis` module contains methods to analyze the tracking output and derive statistics about individual tracks as well as summary statistics of the entire populations of tracks or subsets of the latter. 

6. The :py:mod:`tobac.plotting` module provides methods to visualize the tracking output, for example for creating maps and animations of identified features, segmented areas and tracks.

   
Finally, there are two modules that are primarily **important for developers**:

7. The :py:mod:`tobac.utils` module is a collection of smaller, not necessarily tracking-specific methods that facilitate and support the methods of the main modules. This module has multiple submodules. We separate methods that are rather generic and could also be practical for tobac users who build their own tracking algorithms (:py:mod:`tobac.utils.general`) and methods that mainly facilitate the development of **tobac** (:py:mod:`tobac.utils.internal`). Sometimes, new features come with the need of a whole set of new methods, so it could make sense to save these in their own submodule (see e.g. :py:mod:`tobac.periodic_boundaries`)

8. The :py:mod:`tobac.testing` module provides support for writing of unit tests. This module contains several methods to create simplified test data sets on which the various methods and parameters for feature detection, segmentation, and tracking can be tested. 

For more information on each submodule, refer to the respective source code documentation.

One thing to note is that **tobac** as of now is purely functional. The plan is, however, to move towards a more object-oriented design with base classes for the main operations such as feature detection and tracking. 


========
Examples
========

To help users get started with **tobac** and to demonstrate the various functionalities, **tobac** hosts several detailed and **illustrated examples** in the form of Jupyter notebooks. They are hosted under the directory `examples/` and be executed by the user.


============================
Migrating to xarray and dask
============================

- Basics of xarray (xarray.Dataarray class) and dask
- How these are or could be used in tobac 
- How to work on the transition












tobac - Tracking and Object-based Analysis of Clouds
======
[![Documentation Status](https://readthedocs.org/projects/tobac/badge/?version=latest)](https://tobac.readthedocs.io/en/latest/?badge=latest)[![Build Status](https://travis-ci.org/climate-processes/tobac.svg?branch=master)](https://travis-ci.org/climate-processes/tobac)

What is it?
-----------

**tobac** is a Python package for identifiying, tracking and analysing of clouds in different types of gridded datasets, i.e. 3D model output from cloud resolving model simulations or 2D data of satellite observations.

Documentation
-------------
Individual features are indentified as either maxima or minima in a two dimensional time varying field.
The volume/area associated with the identified objects can be determined based on a time-varying 2D or 3D field and a threshold value. The in thre tracking step, the identified objects are linked into consistent trajectories representing the cloud over its lifecycle.

Installation
------------
tobac now works for both Python 3 and Python 2 installations.

The easiest way is to install the most recent version of tobac via conda and the conda-forge channel:
```
conda install -c conda-forge tobac 
```
This will take care of all necessary dependencies and should do the job for most users and also allows for an easy update of the installation by
```
conda update -c conda-forge tobac 
```


You can also install conda via pip, which is mainly interesting for development purposed or to use specific development branches for the Github repository.

Required packages: trackpy scipy numpy iris scikit-learn cartopy pandas pytables 

If you are using anaconda, the following command should make sure all dependencies are met and up to date:
```
conda install -c conda-forge trackpy scipy numpy iris scikit-learn cartopy pandas pytables 
```
You can directly install the package directly from github with pip and either of the two following commands:
```
pip install --upgrade git+ssh://git@github.com/climate-processes/tobac.git
pip install --upgrade git+https://github.com/climate-processes/tobac.git
```
You can also clone the package with any of the two following commands
```
git clone git@github.com:climate-processes/tobac.git
git clone https://github.com/climate-processes/tobac.git
```
and install the package from the locally cloned version:
```
pip install tobac/
```
